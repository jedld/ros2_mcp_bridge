#!/usr/bin/env python3
"""
ros_node.py — rclpy node for the ROS2 MCP bridge.

Runs in a dedicated daemon thread alongside the FastMCP server.
Subscribes to configured topics, caches the latest message for each, and
exposes synchronous getter / publisher / service-call methods that the
MCP tool handlers can call from any async context.
"""

import base64
import json
import math
import threading
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, LaserScan
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray

logger = logging.getLogger(__name__)


class ROS2BridgeNode(Node):
    """
    rclpy Node that caches the latest message from each subscribed topic
    and provides synchronous helpers for the MCP tool layer.

    Instantiate once; call `get_latest(topic, timeout)` from MCP tools.
    """

    def __init__(self, config: dict):
        super().__init__("ros2_mcp_bridge")
        self._cfg = config
        self._cb = ReentrantCallbackGroup()

        # ------------------------------------------------------------------ #
        # Per-topic cache: {topic_name: {"msg": <msg>, "event": Event}}
        # ------------------------------------------------------------------ #
        self._cache: Dict[str, dict] = {}
        self._cache_lock = threading.Lock()

        # Teleop deadman state
        self._cmd_pub = None
        self._cmd_lock = threading.Lock()
        self._last_cmd_time = 0.0

        # Nav2 action client (created lazily)
        self._nav_client = None
        self._nav_lock = threading.Lock()

        # ------------------------------------------------------------------ #
        # Build subscriptions from config
        # ------------------------------------------------------------------ #
        self._setup_subscriptions()

        # Cmd_vel publisher (always created — zero cost if unused)
        cmd_topic = config.get("cmd_vel_topic", "/cmd_vel")
        self._cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.get_logger().info(f"[ros2_mcp_bridge] Publisher: {cmd_topic}")

        # Deadman watchdog: stop robot if no cmd received for >0.5 s
        self.create_timer(0.1, self._deadman_watchdog, callback_group=self._cb)

        self.get_logger().info("[ros2_mcp_bridge] Node ready.")

    # ------------------------------------------------------------------ #
    # Subscription setup
    # ------------------------------------------------------------------ #

    def _setup_subscriptions(self):
        """Create subscriptions for every topic declared in config."""
        topics = self._cfg.get("topics", {})
        for key, info in topics.items():
            topic = info.get("topic")
            msg_type_str = info.get("type", "")
            if not topic or not msg_type_str:
                continue
            msg_cls = _resolve_msg_type(msg_type_str)
            if msg_cls is None:
                self.get_logger().warn(
                    f"[ros2_mcp_bridge] Unknown msg type '{msg_type_str}' for {topic}, skipping."
                )
                continue
            self._ensure_cache_entry(topic)
            qos = qos_profile_sensor_data
            self.create_subscription(
                msg_cls, topic,
                lambda msg, t=topic: self._cache_cb(t, msg),
                qos, callback_group=self._cb,
            )
            self.get_logger().info(f"[ros2_mcp_bridge] Subscribed: {topic} ({msg_type_str})")

    def _ensure_cache_entry(self, topic: str):
        with self._cache_lock:
            if topic not in self._cache:
                self._cache[topic] = {"msg": None, "event": threading.Event()}

    def _cache_cb(self, topic: str, msg):
        """Generic callback: store latest message and signal waiters."""
        with self._cache_lock:
            entry = self._cache.setdefault(topic, {"msg": None, "event": threading.Event()})
            entry["msg"] = msg
            entry["event"].set()

    # ------------------------------------------------------------------ #
    # Deadman watchdog
    # ------------------------------------------------------------------ #

    def _deadman_watchdog(self):
        """Publish zero velocity if no cmd has been received recently."""
        with self._cmd_lock:
            if self._last_cmd_time > 0.0 and (time.time() - self._last_cmd_time) > 0.5:
                self._cmd_pub.publish(Twist())
                self._last_cmd_time = 0.0

    # ------------------------------------------------------------------ #
    # Public API used by MCP tools
    # ------------------------------------------------------------------ #

    def get_latest(self, topic: str, timeout: float = 3.0) -> Optional[Any]:
        """
        Block until a message arrives on *topic* (or reuse cached), up to
        *timeout* seconds.  Returns the raw rclpy message or None.
        """
        self._ensure_cache_entry(topic)
        entry = self._cache[topic]
        # If nothing cached yet, wait for first message
        if entry["msg"] is None:
            entry["event"].clear()
            if not entry["event"].wait(timeout):
                return None
        return entry["msg"]

    def publish_twist(self, linear_x: float, angular_z: float):
        """Publish a Twist to cmd_vel and reset the deadman timer."""
        t = Twist()
        t.linear.x = float(np.clip(linear_x, -0.22, 0.22))
        t.angular.z = float(np.clip(angular_z, -2.84, 2.84))
        self._cmd_pub.publish(t)
        with self._cmd_lock:
            self._last_cmd_time = time.time()

    def stop(self):
        """Immediately publish a zero-velocity Twist."""
        self._cmd_pub.publish(Twist())
        with self._cmd_lock:
            self._last_cmd_time = 0.0

    def list_topics(self) -> List[Tuple[str, List[str]]]:
        """Return the current topic list from the ROS graph."""
        return self.get_topic_names_and_types()

    def list_services(self) -> List[Tuple[str, List[str]]]:
        """Return the current service list from the ROS graph."""
        return self.get_service_names_and_types()

    # ------------------------------------------------------------------ #
    # Nav2 action client (lazy)
    # ------------------------------------------------------------------ #

    def navigate_to_pose(self, x: float, y: float, yaw: float = 0.0,
                         timeout: float = 60.0) -> dict:
        """
        Send a NavigateToPose goal to nav2 and block until done.
        Returns {"status": "succeeded"|"failed"|"timeout", "message": ...}.
        """
        try:
            from nav2_msgs.action import NavigateToPose
            from rclpy.action import ActionClient
            from geometry_msgs.msg import PoseStamped, Quaternion
        except ImportError:
            return {"status": "failed", "message": "nav2_msgs not available — install nav2."}

        with self._nav_lock:
            if self._nav_client is None:
                self._nav_client = ActionClient(
                    self, NavigateToPose, "/navigate_to_pose"
                )

        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            return {"status": "failed", "message": "NavigateToPose action server not available."}

        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        # Convert yaw → quaternion (rotation about Z)
        pose.pose.orientation = Quaternion(
            x=0.0, y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0),
        )
        goal.pose = pose

        done_event = threading.Event()
        result_holder = [None]

        def done_cb(future):
            result_holder[0] = future.result()
            done_event.set()

        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(
            lambda f: f.result().get_result_async().add_done_callback(done_cb)
        )

        if not done_event.wait(timeout):
            return {"status": "timeout", "message": f"Navigation timed out after {timeout}s."}

        status = result_holder[0].status  # 4 = succeeded, 6 = aborted
        if status == 4:
            return {"status": "succeeded", "message": "Reached goal."}
        return {"status": "failed", "message": f"Navigation failed (status={status})."}


# --------------------------------------------------------------------------- #
# Message type registry
# --------------------------------------------------------------------------- #

_MSG_TYPE_MAP = {
    "sensor_msgs/CompressedImage":   CompressedImage,
    "sensor_msgs/LaserScan":         LaserScan,
    "nav_msgs/Odometry":             Odometry,
    "vision_msgs/Detection2DArray":  Detection2DArray,
}


def _resolve_msg_type(type_str: str):
    """Return the rclpy message class for a 'pkg/Name' string, or None."""
    # Try built-in map first (fast path)
    cls = _MSG_TYPE_MAP.get(type_str)
    if cls:
        return cls
    # Generic resolution: split 'pkg_name/MsgName' and import
    try:
        parts = type_str.split("/")
        if len(parts) == 2:
            pkg, name = parts
            import importlib
            mod = importlib.import_module(f"{pkg}.msg")
            return getattr(mod, name)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Message → JSON serialisers  (used by MCP tools to build return strings)
# --------------------------------------------------------------------------- #

def compressed_image_to_dict(msg: CompressedImage) -> dict:
    """Return a small dict with base64-encoded JPEG bytes for the LLM."""
    return {
        "format": msg.format,
        "width_hint": "see image",
        "data_base64": base64.b64encode(bytes(msg.data)).decode(),
    }


def laser_scan_to_dict(msg: LaserScan) -> dict:
    """Simplify LaserScan into front/left/right/rear sector summaries."""
    ranges = np.array(msg.ranges, dtype=np.float32)
    ranges = np.where(
        (ranges < msg.range_min) | (ranges > msg.range_max),
        np.nan, ranges
    )
    n = len(ranges)

    def sector_min(start_deg, end_deg):
        i0 = int((start_deg - math.degrees(msg.angle_min)) /
                 math.degrees(msg.angle_increment)) % n
        i1 = int((end_deg   - math.degrees(msg.angle_min)) /
                 math.degrees(msg.angle_increment)) % n
        sec = ranges[i0:i1] if i0 < i1 else np.concatenate([ranges[i0:], ranges[:i1]])
        valid = sec[~np.isnan(sec)]
        return float(np.min(valid)) if len(valid) > 0 else None

    return {
        "range_min_m": msg.range_min,
        "range_max_m": msg.range_max,
        "front_min_m":  sector_min(-30, 30),
        "left_min_m":   sector_min(30, 90),
        "rear_min_m":   sector_min(90, 270),
        "right_min_m":  sector_min(270, 330),
        "full_ranges":  [round(float(r), 3) if not math.isnan(r) else None for r in ranges],
    }


def odometry_to_dict(msg: Odometry) -> dict:
    """Extract x, y, yaw from an Odometry message."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    # Yaw from quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return {
        "x": round(p.x, 4),
        "y": round(p.y, 4),
        "z": round(p.z, 4),
        "yaw_rad": round(yaw, 4),
        "yaw_deg": round(math.degrees(yaw), 2),
    }


def detections_to_dict(msg: Detection2DArray) -> dict:
    """Serialise Detection2DArray to a plain dict list."""
    items = []
    for d in msg.detections:
        if not d.results:
            continue
        h = d.results[0].hypothesis
        items.append({
            "label": h.class_id,
            "confidence": round(h.score, 3),
            "bbox": {
                "cx": round(d.bbox.center.position.x, 1),
                "cy": round(d.bbox.center.position.y, 1),
                "w":  round(d.bbox.size_x, 1),
                "h":  round(d.bbox.size_y, 1),
            },
        })
    return {"count": len(items), "detections": items}
