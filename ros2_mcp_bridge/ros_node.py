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
from sensor_msgs.msg import CompressedImage, Imu, JointState, LaserScan, BatteryState, MagneticField
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
        # Robot limits from config (used by publish_twist clipping)
        # ------------------------------------------------------------------ #
        robot_cfg = config.get("robot", {})
        self._max_linear_speed = float(robot_cfg.get("max_linear_speed", 0.22))
        self._max_angular_speed = float(robot_cfg.get("max_angular_speed", 2.84))

        # Wheel-radius calibration scale factor (1.0 = trust odometry as-is).
        # If the robot consistently over/under-shoots, adjust this value.
        # scale > 1.0 means odometry reports more distance than actual.
        self._wheel_radius_scale = float(
            robot_cfg.get("wheel_radius_scale", 1.0)
        )

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
        t.linear.x = float(np.clip(linear_x, -self._max_linear_speed,
                                    self._max_linear_speed))
        t.angular.z = float(np.clip(angular_z, -self._max_angular_speed,
                                     self._max_angular_speed))
        self._cmd_pub.publish(t)
        with self._cmd_lock:
            self._last_cmd_time = time.time()

    def publish_twist_for_duration(
        self, linear_x: float, angular_z: float, duration: float,
        rate_hz: float = 10.0,
    ):
        """
        Publish Twist at *rate_hz* for *duration* seconds, then stop.

        This keeps the deadman alive for the whole interval so the robot
        actually moves for the requested time.
        """
        period = 1.0 / rate_hz
        end_time = time.time() + duration
        while time.time() < end_time:
            self.publish_twist(linear_x, angular_z)
            time.sleep(period)
        self.stop()

    def stop(self):
        """Immediately publish a zero-velocity Twist."""
        self._cmd_pub.publish(Twist())
        with self._cmd_lock:
            self._last_cmd_time = 0.0

    # ------------------------------------------------------------------ #
    # Odometry helpers
    # ------------------------------------------------------------------ #

    def _get_odom_pose(self) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, yaw) from latest odometry, or None."""
        cfg_topics = self._cfg.get("topics", {})
        odom_topic = cfg_topics.get("odom", {}).get("topic", "/odom")
        msg = self.get_latest(odom_topic, timeout=1.0)
        if msg is None:
            return None
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (p.x, p.y, yaw)

    # ------------------------------------------------------------------ #
    # Closed-loop distance / rotation
    # ------------------------------------------------------------------ #

    def move_distance(
        self,
        distance_m: float,
        speed: float = 0.0,
        timeout: float = 30.0,
        collision_avoidance: bool = True,
    ) -> dict:
        """
        Drive *distance_m* metres using odometry feedback.  Positive = forward.
        *speed* is the cruise velocity (m/s); 0 = use 80% of max.
        Returns a dict with status, distance_requested, distance_actual.
        """
        if speed <= 0:
            speed = self._max_linear_speed * 0.8
        speed = min(speed, self._max_linear_speed)
        scale = self._wheel_radius_scale

        start = self._get_odom_pose()
        if start is None:
            return {"status": "failed",
                    "message": "Odometry not available."}

        # Check collision before starting
        direction = 1.0 if distance_m >= 0 else -1.0
        if collision_avoidance:
            ca_result = self.check_collision(direction * speed)
            if ca_result["blocked"]:
                self.stop()
                return {
                    "status": "blocked",
                    "collision_avoidance_activated": True,
                    "message": ca_result["message"],
                    "obstacle_distance_m": ca_result["distance_m"],
                    "distance_requested": round(distance_m, 4),
                    "distance_actual": 0.0,
                }

        target = abs(distance_m)
        rate_hz = 20.0
        period = 1.0 / rate_hz
        deadline = time.time() + timeout
        travelled = 0.0
        prev = start

        while travelled < target and time.time() < deadline:
            # Proportional slow-down in the last 0.05 m
            remaining = target - travelled
            cmd_speed = speed if remaining > 0.05 else max(0.05, speed * (remaining / 0.05))
            self.publish_twist(direction * cmd_speed, 0.0)
            time.sleep(period)

            cur = self._get_odom_pose()
            if cur is None:
                continue
            dx = cur[0] - prev[0]
            dy = cur[1] - prev[1]
            step = math.sqrt(dx * dx + dy * dy) / scale
            travelled += step
            prev = cur

            # Periodic collision check
            if collision_avoidance and int(travelled * 100) % 10 == 0:
                ca_result = self.check_collision(direction * cmd_speed)
                if ca_result["blocked"]:
                    self.stop()
                    return {
                        "status": "blocked",
                        "collision_avoidance_activated": True,
                        "message": ca_result["message"],
                        "distance_requested": round(distance_m, 4),
                        "distance_actual": round(direction * travelled, 4),
                    }

        self.stop()

        timed_out = travelled < target * 0.9
        return {
            "status": "timeout" if timed_out else "succeeded",
            "distance_requested": round(distance_m, 4),
            "distance_actual": round(direction * travelled, 4),
            "message": ("Timed out before reaching target distance."
                        if timed_out else "Distance reached."),
        }

    def rotate_angle(
        self,
        angle_deg: float,
        speed: float = 0.0,
        timeout: float = 20.0,
    ) -> dict:
        """
        Rotate *angle_deg* degrees using odometry feedback.
        Positive = counter-clockwise (left).  Negative = clockwise (right).
        *speed* is angular velocity (rad/s); 0 = use 50% of max.
        Returns dict with status, angle_requested, angle_actual.
        """
        if speed <= 0:
            speed = self._max_angular_speed * 0.5
        speed = min(speed, self._max_angular_speed)

        start = self._get_odom_pose()
        if start is None:
            return {"status": "failed",
                    "message": "Odometry not available."}

        target_rad = abs(math.radians(angle_deg))
        direction = 1.0 if angle_deg >= 0 else -1.0
        rate_hz = 20.0
        period = 1.0 / rate_hz
        deadline = time.time() + timeout
        rotated = 0.0
        prev_yaw = start[2]

        while rotated < target_rad and time.time() < deadline:
            remaining = target_rad - rotated
            # Slow down in the last 10 degrees
            if remaining < math.radians(10):
                cmd_speed = max(0.15, speed * (remaining / math.radians(10)))
            else:
                cmd_speed = speed
            self.publish_twist(0.0, direction * cmd_speed)
            time.sleep(period)

            cur = self._get_odom_pose()
            if cur is None:
                continue
            # Signed angular delta, taking wraparound into account
            d_yaw = cur[2] - prev_yaw
            if d_yaw > math.pi:
                d_yaw -= 2 * math.pi
            elif d_yaw < -math.pi:
                d_yaw += 2 * math.pi
            rotated += abs(d_yaw)
            prev_yaw = cur[2]

        self.stop()

        timed_out = rotated < target_rad * 0.9
        actual_deg = math.degrees(rotated) * direction
        return {
            "status": "timeout" if timed_out else "succeeded",
            "angle_requested_deg": round(angle_deg, 2),
            "angle_actual_deg": round(actual_deg, 2),
            "message": ("Timed out before reaching target angle."
                        if timed_out else "Rotation complete."),
        }

    def calibrate_motion(self, distance_m: float = 1.0, speed: float = 0.0,
                         timeout: float = 30.0) -> dict:
        """
        Drive *distance_m* according to odometry (ignoring scale), stop, and
        return the odom-reported distance.  The user measures actual distance
        and computes: new_scale = odom_reported / actual_measured.
        """
        if speed <= 0:
            speed = self._max_linear_speed * 0.8
        speed = min(speed, self._max_linear_speed)

        start = self._get_odom_pose()
        if start is None:
            return {"status": "failed",
                    "message": "Odometry not available."}

        # Drive using raw odometry (scale = 1.0)
        target = abs(distance_m)
        direction = 1.0 if distance_m >= 0 else -1.0
        rate_hz = 20.0
        period = 1.0 / rate_hz
        deadline = time.time() + timeout
        travelled = 0.0
        prev = start

        while travelled < target and time.time() < deadline:
            remaining = target - travelled
            cmd_speed = speed if remaining > 0.05 else max(0.05, speed * (remaining / 0.05))
            self.publish_twist(direction * cmd_speed, 0.0)
            time.sleep(period)

            cur = self._get_odom_pose()
            if cur is None:
                continue
            dx = cur[0] - prev[0]
            dy = cur[1] - prev[1]
            step = math.sqrt(dx * dx + dy * dy)
            travelled += step
            prev = cur

        self.stop()
        return {
            "status": "succeeded",
            "distance_requested_m": round(distance_m, 4),
            "odometry_reported_m": round(direction * travelled, 4),
            "current_wheel_radius_scale": self._wheel_radius_scale,
            "instructions": (
                "Measure the actual distance the robot moved with a tape measure. "
                "Then compute: new_scale = odometry_reported_m / actual_measured_m. "
                "Set robot.wheel_radius_scale in bridge.yaml to this value. "
                "For example, if odometry reported 1.0 m but the robot only moved "
                "0.85 m, set wheel_radius_scale to 1.176 (= 1.0 / 0.85). "
                "Alternatively, adjust wheel_radius in the ROS 2 param file "
                "(burger.yaml or burger_pico.yaml): "
                "new_radius = current_radius * (actual / odometry_reported)."
            ),
        }

    def check_collision(self, linear_x: float) -> dict:
        """
        Read the latest LiDAR scan and check whether *linear_x* motion would
        drive the robot toward a nearby obstacle.

        Returns a dict::

            {
                "blocked":     bool,          # True → suppress motion
                "distance_m":  float | None,  # measured obstacle distance
                "sector":      str | None,    # "front" | "rear"
                "message":     str,           # human/LLM-readable explanation
            }

        If no LiDAR data is available the call returns ``blocked=False`` so
        that the robot can still move (fail-open; caller should log the
        "message" field).
        """
        ca_cfg = self._cfg.get("collision_avoidance", {})
        min_front = float(ca_cfg.get("min_front_distance", 0.30))
        min_rear  = float(ca_cfg.get("min_rear_distance",  0.20))

        cfg_topics = self._cfg.get("topics", {})
        laser_topic = cfg_topics.get("laser", {}).get("topic", "/scan")
        scan_msg = self.get_latest(laser_topic, timeout=0.5)

        if scan_msg is None:
            return {
                "blocked": False,
                "distance_m": None,
                "sector": None,
                "message": "LiDAR not available; collision avoidance skipped.",
            }

        scan = laser_scan_to_dict(scan_msg)

        if linear_x > 0.0:
            dist = scan.get("front_min_m")
            if dist is not None and dist < min_front:
                return {
                    "blocked": True,
                    "distance_m": round(dist, 3),
                    "sector": "front",
                    "message": (
                        f"Collision avoidance activated: obstacle {dist:.2f} m ahead "
                        f"(threshold {min_front:.2f} m). Forward motion suppressed."
                    ),
                }
        elif linear_x < 0.0:
            dist = scan.get("rear_min_m")
            if dist is not None and dist < min_rear:
                return {
                    "blocked": True,
                    "distance_m": round(dist, 3),
                    "sector": "rear",
                    "message": (
                        f"Collision avoidance activated: obstacle {dist:.2f} m behind "
                        f"(threshold {min_rear:.2f} m). Backward motion suppressed."
                    ),
                }

        return {"blocked": False, "distance_m": None, "sector": None, "message": ""}

    def list_topics(self) -> List[Tuple[str, List[str]]]:
        """Return the current topic list from the ROS graph."""
        return self.get_topic_names_and_types()

    def list_services(self) -> List[Tuple[str, List[str]]]:
        """Return the current service list from the ROS graph."""
        return self.get_service_names_and_types()

    def play_sound(self, value: int, timeout: float = 5.0) -> dict:
        """Call the /sound service to play a TurtleBot3 sound.

        Values: 0=OFF, 1=ON, 2=LOW_BATTERY, 3=ERROR, 4=BUTTON1, 5=BUTTON2.
        Works on both stock OpenCR and Pico variant.
        """
        try:
            from turtlebot3_msgs.srv import Sound
        except ImportError:
            return {"status": "failed",
                    "message": "turtlebot3_msgs not installed."}
        client = self.create_client(Sound, "/sound")
        if not client.wait_for_service(timeout_sec=timeout):
            return {"status": "failed",
                    "message": "/sound service not available."}
        req = Sound.Request()
        req.value = int(value)
        future = client.call_async(req)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            return {"status": "failed", "message": "Service call timed out."}
        result = future.result()
        return {"status": "succeeded" if result.success else "failed",
                "message": result.message}

    def set_motor_power(self, enable: bool, timeout: float = 5.0) -> dict:
        """Call the /motor_power service to enable or disable wheel torque."""
        from std_srvs.srv import SetBool
        client = self.create_client(SetBool, "/motor_power")
        if not client.wait_for_service(timeout_sec=timeout):
            return {"status": "failed",
                    "message": "/motor_power service not available."}
        req = SetBool.Request()
        req.data = bool(enable)
        future = client.call_async(req)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            return {"status": "failed", "message": "Service call timed out."}
        result = future.result()
        return {"status": "succeeded" if result.success else "failed",
                "message": result.message}

    def reset_odometry(self, timeout: float = 5.0) -> dict:
        """Call the /reset_odometry service to zero the odometry pose."""
        from std_srvs.srv import Trigger
        client = self.create_client(Trigger, "/reset_odometry")
        if not client.wait_for_service(timeout_sec=timeout):
            return {"status": "failed",
                    "message": "/reset_odometry service not available."}
        req = Trigger.Request()
        future = client.call_async(req)
        # Block until result (safe — called from MCP thread, not rclpy executor)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            return {"status": "failed", "message": "Service call timed out."}
        result = future.result()
        return {"status": "succeeded" if result.success else "failed",
                "message": result.message}

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
    "sensor_msgs/Imu":               Imu,
    "sensor_msgs/JointState":        JointState,
    "sensor_msgs/LaserScan":         LaserScan,
    "sensor_msgs/BatteryState":      BatteryState,
    "sensor_msgs/MagneticField":     MagneticField,
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


def sensor_state_to_dict(msg) -> dict:
    """Serialise turtlebot3_msgs/SensorState to a plain dict."""
    return {
        "battery_voltage": round(msg.battery, 3),
        "torque_enabled": bool(msg.torque),
        "left_encoder": msg.left_encoder,
        "right_encoder": msg.right_encoder,
        "bumper": msg.bumper,
    }


def battery_state_to_dict(msg) -> dict:
    """Serialise sensor_msgs/BatteryState to a plain dict."""
    return {
        "voltage": round(msg.voltage, 3),
        "percentage": round(msg.percentage, 2) if not math.isnan(msg.percentage) else None,
        "present": bool(msg.present),
    }


def imu_to_dict(msg) -> dict:
    """Serialise sensor_msgs/Imu to a plain dict with Euler angles."""
    q = msg.orientation
    # Quaternion → roll / pitch / yaw
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    av = msg.angular_velocity
    la = msg.linear_acceleration
    return {
        "orientation": {
            "roll_deg":  round(math.degrees(roll), 2),
            "pitch_deg": round(math.degrees(pitch), 2),
            "yaw_deg":   round(math.degrees(yaw), 2),
        },
        "angular_velocity": {
            "x": round(av.x, 4),
            "y": round(av.y, 4),
            "z": round(av.z, 4),
        },
        "linear_acceleration": {
            "x": round(la.x, 4),
            "y": round(la.y, 4),
            "z": round(la.z, 4),
        },
    }


def joint_state_to_dict(msg) -> dict:
    """Serialise sensor_msgs/JointState to a plain dict."""
    joints = {}
    for i, name in enumerate(msg.name):
        joints[name] = {
            "position_rad": round(msg.position[i], 4) if i < len(msg.position) else None,
            "velocity_rad_s": round(msg.velocity[i], 4) if i < len(msg.velocity) else None,
        }
    return {"joints": joints}


def magnetic_field_to_dict(msg) -> dict:
    """Serialise sensor_msgs/MagneticField to a plain dict.

    Values are in Tesla (SI).  Earth's field is typically 25–65 µT total,
    so expect values on the order of ±50e-6 T.
    """
    f = msg.magnetic_field
    magnitude = math.sqrt(f.x**2 + f.y**2 + f.z**2)
    return {
        "x_T": round(f.x, 9),
        "y_T": round(f.y, 9),
        "z_T": round(f.z, 9),
        "magnitude_T": round(magnitude, 9),
        "magnitude_uT": round(magnitude * 1e6, 3),
    }
