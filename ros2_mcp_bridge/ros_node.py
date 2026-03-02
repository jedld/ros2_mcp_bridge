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
from sensor_msgs.msg import CompressedImage, CameraInfo, Imu, JointState, LaserScan, BatteryState, MagneticField
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

        # Stop / cancel event: set by stop() to abort running behaviours
        self._stop_event = threading.Event()

        # Nav2 action client (created lazily)
        self._nav_client = None
        self._nav_lock = threading.Lock()

        # Active Nav2 goal handle (for cancel) and initial-pose publisher (lazy)
        self._nav_goal_handle = None
        self._nav_goal_lock = threading.Lock()
        self._initial_pose_pub = None
        self._initial_pose_lock = threading.Lock()

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
                self._cache[topic] = {"msg": None, "event": threading.Event(), "stamp": 0.0}

    def _cache_cb(self, topic: str, msg):
        """Generic callback: store latest message and signal waiters."""
        with self._cache_lock:
            entry = self._cache.setdefault(topic, {"msg": None, "event": threading.Event(), "stamp": 0.0})
            entry["msg"] = msg
            entry["stamp"] = time.monotonic()
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

    def get_fresh(self, topic: str, timeout: float = 3.0) -> Optional[Any]:
        """
        Block until a NEW message arrives on *topic*, ignoring any
        previously cached value.  Essential for behaviours that need
        the latest sensor reading after the robot has moved or stopped.

        Unlike ``get_latest`` (which returns the cached message
        immediately), this clears the arrival flag and waits for the
        next publication.  Returns None if no message arrives within
        *timeout* seconds.
        """
        self._ensure_cache_entry(topic)
        entry = self._cache[topic]
        entry["event"].clear()
        if not entry["event"].wait(timeout):
            return None
        return entry["msg"]

    def get_cache_age(self, topic: str) -> float:
        """Seconds since the last message on *topic*.  Returns ``inf`` if
        no message has ever been received."""
        with self._cache_lock:
            entry = self._cache.get(topic)
            if entry is None or entry["stamp"] == 0.0:
                return float('inf')
            return time.monotonic() - entry["stamp"]

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
            if self._stop_event.is_set():
                break
            self.publish_twist(linear_x, angular_z)
            time.sleep(period)
        self._cmd_pub.publish(Twist())

    def stop(self):
        """Immediately publish a zero-velocity Twist and signal running behaviours to cancel."""
        self._stop_event.set()
        self._cmd_pub.publish(Twist())
        with self._cmd_lock:
            self._last_cmd_time = 0.0

    def clear_stop_event(self):
        """Clear the cancellation flag.  Call at the start of a new behaviour."""
        self._stop_event.clear()

    def is_stop_requested(self) -> bool:
        """Return True if stop() has been called since last clear_stop_event()."""
        return self._stop_event.is_set()

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

    def _recover_from_stuck(
        self,
        direction: float,
        speed: float,
        travelled: float,
        distance_m: float,
    ) -> dict:
        """
        Called when stuck is detected.  Reverses 0.15 m (ignoring collision
        avoidance) to un-wedge the robot, then returns a 'stuck' result dict
        so the caller (and ultimately the LLM) knows what happened.
        """
        self.stop()
        reverse_dist = 0.15
        rev = self.move_distance(
            -direction * reverse_dist,
            speed=max(0.05, speed * 0.5),
            timeout=6.0,
            collision_avoidance=False,
        )
        return {
            "status": "stuck",
            "distance_requested": round(distance_m, 4),
            "distance_actual": round(direction * travelled, 4),
            "message": (
                f"Robot stuck after {round(travelled, 3)} m — motor commands "
                f"were active but odometry showed no movement. "
                f"Reversed {abs(rev.get('distance_actual', 0.0)):.2f} m to recover."
            ),
            "recovery_action": "reversed",
            "recovery_distance_m": rev.get("distance_actual", 0.0),
        }

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

        # Stuck detection: consecutive loops with negligible odom progress
        # while commanding meaningful speed → robot is wedged / high-centred.
        _STUCK_STEP_THRESH = 0.003   # m per 20 Hz loop ≈ 0.06 m/s effective
        _STUCK_CMD_THRESH  = 0.08    # only flag if cmd ≥ this (ignore slow ramp)
        _STUCK_MAX_COUNT   = 20      # 20 × 50 ms = 1.0 s of zero progress
        stuck_count = 0

        while travelled < target and time.time() < deadline:
            if self._stop_event.is_set():
                self._cmd_pub.publish(Twist())
                return {
                    "status": "cancelled",
                    "distance_requested": round(distance_m, 4),
                    "distance_actual": round(direction * travelled, 4),
                    "message": "Motion cancelled by stop command.",
                }
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

            # Stuck detection
            if cmd_speed >= _STUCK_CMD_THRESH and step < _STUCK_STEP_THRESH:
                stuck_count += 1
                if stuck_count >= _STUCK_MAX_COUNT:
                    return self._recover_from_stuck(direction, speed, travelled, distance_m)
            else:
                stuck_count = 0

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

        The rotation uses a three-phase approach to prevent overshoot:

        1. **Cruise** — full speed until a configurable deceleration zone
           (default 30°) before the target.
        2. **Deceleration / pulse** — instead of ramping to a low continuous
           speed (which the firmware's dead-zone compensation boosts to
           a high duty anyway), the robot uses short motor pulses separated
           by coast+measure gaps.  This gives the firmware no chance to
           apply sustained dead-zone thrust.
        3. **Coast+verify** — after each pulse (or when the target is
           nearly reached) the motors are stopped and odometry is polled
           for ~200 ms to account for rotational inertia before deciding
           whether more rotation is needed.
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

        # ── Tuning knobs ──────────────────────────────────────────────── #
        # Deceleration zone: start pulsing this far from the target.
        DECEL_ZONE_RAD = math.radians(30)
        # Early stop: stop motors this far before the target to allow
        # coasting/inertia to cover the remaining distance.
        EARLY_STOP_RAD = math.radians(5)
        # Pulse mode: in the decel zone, drive for PULSE_ON then coast
        # for PULSE_OFF to measure actual position.
        PULSE_ON_S   = 0.10   # 100 ms motor pulse
        PULSE_OFF_S  = 0.20   # 200 ms coast/measure gap
        # Coast settle: after final stop, wait this long and re-measure.
        COAST_SETTLE_S = 0.35

        # Stuck detection for rotation: wheel drag / carpet / obstacle
        _STUCK_YAW_THRESH = 0.002    # rad per 20 Hz loop ≈ 0.04 rad/s effective
        _STUCK_CMD_THRESH = 0.25     # only flag if cmd ≥ this rad/s (ignore slow ramp)
        _STUCK_MAX_COUNT  = 20       # 20 × 50 ms = 1.0 s of zero angular progress
        stuck_count = 0

        def _read_yaw_delta() -> float:
            """Read odom and accumulate rotation. Returns the delta."""
            nonlocal rotated, prev_yaw
            cur = self._get_odom_pose()
            if cur is None:
                return 0.0
            d_yaw = cur[2] - prev_yaw
            if d_yaw > math.pi:
                d_yaw -= 2 * math.pi
            elif d_yaw < -math.pi:
                d_yaw += 2 * math.pi
            rotated += abs(d_yaw)
            prev_yaw = cur[2]
            return abs(d_yaw)

        def _coast_and_settle(settle_s: float):
            """Stop motors and spin odom for settle_s to catch inertia."""
            self._cmd_pub.publish(Twist())
            t_end = time.time() + settle_s
            while time.time() < t_end:
                time.sleep(period)
                _read_yaw_delta()

        # ── Main rotation loop ────────────────────────────────────────── #
        in_pulse_mode = False

        while rotated < target_rad and time.time() < deadline:
            if self._stop_event.is_set():
                self._cmd_pub.publish(Twist())
                actual_deg = math.degrees(rotated) * direction
                return {
                    "status": "cancelled",
                    "angle_requested_deg": round(angle_deg, 2),
                    "angle_actual_deg": round(actual_deg, 2),
                    "message": "Rotation cancelled by stop command.",
                }

            remaining = target_rad - rotated

            # ── Phase: early stop for coasting ─────────────────────── #
            if remaining < EARLY_STOP_RAD:
                _coast_and_settle(COAST_SETTLE_S)
                break  # good enough — inertia will cover the rest

            # ── Phase: pulse mode (deceleration zone) ──────────────── #
            if remaining < DECEL_ZONE_RAD:
                if not in_pulse_mode:
                    in_pulse_mode = True

                # Short burst
                pulse_speed = max(0.15, speed * 0.35)
                t_end = time.time() + PULSE_ON_S
                while time.time() < t_end and rotated < (target_rad - EARLY_STOP_RAD):
                    self.publish_twist(0.0, direction * pulse_speed)
                    time.sleep(period)
                    _read_yaw_delta()

                # Coast + measure
                _coast_and_settle(PULSE_OFF_S)
                continue

            # ── Phase: cruise (full speed) ─────────────────────────── #
            cmd_speed = speed
            self.publish_twist(0.0, direction * cmd_speed)
            time.sleep(period)
            d_yaw = _read_yaw_delta()

            # Stuck detection (only during cruise)
            if cmd_speed >= _STUCK_CMD_THRESH and d_yaw < _STUCK_YAW_THRESH:
                stuck_count += 1
                if stuck_count >= _STUCK_MAX_COUNT:
                    self.stop()
                    rev = self.move_distance(
                        -0.10,
                        speed=self._max_linear_speed * 0.4,
                        timeout=4.0,
                        collision_avoidance=False,
                    )
                    actual_deg = math.degrees(rotated) * direction
                    return {
                        "status": "stuck",
                        "angle_requested_deg": round(angle_deg, 2),
                        "angle_actual_deg": round(actual_deg, 2),
                        "message": (
                            f"Robot stuck during rotation after "
                            f"{round(math.degrees(rotated), 1)}° — "
                            f"motor commands active but odometry showed no "
                            f"angular movement. Reversed "
                            f"{abs(rev.get('distance_actual', 0.0)):.2f} m to recover."
                        ),
                        "recovery_action": "reversed",
                        "recovery_distance_m": rev.get("distance_actual", 0.0),
                    }
            else:
                stuck_count = 0

        # ── Final coast settle ────────────────────────────────────────── #
        self._cmd_pub.publish(Twist())
        _coast_and_settle(COAST_SETTLE_S)

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
    # Full-resolution image capture (Pi camera service)
    # ------------------------------------------------------------------ #

    def capture_full_res(
        self,
        width: int = 0,
        height: int = 0,
        jpeg_quality: int = 90,
        timeout: float = 20.0,
    ) -> dict:
        """
        Call the /camera/capture_full_res service on the Pi to capture a
        full-resolution image on demand.

        Args:
            width:        Requested image width in pixels (0 = camera native).
            height:       Requested image height in pixels (0 = camera native).
            jpeg_quality: JPEG compression quality 1-100 (default 90).
            timeout:      Maximum seconds to wait for the service (default 20).

        Returns:
            On success::
                {
                    "status": "succeeded",
                    "width": int,
                    "height": int,
                    "jpeg_b64": str,   # base-64 encoded JPEG bytes
                    "message": str,
                }
            On failure::
                {"status": "failed", "message": str}
        """
        try:
            from turtlebot3_camera_interfaces.srv import CaptureImage
        except ImportError:
            return {
                "status": "failed",
                "message": (
                    "turtlebot3_camera_interfaces not installed. "
                    "Build the package and source the workspace."
                ),
            }

        client = self.create_client(CaptureImage, "/camera/capture_full_res")
        if not client.wait_for_service(timeout_sec=min(timeout, 10.0)):
            return {
                "status": "failed",
                "message": (
                    "/camera/capture_full_res service not available. "
                    "Ensure the camera node is running on the robot."
                ),
            }

        req = CaptureImage.Request()
        req.width = int(width)
        req.height = int(height)
        req.jpeg_quality = int(max(1, min(100, jpeg_quality)))

        future = client.call_async(req)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)

        if not future.done():
            return {"status": "failed", "message": "capture_full_res service call timed out."}

        result = future.result()
        if not result.success:
            return {"status": "failed", "message": result.message or "Capture failed."}

        jpeg_bytes = bytes(result.image.data)
        jpeg_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

        # Extract actual dimensions from the image header if available
        actual_w = getattr(result.image, "width", width) or width
        actual_h = getattr(result.image, "height", height) or height

        return {
            "status": "succeeded",
            "width": actual_w,
            "height": actual_h,
            "jpeg_b64": jpeg_b64,
            "message": result.message or "Image captured.",
        }

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

        def goal_response_cb(f):
            goal_handle = f.result()
            with self._nav_goal_lock:
                self._nav_goal_handle = goal_handle
            goal_handle.get_result_async().add_done_callback(done_cb)

        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(goal_response_cb)

        if not done_event.wait(timeout):
            with self._nav_goal_lock:
                self._nav_goal_handle = None
            return {"status": "timeout", "message": f"Navigation timed out after {timeout}s."}

        with self._nav_goal_lock:
            self._nav_goal_handle = None
        status = result_holder[0].status  # 4 = succeeded, 6 = aborted
        if status == 4:
            return {"status": "succeeded", "message": "Reached goal."}
        return {"status": "failed", "message": f"Navigation failed (status={status})."}

    def cancel_navigation(self, timeout: float = 5.0) -> dict:
        """Cancel the currently active Nav2 navigation goal."""
        with self._nav_goal_lock:
            handle = self._nav_goal_handle
        if handle is None:
            return {"status": "no_active_goal"}
        done_ev = threading.Event()
        cancel_future = handle.cancel_goal_async()
        cancel_future.add_done_callback(lambda _f: done_ev.set())
        done_ev.wait(timeout)
        with self._nav_goal_lock:
            self._nav_goal_handle = None
        return {"status": "cancel_requested"}

    def set_initial_pose(
        self,
        x: float,
        y: float,
        yaw_rad: float,
        cov_xy: float = 0.25,
        cov_yaw: float = 0.0685,
    ) -> dict:
        """Publish a PoseWithCovarianceStamped to /initialpose for AMCL."""
        try:
            from geometry_msgs.msg import PoseWithCovarianceStamped
        except ImportError:
            return {"status": "failed", "message": "geometry_msgs not available."}
        with self._initial_pose_lock:
            if self._initial_pose_pub is None:
                self._initial_pose_pub = self.create_publisher(
                    PoseWithCovarianceStamped, "/initialpose", 10
                )
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
        # Diagonal covariance [x, y, z, roll, pitch, yaw] (6x6 row-major)
        cov = [0.0] * 36
        cov[0]  = cov_xy   # x variance
        cov[7]  = cov_xy   # y variance
        cov[35] = cov_yaw  # yaw variance
        msg.pose.covariance = cov
        self._initial_pose_pub.publish(msg)
        self.get_logger().info(
            f"[ros2_mcp_bridge] Published initial pose: x={x:.3f} y={y:.3f} yaw={math.degrees(yaw_rad):.1f}°"
        )
        return {"status": "published", "x": x, "y": y, "yaw_rad": yaw_rad,
                "yaw_deg": round(math.degrees(yaw_rad), 2)}

    def subscribe_on_demand(self, topic: str, msg_cls, timeout: float = 3.0):
        """Subscribe to *topic* lazily (if not already subscribed) and return latest message."""
        with self._cache_lock:
            already_subscribed = topic in self._cache
        if not already_subscribed:
            self._ensure_cache_entry(topic)
            self.create_subscription(
                msg_cls, topic,
                lambda msg, t=topic: self._cache_cb(t, msg),
                qos_profile_sensor_data, callback_group=self._cb,
            )
            self.get_logger().info(f"[ros2_mcp_bridge] On-demand subscription: {topic}")
        return self.get_latest(topic, timeout)

    def get_map_metadata(self, timeout: float = 3.0) -> dict | None:
        """Return the current map metadata from /map_metadata, or None."""
        try:
            from nav_msgs.msg import MapMetaData
        except ImportError:
            return None
        msg = self.subscribe_on_demand("/map_metadata", MapMetaData, timeout)
        if msg is None:
            return None
        width_m  = round(msg.width  * msg.resolution, 2)
        height_m = round(msg.height * msg.resolution, 2)
        return {
            "resolution_m_per_cell": round(msg.resolution, 4),
            "width_cells":  msg.width,
            "height_cells": msg.height,
            "width_m":  width_m,
            "height_m": height_m,
            "origin_x": round(msg.origin.position.x, 4),
            "origin_y": round(msg.origin.position.y, 4),
            "total_cells": msg.width * msg.height,
        }


# --------------------------------------------------------------------------- #
# Message type registry
# --------------------------------------------------------------------------- #

_MSG_TYPE_MAP = {
    "sensor_msgs/CameraInfo":        CameraInfo,
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


# --------------------------------------------------------------------------- #
# Distance estimation helpers (camera ↔ LiDAR fusion)
# --------------------------------------------------------------------------- #

# Approximate real-world heights (metres) for common COCO classes.
# Used by the pinhole-model fallback when LiDAR has no return for an object.
_COCO_TYPICAL_HEIGHT_M: dict[str, float] = {
    "person": 1.70, "bicycle": 1.10, "car": 1.50, "motorcycle": 1.10,
    "bus": 3.00, "truck": 2.80, "cat": 0.30, "dog": 0.50,
    "chair": 0.80, "couch": 0.85, "dining table": 0.75,
    "bed": 0.60, "toilet": 0.45, "tv": 0.40, "laptop": 0.25,
    "mouse": 0.04, "keyboard": 0.04, "cell phone": 0.15,
    "microwave": 0.30, "oven": 0.85, "refrigerator": 1.70,
    "book": 0.25, "clock": 0.30, "vase": 0.30, "cup": 0.12,
    "bottle": 0.25, "wine glass": 0.22, "fork": 0.03, "knife": 0.03,
    "spoon": 0.03, "bowl": 0.10, "banana": 0.05, "apple": 0.08,
    "sandwich": 0.08, "orange": 0.08, "backpack": 0.50, "umbrella": 1.00,
    "handbag": 0.30, "suitcase": 0.55, "potted plant": 0.40,
    "teddy bear": 0.35, "toothbrush": 0.02, "remote": 0.06,
    "scissors": 0.10, "sports ball": 0.22,
}


def lidar_distance_for_bbox(
    scan_msg: LaserScan,
    cx: float,
    bbox_w: float,
    image_width: float = 640.0,
    hfov_deg: float = 62.0,
    min_angular_window_deg: float = 5.0,
) -> tuple[float | None, int]:
    """Map a detection bbox to LiDAR angles and sample those rays.

    Parameters
    ----------
    scan_msg : LaserScan
        Raw LiDAR message.
    cx : float
        Horizontal centre of the bounding box in pixels.
    bbox_w : float
        Width of the bounding box in pixels.
    image_width : float
        Image width in pixels (default 640).
    hfov_deg : float
        Camera horizontal field-of-view in degrees.
    min_angular_window_deg : float
        Minimum half-width of the angular sampling window (degrees).

    Returns
    -------
    (distance_m, n_valid_rays) — distance_m is None if no valid rays found.
    """
    half_hfov = hfov_deg / 2.0

    # Pixel offset from image centre → angle in degrees
    # Positive angle = right of centre in image → negative LiDAR angle
    # (LiDAR convention: counter-clockwise positive, camera x: left-to-right)
    pixel_offset = cx - image_width / 2.0
    centre_angle_deg = -(pixel_offset / (image_width / 2.0)) * half_hfov

    # Angular half-width of the bbox
    bbox_half_angle_deg = max(
        (bbox_w / image_width) * half_hfov,
        min_angular_window_deg,
    )

    start_deg = centre_angle_deg - bbox_half_angle_deg
    end_deg = centre_angle_deg + bbox_half_angle_deg

    # Sample LiDAR rays in [start_deg, end_deg]
    ranges = np.array(scan_msg.ranges, dtype=np.float32)
    ranges = np.where(
        (ranges < scan_msg.range_min) | (ranges > scan_msg.range_max),
        np.nan, ranges,
    )
    n = len(ranges)
    inc_deg = math.degrees(scan_msg.angle_increment)
    min_deg = math.degrees(scan_msg.angle_min)

    i0 = int((start_deg - min_deg) / inc_deg) % n
    i1 = int((end_deg - min_deg) / inc_deg) % n

    if i0 < i1:
        sector = ranges[i0:i1 + 1]
    else:
        sector = np.concatenate([ranges[i0:], ranges[: i1 + 1]])

    valid = sector[~np.isnan(sector)]
    if len(valid) == 0:
        return None, 0
    return float(np.min(valid)), int(len(valid))


def bearing_from_bbox(
    cx: float,
    image_width: float = 640.0,
    hfov_deg: float = 62.0,
) -> float:
    """Convert a bounding-box centre x-pixel to a bearing angle in radians.

    Returns a signed angle: positive = object is to the *left* of the
    camera centre (counter-clockwise positive, matching the LiDAR /
    ROS convention).
    """
    half_hfov = hfov_deg / 2.0
    pixel_offset = cx - image_width / 2.0
    bearing_deg = -(pixel_offset / (image_width / 2.0)) * half_hfov
    return math.radians(bearing_deg)


def motion_stereo_depth(
    node: "ROS2BridgeNode",
    cam_topic: str,
    bbox: dict,
    image_width: float = 640.0,
    hfov_deg: float = 62.0,
    baseline_m: float = 0.08,
) -> float | None:
    """Estimate depth at a bounding-box location using motion parallax.

    The robot nudges forward by *baseline_m*, captures two JPEG frames
    (before / after), matches ORB features inside the bbox region, and
    uses the median horizontal disparity to triangulate depth:

        depth = focal_px * baseline_m / median_disparity

    Returns the estimated depth in metres, or None on failure.  The robot
    is returned to its original position afterward.
    """
    import cv2

    focal_px = (image_width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))

    # Region of interest (expand bbox by 20% for better feature matching)
    cx, cy = bbox["cx"], bbox["cy"]
    bw, bh = bbox["w"], bbox["h"]
    margin = 0.2
    x1 = int(max(0, cx - bw / 2 * (1 + margin)))
    y1 = int(max(0, cy - bh / 2 * (1 + margin)))
    x2 = int(cx + bw / 2 * (1 + margin))
    y2 = int(cy + bh / 2 * (1 + margin))

    # Capture frame 1
    msg1 = node.get_fresh(cam_topic, timeout=1.5)
    if msg1 is None:
        return None
    buf1 = np.frombuffer(bytes(msg1.data), dtype=np.uint8)
    img1 = cv2.imdecode(buf1, cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        return None

    # Nudge forward
    node.move_distance(baseline_m, speed=0.05, timeout=5.0, collision_avoidance=True)

    # Capture frame 2
    msg2 = node.get_fresh(cam_topic, timeout=1.5)
    if msg2 is None:
        # Nudge back
        node.move_distance(-baseline_m, speed=0.05, timeout=5.0, collision_avoidance=False)
        return None
    buf2 = np.frombuffer(bytes(msg2.data), dtype=np.uint8)
    img2 = cv2.imdecode(buf2, cv2.IMREAD_GRAYSCALE)
    if img2 is None:
        node.move_distance(-baseline_m, speed=0.05, timeout=5.0, collision_avoidance=False)
        return None

    # Nudge back to original position
    node.move_distance(-baseline_m, speed=0.05, timeout=5.0, collision_avoidance=False)

    # Clamp ROI to image bounds
    h, w = img1.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    roi1 = img1[y1:y2, x1:x2]
    roi2 = img2[y1:y2, x1:x2]

    # ORB feature matching
    orb = cv2.ORB_create(nfeatures=200)
    kp1, des1 = orb.detectAndCompute(roi1, None)
    kp2, des2 = orb.detectAndCompute(roi2, None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 3:
        return None

    # Compute horizontal disparities (motion is forward → points move outward
    # from the focus of expansion at image centre)
    disparities = []
    for m in matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        dx = abs(pt2[0] - pt1[0])
        if dx > 0.5:  # ignore sub-pixel noise
            disparities.append(dx)

    if len(disparities) < 3:
        return None

    median_disp = float(np.median(disparities))
    if median_disp < 1.0:
        return None  # too small → object is very far away, unreliable

    depth = focal_px * baseline_m / median_disp
    # Sanity: clamp to 0.1 – 10 m
    if depth < 0.1 or depth > 10.0:
        return None

    return round(depth, 3)


def bbox_depth_estimate(
    bbox_h: float,
    label: str,
    image_height: float = 480.0,
    image_width: float = 640.0,
    hfov_deg: float = 62.0,
) -> float | None:
    """Pinhole-model distance estimate from bounding-box height.

    Uses the known typical height of common COCO objects.
    Returns None if the class is not in the lookup table.
    """
    real_h = _COCO_TYPICAL_HEIGHT_M.get(label.lower())
    if real_h is None or bbox_h < 5:
        return None
    focal_px = (image_width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    return round(real_h * focal_px / bbox_h, 3)


def estimate_detection_distance(
    scan_msg,
    det: dict,
    image_width: float = 640.0,
    image_height: float = 480.0,
    hfov_deg: float = 62.0,
) -> dict:
    """Fuse LiDAR + bbox heuristic for one detection.

    Parameters
    ----------
    scan_msg : LaserScan | None
        Raw LiDAR message (may be None).
    det : dict
        Single detection dict with keys: label, confidence, bbox {cx, cy, w, h}.
    image_width, image_height, hfov_deg : float
        Camera parameters.

    Returns
    -------
    dict with keys:
        distance_m : float | None
        distance_source : "lidar" | "bbox_estimate" | "fused" | "none"
        distance_reliable : bool
        lidar_m : float | None
        bbox_estimate_m : float | None
        lidar_n_rays : int
    """
    bbox = det["bbox"]
    lidar_m = None
    lidar_n_rays = 0
    bbox_est = None

    # --- LiDAR-aligned distance ---
    if scan_msg is not None:
        lidar_m, lidar_n_rays = lidar_distance_for_bbox(
            scan_msg, bbox["cx"], bbox["w"], image_width, hfov_deg,
        )

    # --- Bbox pinhole estimate ---
    bbox_est = bbox_depth_estimate(
        bbox["h"], det["label"], image_height, image_width, hfov_deg,
    )

    # --- Fusion logic ---
    distance_m = None
    source = "none"
    reliable = False

    if lidar_m is not None and bbox_est is not None:
        # If they broadly agree (within 2× ratio), prefer LiDAR
        ratio = lidar_m / bbox_est if bbox_est > 0.01 else 999.0
        if 0.3 < ratio < 3.0:
            distance_m = lidar_m
            source = "fused"
            reliable = True
        else:
            # Large disagreement → LiDAR probably hitting background wall,
            # not the object.  Trust bbox estimate but flag unreliable.
            distance_m = bbox_est
            source = "bbox_estimate"
            reliable = False
    elif lidar_m is not None:
        distance_m = lidar_m
        source = "lidar"
        reliable = lidar_n_rays >= 3
    elif bbox_est is not None:
        distance_m = bbox_est
        source = "bbox_estimate"
        reliable = False
    # else: both None → source stays "none", distance stays None

    return {
        "distance_m": round(distance_m, 3) if distance_m is not None else None,
        "distance_source": source,
        "distance_reliable": reliable,
        "lidar_m": round(lidar_m, 3) if lidar_m is not None else None,
        "bbox_estimate_m": round(bbox_est, 3) if bbox_est is not None else None,
        "lidar_n_rays": lidar_n_rays,
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


def camera_info_to_dict(msg: CameraInfo) -> dict:
    """Serialise sensor_msgs/CameraInfo to a plain dict.

    Returns the full pinhole-model intrinsic matrix K, distortion
    coefficients D, projection matrix P, image dimensions, and derived
    convenience values (hfov_deg, vfov_deg, fx, fy, cx, cy) so that the
    LLM can perform 3D ↔ pixel projection without additional lookups.

    Projection of a 3-D point (X, Y, Z) in camera frame to pixel (u, v):
        u = fx * X/Z + cx
        v = fy * Y/Z + cy

    Back-projection of pixel (u, v) at known depth Z:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """
    K = list(msg.k)           # 3×3 row-major intrinsic matrix
    D = list(msg.d)           # distortion coefficients
    P = list(msg.p)           # 3×4 projection matrix (for rectified images)
    R = list(msg.r)           # 3×3 rectification rotation

    fx = K[0] if len(K) >= 1 else None
    fy = K[4] if len(K) >= 5 else None
    cx = K[2] if len(K) >= 3 else None
    cy = K[5] if len(K) >= 6 else None

    w = msg.width
    h = msg.height

    # Derive field-of-view angles from intrinsics when available
    hfov_deg = None
    vfov_deg = None
    if fx and w:
        hfov_deg = round(math.degrees(2.0 * math.atan2(w / 2.0, fx)), 2)
    if fy and h:
        vfov_deg = round(math.degrees(2.0 * math.atan2(h / 2.0, fy)), 2)

    return {
        "image_width":  w,
        "image_height": h,
        "distortion_model": msg.distortion_model,
        # Pinhole intrinsic matrix K (3×3, row-major)
        # [ fx  0  cx ]
        # [  0 fy  cy ]
        # [  0  0   1 ]
        "K": K,
        "fx": round(fx, 4) if fx is not None else None,
        "fy": round(fy, 4) if fy is not None else None,
        "cx": round(cx, 4) if cx is not None else None,
        "cy": round(cy, 4) if cy is not None else None,
        # Distortion coefficients D  (plumb_bob: k1,k2,p1,p2[,k3])
        "D": [round(d, 8) for d in D],
        # Projection matrix P (3×4) for rectified images: P = K [R | t]
        "P": [round(p, 4) for p in P],
        # Rectification rotation R (identity for mono cameras)
        "R": [round(r, 8) for r in R],
        # Derived convenience values
        "hfov_deg": hfov_deg,
        "vfov_deg": vfov_deg,
    }
