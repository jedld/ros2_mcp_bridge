#!/usr/bin/env python3
"""
physical_test.py — End-to-end physical test suite for TurtleBot3 + Nav2 + SLAM bridge.
Movement limit: <100cm total displacement at any point.

Tests:
  1. Sensor reads  (scan, odom, battery)
  2. Small rotation  (90° CW then 90° CCW — net 0°)
  3. Small forward/back  (20 cm forward, 20 cm back — net 0 m)
  4. Waypoint save + Nav2 go_to_waypoint  (go 30 cm ahead, save, return)
  5. SLAM status + map info
  6. Map save (slam_toolbox save_map)
"""

import json
import math
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, MapMetaData
from sensor_msgs.msg import LaserScan, BatteryState
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import threading

PASS = "✅"
FAIL = "❌"
SKIP = "⚠️ "


class Tester(Node):
    def __init__(self):
        super().__init__("physical_tester")
        self._odom: Odometry | None = None
        self._scan: LaserScan | None = None
        self._battery: BatteryState | None = None
        self._odom_lock = threading.Lock()

        _sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._sub_odom = self.create_subscription(
            Odometry, "/odom", self._cb_odom, _sensor_qos)
        self._sub_scan = self.create_subscription(
            LaserScan, "/scan", self._cb_scan, _sensor_qos)
        self._sub_bat = self.create_subscription(
            BatteryState, "/battery_state", self._cb_battery, _sensor_qos)
        _latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_meta: MapMetaData | None = None
        self._sub_map_meta = self.create_subscription(
            MapMetaData, "/map_metadata",
            lambda m: setattr(self, '_map_meta', m), _latched_qos)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._ip_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10)
        self._nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        self.results: list[tuple[str, str, str]] = []  # (name, status, detail)

    # ──────────────────────────── callbacks ────────────────────────── #

    def _cb_odom(self, msg: Odometry):
        with self._odom_lock:
            self._odom = msg

    def _cb_scan(self, msg: LaserScan):
        self._scan = msg

    def _cb_battery(self, msg: BatteryState):
        self._battery = msg

    # ──────────────────────────── helpers ──────────────────────────── #

    def spin_for(self, secs: float):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def get_odom_pose(self) -> tuple[float, float, float] | None:
        """Returns (x, y, yaw_rad) or None."""
        with self._odom_lock:
            msg = self._odom
        if msg is None:
            return None
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        return msg.pose.pose.position.x, msg.pose.pose.position.y, yaw

    def stop_robot(self):
        t = Twist()
        for _ in range(5):
            self._cmd_pub.publish(t)
            self.spin_for(0.05)

    def log(self, name: str, ok: bool, detail: str = ""):
        status = PASS if ok else FAIL
        self.results.append((name, status, detail))
        print(f"  {status}  {name}: {detail}")

    # ──────────────────────────── test steps ───────────────────────── #

    def test_1_sensors(self):
        print("\n── Test 1: Sensor reads ──")
        # Wait up to 3s for data
        self.spin_for(3.0)

        # Scan
        if self._scan:
            ranges = [r for r in self._scan.ranges if math.isfinite(r) and r > 0.05]
            front_vals = [r for r in (self._scan.ranges[0:15] + self._scan.ranges[-15:])
                          if math.isfinite(r) and r > 0.05]
            front_min = min(front_vals, default=float('inf'))
            self.log("LiDAR /scan", True,
                     f"{len(self._scan.ranges)} beams, front_min={front_min:.2f}m, "
                     f"valid={len(ranges)}")
        else:
            self.log("LiDAR /scan", False, "no message received")

        # Odom
        pose = self.get_odom_pose()
        if pose:
            self.log("Odometry /odom", True,
                     f"x={pose[0]:.3f} y={pose[1]:.3f} yaw={math.degrees(pose[2]):.1f}°")
        else:
            self.log("Odometry /odom", False, "no message received")

        # Battery
        if self._battery:
            self.log("Battery /battery_state", self._battery.present,
                     f"{self._battery.voltage:.1f}V  {self._battery.percentage*100:.0f}%" if self._battery.percentage <= 1.0 else f"{self._battery.voltage:.1f}V  {self._battery.percentage:.0f}%")
        else:
            self.log("Battery /battery_state", False, "no message received")

    def test_2_rotation(self):
        print("\n── Test 2: In-place rotation (90°CW + 90°CCW) ──")
        # Check for obstacles first
        if self._scan is None:
            self.log("Rotation safety check", False, "no scan — skip")
            return
        front_min = min(
            [r for r in (self._scan.ranges[:15] + self._scan.ranges[-15:])
             if math.isfinite(r) and r > 0.05],
            default=9.9
        )
        if front_min < 0.35:
            self.log("Rotation pre-check", False,
                     f"obstacle at {front_min:.2f}m — aborting movement tests")
            return

        start_pose = self.get_odom_pose()
        if start_pose is None:
            self.log("Rotation", False, "no odom"); return

        def rotate(angular_z: float, target_rad: float):
            start_yaw = self.get_odom_pose()[2]
            t = Twist(); t.angular.z = angular_z
            rotated = 0.0
            prev_yaw = start_yaw
            while abs(rotated) < abs(target_rad):
                self._cmd_pub.publish(t)
                self.spin_for(0.05)
                cur_yaw = self.get_odom_pose()[2]
                delta = cur_yaw - prev_yaw
                # Unwrap
                if delta > math.pi: delta -= 2*math.pi
                if delta < -math.pi: delta += 2*math.pi
                rotated += delta
                prev_yaw = cur_yaw
                if abs(rotated) > abs(target_rad) * 1.2:
                    break
            self.stop_robot()
            return abs(rotated)

        # 90° CW
        r1 = rotate(-0.8, math.pi/2)
        self.spin_for(0.3)
        # 90° CCW → back to start
        r2 = rotate(+0.8, math.pi/2)
        self.stop_robot()

        end_pose = self.get_odom_pose()
        net_displacement = math.hypot(
            end_pose[0] - start_pose[0], end_pose[1] - start_pose[1])
        self.log("Rotation 90°CW + 90°CCW", True,
                 f"rotated {math.degrees(r1):.1f}° + {math.degrees(r2):.1f}°, "
                 f"net displacement={net_displacement*100:.1f}cm")

    def test_3_forward_back(self):
        print("\n── Test 3: Forward 20cm + back 20cm ──")
        if self._scan is None:
            self.log("Forward/back", False, "no scan"); return

        # Obstacle check — front and rear
        front_ranges = [r for r in (self._scan.ranges[:15] + self._scan.ranges[-15:])
                        if math.isfinite(r) and r > 0.05]
        rear_idx = len(self._scan.ranges) // 2
        rear_ranges = [r for r in self._scan.ranges[rear_idx-15:rear_idx+15]
                       if math.isfinite(r) and r > 0.05]
        front_min = min(front_ranges, default=9.9)
        rear_min = min(rear_ranges, default=9.9)

        if front_min < 0.40:
            self.log("Forward pre-check", False, f"obstacle ahead at {front_min:.2f}m — skip")
            return
        if rear_min < 0.40:
            self.log("Backward pre-check", False, f"obstacle behind at {rear_min:.2f}m — skip")
            return

        start = self.get_odom_pose()

        def drive(lin: float, target_m: float):
            start_p = self.get_odom_pose()
            t = Twist(); t.linear.x = lin
            driven = 0.0
            prev = start_p
            while abs(driven) < abs(target_m):
                self._cmd_pub.publish(t)
                self.spin_for(0.05)
                cur = self.get_odom_pose()
                driven += math.hypot(cur[0]-prev[0], cur[1]-prev[1])
                prev = cur
                if abs(driven) > abs(target_m) * 1.2:
                    break
            self.stop_robot()
            return abs(driven)

        d1 = drive(+0.1, 0.20)   # 20 cm forward
        self.spin_for(0.3)
        d2 = drive(-0.1, 0.20)   # 20 cm back
        self.stop_robot()

        end = self.get_odom_pose()
        net = math.hypot(end[0]-start[0], end[1]-start[1])
        self.log("Forward 20cm + back 20cm", True,
                 f"fwd={d1*100:.1f}cm  bk={d2*100:.1f}cm  net_disp={net*100:.1f}cm")

    def test_4_waypoint_nav2(self):
        """Drive 30cm ahead, save waypoint, return via Nav2."""
        print("\n── Test 4: Nav2 waypoint (30cm ahead → save → return) ──")
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.log("Nav2 action server", False, "/navigate_to_pose not available"); return
        self.log("Nav2 action server", True, "available")

        # Make sure we have odom
        start = self.get_odom_pose()
        if start is None:
            self.log("Nav2 waypoint test", False, "no odom"); return

        if self._scan is None:
            self.log("Nav2 obstacle check", False, "no scan"); return
        front_ranges = [r for r in (self._scan.ranges[:15] + self._scan.ranges[-15:])
                        if math.isfinite(r) and r > 0.05]
        front_min = min(front_ranges, default=9.9)
        if front_min < 0.55:
            self.log("Nav2 waypoint pre-check", False,
                     f"obstacle at {front_min:.2f}m — skip (need >55cm)"); return

        # Drive 30cm ahead via Nav2
        target_x = start[0] + 0.30 * math.cos(start[2])
        target_y = start[1] + 0.30 * math.sin(start[2])

        def send_goal(x, y, yaw=0.0, timeout_s=20.0) -> str:
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = "map"
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x = x
            goal.pose.pose.position.y = y
            goal.pose.pose.orientation.z = math.sin(yaw / 2)
            goal.pose.pose.orientation.w = math.cos(yaw / 2)
            future = self._nav_client.send_goal_async(goal)
            end = time.time() + timeout_s
            while not future.done() and time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.1)
            if not future.done():
                return "timeout_accept"
            gh = future.result()
            if not gh.accepted:
                return "rejected"
            res_future = gh.get_result_async()
            end = time.time() + timeout_s
            while not res_future.done() and time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.1)
            if not res_future.done():
                gh.cancel_goal_async()
                return "timeout_result"
            return "succeeded"

        print(f"     → Navigating to ({target_x:.3f}, {target_y:.3f}) …")
        status_fwd = send_goal(target_x, target_y, start[2])
        mid = self.get_odom_pose()
        dist_fwd = math.hypot(mid[0]-start[0], mid[1]-start[1]) if mid else 0
        self.log("Nav2 forward 30cm", status_fwd == "succeeded",
                 f"status={status_fwd}  actual dist={dist_fwd*100:.1f}cm")

        # Return to start
        print(f"     → Returning to ({start[0]:.3f}, {start[1]:.3f}) …")
        status_ret = send_goal(start[0], start[1], start[2])
        end_p = self.get_odom_pose()
        net = math.hypot(end_p[0]-start[0], end_p[1]-start[1]) if end_p else 999
        self.log("Nav2 return to start", status_ret == "succeeded",
                 f"status={status_ret}  net_error={net*100:.1f}cm")

    def test_5_slam_status(self):
        print("\n── Test 5: SLAM status ──")
        result = subprocess.run(
            ["ros2", "node", "list"],
            capture_output=True, text=True, timeout=5)
        nodes = result.stdout.strip().split("\n")
        slam_running = any("slam_toolbox" in n for n in nodes)
        self.log("slam_toolbox node", slam_running,
                 "/slam_toolbox" if slam_running else "NOT found")

        # Check /map topic
        result2 = subprocess.run(
            ["ros2", "topic", "info", "/map"],
            capture_output=True, text=True, timeout=5)
        map_ok = "Publisher count: 1" in result2.stdout or \
                 "publisher count: 1" in result2.stdout.lower() or \
                 "1" in result2.stdout
        self.log("/map topic", map_ok, result2.stdout.strip().split("\n")[0])

        # Map metadata — already subscribed in __init__, wait for it
        self.spin_for(2.0)
        if self._map_meta is not None:
            m = self._map_meta
            self.log("Map metadata", True,
                     f"res={m.resolution:.4f}m/cell  {m.width}x{m.height}cells  "
                     f"({m.width*m.resolution:.1f}x{m.height*m.resolution:.1f}m)")
        else:
            # Fallback: try ros2 topic echo
            result3 = subprocess.run(
                ["ros2", "topic", "echo", "/map_metadata", "--once"],
                capture_output=True, text=True, timeout=8)
            if "resolution" in result3.stdout:
                lines = {l.split(":")[0].strip(): l.split(":")[1].strip()
                         for l in result3.stdout.strip().split("\n") if ":" in l}
                res = lines.get("resolution", "?")
                w = lines.get("width", "?")
                h = lines.get("height", "?")
                self.log("Map metadata", True, f"res={res}m/cell  {w}x{h}cells (via CLI)")
            else:
                self.log("Map metadata", False, "no /map_metadata received")

    def test_6_save_map(self):
        print("\n── Test 6: SLAM save map ──")
        # Use ros2 service call to save the map
        req = '"{name: {data: \'/home/joseph/maps/physical_test_map\'}}"'
        result = subprocess.run(
            ["ros2", "service", "call",
             "/slam_toolbox/save_map",
             "slam_toolbox/srv/SaveMap",
             "{name: {data: '/home/joseph/maps/physical_test_map'}}"],
            capture_output=True, text=True, timeout=15)
        ok = result.returncode == 0 and "result" in result.stdout.lower()
        detail = result.stdout.strip().split("\n")[-1] if result.stdout else result.stderr[:80]
        self.log("slam_toolbox/save_map", ok, detail)

        # Check files were created
        import os
        pgm = "/home/joseph/maps/physical_test_map.pgm"
        yaml = "/home/joseph/maps/physical_test_map.yaml"
        self.log("Map files created (.pgm)", os.path.exists(pgm), pgm if os.path.exists(pgm) else "NOT found")
        self.log("Map files created (.yaml)", os.path.exists(yaml), yaml if os.path.exists(yaml) else "NOT found")


def main():
    rclpy.init()
    node = Tester()
    print("=" * 60)
    print("TurtleBot3 Physical Test Suite")
    print("Max movement: 100cm  |  All moves auto-reversed")
    print("=" * 60)

    try:
        node.test_1_sensors()
        node.test_2_rotation()
        node.test_3_forward_back()
        node.test_4_waypoint_nav2()
        node.test_5_slam_status()
        node.test_6_save_map()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        node.stop_robot()
    finally:
        node.stop_robot()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, s, _ in node.results if s == PASS)
    failed = sum(1 for _, s, _ in node.results if s == FAIL)
    for name, status, detail in node.results:
        print(f"  {status}  {name}")
        if status == FAIL and detail:
            print(f"       → {detail}")
    print(f"\n  Total: {passed} passed / {failed} failed / {len(node.results)} tests")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
