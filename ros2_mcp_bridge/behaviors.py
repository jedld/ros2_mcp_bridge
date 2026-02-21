#!/usr/bin/env python3
"""
behaviors.py — Higher-level robot behaviors built on top of ROS2BridgeNode.

Each function is a plain synchronous call.  The MCP tools call these from
their (synchronous) handlers so there are no async concerns here.
"""

import math
import time
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ros2_mcp_bridge.ros_node import ROS2BridgeNode


# --------------------------------------------------------------------------- #
# find_object
# --------------------------------------------------------------------------- #

def find_object_behavior(
    node: "ROS2BridgeNode",
    label: str,
    timeout: float,
    collision_avoidance: bool = True,
) -> dict:
    """
    Rotate in place up to one full revolution while watching for *label*.
    Returns {"found": bool, "detection": <dict or None>, "heading_deg": float}.

    When *collision_avoidance* is True an initial LiDAR snapshot is taken and
    its sector distances are included in the return dict so the LLM can reason
    about the surrounding space.  Pure rotation does not change the robot's
    position so forward/rear collision checks are not applied here.
    """
    from ros2_mcp_bridge.ros_node import detections_to_dict, laser_scan_to_dict

    cfg = node._cfg.get("topics", {})
    det_topic = cfg.get("detections", {}).get("topic", "/detections")
    laser_topic = cfg.get("laser", {}).get("topic", "/scan")

    # Snapshot obstacle distances before we start spinning
    obstacle_info: dict | None = None
    if collision_avoidance:
        scan_msg = node.get_latest(laser_topic, timeout=0.5)
        if scan_msg is not None:
            s = laser_scan_to_dict(scan_msg)
            obstacle_info = {
                "front_min_m": s.get("front_min_m"),
                "left_min_m":  s.get("left_min_m"),
                "right_min_m": s.get("right_min_m"),
                "rear_min_m":  s.get("rear_min_m"),
            }

    # How fast to rotate (rad/s) and polling interval
    angular_speed = 0.5   # rad/s — slow enough for the camera to see
    poll_interval = 0.15  # s

    # Full circle = 2π rad; at angular_speed rad/s that takes ~12.6 s
    full_circle_time = (2 * math.pi) / angular_speed
    deadline = time.time() + min(timeout, full_circle_time * 1.05)

    heading_rad = 0.0
    last_det = None

    while time.time() < deadline:
        node.publish_twist(0.0, angular_speed)
        time.sleep(poll_interval)
        heading_rad += angular_speed * poll_interval

        msg = node.get_latest(det_topic, timeout=0.1)
        if msg is not None:
            d = detections_to_dict(msg)
            for det in d["detections"]:
                if det["label"].lower() == label.lower():
                    node.stop()
                    result = {
                        "found": True,
                        "label": label,
                        "detection": det,
                        "heading_deg": round(math.degrees(heading_rad) % 360, 1),
                    }
                    if obstacle_info is not None:
                        result["obstacle_distances_m"] = obstacle_info
                    return result

    node.stop()
    result = {
        "found": False,
        "label": label,
        "detection": None,
        "heading_deg": None,
    }
    if obstacle_info is not None:
        result["obstacle_distances_m"] = obstacle_info
    return result


# --------------------------------------------------------------------------- #
# approach_object
# --------------------------------------------------------------------------- #

def approach_object_behavior(
    node: "ROS2BridgeNode",
    label: str,
    stop_distance: float,
    timeout: float,
    collision_avoidance: bool = True,
) -> dict:
    """
    Drive toward *label* until within *stop_distance* metres.

    Strategy:
    1. If detection bbox centre is left of image centre → turn left.
    2. If bbox centre is right → turn right.
    3. Use front LiDAR sector to determine distance; stop when < stop_distance.

    When *collision_avoidance* is True an additional hard safety stop is applied
    if the front LiDAR distance drops below the configured
    ``collision_avoidance.min_front_distance`` threshold — this fires even when
    *stop_distance* is set smaller than the safety threshold.  The response will
    include ``collision_avoidance_activated: true`` and an explanatory message so
    the LLM knows why the approach was interrupted.
    """
    from ros2_mcp_bridge.ros_node import detections_to_dict, laser_scan_to_dict

    cfg = node._cfg.get("topics", {})
    det_topic = cfg.get("detections", {}).get("topic", "/detections")
    scan_topic = cfg.get("laser", {}).get("topic", "/scan")

    ca_cfg = node._cfg.get("collision_avoidance", {})
    min_front_ca = float(ca_cfg.get("min_front_distance", 0.30))

    # Rough image width used for centering heuristic (pixels)
    # We don't know the exact resolution here, but 640 is typical
    image_width = node._cfg.get("image_width", 640)

    deadline = time.time() + timeout

    while time.time() < deadline:
        det_msg = node.get_latest(det_topic, timeout=0.3)
        scan_msg = node.get_latest(scan_topic, timeout=0.3)

        if det_msg is None:
            node.stop()
            return {"status": "lost_target", "message": "No detections received.",
                    "collision_avoidance_activated": False}

        d = detections_to_dict(det_msg)
        target = next(
            (x for x in d["detections"] if x["label"].lower() == label.lower()),
            None,
        )

        if target is None:
            # Object lost — stop and report
            node.stop()
            return {"status": "lost_target",
                    "message": f"'{label}' no longer detected.",
                    "collision_avoidance_activated": False}

        # Steering: proportional to horizontal offset from image centre
        cx = target["bbox"]["cx"]
        offset = (cx - image_width / 2.0) / (image_width / 2.0)  # [-1, 1]
        angular_z = -0.5 * offset  # negative because image-x is left→right

        if scan_msg is not None:
            scan_d = laser_scan_to_dict(scan_msg)
            front = scan_d.get("front_min_m")

            # ── Target reached (intentional stop) ────────────────────────── #
            if front is not None and front <= stop_distance:
                node.stop()
                return {
                    "status": "reached",
                    "message": f"Stopped {front:.2f} m from '{label}'.",
                    "distance_m": round(front, 3),
                    "collision_avoidance_activated": False,
                }

            # ── Collision avoidance safety stop ───────────────────────────── #
            if collision_avoidance and front is not None and front < min_front_ca:
                node.stop()
                return {
                    "status": "blocked",
                    "collision_avoidance_activated": True,
                    "message": (
                        f"Collision avoidance activated: obstacle {front:.2f} m ahead "
                        f"(threshold {min_front_ca:.2f} m). Approach aborted."
                    ),
                    "distance_m": round(front, 3),
                }
        else:
            front = None

        # Drive forward while steering
        node.publish_twist(0.15, angular_z)
        time.sleep(0.1)

    node.stop()
    return {"status": "timeout",
            "message": f"Approach timed out after {timeout}s.",
            "collision_avoidance_activated": False}


# --------------------------------------------------------------------------- #
# look_around
# --------------------------------------------------------------------------- #

def look_around_behavior(
    node: "ROS2BridgeNode",
    n_stops: int,
    pause_s: float,
    collision_avoidance: bool = True,
) -> dict:
    """
    Rotate the robot in *n_stops* equal steps covering 360°.
    At each stop, capture detections (and optionally an image).

    When *collision_avoidance* is True, the LiDAR sector distances are
    recorded at every observation stop so the LLM can reason about nearby
    obstacles at each heading.  Pure rotation does not move the robot so
    forward/rear blocking is not applied here.

    Returns a list of {heading_deg, detections} observations.
    """
    from ros2_mcp_bridge.ros_node import detections_to_dict, laser_scan_to_dict

    cfg = node._cfg.get("topics", {})
    det_topic = cfg.get("detections", {}).get("topic", "/detections")
    laser_topic = cfg.get("laser", {}).get("topic", "/scan")

    step_rad = (2 * math.pi) / n_stops
    angular_speed = 0.6   # rad/s
    step_time = step_rad / angular_speed

    observations = []

    for i in range(n_stops):
        heading_deg = round(math.degrees(i * step_rad) % 360, 1)

        # Rotate one step
        deadline = time.time() + step_time
        while time.time() < deadline:
            node.publish_twist(0.0, angular_speed)
            time.sleep(0.05)
        node.stop()

        # Pause before capturing
        time.sleep(pause_s)

        det_msg = node.get_latest(det_topic, timeout=0.5)
        detections = detections_to_dict(det_msg)["detections"] if det_msg is not None else []

        obs: dict = {
            "heading_deg": heading_deg,
            "detections": detections,
        }

        # Optionally attach LiDAR snapshot so the LLM knows proximity at this heading
        if collision_avoidance:
            scan_msg = node.get_latest(laser_topic, timeout=0.5)
            if scan_msg is not None:
                s = laser_scan_to_dict(scan_msg)
                obs["obstacle_distances_m"] = {
                    "front_min_m": s.get("front_min_m"),
                    "left_min_m":  s.get("left_min_m"),
                    "right_min_m": s.get("right_min_m"),
                    "rear_min_m":  s.get("rear_min_m"),
                }

        observations.append(obs)

    return {
        "n_stops": n_stops,
        "observations": observations,
        "total_detections": sum(len(o["detections"]) for o in observations),
    }
