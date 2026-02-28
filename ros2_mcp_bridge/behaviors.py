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

    When the object is found, this also computes:
      - **bearing_deg**: the angular offset of the object from the camera
        centre (positive = left).
      - **estimated_distance_m** / **distance_source**: fused LiDAR + bbox
        depth estimate so the LLM can decide whether to approach directly.

    When *collision_avoidance* is True an initial LiDAR snapshot is taken and
    its sector distances are included in the return dict so the LLM can reason
    about the surrounding space.  Pure rotation does not change the robot's
    position so forward/rear collision checks are not applied here.
    """
    from ros2_mcp_bridge.ros_node import (
        detections_to_dict, laser_scan_to_dict,
        estimate_detection_distance, bearing_from_bbox,
    )

    cfg = node._cfg.get("topics", {})
    det_topic = cfg.get("detections", {}).get("topic", "/detections")
    laser_topic = cfg.get("laser", {}).get("topic", "/scan")

    image_width = node._cfg.get("image_width", 640)
    image_height = int(image_width * 480 / 640)
    hfov_deg = float(node._cfg.get("camera_hfov_deg", 62.0))

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

    # Clear any previous stop signal so the loop runs cleanly
    node.clear_stop_event()

    heading_rad = 0.0
    last_det = None

    while time.time() < deadline:
        if node.is_stop_requested():
            node.stop()
            result = {
                "found": False,
                "label": label,
                "detection": None,
                "heading_deg": None,
                "cancelled": True,
                "message": "Search cancelled by stop command.",
            }
            if obstacle_info is not None:
                result["obstacle_distances_m"] = obstacle_info
            return result
        node.publish_twist(0.0, angular_speed)
        time.sleep(poll_interval)
        heading_rad += angular_speed * poll_interval

        msg = node.get_latest(det_topic, timeout=0.1)
        if msg is not None:
            d = detections_to_dict(msg)
            for det in d["detections"]:
                if det["label"].lower() == label.lower():
                    node.stop()

                    # Compute bearing from bbox centre
                    bearing_rad = bearing_from_bbox(
                        det["bbox"]["cx"], image_width, hfov_deg,
                    )
                    bearing_deg_val = round(math.degrees(bearing_rad), 1)

                    # Estimate distance using LiDAR + bbox cascade
                    scan_now = node.get_latest(laser_topic, timeout=0.3)
                    dist_info = estimate_detection_distance(
                        scan_now, det, image_width, image_height, hfov_deg,
                    )

                    result = {
                        "found": True,
                        "label": label,
                        "detection": det,
                        "heading_deg": round(math.degrees(heading_rad) % 360, 1),
                        "bearing_deg": bearing_deg_val,
                        "estimated_distance_m": dist_info["distance_m"],
                        "distance_source": dist_info["distance_source"],
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
# VLM helper for approach_object (optional — used when YOLO loses the target)
# --------------------------------------------------------------------------- #

def _vlm_check_object(node: "ROS2BridgeNode", label: str, cam_topic: str,
                       timeout: float = 15.0) -> dict | None:
    """Ask the VLM agent whether *label* is visible in the current camera frame.

    The robot should be **stopped** before calling this so the frame is sharp.
    Returns ``{"visible": bool, "vlm_response": str}`` or ``None`` if the
    VLM agent is disabled / unreachable.
    """
    import base64
    import json as _json
    import uuid

    vlm_cfg = node._cfg.get("vlm_agent", {})
    if not vlm_cfg.get("enabled", False):
        return None
    vlm_url = vlm_cfg.get("url", "").rstrip("/")
    if not vlm_url:
        return None

    # Capture a fresh frame (robot should already be stopped)
    cam_msg = node.get_fresh(cam_topic, timeout=2.0)
    if cam_msg is None:
        return None

    img_b64 = base64.b64encode(bytes(cam_msg.data)).decode("utf-8")
    fmt = (cam_msg.format or "jpeg").lower().split("/")[-1]
    mime_type = f"image/{fmt}"

    question = (
        f"Is there a '{label}' visible in this image? "
        f"Answer YES or NO on the first line, then briefly describe "
        f"where it is if visible."
    )

    try:
        import urllib.request
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": str(uuid.uuid4()),
                    "parts": [
                        {"kind": "text", "text": question},
                        {"kind": "file", "file": {
                            "bytes": img_b64,
                            "mimeType": mime_type,
                            "name": "camera_frame.jpg",
                        }},
                    ],
                }
            },
        }
        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            vlm_url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = _json.loads(resp.read().decode("utf-8"))

        if "error" in result:
            return None

        # Extract text from A2A response (same structure as mcp_server._call_a2a_with_image)
        text = None
        res = result.get("result", {})
        for artifact in res.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    text = part["text"]
                    break
            if text:
                break
        if text is None:
            task_result = res.get("result")
            if task_result:
                for part in task_result.get("parts", []):
                    if part.get("kind") == "text":
                        text = part["text"]
                        break
        if text is None:
            for part in res.get("parts", []):
                if part.get("kind") == "text":
                    text = part["text"]
                    break
        if text is None:
            return None

        visible = text.strip().upper().startswith("YES")
        return {"visible": visible, "vlm_response": text}

    except Exception:
        return None


# --------------------------------------------------------------------------- #
# approach_object  (two-phase: acquire → blind approach)
# --------------------------------------------------------------------------- #

def approach_object_behavior(
    node: "ROS2BridgeNode",
    label: str,
    stop_distance: float,
    timeout: float,
    collision_avoidance: bool = True,
) -> dict:
    """
    Two-phase approach to a detected object.

    **Phase 1 — Acquisition** (uses object detection):
      1. Get a fresh YOLO detection for *label*.
      2. Compute bearing angle from the bounding-box centre in the image.
      3. Estimate distance using a priority cascade:
         a. LiDAR rays aligned with the bearing (most reliable).
         b. Pinhole-model bbox height estimate (works for known COCO classes).
         c. Motion-stereo depth fallback: nudge the robot forward ~8 cm,
            match ORB features in the bbox region, triangulate depth from
            parallax, then reverse back.
      4. Convert (bearing, distance) into a target point in the odometry frame.

    **Phase 2 — Blind approach** (no more object detection):
      1. Rotate to face the target bearing.
      2. Compute the drive distance = estimated_distance − stop_distance.
      3. Drive forward in small increments using odometry closed-loop,
         with LiDAR collision avoidance at every step.
      4. Yield streaming progress updates so the LLM can track movement.
      5. Stop when the target distance is reached or an obstacle blocks.

    The generator yields dicts with status updates; the final yield has
    ``"phase": "complete"``.
    """
    from ros2_mcp_bridge.ros_node import (
        detections_to_dict, laser_scan_to_dict,
        estimate_detection_distance, bearing_from_bbox,
        motion_stereo_depth, odometry_to_dict,
    )

    cfg = node._cfg.get("topics", {})
    det_topic = cfg.get("detections", {}).get("topic", "/detections")
    scan_topic = cfg.get("laser", {}).get("topic", "/scan")
    cam_topic = cfg.get("camera", {}).get("topic", "/camera/image_raw/compressed")
    odom_topic = cfg.get("odom", {}).get("topic", "/odom")

    ca_cfg = node._cfg.get("collision_avoidance", {})
    min_front_ca = float(ca_cfg.get("min_front_distance", 0.30))

    image_width = node._cfg.get("image_width", 640)
    image_height = int(image_width * 480 / 640)
    hfov_deg = float(node._cfg.get("camera_hfov_deg", 62.0))

    approach_cfg = node._cfg.get("approach_object", {})
    max_speed = float(approach_cfg.get("max_speed", 0.15))
    min_speed = float(approach_cfg.get("min_speed", 0.05))
    step_size = float(approach_cfg.get("step_size", 0.15))

    deadline = time.time() + timeout

    # Clear any previous stop signal so the generator runs cleanly
    node.clear_stop_event()

    # ================================================================== #
    #  PHASE 1 — ACQUISITION  (detection-based)
    # ================================================================== #

    yield {
        "phase": "acquisition",
        "status": "detecting",
        "message": f"Searching for '{label}' in current camera frame...",
    }

    # --- 1a. Get fresh detection ---------------------------------------- #
    det_msg = None
    target = None
    DETECT_RETRIES = 5
    for attempt in range(DETECT_RETRIES):
        if time.time() >= deadline or node.is_stop_requested():
            node.stop()
            yield {"phase": "complete", "status": "cancelled" if node.is_stop_requested() else "timeout",
                   "message": "Cancelled by stop command." if node.is_stop_requested() else "Timed out during acquisition phase."}
            return
        det_msg = node.get_fresh(det_topic, timeout=1.0)
        if det_msg is not None:
            d = detections_to_dict(det_msg)
            target = next(
                (x for x in d["detections"]
                 if x["label"].lower() == label.lower()),
                None,
            )
        if target is not None:
            break
        time.sleep(0.3)

    if target is None:
        node.stop()
        yield {
            "phase": "complete",
            "status": "not_found",
            "message": (
                f"'{label}' not detected after {DETECT_RETRIES} attempts. "
                f"Try find_object first to rotate toward the object, "
                f"or explore_for_object to search a larger area."
            ),
        }
        return

    # --- 1b. Compute bearing -------------------------------------------- #
    bearing_rad = bearing_from_bbox(target["bbox"]["cx"], image_width, hfov_deg)
    bearing_deg = round(math.degrees(bearing_rad), 1)

    yield {
        "phase": "acquisition",
        "status": "bearing_locked",
        "label": label,
        "bearing_deg": bearing_deg,
        "detection": target,
        "message": f"Locked bearing on '{label}' at {bearing_deg}° from centre.",
    }

    # --- 1c. Estimate distance (cascade) -------------------------------- #
    scan_msg = node.get_latest(scan_topic, timeout=0.5)

    dist_info = estimate_detection_distance(
        scan_msg, target, image_width, image_height, hfov_deg,
    )
    est_dist = dist_info["distance_m"]
    dist_source = dist_info["distance_source"]

    # If both LiDAR and bbox failed, try motion stereo
    if est_dist is None:
        yield {
            "phase": "acquisition",
            "status": "motion_stereo",
            "message": (
                "LiDAR and bbox estimates unavailable. "
                "Attempting motion-stereo depth estimation (nudging forward ~8 cm)..."
            ),
        }
        stereo_depth = motion_stereo_depth(
            node, cam_topic, target["bbox"],
            image_width, hfov_deg, baseline_m=0.08,
        )
        if stereo_depth is not None:
            est_dist = stereo_depth
            dist_source = "motion_stereo"
        else:
            node.stop()
            yield {
                "phase": "complete",
                "status": "distance_unknown",
                "bearing_deg": bearing_deg,
                "message": (
                    f"Could not estimate distance to '{label}'. "
                    f"LiDAR, bbox, and motion-stereo all failed. "
                    f"The object may be too small, too far, or featureless. "
                    f"Try driving closer manually and retrying."
                ),
            }
            return

    yield {
        "phase": "acquisition",
        "status": "distance_estimated",
        "estimated_distance_m": round(est_dist, 3),
        "distance_source": dist_source,
        "bearing_deg": bearing_deg,
        "message": (
            f"Distance to '{label}': {est_dist:.2f} m "
            f"(source: {dist_source}). "
            f"Will approach to within {stop_distance:.2f} m."
        ),
    }

    # --- 1d. Already within stop distance? ------------------------------ #
    if est_dist <= stop_distance:
        node.stop()
        yield {
            "phase": "complete",
            "status": "already_reached",
            "distance_m": round(est_dist, 3),
            "distance_source": dist_source,
            "bearing_deg": bearing_deg,
            "message": (
                f"Already within stop distance: {est_dist:.2f} m ≤ {stop_distance:.2f} m. "
                f"No approach needed."
            ),
        }
        return

    # --- 1e. Compute target pose in odom frame -------------------------- #
    odom_msg = node.get_latest(odom_topic, timeout=1.0)
    if odom_msg is None:
        node.stop()
        yield {"phase": "complete", "status": "failed",
               "message": "Odometry not available."}
        return

    odom = odometry_to_dict(odom_msg)
    robot_x, robot_y = odom["x"], odom["y"]
    robot_yaw = odom["yaw_rad"]

    # Target in world frame: robot_pos + distance * direction(yaw + bearing)
    target_angle = robot_yaw + bearing_rad
    target_x = robot_x + est_dist * math.cos(target_angle)
    target_y = robot_y + est_dist * math.sin(target_angle)
    drive_distance = max(0.0, est_dist - stop_distance)

    yield {
        "phase": "acquisition",
        "status": "target_computed",
        "target_x": round(target_x, 3),
        "target_y": round(target_y, 3),
        "drive_distance_m": round(drive_distance, 3),
        "message": (
            f"Target pose: ({target_x:.3f}, {target_y:.3f}). "
            f"Will drive {drive_distance:.2f} m after rotating {bearing_deg:.1f}°."
        ),
    }

    # ================================================================== #
    #  PHASE 2 — BLIND APPROACH  (odometry + LiDAR, no detection)
    # ================================================================== #

    yield {
        "phase": "approach",
        "status": "rotating",
        "bearing_deg": bearing_deg,
        "message": f"Rotating {bearing_deg:.1f}° to face '{label}'...",
    }

    # --- 2a. Rotate to face the target ---------------------------------- #
    if abs(bearing_deg) > 2.0:
        rot_result = node.rotate_angle(bearing_deg, 0.0, min(deadline - time.time(), 10.0))
        if rot_result["status"] == "stuck":
            yield {
                "phase": "complete",
                "status": "stuck_rotating",
                "rotation_result": rot_result,
                "message": f"Got stuck while rotating: {rot_result.get('message', '')}",
            }
            return
        yield {
            "phase": "approach",
            "status": "rotation_complete",
            "rotation_result": rot_result,
            "message": f"Rotation done: {rot_result.get('angle_actual_deg', 0):.1f}° actual.",
        }

    # --- 2b. Drive forward in increments -------------------------------- #
    remaining_dist = drive_distance
    total_driven = 0.0
    step_num = 0

    while remaining_dist > 0.02 and time.time() < deadline:
        # --- Check for cancellation --------------------------------- #
        if node.is_stop_requested():
            node.stop()
            yield {
                "phase": "complete",
                "status": "cancelled",
                "distance_driven_m": round(total_driven, 3),
                "distance_remaining_m": round(remaining_dist, 3),
                "message": "Approach cancelled by stop command.",
            }
            return
        step_num += 1
        this_step = min(step_size, remaining_dist)

        # Adaptive speed: slow down in the last 0.3 m
        if remaining_dist < 0.3:
            speed = min_speed + (max_speed - min_speed) * (remaining_dist / 0.3)
        else:
            speed = max_speed

        # Pre-step collision check via latest LiDAR
        if collision_avoidance:
            scan_msg = node.get_latest(scan_topic, timeout=0.3)
            if scan_msg is not None:
                s = laser_scan_to_dict(scan_msg)
                front = s.get("front_min_m")
                if front is not None and front < min_front_ca:
                    node.stop()
                    yield {
                        "phase": "complete",
                        "status": "blocked",
                        "collision_avoidance_activated": True,
                        "obstacle_distance_m": round(front, 3),
                        "distance_driven_m": round(total_driven, 3),
                        "distance_remaining_m": round(remaining_dist, 3),
                        "message": (
                            f"Obstacle detected at {front:.2f} m (threshold "
                            f"{min_front_ca:.2f} m). Stopped after driving "
                            f"{total_driven:.2f} m of {drive_distance:.2f} m."
                        ),
                    }
                    return

        # Drive one step
        step_result = node.move_distance(
            this_step, speed=speed,
            timeout=min(deadline - time.time(), 10.0),
            collision_avoidance=collision_avoidance,
        )

        actual_step = abs(step_result.get("distance_actual", 0.0))
        total_driven += actual_step
        remaining_dist -= actual_step

        # Handle blocked / stuck during step
        if step_result["status"] == "blocked":
            node.stop()
            yield {
                "phase": "complete",
                "status": "blocked",
                "collision_avoidance_activated": True,
                "distance_driven_m": round(total_driven, 3),
                "distance_remaining_m": round(remaining_dist, 3),
                "step_result": step_result,
                "message": (
                    f"Blocked by obstacle after {total_driven:.2f} m. "
                    f"{remaining_dist:.2f} m remaining. "
                    f"Consider trying a different approach angle."
                ),
            }
            return

        if step_result["status"] == "stuck":
            yield {
                "phase": "complete",
                "status": "stuck",
                "distance_driven_m": round(total_driven, 3),
                "distance_remaining_m": round(remaining_dist, 3),
                "step_result": step_result,
                "message": (
                    f"Robot stuck after driving {total_driven:.2f} m. "
                    f"{step_result.get('message', '')}"
                ),
            }
            return

        # Progress update
        pct = round(total_driven / drive_distance * 100, 0) if drive_distance > 0 else 100
        yield {
            "phase": "approach",
            "status": "driving",
            "step": step_num,
            "step_distance_m": round(actual_step, 3),
            "total_driven_m": round(total_driven, 3),
            "remaining_m": round(max(0, remaining_dist), 3),
            "progress_pct": pct,
            "message": (
                f"Step {step_num}: drove {actual_step:.3f} m. "
                f"Total: {total_driven:.2f}/{drive_distance:.2f} m ({pct:.0f}%)."
            ),
        }

    node.stop()

    # --- Final status ---------------------------------------------------- #
    if remaining_dist <= 0.02:
        yield {
            "phase": "complete",
            "status": "reached",
            "label": label,
            "distance_driven_m": round(total_driven, 3),
            "estimated_final_distance_m": round(stop_distance, 3),
            "distance_source": dist_source,
            "bearing_deg": bearing_deg,
            "message": (
                f"Approached '{label}' successfully. Drove {total_driven:.2f} m. "
                f"Estimated distance from object: ~{stop_distance:.2f} m "
                f"(based on {dist_source} measurement)."
            ),
        }
    else:
        yield {
            "phase": "complete",
            "status": "timeout",
            "distance_driven_m": round(total_driven, 3),
            "distance_remaining_m": round(remaining_dist, 3),
            "message": (
                f"Approach timed out after {timeout:.0f}s. "
                f"Drove {total_driven:.2f} m, {remaining_dist:.2f} m remaining."
            ),
        }


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


# --------------------------------------------------------------------------- #
# explore_for_object
# --------------------------------------------------------------------------- #

def explore_for_object_behavior(
    node: "ROS2BridgeNode",
    label: str,
    step_distance: float,
    max_steps: int,
    timeout: float,
    collision_avoidance: bool = True,
) -> dict:
    """
    Actively explore the environment to find *label*.

    Exploration loop (repeats up to *max_steps* times or until *timeout*):
      1. Rotate a full circle with detection polling (find_object_behavior).
         If the target is found, return immediately.
      2. Read LiDAR to pick the most-open direction:
            front  → continue straight
            left   → rotate +90 °
            right  → rotate -90 °
            (rear used as fallback if all forward sectors are blocked)
      3. Drive forward *step_distance* metres.
      4. Record the visit so the LLM can see the exploration path.

    Returns {"found": bool, "detection": dict|None, "steps_taken": int,
             "path": [{"step": int, "action": str, "pose": {x,y}, ...}]}.
    """
    from ros2_mcp_bridge.ros_node import detections_to_dict, laser_scan_to_dict, odometry_to_dict

    cfg = node._cfg.get("topics", {})
    det_topic  = cfg.get("detections", {}).get("topic", "/detections")
    laser_topic = cfg.get("laser",     {}).get("topic", "/scan")
    odom_topic  = cfg.get("odom",      {}).get("topic", "/odom")

    # How much clearance we require ahead before stepping forward (metres)
    ca_cfg = node._cfg.get("collision_avoidance", {})
    min_front = float(ca_cfg.get("min_front_distance", 0.30))
    step_clearance = max(step_distance + 0.15, min_front + 0.10)

    deadline = time.time() + timeout
    path: list[dict] = []

    # Clear any previous stop signal so the generator runs cleanly
    node.clear_stop_event()

    def _current_pose() -> dict | None:
        msg = node.get_latest(odom_topic, timeout=0.5)
        if msg is None:
            return None
        o = odometry_to_dict(msg)
        return {"x": round(o["x"], 3), "y": round(o["y"], 3), "yaw_deg": round(o.get("yaw_deg", 0), 1)}

    def _scan_sectors() -> dict | None:
        msg = node.get_latest(laser_topic, timeout=0.5)
        if msg is None:
            return None
        s = laser_scan_to_dict(msg)
        return {
            "front_min_m": s.get("front_min_m"),
            "left_min_m":  s.get("left_min_m"),
            "right_min_m": s.get("right_min_m"),
            "rear_min_m":  s.get("rear_min_m"),
        }

    for step in range(max_steps):
        if time.time() >= deadline:
            yield {
                "progress": "timeout",
                "step": step,
                "path": path,
            }
            return

        # --- Check for cancellation --------------------------------- #
        if node.is_stop_requested():
            node.stop()
            yield {
                "progress": "cancelled",
                "step": step,
                "path": path,
                "message": "Exploration cancelled by stop command.",
            }
            return

        remaining = deadline - time.time()
        # --- 1. Rotate and scan for the target at this position --------- #
        scan_timeout = min(remaining, 15.0)
        found_result = find_object_behavior(node, label, scan_timeout, collision_avoidance)
        pose = _current_pose()

        if found_result.get("found"):
            path.append({
                "step": step,
                "action": "found",
                "pose": pose,
                "detection": found_result.get("detection"),
            })
            yield {
                "found": True,
                "label": label,
                "detection": found_result["detection"],
                "heading_deg": found_result.get("heading_deg"),
                "steps_taken": step + 1,
                "path": path,
            }
            return

        if time.time() >= deadline:
            yield {
                "progress": "timeout",
                "step": step,
                "path": path,
            }
            return

        # --- 2. Pick the most-open direction using LiDAR ---------------- #
        sectors = _scan_sectors()
        turn_deg = 0.0
        action_label = "forward"

        if sectors is not None:
            front = sectors.get("front_min_m") or 0.0
            left  = sectors.get("left_min_m")  or 0.0
            right = sectors.get("right_min_m") or 0.0
            rear  = sectors.get("rear_min_m")  or 0.0

            # Choose the sector with the most clearance
            candidates = [
                (front, 0.0,    "forward"),
                (left,  90.0,   "turn_left_90"),
                (right, -90.0,  "turn_right_90"),
                (rear,  180.0,  "turn_around"),
            ]
            best_dist, turn_deg, action_label = max(candidates, key=lambda c: c[0])

            if best_dist < min_front:
                # All sectors blocked — log and bail out
                path.append({
                    "step": step,
                    "action": "all_blocked",
                    "pose": pose,
                    "sectors": sectors,
                })
                yield {
                    "progress": "all_blocked",
                    "step": step,
                    "path": path,
                }
                return
        else:
            action_label = "forward_no_lidar"

        # --- 3. Rotate to chosen direction ------------------------------ #
        if abs(turn_deg) > 5.0:
            remaining = deadline - time.time()
            node.rotate_angle(turn_deg, 0.0, min(remaining, 10.0))

        path.append({
            "step": step,
            "action": action_label,
            "pose": pose,
            "sectors": sectors,
        })

        yield {
            "progress": "step",
            "step": step,
            "action": action_label,
            "pose": pose,
            "sectors": sectors,
            "path": path,
        }

        if time.time() >= deadline:
            yield {
                "progress": "timeout",
                "step": step,
                "path": path,
            }
            return

        # --- 4. Drive forward step_distance ----------------------------- #
        remaining = deadline - time.time()
        node.move_distance(step_distance, 0.0, min(remaining, 20.0), collision_avoidance)

    node.stop()
    yield {
        "found": False,
        "label": label,
        "detection": None,
        "steps_taken": len([p for p in path if p["action"] not in ("all_blocked",)]),
        "path": path,
        "message": (
            f"Object '{label}' not found after exploring {len(path)} positions. "
            "Consider moving to a different room or area and trying again."
        ),
    }


# --------------------------------------------------------------------------- #
# follow_wall
# --------------------------------------------------------------------------- #

def follow_wall_behavior(
    node: "ROS2BridgeNode",
    side: str,
    target_distance: float,
    duration: float,
    speed: float,
) -> dict:
    """
    Follow the wall on *side* ('left' or 'right') for *duration* seconds,
    maintaining approximately *target_distance* metres from the wall.

    Uses a simple proportional controller:
        error = measured_side_distance - target_distance
        angular_z = -Kp * error  (sign flips per side)

    Stops early if a front obstacle is detected within `min_front_distance`.

    Returns {"status": ..., "distance_travelled_m": float, "duration_s": float}.
    """
    from ros2_mcp_bridge.ros_node import laser_scan_to_dict

    Kp = 1.2          # proportional gain for wall-following
    dt  = 0.1         # control loop period (seconds)

    cfg = node._cfg.get("topics", {})
    laser_topic = cfg.get("laser", {}).get("topic", "/scan")
    ca_cfg = node._cfg.get("collision_avoidance", {})
    min_front = float(ca_cfg.get("min_front_distance", 0.30))

    robot_cfg  = node._cfg.get("robot", {})
    max_linear = float(robot_cfg.get("max_linear_speed", 0.22))
    max_angular = float(robot_cfg.get("max_angular_speed", 2.84))

    linear_v = max(0.05, min(float(speed), max_linear))

    deadline = time.time() + duration
    start    = time.time()

    # Clear any previous stop signal
    node.clear_stop_event()

    while time.time() < deadline:
        if node.is_stop_requested():
            node.stop()
            return {
                "status": "cancelled",
                "message": "Wall-follow cancelled by stop command.",
                "distance_travelled_m": round(linear_v * (time.time() - start), 2),
                "duration_s": round(time.time() - start, 2),
            }
        scan_msg = node.get_latest(laser_topic, timeout=0.3)

        if scan_msg is not None:
            s = laser_scan_to_dict(scan_msg)
            front = s.get("front_min_m") or 999.0

            # Front obstacle — stop
            if front < min_front:
                node.stop()
                return {
                    "status": "blocked",
                    "message": f"Front obstacle at {front:.2f} m.",
                    "distance_travelled_m": round(linear_v * (time.time() - start), 2),
                    "duration_s": round(time.time() - start, 2),
                }

            # Wall-following: keep selected side at target_distance
            if side == "left":
                side_dist = s.get("left_min_m") or target_distance
                error = side_dist - target_distance
                angular_z = -Kp * error   # positive error → too far → turn left (+)
            else:
                side_dist = s.get("right_min_m") or target_distance
                error = side_dist - target_distance
                angular_z = Kp * error    # positive error → too far → turn right (-)

            angular_z = max(-max_angular, min(angular_z, max_angular))
        else:
            # No scan data — drive straight
            angular_z = 0.0

        node.publish_twist(linear_v, angular_z)
        time.sleep(dt)

    node.stop()
    return {
        "status": "completed",
        "message": f"Wall-follow on {side} side finished normally.",
        "target_wall_distance_m": target_distance,
        "distance_travelled_m": round(linear_v * duration, 2),
        "duration_s": round(time.time() - start, 2),
    }
