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

    Key robustness improvements over a naive get-and-drive loop:

    1. **Frame synchronization** — uses ``get_fresh()`` for detections so
       the behaviour always acts on the newest detector output, not a stale
       cached message from before the robot moved.

    2. **Stop-and-rescan on lost detection** — when YOLO loses the target
       (common during motion blur), the robot *stops*, waits for the camera
       to capture a sharp frame, and retries detection several times before
       declaring the target lost.  This avoids triggering an unnecessary
       360° find_object scan when the object is right in front of the robot.

    3. **Adaptive speed** — the robot decelerates as it nears the target,
       reducing motion blur and giving the detector more time to lock on.

    4. **VLM fallback** — if YOLO keeps failing after retries and a VLM
       agent is enabled, the behaviour captures a clean frame and asks the
       VLM whether the object is still visible.  If the VLM confirms, the
       robot continues cautiously using LiDAR-only distance guidance.

    5. **Collision avoidance** — unchanged: a hard LiDAR safety stop fires
       if any obstacle enters the configured minimum front distance.
    """
    from ros2_mcp_bridge.ros_node import (
        detections_to_dict, laser_scan_to_dict,
        estimate_detection_distance,
    )

    cfg = node._cfg.get("topics", {})
    det_topic = cfg.get("detections", {}).get("topic", "/detections")
    scan_topic = cfg.get("laser", {}).get("topic", "/scan")
    cam_topic = cfg.get("camera", {}).get("topic", "/camera/image_raw/compressed")

    ca_cfg = node._cfg.get("collision_avoidance", {})
    min_front_ca = float(ca_cfg.get("min_front_distance", 0.30))

    # Camera / image parameters for distance estimation
    image_width = node._cfg.get("image_width", 640)
    image_height = int(image_width * 480 / 640)  # assume 4:3 aspect
    hfov_deg = float(node._cfg.get("camera_hfov_deg", 62.0))

    # ── Approach tuning (bridge.yaml → approach_object section) ───────── #
    approach_cfg = node._cfg.get("approach_object", {})
    max_speed = float(approach_cfg.get("max_speed", 0.15))
    min_speed = float(approach_cfg.get("min_speed", 0.05))
    slow_distance = float(approach_cfg.get("slow_distance", 0.8))
    lost_retries = int(approach_cfg.get("lost_retries", 5))
    rescan_pause = float(approach_cfg.get("rescan_pause", 0.4))
    vlm_fallback_enabled = bool(approach_cfg.get("vlm_fallback", True))

    deadline = time.time() + timeout
    consecutive_lost = 0
    vlm_used = False

    while time.time() < deadline:
        # ── Get FRESH detection (not stale cache) ────────────────────── #
        det_msg = node.get_fresh(det_topic, timeout=0.5)
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

        # ── Stop-and-rescan when target lost (motion blur recovery) ─── #
        if target is None:
            consecutive_lost += 1

            if consecutive_lost <= lost_retries:
                # Stop to eliminate motion blur and let detector see a clean frame
                node.stop()
                time.sleep(rescan_pause)

                # Fetch a genuinely new detection after the pause
                det_msg = node.get_fresh(det_topic, timeout=0.5)
                if det_msg is not None:
                    d = detections_to_dict(det_msg)
                    target = next(
                        (x for x in d["detections"]
                         if x["label"].lower() == label.lower()),
                        None,
                    )
                if target is not None:
                    consecutive_lost = 0
                    # Fall through to steering logic below
                else:
                    continue  # try again on next iteration

            else:
                # ── VLM fallback ─────────────────────────────────────── #
                if vlm_fallback_enabled:
                    vlm_result = _vlm_check_object(node, label, cam_topic)
                    vlm_used = True

                    if vlm_result is not None and vlm_result.get("visible"):
                        # VLM sees the object but YOLO doesn't — continue
                        # cautiously with LiDAR-only distance guidance
                        if scan_msg is not None:
                            scan_d = laser_scan_to_dict(scan_msg)
                            front = scan_d.get("front_min_m")

                            if front is not None and front <= stop_distance:
                                node.stop()
                                return {
                                    "status": "reached",
                                    "message": (
                                        f"VLM confirmed '{label}' ahead; "
                                        f"stopped at {front:.2f} m "
                                        f"(YOLO lost track, LiDAR + VLM guided stop)."
                                    ),
                                    "distance_m": round(front, 3),
                                    "distance_source": "vlm_lidar",
                                    "distance_reliable": False,
                                    "collision_avoidance_activated": False,
                                    "vlm_assisted": True,
                                    "vlm_response": vlm_result.get("vlm_response", ""),
                                }

                            if (collision_avoidance and front is not None
                                    and front < min_front_ca):
                                node.stop()
                                return {
                                    "status": "blocked",
                                    "collision_avoidance_activated": True,
                                    "message": (
                                        f"VLM confirmed '{label}' but obstacle "
                                        f"at {front:.2f} m (threshold "
                                        f"{min_front_ca:.2f} m)."
                                    ),
                                    "distance_m": round(front, 3),
                                    "vlm_assisted": True,
                                }

                        # VLM sees it — drive forward slowly, allow more retries
                        node.publish_twist(min_speed, 0.0)
                        time.sleep(0.15)
                        consecutive_lost = max(0, lost_retries - 2)
                        continue

                # All retries (+ VLM if enabled) exhausted — target truly lost
                node.stop()
                msg = f"'{label}' not detected after {lost_retries} retries."
                if vlm_used:
                    msg += " VLM also could not confirm the object."
                return {"status": "lost_target",
                        "message": msg,
                        "collision_avoidance_activated": False,
                        "vlm_assisted": vlm_used}

        if target is None:
            continue

        # ── Object found — reset lost counter ────────────────────────── #
        consecutive_lost = 0

        # Steering: proportional to horizontal offset from image centre
        cx = target["bbox"]["cx"]
        offset = (cx - image_width / 2.0) / (image_width / 2.0)  # [-1, 1]
        angular_z = -0.5 * offset  # negative because image-x is left→right

        # ── Fused distance estimate (bbox-aligned LiDAR + pinhole) ──── #
        dist_info = estimate_detection_distance(
            scan_msg, target, image_width, image_height, hfov_deg,
        )
        est_dist = dist_info["distance_m"]

        # Also keep the broad front sector for collision avoidance
        front_broad = None
        if scan_msg is not None:
            scan_d = laser_scan_to_dict(scan_msg)
            front_broad = scan_d.get("front_min_m")

        # ── Target reached (intentional stop) ────────────────────────── #
        if est_dist is not None and est_dist <= stop_distance:
            node.stop()
            return {
                "status": "reached",
                "message": (
                    f"Stopped ~{est_dist:.2f} m from '{label}' "
                    f"(source: {dist_info['distance_source']})."
                ),
                "distance_m": round(est_dist, 3),
                "distance_source": dist_info["distance_source"],
                "distance_reliable": dist_info["distance_reliable"],
                "collision_avoidance_activated": False,
            }

        # ── Collision avoidance safety stop (uses broad front sector) ── #
        if collision_avoidance and front_broad is not None and front_broad < min_front_ca:
            node.stop()
            return {
                "status": "blocked",
                "collision_avoidance_activated": True,
                "message": (
                    f"Collision avoidance activated: obstacle {front_broad:.2f} m ahead "
                    f"(threshold {min_front_ca:.2f} m). Approach aborted."
                ),
                "distance_m": round(est_dist, 3) if est_dist is not None else None,
                "distance_source": dist_info["distance_source"],
                "distance_reliable": dist_info["distance_reliable"],
            }

        # ── Adaptive speed: slow down when close to reduce blur ──────── #
        if est_dist is not None and est_dist < slow_distance:
            frac = est_dist / slow_distance          # 0.0 → 1.0
            speed = min_speed + (max_speed - min_speed) * frac
        else:
            speed = max_speed

        # Drive forward while steering
        node.publish_twist(speed, angular_z)
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

    while time.time() < deadline:
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
