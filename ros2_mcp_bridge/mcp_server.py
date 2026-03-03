#!/usr/bin/env python3
"""
mcp_server.py — FastMCP server that exposes ROS 2 topics/services as tools.

Uses the same FastMCP + streamable-http pattern as onit's built-in servers so
that the agent can discover and call all tools automatically.
"""

import asyncio
import base64
import collections
import json
import logging
import math
import os
import re
import subprocess
import threading as _threading
import time
import urllib.request
from typing import Any, Optional

from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.utilities.types import Image
from starlette.requests import Request as _SR
from starlette.responses import JSONResponse as _JR

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# MCP call log — circular buffer, last 200 tool invocations.                  #
# Populated by LoggingMiddleware; served at GET /mcp_api/log.                 #
# --------------------------------------------------------------------------- #

_MAX_LOG      = 200
_mcp_call_log: collections.deque = collections.deque(maxlen=_MAX_LOG)
_mcp_log_lock = _threading.Lock()
_mcp_log_seq: int = 0   # monotonic counter — increments on every call

# Detect base64-encoded blobs so we don't flood the UI with raw image data.
_B64_RE  = re.compile(r'^[A-Za-z0-9+/]{80,}={0,2}$')
_ARG_MAX  = 300   # chars for truncated argument display
_RESP_MAX = 600   # chars for truncated response display


def _trunc_arg(val: Any) -> Any:
    """Return a display-safe version of a single argument value."""
    if not isinstance(val, str):
        return val
    if len(val) > 60 and _B64_RE.match(val[:80]):
        return f"[BASE64 ~{len(val)} chars]"
    if len(val) > _ARG_MAX:
        return val[:_ARG_MAX] + f" ...({len(val) - _ARG_MAX} more chars)"
    return val


def _trunc_resp(text: str) -> str:
    """Return a compact representation of a response string."""
    if len(text) <= _RESP_MAX:
        return text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            summary = {}
            for k, v in list(parsed.items())[:12]:
                if isinstance(v, str) and len(v) > 80:
                    summary[k] = v[:80] + "..."
                elif isinstance(v, (list, dict)):
                    summary[k] = f"[{type(v).__name__} len={len(v)}]"
                else:
                    summary[k] = v
            suffix = f" ...+{len(parsed) - 12} more keys" if len(parsed) > 12 else ""
            return json.dumps(summary, ensure_ascii=False) + suffix
    except Exception:
        pass
    return text[:_RESP_MAX] + f" ...({len(text) - _RESP_MAX} more chars)"


def _result_texts(result: Any) -> tuple[str, str]:
    """Return (short_display, full_text) from a FastMCP CallToolResult."""
    content = getattr(result, "content", None)
    if content is None:
        raw = str(result)
        return _trunc_resp(raw), raw[:8000]
    parts_short: list[str] = []
    parts_full:  list[str] = []
    for item in content:
        mime = getattr(item, "mimeType", None)
        if mime and mime.startswith("image/"):
            data = getattr(item, "data", "")
            size = len(data) if isinstance(data, (bytes, bytearray)) else len(str(data))
            tag  = f"[IMAGE {mime} ~{size} chars/bytes]"
            parts_short.append(tag)
            parts_full.append(tag)
        else:
            text = str(getattr(item, "text", item))
            parts_short.append(_trunc_resp(text))
            parts_full.append(text[:8000])
    return "\n".join(parts_short), "\n".join(parts_full)


class LoggingMiddleware(Middleware):
    """Logs every tool call and appends an entry to _mcp_call_log.

    Large/binary argument values (e.g. base64 images) are summarised so the
    log stays compact and the cam_web /mcp trace UI remains snappy.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> Any:
        global _mcp_log_seq
        params    = context.message
        tool_name = getattr(params, "name", "<unknown>")
        arguments = getattr(params, "arguments", {}) or {}
        args_str  = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        logger.info("[MCP CALL] %s(%s)", tool_name, args_str)

        t0        = time.monotonic()
        wall_time = time.time()

        display_args = {k: _trunc_arg(v)  for k, v in arguments.items()}
        full_args    = {k: str(v)[:4096]  for k, v in arguments.items()}

        def _append(status: str, short_resp: str, full_resp: str, elapsed: float) -> None:
            global _mcp_log_seq
            with _mcp_log_lock:
                _mcp_log_seq += 1
                _mcp_call_log.append({
                    "seq":           _mcp_log_seq,
                    "ts":            wall_time,
                    "tool":          tool_name,
                    "args":          display_args,
                    "args_full":     full_args,
                    "status":        status,
                    "duration_ms":   round(elapsed * 1000),
                    "response":      short_resp,
                    "response_full": full_resp,
                })

        try:
            result   = await call_next(context)
            elapsed  = time.monotonic() - t0
            is_error = getattr(result, "isError", False)
            status   = "error" if is_error else "ok"
            logger.info("[MCP RETURN] %s -> %s in %.3fs", tool_name, status.upper(), elapsed)
            short_r, full_r = _result_texts(result)
            _append(status, short_r, full_r, elapsed)
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("[MCP ERROR] %s raised %s in %.3fs", tool_name, exc, elapsed)
            _append("exception", msg, msg, elapsed)
            raise

from ros2_mcp_bridge.ros_node import (
    ROS2BridgeNode,
    laser_scan_to_dict,
    odometry_to_dict,
    detections_to_dict,
    estimate_detection_distance,
    sensor_state_to_dict,
    battery_state_to_dict,
    imu_to_dict,
    joint_state_to_dict,
    magnetic_field_to_dict,
    camera_info_to_dict,
)

from ros2_mcp_bridge.dsl_runtime import DSLRuntime

# Singleton reference injected by bridge.py before the server starts
_node: Optional[ROS2BridgeNode] = None
_dsl: Optional[DSLRuntime] = None

# --------------------------------------------------------------------------- #
# Session-level state (lives for the lifetime of the bridge process)
# --------------------------------------------------------------------------- #

# Named pose waypoints: {name: {x, y, yaw_deg}}
_waypoints: dict[str, dict] = {}

# LLM scratchpad: {key: {"value": str, "description": str}}
_memory: dict[str, dict] = {}


def set_node(node: ROS2BridgeNode):
    """Called by bridge.py to inject the live rclpy node."""
    global _node, _dsl
    _node = node
    _dsl = DSLRuntime(node, node._cfg, _memory, _waypoints)


# --------------------------------------------------------------------------- #
# ROS 2 CLI service-call helper (used by SLAM management tools)
# --------------------------------------------------------------------------- #

_DEFAULT_MAPS_DIR = os.path.expanduser("~/maps")


def _resolve_maps_dir(directory: str) -> str:
    """Return an absolute maps directory path.

    - Empty / blank → ~/maps  (the default)
    - Starts with ~ → expand ~
    - Already absolute → use as-is
    - Relative (e.g. 'maps') → resolve under $HOME so 'maps' → ~/maps
    """
    d = directory.strip()
    if not d:
        return _DEFAULT_MAPS_DIR
    d = os.path.expanduser(d)
    if not os.path.isabs(d):
        d = os.path.join(os.path.expanduser("~"), d)
    return d


def _ros_service_call(service: str, type_str: str, request: str,
                      timeout: float = 10.0) -> dict:
    """Call a ROS 2 service via `ros2 service call` subprocess.

    Returns {"output": ..., "returncode": 0} on success, or {"error": ...}.
    """
    cmd = ["ros2", "service", "call", service, type_str, request]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return {"error": (r.stderr.strip() or r.stdout.strip() or "service call failed"),
                    "returncode": r.returncode}
        return {"output": r.stdout.strip(), "returncode": 0}
    except subprocess.TimeoutExpired:
        return {"error": f"service call timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


def _service_exists(service_name: str, timeout: float = 3.0) -> bool:
    """Check whether a ROS 2 service is currently advertised."""
    try:
        r = subprocess.run(
            ["ros2", "service", "list"],
            capture_output=True, text=True, timeout=timeout,
        )
        return service_name in r.stdout.splitlines()
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# FastMCP server
# --------------------------------------------------------------------------- #

mcp = FastMCP(
    "ROS2BridgeMCPServer",
    instructions=(
        "Tools for controlling a TurtleBot3 robot via ROS 2.\n"
        "\n"
        "SENSOR TOOLS: get_camera_image (visual), get_full_res_image (high-resolution on-demand capture), "
        "get_sensor_snapshot (pose+lidar+detections+battery "
        "all-in-one), get_camera_info (intrinsic matrix, FOV, distortion — use for pixel↔3D math), "
        "get_laser_scan, get_robot_pose, detect_objects_in_image, get_imu, get_battery_state.\n"
        "\n"
        "DEPTH / FLOOR TOOLS: get_depth_map (colourised monocular depth image + zone distances, "
        "metric-anchored when LiDAR is available), get_depth_zones (left/center/right obstacle "
        "distances without an image), analyse_floor (floor anomaly/obstacle/spill detection). "
        "These require the onit-depth-service to be running.\n"
        "\n"
        "MOTION TOOLS: move_distance (precise, odometry-closed-loop), rotate_angle (precise), "
        "move_robot (timed open-loop), stop_robot.\n"
        "\n"
        "NAVIGATION: navigate_to_pose (Nav2, map frame), nav2_cancel_navigation, "
        "save_waypoint / go_to_waypoint / list_waypoints / delete_waypoint / update_waypoint "
        "(session-level named poses). Use get_map_info() to check the current occupancy "
        "grid bounds and resolution before planning long-range goals.\n"
        "\n"
        "SLAM / MAP MANAGEMENT: slam_get_status (check if slam_toolbox is running and "
        "what mode it is in), get_map_info (map dimensions, resolution, free/occupied cell "
        "counts), slam_save_map (save PGM+YAML map file AND pose graph to ~/maps/), "
        "slam_serialize_map (serialize pose graph only, for later localization resumption), "
        "slam_load_map (deserialize pose graph and switch to localization mode — "
        "match_type 3=LOCALIZE_AT_POSE is best when you know start position), "
        "slam_pause_mapping (pause/resume scan processing during fast motion), "
        "slam_clear_map (wipe current SLAM state and start fresh), "
        "nav2_set_initial_pose (for AMCL-based localization on a static pre-built map), "
        "list_map_files (list ~/maps/ contents). "
        "SLAM WORKFLOW: (1) Robot boots → slam_toolbox starts automatically in mapping mode. "
        "(2) Drive/navigate to explore. (3) slam_save_map to persist map+graph. "
        "(4) On next session, slam_load_map(match_type=3, x, y, yaw_deg) to resume "
        "localization — no initial pose needed for AMCL.\n"
        "\n"
        "VISION SUB-AGENT (A2A): ask_vision_agent → Qwen3-VL-8B, best for scene description, "
        "object finding, OCR, counting, navigation safety assessment.\n"
        "\n"
        "MEMORY: set_memory / get_memory / list_memory / clear_memory — persistent scratchpad for "
        "noting observations, plans, and task state between tool calls.\n"
        "\n"
        "DSL PROGRAMS: Preferred approach for all search, patrol, approach-object, "
        "wall-follow, and any multi-step reactive behaviour. Use dsl_run_inline for "
        "quick one-shot scripts and dsl_store_program + dsl_run_program for reusable "
        "routines. Programs execute locally at ~20 Hz with access to get_scan(), "
        "get_detections() (VLM-first, falls back to YOLO; each detection includes source='vlm'/'yolo'), "
        "get_image() (BGR numpy), move(), rotate(), stop(), "
        "move_distance(), find_object(), approach_object(), follow_wall(), "
        "navigate_to_pose(), cancel_navigation(), set_initial_pose(x,y,yaw_deg), "
        "get_map_info(), and a full OpenCV toolkit (cv_hsv_filter, "
        "cv_find_contours, cv_detect_aruco, cv_canny, cv_hough_lines, …). "
        "WORKFLOW: (1) dsl_validate(source) — instant syntax + forbidden-call check. "
        "(2) dsl_run_inline(source, dry_run=True) — test logic with live sensors, all "
        "motion replaced by log stubs, no robot movement. (3) Run for real. "
        "Runtime errors include DSL-frame tracebacks with line numbers. "
        "Prefer DSL over chaining individual MCP tool calls — it avoids round-trip "
        "latency and does not incur repeated image/VLM costs. "
        "Manage stored programs with dsl_list_programs, dsl_get_source, "
        "dsl_delete_program, dsl_stop_program.\n"
        "\n"
        "STRATEGY: Use get_sensor_snapshot "
        "instead of calling get_camera_image + get_laser_scan + get_robot_pose separately. "
        "Save interesting locations with save_waypoint so you can return to them. "
        "Use set_memory to record what rooms/areas have been checked. "
        "After mapping a new environment call slam_save_map immediately so the map "
        "survives a reboot. Use slam_load_map on the next session to resume localization "
        "without re-exploring. "
        "For complex navigation that reacts to real-time sensor data (obstacle weaving, "
        "tracking a moving object, patrol routes), write a DSL program instead of chaining "
        "many individual move/read tool calls."
    ),
    middleware=[LoggingMiddleware()],
)


# --------------------------------------------------------------------------- #
# MCP trace API  (consumed by the cam_web /mcp trace page)                    #
# --------------------------------------------------------------------------- #

@mcp.custom_route("/mcp_api/log", methods=["GET"])
async def _mcp_api_log(request: _SR) -> _JR:
    """Return the circular call log as JSON for the cam_web trace UI."""
    with _mcp_log_lock:
        entries = list(_mcp_call_log)
    return _JR({"entries": entries, "total_ever": _mcp_log_seq})


@mcp.custom_route("/mcp_api/clear", methods=["POST"])
async def _mcp_api_clear(request: _SR) -> _JR:
    """Clear the call log."""
    global _mcp_log_seq
    with _mcp_log_lock:
        _mcp_call_log.clear()
    return _JR({"ok": True, "seq_after_clear": _mcp_log_seq})


# --------------------------------------------------------------------------- #
# Introspection tools
# --------------------------------------------------------------------------- #


@mcp.tool(
    title="List ROS 2 Topics",
    description=(
        "Return the list of currently active ROS 2 topics and their message "
        "types as seen by the bridge node."
    ),
)
def list_ros2_topics() -> str:
    """List all active ROS 2 topics."""
    topics = _node.list_topics()
    lines = [f"{t}  [{', '.join(ts)}]" for t, ts in sorted(topics)]
    return json.dumps({"topics": lines})


@mcp.tool(
    title="List ROS 2 Services",
    description="Return the list of currently available ROS 2 services.",
)
def list_ros2_services() -> str:
    """List all active ROS 2 services."""
    services = _node.list_services()
    lines = [f"{s}  [{', '.join(ts)}]" for s, ts in sorted(services)]
    return json.dumps({"services": lines})


# --------------------------------------------------------------------------- #
# Sensor tools
# --------------------------------------------------------------------------- #


@mcp.tool(
    title="Get Camera Image",
    description=(
        "Capture the latest image from the robot's camera. "
        "Returns an image that can be directly interpreted visually. "
        "Use this to visually inspect the robot's surroundings."
    ),
)
def get_camera_image(timeout: float = 3.0) -> Image | str:
    """
    Args:
        timeout: Seconds to wait for a frame (default 3.0).

    Returns:
        Image: MCP ImageContent containing the JPEG frame (mimeType image/jpeg).
        str:   JSON error string if no frame was available within the timeout.
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("camera", {}).get("topic", "/camera/image_raw/compressed")
    msg = _node.get_latest(topic, timeout=timeout)
    if msg is None:
        return json.dumps({"error": f"No image received on {topic} within {timeout}s."})
    # Return a proper MCP ImageContent so vision-language models receive the
    # image data rather than a raw JSON blob.  FastMCP converts Image(data, format)
    # → ImageContent(mimeType="image/jpeg", data=<base64>).
    fmt = (msg.format or "jpeg").lower().split("/")[-1]  # normalise e.g. "jpeg" or "image/jpeg"
    return Image(data=bytes(msg.data), format=fmt)


@mcp.tool(
    title="Get Laser Scan",
    description=(
        "Return the latest LiDAR scan from the robot. "
        "Includes minimum obstacle distances in front, left, right, and rear "
        "sectors as well as the full range array."
    ),
)
def get_laser_scan(timeout: float = 2.0) -> str:
    """
    Args:
        timeout: Seconds to wait for a scan (default 2.0).
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("laser", {}).get("topic", "/scan")
    msg = _node.get_latest(topic, timeout=timeout)
    if msg is None:
        return json.dumps({"error": f"No scan received on {topic} within {timeout}s."})
    return json.dumps(laser_scan_to_dict(msg))


@mcp.tool(
    title="Get Robot Pose",
    description=(
        "Return the robot's current position and orientation from odometry. "
        "Pose is expressed as x, y (metres) and yaw (radians and degrees) "
        "in the odometry frame."
    ),
)
def get_robot_pose(timeout: float = 2.0) -> str:
    """
    Args:
        timeout: Seconds to wait for an odometry message (default 2.0).
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("odom", {}).get("topic", "/odom")
    msg = _node.get_latest(topic, timeout=timeout)
    if msg is None:
        return json.dumps({"error": f"No odometry received on {topic} within {timeout}s."})
    return json.dumps(odometry_to_dict(msg))


@mcp.tool(
    title="Get Camera Info",
    description=(
        "Return the camera's intrinsic calibration parameters from the "
        "sensor_msgs/CameraInfo topic (published by the camera driver). "
        "Includes the full 3×3 pinhole intrinsic matrix K, distortion "
        "coefficients D, projection matrix P, image dimensions, and derived "
        "horizontal/vertical field-of-view angles. "
        "Use these parameters for camera-space ↔ physical-space calculations:\n"
        "  Pixel → 3-D ray: X = (u - cx) / fx,  Y = (v - cy) / fy,  Z = 1\n"
        "  3-D → pixel:     u = fx * X/Z + cx,   v = fy * Y/Z + cy\n"
        "If CameraInfo is not being published, estimated values are derived "
        "from the bridge configuration (image_width, camera_hfov_deg) and "
        "returned with calibration_source='config_estimate'."
    ),
)
def get_camera_info(timeout: float = 2.0) -> str:
    """
    Args:
        timeout: Seconds to wait for a CameraInfo message (default 2.0).
                 CameraInfo is latched so it usually arrives immediately.
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("camera_info", {}).get("topic", "/camera/camera_info")
    msg = _node.get_latest(topic, timeout=timeout)

    if msg is not None:
        result = camera_info_to_dict(msg)
        result["calibration_source"] = "camera_info_topic"
        result["topic"] = topic
    else:
        # Fall back to bridge config estimates using the pinhole model
        w = int(_node._cfg.get("image_width", 640))
        hfov_deg = float(_node._cfg.get("camera_hfov_deg", 62.0))
        hfov_rad = math.radians(hfov_deg)
        fx = round((w / 2.0) / math.tan(hfov_rad / 2.0), 4)
        # Assume square pixels and a 4:3 aspect ratio for fy/cy
        h = int(w * 3 / 4)
        fy = fx  # square-pixel assumption
        cx = round(w / 2.0, 4)
        cy = round(h / 2.0, 4)
        vfov_deg = round(math.degrees(2.0 * math.atan2(h / 2.0, fy)), 2)
        K = [fx, 0.0, cx,
             0.0, fy, cy,
             0.0, 0.0, 1.0]
        result = {
            "image_width":  w,
            "image_height": h,
            "distortion_model": "plumb_bob",
            "K":  [round(v, 4) for v in K],
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "D":  [0.0, 0.0, 0.0, 0.0, 0.0],
            "P":  [fx, 0.0, cx, 0.0,
                   0.0, fy, cy, 0.0,
                   0.0, 0.0, 1.0, 0.0],
            "R":  [1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0],
            "hfov_deg": round(hfov_deg, 2),
            "vfov_deg": vfov_deg,
            "calibration_source": "config_estimate",
            "note": (
                f"No CameraInfo received on {topic}. "
                "Values derived from bridge.yaml camera_hfov_deg and image_width. "
                "Run camera calibration and ensure the camera driver publishes "
                f"sensor_msgs/CameraInfo on {topic} for accurate results."
            ),
        }

    # Always append the bridge-level config values for reference
    result["bridge_config"] = {
        "image_width":    _node._cfg.get("image_width", 640),
        "camera_hfov_deg": float(_node._cfg.get("camera_hfov_deg", 62.0)),
        "lidar_height_m": float(_node._cfg.get("lidar_height_m", 0.13)),
    }
    return json.dumps(result)


@mcp.tool(
    title="Get Full Resolution Image",
    description=(
        "Request an on-demand full-resolution JPEG image from the Pi camera. "
        "Unlike get_camera_image (which returns a compressed live stream frame at "
        "reduced resolution), this calls the /camera/capture_full_res ROS 2 service "
        "on the robot's Raspberry Pi and returns the highest-quality image the camera "
        "can produce. "
        "Use this when fine detail, text recognition, or high-quality vision analysis "
        "is required. The capture takes 1-3 seconds. "
        "IMPORTANT: The full-resolution image is intended for LOCAL processing only "
        "(e.g. passing to detect_objects_in_image or saving to disk). "
        "Do NOT send it directly to a VLM / vision-language model or to any external "
        "internet service — use get_camera_image for that purpose instead, as it "
        "provides a suitably sized frame for visual inference. "
        "Returns an MCP Image (JPEG) on success, or an error JSON string on failure."
    ),
)
def get_full_res_image(
    width: int = 0,
    height: int = 0,
    jpeg_quality: int = 90,
    timeout: float = 20.0,
) -> Image | str:
    """
    Args:
        width:        Desired image width in pixels. 0 = camera native resolution.
        height:       Desired image height in pixels. 0 = camera native resolution.
        jpeg_quality: JPEG compression quality 1-100 (default 90).
        timeout:      Seconds to wait for the capture service (default 20).

    Returns:
        Image: MCP ImageContent with the full-resolution JPEG.
        str:   JSON error string if the capture failed.
    """
    result = _node.capture_full_res(
        width=width,
        height=height,
        jpeg_quality=jpeg_quality,
        timeout=timeout,
    )
    if result.get("status") != "succeeded":
        return json.dumps({"error": result.get("message", "Capture failed.")})

    try:
        jpeg_bytes = base64.b64decode(result["jpeg_b64"])
    except Exception as exc:
        return json.dumps({"error": f"Failed to decode captured image: {exc}"})

    return Image(data=jpeg_bytes, format="jpeg")


@mcp.tool(
    title="Detect Objects in Image",
    description=(
        "Run YOLO object detection on a JPEG image and return detected objects "
        "with class labels, confidence scores, bounding boxes in image coordinates, "
        "and a fused LiDAR distance estimate for each detection. "
        "If jpeg_b64 is omitted the latest camera frame is captured automatically. "
        "Distance is computed by aligning the detection bbox to specific LiDAR rays "
        "and cross-validating with a pinhole-model depth estimate from the bbox height. "
        "If the object is below the 2D LiDAR scan plane (e.g. on the floor), the "
        "bbox estimate is used as fallback and distance_reliable will be false."
    ),
)
def detect_objects_in_image(jpeg_b64: str = "", timeout: float = 3.0) -> str:
    """
    Args:
        jpeg_b64: Base64-encoded JPEG image to run detection on.
                  Leave empty to automatically capture the current camera frame.
        timeout:  Seconds to wait for a camera frame when jpeg_b64 is empty (default 3.0).
    """
    # ── Resolve JPEG bytes ────────────────────────────────────────────────
    if jpeg_b64:
        try:
            jpeg_bytes = base64.b64decode(jpeg_b64)
        except Exception as exc:
            return json.dumps({"error": f"Invalid base64 input: {exc}"})
    else:
        cfg_topics = _node._cfg.get("topics", {})
        topic = cfg_topics.get("camera", {}).get("topic", "/camera/image_raw/compressed")
        msg = _node.get_latest(topic, timeout=timeout)
        if msg is None:
            return json.dumps({"error": f"No camera frame on {topic} within {timeout}s."})
        jpeg_bytes = bytes(msg.data)

    # ── Call the detector HTTP service ────────────────────────────────────
    detector_url = _node._cfg.get("detector_service_url", "http://localhost:8082/detect")
    req = urllib.request.Request(
        detector_url,
        data=jpeg_bytes,
        method="POST",
        headers={"Content-Type": "image/jpeg"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            result = json.loads(resp.read())
    except Exception as exc:
        return json.dumps({"error": f"Detector service unreachable at {detector_url}: {exc}"})

    if "error" in result:
        return json.dumps(result)

    # ── Enrich each detection with fused LiDAR distance ──────────────────
    cfg_topics = _node._cfg.get("topics", {})
    laser_topic = cfg_topics.get("laser", {}).get("topic", "/scan")
    scan_msg = _node.get_latest(laser_topic, timeout=0.5)
    image_width = _node._cfg.get("image_width", 640)
    image_height = int(image_width * 480 / 640)
    hfov_deg = float(_node._cfg.get("camera_hfov_deg", 62.0))

    for det in result.get("detections", []):
        dist = estimate_detection_distance(
            scan_msg, det, image_width, image_height, hfov_deg,
        )
        det.update(dist)

    return json.dumps(result)


# --------------------------------------------------------------------------- #
# Depth estimation & floor analysis tools  (backed by depth_service HTTP)     #
# --------------------------------------------------------------------------- #

def _depth_cfg() -> tuple[bool, str]:
    """Return (enabled, base_url) from bridge.yaml depth_service section."""
    ds = _node._cfg.get("depth_service", {})
    return ds.get("enabled", False), ds.get("url", "http://localhost:8083")


def _depth_service_healthy(base_url: str) -> bool:
    """Quick health-check against the depth service."""
    try:
        req = urllib.request.Request(f"{base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def _capture_jpeg_bytes(timeout: float = 3.0) -> bytes | None:
    """Grab the latest camera JPEG from the ROS topic (reusable helper)."""
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("camera", {}).get("topic", "/camera/image_raw/compressed")
    msg = _node.get_latest(topic, timeout=timeout)
    if msg is None:
        return None
    return bytes(msg.data)


@mcp.tool(
    title="Get Depth Map",
    description=(
        "Run monocular depth estimation (Depth Anything V2) on the current camera "
        "frame and return a colourised depth-map image (JPEG). "
        "Closer pixels appear warmer (yellow/red), farther pixels appear cooler (blue/purple).\n"
        "\n"
        "Also returns left/center/right zone obstacle distance estimates (heuristic, "
        "metres) via the X-Depth-Zones header when a LiDAR front distance is available "
        "for metric anchoring.\n"
        "\n"
        "If jpeg_b64 is omitted the latest camera frame is captured automatically."
    ),
)
def get_depth_map(jpeg_b64: str = "", timeout: float = 3.0) -> str | Image:
    """
    Args:
        jpeg_b64: Base64-encoded JPEG to analyse. Leave empty to auto-capture.
        timeout:  Seconds to wait for camera frame when auto-capturing (default 3.0).
    """
    enabled, base_url = _depth_cfg()
    if not enabled:
        return json.dumps({"error": "Depth service is disabled in bridge.yaml (depth_service.enabled=false)."})
    if not _depth_service_healthy(base_url):
        return json.dumps({"error": f"Depth service not reachable at {base_url}. Is onit-depth-service running?"})

    # Resolve JPEG bytes
    if jpeg_b64:
        try:
            jpeg_bytes = base64.b64decode(jpeg_b64)
        except Exception as exc:
            return json.dumps({"error": f"Invalid base64: {exc}"})
    else:
        jpeg_bytes = _capture_jpeg_bytes(timeout)
        if jpeg_bytes is None:
            return json.dumps({"error": "No camera frame available."})

    # Try metric depth if we have a LiDAR front distance
    cfg_topics = _node._cfg.get("topics", {})
    laser_topic = cfg_topics.get("laser", {}).get("topic", "/scan")
    scan_msg = _node.get_latest(laser_topic, timeout=0.5)
    front_m = 0.0
    if scan_msg is not None:
        ranges = list(scan_msg.ranges)
        n = len(ranges)
        # front = index 0 for standard LiDAR
        window = ranges[max(0, n-5):] + ranges[:5]
        valid = [r for r in window if scan_msg.range_min < r < scan_msg.range_max]
        if valid:
            front_m = min(valid)

    if front_m > 0.05:
        # Metric depth via multipart POST
        boundary = b"----DepthBoundary"
        body = (
            b"--" + boundary + b"\r\n"
            b"Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg_bytes + b"\r\n"
            b"--" + boundary + b"\r\n"
            b"Content-Disposition: form-data; name=\"front_m\"\r\n\r\n" +
            f"{front_m:.3f}".encode() + b"\r\n"
            b"--" + boundary + b"--\r\n"
        )
        req = urllib.request.Request(
            f"{base_url}/depth/metric",
            data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        )
    else:
        # Relative depth only
        req = urllib.request.Request(
            f"{base_url}/depth",
            data=jpeg_bytes, method="POST",
            headers={"Content-Type": "image/jpeg"},
        )

    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            depth_jpeg = resp.read()
            zones_hdr = resp.headers.get("X-Depth-Zones")
    except Exception as exc:
        return json.dumps({"error": f"Depth inference failed: {exc}"})

    # Return as an image; attach zone info as a follow-up text block
    result_parts: list = [Image(data=depth_jpeg, format="jpeg")]
    zones = {}
    if zones_hdr:
        try:
            zones = json.loads(zones_hdr)
        except Exception:
            pass
    if not zones and front_m <= 0.05:
        # Fall back to GET /depth_zones if we cached one
        try:
            zr = urllib.request.urlopen(f"{base_url}/depth_zones", timeout=3.0)
            zones = json.loads(zr.read())
        except Exception:
            pass

    meta = {"front_lidar_m": round(front_m, 3) if front_m > 0.05 else None}
    if zones:
        meta["depth_zones"] = zones
    return [*result_parts, json.dumps(meta)]


@mcp.tool(
    title="Get Depth Zones",
    description=(
        "Get obstacle distance estimates (metres) for left, center, and right "
        "image zones from the depth model, without returning an image. "
        "If a LiDAR front distance is available, distances are metric-anchored; "
        "otherwise they are heuristic relative estimates.\n"
        "\n"
        "Returns: {\"left_m\": <float>, \"center_m\": <float>, \"right_m\": <float>}\n"
        "\n"
        "Use this for quick obstacle-clearance checks without the overhead of "
        "transferring a depth-map image."
    ),
)
def get_depth_zones(timeout: float = 3.0) -> str:
    """
    Args:
        timeout: Seconds to wait for camera frame (default 3.0).
    """
    enabled, base_url = _depth_cfg()
    if not enabled:
        return json.dumps({"error": "Depth service disabled in bridge.yaml."})
    if not _depth_service_healthy(base_url):
        return json.dumps({"error": f"Depth service not reachable at {base_url}."})

    jpeg_bytes = _capture_jpeg_bytes(timeout)
    if jpeg_bytes is None:
        return json.dumps({"error": "No camera frame available."})

    req = urllib.request.Request(
        f"{base_url}/depth/zones",
        data=jpeg_bytes, method="POST",
        headers={"Content-Type": "image/jpeg"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            return resp.read().decode()
    except Exception as exc:
        return json.dumps({"error": f"Depth zone estimation failed: {exc}"})


@mcp.tool(
    title="Analyse Floor",
    description=(
        "Analyse the floor region of the current camera frame for anomalies: "
        "obstacles, steps, wet/reflective surfaces, or spills.\n"
        "\n"
        "Returns JSON:\n"
        "  ok     — true if floor appears clear\n"
        "  reason — human-readable summary\n"
        "  details.depth_var   — variance of floor depth (high = obstacle/step)\n"
        "  details.depth_mean  — mean relative depth of floor region\n"
        "  details.texture_lap — Laplacian texture energy (low = wet/reflective)\n"
        "\n"
        "Use before driving forward to check for ground-level hazards that "
        "the 2D LiDAR cannot detect (the LiDAR beam is ~13 cm high and misses "
        "objects below it)."
    ),
)
def analyse_floor(timeout: float = 3.0) -> str:
    """
    Args:
        timeout: Seconds to wait for camera frame (default 3.0).
    """
    enabled, base_url = _depth_cfg()
    if not enabled:
        return json.dumps({"error": "Depth service disabled in bridge.yaml."})
    if not _depth_service_healthy(base_url):
        return json.dumps({"error": f"Depth service not reachable at {base_url}."})

    jpeg_bytes = _capture_jpeg_bytes(timeout)
    if jpeg_bytes is None:
        return json.dumps({"error": "No camera frame available."})

    req = urllib.request.Request(
        f"{base_url}/floor",
        data=jpeg_bytes, method="POST",
        headers={"Content-Type": "image/jpeg"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            return resp.read().decode()
    except Exception as exc:
        return json.dumps({"error": f"Floor analysis failed: {exc}"})


@mcp.tool(
    title="Get Sensor Snapshot",
    description=(
        "Return all non-image sensor data in a single call: robot pose (x, y, yaw), "
        "LiDAR sector distances (front/left/right/rear), current object detections, "
        "and battery state. "
        "Use this instead of calling get_robot_pose + get_laser_scan + get_detections "
        "separately — it is much faster and uses fewer tool-call turns. "
        "Call get_camera_image in addition if you need a visual frame."
    ),
)
def get_sensor_snapshot(timeout: float = 2.0) -> str:
    """
    Args:
        timeout: Seconds to wait for each sensor message (default 2.0).
    """
    cfg_topics = _node._cfg.get("topics", {})
    snapshot: dict = {}

    # Pose
    odom_topic = cfg_topics.get("odom", {}).get("topic", "/odom")
    odom_msg = _node.get_latest(odom_topic, timeout=timeout)
    snapshot["pose"] = odometry_to_dict(odom_msg) if odom_msg is not None else None

    # LiDAR
    laser_topic = cfg_topics.get("laser", {}).get("topic", "/scan")
    laser_msg = _node.get_latest(laser_topic, timeout=timeout)
    if laser_msg is not None:
        s = laser_scan_to_dict(laser_msg)
        snapshot["lidar"] = {
            "front_min_m": s.get("front_min_m"),
            "left_min_m":  s.get("left_min_m"),
            "right_min_m": s.get("right_min_m"),
            "rear_min_m":  s.get("rear_min_m"),
        }
    else:
        snapshot["lidar"] = None

    # Detections (enriched with fused distance estimates)
    det_topic = cfg_topics.get("detections", {}).get("topic", "/detections")
    det_msg = _node.get_latest(det_topic, timeout=timeout)
    if det_msg is not None:
        det_list = detections_to_dict(det_msg)["detections"]
        image_width = _node._cfg.get("image_width", 640)
        image_height = int(image_width * 480 / 640)
        hfov_deg = float(_node._cfg.get("camera_hfov_deg", 62.0))
        for det in det_list:
            dist = estimate_detection_distance(
                laser_msg, det, image_width, image_height, hfov_deg,
            )
            det.update(dist)
        snapshot["detections"] = det_list
    else:
        snapshot["detections"] = []

    # Battery
    bat_topic = cfg_topics.get("battery", {}).get("topic", "/battery_state")
    bat_msg = _node.get_latest(bat_topic, timeout=0.5)  # non-blocking fallback
    snapshot["battery"] = battery_state_to_dict(bat_msg) if bat_msg is not None else None

    return json.dumps(snapshot)


# --------------------------------------------------------------------------- #
# Motion tools
# --------------------------------------------------------------------------- #


@mcp.tool(
    title="Move Robot",
    description=(
        "Low-level velocity command — publishes a velocity for a set duration. "
        "Positive linear_x moves forward; positive angular_z turns left. "
        "WARNING: This is open-loop (time-based), NOT distance-based. "
        "To move an exact distance (e.g. 1 metre), use move_distance instead. "
        "To rotate an exact angle, use rotate_angle instead. "
        "Default duration is only 0.5 s at max 0.22 m/s = ~0.11 m. "
        "When collision_avoidance is enabled (default) and LiDAR detects an "
        "obstacle, the command is suppressed."
    ),
)
def move_robot(
    linear_x: float = 0.0,
    angular_z: float = 0.0,
    duration: float = 0.5,
    collision_avoidance: bool = True,
) -> str:
    """
    Args:
        linear_x: Forward velocity in m/s. Clamped to the configured
                  robot.max_linear_speed (default 0.22).
        angular_z: Rotational velocity in rad/s. Clamped to the configured
                   robot.max_angular_speed (default 2.84).
        duration: How long to drive for, in seconds (default 0.5, max 10).
                  The robot is stopped automatically once this time elapses.
        collision_avoidance: If True (default), check LiDAR before moving and
                             suppress the command if an obstacle is too close.
                             Set to False to override (use with caution).
    """
    duration = max(0.1, min(float(duration), 10.0))  # clamp 0.1–10 s
    _node.clear_stop_event()  # clear any prior stop/cancel before starting fresh motion

    ca_global = _node._cfg.get("collision_avoidance", {}).get("enabled", True)

    if collision_avoidance and ca_global and linear_x != 0.0:
        ca_result = _node.check_collision(linear_x)
        if ca_result["blocked"]:
            _node.stop()
            return json.dumps({
                "status": "blocked",
                "collision_avoidance_activated": True,
                "message": ca_result["message"],
                "obstacle_distance_m": ca_result["distance_m"],
                "sector": ca_result["sector"],
            })

    _node.publish_twist_for_duration(linear_x, angular_z, duration)
    return json.dumps({
        "status": "ok",
        "linear_x": linear_x,
        "angular_z": angular_z,
        "duration": duration,
        "collision_avoidance_activated": False,
    })


@mcp.tool(
    title="Stop Robot",
    description="Immediately stop the robot by publishing a zero-velocity command.",
)
def stop_robot() -> str:
    """Stop all robot motion."""
    _node.stop()
    return json.dumps({"status": "stopped"})


@mcp.tool(
    title="Move Distance",
    description=(
        "Drive the robot a precise distance in metres using closed-loop "
        "odometry feedback. Positive distance = forward, negative = backward. "
        "The robot accelerates, maintains cruise speed, and decelerates to "
        "stop at the target. Much more accurate than move_robot for "
        "specific distances. Collision avoidance is checked before and "
        "during motion. Works on both stock TurtleBot3 and Pico variant. "
        "If the robot consistently over-shoots or under-shoots, run "
        "calibrate_motion and update wheel_radius_scale in bridge.yaml."
    ),
)
def move_distance(
    distance: float,
    speed: float = 0.0,
    timeout: float = 30.0,
    collision_avoidance: bool = True,
) -> str:
    """
    Args:
        distance: Distance to travel in metres. Positive = forward,
                  negative = backward.
        speed: Cruise speed in m/s (default 0 = 80% of max_linear_speed).
        timeout: Maximum seconds to spend driving (default 30).
        collision_avoidance: If True (default), check LiDAR before and
                             during motion.
    """
    ca_global = _node._cfg.get("collision_avoidance", {}).get("enabled", True)
    effective_ca = collision_avoidance and ca_global
    _node.clear_stop_event()  # clear any prior stop/cancel before starting fresh motion
    result = _node.move_distance(distance, speed, timeout, effective_ca)
    return json.dumps(result)


@mcp.tool(
    title="Rotate Angle",
    description=(
        "Rotate the robot a precise angle in degrees using closed-loop "
        "odometry feedback. Positive angle = counter-clockwise (left), "
        "negative = clockwise (right). The robot decelerates as it "
        "approaches the target heading. Much more accurate than using "
        "move_robot with angular_z for precise turns. "
        "Works on both stock TurtleBot3 and Pico variant."
    ),
)
def rotate_angle(
    angle: float,
    speed: float = 0.0,
    timeout: float = 20.0,
) -> str:
    """
    Args:
        angle: Angle to rotate in degrees. Positive = counter-clockwise (left),
               negative = clockwise (right).
        speed: Angular speed in rad/s (default 0 = 50% of max_angular_speed).
        timeout: Maximum seconds to spend rotating (default 20).
    """
    _node.clear_stop_event()  # clear any prior stop/cancel before starting fresh motion
    result = _node.rotate_angle(angle, speed, timeout)
    return json.dumps(result)


@mcp.tool(
    title="Calibrate Motion",
    description=(
        "Calibration helper: drives the robot a specified distance according "
        "to odometry, then stops and reports the odometry-measured distance. "
        "The user should measure the actual distance traveled with a tape "
        "measure, then compute: wheel_radius_scale = odometry_reported / actual. "
        "Set this value in bridge.yaml under robot.wheel_radius_scale to "
        "correct future move_distance calls. Alternatively, adjust the "
        "wheel_radius parameter in the ROS 2 param file (burger.yaml or "
        "burger_pico.yaml) using: new_radius = current × (actual / odom). "
        "This works for both stock TurtleBot3 and Pico variants."
    ),
)
def calibrate_motion(
    distance: float = 1.0,
    speed: float = 0.0,
    timeout: float = 30.0,
) -> str:
    """
    Args:
        distance: Distance to drive in metres (default 1.0). Use a value
                  long enough to measure accurately (0.5-2.0 m recommended).
        speed: Cruise speed in m/s (default 0 = 80% of max speed).
        timeout: Maximum seconds (default 30).
    """
    result = _node.calibrate_motion(distance, speed, timeout)
    return json.dumps(result)


@mcp.tool(
    title="Get Battery State",
    description=(
        "Return the robot's current battery voltage and charge percentage. "
        "Uses sensor_msgs/BatteryState if available."
    ),
)
def get_battery_state(timeout: float = 2.0) -> str:
    """
    Args:
        timeout: Seconds to wait for a battery message (default 2.0).
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("battery", {}).get("topic", "/battery_state")
    msg = _node.get_latest(topic, timeout=timeout)
    if msg is None:
        return json.dumps({"error": f"No battery data received on {topic} within {timeout}s."})
    return json.dumps(battery_state_to_dict(msg))


@mcp.tool(
    title="Get Sensor State",
    description=(
        "Return the robot's low-level sensor state including encoder counts, "
        "torque status, bumper, and battery voltage. "
        "This is a TurtleBot3-specific topic; returns an error if not available."
    ),
)
def get_sensor_state(timeout: float = 2.0) -> str:
    """
    Args:
        timeout: Seconds to wait for a sensor_state message (default 2.0).
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("sensor_state", {}).get("topic", "/sensor_state")
    msg = _node.get_latest(topic, timeout=timeout)
    if msg is None:
        return json.dumps({"error": f"No sensor state received on {topic} within {timeout}s. "
                           "This topic requires turtlebot3_node to be running."})
    return json.dumps(sensor_state_to_dict(msg))


@mcp.tool(
    title="Reset Odometry",
    description=(
        "Reset the robot's odometry to zero. Useful before executing "
        "a precise movement sequence. Calls the /reset_odometry service "
        "provided by turtlebot3_node."
    ),
)
def reset_odometry(timeout: float = 5.0) -> str:
    """
    Args:
        timeout: Seconds to wait for the service (default 5.0).
    """
    result = _node.reset_odometry(timeout)
    return json.dumps(result)


@mcp.tool(
    title="Play Sound",
    description=(
        "Play a sound on the robot's buzzer. Works on both stock TurtleBot3 "
        "(OpenCR) and Pico variant. Available sounds: 0=OFF (silence), "
        "1=ON (startup melody), 2=LOW_BATTERY (warning beeps), "
        "3=ERROR (rapid low-pitch bursts), 4=BUTTON1 (short blip), "
        "5=BUTTON2 (ascending blips). "
        "Use this to provide audible feedback to the user, signal errors, "
        "or confirm actions."
    ),
)
def play_sound(value: int = 1, timeout: float = 5.0) -> str:
    """
    Args:
        value: Sound ID (0-5). 0=OFF, 1=ON, 2=LOW_BATTERY, 3=ERROR,
               4=BUTTON1, 5=BUTTON2. Default 1.
        timeout: Seconds to wait for the service (default 5.0).
    """
    if not 0 <= value <= 5:
        return json.dumps({"error": f"Invalid sound value {value}; must be 0-5."})
    result = _node.play_sound(value, timeout)
    return json.dumps(result)


@mcp.tool(
    title="Get IMU",
    description=(
        "Return the robot's IMU (Inertial Measurement Unit) data: orientation "
        "as roll/pitch/yaw in degrees, angular velocity, and linear acceleration. "
        "On the Pico variant this is the BNO055 9-DOF NDOF fusion sensor providing "
        "absolute orientation (quaternion fused from accelerometer, gyroscope, and "
        "magnetometer). On the stock TurtleBot3 this is the MPU-9250 on OpenCR. "
        "Useful for detecting tilt, stuck conditions, verifying rotation, or "
        "reading the absolute heading of the robot."
    ),
)
def get_imu(timeout: float = 2.0) -> str:
    """
    Args:
        timeout: Seconds to wait for an IMU message (default 2.0).
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("imu", {}).get("topic", "/imu")
    msg = _node.get_latest(topic, timeout=timeout)
    if msg is None:
        return json.dumps({"error": f"No IMU data received on {topic} within {timeout}s."})
    return json.dumps(imu_to_dict(msg))


@mcp.tool(
    title="Get Magnetometer",
    description=(
        "Return the raw magnetometer reading from the IMU (sensor_msgs/MagneticField). "
        "Values are in Tesla (SI); Earth's field is typically 25–65 µT "
        "(2.5e-5 to 6.5e-5 T). Published on /magnetic_field by turtlebot3_node. "
        "NOTE: if all three components are 0.0 the IMU is publishing but the "
        "BNO055 magnetometer is not yet calibrated — move the robot in a "
        "figure-8 pattern to calibrate. A zero reading does NOT mean the sensor "
        "is absent or broken."
    ),
)
def get_magnetometer(timeout: float = 2.0) -> str:
    """
    Args:
        timeout: Seconds to wait for a MagneticField message (default 2.0).
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("mag", {}).get("topic", "/magnetic_field")
    msg = _node.get_latest(topic, timeout=timeout)
    # Fallback: try /magnetic_field if the configured topic yielded nothing
    if msg is None and topic != "/magnetic_field":
        msg = _node.get_latest("/magnetic_field", timeout=timeout)
        if msg is not None:
            topic = "/magnetic_field"
    if msg is None:
        return json.dumps({
            "error": f"No magnetometer data received on {topic} within {timeout}s.",
            "hint": "Check that turtlebot3_node is running and the IMU is connected.",
        })
    result = magnetic_field_to_dict(msg)
    # Annotate zero readings so the LLM understands calibration vs absent sensor
    mf = result.get("magnetic_field", {})
    if mf.get("x", 1) == 0.0 and mf.get("y", 1) == 0.0 and mf.get("z", 1) == 0.0:
        result["calibration_note"] = (
            "All components are 0.0 — the BNO055 magnetometer is not yet calibrated. "
            "Move the robot in a figure-8 pattern to allow the sensor to calibrate. "
            "The sensor IS present and publishing; only calibration is missing."
        )
    result["topic"] = topic
    return json.dumps(result)


@mcp.tool(
    title="Get Joint States",
    description=(
        "Return the current wheel joint positions (radians) and velocities "
        "(rad/s). The TurtleBot3 has two joints: wheel_left_joint and "
        "wheel_right_joint. Useful for verifying that wheels are actually "
        "turning, detecting stuck wheels, or computing distance travelled "
        "from encoder deltas."
    ),
)
def get_joint_states(timeout: float = 2.0) -> str:
    """
    Args:
        timeout: Seconds to wait for a JointState message (default 2.0).
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("joint_states", {}).get("topic", "/joint_states")
    msg = _node.get_latest(topic, timeout=timeout)
    if msg is None:
        return json.dumps({"error": f"No joint state data received on {topic} within {timeout}s."})
    return json.dumps(joint_state_to_dict(msg))


@mcp.tool(
    title="Set Motor Power",
    description=(
        "Enable or disable the robot's wheel motor torque. When disabled, "
        "the wheels can be moved freely by hand (useful for manual positioning). "
        "When enabled, the motors hold position and respond to velocity commands. "
        "Calls the /motor_power service provided by turtlebot3_node."
    ),
)
def set_motor_power(enable: bool = True, timeout: float = 5.0) -> str:
    """
    Args:
        enable: True to enable motor torque, False to disable (default True).
        timeout: Seconds to wait for the service (default 5.0).
    """
    result = _node.set_motor_power(enable, timeout)
    return json.dumps(result)


@mcp.tool(
    title="Navigate to Pose",
    description=(
        "Send the robot to an (x, y) goal in the map frame using Nav2. "
        "Blocks until navigation succeeds or fails. "
        "Requires an active Nav2 stack."
    ),
)
def navigate_to_pose(
    x: float,
    y: float,
    yaw: float = 0.0,
    timeout: float = 60.0,
) -> str:
    """
    Args:
        x: Goal x position in metres (map frame).
        y: Goal y position in metres (map frame).
        yaw: Goal heading in radians (default 0 = facing +X).
        timeout: Maximum seconds to wait for completion (default 60).
    """
    result = _node.navigate_to_pose(x, y, yaw, timeout)
    return json.dumps(result)


# --------------------------------------------------------------------------- #
# Spatial memory — named waypoints
# --------------------------------------------------------------------------- #


@mcp.tool(
    title="Save Waypoint",
    description=(
        "Save the robot's current odometry pose under a name so you can navigate "
        "back to it later with go_to_waypoint. "
        "Useful for marking the start position (e.g. 'home'), the location where an "
        "object was found, or any interesting area visited during exploration. "
        "Waypoints persist for the lifetime of the bridge process."
    ),
)
def save_waypoint(name: str) -> str:
    """
    Args:
        name: A short identifier for this location (e.g. 'home', 'shoe_location').
    """
    cfg_topics = _node._cfg.get("topics", {})
    odom_topic = cfg_topics.get("odom", {}).get("topic", "/odom")
    msg = _node.get_latest(odom_topic, timeout=2.0)
    if msg is None:
        return json.dumps({"error": "No odometry available; cannot save waypoint."})
    pose = odometry_to_dict(msg)
    _waypoints[name] = {"x": pose["x"], "y": pose["y"], "yaw_deg": pose.get("yaw_deg", 0.0)}
    return json.dumps({"status": "saved", "name": name, "pose": _waypoints[name]})


@mcp.tool(
    title="Go To Waypoint",
    description=(
        "Navigate to a previously saved named waypoint using Nav2. "
        "The waypoint must have been saved with save_waypoint during this session. "
        "Requires an active Nav2 stack (same as navigate_to_pose). "
        "Use list_waypoints to see all saved names."
    ),
)
def go_to_waypoint(name: str, timeout: float = 60.0) -> str:
    """
    Args:
        name: The waypoint name to navigate to (must exist in list_waypoints).
        timeout: Maximum seconds to wait for Nav2 to reach the goal (default 60).
    """
    if name not in _waypoints:
        known = list(_waypoints.keys())
        return json.dumps({
            "error": f"Unknown waypoint '{name}'.",
            "known_waypoints": known,
        })
    wp = _waypoints[name]
    yaw_rad = math.radians(wp.get("yaw_deg", 0.0))
    result = _node.navigate_to_pose(wp["x"], wp["y"], yaw_rad, timeout)
    result["waypoint"] = name
    return json.dumps(result)


@mcp.tool(
    title="List Waypoints",
    description=(
        "Return all named waypoints saved in this session with their (x, y, yaw_deg) poses. "
        "Use save_waypoint to add new waypoints and go_to_waypoint to navigate to one."
    ),
)
def list_waypoints() -> str:
    """Return all saved waypoints."""
    return json.dumps({"waypoints": _waypoints, "count": len(_waypoints)})


@mcp.tool(
    title="Delete Waypoint",
    description="Remove a named waypoint from the session.",
)
def delete_waypoint(name: str) -> str:
    """
    Args:
        name: Waypoint name to delete.
    """
    if name not in _waypoints:
        return json.dumps({"error": f"Waypoint '{name}' not found.",
                           "known": list(_waypoints.keys())})
    del _waypoints[name]
    return json.dumps({"status": "deleted", "name": name})


@mcp.tool(
    title="Update Waypoint",
    description=(
        "Manually set or overwrite a waypoint's position without navigating there. "
        "Useful for pre-loading known room coordinates into the session, or for "
        "correcting a saved waypoint after refining the map."
    ),
)
def update_waypoint(name: str, x: float, y: float, yaw_deg: float = 0.0) -> str:
    """
    Args:
        name:    Waypoint identifier.
        x:       X position in map frame (metres).
        y:       Y position in map frame (metres).
        yaw_deg: Heading in degrees (default 0).
    """
    _waypoints[name] = {"x": x, "y": y, "yaw_deg": yaw_deg}
    return json.dumps({"status": "ok", "name": name, "pose": _waypoints[name]})


# --------------------------------------------------------------------------- #
# SLAM / Map management tools                                                  #
# --------------------------------------------------------------------------- #


@mcp.tool(
    title="SLAM: Get Status",
    description=(
        "Report the current state of slam_toolbox and Nav2. "
        "Returns whether the SLAM node is running, current map dimensions, "
        "resolution, and origin. "
        "Call this first to understand the mapping/localization mode before "
        "issuing navigation goals."
    ),
)
def slam_get_status() -> str:
    """Check slam_toolbox + Nav2 status."""
    result: dict[str, Any] = {}

    # --- map metadata ---
    meta = _node.get_map_metadata(timeout=2.0)
    if meta:
        result["map"] = meta
        result["map_available"] = True
    else:
        result["map_available"] = False
        result["map"] = None

    # --- slam_toolbox node alive? ---
    try:
        r = subprocess.run(
            ["ros2", "node", "list"], capture_output=True, text=True, timeout=5
        )
        nodes = r.stdout.splitlines()
        result["slam_toolbox_running"] = any("slam_toolbox" in n for n in nodes)
        result["nav2_running"] = any("bt_navigator" in n for n in nodes)
    except Exception as e:
        result["node_list_error"] = str(e)
        result["slam_toolbox_running"] = False
        result["nav2_running"] = False

    # --- nav2 action server ---
    try:
        r = subprocess.run(
            ["ros2", "action", "list"], capture_output=True, text=True, timeout=5
        )
        result["navigate_to_pose_available"] = "/navigate_to_pose" in r.stdout
    except Exception as e:
        result["navigate_to_pose_available"] = False

    return json.dumps(result)


@mcp.tool(
    title="Get Map Info",
    description=(
        "Return the current occupancy-grid map dimensions, resolution, origin, "
        "and a breakdown of free / occupied / unknown cell counts. "
        "Useful for understanding how much of the environment has been mapped "
        "and choosing valid navigation goal coordinates."
    ),
)
def get_map_info() -> str:
    """Read /map and /map_metadata for detailed occupancy statistics."""
    meta = _node.get_map_metadata(timeout=3.0)
    if meta is None:
        return json.dumps({"error": "No map available. Is slam_toolbox or map_server running?"})

    # Try to also read the full /map OccupancyGrid for cell stats
    try:
        from nav_msgs.msg import OccupancyGrid
        msg = _node.subscribe_on_demand("/map", OccupancyGrid, timeout=3.0)
        if msg is not None:
            data = msg.data
            total = len(data)
            free     = sum(1 for v in data if v == 0)
            occupied = sum(1 for v in data if v > 50)
            unknown  = total - free - occupied
            meta["cells_free"]     = free
            meta["cells_occupied"] = occupied
            meta["cells_unknown"]  = unknown
            meta["pct_free"]       = round(100 * free / total, 1) if total else 0
            meta["pct_occupied"]   = round(100 * occupied / total, 1) if total else 0
            meta["pct_unknown"]    = round(100 * unknown / total, 1) if total else 0
    except Exception as e:
        meta["cell_stats_error"] = str(e)

    return json.dumps(meta)


@mcp.tool(
    title="SLAM: Save Map",
    description=(
        "Save the current SLAM map to disk in two formats:\n"
        "1. PGM/PNG image + YAML metadata (for Nav2 static map server)\n"
        "2. slam_toolbox pose graph .posegraph (for later localization resumption)\n"
        "Files are saved as ~/maps/<filename_stem>.{pgm,yaml,posegraph,data}.\n"
        "After calling this, the robot can be rebooted and navigation can resume "
        "via slam_load_map without re-exploring.\n"
        "\n"
        "If slam_toolbox is not running, falls back to nav2_map_server's "
        "map_saver_cli to save the PGM/YAML only (no pose graph)."
    ),
)
def slam_save_map(filename_stem: str = "map", directory: str = "") -> str:
    """
    Args:
        filename_stem: Base name for saved files (default 'map'). Do not include extension.
        directory:     Directory to save into (default ~/maps/).
    """
    save_dir = _resolve_maps_dir(directory)
    os.makedirs(save_dir, exist_ok=True)
    full_stem = os.path.join(save_dir, filename_stem)
    results: dict[str, Any] = {"directory": save_dir, "stem": filename_stem}

    # 1. Save PGM/YAML via map_saver_cli (always — more reliable than slam_toolbox's
    #    internal map_saver which can fail with "Failed to spin map subscription").
    try:
        r = subprocess.run(
            ["ros2", "run", "nav2_map_server", "map_saver_cli",
             "-f", full_stem, "--ros-args", "-p", "save_map_timeout:=10.0"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0:
            results["pgm_yaml"] = {"output": r.stdout.strip(), "returncode": 0}
        else:
            results["pgm_yaml"] = {"error": (r.stderr.strip() or r.stdout.strip() or
                                             "map_saver_cli failed"), "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        results["pgm_yaml"] = {"error": "map_saver_cli timed out after 20s"}
    except Exception as e:
        results["pgm_yaml"] = {"error": str(e)}

    # 2. Serialize pose graph (only if slam_toolbox is running)
    if _service_exists("/slam_toolbox/serialize_map"):
        req2 = "{" + f"filename: '{full_stem}'" + "}"
        r2 = _ros_service_call("/slam_toolbox/serialize_map",
                               "slam_toolbox/srv/SerializePoseGraph", req2, timeout=15)
        results["pose_graph"] = r2
    else:
        results["pose_graph"] = {"note": "slam_toolbox not running — pose graph skipped (PGM/YAML still saved)"}

    # List created files
    try:
        files = [f for f in os.listdir(save_dir) if f.startswith(filename_stem)]
        results["files_created"] = sorted(files)
    except Exception:
        pass

    pgm_ok = "error" not in results.get("pgm_yaml", {})
    pg_ok = "error" not in results.get("pose_graph", {})
    results["status"] = "ok" if (pgm_ok and pg_ok) else "partial_failure"
    return json.dumps(results)


@mcp.tool(
    title="SLAM: Serialize Map (pose graph only)",
    description=(
        "Serialize only the slam_toolbox pose graph to disk "
        "(as ~/maps/<filename_stem>.posegraph and .data). "
        "This is a lightweight checkpoint — no PGM image is generated. "
        "Use slam_save_map to also get the PGM/YAML files needed by Nav2."
    ),
)
def slam_serialize_map(filename_stem: str = "map", directory: str = "") -> str:
    """
    Args:
        filename_stem: File stem (default 'map').
        directory:     Save directory (default ~/maps/).
    """
    if not _service_exists("/slam_toolbox/serialize_map"):
        return json.dumps({
            "error": "slam_toolbox is not running — /slam_toolbox/serialize_map service unavailable. "
                     "Restart the Nav2+SLAM service or use slam_save_map (which can fall back to map_saver_cli)."
        })
    save_dir = _resolve_maps_dir(directory)
    os.makedirs(save_dir, exist_ok=True)
    full_stem = os.path.join(save_dir, filename_stem)
    req = "{" + f"filename: '{full_stem}'" + "}"
    r = _ros_service_call("/slam_toolbox/serialize_map",
                          "slam_toolbox/srv/SerializePoseGraph", req, timeout=15)
    return json.dumps({"stem": full_stem, **r})


@mcp.tool(
    title="SLAM: Load Map (resume localization)",
    description=(
        "Load a previously serialized slam_toolbox pose graph and switch to "
        "localization mode. This replaces the AMCL initial-pose requirement — "
        "slam_toolbox already knows the map and can localize from the given start pose.\n"
        "\n"
        "match_type options:\n"
        "  0 = UNSET         — start at origin\n"
        "  1 = START_AT_FIRST_NODE — start at first map node\n"
        "  2 = START_AT_GIVEN_POSE — start mapping from given pose (still builds map)\n"
        "  3 = LOCALIZE_AT_POSE   — pure localization at given pose (recommended)\n"
        "\n"
        "Use match_type=3 and pass the robot's approximate current position. "
        "Even a coarse estimate (within ~0.5 m) is sufficient for scan-matching."
    ),
)
def slam_load_map(
    filename_stem: str = "map",
    directory: str = "",
    match_type: int = 3,
    x: float = 0.0,
    y: float = 0.0,
    yaw_deg: float = 0.0,
) -> str:
    """
    Args:
        filename_stem: Base name of the saved map files (no extension).
        directory:     Directory where map files are stored (default ~/maps/).
        match_type:    0/1/2/3 — see description above (default 3 = LOCALIZE_AT_POSE).
        x:             Approximate start X in map frame (metres).
        y:             Approximate start Y in map frame (metres).
        yaw_deg:       Approximate start yaw (degrees).
    """
    if not _service_exists("/slam_toolbox/deserialize_map"):
        return json.dumps({"error": "slam_toolbox is not running — /slam_toolbox/deserialize_map service unavailable."})
    save_dir = _resolve_maps_dir(directory)
    full_stem = os.path.join(save_dir, filename_stem)
    yaw_rad = math.radians(yaw_deg)
    req = ("{" +
           f"filename: '{full_stem}', "
           f"match_type: {match_type}, "
           f"initial_pose: {{x: {x}, y: {y}, theta: {yaw_rad}}}"
           + "}")
    r = _ros_service_call("/slam_toolbox/deserialize_map",
                          "slam_toolbox/srv/DeserializePoseGraph", req, timeout=20)
    return json.dumps({"stem": full_stem, "match_type": match_type,
                       "initial_pose": {"x": x, "y": y, "yaw_deg": yaw_deg}, **r})


@mcp.tool(
    title="SLAM: Pause / Resume Mapping",
    description=(
        "Toggle whether slam_toolbox processes incoming LiDAR scans. "
        "Pause mapping during high-speed motion or when you deliberately want "
        "to stop updating the map (e.g. while carrying the robot). "
        "Each call toggles the state — call once to pause, call again to resume. "
        "Returns the new pause status."
    ),
)
def slam_pause_mapping() -> str:
    """Toggle scan processing in slam_toolbox."""
    if not _service_exists("/slam_toolbox/pause_new_measurements"):
        return json.dumps({"error": "slam_toolbox is not running — service unavailable."})
    r = _ros_service_call("/slam_toolbox/pause_new_measurements",
                          "slam_toolbox/srv/Pause", "{}", timeout=5)
    return json.dumps(r)


@mcp.tool(
    title="SLAM: Clear Map",
    description=(
        "Clear all pending changes / queued scans in slam_toolbox. "
        "This discards unprocessed scan data but does NOT clear the built map graph. "
        "Useful after large jumps or teleportation to prevent stale data from "
        "corrupting the map. For a full map reset, restart the Nav2 service."
    ),
)
def slam_clear_map() -> str:
    """Call /slam_toolbox/clear_changes."""
    if not _service_exists("/slam_toolbox/clear_changes"):
        return json.dumps({"error": "slam_toolbox is not running — service unavailable."})
    r = _ros_service_call("/slam_toolbox/clear_changes",
                          "slam_toolbox/srv/Clear", "{}", timeout=5)
    return json.dumps(r)


@mcp.tool(
    title="Nav2: Set Initial Pose (for AMCL)",
    description=(
        "Publish an /initialpose message to initialize AMCL localization. "
        "Only needed when using a STATIC pre-built map with AMCL (not slam_toolbox). "
        "With slam_toolbox (the default setup), use slam_load_map instead — "
        "it handles localization initialization automatically.\n"
        "\n"
        "covariance_xy and covariance_yaw control the uncertainty of the estimate. "
        "Tighter estimates (< 0.1) converge faster; wider estimates allow AMCL to "
        "search a larger area. Default 0.25 m² and 0.07 rad² are good starting values."
    ),
)
def nav2_set_initial_pose(
    x: float,
    y: float,
    yaw_deg: float = 0.0,
    covariance_xy: float = 0.25,
    covariance_yaw: float = 0.0685,
) -> str:
    """
    Args:
        x:              X position in map frame (metres).
        y:              Y position in map frame (metres).
        yaw_deg:        Heading in degrees.
        covariance_xy:  Position uncertainty variance in m² (default 0.25).
        covariance_yaw: Yaw uncertainty variance in rad² (default 0.07).
    """
    yaw_rad = math.radians(yaw_deg)
    result = _node.set_initial_pose(x, y, yaw_rad, covariance_xy, covariance_yaw)
    return json.dumps(result)


@mcp.tool(
    title="Nav2: Cancel Navigation",
    description=(
        "Cancel the currently active Nav2 navigation goal. "
        "The robot will stop in place. "
        "Returns immediately — the cancellation is sent asynchronously."
    ),
)
def nav2_cancel_navigation() -> str:
    """Cancel active Nav2 goal."""
    result = _node.cancel_navigation()
    return json.dumps(result)


@mcp.tool(
    title="List Map Files",
    description=(
        "List all saved map files in the maps directory (default ~/maps/). "
        "Each map consists of multiple files: .pgm (image), .yaml (metadata), "
        ".posegraph and .data (slam_toolbox serialized pose graph). "
        "Pass the filename_stem (without extension) to slam_load_map to reload."
    ),
)
def list_map_files(directory: str = "") -> str:
    """
    Args:
        directory: Directory to list (default ~/maps/).
    """
    target = _resolve_maps_dir(directory)
    if not os.path.isdir(target):
        return json.dumps({"directory": target, "files": [],
                           "note": "Directory does not exist yet. Use slam_save_map to create it."})
    files = sorted(os.listdir(target))
    # Group by stem
    stems: dict[str, list[str]] = {}
    for f in files:
        stem = f.rsplit(".", 1)[0] if "." in f else f
        stems.setdefault(stem, []).append(f)
    return json.dumps({"directory": target, "files": files, "map_stems": stems})


# --------------------------------------------------------------------------- #
# LLM scratchpad memory
# --------------------------------------------------------------------------- #


@mcp.tool(
    title="Set Memory",
    description=(
        "Store a text note in the bridge's scratchpad under a key. "
        "Use this to record observations, plans, or task state that need to persist "
        "across multiple tool calls — for example: "
        "'checked_kitchen=true', 'shoe_last_seen=near sofa', 'task_step=2'. "
        "Values are plain strings; overwrite an existing key by setting it again. "
        "Optionally attach a short description so list_memory shows what each key is for. "
        "DSL programs can also read/write these entries via get_memory()/set_memory()."
    ),
)
def set_memory(key: str, value: str, description: str = "") -> str:
    """
    Args:
        key:         Short identifier for the note (e.g. 'search_status').
        value:       Text to store (e.g. 'living room checked, no shoe found').
        description: Optional short description of what this key tracks
                     (e.g. 'whether the kitchen has been searched').
                     If omitted and the key already exists, the existing
                     description is preserved.
    """
    existing_desc = _memory.get(key, {}).get("description", "")
    _memory[key] = {
        "value": value,
        "description": description or existing_desc,
    }
    return json.dumps({"status": "ok", "key": key, "value": value,
                       "description": _memory[key]["description"]})


@mcp.tool(
    title="Get Memory",
    description=(
        "Retrieve a note previously stored with set_memory. "
        "Returns an error if the key does not exist. "
        "Use list_memory to see all stored keys."
    ),
)
def get_memory(key: str) -> str:
    """
    Args:
        key: The key to look up.
    """
    if key not in _memory:
        return json.dumps({"error": f"Key '{key}' not found.", "known_keys": list(_memory.keys())})
    entry = _memory[key]
    return json.dumps({"key": key, "value": entry["value"],
                       "description": entry.get("description", "")})


@mcp.tool(
    title="List Memory",
    description=(
        "Return all scratchpad memory entries with their keys, values, and descriptions. "
        "DSL programs share this same memory store."
    ),
)
def list_memory() -> str:
    """Return the full scratchpad."""
    items = {}
    for k, entry in _memory.items():
        items[k] = {
            "value": entry["value"],
            "description": entry.get("description", ""),
        }
    return json.dumps({"memory": items, "count": len(items)})


@mcp.tool(
    title="Clear Memory",
    description=(
        "Delete one or all scratchpad memory entries. "
        "If key is provided, only that entry is removed. "
        "If key is omitted (empty string), all entries are cleared."
    ),
)
def clear_memory(key: str = "") -> str:
    """
    Args:
        key: Key to delete. Pass empty string (default) to wipe all entries.
    """
    if key == "":
        count = len(_memory)
        _memory.clear()
        return json.dumps({"status": "cleared_all", "entries_removed": count})
    if key not in _memory:
        return json.dumps({"error": f"Key '{key}' not found.", "known_keys": list(_memory.keys())})
    del _memory[key]
    return json.dumps({"status": "deleted", "key": key})


# --------------------------------------------------------------------------- #
# A2A VLM sub-agent
# --------------------------------------------------------------------------- #


@mcp.tool(
    title="Ask Vision Agent",
    description=(
        "Capture the current camera frame and send it together with a question "
        "to a dedicated Vision-Language Model (VLM) sub-agent running on the robot. "
        "The VLM agent returns a focused visual analysis without consuming the "
        "planner's context window with raw image data. "
        "Use this instead of get_camera_image when you need a specific visual "
        "question answered — e.g. 'Is there a shoe visible? If so, describe its "
        "position relative to the frame centre.' "
        "The VLM agent is configured separately (see bridge.yaml vlm_agent section). "
        "Returns an error if the VLM agent is disabled or unreachable."
    ),
)
def ask_vision_agent(
    question: str = "Describe the scene in detail. List every distinct object you can see, "
                    "their approximate positions (left/centre/right, near/far), "
                    "and any text or labels visible.",
    timeout: float = 30.0,
    image_timeout: float = 3.0,
) -> str:
    """
    Args:
        question:      What to ask the VLM about the current camera frame.
        timeout:       Seconds to wait for the VLM agent to respond (default 30).
        image_timeout: Seconds to wait for a camera frame (default 3).
    """
    import base64

    vlm_cfg = _node._cfg.get("vlm_agent", {})
    if not vlm_cfg.get("enabled", False):
        return json.dumps({
            "error": "VLM agent is disabled. Set vlm_agent.enabled: true and "
                     "vlm_agent.url in bridge.yaml, then restart the bridge."
        })
    vlm_url = vlm_cfg.get("url", "").rstrip("/")
    if not vlm_url:
        return json.dumps({"error": "vlm_agent.url is not set in bridge.yaml."})

    cfg_topics = _node._cfg.get("topics", {})
    cam_topic  = cfg_topics.get("camera", {}).get("topic", "/camera/image_raw/compressed")
    cam_msg    = _node.get_latest(cam_topic, timeout=image_timeout)
    if cam_msg is None:
        return json.dumps({"error": f"No camera frame on {cam_topic} within {image_timeout}s."})

    img_b64   = base64.b64encode(bytes(cam_msg.data)).decode("utf-8")
    fmt       = (cam_msg.format or "jpeg").lower().split("/")[-1]
    mime_type = f"image/{fmt}"

    try:
        text = _call_a2a_with_image(vlm_url, question, img_b64, mime_type, timeout)
    except Exception as exc:
        return json.dumps({"error": f"VLM agent unreachable at {vlm_url}: {exc}"})

    if text is None:
        return json.dumps({"error": "VLM agent returned no text."})
    return json.dumps({"vlm_response": text, "question": question})


def _call_a2a_with_image(agent_url: str, question: str, img_b64: str,
                         mime_type: str, timeout: float) -> str | None:
    """POST a text+image A2A message/send payload; return the response text or None."""
    import uuid, httpx
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
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(agent_url, json=payload)
        resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        return None
    result = data.get("result", {})
    # A2A Task response
    for artifact in result.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "text":
                return part["text"]
    task_result = result.get("result")
    if task_result:
        for part in task_result.get("parts", []):
            if part.get("kind") == "text":
                return part["text"]
    # Direct message response
    for part in result.get("parts", []):
        if part.get("kind") == "text":
            return part["text"]
    return None


# --------------------------------------------------------------------------- #
# DSL runtime tools
# --------------------------------------------------------------------------- #


def _dsl_result_to_dict(r) -> dict:
    """Convert a DSLRunResult to a JSON-serialisable dict."""
    d = {
        "name": r.name,
        "status": r.status,
        "duration_s": r.duration_s,
        "log": r.log[-50:],  # last 50 lines to stay within token budget
    }
    if r.return_value is not None:
        d["return_value"] = r.return_value
    if r.error is not None:
        d["error"] = r.error
    return d


@mcp.tool(
    title="DSL: Validate",
    description=(
        "Statically validate DSL source code without executing it or moving the robot. "
        "Reports syntax errors (with line/column numbers), forbidden statements "
        "(imports, open, eval, exec, …), and warnings about possibly-undefined names. "
        "Always call this before dsl_run_inline or dsl_store_program when writing new "
        "programs — it catches errors instantly without consuming any execution time."
    ),
)
def dsl_validate(source: str) -> str:
    """
    Args:
        source: Python-DSL source code to validate.
    """
    return json.dumps(_dsl.validate_source(source))


@mcp.tool(
    title="DSL: Store Program",
    description=(
        "Store a named Python-DSL program for later execution on the robot. "
        "Programs run locally at ~20 Hz with a restricted namespace that provides:\n"
        "\n"
        "SENSORS: get_scan() → {front_min_m, left_min_m, right_min_m, rear_min_m, ...}, "
        "get_detections(timeout_s=0.5, vlm_timeout_s=8.0) → tries VLM first (if vlm_agent.enabled), "
        "falls back to YOLO/topic; returns [{label, confidence, source, bbox, distance_m, distance_source, ...}]. "
        "Set vlm_timeout_s=0 to force YOLO. source='vlm' or source='yolo' per detection. "
        "get_odom() → {x, y, yaw_rad, yaw_deg}, get_imu(), get_battery(), "
        "get_image() → BGR numpy array from camera (or None)\n"
        "\n"
        "MOTION: move(linear, angular, duration=0) — publish twist (duration=0 means single "
        "publish, >0 means run for that duration then stop), stop() — immediate halt, "
        "move_distance(distance_m) — closed-loop drive, rotate(angle_deg) — closed-loop turn, "
        "check_collision(linear_x) → {blocked, distance_m, ...}\n"
        "\n"
        "NAVIGATION: navigate_to_pose(x, y, yaw=0, timeout_s=60) — send goal to Nav2 (blocks), "
        "cancel_navigation() → {status}, "
        "save_waypoint(name) → saves current pose as named waypoint, "
        "go_to_waypoint(name, timeout_s=60) — navigate to a saved waypoint via Nav2, "
        "list_waypoints() → {name: {x, y, yaw_deg}}\n"
        "\n"
        "SLAM/MAP: set_initial_pose(x, y, yaw_deg, cov_xy=0.25, cov_yaw=0.07) — publish /initialpose "
        "for AMCL (only needed with static-map mode; slam_toolbox handles this automatically), "
        "get_map_info() → {width_m, height_m, resolution, origin_x, origin_y, pct_free, "
        "pct_occupied, pct_unknown} — use to check whether goal coordinates are inside the mapped area\n"
        "\n"
        "BEHAVIORS: find_object(label, timeout_s=20) — rotate searching for object, returns {found, detection}, "
        "approach_object(label, stop_distance=0.5, timeout_s=30) — drive toward detected object, "
        "follow_wall(side='left', target_distance=0.35, duration_s=10) — follow a wall, "
        "\n"
        "MEMORY: get_memory(key) → value or None, set_memory(key, value, description='') — "
        "shared with the LLM scratchpad tools, list_memory() → {key: value}, "
        "delete_memory(key) → bool\n"
        "\n"
        "OPENCV VISION: get_image() → BGR numpy array; "
        "cv_gray(img), cv_resize(img,w,h), cv_blur(img,k=5), "
        "cv_canny(img, low=50, high=150) → edge mask, "
        "cv_hsv_filter(img, [H,S,V]_lower, [H,S,V]_upper) → binary mask, "
        "cv_find_contours(mask, min_area=100) → [{area,cx,cy,x,y,w,h},...], "
        "cv_largest_blob(mask) → {cx,cy,area,x,y,w,h} or None, "
        "cv_hough_lines(edges, threshold=50) → [{x1,y1,x2,y2,length,angle_deg},...], "
        "cv_detect_aruco(img, dict_type='DICT_4X4_50') → [{id,corners,cx,cy},...], "
        "cv_image_stats(img) → {height,width,channels,mean_brightness}, "
        "cv_encode_jpg(img, quality=85) → JPEG bytes (use with set_result()), "
        "cv_draw_boxes(img, contours) → annotated copy. "
        "All cv_* functions import opencv internally — no explicit import needed.\n"
        "\n"
        "CONTROL: sleep(seconds) — interruptible, log(msg) — append to output, "
        "elapsed() → seconds since start, set_result(value) — set return value, "
        "print() → redirected to log(), params dict — runtime parameters, "
        "dry_run bool — True when called via dsl_run_inline/dsl_run_program with dry_run=True\n"
        "\n"
        "MATH: math module, pi, sqrt, sin, cos, atan2, radians, degrees\n"
        "\n"
        "WORKFLOW: 1) Write your program. 2) Call dsl_validate(source) \u2014 check for syntax "
        "errors and forbidden statements BEFORE running. 3) Call dsl_run_inline(source, dry_run=True) "
        "to test logic against live sensors without moving the robot. 4) Run for real via "
        "dsl_run_inline(source) or store + dsl_run_program.\n"
        "\n"
        "RULES: No imports, no file I/O, no network. Use 'while True:' loops "
        "with sleep() for continuous behaviours — the program will be stopped "
        "via dsl_stop_program or timeout. Use set_result({...}) to return data. "
        "The robot is always stopped when the program ends.\n"
        "\n"
        "EXAMPLE — follow a person:\n"
        "  while True:\n"
        "      dets = get_detections()\n"
        "      person = None\n"
        "      for d in dets:\n"
        "          if d['label'] == 'person':\n"
        "              person = d\n"
        "              break\n"
        "      if person is None:\n"
        "          move(0, 0.3)  # rotate to search\n"
        "      else:\n"
        "          cx = person['bbox']['cx']\n"
        "          offset = (cx - 320) / 320\n"
        "          dist = person.get('distance_m')\n"
        "          if dist and dist < params.get('stop_dist', 0.5):\n"
        "              stop()\n"
        "              log(f'Person {dist:.2f}m away — holding')\n"
        "          else:\n"
        "              move(0.12, -0.4 * offset)\n"
        "      sleep(0.1)\n"
    ),
)
def dsl_store_program(
    name: str,
    source: str,
    description: str = "",
    default_params: str = "{}",
) -> str:
    """
    Args:
        name:           Unique program name (e.g. 'follow_person', 'patrol_hallway').
        source:         Python-DSL source code.  Must be valid Python 3.
        description:    Human-readable description of what the program does.
        default_params: JSON object of default parameter values.
                        These can be overridden at run time.
    """
    try:
        dp = json.loads(default_params) if isinstance(default_params, str) else default_params
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"default_params is not valid JSON: {e}"})
    return json.dumps(_dsl.store_program(name, source, description, dp))


@mcp.tool(
    title="DSL: Run Program",
    description=(
        "Execute a previously stored DSL program by name. "
        "The program runs synchronously — this tool call blocks until the "
        "program finishes, is stopped, or times out. "
        "Pass runtime params as a JSON object to override defaults. "
        "Returns the program's log output, return value, and status.\n"
        "\n"
        "Set dry_run=true to test program logic without physically moving the "
        "robot — all motion, navigation, and behavior calls are replaced with "
        "no-op stubs that log what they would do. Sensors still return live data."
    ),
)
def dsl_run_program(
    name: str,
    params: str = "{}",
    timeout: float = 30.0,
    dry_run: bool = False,
) -> str:
    """
    Args:
        name:     Name of a previously stored program.
        params:   JSON object of runtime parameters (merged with defaults).
        timeout:  Maximum execution time in seconds (default 30, max 300).
        dry_run:  If true, replace all motion calls with logging stubs.
    """
    try:
        p = json.loads(params) if isinstance(params, str) else params
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"params is not valid JSON: {e}"})
    result = _dsl.run_program(name, p, timeout, dry_run=dry_run)
    return json.dumps(_dsl_result_to_dict(result))


@mcp.tool(
    title="DSL: Run Inline",
    description=(
        "Compile and execute a DSL script directly without storing it. "
        "Use this for quick one-off reactive behaviours. "
        "For re-usable programs, prefer dsl_store_program + dsl_run_program. "
        "Same namespace and rules as stored programs.\n"
        "\n"
        "Set dry_run=true to test program logic without physically moving the "
        "robot — all motion, navigation, and behavior calls are replaced with "
        "no-op stubs that log what they would do. Sensors (scan, odom, camera) "
        "still return live data so vision / logic branches are exercised normally."
    ),
)
def dsl_run_inline(
    source: str,
    params: str = "{}",
    timeout: float = 30.0,
    dry_run: bool = False,
) -> str:
    """
    Args:
        source:   Python-DSL source code to execute.
        params:   JSON object of runtime parameters.
        timeout:  Maximum execution time in seconds (default 30, max 300).
        dry_run:  If true, replace all motion calls with logging stubs.
    """
    try:
        p = json.loads(params) if isinstance(params, str) else params
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"params is not valid JSON: {e}"})
    result = _dsl.run_inline(source, p, timeout, dry_run=dry_run)
    return json.dumps(_dsl_result_to_dict(result))


@mcp.tool(
    title="DSL: Stop Program",
    description="Stop the currently running DSL program. The robot will be halted.",
)
def dsl_stop_program() -> str:
    """Stop the running DSL program."""
    return json.dumps(_dsl.stop_running())


@mcp.tool(
    title="DSL: List Programs",
    description=(
        "List all stored DSL programs in this session with their names, "
        "descriptions, default parameters, and run counts."
    ),
)
def dsl_list_programs() -> str:
    """List all stored DSL programs."""
    return json.dumps(_dsl.list_programs())


@mcp.tool(
    title="DSL: Get Source",
    description="Retrieve the source code and metadata of a stored DSL program.",
)
def dsl_get_source(name: str) -> str:
    """
    Args:
        name: Name of the program to retrieve.
    """
    return json.dumps(_dsl.get_program_source(name))


@mcp.tool(
    title="DSL: Delete Program",
    description="Delete a stored DSL program from the session.",
)
def dsl_delete_program(name: str) -> str:
    """
    Args:
        name: Name of the program to delete.
    """
    return json.dumps(_dsl.delete_program(name))


# --------------------------------------------------------------------------- #
# Server entry point (called by bridge.py)
# --------------------------------------------------------------------------- #


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 18210,
    path: str = "/ros2",
    options: dict = None,
):
    """Start the FastMCP server.  Blocks until the process exits."""
    mcp.run(transport=transport, host=host, port=port, path=path)
