#!/usr/bin/env python3
"""
mcp_server.py — FastMCP server that exposes ROS 2 topics/services as tools.

Uses the same FastMCP + streamable-http pattern as onit's built-in servers so
that the agent can discover and call all tools automatically.
"""

import asyncio
import base64
import json
import logging
import math
import time
from typing import Any, Optional

from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.utilities.types import Image

logger = logging.getLogger(__name__)


class LoggingMiddleware(Middleware):
    """Logs every tool call with name, arguments, duration, and error status."""

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> Any:
        params = context.message
        tool_name = getattr(params, "name", "<unknown>")
        arguments = getattr(params, "arguments", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in (arguments or {}).items())
        logger.info("[MCP CALL] %s(%s)", tool_name, args_str)
        t0 = time.monotonic()
        try:
            result = await call_next(context)
            elapsed = time.monotonic() - t0
            is_error = getattr(result, "isError", False)
            logger.info(
                "[MCP RETURN] %s → %s in %.3fs",
                tool_name,
                "ERROR" if is_error else "OK",
                elapsed,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            logger.error("[MCP ERROR] %s raised %s in %.3fs", tool_name, exc, elapsed)
            raise

from ros2_mcp_bridge.ros_node import (
    ROS2BridgeNode,
    laser_scan_to_dict,
    odometry_to_dict,
    detections_to_dict,
    estimate_detection_distance,
    bearing_from_bbox,
    motion_stereo_depth,
    sensor_state_to_dict,
    battery_state_to_dict,
    imu_to_dict,
    joint_state_to_dict,
    magnetic_field_to_dict,
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
# FastMCP server
# --------------------------------------------------------------------------- #

mcp = FastMCP(
    "ROS2BridgeMCPServer",
    instructions=(
        "Tools for controlling a TurtleBot3 robot via ROS 2.\n"
        "\n"
        "SENSOR TOOLS: get_camera_image (visual), get_sensor_snapshot (pose+lidar+detections+battery "
        "all-in-one), get_laser_scan, get_robot_pose, get_detections, get_imu, get_battery_state.\n"
        "\n"
        "MOTION TOOLS: move_distance (precise, odometry-closed-loop), rotate_angle (precise), "
        "move_robot (timed open-loop), stop_robot.\n"
        "\n"
        "NAVIGATION: navigate_to_pose (Nav2), save_waypoint / go_to_waypoint / list_waypoints "
        "(session-level named poses).\n"
        "\n"
        "SEARCH BEHAVIORS: explore_for_object (drives + searches a room), find_object (rotates in place, "
        "returns bearing + distance), approach_object (two-phase: acquires bearing+distance via detection "
        "then switches to blind odometry approach with LiDAR obstacle avoidance — streams progress), "
        "look_around (360° detection sweep in place), "
        "panoramic_images (360° image sweep for VLM reasoning), follow_wall (perimeter exploration).\n"
        "\n"
        "VISION SUB-AGENTS (A2A): ask_vision_agent → Qwen3-VL-8B, best for scene description, "
        "object finding, OCR, counting. ask_cosmos_agent → NVIDIA Cosmos-Reason2-8B, best for "
        "navigation safety ('is it safe to move forward?'), obstacle bounding boxes, "
        "embodied action planning ('what should the robot do next?').\n"
        "\n"
        "MEMORY: set_memory / get_memory / list_memory / clear_memory — persistent scratchpad for "
        "noting observations, plans, and task state between tool calls.\n"
        "\n"
        "DSL PROGRAMS: For real-time reactive behaviours that need sensor-feedback loops "
        "faster than MCP round-trip latency allows, write a Python-DSL program using "
        "dsl_store_program, then run it with dsl_run_program. Programs execute locally on "
        "the robot at ~20 Hz with access to get_scan(), get_detections(), move(), rotate(), "
        "stop(), etc. Use dsl_run_inline for one-shot scripts. Manage with dsl_list_programs, "
        "dsl_get_source, dsl_delete_program, dsl_stop_program.\n"
        "\n"
        "OBJECT DETECTION — COCO-80 CLASS LIMITATION:\n"
        "The onboard YOLO detector recognises ONLY the 80 COCO classes listed below. "
        "Tools that take a 'label' parameter (find_object, approach_object, explore_for_object, "
        "get_detections, look_around) can ONLY match these exact class names:\n"
        "  person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, "
        "fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, "
        "elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, "
        "skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, "
        "tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, "
        "sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, "
        "potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, "
        "cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, "
        "scissors, teddy bear, hair drier, toothbrush.\n"
        "\n"
        "When the user asks for an object that is NOT one of these 80 classes, follow this strategy:\n"
        "  1. MAP to the closest COCO class if a reasonable equivalent exists. Examples:\n"
        "     • 'mug' or 'coffee cup' → 'cup'\n"
        "     • 'sofa' or 'loveseat' → 'couch'\n"
        "     • 'monitor' or 'screen' or 'television' → 'tv'\n"
        "     • 'phone' or 'smartphone' or 'iphone' → 'cell phone'\n"
        "     • 'flower pot' or 'plant' → 'potted plant'\n"
        "     • 'puppy' or 'hound' → 'dog'\n"
        "     • 'kitten' → 'cat'\n"
        "     • 'ball' or 'basketball' or 'soccer ball' → 'sports ball'\n"
        "     • 'bag' or 'purse' → 'handbag'\n"
        "     • 'notebook' or 'textbook' → 'book'\n"
        "     Tell the user which COCO class you are using as a proxy.\n"
        "  2. FALL BACK TO VLM if there is no reasonable COCO equivalent. Use "
        "ask_vision_agent or panoramic_images to visually search for the object instead. "
        "For example, 'shoe', 'pen', 'glasses', 'wallet', 'keys' have no COCO match — "
        "use the VLM to scan the camera image and locate them.\n"
        "  3. COMBINE both strategies when useful: use the YOLO detector to quickly narrow "
        "the search area (e.g. find a 'dining table' first), then use ask_vision_agent to "
        "look for the specific non-COCO object on/near it (e.g. 'keys on the table').\n"
        "\n"
        "STRATEGY: prefer explore_for_object over repeated find_object calls. Use get_sensor_snapshot "
        "instead of calling get_camera_image + get_laser_scan + get_robot_pose separately. "
        "Save interesting locations with save_waypoint so you can return to them. "
        "Use set_memory to record what rooms/areas have been checked. "
        "For complex navigation that reacts to real-time sensor data (obstacle weaving, "
        "tracking a moving object, patrol routes), write a DSL program instead of chaining "
        "many individual move/read tool calls."
    ),
    middleware=[LoggingMiddleware()],
)


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
    title="Get Object Detections",
    description=(
        "Return the latest object detections from the vision pipeline. "
        "Each detection includes a class label, confidence score, bounding "
        "box in image coordinates, and a fused distance estimate. "
        "Distance is computed by aligning the detection bbox to specific "
        "LiDAR rays and cross-validating with a pinhole-model depth "
        "estimate from the bbox height. If the object is below the 2D "
        "LiDAR scan plane (e.g. on the floor), the bbox estimate is used "
        "as fallback and distance_reliable will be false. "
        "NOTE: the detector only recognises the 80 COCO classes — see the "
        "server instructions for the full list. If you need to find an "
        "object outside those classes, use ask_vision_agent instead."
    ),
)
def get_detections(timeout: float = 2.0) -> str:
    """
    Args:
        timeout: Seconds to wait for a detection message (default 2.0).
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("detections", {}).get("topic", "/detections")
    msg = _node.get_latest(topic, timeout=timeout)
    if msg is None:
        return json.dumps({"error": f"No detections received on {topic} within {timeout}s."})

    result = detections_to_dict(msg)

    # Enrich each detection with fused distance estimate
    laser_topic = cfg_topics.get("laser", {}).get("topic", "/scan")
    scan_msg = _node.get_latest(laser_topic, timeout=0.5)
    image_width = _node._cfg.get("image_width", 640)
    image_height = int(image_width * 480 / 640)
    hfov_deg = float(_node._cfg.get("camera_hfov_deg", 62.0))

    for det in result["detections"]:
        dist = estimate_detection_distance(
            scan_msg, det, image_width, image_height, hfov_deg,
        )
        det.update(dist)

    return json.dumps(result)


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
    description=(
        "Immediately stop the robot by publishing a zero-velocity command. "
        "This also cancels any running behaviour (explore_for_object, "
        "approach_object, find_object, follow_wall, etc.) — the behaviour "
        "will terminate at the next checkpoint and return a 'cancelled' status. "
        "Call this whenever you need the robot to halt immediately."
    ),
)
def stop_robot() -> str:
    """Stop all robot motion and cancel running behaviours."""
    _node.stop()
    return json.dumps({"status": "stopped", "message": "Robot stopped and running behaviours cancelled."})


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
# Behavioural tools (higher-level)
# --------------------------------------------------------------------------- #


@mcp.tool(
    title="Find Object",
    description=(
        "Rotate in place (one full circle) while watching for a specific object class. "
        "IMPORTANT: this tool does NOT move the robot to a new location — it only spins "
        "on the spot. If the object is not found, you MUST physically drive the robot to "
        "a new area before calling find_object again, or use explore_for_object instead "
        "(which automatically moves the robot to new positions). "
        "When the object IS found, returns its detection plus bearing_deg (angular offset "
        "from camera centre) and estimated_distance_m (fused LiDAR + bbox estimate) so "
        "you can then call approach_object to drive to it. "
        "Collision avoidance (LiDAR) is checked at the start and reported in the "
        "response if any sectors are obstructed. "
        "IMPORTANT: the 'label' must be one of the 80 COCO class names (see server "
        "instructions). Map synonyms to the closest COCO class (e.g. 'mug' → 'cup'). "
        "If the object has no COCO equivalent, use ask_vision_agent or panoramic_images "
        "with a VLM query instead."
    ),
)
def find_object(
    label: str,
    timeout: float = 20.0,
    collision_avoidance: bool = True,
) -> str:
    """
    Args:
        label: Object class label to search for (e.g. 'person', 'bottle').
        timeout: Maximum seconds to spend rotating (default 20).
        collision_avoidance: If True (default), report nearby obstacles detected
                             by LiDAR in the response.
    """
    from ros2_mcp_bridge.behaviors import find_object_behavior
    ca_global = _node._cfg.get("collision_avoidance", {}).get("enabled", True)
    effective_ca = collision_avoidance and ca_global
    return json.dumps(find_object_behavior(_node, label, timeout, effective_ca))


@mcp.tool(
    title="Approach Object",
    description=(
        "Two-phase approach to a detected object. "
        "\n\n"
        "**Phase 1 — Acquisition** (uses YOLO detection): locks on the target's "
        "bearing from the bounding-box centre, then estimates distance using a "
        "priority cascade: (1) LiDAR rays aligned with the bearing, (2) pinhole-model "
        "bbox height estimate, (3) motion-stereo depth fallback (nudges the robot "
        "~8 cm forward, matches ORB features, triangulates from parallax). "
        "\n\n"
        "**Phase 2 — Blind approach** (no more object detection): rotates to face "
        "the target, then drives forward using odometry closed-loop with LiDAR "
        "obstacle avoidance, in small increments. Does NOT rely on YOLO during the "
        "drive — this avoids motion-blur detection failures. "
        "\n\n"
        "Returns streaming progress updates: acquisition status, rotation, and "
        "per-step driving progress so you can monitor the approach in real time. "
        "The final update has phase='complete'. "
        "IMPORTANT: the 'label' must be one of the 80 COCO class names (see server "
        "instructions). The object must already be visible in the current detections."
    ),
)
def approach_object(
    label: str,
    stop_distance: float = 0.5,
    timeout: float = 30.0,
    collision_avoidance: bool = True,
) -> str:
    """
    Args:
        label: Object class label to approach (must be currently visible).
        stop_distance: Desired stop distance from object in metres (default 0.5).
        timeout: Maximum seconds for the full approach (default 30).
        collision_avoidance: If True (default), apply LiDAR safety stops during the
                             blind approach phase if an obstacle enters the minimum
                             front distance.
    """
    from ros2_mcp_bridge.behaviors import approach_object_behavior
    ca_global = _node._cfg.get("collision_avoidance", {}).get("enabled", True)
    effective_ca = collision_avoidance and ca_global
    def generator():
        for progress in approach_object_behavior(_node, label, stop_distance, timeout, effective_ca):
            yield json.dumps(progress)
    return generator()


@mcp.tool(
    title="Look Around",
    description=(
        "Rotate the robot in a full circle, pausing at evenly-spaced headings "
        "to capture a camera image and detection snapshot at each stop. "
        "Returns a list of observations (detections per heading). "
        "NOTE: like find_object, this tool only rotates in place; it does not "
        "move the robot to a new location. Use explore_for_object if you need "
        "the robot to actively drive around and search a larger area. "
        "When collision_avoidance is enabled, each observation includes the "
        "current LiDAR sector distances so the LLM can reason about nearby obstacles."
    ),
)
def look_around(
    n_stops: int = 8,
    pause_s: float = 1.0,
    collision_avoidance: bool = True,
) -> str:
    """
    Args:
        n_stops: Number of evenly-spaced headings to sample (default 8 = 45° apart).
        pause_s: Seconds to pause at each heading before capturing (default 1.0).
        collision_avoidance: If True (default), include LiDAR obstacle distances
                             in each observation snapshot.
    """
    from ros2_mcp_bridge.behaviors import look_around_behavior
    ca_global = _node._cfg.get("collision_avoidance", {}).get("enabled", True)
    effective_ca = collision_avoidance and ca_global
    return json.dumps(look_around_behavior(_node, n_stops, pause_s, effective_ca))


@mcp.tool(
    title="Explore For Object",
    description=(
        "Actively explore the environment to find a specific object class. "
        "Unlike find_object or look_around (which only rotate in place), this tool "
        "physically moves the robot to new positions while searching. "
        "At each position it performs a full 360° detection sweep; if the target is "
        "not found it reads LiDAR to choose the most-open direction, then drives "
        "forward step_distance metres and repeats. "
        "Use this when find_object has already failed at the current location, or "
        "when the task requires searching a room or larger area. "
        "Returns found status, the detection details if found, steps taken, and "
        "an exploration path log. "
        "When collision_avoidance is enabled (default), the robot will not drive "
        "into obstacles and will attempt to navigate around them. "
        "IMPORTANT: the 'label' must be one of the 80 COCO class names (see server "
        "instructions). Map synonyms to the closest COCO class (e.g. 'sofa' → 'couch'). "
        "If the object has no COCO equivalent, combine exploration with "
        "ask_vision_agent to visually locate it."
    ),
)
def explore_for_object(
    label: str,
    step_distance: float = 0.5,
    max_steps: int = 10,
    timeout: float = 180.0,
    collision_avoidance: bool = True,
) -> str:
    """
    Args:
        label: Object class label to search for (e.g. 'shoe', 'bottle', 'person').
        step_distance: How far to drive between scan positions in metres (default 0.5).
                       Increase for large rooms; decrease for small cluttered spaces.
        max_steps: Maximum number of drive-forward steps before giving up (default 10).
        timeout: Maximum total seconds for the entire exploration (default 180).
        collision_avoidance: If True (default), use LiDAR to avoid obstacles while
                             navigating between scan positions.
    """
    from ros2_mcp_bridge.behaviors import explore_for_object_behavior
    ca_global = _node._cfg.get("collision_avoidance", {}).get("enabled", True)
    effective_ca = collision_avoidance and ca_global
    def generator():
        for progress in explore_for_object_behavior(_node, label, step_distance, max_steps, timeout, effective_ca):
            yield json.dumps(progress)
    return generator()


@mcp.tool(
    title="Panoramic Images",
    description=(
        "Rotate the robot in a full circle, capture a camera image at each of "
        "n_stops evenly-spaced headings, and return all frames together. "
        "This lets a vision-language model reason about the entire 360° scene in "
        "one tool call rather than issuing many separate get_camera_image calls. "
        "Useful for: initial room survey, verifying 'shoe not visible anywhere', "
        "choosing which direction to explore next. "
        "Each image is returned as an MCP ImageContent so the VLM can inspect it directly. "
        "A JSON summary of detections at each heading is appended as the last content item."
    ),
)
def panoramic_images(
    n_stops: int = 6,
    pause_s: float = 0.8,
) -> list:
    """
    Args:
        n_stops: Number of evenly-spaced headings to photograph (default 6 = every 60°).
        pause_s: Seconds to pause at each stop before capturing (default 0.8).
    """
    import math as _math

    cfg_topics = _node._cfg.get("topics", {})
    cam_topic = cfg_topics.get("camera", {}).get("topic", "/camera/image_raw/compressed")
    det_topic = cfg_topics.get("detections", {}).get("topic", "/detections")

    step_rad   = (2 * _math.pi) / n_stops
    ang_speed  = 0.6  # rad/s
    step_time  = step_rad / ang_speed

    content = []
    summary = []

    for i in range(n_stops):
        heading_deg = round(_math.degrees(i * step_rad) % 360, 1)

        # Rotate one step
        deadline = time.time() + step_time
        while time.time() < deadline:
            _node.publish_twist(0.0, ang_speed)
            time.sleep(0.05)
        _node.stop()
        time.sleep(pause_s)

        # Capture image → MCP ImageContent
        cam_msg = _node.get_latest(cam_topic, timeout=1.5)
        if cam_msg is not None:
            fmt = (cam_msg.format or "jpeg").lower().split("/")[-1]
            content.append(Image(data=bytes(cam_msg.data), format=fmt))
        else:
            content.append(json.dumps({"heading_deg": heading_deg, "error": "no_image"}))

        # Capture detections for summary
        det_msg = _node.get_latest(det_topic, timeout=0.5)
        dets = detections_to_dict(det_msg)["detections"] if det_msg is not None else []
        summary.append({"heading_deg": heading_deg, "detections": dets})

    # Append JSON summary as the last content item
    content.append(json.dumps({
        "panorama_summary": summary,
        "total_unique_labels": list({d["label"] for obs in summary for d in obs["detections"]}),
    }))
    return content


@mcp.tool(
    title="Follow Wall",
    description=(
        "Drive along a wall, maintaining a set distance from it, for a specified duration. "
        "This is a classic exploration primitive for mapping room perimeters: "
        "the robot hugs one wall and moves forward until it hits a corner or obstacle. "
        "Use side='left' to follow the left wall, side='right' for the right wall. "
        "Stops early if a front obstacle is detected (collision avoidance). "
        "Combine with rotate_angle (turn at corners) to circuit an entire room."
    ),
)
def follow_wall(
    side: str = "left",
    target_distance: float = 0.35,
    duration: float = 10.0,
    speed: float = 0.0,
) -> str:
    """
    Args:
        side: Which wall to track — 'left' or 'right' (default 'left').
        target_distance: Desired distance from the wall in metres (default 0.35).
        duration: How long to follow for in seconds (default 10, max 120).
        speed: Forward speed in m/s (default 0 = 60% of max_linear_speed).
    """
    if side not in ("left", "right"):
        return json.dumps({"error": "side must be 'left' or 'right'."})
    duration = max(1.0, min(float(duration), 120.0))
    rob_cfg  = _node._cfg.get("robot", {})
    if speed == 0.0:
        speed = float(rob_cfg.get("max_linear_speed", 0.22)) * 0.6
    from ros2_mcp_bridge.behaviors import follow_wall_behavior
    result = follow_wall_behavior(_node, side, target_distance, duration, speed)
    return json.dumps(result)


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


@mcp.tool(
    title="Ask Cosmos Reasoning Agent",
    description=(
        "Capture the current camera frame and send it to the NVIDIA "
        "Cosmos-Reason2-8B embodied reasoning agent. "
        "Unlike the general VLM agent, Cosmos was specifically fine-tuned for "
        "robotics and physical AI: it reasons about space, physics, and time, "
        "provides bounding-box object coordinates, and can plan robot actions. "
        "Use this for navigation decisions: "
        "'Is it safe to move forward?', "
        "'What obstacles are ahead and where exactly?', "
        "'What action should the robot take next to reach the goal?', "
        "'Describe the geometry of the path ahead.' "
        "Returns a chain-of-thought reasoning trace followed by a concise answer. "
        "Requires the Cosmos A2A agent to be running (see bridge.yaml cosmos_agent section)."
    ),
)
def ask_cosmos_agent(
    question: str = "Assess the scene from a robot navigation perspective. "
                    "Is it safe to drive straight ahead? "
                    "List any obstacles with their clock-position and estimated distance. "
                    "Recommend: move forward / turn left / turn right / stop.",
    timeout: float = 45.0,
    image_timeout: float = 3.0,
) -> str:
    """
    Args:
        question:      What to ask Cosmos about the current camera frame.
                       Best suited for navigation safety, obstacle detection,
                       spatial bounding boxes, and action planning.
        timeout:       Seconds to wait for Cosmos to respond (default 45 —
                       Cosmos uses chain-of-thought so takes a little longer).
        image_timeout: Seconds to wait for a camera frame (default 3).
    """
    import base64

    cosmos_cfg = _node._cfg.get("cosmos_agent", {})
    if not cosmos_cfg.get("enabled", False):
        return json.dumps({
            "error": "Cosmos agent is disabled. Set cosmos_agent.enabled: true and "
                     "cosmos_agent.url in bridge.yaml, then restart the bridge. "
                     "Start the agent on the robot with: "
                     "onit --a2a --config ~/turtlebot3_ws/onit/configs/cosmos_agent.yaml"
        })
    cosmos_url = cosmos_cfg.get("url", "").rstrip("/")
    if not cosmos_url:
        return json.dumps({"error": "cosmos_agent.url is not set in bridge.yaml."})

    cfg_topics = _node._cfg.get("topics", {})
    cam_topic  = cfg_topics.get("camera", {}).get("topic", "/camera/image_raw/compressed")
    cam_msg    = _node.get_latest(cam_topic, timeout=image_timeout)
    if cam_msg is None:
        return json.dumps({"error": f"No camera frame on {cam_topic} within {image_timeout}s."})

    raw_bytes = bytes(cam_msg.data)
    img_b64   = base64.b64encode(raw_bytes).decode("utf-8")
    fmt       = (cam_msg.format or "jpeg").lower().split("/")[-1]
    mime_type = f"image/{fmt}"

    try:
        text = _call_a2a_with_image(cosmos_url, question, img_b64, mime_type, timeout)
    except Exception as exc:
        return json.dumps({"error": f"Cosmos agent unreachable at {cosmos_url}: {exc}"})

    if text is None:
        return json.dumps({"error": "Cosmos agent returned no text."})
    return json.dumps({"cosmos_response": text, "question": question})


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
    title="DSL: Store Program",
    description=(
        "Store a named Python-DSL program for later execution on the robot. "
        "Programs run locally at ~20 Hz with a restricted namespace that provides:\n"
        "\n"
        "SENSORS: get_scan() → {front_min_m, left_min_m, right_min_m, rear_min_m, ...}, "
        "get_detections() → [{label, confidence, bbox, distance_m, distance_source, ...}], "
        "get_odom() → {x, y, yaw_rad, yaw_deg}, get_imu(), get_battery()\n"
        "\n"
        "MOTION: move(linear, angular, duration=0) — publish twist (duration=0 means single "
        "publish, >0 means run for that duration then stop), stop() — immediate halt, "
        "move_distance(distance_m) — closed-loop drive, rotate(angle_deg) — closed-loop turn, "
        "check_collision(linear_x) → {blocked, distance_m, ...}\n"
        "\n"
        "NAVIGATION: navigate_to_pose(x, y, yaw=0, timeout_s=60) — send goal to Nav2 (blocks), "
        "save_waypoint(name) → saves current pose as named waypoint, "
        "go_to_waypoint(name, timeout_s=60) — navigate to a saved waypoint via Nav2, "
        "list_waypoints() → {name: {x, y, yaw_deg}}\n"
        "\n"
        "BEHAVIORS: find_object(label, timeout_s=20) — rotate searching for object, returns {found, detection}, "
        "approach_object(label, stop_distance=0.5, timeout_s=30) — drive toward detected object, "
        "follow_wall(side='left', target_distance=0.35, duration_s=10) — follow a wall, "
        "explore_for_object(label, step_distance=0.5, max_steps=10, timeout_s=180) — explore to find object\n"
        "\n"
        "MEMORY: get_memory(key) → value or None, set_memory(key, value, description='') — "
        "shared with the LLM scratchpad tools, list_memory() → {key: value}, "
        "delete_memory(key) → bool\n"
        "\n"
        "CONTROL: sleep(seconds) — interruptible, log(msg) — append to output, "
        "elapsed() → seconds since start, set_result(value) — set return value, "
        "print() → redirected to log(), params dict — runtime parameters\n"
        "\n"
        "MATH: math module, pi, sqrt, sin, cos, atan2, radians, degrees\n"
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
        "Returns the program's log output, return value, and status."
    ),
)
def dsl_run_program(
    name: str,
    params: str = "{}",
    timeout: float = 30.0,
) -> str:
    """
    Args:
        name:    Name of a previously stored program.
        params:  JSON object of runtime parameters (merged with defaults).
        timeout: Maximum execution time in seconds (default 30, max 300).
    """
    try:
        p = json.loads(params) if isinstance(params, str) else params
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"params is not valid JSON: {e}"})
    result = _dsl.run_program(name, p, timeout)
    return json.dumps(_dsl_result_to_dict(result))


@mcp.tool(
    title="DSL: Run Inline",
    description=(
        "Compile and execute a DSL script directly without storing it. "
        "Use this for quick one-off reactive behaviours. "
        "For re-usable programs, prefer dsl_store_program + dsl_run_program. "
        "Same namespace and rules as stored programs."
    ),
)
def dsl_run_inline(
    source: str,
    params: str = "{}",
    timeout: float = 30.0,
) -> str:
    """
    Args:
        source:  Python-DSL source code to execute.
        params:  JSON object of runtime parameters.
        timeout: Maximum execution time in seconds (default 30, max 300).
    """
    try:
        p = json.loads(params) if isinstance(params, str) else params
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"params is not valid JSON: {e}"})
    result = _dsl.run_inline(source, p, timeout)
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
