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
    compressed_image_to_dict,
    laser_scan_to_dict,
    odometry_to_dict,
    detections_to_dict,
)

# Singleton reference injected by bridge.py before the server starts
_node: Optional[ROS2BridgeNode] = None


def set_node(node: ROS2BridgeNode):
    """Called by bridge.py to inject the live rclpy node."""
    global _node
    _node = node


# --------------------------------------------------------------------------- #
# FastMCP server
# --------------------------------------------------------------------------- #

mcp = FastMCP(
    "ROS2BridgeMCPServer",
    instructions=(
        "Tools for controlling a TurtleBot3 robot via ROS 2. "
        "Provides access to camera, laser scan, odometry, object detections, "
        "velocity commands, and Nav2 navigation."
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
        "Returns a base64-encoded JPEG. "
        "Use this to visually inspect the robot's surroundings."
    ),
)
def get_camera_image(timeout: float = 3.0) -> str:
    """
    Args:
        timeout: Seconds to wait for a frame (default 3.0).
    """
    cfg_topics = _node._cfg.get("topics", {})
    topic = cfg_topics.get("camera", {}).get("topic", "/camera/image_raw/compressed")
    msg = _node.get_latest(topic, timeout=timeout)
    if msg is None:
        return json.dumps({"error": f"No image received on {topic} within {timeout}s."})
    return json.dumps(compressed_image_to_dict(msg))


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
        "Each detection includes a class label, confidence score, and bounding "
        "box in image coordinates."
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
    return json.dumps(detections_to_dict(msg))


# --------------------------------------------------------------------------- #
# Motion tools
# --------------------------------------------------------------------------- #


@mcp.tool(
    title="Move Robot",
    description=(
        "Drive the robot by publishing a velocity command. "
        "Positive linear_x moves forward; positive angular_z turns left "
        "(counter-clockwise). "
        "The command is sent once; the robot will stop automatically after "
        "~0.5 s unless you keep calling this tool."
    ),
)
def move_robot(linear_x: float = 0.0, angular_z: float = 0.0) -> str:
    """
    Args:
        linear_x: Forward velocity in m/s. Range [-0.22, 0.22] (TurtleBot3 Burger).
        angular_z: Rotational velocity in rad/s. Range [-2.84, 2.84].
    """
    _node.publish_twist(linear_x, angular_z)
    return json.dumps({"status": "ok", "linear_x": linear_x, "angular_z": angular_z})


@mcp.tool(
    title="Stop Robot",
    description="Immediately stop the robot by publishing a zero-velocity command.",
)
def stop_robot() -> str:
    """Stop all robot motion."""
    _node.stop()
    return json.dumps({"status": "stopped"})


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
# Behavioural tools (higher-level)
# --------------------------------------------------------------------------- #


@mcp.tool(
    title="Find Object",
    description=(
        "Rotate in place while watching for a specific object class. "
        "The robot completes up to one full rotation looking for the target. "
        "Returns whether the object was found and, if so, its last known position."
    ),
)
def find_object(label: str, timeout: float = 20.0) -> str:
    """
    Args:
        label: Object class label to search for (e.g. 'person', 'bottle').
        timeout: Maximum seconds to spend rotating (default 20).
    """
    from ros2_mcp_bridge.behaviors import find_object_behavior
    return json.dumps(find_object_behavior(_node, label, timeout))


@mcp.tool(
    title="Approach Object",
    description=(
        "Drive toward a detected object until within stop_distance metres "
        "or the object is no longer detected. "
        "Fuses camera detections with LiDAR to estimate approach distance."
    ),
)
def approach_object(label: str, stop_distance: float = 0.5, timeout: float = 30.0) -> str:
    """
    Args:
        label: Object class label to approach.
        stop_distance: Desired stop distance in metres (default 0.5).
        timeout: Maximum seconds to attempt approach (default 30).
    """
    from ros2_mcp_bridge.behaviors import approach_object_behavior
    return json.dumps(approach_object_behavior(_node, label, stop_distance, timeout))


@mcp.tool(
    title="Look Around",
    description=(
        "Rotate the robot in a full circle, pausing at evenly-spaced headings "
        "to capture a camera image and detection snapshot at each stop. "
        "Returns a list of observations (detections per heading)."
    ),
)
def look_around(n_stops: int = 8, pause_s: float = 1.0) -> str:
    """
    Args:
        n_stops: Number of evenly-spaced headings to sample (default 8 = 45° apart).
        pause_s: Seconds to pause at each heading before capturing (default 1.0).
    """
    from ros2_mcp_bridge.behaviors import look_around_behavior
    return json.dumps(look_around_behavior(_node, n_stops, pause_s))


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
