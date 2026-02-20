#!/usr/bin/env python3
"""
bridge.py — Main entry point for ros2_mcp_bridge.

Starts the rclpy node in a daemon thread and runs the FastMCP server
(uvicorn) in the main thread.  Mirrors the cam_web.py pattern used by
the camera streamer in this workspace.

Usage (after colcon build + source install/setup.bash):
    ros2 run ros2_mcp_bridge bridge

Or via the launch file:
    ros2 launch ros2_mcp_bridge bridge.launch.py
"""

import logging
import os
import sys
import threading
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


def _load_config(path: str) -> dict:
    """Load a YAML config file; return an empty dict on failure."""
    try:
        import yaml  # PyYAML
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        # Unwrap ROS 2 parameter node namespace if present
        for v in data.values():
            if isinstance(v, dict) and "ros__parameters" in v:
                return v["ros__parameters"]
        return data
    except FileNotFoundError:
        logger.warning(f"Config file not found: {path}  — using defaults.")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config {path}: {e}")
        return {}


def _find_config() -> str:
    """
    Locate bridge.yaml in standard locations:
    1. ROS_CONFIG env var
    2. <package_share>/config/bridge.yaml via ament resource lookup
    3. Relative to this file (development fallback)
    """
    env_path = os.environ.get("ROS2_MCP_BRIDGE_CONFIG")
    if env_path and Path(env_path).exists():
        return env_path

    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("ros2_mcp_bridge")
        candidate = Path(share) / "config" / "bridge.yaml"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass

    # Fallback: relative to this source file
    fallback = Path(__file__).parent.parent / "config" / "bridge.yaml"
    return str(fallback)


# --------------------------------------------------------------------------- #
# rclpy thread
# --------------------------------------------------------------------------- #


def _spin_node(node, executor: MultiThreadedExecutor):
    """Target function for the rclpy daemon thread."""
    try:
        executor.add_node(node)
        executor.spin()
    except Exception as e:
        logger.error(f"rclpy executor error: {e}")
    finally:
        executor.remove_node(node)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # -- Config -------------------------------------------------------------- #
    cfg_path = _find_config()
    logger.info(f"Loading config from: {cfg_path}")
    cfg = _load_config(cfg_path)

    bridge_cfg = cfg.get("ros2_mcp_bridge", cfg)  # tolerate flat or nested YAML

    # -- rclpy --------------------------------------------------------------- #
    rclpy.init(args=args)
    from ros2_mcp_bridge.ros_node import ROS2BridgeNode
    node = ROS2BridgeNode(bridge_cfg)

    executor = MultiThreadedExecutor()
    ros_thread = threading.Thread(
        target=_spin_node,
        args=(node, executor),
        daemon=True,
        name="rclpy-executor",
    )
    ros_thread.start()
    logger.info("rclpy executor started in background thread.")

    # -- FastMCP server ------------------------------------------------------ #
    from ros2_mcp_bridge import mcp_server
    mcp_server.set_node(node)

    transport = bridge_cfg.get("transport", "streamable-http")
    host      = bridge_cfg.get("host",      "0.0.0.0")
    port      = int(bridge_cfg.get("port",  18210))
    path      = bridge_cfg.get("path",      "/ros2")

    logger.info(f"Starting MCP server on {transport}://{host}:{port}{path}")

    try:
        mcp_server.run(transport=transport, host=host, port=port, path=path)
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        node.stop()
        rclpy.shutdown()
        ros_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
