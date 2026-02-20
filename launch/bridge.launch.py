"""
bridge.launch.py — ROS 2 launch file for ros2_mcp_bridge.

Usage:
    ros2 launch ros2_mcp_bridge bridge.launch.py
    ros2 launch ros2_mcp_bridge bridge.launch.py port:=18210 host:=0.0.0.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("ros2_mcp_bridge")

    # ── Declare arguments ─────────────────────────────────────────────── #
    host_arg = DeclareLaunchArgument(
        "host", default_value="0.0.0.0",
        description="Host address for the MCP HTTP server.",
    )
    port_arg = DeclareLaunchArgument(
        "port", default_value="18210",
        description="Port for the MCP HTTP server.",
    )
    config_arg = DeclareLaunchArgument(
        "config",
        default_value=PathJoinSubstitution([pkg_share, "config", "bridge.yaml"]),
        description="Path to bridge.yaml configuration file.",
    )

    # ── Set config env var so bridge.py can find it ────────────────────── #
    set_config_env = SetEnvironmentVariable(
        name="ROS2_MCP_BRIDGE_CONFIG",
        value=LaunchConfiguration("config"),
    )

    # ── Bridge node ────────────────────────────────────────────────────── #
    bridge_node = Node(
        package="ros2_mcp_bridge",
        executable="bridge",
        name="ros2_mcp_bridge",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("config")],
    )

    return LaunchDescription([
        host_arg,
        port_arg,
        config_arg,
        set_config_env,
        bridge_node,
    ])
