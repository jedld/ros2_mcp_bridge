# ros2_mcp_bridge

A ROS 2 package that exposes robot topics and services as **MCP (Model Context Protocol) tools**, allowing LLM agents to directly perceive and control a robot.

The bridge subscribes to your robot's sensor topics, caches the latest data, and serves it through a FastMCP HTTP server.  Any MCP-compatible agent (such as [onit](https://github.com/onit-labs/onit)) can then call these tools to read sensors, drive the robot, navigate to goals, and run higher-level behaviors.

```
LLM Agent (onit)
      │  HTTP / streamable-http
      ▼
ros2_mcp_bridge  (port 18210)
      │  rclpy subscriptions / publishers
      ▼
ROS 2 Robot  (camera, LiDAR, odom, detections, cmd_vel, Nav2)
```

---

## Features

- **12 ready-to-use MCP tools** — camera image, laser scan, odometry, object detections, velocity commands, Nav2 navigation, and three behavioral tools
- **YAML-configurable** — change topics, port, and speed limits without rebuilding
- **Drop-in for any ROS 2 robot** — topic names and message types are all configurable
- **Auto-discovery safe** — tools appear in the agent's tool list automatically on first connect
- **Deadman watchdog** — robot stops automatically 0.5 s after the last motion command; safe if the network drops

---

## Requirements

| Requirement | Version |
|---|---|
| ROS 2 | Humble, Iron, or Jazzy |
| Python | 3.10+ |
| [FastMCP](https://github.com/jlowin/fastmcp) | ≥ 2.0.0 |
| PyYAML | any recent |
| NumPy | any recent |

**Optional** (only needed if you use the `navigate_to_pose` tool):

- `nav2_msgs` — installed with a standard Nav2 stack

---

## Installation

### 1. Install Python dependencies

Install FastMCP into whichever Python environment your ROS 2 node will run in:

```bash
pip install "fastmcp>=2.0.0" pyyaml numpy
```

> If your system Python and your ROS 2 Python are different (common on Jetson), make sure you run `pip` for the correct interpreter:
> ```bash
> python3 -c "import rclpy; print('ok')"   # verify rclpy works
> pip3 install "fastmcp>=2.0.0" pyyaml numpy
> ```

### 2. Clone / place the package

The package should live under your workspace's `src/` directory:

```bash
# If you already have a turtlebot3_ws (or any colcon workspace):
cd ~/your_ws/src
git clone <this-repo-url> ros2_mcp_bridge
# — or copy the folder directly —
```

### 3. Build

```bash
cd ~/your_ws
colcon build --packages-select ros2_mcp_bridge
source install/setup.bash
```

---

## Quick Start

```bash
# Terminal 1 — start your robot / sensors as normal
ros2 launch turtlebot3_bringup robot.launch.py

# Terminal 2 — start the bridge
source ~/your_ws/install/setup.bash
ros2 launch ros2_mcp_bridge bridge.launch.py
```

You should see:

```
[ros2_mcp_bridge] Subscribed: /camera/image_raw/compressed
[ros2_mcp_bridge] Subscribed: /scan
[ros2_mcp_bridge] Subscribed: /odom
[ros2_mcp_bridge] Subscribed: /detections
[ros2_mcp_bridge] Node ready.
Starting MCP server on streamable-http://0.0.0.0:18210/ros2
```

Verify with:

```bash
curl http://localhost:18210/ros2
```

---

## Configuration

All runtime settings live in `config/bridge.yaml`.  You do **not** need to rebuild after changing this file — just restart the bridge.

```yaml
ros2_mcp_bridge:

  # MCP server settings
  transport: streamable-http
  host: 0.0.0.0     # bind to all interfaces; change to 127.0.0.1 to restrict to localhost
  port: 18210
  path: /ros2        # agent connects to http://<robot-ip>:18210/ros2

  # Robot motion limits (m/s and rad/s)
  robot:
    max_linear_speed: 0.22
    max_angular_speed: 2.84

  # Topic for velocity commands
  cmd_vel_topic: /cmd_vel

  # Approximate camera image width in pixels (used by approach_object steering)
  image_width: 640

  # Topics to subscribe to — add/rename/remove as needed
  topics:
    camera:
      topic: /camera/image_raw/compressed
      type: sensor_msgs/CompressedImage
    laser:
      topic: /scan
      type: sensor_msgs/LaserScan
    odom:
      topic: /odom
      type: nav_msgs/Odometry
    detections:
      topic: /detections
      type: vision_msgs/Detection2DArray
```

### Using a custom config file

Point the bridge at a different YAML file using an environment variable:

```bash
export ROS2_MCP_BRIDGE_CONFIG=/path/to/my_bridge.yaml
ros2 run ros2_mcp_bridge bridge
```

Or pass it as a launch argument:

```bash
ros2 launch ros2_mcp_bridge bridge.launch.py config:=/path/to/my_bridge.yaml
```

### Adapting to your robot

| Your robot | Change in bridge.yaml |
|---|---|
| Different camera topic | `topics.camera.topic` |
| Raw image instead of compressed | `topics.camera.type: sensor_msgs/Image` |
| Different scan topic name | `topics.laser.topic` |
| No object detector | Remove the `detections` entry |
| Higher speed limits | `robot.max_linear_speed`, `robot.max_angular_speed` |
| Different port | `port` |

Any message type matching `pkg_name/MessageName` will be resolved automatically at runtime via Python's import system — no code changes needed.

---

## Available MCP Tools

These are the tools the agent can call. All tools return JSON strings.

### Introspection

| Tool | Description |
|---|---|
| `list_ros2_topics` | List all active topics and their message types |
| `list_ros2_services` | List all active services |

### Sensors

| Tool | Key parameters | Description |
|---|---|---|
| `get_camera_image` | `timeout` | Latest camera frame as base64-encoded JPEG |
| `get_laser_scan` | `timeout` | Front / left / right / rear minimum distances + full range array |
| `get_robot_pose` | `timeout` | x, y, yaw from `/odom` |
| `get_detections` | `timeout` | YOLO bounding boxes: label, confidence, bbox (cx, cy, w, h) |

### Motion

| Tool | Key parameters | Description |
|---|---|---|
| `move_robot` | `linear_x`, `angular_z` | Publish one velocity command; auto-stops after 0.5 s |
| `stop_robot` | — | Immediately publish zero velocity |
| `navigate_to_pose` | `x`, `y`, `yaw`, `timeout` | Send Nav2 goal and block until done; requires Nav2 |

### Behaviors

| Tool | Key parameters | Description |
|---|---|---|
| `find_object` | `label`, `timeout` | Rotate up to 360° searching for an object by class label |
| `approach_object` | `label`, `stop_distance`, `timeout` | Drive toward a detected object; fuses camera + LiDAR |
| `look_around` | `n_stops`, `pause_s` | Full-circle scan; returns detections at each heading |

---

## Connecting to onit

Add one entry to `onit/configs/default.yaml` under `mcp.servers`:

```yaml
mcp:
  servers:
    - name: ROS2BridgeMCPServer
      description: "ROS 2 robot control — camera, LiDAR, odometry, YOLO detections, velocity, Nav2"
      url: http://192.168.0.153:18210/ros2   # ← your robot's IP
      enabled: true
```

Restart onit — it calls `list_tools()` at startup and registers every tool automatically.

For a full walkthrough see [docs/onit_configuration.md](docs/onit_configuration.md).

---

## Connecting to other MCP clients

The bridge uses **FastMCP streamable-http** transport, which is standard MCP over HTTP.  Any MCP client that supports HTTP transport can connect.

Example using the FastMCP Python client:

```python
from fastmcp import Client

async with Client("http://192.168.0.153:18210/ros2") as c:
    tools = await c.list_tools()
    result = await c.call_tool("get_laser_scan", {})
    print(result)
```

---

## Running without a full robot (testing)

You can start the bridge before any robot topics are publishing.  Every sensor tool accepts a `timeout` parameter; if no data arrives within that window it returns a JSON error rather than crashing.

Simulate topics with `ros2 topic pub` to test individual tools:

```bash
# Feed a dummy laser scan
ros2 topic pub /scan sensor_msgs/msg/LaserScan "{header: {frame_id: 'laser'}, \
  angle_min: -3.14, angle_max: 3.14, angle_increment: 0.01, \
  range_min: 0.1, range_max: 10.0, ranges: [1.0]}"
```

---

## Launch file arguments

```bash
ros2 launch ros2_mcp_bridge bridge.launch.py \
    host:=0.0.0.0 \
    port:=18210 \
    config:=/path/to/bridge.yaml
```

---

## Running as a systemd service

Create `/etc/systemd/system/ros2_mcp_bridge.service`:

```ini
[Unit]
Description=ROS 2 MCP Bridge
After=network.target

[Service]
User=<your-user>
Environment="HOME=/home/<your-user>"
ExecStart=/bin/bash -c \
  "source /home/<your-user>/your_ws/install/setup.bash && \
   ros2 launch ros2_mcp_bridge bridge.launch.py"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ros2_mcp_bridge
sudo journalctl -fu ros2_mcp_bridge   # view logs
```

---

## Architecture

```
bridge.py  ──────────────────────────────────────────────
│  loads bridge.yaml
│  starts rclpy MultiThreadedExecutor in a daemon thread
│  calls mcp_server.run()  (blocks — uvicorn event loop)
│
├── ros_node.py  (ROS2BridgeNode)
│    subscribes to configured topics
│    caches latest message per topic (thread-safe)
│    exposes get_latest(topic, timeout)
│    publishes Twist to cmd_vel
│    wraps Nav2 NavigateToPose action client
│
├── mcp_server.py  (FastMCP)
│    12 @mcp.tool functions
│    calls ros_node helpers synchronously
│    serves over streamable-http
│
└── behaviors.py
     find_object_behavior     — rotate + detect
     approach_object_behavior — drive + fuse camera/LiDAR
     look_around_behavior     — full-circle sweep
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: fastmcp` | `pip install "fastmcp>=2.0.0"` |
| `ModuleNotFoundError: rclpy` | Source your ROS 2 workspace: `source /opt/ros/<distro>/setup.bash` |
| No data from a sensor tool | Check the topic is publishing: `ros2 topic hz <topic>` |
| Agent shows no tools | Verify the URL is reachable: `curl http://<ip>:18210/ros2` |
| `navigate_to_pose` returns "not available" | Start the Nav2 stack: `ros2 launch nav2_bringup navigation_launch.py` |
| Robot keeps moving | The deadman fires after 0.5 s — this is expected if commands are not being sent continuously |
| Port already in use | Change `port` in `bridge.yaml` and update the agent's URL |

---

## License

Apache 2.0
