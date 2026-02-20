# Integrating ros2_mcp_bridge with onit

This guide explains how to connect the `ros2_mcp_bridge` MCP server to the
**onit** AI agent framework so that your LLM agent can control the TurtleBot3.

---

## Prerequisites

| Component | Where |
|---|---|
| ROS 2 Humble (or later) | Robot / Jetson Orin Nano |
| `ros2_mcp_bridge` built | `turtlebot3_ws` |
| `fastmcp >= 2.0.0` | Robot Python env |
| onit | Developer machine (or same machine) |
| Network connectivity | Robot IP reachable from onit host |

---

## 1. Install fastmcp on the robot

`ros2_mcp_bridge` uses **FastMCP** for its transport layer.  Install it into
the same Python environment that runs ROS 2:

```bash
pip install "fastmcp>=2.0.0"
```

> **Tip:** If you use a virtual environment, activate it before running
> `colcon build` so the `fastmcp` import is available at runtime.

---

## 2. Build the package

```bash
cd ~/turtlebot3_ws
colcon build --packages-select ros2_mcp_bridge
source install/setup.bash
```

---

## 3. Start the bridge

### Directly

```bash
source ~/turtlebot3_ws/install/setup.bash
ros2 run ros2_mcp_bridge bridge
```

### Via the launch file (recommended)

```bash
ros2 launch ros2_mcp_bridge bridge.launch.py
```

Optional arguments:

```bash
ros2 launch ros2_mcp_bridge bridge.launch.py \
    port:=18210 \
    host:=0.0.0.0 \
    config:=/path/to/custom_bridge.yaml
```

You should see log output like:

```
[ros2_mcp_bridge] Subscribed: /camera/image_raw/compressed
[ros2_mcp_bridge] Subscribed: /scan
[ros2_mcp_bridge] Subscribed: /odom
[ros2_mcp_bridge] Subscribed: /detections
[ros2_mcp_bridge] Node ready.
Starting MCP server on streamable-http://0.0.0.0:18210/ros2
```

Verify it is running:

```bash
curl http://<robot-ip>:18210/ros2
# Should return HTTP 200 or a FastMCP methods response
```

---

## 4. Configure onit

Edit **`onit/configs/default.yaml`** (or your active config file) and add an
entry under `mcp.servers`:

```yaml
mcp:
  servers:
    # ... your existing servers ...

    - name: ROS2BridgeMCPServer
      description: >
        ROS 2 robot control — camera, LiDAR, odometry, YOLO detections,
        velocity commands, and Nav2 navigation for TurtleBot3.
      url: http://192.168.0.153:18210/ros2   # ← replace with your robot's IP
      enabled: true
```

> Replace `192.168.0.153` with the actual IP address of your robot.
> Find it with `hostname -I` on the Jetson.

---

## 5. Restart onit

```bash
# From your onit directory
python -m onit   # or however you start onit
```

onit will call `fastmcp.Client.list_tools()` against the URL and auto-register
all tools.  You should see them appear in the agent's tool list.

---

## 6. Available MCP tools

Once connected, the agent has access to these tools:

| Tool | What it does |
|---|---|
| `list_ros2_topics` | List all active ROS 2 topics and types |
| `list_ros2_services` | List all active ROS 2 services |
| `get_camera_image` | Return a base64-encoded JPEG from the camera |
| `get_laser_scan` | Front/left/rear/right obstacle distances + full scan |
| `get_robot_pose` | x, y, yaw from `/odom` |
| `get_detections` | Latest YOLO bounding boxes from `/detections` |
| `move_robot` | Publish a velocity command (linear_x, angular_z) |
| `stop_robot` | Immediately stop all motion |
| `navigate_to_pose` | Send a Nav2 goal and wait for completion |
| `find_object` | Rotate and search for a labelled object |
| `approach_object` | Drive toward a detected object until a stop distance |
| `look_around` | Full-circle scan with detections at each heading |

---

## 7. Example agent prompts

```
Look around and tell me what objects you can see.
```

```
Find a person.  If you find one, approach to 1.5 metres and stop.
```

```
Go to map coordinates x=1.5, y=0.8 and report what the camera sees.
```

---

## 8. Configuring topics

Edit `config/bridge.yaml` to match your robot's actual topic names:

```yaml
ros2_mcp_bridge:
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

After changing the YAML, restart the bridge (no rebuild needed).

---

## 9. Autostart with systemd (optional)

Create `/etc/systemd/system/ros2_mcp_bridge.service`:

```ini
[Unit]
Description=ROS 2 MCP Bridge
After=network.target

[Service]
User=<your-user>
Environment="HOME=/home/<your-user>"
ExecStart=/bin/bash -c \
  "source /home/<your-user>/turtlebot3_ws/install/setup.bash && \
   ros2 launch ros2_mcp_bridge bridge.launch.py"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ros2_mcp_bridge
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: fastmcp` | `pip install "fastmcp>=2.0.0"` |
| `No image received on /camera/…` | Check `ros2 topic hz /camera/image_raw/compressed` |
| onit shows no tools from this server | Confirm the URL is reachable: `curl http://<ip>:18210/ros2` |
| Nav2 `navigate_to_pose` returns "not available" | Ensure Nav2 stack is running: `ros2 launch turtlebot3_navigation2 navigation2.launch.py` |
| Robot doesn't stop after `move_robot` | The deadman watchdog fires after 0.5 s — expected behaviour |
