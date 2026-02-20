# ros2_mcp_bridge — Agent Guide

This document is written for AI coding agents (Copilot, Cursor, Claude, GPT, etc.).
It describes what this repo does, how the code is structured, and how to make changes safely.

---

## What this repo is

`ros2_mcp_bridge` is a **ROS 2 Python package** that acts as a bridge between a physical (or simulated) robot and LLM agents.

It exposes robot sensor data and actuator commands as **MCP (Model Context Protocol) tools** served over HTTP.  An LLM agent connects to the bridge's HTTP endpoint and can then:

- Read camera images, LiDAR scans, odometry, and YOLO object detections
- Publish velocity commands (`cmd_vel`)
- Send navigation goals to Nav2
- Run higher-level behaviors (find object, approach object, look around)

The intended agent framework is **onit**, but any MCP-over-HTTP client works.

---

## Tech stack

| Layer | Technology |
|---|---|
| ROS 2 | rclpy (Python), ament_python build type |
| MCP server | [FastMCP](https://github.com/jlowin/fastmcp) ≥ 2.0.0, `streamable-http` transport |
| Concurrency model | rclpy `MultiThreadedExecutor` in a **daemon thread**; FastMCP/uvicorn in **main thread** |
| Config | PyYAML (`config/bridge.yaml`) |
| Math / arrays | NumPy |
| Target hardware | Jetson Orin Nano + TurtleBot3 Burger, but generic enough for any ROS 2 robot |

---

## File map

```
ros2_mcp_bridge/          ← Python package (importable)
├── __init__.py
├── bridge.py             ← Entry point.  Loads config, starts rclpy thread, runs mcp_server.run()
├── ros_node.py           ← rclpy Node.  All ROS 2 interaction lives here.
├── mcp_server.py         ← FastMCP server.  All @mcp.tool definitions live here.
└── behaviors.py          ← Blocking behavioral routines called by MCP tools.

config/
└── bridge.yaml           ← Runtime config: port, topics, speed limits.  Edit this, not Python.

launch/
└── bridge.launch.py      ← ros2 launch entry point.

docs/
└── onit_configuration.md ← Walkthrough for connecting onit to this bridge.

package.xml               ← ROS 2 package manifest (dependencies declared here).
setup.py                  ← ament_python build; entry_point: bridge = ros2_mcp_bridge.bridge:main
setup.cfg                 ← Standard ament_python script_dir setting.
README.md                 ← Human-facing docs.
AGENTS.md                 ← This file.
```

---

## Module responsibilities — where to make changes

### `ros_node.py` — ROS 2 interface

Contains `ROS2BridgeNode(rclpy.node.Node)`.

**Edit this file when:**
- Adding a new topic subscription
- Adding a new publisher (e.g. a different message type)
- Changing how messages are serialised to plain dicts (`laser_scan_to_dict`, `odometry_to_dict`, etc.)
- Adjusting the deadman watchdog timeout
- Adding a new action client (e.g. a different Nav2 action)

**Key patterns:**
- `_cache_cb(topic, msg)` — generic subscription callback; stores to `self._cache[topic]`
- `get_latest(topic, timeout)` — blocking call; waits up to `timeout` seconds then returns the raw rclpy message or `None`
- All serialisation helpers at module level return plain `dict`s — they do NOT depend on `self`

**Adding a new subscription:**

1. Add an entry to `config/bridge.yaml` under `topics:`
2. Ensure the message type string resolves through `_resolve_msg_type()` — it handles any `pkg/MsgName` string automatically via `importlib`
3. Add a serialiser function at the bottom of `ros_node.py` if the type is new (follow existing pattern)

### `mcp_server.py` — MCP tools

Contains the FastMCP instance (`mcp = FastMCP(...)`) and all `@mcp.tool` decorated functions.

**Edit this file when:**
- Adding a new MCP tool
- Changing a tool's name, description, or parameters (these are what the LLM agent sees)
- Changing what data a tool returns

**Key patterns:**
- Tools are plain synchronous functions — no `async def`
- Each tool calls `_node.get_latest(topic, timeout)` to get sensor data, then calls a serialiser from `ros_node.py`
- `set_node(node)` must be called by `bridge.py` before the server starts
- The `run(transport, host, port, path, options)` function at the bottom is the FastMCP entry point — matches onit's expected MCP server API exactly

**Adding a new tool:**

```python
@mcp.tool(
    title="Human-readable tool name",
    description="What this tool does. This text is what the LLM agent reads.",
)
def my_new_tool(param1: str, param2: float = 1.0) -> str:
    """
    Args:
        param1: Description of param1.
        param2: Description of param2 (default 1.0).
    """
    msg = _node.get_latest("/some/topic", timeout=2.0)
    if msg is None:
        return json.dumps({"error": "No data received."})
    return json.dumps({"result": "..."})
```

### `behaviors.py` — High-level behaviors

Contains blocking behavioral functions called by the `find_object`, `approach_object`, and `look_around` tools.

**Edit this file when:**
- Improving the logic of existing behaviors (rotation speed, steering gain, etc.)
- Adding a new multi-step behavior that combines sensing and acting

**Key pattern:** Each function takes `node: ROS2BridgeNode` as first argument and returns a plain `dict`.  It calls `node.publish_twist()` and `node.get_latest()` in a timed loop.

### `bridge.py` — Entry point (rarely needs editing)

Handles config loading, starts the rclpy daemon thread, and calls `mcp_server.run()`.

**Edit this file only when:**
- Changing how the config file is located (the `_find_config()` function)
- Changing startup/shutdown order

### `config/bridge.yaml` — Runtime configuration

**Prefer editing this over Python** for:
- Topic names and message types
- Port / host / path
- Speed limits
- Adding new topics

---

## Concurrency model — important

```
Main thread:   mcp_server.run()  →  uvicorn  →  asyncio event loop
Daemon thread: rclpy MultiThreadedExecutor.spin()
```

- MCP tool functions run on uvicorn worker threads (synchronous, not async)
- `get_latest()` uses `threading.Event.wait()` — safe to call from any thread
- `publish_twist()` and `stop()` are thread-safe (no lock needed for publisher)
- **Do NOT call `rclpy.spin()` from inside a tool** — the executor is already running in the daemon thread
- **Do NOT use `asyncio.run()` inside a tool** — you are already inside a uvicorn asyncio loop on the main thread

---

## Message type resolution

`_resolve_msg_type(type_str)` in `ros_node.py` resolves a `"pkg/MsgName"` string to an rclpy class.

Built-in fast-path map covers the four default types.  Everything else falls through to `importlib.import_module(f"{pkg}.msg")`.  No code changes are needed to add support for a new message type — just add it to `bridge.yaml`.

---

## Adding support for a new sensor (end-to-end example)

**Goal:** add a `get_imu` tool that reads from `/imu/data`.

1. **`config/bridge.yaml`** — add under `topics:`:
   ```yaml
   imu:
     topic: /imu/data
     type: sensor_msgs/Imu
   ```

2. **`ros_node.py`** — add a serialiser at the bottom:
   ```python
   from sensor_msgs.msg import Imu

   def imu_to_dict(msg: Imu) -> dict:
       return {
           "linear_accel": {"x": msg.linear_acceleration.x, ...},
           "angular_vel":  {"x": msg.angular_velocity.x, ...},
       }
   ```

3. **`mcp_server.py`** — import the serialiser and add a tool:
   ```python
   from ros2_mcp_bridge.ros_node import imu_to_dict

   @mcp.tool(title="Get IMU", description="Return linear acceleration and angular velocity.")
   def get_imu(timeout: float = 2.0) -> str:
       topic = _node._cfg.get("topics", {}).get("imu", {}).get("topic", "/imu/data")
       msg = _node.get_latest(topic, timeout=timeout)
       if msg is None:
           return json.dumps({"error": "No IMU data."})
       return json.dumps(imu_to_dict(msg))
   ```

4. Rebuild and restart — no other files need touching.

---

## Build and run

```bash
# Install Python deps (once)
pip install "fastmcp>=2.0.0" pyyaml numpy

# Build
cd ~/turtlebot3_ws
colcon build --packages-select ros2_mcp_bridge
source install/setup.bash

# Run
ros2 launch ros2_mcp_bridge bridge.launch.py
```

---

## Testing a tool without an LLM agent

```bash
# Check the server is up
curl http://localhost:18210/ros2

# Use the FastMCP Python client
python3 - <<'EOF'
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:18210/ros2") as c:
        print(await c.list_tools())
        print(await c.call_tool("get_laser_scan", {}))

asyncio.run(main())
EOF
```

---

## Constraints and invariants to preserve

1. **All MCP tool functions must be synchronous** (`def`, not `async def`).  FastMCP wraps them in a thread internally.
2. **`ros_node.py` must not import `fastmcp`** — keep the ROS and MCP layers separate.
3. **`mcp_server.py` must not import `rclpy` directly** — use `_node` methods only.
4. **`behaviors.py` must not import `fastmcp` or `rclpy`** — only `ros_node` types.
5. **Serialiser functions return plain `dict`**, never raw rclpy messages.
6. **The deadman watchdog in `ros_node.py` must not be removed** — it is a safety requirement.
7. **The `run()` function signature in `mcp_server.py`** must stay as `run(transport, host, port, path, options=None)` — onit calls it with this signature.
