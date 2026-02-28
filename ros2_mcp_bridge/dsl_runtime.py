#!/usr/bin/env python3
"""
dsl_runtime.py — Lightweight Python-DSL execution engine for real-time
robot control programs.

The LLM writes small Python programs that run **locally** on the robot,
closing the sense → decide → act loop at ~10-20 Hz without round-tripping
every frame back to the cloud.  Programs are stored in a session-level
registry so they can be re-invoked with different parameters.

Design principles:
 • Restricted namespace — only whitelisted builtins and robot-specific
   functions are available; no file I/O, no imports, no network access.
 • Interruptible — every program runs in a daemon thread with a stop-flag
   checked automatically at every sensor read and sleep call.
 • Timeout — hard wall-clock limit configurable per invocation.
 • Safe — the robot is always stopped when a program ends (normally or
   via exception/timeout/cancellation).
"""

import copy
import logging
import math
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ros2_mcp_bridge.ros_node import ROS2BridgeNode

logger = logging.getLogger(__name__)

# Maximum number of stored programs and log lines per run.
_MAX_PROGRAMS = 50
_MAX_LOG_LINES = 500
_DEFAULT_TIMEOUT = 30.0
_MAX_TIMEOUT = 300.0


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class DSLProgram:
    """A stored DSL program."""
    name: str
    source: str
    description: str = ""
    default_params: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    run_count: int = 0


@dataclass
class DSLRunResult:
    """Result of a single DSL execution."""
    name: str
    status: str  # "completed", "stopped", "timeout", "error"
    return_value: Any = None
    log: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    error: str | None = None


# --------------------------------------------------------------------------- #
# Stop-signal exception (internal)
# --------------------------------------------------------------------------- #


class _StopRequested(Exception):
    """Raised inside the DSL program when the stop flag is set."""


# --------------------------------------------------------------------------- #
# DSLRuntime
# --------------------------------------------------------------------------- #


class DSLRuntime:
    """Session-level DSL program manager and executor.

    One instance is created per bridge process, injected by bridge.py.
    """

    def __init__(self, node: "ROS2BridgeNode", config: dict,
                 memory: dict | None = None,
                 waypoints: dict | None = None):
        self._node = node
        self._cfg = config
        self._dsl_cfg = config.get("dsl_runtime", {})
        self._enabled: bool = self._dsl_cfg.get("enabled", True)
        self._default_timeout: float = min(
            float(self._dsl_cfg.get("default_timeout", _DEFAULT_TIMEOUT)),
            _MAX_TIMEOUT,
        )
        self._max_timeout: float = float(
            self._dsl_cfg.get("max_timeout", _MAX_TIMEOUT)
        )

        # Shared memory dict (same object as mcp_server._memory)
        self._memory: dict[str, dict] = memory if memory is not None else {}

        # Shared waypoints dict (same object as mcp_server._waypoints)
        self._waypoints: dict[str, dict] = waypoints if waypoints is not None else {}

        # Program registry: {name: DSLProgram}
        self._programs: dict[str, DSLProgram] = {}
        self._lock = threading.Lock()

        # Currently running program state
        self._run_lock = threading.Lock()
        self._running_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._current_name: str | None = None

    # ------------------------------------------------------------------ #
    # Public API — program management
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        return self._enabled

    def store_program(
        self, name: str, source: str, description: str = "",
        default_params: dict | None = None,
    ) -> dict:
        """Store (or overwrite) a named DSL program."""
        if not self._enabled:
            return {"error": "DSL runtime is disabled in bridge.yaml."}
        if len(self._programs) >= _MAX_PROGRAMS and name not in self._programs:
            return {"error": f"Maximum {_MAX_PROGRAMS} programs reached. Delete some first."}
        prog = DSLProgram(
            name=name,
            source=source,
            description=description,
            default_params=default_params or {},
        )
        # Compile check
        try:
            compile(source, f"<dsl:{name}>", "exec")
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
        with self._lock:
            self._programs[name] = prog
        return {"stored": name, "description": description}

    def delete_program(self, name: str) -> dict:
        with self._lock:
            if name in self._programs:
                del self._programs[name]
                return {"deleted": name}
            return {"error": f"Program '{name}' not found."}

    def list_programs(self) -> dict:
        with self._lock:
            items = []
            for p in self._programs.values():
                items.append({
                    "name": p.name,
                    "description": p.description,
                    "default_params": p.default_params,
                    "run_count": p.run_count,
                })
            return {"programs": items, "count": len(items)}

    def get_program_source(self, name: str) -> dict:
        with self._lock:
            p = self._programs.get(name)
            if p is None:
                return {"error": f"Program '{name}' not found."}
            return {
                "name": p.name,
                "source": p.source,
                "description": p.description,
                "default_params": p.default_params,
            }

    # ------------------------------------------------------------------ #
    # Public API — execution
    # ------------------------------------------------------------------ #

    def run_program(
        self, name: str, params: dict | None = None,
        timeout: float | None = None,
    ) -> DSLRunResult:
        """Execute a stored DSL program synchronously (blocks until done)."""
        if not self._enabled:
            return DSLRunResult(name=name, status="error",
                                error="DSL runtime is disabled in bridge.yaml.")
        with self._lock:
            prog = self._programs.get(name)
            if prog is None:
                return DSLRunResult(name=name, status="error",
                                    error=f"Program '{name}' not found.")

        merged_params = {**prog.default_params, **(params or {})}
        timeout = min(float(timeout or self._default_timeout), self._max_timeout)
        return self._execute(prog, merged_params, timeout)

    def run_inline(
        self, source: str, params: dict | None = None,
        timeout: float | None = None,
    ) -> DSLRunResult:
        """Compile and execute DSL source directly (not stored)."""
        if not self._enabled:
            return DSLRunResult(name="<inline>", status="error",
                                error="DSL runtime is disabled in bridge.yaml.")
        try:
            compile(source, "<dsl:inline>", "exec")
        except SyntaxError as e:
            return DSLRunResult(name="<inline>", status="error",
                                error=f"Syntax error: {e}")
        prog = DSLProgram(name="<inline>", source=source)
        timeout = min(float(timeout or self._default_timeout), self._max_timeout)
        return self._execute(prog, params or {}, timeout)

    def stop_running(self) -> dict:
        """Request the currently running DSL program to stop."""
        with self._run_lock:
            if self._running_thread is None or not self._running_thread.is_alive():
                return {"status": "no_program_running"}
            self._stop_event.set()
            name = self._current_name
        # Wait briefly for clean shutdown
        self._running_thread.join(timeout=3.0)
        return {"status": "stop_requested", "program": name}

    def is_running(self) -> bool:
        with self._run_lock:
            return (self._running_thread is not None
                    and self._running_thread.is_alive())

    # ------------------------------------------------------------------ #
    # Internal — execution engine
    # ------------------------------------------------------------------ #

    def _execute(
        self, prog: DSLProgram, params: dict, timeout: float,
    ) -> DSLRunResult:
        """Run a DSLProgram, blocking the caller until it finishes."""
        # Only one program at a time
        with self._run_lock:
            if self._running_thread is not None and self._running_thread.is_alive():
                return DSLRunResult(
                    name=prog.name, status="error",
                    error="Another DSL program is already running. Stop it first.",
                )
            self._stop_event.clear()
            self._current_name = prog.name

        log_lines: list[str] = []
        result_holder: list[DSLRunResult] = []

        def _target():
            start = time.time()
            try:
                ns = self._build_namespace(params, log_lines, timeout, start)
                code = compile(prog.source, f"<dsl:{prog.name}>", "exec")
                exec(code, ns)  # noqa: S102
                ret = ns.get("__result__")
                result_holder.append(DSLRunResult(
                    name=prog.name, status="completed",
                    return_value=ret, log=log_lines,
                    duration_s=round(time.time() - start, 3),
                ))
            except _StopRequested:
                result_holder.append(DSLRunResult(
                    name=prog.name, status="stopped",
                    log=log_lines,
                    duration_s=round(time.time() - start, 3),
                ))
            except Exception as e:
                result_holder.append(DSLRunResult(
                    name=prog.name, status="error",
                    log=log_lines, error=f"{type(e).__name__}: {e}",
                    duration_s=round(time.time() - start, 3),
                ))
            finally:
                try:
                    self._node.stop()
                except Exception:
                    pass

        thread = threading.Thread(target=_target, daemon=True,
                                  name=f"dsl-{prog.name}")
        with self._run_lock:
            self._running_thread = thread
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Timed out — signal stop and wait a bit
            self._stop_event.set()
            thread.join(timeout=3.0)
            try:
                self._node.stop()
            except Exception:
                pass
            with self._run_lock:
                self._running_thread = None
                self._current_name = None
            return DSLRunResult(
                name=prog.name, status="timeout", log=log_lines,
                duration_s=timeout,
                error=f"Program exceeded {timeout}s timeout.",
            )

        with self._run_lock:
            self._running_thread = None
            self._current_name = None

        # Bump run count
        with self._lock:
            if prog.name in self._programs:
                self._programs[prog.name].run_count += 1

        if result_holder:
            return result_holder[0]
        return DSLRunResult(name=prog.name, status="error",
                            error="No result produced (internal error).")

    # ------------------------------------------------------------------ #
    # Namespace builder
    # ------------------------------------------------------------------ #

    def _build_namespace(
        self, params: dict, log_lines: list[str],
        timeout: float, start_time: float,
    ) -> dict:
        """Build the restricted global namespace for DSL execution."""
        node = self._node
        cfg = self._cfg
        stop_event = self._stop_event

        # ── guard: check stop + timeout ──────────────────────────────── #
        def _guard():
            if stop_event.is_set():
                raise _StopRequested()
            if time.time() - start_time > timeout:
                raise _StopRequested()

        # ── sensor functions ─────────────────────────────────────────── #
        from ros2_mcp_bridge.ros_node import (
            laser_scan_to_dict,
            odometry_to_dict,
            detections_to_dict,
            estimate_detection_distance,
            imu_to_dict,
            battery_state_to_dict,
        )

        cfg_topics = cfg.get("topics", {})
        image_width = cfg.get("image_width", 640)
        image_height = int(image_width * 480 / 640)
        hfov_deg = float(cfg.get("camera_hfov_deg", 62.0))

        def get_scan(timeout_s: float = 0.5) -> dict | None:
            """Read latest LiDAR scan → {front_min_m, left_min_m, ...} or None."""
            _guard()
            topic = cfg_topics.get("laser", {}).get("topic", "/scan")
            msg = node.get_latest(topic, timeout=timeout_s)
            if msg is None:
                return None
            return laser_scan_to_dict(msg)

        def get_odom(timeout_s: float = 0.5) -> dict | None:
            """Read odometry → {x, y, yaw_rad, yaw_deg} or None."""
            _guard()
            topic = cfg_topics.get("odom", {}).get("topic", "/odom")
            msg = node.get_latest(topic, timeout=timeout_s)
            if msg is None:
                return None
            return odometry_to_dict(msg)

        def get_detections(timeout_s: float = 0.5) -> list[dict]:
            """Read detection array → list of {label, confidence, bbox, distance_m, ...}."""
            _guard()
            topic = cfg_topics.get("detections", {}).get("topic", "/detections")
            msg = node.get_latest(topic, timeout=timeout_s)
            if msg is None:
                return []
            d = detections_to_dict(msg)
            # Enrich with distance estimates
            laser_topic = cfg_topics.get("laser", {}).get("topic", "/scan")
            scan_msg = node.get_latest(laser_topic, timeout=0.3)
            for det in d["detections"]:
                dist = estimate_detection_distance(
                    scan_msg, det, image_width, image_height, hfov_deg,
                )
                det.update(dist)
            return d["detections"]

        def get_imu(timeout_s: float = 0.5) -> dict | None:
            """Read IMU → {orientation, angular_velocity, linear_acceleration} or None."""
            _guard()
            topic = cfg_topics.get("imu", {}).get("topic", "/imu")
            msg = node.get_latest(topic, timeout=timeout_s)
            if msg is None:
                return None
            return imu_to_dict(msg)

        def get_battery(timeout_s: float = 0.5) -> dict | None:
            """Read battery state or None."""
            _guard()
            topic = cfg_topics.get("battery", {}).get("topic", "/battery_state")
            msg = node.get_latest(topic, timeout=timeout_s)
            if msg is None:
                return None
            return battery_state_to_dict(msg)

        # ── motion functions ─────────────────────────────────────────── #
        collision_cfg = cfg.get("collision_avoidance", {})
        ca_enabled = collision_cfg.get("enabled", True)

        def move(linear: float = 0.0, angular: float = 0.0,
                 duration: float = 0.0) -> None:
            """Publish twist. If duration>0, run for that long then stop."""
            _guard()
            if duration > 0:
                deadline = time.time() + min(duration, 10.0)
                while time.time() < deadline:
                    _guard()
                    node.publish_twist(linear, angular)
                    time.sleep(0.05)
                node.stop()
            else:
                node.publish_twist(linear, angular)

        def stop() -> None:
            """Immediately stop the robot."""
            node.stop()

        def move_distance(distance_m: float, speed: float = 0.0,
                          timeout_s: float = 20.0,
                          collision_avoidance: bool = True) -> dict:
            """Drive a precise distance (closed-loop odometry). Returns status dict."""
            _guard()
            ca = ca_enabled and collision_avoidance
            return node.move_distance(distance_m, speed,
                                      min(timeout_s, timeout), ca)

        def rotate(angle_deg: float, speed: float = 0.0,
                   timeout_s: float = 15.0) -> dict:
            """Rotate a precise angle (closed-loop). Returns status dict."""
            _guard()
            return node.rotate_angle(angle_deg, speed, min(timeout_s, timeout))

        def check_collision(linear_x: float = 0.15) -> dict:
            """Check LiDAR for collision risk. Returns {blocked, distance_m, ...}."""
            _guard()
            return node.check_collision(linear_x)

        # ── navigation functions ─────────────────────────────────────── #
        from ros2_mcp_bridge.ros_node import odometry_to_dict as _odom_to_dict

        waypoints = self._waypoints

        def navigate_to_pose(x: float, y: float, yaw: float = 0.0,
                             timeout_s: float = 60.0) -> dict:
            """Send robot to (x, y, yaw) via Nav2. Blocks until done. Returns status dict."""
            _guard()
            return node.navigate_to_pose(x, y, yaw, min(timeout_s, timeout))

        def save_waypoint(name: str) -> dict | None:
            """Save current odometry pose as a named waypoint. Returns pose or None."""
            _guard()
            odom_topic = cfg_topics.get("odom", {}).get("topic", "/odom")
            msg = node.get_latest(odom_topic, timeout=2.0)
            if msg is None:
                return None
            pose = _odom_to_dict(msg)
            waypoints[name] = {
                "x": pose["x"], "y": pose["y"],
                "yaw_deg": pose.get("yaw_deg", 0.0),
            }
            return waypoints[name]

        def go_to_waypoint(name: str, timeout_s: float = 60.0) -> dict:
            """Navigate to a previously saved waypoint via Nav2. Returns status dict."""
            _guard()
            if name not in waypoints:
                return {"status": "error", "message": f"Unknown waypoint '{name}'."}
            wp = waypoints[name]
            yaw_rad = math.radians(wp.get("yaw_deg", 0.0))
            return node.navigate_to_pose(wp["x"], wp["y"], yaw_rad,
                                         min(timeout_s, timeout))

        def list_waypoints() -> dict[str, dict]:
            """Return all saved waypoints as {name: {x, y, yaw_deg}}."""
            return dict(waypoints)

        # ── behavior functions ───────────────────────────────────────── #
        ca_global = ca_enabled

        def find_object(label: str, timeout_s: float = 20.0,
                        collision_avoidance: bool = True) -> dict:
            """Rotate in place searching for an object. Returns {found, detection, ...}."""
            _guard()
            from ros2_mcp_bridge.behaviors import find_object_behavior
            ca = collision_avoidance and ca_global
            return find_object_behavior(node, label, min(timeout_s, timeout), ca)

        def approach_object(label: str, stop_distance: float = 0.5,
                            timeout_s: float = 30.0,
                            collision_avoidance: bool = True) -> dict:
            """Drive toward a detected object. Returns {status, distance_m, ...}."""
            _guard()
            from ros2_mcp_bridge.behaviors import approach_object_behavior
            ca = collision_avoidance and ca_global
            return approach_object_behavior(node, label, stop_distance,
                                            min(timeout_s, timeout), ca)

        def follow_wall(side: str = "left", target_distance: float = 0.35,
                        duration_s: float = 10.0, speed: float = 0.0) -> dict:
            """Follow a wall on the given side. Returns {status, duration_s, ...}."""
            _guard()
            from ros2_mcp_bridge.behaviors import follow_wall_behavior
            if speed == 0.0:
                robot_cfg = cfg.get("robot", {})
                speed = float(robot_cfg.get("max_linear_speed", 0.22)) * 0.6
            return follow_wall_behavior(node, side, target_distance,
                                        min(duration_s, timeout), speed)

        def explore_for_object(label: str, step_distance: float = 0.5,
                               max_steps: int = 10, timeout_s: float = 180.0,
                               collision_avoidance: bool = True) -> dict:
            """Explore the environment to find an object. Returns {found, path, ...}."""
            _guard()
            from ros2_mcp_bridge.behaviors import explore_for_object_behavior
            ca = collision_avoidance and ca_global
            return explore_for_object_behavior(node, label, step_distance,
                                               max_steps,
                                               min(timeout_s, timeout), ca)

        # ── memory functions ───────────────────────────────────────────── #
        mem = self._memory

        def mem_get(key: str) -> str | None:
            """Read a scratchpad memory value by key. Returns None if missing."""
            entry = mem.get(key)
            return entry["value"] if entry is not None else None

        def mem_set(key: str, value: str, description: str = "") -> None:
            """Write a scratchpad memory entry (shared with LLM tools)."""
            existing_desc = mem.get(key, {}).get("description", "")
            mem[key] = {
                "value": str(value),
                "description": description or existing_desc,
            }

        def mem_list() -> dict[str, str]:
            """Return all memory entries as {key: value} dict."""
            return {k: v["value"] for k, v in mem.items()}

        def mem_delete(key: str) -> bool:
            """Delete a memory entry. Returns True if it existed."""
            if key in mem:
                del mem[key]
                return True
            return False

        # ── control flow helpers ─────────────────────────────────────── #
        def dsl_sleep(seconds: float) -> None:
            """Sleep with stop-flag checking (resolution ~50ms)."""
            end = time.time() + seconds
            while time.time() < end:
                _guard()
                time.sleep(min(0.05, end - time.time()))

        def log(msg: str) -> None:
            """Append a line to the program's output log."""
            if len(log_lines) < _MAX_LOG_LINES:
                ts = round(time.time() - start_time, 2)
                line = f"[{ts:6.2f}s] {msg}"
                log_lines.append(line)
                logger.info(f"DSL log: {line}")

        def elapsed() -> float:
            """Seconds since program start."""
            return round(time.time() - start_time, 3)

        def set_result(value: Any) -> None:
            """Set the return value of the program."""
            ns["__result__"] = value

        # ── build namespace ──────────────────────────────────────────── #
        # Whitelisted builtins only
        safe_builtins = {
            "True": True, "False": False, "None": None,
            "abs": abs, "all": all, "any": any, "bool": bool,
            "dict": dict, "enumerate": enumerate, "filter": filter,
            "float": float, "int": int, "isinstance": isinstance,
            "len": len, "list": list, "map": map, "max": max, "min": min,
            "print": lambda *a, **kw: log(" ".join(str(x) for x in a)),
            "range": range, "reversed": reversed, "round": round,
            "set": set, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "type": type, "zip": zip,
        }

        ns: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "__result__": None,

            # Parameters
            "params": copy.deepcopy(params),

            # Sensor functions
            "get_scan": get_scan,
            "get_odom": get_odom,
            "get_detections": get_detections,
            "get_imu": get_imu,
            "get_battery": get_battery,

            # Motion functions
            "move": move,
            "stop": stop,
            "move_distance": move_distance,
            "rotate": rotate,
            "check_collision": check_collision,

            # Navigation
            "navigate_to_pose": navigate_to_pose,
            "save_waypoint": save_waypoint,
            "go_to_waypoint": go_to_waypoint,
            "list_waypoints": list_waypoints,

            # Behaviors
            "find_object": find_object,
            "approach_object": approach_object,
            "follow_wall": follow_wall,
            "explore_for_object": explore_for_object,

            # Memory (shared with LLM tools)
            "get_memory": mem_get,
            "set_memory": mem_set,
            "list_memory": mem_list,
            "delete_memory": mem_delete,

            # Control / utility
            "sleep": dsl_sleep,
            "log": log,
            "elapsed": elapsed,
            "set_result": set_result,
            "time": time.time,

            # Math
            "math": math,
            "pi": math.pi,
            "sqrt": math.sqrt,
            "atan2": math.atan2,
            "sin": math.sin,
            "cos": math.cos,
            "radians": math.radians,
            "degrees": math.degrees,
        }

        return ns
