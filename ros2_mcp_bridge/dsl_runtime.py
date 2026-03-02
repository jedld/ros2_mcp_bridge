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

import ast
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

    def validate_source(self, source: str) -> dict:
        """Static-check DSL source without executing it.

        Performs:
          1. Python syntax check (with line/column info)
          2. Forbidden-statement detection (import, open, eval, exec, …)
          3. Undefined-name warnings for common DSL typos

        Returns::
            {
              "ok": bool,
              "errors": [{"type", "message", "line"}, ...],
              "warnings": [{"type", "message", "line"}, ...],
            }
        """
        errors: list[dict] = []
        warnings: list[dict] = []

        # 1. Syntax check ------------------------------------------------
        try:
            tree = ast.parse(source, "<dsl:validate>")
        except SyntaxError as e:
            return {
                "ok": False,
                "errors": [{
                    "type": "SyntaxError",
                    "message": str(e.msg),
                    "line": e.lineno,
                    "col": e.offset,
                    "text": (e.text or "").rstrip(),
                }],
                "warnings": [],
            }

        # 2. Forbidden / dangerous patterns --------------------------------
        _FORBIDDEN_CALLS = frozenset({
            "open", "eval", "exec", "__import__", "compile",
            "breakpoint", "input",
        })
        # Names available in the DSL namespace (for undefined-name check)
        _KNOWN_NAMES = frozenset({
            "params", "dry_run",
            # sensors
            "get_scan", "get_odom", "get_detections", "get_imu",
            "get_battery", "get_image",
            # motion
            "move", "stop", "move_distance", "rotate", "check_collision",
            # navigation
            "navigate_to_pose", "save_waypoint", "go_to_waypoint",
            "list_waypoints",
            # behaviors
            "find_object", "approach_object", "follow_wall",
            "explore_for_object",
            # memory
            "get_memory", "set_memory", "list_memory", "delete_memory",
            # control
            "sleep", "log", "elapsed", "set_result", "time",
            # math
            "math", "pi", "sqrt", "atan2", "sin", "cos",
            "radians", "degrees",
            # opencv
            "cv_gray", "cv_resize", "cv_blur", "cv_canny",
            "cv_hsv_filter", "cv_find_contours", "cv_largest_blob",
            "cv_hough_lines", "cv_detect_aruco", "cv_image_stats",
            "cv_encode_jpg", "cv_draw_boxes",
            # safe builtins
            "True", "False", "None",
            "abs", "all", "any", "bool", "dict", "enumerate", "filter",
            "float", "int", "isinstance", "len", "list", "map", "max",
            "min", "print", "range", "reversed", "round", "set",
            "sorted", "str", "sum", "tuple", "type", "zip",
        })

        # Collect all names assigned (defined) in the top-level scope
        _assigned: set[str] = set()
        for node_item in ast.walk(tree):
            if isinstance(node_item, (ast.Assign, ast.AugAssign)):
                targets = (
                    node_item.targets
                    if isinstance(node_item, ast.Assign)
                    else [node_item.target]
                )
                for t in targets:
                    if isinstance(t, ast.Name):
                        _assigned.add(t.id)
            elif isinstance(node_item, (ast.FunctionDef, ast.AsyncFunctionDef,
                                        ast.ClassDef)):
                _assigned.add(node_item.name)
            elif isinstance(node_item, ast.For):
                if isinstance(node_item.target, ast.Name):
                    _assigned.add(node_item.target.id)
            elif isinstance(node_item, ast.NamedExpr):
                if isinstance(node_item.target, ast.Name):
                    _assigned.add(node_item.target.id)

        _all_known = _KNOWN_NAMES | _assigned

        for node_item in ast.walk(tree):
            lineno = getattr(node_item, "lineno", None)

            # Forbidden: import statements
            if isinstance(node_item, (ast.Import, ast.ImportFrom)):
                errors.append({
                    "type": "ForbiddenStatement",
                    "message": (
                        "import statements are not allowed in DSL programs. "
                        "All needed modules are pre-imported."
                    ),
                    "line": lineno,
                })

            # Forbidden: dangerous calls
            elif isinstance(node_item, ast.Call):
                func = node_item.func
                name_id = (
                    func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute)
                    else None
                )
                if name_id in _FORBIDDEN_CALLS:
                    errors.append({
                        "type": "ForbiddenCall",
                        "message": (
                            f"'{name_id}()' is not available in the DSL sandbox."
                        ),
                        "line": lineno,
                    })

            # Warning: __dunder__ attribute access (likely to fail)
            elif isinstance(node_item, ast.Attribute):
                if (node_item.attr.startswith("__")
                        and node_item.attr.endswith("__")):
                    warnings.append({
                        "type": "DunderAccess",
                        "message": (
                            f"Accessing dunder attribute '{node_item.attr}' "
                            "is likely to fail in the sandbox."
                        ),
                        "line": lineno,
                    })

            # Warning: Name used as load that is not in the known set
            elif isinstance(node_item, ast.Name):
                if (isinstance(node_item.ctx, ast.Load)
                        and node_item.id not in _all_known
                        and not node_item.id.startswith("_")):
                    warnings.append({
                        "type": "PossiblyUndefined",
                        "message": (
                            f"'{node_item.id}' is not a built-in DSL name. "
                            "Ensure it is assigned before use."
                        ),
                        "line": lineno,
                    })

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------ #
    # Public API — execution
    # ------------------------------------------------------------------ #

    def run_program(
        self, name: str, params: dict | None = None,
        timeout: float | None = None,
        dry_run: bool = False,
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
        return self._execute(prog, merged_params, timeout, dry_run=dry_run)

    def run_inline(
        self, source: str, params: dict | None = None,
        timeout: float | None = None,
        dry_run: bool = False,
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
        return self._execute(prog, params or {}, timeout, dry_run=dry_run)

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
        dry_run: bool = False,
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
            self._node.clear_stop_event()  # clear node-level stop so DSL motion starts clean
            self._current_name = prog.name

        log_lines: list[str] = []
        result_holder: list[DSLRunResult] = []

        def _target():
            start = time.time()
            try:
                ns = self._build_namespace(params, log_lines, timeout, start,
                                           dry_run=dry_run)
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
                # Build a clean traceback pointing at the DSL source lines
                full_tb = traceback.format_exc()
                tb_lines = full_tb.splitlines()
                # Keep only frames that reference the user's DSL file
                dsl_frames: list[str] = []
                collect = False
                for ln in tb_lines:
                    if "<dsl:" in ln:
                        collect = True
                    if collect:
                        dsl_frames.append(ln)
                error_detail = (
                    "\n".join(dsl_frames)
                    if dsl_frames
                    else f"{type(e).__name__}: {e}"
                )
                result_holder.append(DSLRunResult(
                    name=prog.name, status="error",
                    log=log_lines, error=error_detail,
                    duration_s=round(time.time() - start, 3),
                ))
            finally:
                try:
                    if not dry_run:
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
        dry_run: bool = False,
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

        # VLM detection prompt — asks the model to return a structured JSON list
        # of every visible object with normalised bounding-box coordinates.
        _VLM_DETECT_PROMPT = (
            "Identify every distinct object visible in this image.\n"
            "Reply with ONLY a JSON object (no markdown, no explanation):\n"
            '{"objects": [{"label": "bottle", "confidence": 0.9, '
            '"cx": 0.52, "cy": 0.45, "w": 0.12, "h": 0.28}]}\n'
            "cx and cy are the fractional centre of each object "
            "(0.0=left/top … 1.0=right/bottom); "
            "w and h are the fractional width and height of its bounding box."
        )

        def _try_vlm_detections(vlm_timeout_s: float) -> list[dict] | None:
            """Attempt VLM-based object detection.

            Returns a list (possibly empty) on success, or None if VLM is
            disabled / unavailable — caller should then fall back to YOLO.
            """
            vlm_cfg = cfg.get("vlm_agent", {})
            if not vlm_cfg.get("enabled", False):
                return None
            vlm_url = vlm_cfg.get("url", "").rstrip("/")
            if not vlm_url:
                return None

            # Get current camera frame
            import base64
            cam_topic = cfg_topics.get("camera", {}).get(
                "topic", "/camera/image_raw/compressed")
            cam_msg = node.get_latest(cam_topic, timeout=min(2.0, vlm_timeout_s))
            if cam_msg is None:
                logger.debug("VLM detection: no camera frame available")
                return None

            img_b64 = base64.b64encode(bytes(cam_msg.data)).decode("utf-8")
            fmt = (cam_msg.format or "jpeg").lower().split("/")[-1]
            mime_type = f"image/{fmt}"

            # Build A2A message/send payload
            try:
                import uuid as _uuid
                import httpx as _httpx
                import json as _json
                payload = {
                    "jsonrpc": "2.0",
                    "id": str(_uuid.uuid4()),
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "messageId": str(_uuid.uuid4()),
                            "parts": [
                                {"kind": "text", "text": _VLM_DETECT_PROMPT},
                                {"kind": "file", "file": {
                                    "bytes": img_b64,
                                    "mimeType": mime_type,
                                    "name": "camera_frame.jpg",
                                }},
                            ],
                        }
                    },
                }
                with _httpx.Client(timeout=vlm_timeout_s) as _client:
                    resp = _client.post(vlm_url, json=payload)
                    resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.debug("VLM detection HTTP error, falling back to YOLO: %s", exc)
                return None

            # Extract text from A2A response
            text: str | None = None
            result = data.get("result", {})
            for artifact in result.get("artifacts", []):
                for part in artifact.get("parts", []):
                    if part.get("kind") == "text":
                        text = part["text"]
                        break
            if text is None:
                task_result = result.get("result")
                if task_result:
                    for part in task_result.get("parts", []):
                        if part.get("kind") == "text":
                            text = part["text"]
                            break
            if text is None:
                for part in result.get("parts", []):
                    if part.get("kind") == "text":
                        text = part["text"]
                        break
            if not text:
                logger.debug("VLM detection: empty response, falling back to YOLO")
                return None

            # Parse JSON (handle optional markdown code fences)
            try:
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                parsed = _json.loads(text)
            except Exception as exc:
                logger.debug("VLM detection: JSON parse error (%s), falling back to YOLO", exc)
                return None

            objects = parsed.get("objects", [])
            if not isinstance(objects, list):
                return None

            # Convert fractional bbox → pixel coords (matching YOLO bbox format)
            detections: list[dict] = []
            for obj in objects:
                if not isinstance(obj, dict) or "label" not in obj:
                    continue
                try:
                    cx_px = float(obj.get("cx", 0.5)) * image_width
                    cy_px = float(obj.get("cy", 0.5)) * image_height
                    w_px  = float(obj.get("w",  0.1)) * image_width
                    h_px  = float(obj.get("h",  0.1)) * image_height
                    detections.append({
                        "label":      str(obj["label"]),
                        "confidence": round(float(obj.get("confidence", 0.8)), 3),
                        "source":     "vlm",
                        "bbox": {
                            "cx": round(cx_px, 1),
                            "cy": round(cy_px, 1),
                            "w":  round(w_px,  1),
                            "h":  round(h_px,  1),
                        },
                    })
                except (TypeError, ValueError):
                    continue
            return detections  # may be empty list — that's valid (nothing seen)

        def get_detections(timeout_s: float = 0.5,
                           vlm_timeout_s: float = 8.0) -> list[dict]:
            """Detect objects: tries VLM first, falls back to YOLO/topic.

            Returns list of {label, confidence, source, bbox, distance_m, ...}.
            ``source`` is "vlm" when the VLM answered, "yolo" otherwise.
            Set vlm_timeout_s=0 to skip VLM and go straight to YOLO.
            """
            _guard()
            # Pre-fetch LiDAR for distance enrichment (both paths)
            laser_topic = cfg_topics.get("laser", {}).get("topic", "/scan")
            scan_msg = node.get_latest(laser_topic, timeout=0.3)

            # ── VLM path ─────────────────────────────────────────────── #
            dets: list[dict] | None = None
            if vlm_timeout_s > 0:
                dets = _try_vlm_detections(vlm_timeout_s)

            # ── YOLO fallback ─────────────────────────────────────────── #
            if dets is None:
                topic = cfg_topics.get("detections", {}).get("topic", "/detections")
                msg = node.get_latest(topic, timeout=timeout_s)
                if msg is None:
                    return []
                d = detections_to_dict(msg)
                dets = d["detections"]
                for det in dets:
                    det.setdefault("source", "yolo")

            # ── Distance enrichment (shared) ──────────────────────────── #
            for det in dets:
                dist = estimate_detection_distance(
                    scan_msg, det, image_width, image_height, hfov_deg,
                )
                det.update(dist)
            return dets

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

        def get_image(timeout_s: float = 0.5):
            """Get the latest camera frame as a BGR numpy array, or None if unavailable."""
            _guard()
            import cv2 as _cv2
            import numpy as _np
            topic = cfg_topics.get("camera", {}).get(
                "topic", "/camera/image_raw/compressed")
            msg = node.get_latest(topic, timeout=timeout_s)
            if msg is None:
                return None
            arr = _np.frombuffer(bytes(msg.data), dtype=_np.uint8)
            return _cv2.imdecode(arr, _cv2.IMREAD_COLOR)

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

        def cancel_navigation() -> dict:
            """Cancel the currently active Nav2 goal. Returns status dict."""
            _guard()
            return node.cancel_navigation()

        def set_initial_pose(
            x: float, y: float, yaw_deg: float = 0.0,
            cov_xy: float = 0.25, cov_yaw: float = 0.0685,
        ) -> dict:
            """Publish /initialpose for AMCL (static-map mode only)."""
            _guard()
            return node.set_initial_pose(x, y, math.radians(yaw_deg), cov_xy, cov_yaw)

        def get_map_info() -> dict | None:
            """Return current map metadata and occupancy statistics, or None."""
            _guard()
            return node.get_map_metadata(timeout=3.0)

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

        # def explore_for_object(label: str, step_distance: float = 0.5,
        #                        max_steps: int = 10, timeout_s: float = 180.0,
        #                        collision_avoidance: bool = True) -> dict:
        #     """Explore the environment to find an object. Returns {found, path, ...}."""
        #     _guard()
        #     from ros2_mcp_bridge.behaviors import explore_for_object_behavior
        #     ca = collision_avoidance and ca_global
        #     return explore_for_object_behavior(node, label, step_distance,
        #                                        max_steps,
        #                                        min(timeout_s, timeout), ca)

        def explore_for_object(label: str, step_distance: float = 0.5,
                               max_steps: int = 10, timeout_s: float = 180.0,
                               collision_avoidance: bool = True) -> dict:
            """Explore the environment searching for an object (not yet fully implemented)."""
            _guard()
            log(f"explore_for_object('{label}') is not yet fully implemented; "
                "falling back to find_object.")
            return find_object(label, min(timeout_s, timeout), collision_avoidance)

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

        # ── opencv utilities ─────────────────────────────────────────── #
        # All cv_* functions require get_image() or a numpy array from it.
        def cv_gray(img):
            """Convert a BGR image to grayscale."""
            import cv2 as _cv2
            return _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)

        def cv_resize(img, width: int, height: int):
            """Resize image to (width, height) pixels."""
            import cv2 as _cv2
            return _cv2.resize(img, (int(width), int(height)))

        def cv_blur(img, kernel_size: int = 5):
            """Apply Gaussian blur. kernel_size must be odd (auto-corrected)."""
            import cv2 as _cv2
            k = int(kernel_size)
            if k % 2 == 0:
                k += 1
            return _cv2.GaussianBlur(img, (k, k), 0)

        def cv_canny(img, low: float = 50.0, high: float = 150.0):
            """Canny edge detection. Works on BGR or grayscale images."""
            import cv2 as _cv2
            gray = img if len(img.shape) == 2 else _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
            return _cv2.Canny(gray, low, high)

        def cv_hsv_filter(img, lower: list, upper: list):
            """
            Return a binary mask where pixels fall within the HSV range.
            lower/upper are [H, S, V].  H ∈ [0,179], S/V ∈ [0,255].
            Example: red ≈ lower=[0,120,70], upper=[10,255,255].
            """
            import cv2 as _cv2
            import numpy as _np
            hsv = _cv2.cvtColor(img, _cv2.COLOR_BGR2HSV)
            return _cv2.inRange(
                hsv,
                _np.array(lower, dtype=_np.uint8),
                _np.array(upper, dtype=_np.uint8),
            )

        def cv_find_contours(mask, min_area: float = 100.0) -> list:
            """
            Find contours in a binary mask.
            Returns [{area, cx, cy, x, y, w, h}, ...] sorted by area descending.
            """
            import cv2 as _cv2
            contours, _ = _cv2.findContours(
                mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
            result = []
            for c in contours:
                area = _cv2.contourArea(c)
                if area < min_area:
                    continue
                x, y, w, h = _cv2.boundingRect(c)
                M = _cv2.moments(c)
                cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
                cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2
                result.append({
                    "area": float(area), "cx": cx, "cy": cy,
                    "x": x, "y": y, "w": w, "h": h,
                })
            return sorted(result, key=lambda c: c["area"], reverse=True)

        def cv_largest_blob(mask) -> dict | None:
            """
            Return the largest connected blob in a binary mask as
            {cx, cy, area, x, y, w, h}, or None if no blob found.
            """
            blobs = cv_find_contours(mask, min_area=1)
            return blobs[0] if blobs else None

        def cv_hough_lines(edges, threshold: int = 50,
                           min_length: float = 30.0,
                           max_gap: float = 10.0) -> list:
            """
            Probabilistic Hough line detection on a Canny edge image.
            Returns [{x1,y1,x2,y2,length,angle_deg}, ...].
            """
            import cv2 as _cv2
            import numpy as _np
            lines = _cv2.HoughLinesP(
                edges, 1, _np.pi / 180, int(threshold),
                minLineLength=min_length, maxLineGap=max_gap,
            )
            if lines is None:
                return []
            result = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = float(_np.hypot(x2 - x1, y2 - y1))
                angle = float(_np.degrees(_np.arctan2(y2 - y1, x2 - x1)))
                result.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "length": round(length, 1),
                    "angle_deg": round(angle, 1),
                })
            return result

        def cv_detect_aruco(img, dict_type: str = "DICT_4X4_50") -> list:
            """
            Detect ArUco markers in a BGR image.
            Returns [{id, corners, cx, cy}, ...].
            dict_type options: DICT_4X4_50, DICT_4X4_100, DICT_5X5_50, DICT_6X6_50.
            """
            import cv2 as _cv2
            _dict_map = {
                "DICT_4X4_50":  _cv2.aruco.DICT_4X4_50,
                "DICT_4X4_100": _cv2.aruco.DICT_4X4_100,
                "DICT_5X5_50":  _cv2.aruco.DICT_5X5_50,
                "DICT_6X6_50":  _cv2.aruco.DICT_6X6_50,
            }
            aruco_dict = _cv2.aruco.getPredefinedDictionary(
                _dict_map.get(dict_type, _cv2.aruco.DICT_4X4_50))
            detector = _cv2.aruco.ArucoDetector(
                aruco_dict, _cv2.aruco.DetectorParameters())
            corners, ids, _ = detector.detectMarkers(img)
            if ids is None:
                return []
            result = []
            for i, corner in enumerate(corners):
                pts = corner[0]
                cx = float(pts[:, 0].mean())
                cy = float(pts[:, 1].mean())
                result.append({
                    "id": int(ids[i][0]),
                    "corners": pts.tolist(),
                    "cx": round(cx, 1),
                    "cy": round(cy, 1),
                })
            return result

        def cv_image_stats(img) -> dict:
            """Return {height, width, channels, mean_brightness} for an image."""
            import numpy as _np
            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            return {
                "height": h, "width": w, "channels": channels,
                "mean_brightness": round(float(_np.mean(img)), 1),
            }

        def cv_encode_jpg(img, quality: int = 85) -> bytes:
            """
            Encode a numpy image to JPEG bytes.
            Example: set_result(cv_encode_jpg(img)) to surface a debug frame.
            """
            import cv2 as _cv2
            _, buf = _cv2.imencode(
                ".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            return bytes(buf)

        def cv_draw_boxes(img, contours: list, color: tuple = (0, 255, 0),
                          thickness: int = 2):
            """
            Draw bounding boxes from cv_find_contours / cv_detect_aruco onto
            a BGR image. Returns a new annotated copy (does not modify in-place).
            """
            import cv2 as _cv2
            out = img.copy()
            for c in contours:
                if "x" in c:  # contour dict
                    _cv2.rectangle(
                        out, (c["x"], c["y"]),
                        (c["x"] + c["w"], c["y"] + c["h"]),
                        color, thickness)
                if "cx" in c and "cy" in c:
                    _cv2.circle(out, (int(c["cx"]), int(c["cy"])), 4, color, -1)
            return out

        # ── dry-run motion stubs (overrides) ─────────────────────────── #
        # These replace the real motion / navigation / behavior functions
        # when dry_run=True so the program can be tested without physically
        # moving the robot.
        if dry_run:
            _real_stop = stop  # keep real stop so _execute finally still works

            def move(linear: float = 0.0, angular: float = 0.0,
                     duration: float = 0.0) -> None:
                log(f"[DRY-RUN] move(linear={linear}, angular={angular}, "
                    f"duration={duration})")
                if duration > 0:
                    dsl_sleep(min(duration, 1.0))

            def stop() -> None:
                log("[DRY-RUN] stop()")

            def move_distance(distance_m: float, speed: float = 0.0,
                              timeout_s: float = 20.0,
                              collision_avoidance: bool = True) -> dict:
                log(f"[DRY-RUN] move_distance({distance_m}m, speed={speed})")
                return {"status": "completed", "distance_m": distance_m,
                        "dry_run": True}

            def rotate(angle_deg: float, speed: float = 0.0,
                       timeout_s: float = 15.0) -> dict:
                log(f"[DRY-RUN] rotate({angle_deg}°, speed={speed})")
                return {"status": "completed", "angle_deg": angle_deg,
                        "dry_run": True}

            def check_collision(linear_x: float = 0.15) -> dict:
                scan = get_scan(timeout_s=0.3)  # still read real sensor
                if scan:
                    return {"blocked": scan.get("front_min_m", 1.0) < linear_x + 0.2,
                            "distance_m": scan.get("front_min_m", 1.0),
                            "dry_run": True}
                return {"blocked": False, "distance_m": 1.0, "dry_run": True}

            def navigate_to_pose(x: float, y: float, yaw: float = 0.0,
                                 timeout_s: float = 60.0) -> dict:
                log(f"[DRY-RUN] navigate_to_pose(x={x}, y={y}, yaw={yaw:.3f})")
                return {"status": "completed", "dry_run": True}

            def go_to_waypoint(name: str, timeout_s: float = 60.0) -> dict:
                log(f"[DRY-RUN] go_to_waypoint('{name}')")
                return {"status": "completed", "dry_run": True}

            def cancel_navigation() -> dict:
                log("[DRY-RUN] cancel_navigation()")
                return {"status": "cancel_requested", "dry_run": True}

            def set_initial_pose(
                x: float, y: float, yaw_deg: float = 0.0,
                cov_xy: float = 0.25, cov_yaw: float = 0.0685,
            ) -> dict:
                log(f"[DRY-RUN] set_initial_pose(x={x}, y={y}, yaw={yaw_deg}°)")
                return {"status": "published", "dry_run": True}

            # get_map_info reads real sensor data, no stub needed

            def find_object(label: str, timeout_s: float = 20.0,
                            collision_avoidance: bool = True) -> dict:
                log(f"[DRY-RUN] find_object('{label}')")
                return {"found": False, "dry_run": True}

            def approach_object(label: str, stop_distance: float = 0.5,
                                timeout_s: float = 30.0,
                                collision_avoidance: bool = True) -> dict:
                log(f"[DRY-RUN] approach_object('{label}', "
                    f"stop_dist={stop_distance})")
                return {"status": "not_found", "dry_run": True}

            def follow_wall(side: str = "left", target_distance: float = 0.35,
                            duration_s: float = 10.0, speed: float = 0.0) -> dict:
                log(f"[DRY-RUN] follow_wall(side='{side}', "
                    f"target={target_distance}m, dur={duration_s}s)")
                dsl_sleep(min(duration_s, 1.0))
                return {"status": "completed", "dry_run": True}

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
            "get_image": get_image,

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
            "cancel_navigation": cancel_navigation,
            "set_initial_pose": set_initial_pose,
            "get_map_info": get_map_info,

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
            "dry_run": dry_run,  # programs can check this flag

            # Math
            "math": math,
            "pi": math.pi,
            "sqrt": math.sqrt,
            "atan2": math.atan2,
            "sin": math.sin,
            "cos": math.cos,
            "radians": math.radians,
            "degrees": math.degrees,

            # OpenCV vision utilities
            "cv_gray": cv_gray,
            "cv_resize": cv_resize,
            "cv_blur": cv_blur,
            "cv_canny": cv_canny,
            "cv_hsv_filter": cv_hsv_filter,
            "cv_find_contours": cv_find_contours,
            "cv_largest_blob": cv_largest_blob,
            "cv_hough_lines": cv_hough_lines,
            "cv_detect_aruco": cv_detect_aruco,
            "cv_image_stats": cv_image_stats,
            "cv_encode_jpg": cv_encode_jpg,
            "cv_draw_boxes": cv_draw_boxes,
        }

        return ns
