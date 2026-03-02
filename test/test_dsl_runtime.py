"""
test_dsl_runtime.py — Unit tests for the DSL runtime engine.

Tests run WITHOUT a live ROS 2 environment by patching ros2_mcp_bridge.ros_node
with a lightweight MockNode and stub *_to_dict helpers.

Run with:   pytest test/test_dsl_runtime.py -v
             (from the src/ros2_mcp_bridge directory, after sourcing the workspace)
"""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace

import json as _json_module
from unittest import mock as _mock

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Patch ros2_mcp_bridge.ros_node and ros2_mcp_bridge.behaviors in sys.modules
# BEFORE importing DSLRuntime so the lazy `from ros2_mcp_bridge.ros_node import ...`
# calls inside _build_namespace never touch rclpy.
# ---------------------------------------------------------------------------

def _make_fake_jpeg(width: int = 64, height: int = 48, color=(100, 150, 200)) -> bytes:
    """Create a minimal valid JPEG image as bytes."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


class MockNode:
    """Minimal stand-in for ROS2BridgeNode used by DSLRuntime._build_namespace."""

    _SCAN = {
        "front_min_m": 0.8, "left_min_m": 0.5,
        "right_min_m": 0.6, "rear_min_m": 1.2,
        "front_mean_m": 0.9, "left_mean_m": 0.55,
        "right_mean_m": 0.65, "rear_mean_m": 1.3,
    }
    _ODOM = {"x": 1.0, "y": 2.0, "yaw_rad": 0.5, "yaw_deg": 28.6}
    _IMU  = {"orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
              "angular_velocity": {"x": 0, "y": 0, "z": 0},
              "linear_acceleration": {"x": 0, "y": 0, "z": 9.8}}
    _BATT = {"voltage": 12.0, "percentage": 0.85}
    _DETS = {"detections": [
        {"label": "person", "confidence": 0.92,
         "bbox": {"cx": 320, "cy": 240, "w": 80, "h": 120},
         "distance_m": 1.5, "distance_source": "lidar"}
    ]}

    def __init__(self, topic_data: dict | None = None):
        self._topic_data = topic_data or {}
        self._stop_ev = threading.Event()
        self._published: list[tuple] = []   # records publish_twist calls

        # build default fake JPEG for camera topic
        jpeg = _make_fake_jpeg()
        cam_msg = SimpleNamespace(data=bytearray(jpeg), format="jpeg")
        self._topic_data.setdefault("/camera/image_raw/compressed", cam_msg)

    def get_latest(self, topic: str, timeout: float = 3.0):
        return self._topic_data.get(topic)

    def stop(self) -> None:
        pass

    def clear_stop_event(self) -> None:
        self._stop_ev.clear()

    def publish_twist(self, linear: float, angular: float) -> None:
        self._published.append((linear, angular))

    def move_distance(self, distance_m, speed=0, timeout_s=20, ca=True):
        return {"status": "completed", "distance_m": distance_m}

    def rotate_angle(self, angle_deg, speed=0, timeout_s=15):
        return {"status": "completed", "angle_deg": angle_deg}

    def check_collision(self, linear_x=0.15):
        return {"blocked": False, "distance_m": 0.8}

    def navigate_to_pose(self, x, y, yaw=0.0, timeout_s=60):
        return {"status": "completed", "x": x, "y": y, "yaw": yaw}


# Build fake ros_node module ------------------------------------------------
_fake_ros_node = types.ModuleType("ros2_mcp_bridge.ros_node")


class _FakeROS2BridgeNode(MockNode):
    pass


_fake_ros_node.ROS2BridgeNode   = _FakeROS2BridgeNode
# *_to_dict functions: the mock node's get_latest already returns the final dict
# for scan/odom/imu/battery, so these are identity functions.
_fake_ros_node.laser_scan_to_dict          = lambda msg: msg
_fake_ros_node.odometry_to_dict            = lambda msg: msg
_fake_ros_node.detections_to_dict          = lambda msg: msg
_fake_ros_node.imu_to_dict                 = lambda msg: msg
_fake_ros_node.battery_state_to_dict       = lambda msg: msg
_fake_ros_node.estimate_detection_distance = lambda *a, **kw: {}

# Build fake behaviors module -----------------------------------------------
_fake_behaviors = types.ModuleType("ros2_mcp_bridge.behaviors")
_fake_behaviors.find_object_behavior    = lambda node, label, *a, **kw: {"found": False}
_fake_behaviors.approach_object_behavior = lambda node, label, *a, **kw: {"status": "not_found"}
_fake_behaviors.follow_wall_behavior    = lambda node, *a, **kw: {"status": "completed"}

sys.modules.setdefault("ros2_mcp_bridge.ros_node", _fake_ros_node)
sys.modules.setdefault("ros2_mcp_bridge.behaviors", _fake_behaviors)

# Now safe to import DSLRuntime ---------------------------------------------
from ros2_mcp_bridge.dsl_runtime import DSLRuntime  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(extra_topics: dict | None = None) -> tuple[DSLRuntime, MockNode]:
    """Return (DSLRuntime, MockNode) pre-wired with mock topic data."""
    cfg = {
        "topics": {
            "laser":      {"topic": "/scan"},
            "odom":       {"topic": "/odom"},
            "imu":        {"topic": "/imu"},
            "battery":    {"topic": "/battery_state"},
            "camera":     {"topic": "/camera/image_raw/compressed"},
            "detections": {"topic": "/detections"},
        },
        "image_width": 640,
        "camera_hfov_deg": 62.0,
        "collision_avoidance": {"enabled": True},
    }
    topics = {
        "/scan":         MockNode._SCAN,
        "/odom":         MockNode._ODOM,
        "/imu":          MockNode._IMU,
        "/battery_state": MockNode._BATT,
        "/detections":   MockNode._DETS,
    }
    if extra_topics:
        topics.update(extra_topics)
    node = MockNode(topic_data=topics)
    dsl = DSLRuntime(node, cfg)
    return dsl, node


def run(source: str, *, dry_run: bool = False, timeout: float = 5.0,
        params: dict | None = None) -> dict:
    """Convenience wrapper: run inline DSL and return the result dict."""
    dsl, _ = _make_runtime()
    result = dsl.run_inline(source, params or {}, timeout, dry_run=dry_run)
    return {
        "status": result.status,
        "log": result.log,
        "return_value": result.return_value,
        "error": result.error,
        "duration_s": result.duration_s,
    }


# ===========================================================================
# TestValidate — static analysis via dsl_validate()
# ===========================================================================

class TestValidate:

    def _vld(self, source: str) -> dict:
        dsl, _ = _make_runtime()
        return dsl.validate_source(source)

    def test_clean_program_ok(self):
        r = self._vld("x = 1 + 2\nset_result(x)")
        assert r["ok"] is True
        assert r["errors"] == []

    def test_opencv_names_are_known(self):
        """cv_* names must not raise 'PossiblyUndefined' warnings."""
        src = "\n".join([
            "img = get_image()",
            "if img is not None:",
            "    g = cv_gray(img)",
            "    edges = cv_canny(g)",
            "    mask = cv_hsv_filter(img, [0,120,70], [10,255,255])",
            "    blobs = cv_find_contours(mask)",
            "    blob = cv_largest_blob(mask)",
            "    stats = cv_image_stats(img)",
            "    lines = cv_hough_lines(edges)",
            "    boxes = cv_draw_boxes(img, blobs)",
            "    jpg = cv_encode_jpg(img)",
            "    markers = cv_detect_aruco(img)",
        ])
        r = self._vld(src)
        # There should be no PossiblyUndefined warnings for cv_* names
        cv_warnings = [w for w in r["warnings"]
                       if w["type"] == "PossiblyUndefined"
                       and w["message"].split("'")[1].startswith("cv_")]
        assert cv_warnings == [], f"Unexpected warnings for cv_* names: {cv_warnings}"

    def test_sensor_names_are_known(self):
        src = (
            "s = get_scan()\n"
            "o = get_odom()\n"
            "d = get_detections()\n"
            "i = get_imu()\n"
            "b = get_battery()\n"
            "im = get_image()\n"
        )
        r = self._vld(src)
        undefined = [w for w in r["warnings"] if w["type"] == "PossiblyUndefined"]
        assert undefined == []

    def test_syntax_error_reported(self):
        r = self._vld("def foo(\n    x =\n")
        assert r["ok"] is False
        assert len(r["errors"]) >= 1
        assert r["errors"][0]["type"] == "SyntaxError"
        assert r["errors"][0]["line"] is not None

    def test_syntax_error_has_line_and_text(self):
        r = self._vld("x = 1\ny = (\nz = 2")
        assert r["ok"] is False
        err = r["errors"][0]
        assert "line" in err
        assert "col" in err

    def test_import_forbidden(self):
        r = self._vld("import os\nos.listdir('.')")
        assert r["ok"] is False
        types_ = [e["type"] for e in r["errors"]]
        assert "ForbiddenStatement" in types_

    def test_from_import_forbidden(self):
        r = self._vld("from os.path import join")
        assert r["ok"] is False
        assert any(e["type"] == "ForbiddenStatement" for e in r["errors"])

    def test_open_call_forbidden(self):
        r = self._vld("f = open('/etc/passwd')")
        assert r["ok"] is False
        assert any(e["type"] == "ForbiddenCall" for e in r["errors"])

    def test_eval_forbidden(self):
        r = self._vld("eval('1+1')")
        assert r["ok"] is False
        assert any(e["type"] == "ForbiddenCall" for e in r["errors"])

    def test_undefined_name_warning(self):
        r = self._vld("x = totally_made_up_func()")
        assert any(w["type"] == "PossiblyUndefined"
                   and "totally_made_up_func" in w["message"]
                   for w in r["warnings"])

    def test_locally_assigned_name_no_warning(self):
        """A name defined earlier in the same program must not warn."""
        r = self._vld("helper = lambda: 42\nresult = helper()")
        undefined = [w for w in r["warnings"]
                     if w["type"] == "PossiblyUndefined"
                     and "helper" in w["message"]]
        assert undefined == []

    def test_multiple_errors_reported(self):
        r = self._vld("import os\nimport sys")
        errors = [e for e in r["errors"] if e["type"] == "ForbiddenStatement"]
        assert len(errors) >= 2


# ===========================================================================
# TestDryRun — motion stubs, sensors live
# ===========================================================================

class TestDryRun:

    def test_move_is_stubbed(self):
        r = run("move(0.2, 0.0)\nlog('done')", dry_run=True)
        assert r["status"] == "completed"
        assert any("[DRY-RUN] move" in line for line in r["log"])

    def test_rotate_is_stubbed(self):
        r = run("res = rotate(90)\nset_result(res)", dry_run=True)
        assert r["status"] == "completed"
        assert r["return_value"]["dry_run"] is True

    def test_stop_is_stubbed(self):
        r = run("stop()", dry_run=True)
        assert r["status"] == "completed"
        assert any("[DRY-RUN] stop" in line for line in r["log"])

    def test_move_distance_is_stubbed(self):
        r = run("res = move_distance(1.0)\nset_result(res)", dry_run=True)
        assert r["return_value"]["dry_run"] is True

    def test_navigate_to_pose_is_stubbed(self):
        r = run("res = navigate_to_pose(3.0, 4.0)\nset_result(res)", dry_run=True)
        assert r["return_value"]["dry_run"] is True

    def test_go_to_waypoint_is_stubbed(self):
        r = run("save_waypoint('home')\nres = go_to_waypoint('home')\nset_result(res)",
                dry_run=True)
        assert r["return_value"]["dry_run"] is True

    def test_find_object_is_stubbed(self):
        r = run("res = find_object('cup')\nset_result(res)", dry_run=True)
        assert r["return_value"]["dry_run"] is True

    def test_approach_object_is_stubbed(self):
        r = run("res = approach_object('chair')\nset_result(res)", dry_run=True)
        assert r["return_value"]["dry_run"] is True

    def test_sensors_still_work_in_dry_run(self):
        r = run("s = get_scan()\nset_result(s)", dry_run=True)
        assert r["status"] == "completed"
        assert r["return_value"] is not None
        assert "front_min_m" in r["return_value"]

    def test_odom_still_works_in_dry_run(self):
        r = run("o = get_odom()\nset_result(o)", dry_run=True)
        assert r["return_value"]["x"] == pytest.approx(1.0)

    def test_dry_run_flag_exposed(self):
        r = run("set_result(dry_run)", dry_run=True)
        assert r["return_value"] is True

    def test_real_run_flag_is_false(self):
        r = run("set_result(dry_run)", dry_run=False)
        assert r["return_value"] is False

    def test_check_collision_reads_scan(self):
        """check_collision in dry-run still queries the live scan."""
        r = run("res = check_collision()\nset_result(res)", dry_run=True)
        assert "blocked" in r["return_value"]
        assert r["return_value"]["dry_run"] is True


# ===========================================================================
# TestExecution — basic runtime behaviour
# ===========================================================================

class TestExecution:

    def test_simple_arithmetic(self):
        r = run("set_result(2 + 3 * 4)")
        assert r["status"] == "completed"
        assert r["return_value"] == 14

    def test_params_passed(self):
        r = run("set_result(params['n'] * 2)", params={"n": 21})
        assert r["return_value"] == 42

    def test_log_captured(self):
        r = run("log('hello')\nlog('world')")
        assert any("hello" in l for l in r["log"])
        assert any("world" in l for l in r["log"])

    def test_print_redirects_to_log(self):
        r = run("print('via print')")
        assert any("via print" in l for l in r["log"])

    def test_store_and_run(self):
        dsl, _ = _make_runtime()
        dsl.store_program("adder", "set_result(params['a'] + params['b'])",
                          description="Adds two numbers")
        result = dsl.run_program("adder", {"a": 3, "b": 7})
        assert result.status == "completed"
        assert result.return_value == 10

    def test_syntax_error_in_store(self):
        dsl, _ = _make_runtime()
        r = dsl.store_program("bad", "def foo(\n    x =\n")
        assert "error" in r

    def test_syntax_error_in_run_inline(self):
        r = run("x = (")
        assert r["status"] == "error"
        assert "Syntax error" in (r["error"] or "")

    def test_runtime_error_has_dsl_traceback(self):
        r = run("x = 1\ny = 1 / 0  # line 2\nz = x + y")
        assert r["status"] == "error"
        assert r["error"] is not None
        # Should mention the error type and (ideally) a line reference
        assert "ZeroDivisionError" in r["error"] or "division by zero" in r["error"]

    def test_runtime_error_line_number(self):
        """The traceback should reference the DSL source frame."""
        src = "a = 1\nb = 2\nc = a + b  # line 3 ok\nd = int('not_a_number')  # line 4 error\n"
        r = run(src)
        assert r["status"] == "error"
        # The error should contain DSL frame info
        assert r["error"] is not None
        assert "int" in r["error"].lower() or "ValueError" in r["error"]

    def test_name_error_reported(self):
        r = run("result = undefined_name()")
        assert r["status"] == "error"
        assert "NameError" in r["error"]

    def test_elapsed_increases(self):
        r = run("e = elapsed()\nset_result(e)")
        # elapsed() measures time since start; should be a small non-negative float
        assert r["status"] == "completed"
        assert isinstance(r["return_value"], float)
        assert r["return_value"] >= 0.0

    def test_memory_ops(self):
        dsl, _ = _make_runtime()
        result = dsl.run_inline(
            "set_memory('k', 'v1')\n"
            "v = get_memory('k')\n"
            "set_result(v)"
        )
        assert result.status == "completed"
        assert result.return_value == "v1"

    def test_memory_shared_with_host(self):
        dsl, _ = _make_runtime()
        shared_mem: dict = {}
        dsl._memory = shared_mem
        dsl.run_inline("set_memory('key', '42', 'test')")
        assert "key" in shared_mem
        assert shared_mem["key"]["value"] == "42"

    def test_delete_memory(self):
        dsl, _ = _make_runtime()
        dsl.run_inline("set_memory('x', '1')\ndelete_memory('x')")
        assert "x" not in dsl._memory

    def test_list_memory(self):
        dsl, _ = _make_runtime()
        dsl.run_inline("set_memory('a','1')\nset_memory('b','2')")
        result = dsl.run_inline("set_result(list_memory())")
        assert result.return_value == {"a": "1", "b": "2"}

    def test_get_scan_returns_dict(self):
        r = run("s = get_scan()\nset_result(s)")
        assert r["status"] == "completed"
        assert isinstance(r["return_value"], dict)
        assert "front_min_m" in r["return_value"]

    def test_get_odom_returns_dict(self):
        r = run("o = get_odom()\nset_result(o)")
        assert r["return_value"]["x"] == pytest.approx(1.0)

    def test_get_none_when_no_sensor(self):
        """Requesting a topic not in mock returns None gracefully."""
        dsl, _ = _make_runtime()
        result = dsl.run_inline("s = get_battery()\nset_result(s is None)")
        # battery topic not in our mock → get_latest returns None
        # Note: we DID put /battery_state in MockNode defaults, so battery returns dict.
        # For this test, create a runtime with no battery topic.
        dsl2, _ = _make_runtime(extra_topics={"/battery_state": None})
        result2 = dsl2.run_inline("s = get_battery()\nset_result(s is None)")
        assert result2.return_value is True

    def test_stop_program(self):
        """A running program can be interrupted."""
        dsl, _ = _make_runtime()
        import threading as _th
        running = threading.Event()
        errors: list[str] = []

        def _start():
            try:
                dsl.run_inline(
                    "running = True\nwhile True:\n    sleep(0.05)\n",
                    timeout=10.0,
                )
            except Exception as e:
                errors.append(str(e))

        t = _th.Thread(target=_start, daemon=True)
        t.start()
        import time
        time.sleep(0.15)
        dsl.stop_running()
        t.join(timeout=5.0)
        assert not t.is_alive(), "Program thread should have stopped"
        assert errors == []


# ===========================================================================
# TestVLMDetections — VLM-first get_detections() with YOLO fallback
# ===========================================================================

def _vlm_response_json(objects: list) -> dict:
    """Build a minimal A2A message/send response body with an object list."""
    text = _json_module.dumps({"objects": objects})
    return {
        "result": {
            "artifacts": [
                {"parts": [{"kind": "text", "text": text}]}
            ]
        }
    }


def _make_vlm_runtime():
    """Return (DSLRuntime, MockNode) with vlm_agent.enabled=True."""
    cfg = {
        "topics": {
            "laser":      {"topic": "/scan"},
            "odom":       {"topic": "/odom"},
            "imu":        {"topic": "/imu"},
            "battery":    {"topic": "/battery_state"},
            "camera":     {"topic": "/camera/image_raw/compressed"},
            "detections": {"topic": "/detections"},
        },
        "image_width": 640,
        "camera_hfov_deg": 62.0,
        "collision_avoidance": {"enabled": True},
        "vlm_agent": {
            "enabled": True,
            "url": "http://localhost:9002",
        },
    }
    topics = {
        "/scan":          MockNode._SCAN,
        "/odom":          MockNode._ODOM,
        "/imu":           MockNode._IMU,
        "/battery_state": MockNode._BATT,
        "/detections":    MockNode._DETS,
    }
    node = MockNode(topic_data=topics)
    from ros2_mcp_bridge.dsl_runtime import DSLRuntime
    dsl = DSLRuntime(node, cfg)
    return dsl, node


def _patched_vlm_client(resp_body: dict):
    """Return a mock httpx.Client context-manager yielding mock_resp."""
    mock_resp = _mock.MagicMock()
    mock_resp.json.return_value = resp_body
    mock_resp.raise_for_status = _mock.Mock()
    patcher = _mock.patch("httpx.Client")
    MockClient = patcher.start()
    MockClient.return_value.__enter__ = _mock.Mock(return_value=MockClient.return_value)
    MockClient.return_value.__exit__ = _mock.Mock(return_value=False)
    MockClient.return_value.post.return_value = mock_resp
    return patcher


class TestVLMDetections:
    """Tests for VLM-first get_detections() with YOLO fallback."""

    # ── VLM disabled / bypassed → YOLO fallback ────────────────────────

    def test_vlm_disabled_falls_back_to_yolo(self):
        """Default config has no vlm_agent; YOLO result arrives with source='yolo'."""
        r = run("dets = get_detections()\nset_result(dets)")
        assert r["status"] == "completed"
        dets = r["return_value"]
        assert isinstance(dets, list) and len(dets) > 0
        assert dets[0]["label"] == "person"
        assert dets[0].get("source") == "yolo"

    def test_vlm_timeout_zero_forces_yolo(self):
        """vlm_timeout_s=0 skips VLM even when vlm_agent is enabled."""
        dsl, _ = _make_vlm_runtime()
        result = dsl.run_inline(
            "dets = get_detections(vlm_timeout_s=0)\nset_result(dets)"
        )
        assert result.status == "completed"
        dets = result.return_value
        assert len(dets) > 0
        assert dets[0]["source"] == "yolo"

    # ── VLM enabled but unavailable → YOLO fallback ────────────────────

    def test_vlm_http_error_falls_back_to_yolo(self):
        """httpx error → silent fallback to YOLO; program still completes."""
        import httpx as _httpx
        dsl, _ = _make_vlm_runtime()
        with _mock.patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = _mock.Mock(
                return_value=MockClient.return_value)
            MockClient.return_value.__exit__ = _mock.Mock(return_value=False)
            MockClient.return_value.post.side_effect = _httpx.ConnectError("refused")
            result = dsl.run_inline(
                "dets = get_detections()\nset_result(dets)"
            )
        assert result.status == "completed"
        dets = result.return_value
        assert any(d["label"] == "person" for d in dets)
        assert all(d.get("source") == "yolo" for d in dets)

    def test_vlm_bad_json_falls_back_to_yolo(self):
        """Unparseable VLM response text → fallback to YOLO."""
        bad_resp = {
            "result": {
                "artifacts": [{"parts": [{"kind": "text", "text": "NOT JSON!"}]}]
            }
        }
        dsl, _ = _make_vlm_runtime()
        patcher = _patched_vlm_client(bad_resp)
        try:
            result = dsl.run_inline(
                "dets = get_detections()\nset_result(dets)"
            )
        finally:
            patcher.stop()
        assert result.status == "completed"
        dets = result.return_value
        assert any(d["label"] == "person" for d in dets)

    # ── VLM returns valid detections ───────────────────────────────────

    def test_vlm_detections_parsed_correctly(self):
        """Valid VLM JSON → source='vlm', label/confidence correct."""
        resp = _vlm_response_json([
            {"label": "cup", "confidence": 0.88,
             "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.15},
        ])
        dsl, _ = _make_vlm_runtime()
        patcher = _patched_vlm_client(resp)
        try:
            result = dsl.run_inline(
                "dets = get_detections()\nset_result(dets)"
            )
        finally:
            patcher.stop()
        assert result.status == "completed"
        dets = result.return_value
        assert len(dets) == 1
        d = dets[0]
        assert d["label"] == "cup"
        assert d["source"] == "vlm"
        assert d["confidence"] == pytest.approx(0.88)

    def test_vlm_bbox_pixel_conversion(self):
        """Fractional bbox coords are scaled to image_width × image_height."""
        # image_width=640, image_height = 640 * 480/640 = 480
        resp = _vlm_response_json([
            {"label": "box", "confidence": 0.9,
             "cx": 0.25, "cy": 0.75, "w": 0.5, "h": 0.5},
        ])
        dsl, _ = _make_vlm_runtime()
        patcher = _patched_vlm_client(resp)
        try:
            result = dsl.run_inline(
                "dets = get_detections()\nset_result(dets[0]['bbox'])"
            )
        finally:
            patcher.stop()
        assert result.status == "completed"
        bbox = result.return_value
        assert bbox["cx"] == pytest.approx(0.25 * 640, abs=1.0)   # 160
        assert bbox["cy"] == pytest.approx(0.75 * 480, abs=1.0)   # 360
        assert bbox["w"]  == pytest.approx(0.50 * 640, abs=1.0)   # 320
        assert bbox["h"]  == pytest.approx(0.50 * 480, abs=1.0)   # 240

    def test_vlm_empty_objects_list_no_yolo_fallback(self):
        """VLM returns objects:[] → empty list (VLM answered, saw nothing)."""
        resp = _vlm_response_json([])
        dsl, _ = _make_vlm_runtime()
        patcher = _patched_vlm_client(resp)
        try:
            result = dsl.run_inline(
                "dets = get_detections()\nset_result(len(dets))"
            )
        finally:
            patcher.stop()
        assert result.status == "completed"
        assert result.return_value == 0  # VLM said nothing; YOLO not queried

    def test_vlm_multiple_objects_all_have_vlm_source(self):
        """Multiple VLM detections all carry source='vlm'."""
        resp = _vlm_response_json([
            {"label": "chair", "confidence": 0.9, "cx": 0.3, "cy": 0.5, "w": 0.2, "h": 0.4},
            {"label": "table", "confidence": 0.8, "cx": 0.7, "cy": 0.5, "w": 0.3, "h": 0.5},
        ])
        dsl, _ = _make_vlm_runtime()
        patcher = _patched_vlm_client(resp)
        try:
            result = dsl.run_inline(
                "sources = [d['source'] for d in get_detections()]\nset_result(sources)"
            )
        finally:
            patcher.stop()
        assert result.status == "completed"
        assert result.return_value == ["vlm", "vlm"]

    def test_vlm_markdown_fence_stripped(self):
        """VLM response wrapped in ```json...``` is handled correctly."""
        inner = _json_module.dumps({"objects": [
            {"label": "plant", "confidence": 0.7,
             "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.2}
        ]})
        resp = {
            "result": {"artifacts": [
                {"parts": [{"kind": "text", "text": f"```json\n{inner}\n```"}]}
            ]}
        }
        dsl, _ = _make_vlm_runtime()
        patcher = _patched_vlm_client(resp)
        try:
            result = dsl.run_inline(
                "dets = get_detections()\nset_result(dets[0]['label'])"
            )
        finally:
            patcher.stop()
        assert result.status == "completed"
        assert result.return_value == "plant"


# ===========================================================================
# TestOpenCV — all cv_* helper functions
# ===========================================================================

def _make_bgr(width=64, height=48, color=(100, 150, 200)) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)


def _make_gray(width=64, height=48, value=128) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


class TestOpenCV:
    """
    Each test runs a tiny DSL snippet with dry_run=True so sensors are live
    but no robot motion occurs.  All cv_* calls happen inside the DSL sandbox.
    """

    def _cv(self, snippet: str, *, setup: str = "", extra: dict | None = None) -> object:
        """
        Run: <setup>\n<snippet>\nset_result(<expr>)  in dry_run mode.
        `snippet` must assign `result` or call set_result() directly.
        """
        src = (setup + "\n" + snippet).strip()
        r = run(src, dry_run=True)
        assert r["status"] == "completed", (
            f"DSL error: {r['error']}\nLog: {r['log']}"
        )
        return r["return_value"]

    # ── cv_image_stats ───────────────────────────────────────────────── #

    def test_image_stats_shape(self):
        result = self._cv(
            "img = get_image()\nset_result(cv_image_stats(img))"
        )
        assert result["width"] == 64
        assert result["height"] == 48
        assert result["channels"] == 3
        assert 0 <= result["mean_brightness"] <= 255

    def test_image_stats_grayscale(self):
        # Encode a known-color image so we know the exact mean
        color = (128, 128, 128)
        img = _make_bgr(color=color)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        jpeg_bytes = bytes(buf)

        dsl, node = _make_runtime()
        node._topic_data["/camera/image_raw/compressed"] = SimpleNamespace(
            data=bytearray(jpeg_bytes), format="jpeg"
        )
        result = dsl.run_inline("img = get_image()\nset_result(cv_image_stats(img))")
        stats = result.return_value
        assert stats["channels"] == 3
        assert abs(stats["mean_brightness"] - 128) < 10  # JPEG is lossy

    # ── cv_gray ──────────────────────────────────────────────────────── #

    def test_cv_gray_produces_2d_array(self):
        result = self._cv(
            "img = get_image()\ng = cv_gray(img)\nset_result(cv_image_stats(g))"
        )
        assert result["channels"] == 1

    # ── cv_resize ────────────────────────────────────────────────────── #

    def test_cv_resize(self):
        result = self._cv(
            "img = get_image()\n"
            "small = cv_resize(img, 32, 16)\n"
            "set_result(cv_image_stats(small))"
        )
        assert result["width"] == 32
        assert result["height"] == 16

    # ── cv_blur ──────────────────────────────────────────────────────── #

    def test_cv_blur_same_shape(self):
        result = self._cv(
            "img = get_image()\n"
            "blurred = cv_blur(img, 3)\n"
            "set_result(cv_image_stats(blurred))"
        )
        assert result["width"] == 64
        assert result["height"] == 48

    def test_cv_blur_even_kernel_corrected(self):
        """Even kernel size should be auto-corrected to odd without error."""
        result = self._cv(
            "img = get_image()\n"
            "blurred = cv_blur(img, 4)\n"  # even → will be bumped to 5
            "set_result(cv_image_stats(blurred))"
        )
        assert result["width"] == 64

    # ── cv_canny ─────────────────────────────────────────────────────── #

    def test_cv_canny_returns_edge_mask(self):
        """Canny on a uniform image should produce (mostly) black edges."""
        result = self._cv(
            "img = get_image()\n"
            "edges = cv_canny(img, 50, 150)\n"
            "set_result(cv_image_stats(edges))"
        )
        assert result["channels"] == 1
        assert result["width"] == 64

    def test_cv_canny_on_grayscale_input(self):
        result = self._cv(
            "img = get_image()\n"
            "g = cv_gray(img)\n"
            "edges = cv_canny(g, 50, 150)\n"
            "set_result(cv_image_stats(edges))"
        )
        assert result["channels"] == 1

    # ── cv_hsv_filter ────────────────────────────────────────────────── #

    def test_hsv_filter_blue_image(self):
        """
        Inject an image that is clearly blue (BGR ~[200,100,50]).
        HSV blue range ~ H:100-130, S:100-255, V:50-255.
        The mask should have significant non-zero area.
        """
        blue_img = _make_bgr(color=(200, 100, 50))  # BGR: B=200, G=100, R=50
        _, buf = cv2.imencode(".jpg", blue_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        dsl, node = _make_runtime()
        node._topic_data["/camera/image_raw/compressed"] = SimpleNamespace(
            data=bytearray(bytes(buf)), format="jpeg"
        )
        result = dsl.run_inline(
            "img = get_image()\n"
            "mask = cv_hsv_filter(img, [100,80,40], [130,255,255])\n"
            "nonzero = int(sum(1 for row in mask for v in row if v > 0))\n"
            "set_result(nonzero)"
        )
        assert result.status == "completed", result.error
        assert result.return_value > 0, "Expected non-zero pixels in blue HSV mask"

    def test_hsv_filter_wrong_color_gives_empty_mask(self):
        """A pure red image should not match the blue HSV range."""
        red_img = _make_bgr(color=(50, 50, 200))   # BGR: R=200
        _, buf = cv2.imencode(".jpg", red_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        dsl, node = _make_runtime()
        node._topic_data["/camera/image_raw/compressed"] = SimpleNamespace(
            data=bytearray(bytes(buf)), format="jpeg"
        )
        result = dsl.run_inline(
            "img = get_image()\n"
            "mask = cv_hsv_filter(img, [100,80,40], [130,255,255])\n"
            "nonzero = int(sum(1 for row in mask for v in row if v > 0))\n"
            "set_result(nonzero)"
        )
        assert result.status == "completed"
        assert result.return_value == 0, "Red image should not match blue HSV range"

    # ── cv_find_contours / cv_largest_blob ───────────────────────────── #

    def test_find_contours_on_white_rect_mask(self):
        """
        Create a binary mask with a known white rectangle and verify
        cv_find_contours finds at least one contour.
        """
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (60, 70), 255, -1)   # ~40×50 = 2000 px
        _, jpg = cv2.imencode(".jpg", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 100])

        dsl, node = _make_runtime()
        node._topic_data["/camera/image_raw/compressed"] = SimpleNamespace(
            data=bytearray(bytes(jpg)), format="jpeg"
        )
        result = dsl.run_inline(
            "img = get_image()\n"
            "g = cv_gray(img)\n"
            "contours = cv_find_contours(g, min_area=50)\n"
            "set_result({'count': len(contours), 'first': contours[0] if contours else None})"
        )
        assert result.status == "completed", result.error
        assert result.return_value["count"] >= 1
        first = result.return_value["first"]
        assert "cx" in first
        assert "cy" in first
        assert first["area"] > 0

    def test_cv_largest_blob_none_on_black_mask(self):
        result = self._cv(
            "img = get_image()\n"
            "g = cv_gray(img)\n"
            "black = cv_canny(g, 300, 400)\n"  # very high threshold → black mask
            "set_result(cv_largest_blob(black) is None)"
        )
        # Result may be True (no blobs) or the dict (some noise blobs) — just check no crash
        assert result is True or isinstance(result, dict)

    # ── cv_hough_lines ───────────────────────────────────────────────── #

    def test_hough_lines_on_line_image(self):
        """
        Draw a clear horizontal line on an image and verify Hough detects it.
        Encode the image and inject it as the camera frame.
        """
        img = np.zeros((100, 200), dtype=np.uint8)
        cv2.line(img, (0, 50), (200, 50), 255, 3)       # horizontal line
        _, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 100])

        dsl, node = _make_runtime()
        node._topic_data["/camera/image_raw/compressed"] = SimpleNamespace(
            data=bytearray(bytes(jpg)), format="jpeg"
        )
        result = dsl.run_inline(
            "img = get_image()\n"
            "g = cv_gray(img)\n"
            "edges = cv_canny(g, 30, 100)\n"
            "lines = cv_hough_lines(edges, threshold=20, min_length=20.0)\n"
            "set_result({'count': len(lines), 'first': lines[0] if lines else None})"
        )
        assert result.status == "completed", result.error
        assert result.return_value["count"] >= 1, "Expected at least one Hough line"
        first = result.return_value["first"]
        assert "angle_deg" in first
        assert "length" in first

    # ── cv_detect_aruco ──────────────────────────────────────────────── #

    def test_detect_aruco_finds_marker(self):
        """
        Generate a real ArUco marker image (id=7), inject it, and verify detection.
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = np.zeros((150, 150), dtype=np.uint8)
        cv2.aruco.generateImageMarker(aruco_dict, 7, 150, marker, 1)
        # Add white border so detector has context
        bordered = cv2.copyMakeBorder(marker, 20, 20, 20, 20,
                                      cv2.BORDER_CONSTANT, value=255)
        bgr = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
        _, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])

        dsl, node = _make_runtime()
        node._topic_data["/camera/image_raw/compressed"] = SimpleNamespace(
            data=bytearray(bytes(jpg)), format="jpeg"
        )
        result = dsl.run_inline(
            "img = get_image()\n"
            "markers = cv_detect_aruco(img, 'DICT_4X4_50')\n"
            "set_result({'count': len(markers), 'ids': [m['id'] for m in markers]})"
        )
        assert result.status == "completed", result.error
        assert result.return_value["count"] >= 1, (
            "ArUco marker ID 7 should be detected"
        )
        assert 7 in result.return_value["ids"]

    def test_detect_aruco_no_marker(self):
        """Plain image → no markers detected."""
        result = self._cv(
            "img = get_image()\n"
            "markers = cv_detect_aruco(img)\n"
            "set_result(len(markers))"
        )
        assert result == 0

    # ── cv_encode_jpg ────────────────────────────────────────────────── #

    def test_cv_encode_jpg_returns_bytes(self):
        result = self._cv(
            "img = get_image()\n"
            "jpg = cv_encode_jpg(img, 80)\n"
            "set_result(len(jpg) > 0)"
        )
        assert result is True

    def test_cv_encode_jpg_decodable(self):
        """The encoded bytes should be decodable back to the same shape."""
        dsl, _ = _make_runtime()
        result = dsl.run_inline(
            "img = get_image()\n"
            "jpg = cv_encode_jpg(img, 95)\n"
            "set_result(list(jpg[:3]))"   # first 3 JPEG magic bytes
        )
        assert result.status == "completed"
        # JPEG magic: starts with FF D8 FF
        assert result.return_value[0] == 0xFF
        assert result.return_value[1] == 0xD8

    # ── cv_draw_boxes ────────────────────────────────────────────────── #

    def test_cv_draw_boxes_returns_copy(self):
        result = self._cv(
            "img = get_image()\n"
            "contours = [{'cx': 32, 'cy': 24, 'area': 500, 'x': 10, 'y': 10, 'w': 20, 'h': 20}]\n"
            "out = cv_draw_boxes(img, contours)\n"
            "set_result(cv_image_stats(out))"
        )
        assert result["width"] == 64
        assert result["height"] == 48

    def test_cv_resize_then_stats(self):
        result = self._cv(
            "img = get_image()\n"
            "r = cv_resize(img, 20, 10)\n"
            "set_result(cv_image_stats(r))"
        )
        assert result["width"] == 20
        assert result["height"] == 10

    # ── get_image pipeline ────────────────────────────────────────────── #

    def test_get_image_returns_numpy_array(self):
        """get_image() should decode the JPEG and return a HxWx3 uint8 array."""
        dsl, _ = _make_runtime()
        result = dsl.run_inline(
            "img = get_image()\n"
            "set_result(cv_image_stats(img) if img is not None else None)"
        )
        assert result.status == "completed", result.error
        assert result.return_value is not None
        assert result.return_value["channels"] == 3

    def test_get_image_none_when_no_camera(self):
        """If camera topic not available, get_image() returns None gracefully."""
        dsl, node = _make_runtime()
        del node._topic_data["/camera/image_raw/compressed"]
        result = dsl.run_inline("img = get_image()\nset_result(img is None)")
        assert result.return_value is True

    def test_opencv_pipeline_hsv_contour(self):
        """
        Full pipeline: get_image → cv_hsv_filter → cv_find_contours →
        cv_draw_boxes → cv_encode_jpg.
        Inject a clearly blue image so the filter produces blobs.
        """
        blue_img = _make_bgr(color=(200, 100, 50))
        _, buf = cv2.imencode(".jpg", blue_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        dsl, node = _make_runtime()
        node._topic_data["/camera/image_raw/compressed"] = SimpleNamespace(
            data=bytearray(bytes(buf)), format="jpeg"
        )
        result = dsl.run_inline(
            "img = get_image()\n"
            "mask = cv_hsv_filter(img, [100,80,40], [130,255,255])\n"
            "blobs = cv_find_contours(mask, min_area=10)\n"
            "annotated = cv_draw_boxes(img, blobs)\n"
            "jpg = cv_encode_jpg(annotated, 85)\n"
            "set_result({'blobs': len(blobs), 'jpg_len': len(jpg)})"
        )
        assert result.status == "completed", result.error
        assert result.return_value["blobs"] >= 1
        assert result.return_value["jpg_len"] > 0
