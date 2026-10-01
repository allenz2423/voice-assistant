from __future__ import annotations

import re
import json
import struct
import subprocess
import threading

import pytest

import src.tools.computer_control as computer


def _png(width: int = 1000, height: int = 700) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00" + b"\x00\x00\x00\x00"


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-1", "DISPLAY": ":1"}, "wayland"),
        ({"WAYLAND_DISPLAY": "wayland-1", "DISPLAY": ":1"}, "wayland"),
        ({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"}, "x11"),
        ({"DISPLAY": ":1"}, "x11"),
        ({}, "unsupported"),
    ],
)
def test_detect_display_backend(environment, expected):
    assert computer.detect_display_backend(environment) == expected


def test_coordinate_mode_is_model_specific():
    assert computer.coordinate_mode_for_model("gui-owl-1.5-4b-vision") == "normalized_1000"
    assert computer.coordinate_mode_for_model("gemma4:e4b") == "pixels"


def test_x11_actions_require_fresh_screenshot_and_use_xdotool(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    commands = []

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
    )
    first = controller.run("inspect")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", first.message).group(1)
    stale = controller.run("click", snapshot_id="old-id", x=10, y=20)
    assert "latest observation" in stale.message.lower()
    assert commands == []
    assert controller.snapshot_id == ""
    assert "call inspect" in stale.message.lower()

    current_id = re.search(r"Snapshot ID: (\w+)", controller.run("inspect").message).group(1)
    clicked = controller.run("click", snapshot_id=current_id, x=10, y=20)
    assert clicked.screenshot == _png()
    assert commands == [
        ["xdotool", "mousemove", "--sync", "10", "20"],
        ["xdotool", "click", "--delay", "80", "1"],
    ]

    stale_input = controller.run("type", snapshot_id=current_id, text="This must be rejected as stale")
    assert "latest observation" in stale_input.message.lower()


def test_x11_type_shortcut_and_scroll_have_xdotool_counterparts(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    commands = []

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
    )
    inspected = controller.run("inspect")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)
    typed = controller.run("type", snapshot_id=snapshot_id, text="safe local text")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", typed.message).group(1)
    pressed = controller.run("press", snapshot_id=snapshot_id, key="ctrl+s")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", pressed.message).group(1)
    controller.run("scroll", snapshot_id=snapshot_id, direction="down", amount=2)
    snapshot_id = re.search(r"Snapshot ID: (\w+)", controller.run("inspect").message).group(1)
    controller.run("press", snapshot_id=snapshot_id, key="ctrl+plus")

    assert commands == [
        ["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", "safe local text"],
        ["xdotool", "key", "--clearmodifiers", "ctrl+s"],
        ["xdotool", "click", "--repeat", "2", "5"],
        ["xdotool", "key", "--clearmodifiers", "ctrl+shift+equal"],
    ]


def test_sequence_allows_selected_field_click_then_typing(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    commands = []

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
    )
    snapshot_id = controller.run("inspect").snapshot_id
    result = controller.run_sequence(
        snapshot_id=snapshot_id,
        actions=[{"action": "click", "x": 100, "y": 80}, {"action": "type", "text": "search text"}],
    )

    assert result.status == "ok"
    assert result.dispatched is True
    assert result.snapshot_id
    assert "Step 2/2 (type): ok" in result.message
    assert commands[-1] == ["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", "search text"]


def test_sequence_rejects_aggregate_text_over_limit_before_any_input(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    commands = []

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        max_text_length=10,
        max_sequence_text_length=10,
    )
    snapshot_id = controller.run("inspect").snapshot_id
    result = controller.run_sequence(
        snapshot_id=snapshot_id,
        actions=[
            {"action": "click", "x": 100, "y": 80},
            {"action": "type", "text": "123456"},
            {"action": "type", "text": "789012"},
        ],
    )

    assert result.status == "invalid_input"
    assert "No actions were run" in result.message
    assert commands == []


def test_sequence_rejects_single_text_action_over_limit_before_click(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    commands = []

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        max_text_length=5,
    )
    snapshot_id = controller.run("inspect").snapshot_id
    result = controller.run_sequence(
        snapshot_id=snapshot_id,
        actions=[{"action": "click", "x": 100, "y": 80}, {"action": "type", "text": "too long"}],
    )

    assert result.status == "invalid_input"
    assert "per-action text limit" in result.message
    assert commands == []


def test_sequence_cancellation_stops_before_next_action(monkeypatch):
    controller = computer.ComputerController(
        screenshot_fn=_png,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
    )
    cancellation = threading.Event()
    actions_run = []

    def run(action, **kwargs):
        actions_run.append(action)
        cancellation.set()
        return computer.ComputerControlResult(
            "Typed the requested text.", screenshot=_png(), status="ok", dispatched=True, snapshot_id="fresh"
        )

    controller.run = run
    result = controller.run_sequence(
        snapshot_id="initial",
        actions=[{"action": "type", "text": "first"}, {"action": "type", "text": "second"}],
        cancel_event=cancellation,
    )

    assert result.status == "cancelled"
    assert result.dispatched is True
    assert result.snapshot_id == "fresh"
    assert actions_run == ["type"]


def test_sequence_time_budget_is_checked_at_action_boundaries(monkeypatch):
    controller = computer.ComputerController(
        screenshot_fn=_png,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        sequence_timeout_seconds=1,
    )
    clock_values = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(computer.time, "monotonic", lambda: next(clock_values))
    actions_run = []

    def run(action, **kwargs):
        actions_run.append(action)
        return computer.ComputerControlResult(
            "Typed the requested text.", screenshot=_png(), status="ok", dispatched=True, snapshot_id="fresh"
        )

    controller.run = run
    result = controller.run_sequence(
        snapshot_id="initial",
        actions=[{"action": "type", "text": "first"}, {"action": "type", "text": "second"}],
    )

    assert result.status == "timed_out"
    assert "action boundary" in result.message
    assert "active action was allowed to finish" in result.message
    assert result.snapshot_id == "fresh"
    assert actions_run == ["type"]


def test_sequence_pauses_before_reusing_geometry_after_a_click(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    commands = []

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
    )
    snapshot_id = controller.run("inspect").snapshot_id
    result = controller.run_sequence(
        snapshot_id=snapshot_id,
        actions=[{"action": "click", "x": 100, "y": 80}, {"action": "click", "x": 200, "y": 90}],
    )

    assert result.status == "partial"
    assert result.dispatched is True
    assert result.snapshot_id
    assert "paused before this input" in result.message
    assert commands == [
        ["xdotool", "mousemove", "--sync", "100", "80"],
        ["xdotool", "click", "--delay", "80", "1"],
    ]


def test_sequence_does_not_press_a_key_after_click_without_reobservation(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    commands = []

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
    )
    snapshot_id = controller.run("inspect").snapshot_id
    result = controller.run_sequence(
        snapshot_id=snapshot_id,
        actions=[{"action": "click", "x": 100, "y": 80}, {"action": "press", "key": "enter"}],
    )

    assert result.status == "partial"
    assert "paused before this input" in result.message
    assert len(commands) == 2


def test_action_is_uncertain_if_post_action_focus_is_not_stable(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    commands = []

    def runner(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
    )
    identities = iter(["window-a", "window-a", "window-a", "window-a", "window-b"])
    controller._read_active_window_state = lambda: (next(identities), (0, 0, 1000, 700))
    snapshot_id = controller.run("inspect").snapshot_id
    result = controller.run("click", snapshot_id=snapshot_id, x=100, y=80)

    assert result.status == "uncertain"
    assert result.dispatched is True
    assert not result.snapshot_id
    assert "focus could not be confirmed stable" in result.message.lower()


def test_wayland_actions_use_ydotool_mouse_and_wtype_keyboard(monkeypatch):
    available_commands = {"ydotool": "/usr/bin/ydotool", "wtype": "/usr/bin/wtype", "swaymsg": "/usr/bin/swaymsg"}
    monkeypatch.setattr(computer.shutil, "which", lambda name: available_commands.get(name))
    commands = []

    def runner(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"WAYLAND_DISPLAY": "wayland-1", "SWAYSOCK": "/run/user/1000/sway-ipc.sock", "DISPLAY": ":1"},
        ensure_wayland_daemon=lambda: True,
    )
    controller._read_active_window_state = lambda: ("sway:test-window", (0, 0, 1000, 700))
    inspected = controller.run("inspect")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)
    clicked = controller.run("click", snapshot_id=snapshot_id, x=99, y=200)
    next_id = re.search(r"Snapshot ID: (\w+)", clicked.message).group(1)
    typed = controller.run("type", snapshot_id=next_id, text="local sample")
    key_id = re.search(r"Snapshot ID: (\w+)", typed.message).group(1)
    pressed = controller.run("press", snapshot_id=key_id, key="ctrl+a")

    assert commands[:2] == [
        ["ydotool", "mousemove", "--absolute", "99", "200"],
        ["ydotool", "click", "--next-delay", "80", "0xC0"],
    ]
    assert commands[2] == ["wtype", "--", "local sample"]
    assert commands[3] == ["wtype", "-M", "ctrl", "a", "-m", "ctrl"]
    assert "screenshot" in pressed.message.lower()


def test_wayland_adapter_uses_session_identity_when_multiple_tools_are_installed(monkeypatch):
    controller = computer.ComputerController(environ={
        "WAYLAND_DISPLAY": "wayland-1",
        "XDG_CURRENT_DESKTOP": "sway",
        "SWAYSOCK": "/run/user/1000/sway-ipc.sock",
    })
    monkeypatch.setattr(computer.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert controller._wayland_window_adapter() == "sway"


def test_wayland_control_is_not_advertised_without_a_supported_window_adapter(monkeypatch):
    controller = computer.ComputerController(environ={
        "WAYLAND_DISPLAY": "wayland-1",
        "XDG_CURRENT_DESKTOP": "unknown",
    })
    monkeypatch.setattr(computer.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert not controller.available


def test_normalized_1000_coordinates_are_scaled_to_screenshot_pixels(monkeypatch):
    monkeypatch.setattr(
        computer.shutil,
        "which",
        lambda name: {"ydotool": "/usr/bin/ydotool", "wtype": "/usr/bin/wtype", "swaymsg": "/usr/bin/swaymsg"}.get(name),
    )
    commands = []

    def runner(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"WAYLAND_DISPLAY": "wayland-1", "SWAYSOCK": "/run/user/1000/sway-ipc.sock"},
        ensure_wayland_daemon=lambda: True,
        coordinate_mode="normalized_1000",
    )
    controller._read_active_window_state = lambda: ("sway:test-window", (0, 0, 1000, 700))
    inspected = controller.run("inspect", scope="window")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)
    assert "normalized coordinates from 0 to 1000" in inspected.message

    clicked = controller.run("click", snapshot_id=snapshot_id, x=500, y=700)

    assert "normalized (500, 700)" in clicked.message
    assert "pixels (500, 489)" in clicked.message
    assert commands == [
        ["ydotool", "mousemove", "--absolute", "500", "489"],
        ["ydotool", "click", "--next-delay", "80", "0xC0"],
    ]


def test_x11_drag_uses_a_held_mouse_button_and_current_snapshot(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    commands = []

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
    )
    inspected = controller.run("inspect", scope="window")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)
    moved = controller.run("drag", snapshot_id=snapshot_id, x=200, y=20, end_x=500, end_y=200)

    assert "Dragged left" in moved.message
    assert commands == [
        ["xdotool", "mousemove", "--sync", "200", "20"],
        ["xdotool", "mousedown", "1"],
        ["xdotool", "mousemove", "--sync", "--duration", "0.35", "500", "200"],
        ["xdotool", "mouseup", "1"],
    ]


def test_wayland_window_drag_holds_detected_modifier(monkeypatch):
    commands = []
    available = {"ydotool": "/usr/bin/ydotool", "wtype": "/usr/bin/wtype", "swaymsg": "/usr/bin/swaymsg"}
    monkeypatch.setattr(computer.shutil, "which", lambda name: available.get(name))

    def runner(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"WAYLAND_DISPLAY": "wayland-1", "SWAYSOCK": "/run/user/1000/sway-ipc.sock"},
        ensure_wayland_daemon=lambda: True,
        screenshot_delay_seconds=0,
    )
    controller._window_move_modifier = lambda: "alt"
    controller._read_active_window_state = lambda: ("sway:test-window", (0, 0, 1000, 700))
    inspected = controller.run("inspect", scope="monitor", screenshot_delay_seconds=0)
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)
    dragged = controller.run(
        "drag", snapshot_id=snapshot_id, x=100, y=100, end_x=300, end_y=200,
        modifier="window", screenshot_delay_seconds=0,
    )

    assert "alt modifier" in dragged.message.lower()
    assert commands == [
        ["ydotool", "key", "56:1"],
        ["ydotool", "mousemove", "--absolute", "100", "100"],
        ["ydotool", "click", "--next-delay", "60", "0x40"],
        ["ydotool", "mousemove", "--absolute", "300", "200"],
        ["ydotool", "click", "--next-delay", "60", "0x80"],
        ["ydotool", "key", "56:0"],
    ]


def test_hyprland_drag_uses_real_relative_motion_and_corrects_acceleration(monkeypatch):
    commands = []
    cursor_states = iter([
        {"x": 100, "y": 100},   # Exact compositor-positioned drag start.
        {"x": 280, "y": 200},   # Relative movement overshot the requested x=300 by 20.
        {"x": 300, "y": 200},   # Feedback correction reaches the requested endpoint.
    ])
    monkeypatch.setattr(computer.shutil, "which", lambda name: f"/usr/bin/{name}")

    def runner(args, **kwargs):
        commands.append(args)
        if args[:3] == ["hyprctl", "activewindow", "-j"]:
            return subprocess.CompletedProcess(args, 0, stdout='{"at":[0,0],"size":[1000,700]}', stderr="")
        if args[:3] == ["hyprctl", "cursorpos", "-j"]:
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(next(cursor_states)), stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "src.tools.desktop._hyprland_dispatch",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="ok", stderr=""),
    )
    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"WAYLAND_DISPLAY": "wayland-1", "HYPRLAND_INSTANCE_SIGNATURE": "instance"},
        ensure_wayland_daemon=lambda: True,
        screenshot_delay_seconds=0,
    )
    inspected = controller.run("inspect", scope="monitor", screenshot_delay_seconds=0)
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)
    controller.run(
        "drag", snapshot_id=snapshot_id, x=100, y=100, end_x=300, end_y=200,
        modifier="alt", screenshot_delay_seconds=0,
    )

    assert ["ydotool", "mousemove", "--", "22", "0"] in commands
    assert ["ydotool", "click", "--next-delay", "60", "0x40"] in commands
    assert ["ydotool", "click", "--next-delay", "60", "0x80"] in commands


def test_failed_post_action_screenshot_is_not_retried_as_a_second_action(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    captures = 0
    commands = []

    def screenshot():
        nonlocal captures
        captures += 1
        if captures > 1:
            raise TimeoutError("compositor capture timed out")
        return _png()

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=screenshot,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        screenshot_delay_seconds=0,
    )
    inspected = controller.run("inspect")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)
    result = controller.run("click", snapshot_id=snapshot_id, x=10, y=10)

    assert commands == [
        ["xdotool", "mousemove", "--sync", "10", "10"],
        ["xdotool", "click", "--delay", "80", "1"],
    ]
    assert captures == 2
    assert "screenshot refresh failed" in result.message.lower()
    assert "inspect again before another action" in result.message.lower()


def test_normalized_coordinates_outside_zero_to_1000_are_rejected(monkeypatch):
    monkeypatch.setattr(
        computer.shutil,
        "which",
        lambda name: {"ydotool": "/usr/bin/ydotool", "wtype": "/usr/bin/wtype", "swaymsg": "/usr/bin/swaymsg"}.get(name),
    )
    controller = computer.ComputerController(
        screenshot_fn=_png,
        environ={"WAYLAND_DISPLAY": "wayland-1", "SWAYSOCK": "/run/user/1000/sway-ipc.sock"},
        ensure_wayland_daemon=lambda: True,
        coordinate_mode="normalized_1000",
    )
    controller._read_active_window_state = lambda: ("sway:test-window", (0, 0, 1000, 700))
    inspected = controller.run("inspect")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)

    rejected = controller.run("click", snapshot_id=snapshot_id, x=1001, y=500)

    assert "normalized click coordinates must be between 0 and 1000" in rejected.message.lower()
    assert controller.snapshot_id == ""
    assert "no action token" in rejected.message.lower()


def test_snapshot_is_bound_to_the_focused_window(monkeypatch):
    monkeypatch.setattr(
        computer.shutil,
        "which",
        lambda name: {"ydotool": "/usr/bin/ydotool", "wtype": "/usr/bin/wtype", "swaymsg": "/usr/bin/swaymsg"}.get(name),
    )
    commands = []
    state = {"identity": "sway:window-1"}

    def runner(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"WAYLAND_DISPLAY": "wayland-1", "SWAYSOCK": "/run/user/1000/sway-ipc.sock"},
        ensure_wayland_daemon=lambda: True,
        screenshot_delay_seconds=0,
    )
    controller._read_active_window_state = lambda: (state["identity"], (0, 0, 1000, 700))
    inspected = controller.run("inspect", screenshot_delay_seconds=0)
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)

    state["identity"] = "sway:window-2"
    rejected = controller.run("type", snapshot_id=snapshot_id, text="must not leak to another window", screenshot_delay_seconds=0)

    assert "focused window changed" in rejected.message.lower()
    assert all("type" not in args for args in commands)
    assert controller.snapshot_id == ""


def test_hyprland_shortcuts_use_compositor_dispatcher(monkeypatch):
    monkeypatch.setattr(
        computer.shutil,
        "which",
        lambda name: {"ydotool": "/usr/bin/ydotool", "wtype": "/usr/bin/wtype", "hyprctl": "/usr/bin/hyprctl"}.get(name),
    )
    from src.tools import desktop

    calls = []

    def dispatch(legacy, args, *, lua_expression):
        calls.append((legacy, args, lua_expression))
        return subprocess.CompletedProcess(["hyprctl"], 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(desktop, "_hyprland_dispatch", dispatch)
    commands = []

    def runner(args, **kwargs):
        if args[:2] == ["hyprctl", "activewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout='{"at":[0,0],"size":[1000,700]}', stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"WAYLAND_DISPLAY": "wayland-1", "HYPRLAND_INSTANCE_SIGNATURE": "test"},
        ensure_wayland_daemon=lambda: True,
    )
    inspected = controller.run("inspect")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)
    result = controller.run("press", snapshot_id=snapshot_id, key="ctrl+s")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", result.message).group(1)
    plus = controller.run("press", snapshot_id=snapshot_id, key="ctrl+plus")

    assert calls == [
        ("sendshortcut", "CTRL,S,", 'hl.dsp.send_shortcut({ mods = "CTRL", key = "S" })'),
        ("sendshortcut", "CTRL+SHIFT,EQUAL,", 'hl.dsp.send_shortcut({ mods = "CTRL+SHIFT", key = "EQUAL" })'),
    ]
    assert commands == []
    assert "Pressed ctrl+plus" in plus.message


def test_wayland_click_is_limited_to_active_window_and_blocks_close_corner(monkeypatch):
    available_commands = {"ydotool": "/usr/bin/ydotool", "wtype": "/usr/bin/wtype", "hyprctl": "/usr/bin/hyprctl"}
    monkeypatch.setattr(computer.shutil, "which", lambda name: available_commands.get(name))
    commands = []

    def runner(args, **kwargs):
        if args[:2] == ["hyprctl", "activewindow"]:
            return subprocess.CompletedProcess(
                args, 0, stdout='{"at":[100,50],"size":[500,300]}', stderr=""
            )
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=lambda: (_png(), (100, 50)),
        runner=runner,
        environ={"WAYLAND_DISPLAY": "wayland-1", "HYPRLAND_INSTANCE_SIGNATURE": "test", "DISPLAY": ":1"},
        ensure_wayland_daemon=lambda: True,
    )
    inspected = controller.run("inspect")
    assert "Active window bounds: x=0..499, y=0..299" in inspected.message
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)

    background = controller.run("click", snapshot_id=snapshot_id, x=600, y=100)
    assert "outside the active window" in background.message.lower()
    assert controller.snapshot_id == ""
    snapshot_id = re.search(r"Snapshot ID: (\w+)", controller.run("inspect").message).group(1)

    close_corner = controller.run("click", snapshot_id=snapshot_id, x=490, y=10)
    assert "window-close corner" in close_corner.message.lower()
    assert commands == []


def test_rejects_out_of_bounds_click_and_unsupported_key():
    controller = computer.ComputerController(screenshot_fn=_png, environ={"DISPLAY": ":1"})
    controller._read_active_window_state = lambda: ("x11:test-window", (0, 0, 640, 480))
    inspected = controller.run("inspect")
    snapshot_id = re.search(r"Snapshot ID: (\w+)", inspected.message).group(1)
    result = controller.run("click", snapshot_id=snapshot_id, x=1000, y=200)
    assert "coordinates must be within" in result.message.lower()

    assert controller.snapshot_id == ""
    snapshot_id = re.search(r"Snapshot ID: (\w+)", controller.run("inspect").message).group(1)
    unsupported = controller.run("press", snapshot_id=snapshot_id, key="alt+f4")
    assert "not allowed" in unsupported.message.lower()


def test_adam_brain_dispatches_computer_control_tool():
    import asyncio

    from src.llm.brain import AdamBrain

    class CustomTools:
        @staticmethod
        def has_tool(_name):
            return False

    class Controller:
        available = True

        def run(self, **kwargs):
            return computer.ComputerControlResult(f"controlled: {kwargs['action']}", _png())

    brain = object.__new__(AdamBrain)
    brain.speculative_router = None
    brain.custom_tool_mgr = CustomTools()
    brain.computer_controller = Controller()
    brain._pending_screenshot = None
    brain.messages = []
    response = asyncio.run(brain._execute_tool("computer_control", {"action": "inspect"}))
    assert isinstance(response, computer.ComputerControlResult)
    assert response.message == "controlled: inspect"
    assert brain._pending_screenshot == _png()
