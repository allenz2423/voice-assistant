from __future__ import annotations

import re
import json
import struct
import subprocess
import threading
from io import BytesIO

import pytest
from PIL import Image, ImageDraw, ImageFont

import src.tools.computer_control as computer


def _png(width: int = 1000, height: int = 700) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00" + b"\x00\x00\x00\x00"


def _valid_png(width: int = 1000, height: int = 700) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def test_model_requested_screenshot_wait_is_capped_by_configured_app_delay(monkeypatch):
    waits = []
    monkeypatch.setattr(computer, "screenshot_delay_for_focused_window", lambda *_args: 0.25)
    monkeypatch.setattr(computer.time, "sleep", waits.append)
    controller = computer.ComputerController(screenshot_fn=_png, screenshot_delay_seconds=0.25)
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))

    result = controller.run("inspect", scope="window", screenshot_delay_seconds=8)

    assert result.status == "ok"
    assert waits == [0.25]


def test_browser_inspection_waits_for_app_but_input_refresh_is_fast_by_default(monkeypatch):
    waits = []
    monkeypatch.setattr(computer, "screenshot_delay_for_focused_window", lambda *_args: 3.0)
    monkeypatch.setattr(computer.time, "sleep", waits.append)
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    controller = computer.ComputerController(
        screenshot_fn=_valid_png,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        screenshot_delay_seconds=0.25,
        browser_screenshot_delay_seconds=3.0,
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))

    inspected = controller.run("inspect")
    assert inspected.status == "ok"
    assert waits == [3.0]

    waits.clear()
    clicked = controller.run("click", snapshot_id=inspected.snapshot_id, x=50, y=50)
    assert clicked.status == "ok"
    assert waits == [0.25]


def test_visual_mode_clicks_unambiguous_ocr_label_without_model_coordinates(monkeypatch):
    region = computer.OCRRegion("O1", "Tomorrow · Oct 5", 0.99, 115, 383, 461, 429)

    class Reader:
        def read(self, _image):
            return [region]

    controller = computer.ComputerController(
        screenshot_fn=_valid_png,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        ocr_reader=Reader(),
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))
    monkeypatch.setattr(computer.ComputerController, "available", property(lambda _self: True))
    clicks = []
    monkeypatch.setattr(controller, "_click", lambda x, y, button: clicks.append((x, y, button)))

    inspected = controller.run("inspect", include_ocr=True)
    clicked = controller.run(
        "click", snapshot_id=inspected.snapshot_id, target_text="tomorrow oct 5",
    )

    assert clicks == [(288, 406, "left")]
    assert "Matched visible OCR label" in clicked.message


def test_visual_mode_refuses_ambiguous_ocr_label_click(monkeypatch):
    regions = [
        computer.OCRRegion("O1", "5:00 PM", 0.99, 100, 200, 200, 240),
        computer.OCRRegion("O2", "5:00 PM", 0.99, 300, 200, 400, 240),
    ]

    class Reader:
        def read(self, _image):
            return regions

    controller = computer.ComputerController(
        screenshot_fn=_valid_png,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        ocr_reader=Reader(),
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))
    monkeypatch.setattr(computer.ComputerController, "available", property(lambda _self: True))
    clicks = []
    monkeypatch.setattr(controller, "_click", lambda x, y, button: clicks.append((x, y, button)))

    inspected = controller.run("inspect", include_ocr=True)
    result = controller.run("click", snapshot_id=inspected.snapshot_id, target_text="5:00 PM")

    assert result.status == "invalid_input"
    assert "ambiguous" in result.message
    assert clicks == []


def test_visual_mode_can_ocr_and_click_label_in_one_action(monkeypatch):
    region = computer.OCRRegion("O1", "Tomorrow · Oct 5", 0.99, 115, 383, 461, 429)
    calls = []

    class Reader:
        def read(self, image):
            calls.append(image)
            return [region]

    controller = computer.ComputerController(
        screenshot_fn=_valid_png,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        ocr_reader=Reader(),
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))
    monkeypatch.setattr(computer.ComputerController, "available", property(lambda _self: True))
    clicks = []
    monkeypatch.setattr(controller, "_click", lambda x, y, button: clicks.append((x, y, button)))

    inspected = controller.run("inspect", include_ocr=False)
    clicked = controller.run(
        "click", snapshot_id=inspected.snapshot_id,
        target_text="Tomorrow Oct 5", include_ocr=True,
    )

    assert clicks == [(288, 406, "left")]
    assert len(calls) == 2  # One read of the inspected image and one post-action read.
    assert "Matched visible OCR label" in clicked.message


def test_sequence_can_re_resolve_multiple_ocr_text_clicks_from_fresh_screens(monkeypatch):
    regions = [
        computer.OCRRegion("O1", "Tomorrow · Oct 5", 0.99, 115, 383, 461, 429),
        computer.OCRRegion("O2", "2 people", 0.99, 112, 632, 295, 690),
        computer.OCRRegion("O3", "5:00 PM", 0.99, 375, 887, 548, 932),
    ]
    reads = []

    class Reader:
        def read(self, image):
            reads.append(image)
            return regions

    controller = computer.ComputerController(
        screenshot_fn=_valid_png,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        ocr_reader=Reader(),
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))
    monkeypatch.setattr(computer.ComputerController, "available", property(lambda _self: True))
    clicks = []
    monkeypatch.setattr(controller, "_click", lambda x, y, button: clicks.append((x, y, button)))
    inspected = controller.run("inspect", include_ocr=True)

    result = controller.run_sequence(
        snapshot_id=inspected.snapshot_id,
        include_ocr=True,
        actions=[
            {"action": "click", "target_text": "Tomorrow Oct 5"},
            {"action": "click", "target_text": "2 people"},
            {"action": "click", "target_text": "5:00 PM"},
        ],
    )

    assert result.status == "ok"
    assert clicks == [(288, 406, "left"), (203, 661, "left"), (461, 909, "left")]
    assert len(reads) == 4  # Initial OCR plus fresh OCR after each click.


def test_sequence_can_retarget_fresh_ocr_after_a_coordinate_click(monkeypatch):
    regions = [computer.OCRRegion("O1", "100", 0.99, 416, 419, 458, 451)]
    reads = []

    class Reader:
        def read(self, image):
            reads.append(image)
            return regions

    controller = computer.ComputerController(
        screenshot_fn=_valid_png,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        ocr_reader=Reader(),
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))
    monkeypatch.setattr(computer.ComputerController, "available", property(lambda _self: True))
    clicks = []
    monkeypatch.setattr(controller, "_click", lambda x, y, button: clicks.append((x, y, button)))
    inspected = controller.run("inspect", include_ocr=True)

    result = controller.run_sequence(
        snapshot_id=inspected.snapshot_id,
        include_ocr=True,
        actions=[
            {"action": "click", "x": 435, "y": 259},
            {"action": "click", "target_text": "100"},
        ],
    )

    assert result.status == "ok"
    assert clicks == [(435, 259, "left"), (437, 435, "left")]
    assert len(reads) == 3  # The text target is resolved from OCR after the first click.


def test_sequence_can_type_between_fresh_ocr_target_clicks(monkeypatch):
    regions = [
        computer.OCRRegion("O1", "Enter a note", 0.99, 100, 100, 300, 140),
        computer.OCRRegion("O2", "Save note", 0.99, 100, 200, 280, 240),
    ]
    reads = []

    class Reader:
        def read(self, image):
            reads.append(image)
            return regions

    controller = computer.ComputerController(
        screenshot_fn=_valid_png,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        ocr_reader=Reader(),
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))
    monkeypatch.setattr(computer.ComputerController, "available", property(lambda _self: True))
    clicks = []
    typed = []
    monkeypatch.setattr(controller, "_click", lambda x, y, button: clicks.append((x, y, button)))
    monkeypatch.setattr(controller, "_type", lambda value: typed.append(value))
    inspected = controller.run("inspect", include_ocr=True)

    result = controller.run_sequence(
        snapshot_id=inspected.snapshot_id,
        include_ocr=True,
        actions=[
            {"action": "click", "target_text": "Enter a note"},
            {"action": "type", "text": "Call the dentist Tuesday at 2 pm"},
            {"action": "click", "target_text": "Save note"},
        ],
    )

    assert result.status == "ok"
    assert clicks == [(200, 120, "left"), (190, 220, "left")]
    assert typed == ["Call the dentist Tuesday at 2 pm"]
    assert len(reads) == 4  # Initial OCR plus fresh OCR after each dispatched step.


def test_sequence_can_replace_text_after_an_ocr_targeted_current_value(monkeypatch):
    regions = [computer.OCRRegion("O1", "100", 0.99, 416, 419, 458, 451)]
    reads = []

    class Reader:
        def read(self, image):
            reads.append(image)
            return regions

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
        screenshot_fn=_valid_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        ocr_reader=Reader(),
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))
    monkeypatch.setattr(computer.ComputerController, "available", property(lambda _self: True))
    clicks = []
    monkeypatch.setattr(controller, "_click", lambda x, y, button: clicks.append((x, y, button)))
    inspected = controller.run("inspect", include_ocr=True)

    result = controller.run_sequence(
        snapshot_id=inspected.snapshot_id,
        include_ocr=True,
        actions=[
            {"action": "click", "target_text": "100"},
            {"action": "press", "key": "Home"},
            {"action": "press", "key": "Shift+End"},
            {"action": "type", "text": "500"},
        ],
    )

    assert result.status == "ok"
    assert result.dispatched is True
    assert clicks == [(437, 435, "left")]
    assert [command for command in commands if command[:2] == ["xdotool", "key"]] == [
        ["xdotool", "key", "--clearmodifiers", "Home"],
        ["xdotool", "key", "--clearmodifiers", "shift+End"],
    ]
    assert commands[-1] == ["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", "500"]
    assert len(reads) == 5  # Initial OCR, then OCR lookup plus a fresh capture after each input.


def test_sequence_explains_linux_ctrl_a_replacement_failure_after_ocr_click(monkeypatch):
    class Reader:
        def read(self, _image):
            return [computer.OCRRegion("O1", "100", 0.99, 416, 419, 458, 451)]

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
        screenshot_fn=_valid_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        ocr_reader=Reader(),
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))
    monkeypatch.setattr(computer.ComputerController, "available", property(lambda _self: True))
    clicks = []
    monkeypatch.setattr(controller, "_click", lambda x, y, button: clicks.append((x, y, button)))
    inspected = controller.run("inspect", include_ocr=True)

    result = controller.run_sequence(
        snapshot_id=inspected.snapshot_id,
        include_ocr=True,
        actions=[
            {"action": "click", "target_text": "100"},
            {"action": "press", "key": "Ctrl+A"},
            {"action": "type", "text": "500"},
        ],
    )

    assert result.status == "partial"
    assert "Ctrl+A was not sent" in result.message
    assert "press Home, press Shift+End" in result.message
    assert clicks == [(437, 435, "left")]
    assert not any(command[:2] == ["xdotool", "key"] for command in commands)


def test_small_status_text_change_counts_as_visible_progress():
    before = Image.new("RGB", (1440, 1800), "#fbfbfb")
    after = before.copy()
    font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 22)
    ImageDraw.Draw(before).text((66, 1037), "Request not placed.", fill="#075", font=font)
    ImageDraw.Draw(after).text(
        (66, 1037), "Selected: 2 people · 5:00 PM. Not placed.", fill="#075", font=font
    )
    before_bytes, after_bytes = BytesIO(), BytesIO()
    before.save(before_bytes, format="PNG")
    after.save(after_bytes, format="PNG")

    assert not computer._screens_visually_unchanged(before_bytes.getvalue(), after_bytes.getvalue())


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
        ["xdotool", "mousemove", "10", "20"],
        ["xdotool", "click", "--delay", "80", "1"],
    ]

    stale_input = controller.run("type", snapshot_id=current_id, text="This must be rejected as stale")
    assert "latest observation" in stale_input.message.lower()


def test_click_reports_when_fresh_screenshot_is_visually_unchanged(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = computer.ComputerController(
        screenshot_fn=_valid_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
    )
    inspected = controller.run("inspect", screenshot_delay_seconds=0)
    clicked = controller.run("click", snapshot_id=inspected.snapshot_id, x=10, y=20, screenshot_delay_seconds=0)

    assert clicked.status == "ok"
    assert clicked.dispatched is True
    assert "screen appears unchanged" in clicked.message
    assert "effect is not visually confirmed" in clicked.message


def test_click_treats_changed_ocr_text_as_visible_progress(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    class Reader:
        def __init__(self):
            self.reads = 0

        def read(self, _image):
            self.reads += 1
            text = "Normal" if self.reads == 1 else "Quiet"
            return [computer.OCRRegion("O1", text, 0.99, 100, 100, 180, 130)]

    controller = computer.ComputerController(
        screenshot_fn=_valid_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        ocr_reader=Reader(),
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 1000, 700))
    monkeypatch.setattr(computer.ComputerController, "available", property(lambda _self: True))
    inspected = controller.run("inspect", include_ocr=True, screenshot_delay_seconds=0)

    clicked = controller.run(
        "click", snapshot_id=inspected.snapshot_id, x=120, y=115,
        include_ocr=True, screenshot_delay_seconds=0,
    )

    assert clicked.status == "ok"
    assert "Quiet" in clicked.message
    assert "screen appears unchanged" not in clicked.message


def test_missing_ocr_comparison_does_not_claim_that_the_screen_was_unchanged():
    screenshot = _valid_png()

    assert computer._screen_state_unchanged(
        screenshot, screenshot, None, ("quiet",)
    ) is False
    assert computer._screen_state_unchanged(
        screenshot, screenshot, ("normal",), None
    ) is False
    assert computer._screen_state_unchanged(
        screenshot, screenshot, ("quiet",), ("quiet",)
    ) is True


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
        ["xdotool", "mousemove", "100", "80"],
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


def test_sequence_allows_wait_step_and_captures_fresh_observation(monkeypatch):
    monkeypatch.setattr(computer.shutil, "which", lambda name: "/usr/bin/xdotool" if name == "xdotool" else None)
    commands = []

    def runner(args, **kwargs):
        if args == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(args, 0, stdout="123", stderr="")
        if args[:3] == ["xdotool", "getactivewindow", "getwindowgeometry"]:
            return subprocess.CompletedProcess(args, 0, stdout="WINDOW=123\nX=0\nY=0\nWIDTH=1000\nHEIGHT=700\n", stderr="")
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    slept = []
    monkeypatch.setattr(computer.time, "sleep", lambda s: slept.append(s))

    controller = computer.ComputerController(
        screenshot_fn=_png,
        runner=runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
    )
    snapshot_id = controller.run("inspect").snapshot_id
    result = controller.run_sequence(
        snapshot_id=snapshot_id,
        actions=[
            {"action": "click", "x": 100, "y": 80},
            {"action": "type", "text": "deposit"},
            {"action": "press", "key": "enter"},
            {"action": "wait", "seconds": 3},
        ],
    )

    assert result.status == "ok"
    assert 3.0 in slept
    assert "Step 4/4 (wait): ok; Waited 3s." in result.message
    assert result.snapshot_id


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
    scroll_id = re.search(r"Snapshot ID: (\w+)", pressed.message).group(1)
    scrolled = controller.run("scroll", snapshot_id=scroll_id, direction="down", amount=3, x=150, y=250)

    assert commands[:2] == [
        ["ydotool", "mousemove", "--absolute", "99", "200"],
        ["ydotool", "click", "--next-delay", "80", "0xC0"],
    ]
    assert commands[2] == ["wtype", "--", "local sample"]
    assert commands[3] == ["wtype", "-M", "ctrl", "a", "-m", "ctrl"]
    assert commands[4:6] == [
        ["ydotool", "mousemove", "--absolute", "150", "250"],
        ["ydotool", "mousemove", "-w", "--", "0", "-3"],
    ]
    assert "screenshot" in scrolled.message.lower()


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
    assert "normalized values from 0 to 1000" in inspected.message

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
    assert commands[:2] == [
        ["xdotool", "mousemove", "200", "20"],
        ["xdotool", "mousedown", "1"],
    ]
    assert commands[-1] == ["xdotool", "mouseup", "1"]
    moves = [command for command in commands if command[:2] == ["xdotool", "mousemove"]]
    assert moves[-1] == ["xdotool", "mousemove", "500", "200"]
    assert all("--duration" not in command and "--sync" not in command for command in moves)


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
        ["xdotool", "mousemove", "10", "10"],
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


def test_hidpi_screenshot_coordinates_map_back_to_compositor_space():
    controller = computer.ComputerController(screenshot_fn=_png, environ={"DISPLAY": ":1"})
    controller._capture_scale = (2.0, 2.0)
    controller._origin_x, controller._origin_y = (100, 50)

    assert controller._screenshot_to_desktop(60, 40) == (130, 70)


def test_full_desktop_inspection_does_not_issue_input_snapshot():
    controller = computer.ComputerController(screenshot_fn=_png, environ={"DISPLAY": ":1"})
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 640, 480))
    inspected = controller.run("inspect", scope="desktop")

    assert "read-only" in inspected.message
    assert "Snapshot ID:" not in inspected.message
    assert inspected.snapshot_id == ""
