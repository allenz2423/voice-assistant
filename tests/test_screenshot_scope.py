from io import BytesIO
import socket

import pytest
from PIL import Image

from src.tools import desktop


def _png(size=(100, 80), color=(20, 40, 60)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeBackend(desktop.BaseDesktopBackend):
    def __init__(self, name="test", image=None):
        super().__init__()
        self.name = name
        self.image = image or _png()
        self.capture_calls = 0
        self.desktop_calls = 0

    def capture_screenshot(self):
        self.capture_calls += 1
        return self.image

    def capture_desktop_screenshot(self):
        self.desktop_calls += 1
        return self.image


def _prepare(monkeypatch, backend):
    monkeypatch.setattr(desktop, "ensure_gui_environment", lambda: None)
    monkeypatch.setattr(desktop, "wait_for_application_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(desktop, "get_active_backend", lambda: backend)


def test_window_scope_returns_only_exact_focused_window_crop(monkeypatch):
    backend = FakeBackend()
    _prepare(monkeypatch, backend)
    monkeypatch.setattr(desktop, "_active_window_geometry", lambda: (10, 12, 30, 20))

    captured, origin = desktop.capture_screenshot_with_origin("window")

    with Image.open(BytesIO(captured)) as image:
        assert image.size == (30, 20)
    assert origin == (10, 12)


def test_window_scope_rejects_partial_window_crop(monkeypatch):
    backend = FakeBackend()
    _prepare(monkeypatch, backend)
    monkeypatch.setattr(desktop, "_active_window_geometry", lambda: (90, 70, 30, 20))

    with pytest.raises(RuntimeError, match="fully contained"):
        desktop.capture_screenshot_with_origin("window")


def test_window_scope_rescales_hidpi_window_geometry(monkeypatch):
    backend = FakeBackend(image=_png((200, 160)))
    backend.last_screenshot_scale = (2.0, 2.0)
    _prepare(monkeypatch, backend)
    monkeypatch.setattr(desktop, "_active_window_geometry", lambda: (10, 12, 30, 20))

    captured = desktop.capture_screenshot_with_origin("window")

    with Image.open(BytesIO(captured[0])) as image:
        assert image.size == (60, 40)
    assert captured[1] == (10, 12)
    assert captured.scale == (2.0, 2.0)


def test_monitor_scope_fails_closed_when_backend_cannot_prove_scope(monkeypatch):
    backend = FakeBackend(name="sway")
    _prepare(monkeypatch, backend)

    with pytest.raises(RuntimeError, match="not verified"):
        desktop.capture_screenshot_with_origin("monitor")
    assert backend.capture_calls == 0


def test_desktop_scope_calls_explicit_full_desktop_capture(monkeypatch):
    backend = FakeBackend()
    _prepare(monkeypatch, backend)

    captured = desktop.capture_screenshot_with_origin("desktop")

    assert captured[0] == backend.image
    assert captured[1] == (0, 0)
    assert backend.desktop_calls == 1
    assert backend.capture_calls == 0
    assert captured.backend == "test"
    assert captured.target == "full desktop (explicitly requested)"


def test_capture_rejects_invalid_png(monkeypatch):
    backend = FakeBackend(image=b"not a PNG")
    _prepare(monkeypatch, backend)

    with pytest.raises(RuntimeError, match="could not be decoded"):
        desktop.capture_screenshot_with_origin("desktop")


def test_hyprland_window_capture_rejects_focus_aba_events(monkeypatch):
    backend = FakeBackend(name="hyprland", image=_png((20, 20)))
    monitor = ("1", "DP-1", 0, 0, 20, 20, 1.0)
    backend.last_screenshot_monitor_id = "1"
    backend.last_screenshot_target = "DP-1"
    backend.last_screenshot_origin = (0, 0)
    backend.last_screenshot_bounds = (0, 0, 20, 20)
    backend.last_screenshot_scale = (1.0, 1.0)
    monkeypatch.setattr(desktop, "ensure_gui_environment", lambda: None)
    monkeypatch.setattr(desktop, "wait_for_application_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(desktop, "get_active_backend", lambda **_kwargs: backend)
    snapshot = ("window-A", (0, 0, 10, 10), monitor)
    monkeypatch.setattr(desktop, "_active_window_snapshot", lambda **_kwargs: snapshot)
    monkeypatch.setattr(desktop, "_active_hyprland_monitor_snapshot", lambda **_kwargs: monitor)
    event_reader, event_writer = socket.socketpair()
    event_reader.setblocking(False)
    monkeypatch.setattr(
        desktop, "_connect_hyprland_focus_event_socket", lambda **_kwargs: event_reader
    )
    select_ready = desktop.select.select

    def finish_split_event(readers, writers, errors, timeout):
        event_writer.sendall(
            b"A\nactivewindowv2>>window-B\nactivewindowv2>>window-A\n"
        )
        return select_ready(readers, writers, errors, timeout)

    monkeypatch.setattr(desktop.select, "select", finish_split_event)

    def capture_with_focus_aba():
        event_writer.sendall(b"activewindowv2>>window-")
        return backend.image

    monkeypatch.setattr(backend, "capture_screenshot", capture_with_focus_aba)
    try:
        with pytest.raises(RuntimeError, match="focus or focused monitor changed"):
            desktop.capture_screenshot_with_origin("window")
    finally:
        event_writer.close()
