import subprocess

import pytest

from src.tools import terminal_text
from src.tools import terminal_atspi_reader


def test_terminal_name_matching_recognizes_common_window_classes():
    assert terminal_text._matches_terminal("Alacritty") == "alacritty"
    assert terminal_text._matches_terminal("org.gnome.Terminal") == "gnome-terminal"
    assert terminal_text._matches_terminal("org.wezfurlong.wezterm") == "wezterm"
    assert terminal_text._matches_terminal("org.gnome.Ptyxis") == "ptyxis"
    assert terminal_text._matches_terminal("org.gnome.Console") == "gnome-console"
    expected = {
        "blackbox-terminal": "blackbox-terminal", "com.raggesilver.BlackBox": "blackbox",
        "deepin-terminal": "deepin-terminal",
        "lxterminal": "lxterminal", "mate-terminal": "mate-terminal",
        "qterminal": "qterminal", "rxvt": "rxvt", "URxvt": "urxvt", "sakura": "sakura",
        "st": "st", "terminology": "terminology", "tilda": "tilda",
        "guake": "guake", "yakuake": "yakuake",
    }
    for name, terminal in expected.items():
        assert terminal_text._matches_terminal(name) == terminal


def test_atspi_reader_calls_text_interface_for_ranges():
    class TextInterface:
        @staticmethod
        def get_text(node, start, end):
            return node[start:end]

    class Atspi:
        Text = TextInterface

    assert terminal_atspi_reader._read_text_range(Atspi, "sentinel output", 0, 8) == "sentinel"


def test_detect_terminal_reports_unsupported_alacritty_without_buffer_api(monkeypatch):
    focused = terminal_text.FocusedWindow("db-shell", "Alacritty", 42, "Hyprland")
    monkeypatch.setattr(terminal_text, "_focused_window", lambda: focused)
    monkeypatch.setattr(terminal_text, "_process_snapshot", lambda: ({42: {"comm": "alacritty", "cmdline": "alacritty"}}, {}))
    monkeypatch.setattr(terminal_text, "_tmux_client_for_focused_window", lambda *args: (None, None))
    monkeypatch.setattr(terminal_text, "_environment_for_focused_terminal", lambda *args: {})
    monkeypatch.setattr(terminal_text, "_probe_atspi", lambda detection: False)

    result = terminal_text.detect_terminal()

    assert result["terminal"] == "alacritty"
    assert result["available"] is False
    assert "No AT-SPI terminal text widget" in result["reason"]


def test_detect_terminal_reports_focused_tmux_pane(monkeypatch):
    focused = terminal_text.FocusedWindow("Alacritty", "Alacritty", 42, "Hyprland")
    monkeypatch.setattr(terminal_text, "_focused_window", lambda: focused)
    monkeypatch.setattr(terminal_text, "_process_snapshot", lambda: ({42: {"comm": "alacritty", "cmdline": "alacritty"}}, {}))
    monkeypatch.setattr(terminal_text, "_tmux_client_for_focused_window", lambda *args: ("/dev/pts/1", "%7"))

    result = terminal_text._detect_terminal(probe_accessibility=True)

    assert result["ok"] is True
    assert result["terminal"] == "alacritty"
    assert result["backend"] == "tmux"
    assert result["pane_id"] == "%7"
    assert result["client_name"] == "/dev/pts/1"


def test_read_tmux_limits_output_and_returns_untrusted_text(monkeypatch):
    monkeypatch.setattr(terminal_text.shutil, "which", lambda _: "/usr/bin/tmux")
    monkeypatch.setattr(
        terminal_text,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "a\nb\nc\n", ""),
    )

    result = terminal_text._read_tmux(
        {"terminal": "alacritty", "pane_id": "%4"}, "screen", 3
    )

    assert result["ok"] is True
    assert result["text"] == "c\n"
    assert result["truncated"] is True
    assert result["text_is_untrusted"] is True


def test_kitty_reader_uses_socket_without_password(monkeypatch):
    focused = terminal_text.FocusedWindow("kitty", "kitty", 42, "Hyprland")
    monkeypatch.setattr(terminal_text, "_focused_window", lambda: focused)
    monkeypatch.setattr(terminal_text, "_process_snapshot", lambda: ({}, {}))
    monkeypatch.setattr(
        terminal_text,
        "_environment_for_focused_terminal",
        lambda *_args: {"KITTY_LISTEN_ON": "unix:/run/user/1000/adam-kitty-42"},
    )
    monkeypatch.setattr(terminal_text.shutil, "which", lambda _: "/usr/bin/kitten")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "visible text\n", "")

    monkeypatch.setattr(terminal_text, "_run", run)

    result = terminal_text._read_kitty({"terminal": "kitty"}, "screen", 12000)

    assert result["ok"] is True
    assert result["text"] == "visible text\n"
    assert commands[0][commands[0].index("--to") + 1] == "unix:/run/user/1000/adam-kitty-42"
    assert "--use-password=never" in commands[0]
    assert "--password" not in commands[0]
    assert commands[0].index("--use-password=never") < commands[0].index("get-text")


def test_read_terminal_rejects_invalid_scope_without_accessing_desktop(monkeypatch):
    monkeypatch.setattr(terminal_text, "_detect_terminal", lambda **kwargs: (_ for _ in ()).throw(AssertionError()))

    result = terminal_text.read_terminal_text("clipboard")

    assert result["ok"] is False
    assert "screen, recent, or all" in result["error"]


@pytest.mark.parametrize(
    ("backend", "reader_name"),
    [
        ("tmux", "_read_tmux"),
        ("kitty", "_read_kitty"),
        ("wezterm", "_read_wezterm"),
        ("atspi", "_read_atspi"),
    ],
)
def test_read_terminal_routes_to_detected_reader(monkeypatch, backend, reader_name):
    detection = {"ok": True, "terminal": "detected-terminal", "backend": backend}
    monkeypatch.setattr(terminal_text, "_detect_terminal", lambda **kwargs: detection)
    calls = []

    def reader(name):
        def read(_detection, scope, max_chars):
            calls.append((name, scope, max_chars))
            return {"ok": True, "backend": name}

        return read

    for name in ("_read_tmux", "_read_kitty", "_read_wezterm", "_read_atspi"):
        monkeypatch.setattr(terminal_text, name, reader(name))

    result = terminal_text.read_terminal_text("recent", 321)

    assert result == {"ok": True, "backend": reader_name}
    assert calls == [(reader_name, "recent", 321)]


@pytest.mark.parametrize("backend,reader_name", [("kitty", "_read_kitty"), ("wezterm", "_read_wezterm")])
def test_native_reader_falls_back_to_atspi_when_unavailable(monkeypatch, backend, reader_name):
    detection = {"ok": True, "terminal": "detected-terminal", "backend": backend}
    monkeypatch.setattr(terminal_text, "_detect_terminal", lambda **kwargs: detection)
    monkeypatch.setattr(terminal_text, reader_name, lambda *args: None)
    monkeypatch.setattr(
        terminal_text,
        "_read_atspi",
        lambda *_args: {"ok": True, "backend": "atspi"},
    )

    result = terminal_text.read_terminal_text()

    assert result == {"ok": True, "backend": "atspi"}


def test_detection_does_not_trust_terminal_name_in_unrelated_window_title(monkeypatch):
    focused = terminal_text.FocusedWindow("Alacritty is a nice terminal", "firefox", 42, "X11")
    monkeypatch.setattr(terminal_text, "_focused_window", lambda: focused)
    monkeypatch.setattr(
        terminal_text,
        "_process_snapshot",
        lambda: ({42: {"ppid": 1, "comm": "firefox", "cmdline": "firefox"}}, {}),
    )

    result = terminal_text._detect_terminal(probe_accessibility=False)

    assert result["ok"] is False
    assert result["terminal"] is None
