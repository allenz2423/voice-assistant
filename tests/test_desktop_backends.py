import os
import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.tools.desktop import (
    detect_desktop_environment,
    get_active_backend,
    is_capability_supported,
    is_tool_enabled,
    get_active_capabilities,
    focus_window,
    workspace_control,
    close_application,
    list_windows,
    swap_windows,
    capture_screenshot,
    execute_desktop_macro,
    list_desktop_macros,
    configure_desktop_macros,
    configure_disabled_capabilities,
    KdePlasmaBackend,
    NiriBackend,
    GnomeBackend,
    CosmicBackend,
    HyprlandBackend,
    SwayBackend
)


def test_kde_plasma_capabilities_and_disabling(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.delenv("I3SOCK", raising=False)
    monkeypatch.delenv("NIRI_SOCKET", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setenv("KDE_SESSION_VERSION", "6")

    backend = get_active_backend(refresh_env=False)
    assert isinstance(backend, KdePlasmaBackend)

    # Window swap should be DISABLED on KDE Plasma
    assert backend.supports("window_swap") is False
    assert is_capability_supported("window_swap") is False
    assert is_tool_enabled("swap_windows") is False

    # Focus, move, switch, close, screenshot, macros should be ENABLED
    assert backend.supports("window_focus") is True
    assert backend.supports("workspace_switch") is True
    assert backend.supports("window_move") is True
    assert backend.supports("window_close") is True
    assert backend.supports("screenshot") is True
    assert backend.supports("desktop_macro") is True

    # Calling swap_windows on KDE returns disabled/unsupported message
    swap_res = swap_windows("Firefox", "Terminal")
    assert "not supported on kde_plasma" in swap_res.lower() or "not supported on kde" in swap_res.lower()


def test_kde_plasma_focus_window(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setenv("KDE_SESSION_VERSION", "6")

    # 1. With kdotool
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/kdotool" if x == "kdotool" else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res = focus_window("Firefox")
        assert "Focused window matching 'Firefox' via kdotool" in res
        assert mock_run.call_args[0][0] == ["kdotool", "search", "--name", "Firefox", "windowactivate"]

    # 2. Fallback to wmctrl
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/wmctrl" if x == "wmctrl" else None), \
         patch("subprocess.run") as mock_run:
        def run_side_effect(cmd, *args, **kwargs):
            if cmd == ["wmctrl", "-l", "-x"]:
                return MagicMock(returncode=0, stdout="0x123 0 firefox.Firefox myhost Firefox Browser\n", stderr="")
            elif cmd == ["wmctrl", "-i", "-a", "0x123"]:
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="")
        mock_run.side_effect = run_side_effect
        res = focus_window("Firefox")
        assert "Focused 'Firefox Browser' in KDE Plasma" in res


def test_kde_plasma_workspace_control(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setenv("KDE_SESSION_VERSION", "6")

    # Switch desktop via qdbus6
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/qdbus6" if x in ["qdbus6", "qdbus"] else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res = workspace_control("switch", "3")
        assert "Switched to virtual desktop 3 in KDE Plasma" in res
        mock_run.assert_called_with(["/usr/bin/qdbus6", "org.kde.KWin", "/KWin", "setCurrentDesktop", "3"], capture_output=True, text=True, timeout=2)

    # Next / Previous desktop
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/qdbus6" if x in ["qdbus6", "qdbus"] else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res_next = workspace_control("switch", "next")
        assert "Switched to next desktop in KDE Plasma" in res_next
        res_prev = workspace_control("switch", "previous")
        assert "Switched to previous desktop in KDE Plasma" in res_prev

    # Move targeted window via kdotool
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/kdotool" if x == "kdotool" else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res_move = workspace_control("move", "2", "Spotify")
        assert "Moved 'Spotify' to desktop 2 in KDE Plasma" in res_move

    # Move active window via qdbus6 shortcut
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/qdbus6" if x in ["qdbus6", "qdbus"] else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res_move_act = workspace_control("move", "2")
        assert "Moved active window to desktop 2 in KDE Plasma" in res_move_act


def test_kde_plasma_close_application(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setenv("KDE_SESSION_VERSION", "6")

    # Close active window via shortcut
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/qdbus6" if x in ["qdbus6", "qdbus"] else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res_act = close_application("this window")
        assert "Closed active window in KDE Plasma" in res_act

    # Close targeted window via kdotool
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/kdotool" if x == "kdotool" else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res_tgt = close_application("Firefox")
        assert "Closed window matching 'Firefox' in KDE Plasma" in res_tgt


def test_kde_plasma_default_macros(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setenv("KDE_SESSION_VERSION", "6")

    macros = list_desktop_macros()
    assert "overview" in macros
    assert "show_desktop" in macros
    assert "grid" in macros
    assert "night_mode" in macros
    assert "lock" in macros

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res = execute_desktop_macro("overview")
        assert "Triggered desktop macro 'overview'" in res


def test_user_defined_macros():
    configure_desktop_macros({
        "my_custom_split": "echo 'custom split action'",
        "work_setup": ["echo 'step1'", "echo 'step2'"]
    })

    macros = list_desktop_macros()
    assert "my_custom_split" in macros
    assert "work_setup" in macros

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res = execute_desktop_macro("my_custom_split")
        assert "Triggered desktop macro 'my_custom_split'" in res


def test_niri_environment_and_operations(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.setenv("NIRI_SOCKET", "/run/user/1000/niri.sock")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    with patch("shutil.which", side_effect=lambda x: "/usr/bin/niri" if x == "niri" else None):
        backend = get_active_backend(refresh_env=False)
        assert isinstance(backend, NiriBackend)
        assert backend.supports("window_swap") is True
        assert backend.supports("window_focus") is True

    # Focus window in Niri
    mock_niri_windows = json.dumps([
        {"id": 10, "app_id": "org.mozilla.firefox", "title": "Mozilla Firefox", "workspace_id": 1, "is_focused": False},
        {"id": 20, "app_id": "Alacritty", "title": "term", "workspace_id": 2, "is_focused": True}
    ])
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/niri" if x == "niri" else None), \
         patch("subprocess.run") as mock_run:
        def niri_side_effect(cmd, *args, **kwargs):
            if cmd == ["niri", "msg", "-j", "windows"]:
                return MagicMock(returncode=0, stdout=mock_niri_windows, stderr="")
            elif cmd == ["niri", "msg", "action", "focus-window", "--id", "10"]:
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        mock_run.side_effect = niri_side_effect

        res = focus_window("Firefox")
        assert "Focused 'Mozilla Firefox' in Niri" in res

    # Switch workspace in Niri
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/niri" if x == "niri" else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res_ws = workspace_control("switch", "2")
        assert "Switched to workspace 2 in Niri" in res_ws


def test_gnome_and_cosmic_swap_disabled(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)

    # GNOME
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    backend_gnome = GnomeBackend()
    assert backend_gnome.supports("window_swap") is False
    assert "not supported on gnome" in backend_gnome.swap_windows("A", "B").lower()

    # COSMIC
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "COSMIC")
    backend_cosmic = CosmicBackend()
    assert backend_cosmic.supports("window_swap") is False
    assert "not supported on cosmic" in backend_cosmic.swap_windows("A", "B").lower()


@pytest.mark.asyncio
async def test_brain_omits_unsupported_tools_on_kde(monkeypatch):
    from src.llm.brain import AdamBrain

    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setenv("KDE_SESSION_VERSION", "6")

    config = SimpleNamespace(
        llm=SimpleNamespace(
            provider="local",
            local_model="qwen3.5:4b",
            cloud_model="",
            ollama_host="http://localhost:11434",
            temperature=0.3,
            num_ctx=16384,
            think="low",
        ),
        execution=SimpleNamespace(downloads_dir="~/Downloads", workspace_dir="~/workspace")
    )
    brain = AdamBrain(config=config, supervisor=None, probe=None, confirmation_mgr=None, tts_engine=MagicMock())

    tools = brain.get_tools()
    tool_names = [getattr(t, "name", None) or t.get("name") for t in tools]

    # swap_windows must NOT be in tools for KDE
    assert "swap_windows" not in tool_names
    # focus_window, workspace_control, desktop_macro MUST be in tools
    assert "focus_window" in tool_names
    assert "workspace_control" in tool_names
    assert "desktop_macro" in tool_names


@pytest.mark.asyncio
async def test_brain_includes_swap_windows_on_hyprland(monkeypatch):
    from src.llm.brain import AdamBrain

    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "sig_123")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "Hyprland")

    config = SimpleNamespace(
        llm=SimpleNamespace(
            provider="local",
            local_model="qwen3.5:4b",
            cloud_model="",
            ollama_host="http://localhost:11434",
            temperature=0.3,
            num_ctx=16384,
            think="low",
        ),
        execution=SimpleNamespace(downloads_dir="~/Downloads", workspace_dir="~/workspace")
    )
    with patch("shutil.which", return_value="/usr/bin/hyprctl"):
        brain = AdamBrain(config=config, supervisor=None, probe=None, confirmation_mgr=None, tts_engine=MagicMock())
        tools = brain.get_tools()
        tool_names = [getattr(t, "name", None) or t.get("name") for t in tools]

        # swap_windows MUST be in tools for Hyprland
        assert "swap_windows" in tool_names
        assert "desktop_macro" in tool_names


def test_default_browser_configuration_and_resolution():
    from src.tools.desktop import (
        configure_default_browser,
        get_default_browser,
        _resolve_application_entry,
        APPLICATION_ALIASES,
        WINDOW_ALIASES,
    )

    configure_default_browser("microsoft-edge-stable")
    assert get_default_browser() == "microsoft-edge-stable"
    assert "microsoft-edge-stable" in APPLICATION_ALIASES["browser"]
    assert "microsoft-edge" in WINDOW_ALIASES["browser"]

    mock_apps = {
        "firefox": {"name": "Firefox", "exec": "/usr/bin/firefox", "desktop_id": "firefox.desktop"},
        "microsoft edge": {"name": "Microsoft Edge", "exec": "/usr/bin/microsoft-edge-stable", "desktop_id": "microsoft-edge.desktop"}
    }
    resolved = _resolve_application_entry("browser", mock_apps)
    assert resolved is not None
    assert "/usr/bin/microsoft-edge-stable" in resolved["exec"]

    # Test dynamic switch to firefox
    configure_default_browser("firefox")
    assert get_default_browser() == "firefox"
    resolved_ff = _resolve_application_entry("browser", mock_apps)
    assert resolved_ff is not None
    assert "/usr/bin/firefox" in resolved_ff["exec"]

    # Revert back to microsoft-edge-stable
    configure_default_browser("microsoft-edge-stable")


def test_desktop_config_loads_default_browser(tmp_path):
    from src.tools.desktop import load_desktop_config, get_default_browser

    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text("""
desktop:
  default_browser: "microsoft-edge-stable"
""")
    load_desktop_config(str(cfg_file))
    assert get_default_browser() == "microsoft-edge-stable"


def test_open_in_browser_search_and_url():
    from src.tools.desktop import open_in_browser

    with patch("src.tools.desktop.launch_application") as mock_launch:
        mock_launch.return_value = "Launched browser."

        # Search query
        open_in_browser("Santal 33")
        mock_launch.assert_called_with("browser", args="https://www.google.com/search?q=Santal+33")

        # Full URL
        open_in_browser("https://github.com")
        mock_launch.assert_called_with("browser", args="https://github.com")

        # Domain
        open_in_browser("youtube.com")
        mock_launch.assert_called_with("browser", args="https://youtube.com")

        # Empty
        open_in_browser("")
        mock_launch.assert_called_with("browser")


def test_close_browser_tab():
    from src.tools.desktop import close_browser_tab

    states = iter([
        {"id": "edge-1", "class": "microsoft-edge", "title": "Video - Personal - Microsoft Edge"},
        {"id": "edge-1", "class": "microsoft-edge", "title": "Home - Personal - Microsoft Edge"},
    ])
    with patch("src.tools.desktop.focus_window", return_value="Focused Microsoft Edge") as mock_focus, \
         patch("src.tools.desktop._active_window_metadata", side_effect=lambda: next(states)), \
         patch("src.tools.desktop._send_browser_shortcut", return_value=(True, "")) as mock_key:
        res = close_browser_tab("browser")
        assert "verified the active tab changed" in res
        mock_focus.assert_called_with("browser")
        mock_key.assert_called_once_with("w")


def test_close_browser_tab_finds_matching_title_before_closing():
    from src.tools.desktop import close_browser_tab

    states = iter([
        {"id": "edge-1", "class": "microsoft-edge", "title": "Music - Personal - Microsoft Edge and 2 more pages"},
        {"id": "edge-1", "class": "microsoft-edge", "title": "Hugging Face - Personal - Microsoft Edge and 2 more pages"},
        {"id": "edge-1", "class": "microsoft-edge", "title": "Docs - Personal - Microsoft Edge and 1 more pages"},
    ])
    with patch("src.tools.desktop.focus_window", return_value="Focused Microsoft Edge"), \
         patch("src.tools.desktop._active_window_metadata", side_effect=lambda: next(states)), \
         patch("src.tools.desktop._send_browser_shortcut", return_value=(True, "")) as mock_key:
        res = close_browser_tab("edge", title_contains="hugging face")
    assert "Closed browser tab 'Hugging Face'" in res
    assert [call.args[0] for call in mock_key.call_args_list] == ["Tab", "w"]


def test_close_browser_tab_closes_all_matching_tabs_with_verification():
    from src.tools.desktop import close_browser_tab

    states = iter([
        {"id": "edge-1", "class": "microsoft-edge", "title": "Music - Personal - Microsoft Edge and 3 more pages"},
        {"id": "edge-1", "class": "microsoft-edge", "title": "Hugging Face A - Personal - Microsoft Edge and 3 more pages"},
        {"id": "edge-1", "class": "microsoft-edge", "title": "Hugging Face B - Personal - Microsoft Edge and 2 more pages"},
        {"id": "edge-1", "class": "microsoft-edge", "title": "Music - Personal - Microsoft Edge and 1 more pages"},
        {"id": "edge-1", "class": "microsoft-edge", "title": "Music - Personal - Microsoft Edge and 1 more pages"},
    ])
    with patch("src.tools.desktop.focus_window", return_value="Focused Microsoft Edge"), \
         patch("src.tools.desktop._active_window_metadata", side_effect=lambda: next(states)), \
         patch("src.tools.desktop._send_browser_shortcut", return_value=(True, "")) as mock_key:
        res = close_browser_tab("edge", title_contains="hugging face", all_matches=True)
    assert "Closed and verified 2 tab(s)" in res
    assert [call.args[0] for call in mock_key.call_args_list] == ["Tab", "w", "w", "Tab"]


def test_close_browser_tab_retries_when_first_shortcut_does_not_change_title():
    from src.tools.desktop import close_browser_tab

    same = {"id": "edge-1", "class": "microsoft-edge", "title": "Music - Personal - Microsoft Edge"}
    next_tab = {"id": "edge-1", "class": "microsoft-edge", "title": "Home - Personal - Microsoft Edge"}
    states = iter([same, same, next_tab])
    with patch("src.tools.desktop.focus_window", return_value="Focused Microsoft Edge"), \
         patch("src.tools.desktop._active_window_metadata", side_effect=lambda: next(states)), \
         patch("src.tools.desktop._send_browser_shortcut", return_value=(True, "")) as mock_key, \
         patch("src.tools.desktop.time.sleep"):
        result = close_browser_tab("edge")

    assert "verified the active tab changed" in result
    assert [call.args[0] for call in mock_key.call_args_list] == ["w", "w"]


def test_close_browser_tab_does_not_claim_success_after_two_ineffective_shortcuts():
    from src.tools.desktop import close_browser_tab

    same = {"id": "edge-1", "class": "microsoft-edge", "title": "Music - Personal - Microsoft Edge"}
    states = iter([same, same, same])
    with patch("src.tools.desktop.focus_window", return_value="Focused Microsoft Edge"), \
         patch("src.tools.desktop._active_window_metadata", side_effect=lambda: next(states)), \
         patch("src.tools.desktop._send_browser_shortcut", return_value=(True, "")) as mock_key, \
         patch("src.tools.desktop.time.sleep"):
        result = close_browser_tab("edge")

    assert "sent twice" in result
    assert mock_key.call_count == 2


def test_hyprland_screenshot_uses_focused_output_name(monkeypatch):
    from types import SimpleNamespace

    screenshot = b"\x89PNG\r\n\x1a\nfixture"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command == ["hyprctl", "activewindow", "-j"]:
            return SimpleNamespace(returncode=0, stdout='{"monitor":1}', stderr=b"")
        if command == ["hyprctl", "monitors", "-j"]:
            return SimpleNamespace(returncode=0, stdout='[{"id":1,"name":"DP-5","x":0,"y":0,"width":2560,"height":1440}]', stderr=b"")
        if command == ["grim", "-l", "0", "-o", "DP-5", "-"]:
            return SimpleNamespace(returncode=0, stdout=screenshot, stderr=b"")
        raise AssertionError(f"Unexpected screenshot command: {command}")

    monkeypatch.setattr("src.tools.desktop.subprocess.run", run)
    image = HyprlandBackend().capture_screenshot()

    assert image == screenshot
    assert calls[-1] == ["grim", "-l", "0", "-o", "DP-5", "-"]
