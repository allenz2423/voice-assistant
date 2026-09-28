import os
import json
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from src.skills import SkillManager
from src.tools.desktop import (
    workspace_control,
    focus_window,
    close_application,
    list_windows,
    launch_application,
    swap_windows,
)

@pytest.fixture
def skill_mgr():
    return SkillManager()

def test_skills_discovery_and_content(skill_mgr):
    skills = skill_mgr.list_skills()
    skill_ids = [s["id"] for s in skills]
    expected = ["hyprland", "sway", "i3", "kde_plasma", "gnome", "cosmic", "generic_desktop"]
    for exp in expected:
        assert exp in skill_ids
        content = skill_mgr.load_skill(exp)
        assert content is not None
        assert len(content) > 100
        assert "# Skill:" in content

def test_skill_aliases(skill_mgr):
    assert skill_mgr.load_skill("plasma") == skill_mgr.load_skill("kde_plasma")
    assert skill_mgr.load_skill("kde") == skill_mgr.load_skill("kde_plasma")
    assert skill_mgr.load_skill("kwin") == skill_mgr.load_skill("kde_plasma")
    assert skill_mgr.load_skill("swaywm") == skill_mgr.load_skill("sway")
    assert skill_mgr.load_skill("i3wm") == skill_mgr.load_skill("i3")
    assert skill_mgr.load_skill("gnome-shell") == skill_mgr.load_skill("gnome")

def test_detect_environment_hyprland(monkeypatch):
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "test_sig_123")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    sm = SkillManager()
    assert sm.detect_desktop_environment(refresh_env=False) == "hyprland"

def test_detect_environment_sway(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.setenv("SWAYSOCK", "/run/user/1000/sway-ipc.sock")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "sway")
    sm = SkillManager()
    assert sm.detect_desktop_environment(refresh_env=False) == "sway"

def test_detect_environment_i3(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.setenv("I3SOCK", "/run/user/1000/i3/ipc.sock")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "i3")
    sm = SkillManager()
    assert sm.detect_desktop_environment(refresh_env=False) == "i3"

def test_detect_environment_kde(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.delenv("I3SOCK", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setenv("KDE_SESSION_VERSION", "6")
    sm = SkillManager()
    assert sm.detect_desktop_environment(refresh_env=False) == "kde_plasma"

def test_detect_environment_gnome(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.delenv("I3SOCK", raising=False)
    monkeypatch.delenv("KDE_SESSION_VERSION", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    sm = SkillManager()
    assert sm.detect_desktop_environment(refresh_env=False) == "gnome"

def test_detect_environment_cosmic(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.delenv("I3SOCK", raising=False)
    monkeypatch.delenv("KDE_SESSION_VERSION", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "COSMIC")
    sm = SkillManager()
    assert sm.detect_desktop_environment(refresh_env=False) == "cosmic"

def test_detect_environment_generic(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.delenv("I3SOCK", raising=False)
    monkeypatch.delenv("KDE_SESSION_VERSION", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "XFCE")
    monkeypatch.setenv("DESKTOP_SESSION", "xfce")
    with patch("shutil.which", return_value=None):
        sm = SkillManager()
        assert sm.detect_desktop_environment(refresh_env=False) == "generic_desktop"

def test_custom_user_skills_override(tmp_path):
    custom_dir = tmp_path / "custom_skills"
    custom_dir.mkdir()
    custom_hypr = custom_dir / "hyprland.md"
    custom_hypr.write_text("# Skill: Custom Hyprland User Override\nSpecial custom instructions.", encoding="utf-8")

    sm = SkillManager(custom_skills_dir=custom_dir)
    loaded = sm.load_skill("hyprland")
    assert "Custom Hyprland User Override" in loaded

def test_active_de_context_format(skill_mgr):
    context = skill_mgr.get_active_de_context()
    assert "=== ACTIVE DESKTOP & WINDOW MANAGER SKILL" in context
    assert "=== END DESKTOP SKILL ===" in context

@pytest.mark.asyncio
async def test_brain_tool_execution_for_skills():
    from src.llm.brain import AdamBrain
    from types import SimpleNamespace

    config = SimpleNamespace(
        llm=SimpleNamespace(
            provider="ollama",
            local_model="gemma4:e4b",
            cloud_model="gemini-2.5-flash",
            ollama_host="http://localhost:11434",
            temperature=0.2,
            num_ctx=16384
        ),
        execution=SimpleNamespace(downloads_dir="~/Downloads", workspace_dir="~/workspace")
    )
    brain = AdamBrain(
        config=config,
        supervisor=MagicMock(),
        probe=MagicMock(),
        confirmation_mgr=MagicMock(),
        tts_engine=MagicMock()
    )

    list_res = await brain._execute_tool("list_skills", {})
    assert "Installed desktop skills" in list_res
    assert "hyprland" in list_res

    get_res = await brain._execute_tool("get_skill_context", {"skill_name": "hyprland"})
    assert "Skill 'hyprland'" in get_res or "ACTIVE DESKTOP" in get_res

    missing_res = await brain._execute_tool("get_skill_context", {"skill_name": "non_existent_wm"})
    assert "not found" in missing_res

def test_sway_workspace_and_focus(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.setenv("SWAYSOCK", "/run/user/1000/sway.sock")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    with patch("shutil.which") as mock_which, patch("subprocess.run") as mock_run:
        def which_side_effect(cmd):
            if cmd == "swaymsg": return "/usr/bin/swaymsg"
            return None
        mock_which.side_effect = which_side_effect
        mock_run.return_value = MagicMock(returncode=0, stdout="true")

        res = workspace_control("switch", "3")
        assert "Switched to workspace 3 in Sway" in res

        res_move = workspace_control("move", "2")
        assert "Moved active window to workspace 2 in Sway" in res_move

        res_focus = focus_window("Alacritty")
        assert "Focused window matching 'Alacritty' in Sway" in res_focus

def test_i3_workspace_and_focus(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("I3SOCK", "/run/user/1000/i3.sock")
    monkeypatch.setenv("DISPLAY", ":0")

    with patch("shutil.which") as mock_which, patch("subprocess.run") as mock_run:
        def which_side_effect(cmd):
            if cmd == "i3-msg": return "/usr/bin/i3-msg"
            return None
        mock_which.side_effect = which_side_effect
        mock_run.return_value = MagicMock(returncode=0, stdout="true")

        res = workspace_control("switch", "4")
        assert "Switched to workspace 4 in i3" in res

        res_move = workspace_control("move", "1")
        assert "Moved active window to workspace 1 in i3" in res_move

        res_focus = focus_window("Firefox")
        assert "Focused window matching 'Firefox' in i3" in res_focus

def test_find_best_window_fuzzy_and_aliases():
    from src.tools.desktop import _find_best_window
    windows = [
        {"class": "microsoft-edge", "title": "YouTube - Personal - Microsoft\u200b Edge", "address": "0x123", "workspace": {"id": 2}},
        {"class": "vesktop", "title": "Discord | General", "address": "0x456", "workspace": {"id": 1}},
        {"class": "Spotify", "title": "Spotify Premium", "address": "0x789", "workspace": {"id": 9}},
        {"class": "Alacritty", "title": "bash - term", "address": "0xabc", "workspace": {"id": 2}}
    ]

    # Test exact normalized
    assert _find_best_window("microsoft-edge", windows)["address"] == "0x123"
    # Test title with zero-width space
    assert _find_best_window("Microsoft Edge", windows)["address"] == "0x123"
    # Test semantic alias browser
    assert _find_best_window("browser", windows)["address"] == "0x123"
    # Test semantic alias discord -> vesktop
    assert _find_best_window("discord", windows)["address"] == "0x456"
    # Test terminal
    assert _find_best_window("terminal", windows)["address"] == "0xabc"
    # Test direct address
    assert _find_best_window("0x789", windows)["address"] == "0x789"
    # Test non-existent
    assert _find_best_window("nonexistent_xyz", windows) is None

def test_hyprland_targeted_window_move(monkeypatch):
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "test_sig_123")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    mock_clients = json.dumps([
        {"class": "Alacritty", "title": "term", "address": "0x111", "workspace": {"id": 2}},
        {"class": "Spotify", "title": "Spotify Premium", "address": "0x222", "workspace": {"id": 9}},
    ])

    with patch("shutil.which", return_value="/usr/bin/hyprctl"), \
         patch("subprocess.run") as mock_run:
        def run_side_effect(cmd, *args, **kwargs):
            if cmd == ["hyprctl", "clients", "-j"]:
                return MagicMock(returncode=0, stdout=mock_clients)
            elif cmd == ["hyprctl", "activeworkspace", "-j"]:
                return MagicMock(returncode=0, stdout=json.dumps({"id": 3, "name": "3"}))
            elif cmd == ["hyprctl", "dispatch", "movetoworkspacesilent", "1,address:0x222"]:
                return MagicMock(returncode=0, stdout="ok\n")
            elif cmd == ["hyprctl", "dispatch", "movetoworkspacesilent", "3,address:0x222"]:
                return MagicMock(returncode=0, stdout="ok\n")
            return MagicMock(returncode=0, stdout="ok\n")

        mock_run.side_effect = run_side_effect

        res = workspace_control("move", "1", "Spotify")
        assert "Moved 'Spotify Premium' to workspace 1" in res

        # Verify word numbers and 'workspace' prefix
        res_prefix = workspace_control("move", "Workspace One", "Spotify")
        assert "Moved 'Spotify Premium' to workspace 1" in res_prefix

        # Verify 'current' resolution
        res_current = workspace_control("move", "current", "Spotify")
        assert "Moved 'Spotify Premium' to workspace 3" in res_current

        # Verify 'back' resolution (Spotify was previously on WS 9)
        res_back = workspace_control("move", "back", "Spotify")
        assert "Moved 'Spotify Premium' to workspace 9" in res_back


def test_hyprland_swap_named_windows(monkeypatch):
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "test_sig_123")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    mock_clients = json.dumps([
        {"class": "Firefox", "title": "Browser", "address": "0x111", "workspace": {"id": 2}},
        {"class": "Alacritty", "title": "Terminal", "address": "0x222", "workspace": {"id": 2}},
    ])

    with patch("shutil.which", return_value="/usr/bin/hyprctl"), patch("subprocess.run") as mock_run:
        def run_side_effect(cmd, *args, **kwargs):
            if cmd == ["hyprctl", "clients", "-j"]:
                return MagicMock(returncode=0, stdout=mock_clients)
            return MagicMock(returncode=0, stdout="ok\n", stderr="")

        mock_run.side_effect = run_side_effect
        result = swap_windows("Firefox", "Terminal")

    assert "Swapped the positions" in result
    assert mock_run.call_args_list[1].args[0] == [
        "hyprctl", "dispatch", "focuswindow", "address:0x111"
    ]
    assert mock_run.call_args_list[2].args[0] == [
        "hyprctl", "dispatch", "swapwindow", "address:0x222"
    ]


def test_hyprland_swap_refuses_windows_on_different_workspaces(monkeypatch):
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "test_sig_123")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    mock_clients = json.dumps([
        {"class": "Firefox", "title": "Browser", "address": "0x111", "workspace": {"id": 2}},
        {"class": "Alacritty", "title": "Terminal", "address": "0x222", "workspace": {"id": 3}},
    ])

    with patch("shutil.which", return_value="/usr/bin/hyprctl"), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_clients)
        result = swap_windows("Firefox", "Terminal")

    assert "same workspace" in result
    assert mock_run.call_count == 1

def test_sway_targeted_window_move(monkeypatch):
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.setenv("SWAYSOCK", "/run/user/1000/sway-ipc.sock")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    mock_tree = json.dumps({
        "type": "root",
        "nodes": [
            {
                "type": "workspace", "name": "1",
                "nodes": [
                    {"id": 42, "app_id": "Spotify", "name": "Spotify Premium"}
                ]
            }
        ]
    })

    with patch("shutil.which", side_effect=lambda x: "/usr/bin/swaymsg" if x == "swaymsg" else None), \
         patch("subprocess.run") as mock_run:
        def run_side_effect(cmd, *args, **kwargs):
            if cmd == ["swaymsg", "-t", "get_tree"]:
                return MagicMock(returncode=0, stdout=mock_tree)
            elif cmd == ["swaymsg", "[con_id=42] move container to workspace 2"]:
                return MagicMock(returncode=0, stdout="true")
            return MagicMock(returncode=0, stdout="true")

        mock_run.side_effect = run_side_effect

        res = workspace_control("move", "2", "Spotify")
        assert "Moved 'Spotify Premium' to workspace 2 in Sway" in res

@pytest.mark.asyncio
async def test_brain_fast_path_action_tools():
    from src.llm.brain import AdamBrain

    class DummyClient:
        def __init__(self):
            self.chat_calls = 0
        async def chat(self, messages, tools=None):
            self.chat_calls += 1
            if self.chat_calls == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "workspace_control",
                            "arguments": {"action": "move", "workspace": "1", "target": "Spotify"}
                        }
                    }]
                }
            return {"content": "I should not be called!", "tool_calls": []}
        def format_tool_response(self, tool_call_id, tool_name, result):
            return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}

    class DummyTTS:
        def __init__(self):
            self.spoken = []
        async def speak_async(self, text):
            self.spoken.append(text)

    dummy_config = SimpleNamespace(
        llm=SimpleNamespace(
            provider="local",
            local_model="qwen",
            cloud_model="gemini",
            ollama_host="http://localhost:11434",
            temperature=0.7,
            num_ctx=8192
        )
    )
    brain = AdamBrain(config=dummy_config, supervisor=None, probe=None, confirmation_mgr=None, tts_engine=DummyTTS())
    brain.llm_client = DummyClient()

    with patch("src.llm.brain.workspace_control", return_value="Moved 'Spotify Premium' to workspace 1."):
        await brain.process_user_utterance("could you move Spotify to Workspace One?")

    # Verify that fast-path spoke "Done." without calling chat a 2nd time!
    assert brain.tts.spoken == ["Done."]
    assert brain.llm_client.chat_calls == 1
    assert brain.messages[-1]["content"] == "Done."

@pytest.mark.asyncio
async def test_brain_reprompts_for_summary_instead_of_speaking_raw_tool_result():
    from src.llm.brain import AdamBrain

    class DummyClient:
        def __init__(self):
            self.chat_calls = 0

        async def chat(self, messages, tools=None):
            self.chat_calls += 1
            if self.chat_calls == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "get_now_playing", "arguments": {}},
                    }],
                }
            if self.chat_calls == 2:
                return {"content": "", "tool_calls": []}
            return {"content": "The browser video is paused.", "tool_calls": []}

        def format_tool_response(self, tool_call_id, tool_name, result):
            return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}

    class DummyTTS:
        def __init__(self):
            self.spoken = []

        async def speak_async(self, text):
            self.spoken.append(text)

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="local", local_model="qwen", cloud_model="", ollama_host="",
        temperature=0.3, num_ctx=4096,
    ))
    tts = DummyTTS()
    brain = AdamBrain(config, None, None, None, tts)
    brain.llm_client = DummyClient()

    async def fake_execute_tool(name, args):
        return "Currently playing: browser video (Paused)"

    brain._execute_tool = fake_execute_tool

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Test desktop"):
        await brain.process_user_utterance("What's on my browser right now?")

    assert tts.spoken == ["The browser video is paused."]
    assert brain.messages[-1]["content"] == tts.spoken[0]
    assert brain.llm_client.chat_calls == 3


@pytest.mark.asyncio
async def test_brain_uses_generic_fallback_if_summary_retry_is_empty():
    from src.llm.brain import AdamBrain

    class DummyClient:
        def __init__(self):
            self.chat_calls = 0

        async def chat(self, messages, tools=None):
            self.chat_calls += 1
            if self.chat_calls == 1:
                return {"content": "", "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "get_now_playing", "arguments": {}},
                }]}
            return {"content": "", "tool_calls": []}

        def format_tool_response(self, tool_call_id, tool_name, result):
            return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}

    class DummyTTS:
        def __init__(self):
            self.spoken = []

        async def speak_async(self, text):
            self.spoken.append(text)

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="local", local_model="qwen", cloud_model="", ollama_host="",
        temperature=0.3, num_ctx=4096,
    ))
    tts = DummyTTS()
    brain = AdamBrain(config, None, None, None, tts)
    brain.llm_client = DummyClient()
    brain._execute_tool = lambda name, args: None

    async def fake_execute_tool(name, args):
        return "SENSITIVE RAW RESULT THAT MUST NOT BE SPOKEN"

    brain._execute_tool = fake_execute_tool

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Test desktop"):
        await brain.process_user_utterance("What's on my browser right now?")

    assert tts.spoken == ["I got the result, but couldn't summarize it just now."]
    assert "SENSITIVE RAW RESULT" not in " ".join(tts.spoken)

def test_multilingual_wake_word_detection():
    from src.wake.engine import WakeWordDetector

    detector = WakeWordDetector(wake_word="hey adam", aliases=["ペアラン", "アダム"])
    assert detector.is_custom_mode is True

    # Test Japanese comma and punctuation
    matched, rem = detector.match_custom_wake_word("Adam、日本語をしゃべりますか？")
    assert matched is True
    assert rem == "日本語をしゃべりますか？"

    # Test Katakana phonetic alias
    matched, rem = detector.match_custom_wake_word("ペアラン、日本語をしゃ、ま喋りますか。")
    assert matched is True
    assert rem == "日本語をしゃ、ま喋りますか。"

    # Test Katakana Adam
    matched, rem = detector.match_custom_wake_word("アダム、日本語をしゃべりますか？")
    assert matched is True
    assert rem == "日本語をしゃべりますか？"

    # Test Hey Adam
    matched, rem = detector.match_custom_wake_word("Hey Adam, move it back")
    assert matched is True
    assert rem == "move it back"

    # Test unaddressed utterance
    matched, rem = detector.match_custom_wake_word("Do you know the way?")
    assert matched is False
