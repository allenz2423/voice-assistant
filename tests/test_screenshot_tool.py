from types import SimpleNamespace
from unittest.mock import patch
from io import BytesIO

import pytest
from PIL import Image


def _png_fixture(size=(100, 80)):
    output = BytesIO()
    Image.new("RGB", size, (30, 50, 70)).save(output, format="PNG")
    return output.getvalue()


PNG_FIXTURE = _png_fixture()


def test_capture_screenshot_returns_png_bytes():
    from src.tools import desktop

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command == ["hyprctl", "activewindow", "-j"]:
            return SimpleNamespace(returncode=0, stdout='{"monitor":1}', stderr=b"")
        if command == ["hyprctl", "monitors", "-j"]:
            return SimpleNamespace(
                returncode=0,
                stdout='[{"id":1,"name":"DP-5","x":0,"y":0,"width":100,"height":80}]',
                stderr=b"",
            )
        if command == ["grim", "-l", "1", "-o", "DP-5", "-"]:
            return SimpleNamespace(returncode=0, stdout=PNG_FIXTURE, stderr=b"")
        raise AssertionError(f"Unexpected screenshot command: {command}")

    with patch("src.tools.desktop.ensure_gui_environment"), patch(
        "src.tools.desktop.wait_for_application_ready"
    ), patch(
        "src.tools.desktop.get_active_backend", return_value=desktop.HyprlandBackend()
    ), patch(
        "src.tools.desktop.subprocess.run", side_effect=run
    ), patch("src.tools.desktop._active_window_geometry", return_value=(10, 10, 40, 30)):
        image = desktop.capture_screenshot()

    with Image.open(BytesIO(image)) as captured:
        assert captured.format == "PNG"
        assert captured.size == (40, 30)
    assert ["grim", "-l", "1", "-o", "DP-5", "-"] in calls
    assert ["grim", "-l", "1", "-"] not in calls


@pytest.mark.asyncio
async def test_screenshot_tool_attaches_image_to_followup_turn():
    from src.llm.brain import AdamBrain

    class DummyClient:
        def __init__(self):
            self.calls = []

        async def chat(self, messages, tools=None):
            self.calls.append([dict(message) for message in messages])
            if len(self.calls) == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "screenshot-call",
                        "function": {"name": "capture_screenshot", "arguments": {}},
                    }],
                }
            if len(self.calls) == 2:
                return {"content": "I can see the desktop.", "tool_calls": []}
            return {"content": "The desktop is open with several application windows.", "tool_calls": []}

        def format_tool_response(self, tool_call_id, tool_name, result):
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }

    class DummyTTS:
        def __init__(self):
            self.spoken = []

        async def speak_async(self, text):
            self.spoken.append(text)

    config = SimpleNamespace(
        llm=SimpleNamespace(
            provider="local",
            local_model="qwen3.5:4b",
            cloud_model="",
            ollama_host="http://localhost:11434",
            temperature=0.3,
            num_ctx=16384,
            think="low",
        )
    )
    brain = AdamBrain(
        config=config,
        supervisor=None,
        probe=None,
        confirmation_mgr=None,
        tts_engine=DummyTTS(),
    )
    brain.llm_client = DummyClient()

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Test desktop"), patch.object(
        brain.computer_controller, "run",
        return_value=SimpleNamespace(screenshot=PNG_FIXTURE, message="Snapshot ID: test-snapshot"),
    ):
        await brain.process_user_utterance("What is on my screen?")

    followup_messages = brain.llm_client.calls[1]
    image_message = followup_messages[-1]
    assert image_message["role"] == "user"
    assert image_message["images"] == [PNG_FIXTURE]
    assert followup_messages[-2]["role"] == "tool"
    assert brain.tts.spoken == [
        "I’ve got the request. I’m checking the screen now.",
        "I can see the desktop.",
    ]


@pytest.mark.asyncio
async def test_chat_inspection_does_not_stop_after_focusing_window():
    from src.llm.brain import AdamBrain

    class DummyClient:
        def __init__(self):
            self.calls = []

        async def chat(self, messages, tools=None):
            self.calls.append([dict(message) for message in messages])
            if len(self.calls) == 1:
                return {"content": "", "tool_calls": [{
                    "id": "focus-call",
                    "function": {"name": "focus_window", "arguments": {"target": "vesktop", "screenshot": True}},
                }]}
            if len(self.calls) == 2:
                return {"content": "Looking at the chat, I can see a discussion about the event.", "tool_calls": []}
            return {"content": "They're discussing the event.", "tool_calls": []}

        def format_tool_response(self, tool_call_id, tool_name, result):
            return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}

    class DummyTTS:
        def __init__(self):
            self.spoken = []

        async def speak_async(self, text):
            self.spoken.append(text)

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="local", local_model="qwen3.5:4b", cloud_model="",
        ollama_host="http://localhost:11434", temperature=0.3, num_ctx=16384, think="low",
    ), desktop=SimpleNamespace(default_browser="microsoft-edge-stable"))
    tts = DummyTTS()
    brain = AdamBrain(config, None, None, None, tts)
    brain.llm_client = DummyClient()

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Discord window"), patch(
        "src.llm.brain.focus_window", return_value="Focused 'Discord' on workspace 1."
    ), patch.object(
        brain.computer_controller, "run",
        return_value=SimpleNamespace(screenshot=PNG_FIXTURE, message="Snapshot ID: test-snapshot"),
    ) as capture:
        await brain.process_user_utterance("What is the Discord chat window showing?")

    assert brain.llm_client.calls[1][-1]["images"] == [PNG_FIXTURE]
    assert capture.call_args.kwargs["scope"] == "window"
    assert "after opening the browser" not in brain.llm_client.calls[1][-1]["content"]
    assert tts.spoken == [
        "I’ve got the request. I’m checking the screen now.",
        "Looking at the chat, I can see a discussion about the event.",
    ]
    assert len(brain.llm_client.calls) == 2
