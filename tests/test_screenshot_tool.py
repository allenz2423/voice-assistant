from types import SimpleNamespace
from unittest.mock import patch

import pytest


PNG_FIXTURE = b"\x89PNG\r\n\x1a\nfixture"


def test_capture_screenshot_returns_png_bytes():
    from src.tools import desktop

    with patch("src.tools.desktop.ensure_gui_environment"), patch(
        "src.tools.desktop.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout=PNG_FIXTURE, stderr=b""),
    ) as run:
        assert desktop.capture_screenshot() == PNG_FIXTURE

    run.assert_called_once_with(
        ["grim", "-"], capture_output=True, timeout=15, env=desktop.os.environ
    )


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

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Test desktop"), patch(
        "src.llm.brain.capture_screenshot", return_value=PNG_FIXTURE
    ):
        await brain.process_user_utterance("What is on my screen?")

    followup_messages = brain.llm_client.calls[1]
    image_message = followup_messages[-1]
    assert image_message["role"] == "user"
    assert image_message["images"] == [PNG_FIXTURE]
    assert followup_messages[-2]["role"] == "tool"
    assert brain.tts.spoken == ["The desktop is open with several application windows."]


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
                    "function": {"name": "focus_window", "arguments": {"target": "vesktop"}},
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
    ))
    tts = DummyTTS()
    brain = AdamBrain(config, None, None, None, tts)
    brain.llm_client = DummyClient()

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Discord window"), patch(
        "src.llm.brain.focus_window", return_value="Focused 'Discord' on workspace 1."
    ), patch("src.llm.brain.capture_screenshot", return_value=PNG_FIXTURE):
        await brain.process_user_utterance("What's in the Discord mod chat?")

    assert brain.llm_client.calls[1][-1]["images"] == [PNG_FIXTURE]
    assert tts.spoken == ["They're discussing the event."]
