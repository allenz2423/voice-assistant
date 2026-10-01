import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _DummyTTS:
    def __init__(self):
        self.spoken = []

    async def speak_async(self, text):
        self.spoken.append(text)


class _DummyClient:
    def __init__(self, provider, responses):
        self.provider = provider
        self.responses = list(responses)
        self.requests = []

    async def chat(self, messages, tools=None):
        self.requests.append([dict(message) for message in messages])
        return self.responses.pop(0)

    def format_tool_response(self, tool_call_id, tool_name, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}


def _config(provider="local"):
    return SimpleNamespace(llm=SimpleNamespace(
        provider=provider,
        local_model="test-model",
        cloud_model="test-model",
        ollama_host="http://localhost:11434",
        api_base="https://example.test/v1",
        api_key="test-key",
        temperature=0,
        num_ctx=8192,
    ))


@pytest.mark.asyncio
async def test_confirmation_pause_correlates_every_call_in_the_tool_batch():
    from src.llm.brain import AdamBrain

    class Confirmation:
        async def request_confirmation(self, payload, question):
            self.payload = payload
            self.question = question

    tts = _DummyTTS()
    confirmation = Confirmation()
    brain = AdamBrain(_config(), None, None, confirmation, tts)
    brain.llm_client = _DummyClient("local", [{
        "content": "",
        "tool_calls": [
            {
                "id": "confirm-call-7",
                "function": {"name": "ask_user_confirmation", "arguments": {
                    "question": "Restart now?", "summary": "Restart system", "command": "systemctl reboot"
                }},
            },
            {
                "id": "time-call-9",
                "function": {"name": "get_current_time", "arguments": {"location": "local"}},
            },
        ],
    }])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Restart after telling me the time")

    assistant = next(message for message in brain.messages if message.get("tool_calls"))
    result_ids = {message.get("tool_call_id") for message in brain.messages if message.get("role") == "tool"}
    assert {call["id"] for call in assistant["tool_calls"]} == result_ids
    results = [json.loads(message["content"]) for message in brain.messages if message.get("role") == "tool"]
    assert results[0]["status"] == "returned"
    assert results[1]["status"] == "cancelled"
    assert results[1]["effect_status"] == "not_dispatched"


@pytest.mark.asyncio
async def test_openai_compatible_history_serializes_native_argument_objects():
    from src.llm.brain import AdamBrain

    tts = _DummyTTS()
    brain = AdamBrain(_config("custom"), None, None, None, tts)
    brain.llm_client = _DummyClient("custom", [
        {
            "content": "",
            "tool_calls": [{
                "id": "math-call-4",
                "function": {"name": "calculate_math", "arguments": {"expression": "6 * 7"}},
            }],
        },
        {"content": "The answer is 42.", "tool_calls": []},
    ])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"), patch(
        "src.llm.brain.calculate_math", return_value="42"
    ):
        await brain.process_user_utterance("What is six times seven?")

    assistant_call = next(message for message in brain.llm_client.requests[1] if message.get("tool_calls"))
    tool_call = assistant_call["tool_calls"][0]
    assert tool_call["id"] == "math-call-4"
    assert tool_call["function"]["arguments"] == '{"expression": "6 * 7"}'
    tool_result = next(message for message in brain.llm_client.requests[1] if message.get("role") == "tool")
    assert tool_result["tool_call_id"] == "math-call-4"


@pytest.mark.asyncio
async def test_agent_loop_supports_file_creation_then_readback(tmp_path):
    from src.llm.brain import AdamBrain

    target = tmp_path / "meeting-notes.txt"
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {
            "content": "",
            "tool_calls": [{
                "id": "create-notes",
                "function": {"name": "create_file", "arguments": {
                    "path": str(target), "content": "Decisions\n- Ship the prototype\n"
                }},
            }],
        },
        {
            "content": "",
            "tool_calls": [{
                "id": "read-notes",
                "function": {"name": "read_file", "arguments": {"path": str(target)}},
            }],
        },
        {"content": "I created the notes and confirmed their contents.", "tool_calls": []},
    ])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Create meeting notes and check what you wrote")

    assert target.read_text() == "Decisions\n- Ship the prototype\n"
    results = [message for message in brain.messages if message.get("role") == "tool"]
    assert [message["tool_call_id"] for message in results] == ["create-notes", "read-notes"]
    assert "Decisions" in results[1]["content"]


@pytest.mark.asyncio
async def test_file_write_failure_returns_failed_execution_status(tmp_path):
    from src.llm.brain import AdamBrain

    target = tmp_path / "existing.txt"
    target.write_text("keep existing content")
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {
            "content": "",
            "tool_calls": [{
                "id": "protected-write",
                "function": {"name": "write_file", "arguments": {
                    "path": str(target), "content": "replacement"
                }},
            }],
        },
        {"content": "The file already exists, so I left it unchanged.", "tool_calls": []},
    ])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Update the existing note")

    result = next(message for message in brain.llm_client.requests[1] if message.get("role") == "tool")
    payload = json.loads(result["content"])
    assert payload["status"] == "failed"
    assert payload["effect_status"] == "unknown"
    assert "refusing to overwrite" in payload["data"]
    assert target.read_text() == "keep existing content"
