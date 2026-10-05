"""Provider-free AdamBrain replay for the deterministic F2 file-status fixture."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _ScriptedClient:
    """Stand in for the model while leaving Brain's real read_file dispatch intact."""

    def __init__(self, source_path: str, final_answer: str, repetition: int) -> None:
        self.provider = "local"
        self.source_path = source_path
        self.final_answer = final_answer
        self.call_id = f"f2-read-{repetition}"
        self.requests: list[dict] = []

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        self.requests.append({"messages": messages, "tools": tools})
        if len(self.requests) == 1:
            return {
                "content": "",
                "tool_calls": [{
                    "id": self.call_id,
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": self.source_path},
                    },
                }],
            }
        return {"content": self.final_answer, "tool_calls": []}

    def format_tool_response(self, tool_call_id, tool_name, result):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }


class _SilentTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak_async(self, text: str) -> None:
        self.spoken.append(text)


def _test_config():
    return SimpleNamespace(llm=SimpleNamespace(
        provider="local",
        local_model="test-model",
        cloud_model="test-model",
        ollama_host="http://localhost:11434",
        api_base="https://example.test/v1",
        api_key="",
        temperature=0,
        num_ctx=8192,
    ))


@pytest.mark.asyncio
async def test_file_status_fixture_round_trip_through_adam_brain(tmp_path):
    from src.llm.brain import AdamBrain
    from tools.create_implementation_fixtures import create_fixtures

    memory = SimpleNamespace(retrieve_context=lambda _text: None)

    for repetition in range(1, 4):
        fixture = create_fixtures(
            tmp_path / "fixtures", f"file-status-brain-replay-{repetition}"
        )
        expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
        request = expected["prompts"]["file_status"]
        file_oracle = expected["oracles"]["file_status"]
        source_oracle = expected["oracles"]["terminal"]["source_sha256"]
        source_path = expected["paths"]["source_file"]
        source_file = fixture / "source.txt"

        assert source_path == str(source_file.resolve())
        assert request == (
            f'Read "{source_path}" and tell me its byte size and exact first line.'
        )
        source_before = source_file.read_bytes()
        source_sha256 = hashlib.sha256(source_before).hexdigest()
        first_line = source_before.decode("utf-8").splitlines()[0]
        byte_length = len(source_before)
        assert byte_length == file_oracle["byte_length"] == 75
        assert first_line == file_oracle["first_line"] == "ITEM: olive"
        assert source_sha256 == source_oracle

        final_answer = (
            f"The file is {file_oracle['byte_length']} bytes; "
            f"its first line is {file_oracle['first_line']}."
        )
        client = _ScriptedClient(source_path, final_answer, repetition)
        tts = _SilentTTS()
        brain = AdamBrain(
            _test_config(), None, None, None, tts, memory_mgr=memory,
        )
        brain.computer_controller = SimpleNamespace(
            available=False,
            drag_active=False,
            coordinate_mode="pixels",
            invalidate_snapshot=lambda: None,
        )
        brain.llm_client = client

        with patch(
            "src.llm.brain.get_open_windows_prompt_context",
            return_value="Synthetic fixture context",
        ):
            await brain.process_user_utterance(request)

        assert len(client.requests) == 2
        offered_tools = client.requests[0]["tools"]
        assert offered_tools is not None
        assert any(tool.name == "read_file" for tool in offered_tools)

        assistant_tool_calls = [
            call for message in brain.messages
            if message.get("role") == "assistant"
            for call in message.get("tool_calls", [])
        ]
        assert len(assistant_tool_calls) == 1
        requested_tool = assistant_tool_calls[0]["function"]
        assert requested_tool["name"] == "read_file"
        assert requested_tool["arguments"] == {"path": source_path}

        tool_reply = next(
            message for message in brain.messages
            if message.get("role") == "tool"
            and message.get("tool_call_id") == client.call_id
        )
        assert tool_reply["name"] == "read_file"
        outer_result = json.loads(tool_reply["content"])
        assert outer_result["call_id"] == client.call_id
        assert outer_result["status"] == "returned"
        assert outer_result["goal_status"] == "not_assessed"

        inner_result = json.loads(outer_result["data"])
        assert inner_result["ok"] is True
        assert inner_result["readback"] == (
            f"File: {source_path}\n"
            f"Bytes: {byte_length}\n"
            f"Text:\n{source_before.decode('utf-8')}"
        )
        assert f"Bytes: {file_oracle['byte_length']}\n" in inner_result["readback"]
        assert f"Text:\n{file_oracle['first_line']}\n" in inner_result["readback"]

        assistant_responses = [
            message["content"] for message in brain.messages
            if message.get("role") == "assistant" and message.get("content")
        ]
        assert assistant_responses == [final_answer]
        assert tts.spoken[-1:] == [final_answer]

        source_after = source_file.read_bytes()
        assert source_after == source_before
        assert hashlib.sha256(source_after).hexdigest() == source_oracle
