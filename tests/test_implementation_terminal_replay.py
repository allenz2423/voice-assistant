"""Provider-free AdamBrain replay for the deterministic terminal fixture."""

from __future__ import annotations

import hashlib
import json
import shlex
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _ScriptedClient:
    def __init__(self, command: str) -> None:
        self.provider = "local"
        self.command = command
        self.requests: list[dict] = []

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        self.requests.append({"messages": messages, "tools": tools})
        if len(self.requests) == 1:
            return {
                "content": "",
                "tool_calls": [{
                    "id": "terminal-count",
                    "function": {
                        "name": "run_bash_command",
                        "arguments": {"command": self.command},
                    },
                }],
            }
        return {"content": "The count was written to answer.txt.", "tool_calls": []}

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
async def test_terminal_fixture_round_trip_through_adam_brain(tmp_path):
    from src.llm.brain import AdamBrain
    from tools.create_implementation_fixtures import create_fixtures

    memory = SimpleNamespace(retrieve_context=lambda _text: None)
    tts = _SilentTTS()

    for repetition in range(1, 4):
        fixture = create_fixtures(tmp_path / "fixtures", f"terminal-brain-replay-{repetition}")
        expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
        request = expected["prompts"]["terminal"]
        terminal_oracle = expected["oracles"]["terminal"]
        source_path = fixture / "source.txt"
        answer_path = fixture / "answer.txt"
        source_before = source_path.read_bytes()
        source_sha256 = hashlib.sha256(source_before).hexdigest()

        # The command reads only the synthetic source and creates only the
        # requested answer file under this repetition's private fixture.
        command = (
            f"cd {shlex.quote(str(fixture))} && "
            "awk '/^ITEM:/ {count += 1} END {print count}' source.txt > answer.txt"
        )
        brain = AdamBrain(
            _test_config(), None, None, None, tts, memory_mgr=memory,
        )
        brain.computer_controller = SimpleNamespace(
            available=False,
            drag_active=False,
            coordinate_mode="pixels",
            invalidate_snapshot=lambda: None,
        )
        client = _ScriptedClient(command)
        brain.llm_client = client

        with patch(
            "src.llm.brain.get_open_windows_prompt_context",
            return_value="Synthetic fixture context",
        ):
            await brain.process_user_utterance(request)

        assert len(client.requests) == 2
        offered_tools = client.requests[0]["tools"]
        assert offered_tools is not None
        assert any(tool.name == "run_bash_command" for tool in offered_tools)

        tool_reply = next(
            message for message in brain.messages
            if message.get("role") == "tool" and message.get("tool_call_id") == "terminal-count"
        )
        tool_result = json.loads(tool_reply["content"])
        assert tool_result["status"] == "returned"
        assert tool_result["goal_status"] == "not_assessed"
        assert tool_result["data"] == ""

        assert answer_path.read_bytes() == terminal_oracle["answer_text"].encode("utf-8") == b"3\n"
        source_after = source_path.read_bytes()
        assert len(source_before) == expected["oracles"]["file_status"]["byte_length"] == 75
        assert source_sha256 == terminal_oracle["source_sha256"]
        assert hashlib.sha256(source_after).hexdigest() == terminal_oracle["source_sha256"]
        assert source_after == source_before
