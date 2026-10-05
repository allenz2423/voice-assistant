"""Provider-free Brain dispatch against a deterministic synthetic F3 host snapshot."""

from __future__ import annotations

import builtins
import io
import json
import os
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _NoCallClient:
    """Fails if a factual system-status request is sent to a model."""

    provider = "local"

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        self.requests.append({"messages": messages, "tools": tools})
        raise AssertionError("The F3 status request should use direct tool dispatch.")

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
async def test_generated_f3_prompt_uses_real_brain_and_status_handler(tmp_path, monkeypatch):
    from src.llm.brain import AdamBrain
    from src.tools import system_telemetry
    from tools.create_implementation_fixtures import create_fixtures

    fixture = create_fixtures(tmp_path / "fixtures", "f3-system-status-replay")
    expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    request = expected["prompts"]["system_status"]

    expected_cpu_count = 8
    synthetic_cpuinfo = (
        "processor\t: 0\n\n"
        "processor\t: 1\n\n"
        "processor\t: 2\n\n"
        "processor\t: 3\n\n"
        "processor\t: 4\n\n"
        "processor\t: 5\n\n"
        "processor\t: 6\n\n"
        "processor\t: 7\n\n"
    )
    assert sum(
        line.startswith("processor") for line in synthetic_cpuinfo.splitlines()
    ) == expected_cpu_count

    mem_total_kib = 16_000_000
    mem_available_kib = 5_760_000
    synthetic_proc = {
        "/proc/stat": "",
        "/proc/loadavg": "0.10 0.20 0.30 1/100 1234\n",
        "/proc/cpuinfo": synthetic_cpuinfo,
        "/proc/meminfo": (
            f"MemTotal:       {mem_total_kib} kB\n"
            f"MemAvailable:    {mem_available_kib} kB\n"
        ),
    }
    real_open = builtins.open
    proc_reads: list[str] = []

    def open_synthetic_proc(path, mode="r", *args, **kwargs):
        path_text = os.fspath(path)
        if path_text.startswith("/proc/"):
            if path_text not in synthetic_proc or mode != "r":
                raise AssertionError(f"Unexpected /proc access: {path_text} ({mode})")
            proc_reads.append(path_text)
            return io.StringIO(synthetic_proc[path_text])
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(system_telemetry, "open", open_synthetic_proc, raising=False)
    monkeypatch.setattr(system_telemetry.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        system_telemetry.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_blocks=2, f_frsize=1024**3, f_bavail=1),
    )

    tts = _SilentTTS()
    client = _NoCallClient()
    brain = AdamBrain(
        _test_config(), None, None, None, tts,
        memory_mgr=SimpleNamespace(retrieve_context=lambda _text: None),
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
        return_value="",
    ):
        await brain.process_user_utterance(request)

    assert client.requests == []
    assert sorted(proc_reads) == sorted(synthetic_proc)
    tool_reply = next(
        message for message in brain.messages if message.get("role") == "tool"
    )
    assert tool_reply["name"] == "get_system_status"
    result = json.loads(tool_reply["content"])
    assert result["status"] == "returned"
    assert result["goal_status"] == "not_assessed"
    status_text = result["data"]

    cpu_match = re.search(r"CPU has (\d+) logical cores", status_text)
    memory_match = re.search(r"Memory is (\d+) percent in use", status_text)
    assert cpu_match is not None
    assert int(cpu_match.group(1)) == expected_cpu_count
    assert memory_match is not None

    expected_memory_percent = (
        (mem_total_kib - mem_available_kib) / mem_total_kib * 100
    )
    assert abs(int(memory_match.group(1)) - expected_memory_percent) <= 1
    assert int(memory_match.group(1)) == round(expected_memory_percent)

    assert brain.messages[-1] == {"role": "assistant", "content": status_text}
    assert tts.spoken[-1:] == [status_text]
