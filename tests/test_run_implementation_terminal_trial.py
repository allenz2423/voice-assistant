"""Offline safety tests for the bounded headless T1 terminal runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools import create_implementation_fixtures
from tools import run_implementation_terminal_trial as trial


MODEL_ID = "deepseek/deepseek-v4.1-flash"


def _config_file(tmp_path: Path, *, fallbacks: bool = False) -> Path:
    path = tmp_path / "protected-config.yaml"
    path.write_text(
        "llm:\n"
        "  provider: custom\n"
        f"  cloud_model: {MODEL_ID}\n"
        f"  api_base: {trial.f2.EXPECTED_API_BASE}\n"
        "  api_key: synthetic-test-secret\n"
        f"  allow_provider_fallbacks: {'true' if fallbacks else 'false'}\n"
        "  tool_free_model: secondary/model\n"
        "  tool_free_provider_only: [secondary-route]\n"
        "  tool_free_allow_provider_fallbacks: true\n"
        "computer_control:\n  enabled: false\n"
        "computer_vision:\n  enabled: false\n"
        "browser_navigation:\n  enabled: false\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700)
    path.chmod(0o700)


def _args(tmp_path: Path, *, fixture: Path | None = None, run_id: str = "t1-run-001") -> dict:
    fixture = fixture or create_implementation_fixtures.create_fixtures(
        tmp_path / "fixtures", "t1-fixture-001",
    )
    output_dir = tmp_path / "private-runs"
    if not output_dir.exists():
        _private_dir(output_dir)
    return {
        "fixture_dir": fixture,
        "config_path": _config_file(tmp_path),
        "run_id": run_id,
        "config_id": "deepseek-openrouter-fallbacks-off-v1",
        "output_path": output_dir / "runs.jsonl",
        "route_ready_confirmed": True,
        "expected_model": MODEL_ID,
        "metadata_http_status": 200,
        "metadata_model_id": MODEL_ID,
        "metadata_endpoint_count": 31,
        "metadata_checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "confirm_provider_inference": True,
    }


class _FakeProvider:
    provider = "custom"

    def __init__(self, command: str, *, duplicate_calls: bool = False, final_answer: str = "Done.") -> None:
        self.command = command
        self.duplicate_calls = duplicate_calls
        self.final_answer = final_answer
        self.calls = 0
        self.request_tools: list[list[str]] = []

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        self.calls += 1
        self.request_tools.append([tool.name for tool in tools or []])
        if self.calls == 1:
            tool_call = {
                "id": "t1-terminal-1",
                "function": {"name": "run_bash_command", "arguments": {"command": self.command}},
            }
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call, dict(tool_call, id="t1-terminal-2")] if self.duplicate_calls else [tool_call],
            }
        return {"role": "assistant", "content": self.final_answer, "tool_calls": []}

    def format_tool_response(self, tool_call_id, tool_name, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}


def _patch_provider(monkeypatch, fake: _FakeProvider) -> list:
    import src.llm.brain as brain_module

    constructed = []

    def construct(config):
        constructed.append(config)
        return fake

    monkeypatch.setattr(brain_module, "UniversalLLMClient", construct)
    return constructed


def test_exact_fixture_terminal_operation_writes_exact_oracle_and_preserves_source(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    command = trial._expected_command(fixture["source_path"], fixture["answer_path"])
    fake = _FakeProvider(command)
    constructed = _patch_provider(monkeypatch, fake)
    built = []

    def build_brain(config, fixture_data, secrets):
        result = trial._build_brain(config, fixture_data, secrets)
        built.append(result)
        return result

    before_source = (args["fixture_dir"] / "source.txt").read_bytes()
    record = trial.run_trial(**args, brain_builder=build_brain)

    answer = (args["fixture_dir"] / "answer.txt").read_bytes()
    expected = json.loads((args["fixture_dir"] / "expected.json").read_text(encoding="utf-8"))
    assert fake.calls == 2
    assert fake.request_tools == [["run_bash_command"], ["run_bash_command"]]
    assert len(constructed) == 1
    assert constructed[0].llm.tool_free_model == ""
    assert constructed[0].computer_control.enabled is False
    assert constructed[0].computer_vision.enabled is False
    assert constructed[0].browser_navigation.enabled is False
    assert built[0][1].engine == "capture-only"
    assert built[0][1].spoken[-1:] == ["Done."]
    assert built[0][0].tool_free_llm_client is None
    assert built[0][0].llm_client._client is fake
    offered_tool = built[0][0].get_tools()[0]
    serialized_tool = json.dumps(offered_tool.to_openai())
    assert command not in serialized_tool
    assert str(fixture["source_path"]) not in serialized_tool
    assert str(fixture["answer_path"]) not in serialized_tool
    assert "source.txt" in offered_tool.description
    assert "answer.txt" in offered_tool.description
    assert "ITEM:" in offered_tool.description
    assert record["prompt"] == expected["prompts"]["terminal"]
    assert record["oracle"]["status"] == "pass"
    assert record["oracle"]["answer_file_matches_exact_bytes"] is True
    assert record["oracle"]["source_unchanged"] is True
    assert answer == b"3\n" == expected["oracles"]["terminal"]["answer_text"].encode("ascii")
    assert (args["fixture_dir"] / "source.txt").read_bytes() == before_source
    assert record["source_integrity"]["before"]["sha256"] == expected["oracles"]["terminal"]["source_sha256"]
    assert record["dispatch_trace"][0]["interpreted_in_process"] is True
    assert record["dispatch_trace"][0]["shell_started"] is False
    assert record["dispatch_trace"][0]["status"] == "returned"
    assert record["dispatch_trace"][0]["result"] == "3\n"
    assert record["telemetry"]["logical_primary_client_chat_calls"] == 2
    assert args["output_path"].stat().st_mode & 0o777 == 0o600
    assert (args["output_path"].parent / "t1-run-001.events.jsonl").stat().st_mode & 0o777 == 0o600
    assert "synthetic-test-secret" not in args["output_path"].read_text(encoding="utf-8")


def test_unrecognized_shell_command_is_refused_without_execution_or_answer_write(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    marker = tmp_path / "must-not-exist"
    malicious = f"{trial._expected_command(fixture['source_path'], fixture['answer_path'])}; touch {marker}"
    fake = _FakeProvider(malicious)
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert not marker.exists()
    assert not fixture["answer_path"].exists()
    assert record["oracle"]["status"] == "fail"
    assert record["oracle"]["answer_file_matches_exact_bytes"] is False
    assert record["dispatch_trace"][0]["status"] == "refused"
    assert "exactly match" in record["dispatch_trace"][0]["reason"]
    assert record["source_integrity"]["unchanged"] is True


def test_duplicate_dispatch_attempts_do_not_overwrite_answer_and_fail_oracle(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    command = trial._expected_command(fixture["source_path"], fixture["answer_path"])
    fake = _FakeProvider(command, duplicate_calls=True)
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert (fixture["answer_path"]).read_bytes() == b"3\n"
    assert len(record["dispatch_trace"]) == 2
    assert record["dispatch_trace"][0]["status"] == "returned"
    assert record["dispatch_trace"][1]["status"] == "refused"
    assert record["oracle"]["status"] == "fail"
    assert "more than one" in record["oracle"]["objective_evidence_reasons"][-1]


def test_changed_source_or_existing_answer_is_refused_before_provider_construction(tmp_path, monkeypatch):
    fixture_dir = create_implementation_fixtures.create_fixtures(tmp_path / "fixtures", "t1-changed-source")
    source_path = fixture_dir / "source.txt"
    source_path.write_bytes(source_path.read_bytes() + b"EXTRA\n")
    args = _args(tmp_path, fixture=fixture_dir)
    called = []
    _patch_provider(monkeypatch, _FakeProvider("unused"))

    with pytest.raises(trial.TrialError, match="expected source SHA-256"):
        trial.run_trial(**args, brain_builder=lambda *_: called.append(True))

    assert called == []
    assert not args["output_path"].exists()

    fresh = create_implementation_fixtures.create_fixtures(tmp_path / "fixtures", "t1-existing-answer")
    (fresh / "answer.txt").write_bytes(b"user data\n")
    args = _args(tmp_path, fixture=fresh, run_id="t1-existing-answer-run")
    with pytest.raises(trial.TrialError, match="answer.txt must not exist"):
        trial.run_trial(**args, brain_builder=lambda *_: called.append(True))
    assert called == []


def test_route_confirmation_is_required_before_live_provider_construction(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args["route_ready_confirmed"] = False
    called = []

    with pytest.raises(trial.TrialError, match="fresh external metadata check"):
        trial.run_trial(**args, brain_builder=lambda *_: called.append(True))

    assert called == []


def test_fixture_answer_oracle_must_match_independent_source_count(tmp_path):
    fixture_dir = create_implementation_fixtures.create_fixtures(tmp_path / "fixtures", "t1-bad-oracle")
    metadata_path = fixture_dir / "expected.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["oracles"]["terminal"]["answer_text"] = "4\n"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(trial.TrialError, match="oracle does not match source"):
        trial._validate_fixture(fixture_dir)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["oracles"]["terminal"]["answer_text"] = "3\n"
    metadata["oracles"]["file_status"]["byte_length"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(trial.TrialError, match="byte-length oracle"):
        trial._validate_fixture(fixture_dir)


def test_output_must_be_outside_fixture_and_new_event_file(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args["output_path"] = args["fixture_dir"] / "runs.jsonl"

    with pytest.raises(trial.f2.TrialError, match="outside the immutable fixture"):
        trial.run_trial(**args)
