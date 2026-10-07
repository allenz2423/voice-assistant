"""Offline safety tests for the bounded headless T1 terminal runner."""

from __future__ import annotations

import json
import shlex
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

    def __init__(self, command: str | list[str], *, duplicate_calls: bool = False,
                 batch_calls: bool = False, final_answer: str = "Done.") -> None:
        self.commands = [command] if isinstance(command, str) else list(command)
        self.duplicate_calls = duplicate_calls
        self.batch_calls = batch_calls
        self.final_answer = final_answer
        self.calls = 0
        self.request_tools: list[list[str]] = []

    @staticmethod
    def _tool_call(command: str, call_id: str) -> dict:
        return {
            "id": call_id,
            "function": {"name": "run_bash_command", "arguments": {"command": command}},
        }

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        self.calls += 1
        self.request_tools.append([tool.name for tool in tools or []])
        if self.batch_calls and self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    self._tool_call(command, f"t1-terminal-{index}")
                    for index, command in enumerate(self.commands, 1)
                ],
            }
        if self.calls <= len(self.commands):
            tool_call = self._tool_call(self.commands[self.calls - 1], f"t1-terminal-{self.calls}")
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call, dict(tool_call, id="t1-terminal-duplicate")]
                if self.duplicate_calls and self.calls == 1 else [tool_call],
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
    assert "only the exact generated command" not in offered_tool.description
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


def test_duplicate_write_attempt_is_refused_and_tracked_separately_from_file_oracle(tmp_path, monkeypatch):
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
    assert record["dispatch_trace"][1]["refused_without_side_effect"] is True
    assert record["oracle"]["status"] == "pass"
    assert record["dispatch_policy"]["successful_fixture_write_count"] == 1
    assert record["dispatch_policy"]["refused_dispatch_count"] == 1
    assert record["dispatch_policy"]["post_write_dispatch_count"] == 1


def test_relative_cd_grep_form_is_accepted_and_interpreted_in_process(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    command = trial._expected_relative_command(fixture["fixture_dir"])
    fake = _FakeProvider(command)
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert record["oracle"]["status"] == "pass"
    assert record["dispatch_policy"]["successful_fixture_write_count"] == 1
    assert record["dispatch_trace"][0]["arguments"]["command"] == command
    assert record["dispatch_trace"][0]["interpreted_in_process"] is True
    assert record["dispatch_trace"][0]["shell_started"] is False
    assert (fixture["answer_path"]).read_bytes() == b"3\n"


def test_refused_benign_spelling_does_not_consume_success_slot_before_correction(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    benign_unrecognized = (
        f"grep -c '^ITEM:' '{fixture['source_path']}' > '{fixture['answer_path']}'"
    )
    corrected = trial._expected_relative_command(fixture["fixture_dir"])
    fake = _FakeProvider([benign_unrecognized, benign_unrecognized, corrected])
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert fake.calls == 4
    assert len(record["dispatch_trace"]) == 3
    assert record["dispatch_trace"][0]["status"] == "refused"
    assert record["dispatch_trace"][0]["refused_without_side_effect"] is True
    assert record["dispatch_trace"][1]["status"] == "refused"
    assert record["dispatch_trace"][1]["refused_without_side_effect"] is True
    assert record["dispatch_trace"][2]["status"] == "returned"
    assert record["oracle"]["status"] == "pass"
    assert record["dispatch_policy"]["successful_fixture_write_count"] == 1
    assert record["dispatch_policy"]["refused_dispatch_count"] == 2
    assert record["dispatch_policy"]["terminal_tool_call_attempt_count"] == 3
    assert record["telemetry"]["logical_primary_client_chat_call_limit"] == 4
    assert (fixture["answer_path"]).read_bytes() == b"3\n"


def test_path_escape_is_refused_without_writing_outside_fixture(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    escaped_answer = fixture["fixture_dir"].parent / "outside-answer.txt"
    command = (
        f"cd {shlex.quote(str(fixture['fixture_dir']))} && "
        "grep -c '^ITEM:' source.txt > ../outside-answer.txt"
    )
    _patch_provider(monkeypatch, _FakeProvider(command))

    record = trial.run_trial(**args)

    assert not escaped_answer.exists()
    assert not fixture["answer_path"].exists()
    assert record["dispatch_trace"][0]["status"] == "refused"
    assert record["dispatch_trace"][0]["refused_without_side_effect"] is True
    assert record["oracle"]["status"] == "fail"


def test_post_write_absolute_cat_readback_is_safe_and_preserves_objective_pass(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    write_command = trial._expected_relative_command(fixture["fixture_dir"])
    read_command = trial._expected_answer_read_command(fixture["answer_path"])
    fake = _FakeProvider([write_command, read_command])
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert fake.calls == 3
    assert record["oracle"]["status"] == "pass"
    assert record["oracle"]["answer_file_matches_exact_bytes"] is True
    assert record["oracle"]["source_unchanged"] is True
    assert record["dispatch_trace"][0]["dispatch_kind"] == "fixture_write"
    assert record["dispatch_trace"][0]["status"] == "returned"
    assert record["dispatch_trace"][1]["dispatch_kind"] == "answer_readback"
    assert record["dispatch_trace"][1]["status"] == "returned"
    assert record["dispatch_trace"][1]["result"] == "3\n"
    assert record["dispatch_trace"][1]["readback_matches_expected_bytes"] is True
    assert record["dispatch_trace"][1]["shell_started"] is False
    assert record["dispatch_policy"]["successful_fixture_write_count"] == 1
    assert record["dispatch_policy"]["successful_answer_readback_count"] == 1
    assert record["dispatch_policy"]["post_write_dispatch_count"] == 1
    assert record["dispatch_policy"]["refused_dispatch_count"] == 0


def test_post_write_fixture_scoped_cd_cat_readback_is_safe(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    write_command = trial._expected_relative_command(fixture["fixture_dir"])
    read_command = trial._expected_relative_answer_read_command(fixture["fixture_dir"])
    fake = _FakeProvider([write_command, read_command])
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert fake.calls == 3
    assert record["oracle"]["status"] == "pass"
    assert record["dispatch_trace"][1]["arguments"]["command"] == read_command
    assert record["dispatch_trace"][1]["dispatch_kind"] == "answer_readback"
    assert record["dispatch_trace"][1]["status"] == "returned"
    assert record["dispatch_trace"][1]["result"] == "3\n"
    assert record["dispatch_trace"][1]["readback_matches_expected_bytes"] is True
    assert record["dispatch_policy"]["successful_fixture_write_count"] == 1
    assert record["dispatch_policy"]["successful_answer_readback_count"] == 1
    assert record["dispatch_policy"]["refused_dispatch_count"] == 0


def test_bare_cat_answer_path_is_refused_without_grounded_fixture_directory(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    write_command = trial._expected_relative_command(fixture["fixture_dir"])
    fake = _FakeProvider([write_command, "cat answer.txt"])
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert record["oracle"]["status"] == "pass"
    assert record["dispatch_trace"][1]["status"] == "refused"
    assert record["dispatch_trace"][1]["refused_without_side_effect"] is True
    assert "exactly match" in record["dispatch_trace"][1]["reason"]
    assert record["dispatch_policy"]["refused_dispatch_count"] == 1
    assert (fixture["answer_path"]).read_bytes() == b"3\n"


def test_unsafe_post_write_command_is_refused_and_tracked_without_changing_file_oracle(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    write_command = trial._expected_relative_command(fixture["fixture_dir"])
    fake = _FakeProvider([write_command, "cat /etc/passwd"])
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert record["oracle"]["status"] == "pass"
    assert record["oracle"]["answer_file_matches_exact_bytes"] is True
    assert record["oracle"]["source_unchanged"] is True
    assert record["dispatch_trace"][1]["status"] == "refused"
    assert record["dispatch_trace"][1]["refused_without_side_effect"] is True
    assert "exactly match" in record["dispatch_trace"][1]["reason"]
    assert record["dispatch_policy"]["status"] == "refused_dispatches"
    assert record["dispatch_policy"]["refused_dispatch_count"] == 1
    assert (fixture["answer_path"]).read_bytes() == b"3\n"


def test_batch_over_attempt_limit_is_stopped_before_any_tool_dispatch(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fixture = trial._validate_fixture(args["fixture_dir"])
    command = trial._expected_relative_command(fixture["fixture_dir"])
    fake = _FakeProvider([command, command, command, command], batch_calls=True)
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert record["dispatch_policy"]["terminal_tool_call_attempt_count"] == 4
    assert record["dispatch_policy"]["terminal_tool_call_attempt_limit"] == 3
    assert record["oracle"]["status"] == "fail"
    assert record["dispatch_trace"] == []
    assert not fixture["answer_path"].exists()


def test_cli_returns_failure_when_brain_returns_but_objective_oracle_fails(monkeypatch, capsys):
    monkeypatch.setattr(trial, "run_trial", lambda **_kwargs: {
        "run_id": "t1-cli-failed-oracle",
        "fixture_id": "t1-fixture-cli",
        "scenario": "T1",
        "oracle": {"status": "fail"},
        "brain_execution_status": "returned",
        "telemetry": {"logical_primary_client_chat_calls": 2},
        "private_output_paths": {"record": "/private/runs.jsonl", "events": "/private/events.jsonl"},
    })

    exit_status = trial.main([
        "--fixture-dir", "/private/t1-fixture",
        "--config", "/private/config.yaml",
        "--run-id", "t1-cli-run",
        "--config-id", "route-v1",
        "--output", "/private/runs.jsonl",
        "--expected-model", MODEL_ID,
        "--metadata-http-status", "200",
        "--metadata-model-id", MODEL_ID,
        "--metadata-endpoint-count", "1",
        "--metadata-checked-at", "2026-10-06T12:00:00Z",
        "--route-ready-confirmed",
        "--confirm-provider-inference",
    ])

    assert exit_status == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["brain_execution_status"] == "returned"
    assert printed["outcome"] == "fail"


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
