"""Offline safety tests for the bounded headless F2 Brain runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools import create_implementation_fixtures
from tools import run_implementation_file_status_trial as trial


MODEL_ID = "deepseek/deepseek-v4.1-flash"


def _config_file(tmp_path: Path, *, fallbacks: bool = False) -> Path:
    path = tmp_path / "protected-config.yaml"
    path.write_text(
        "llm:\n"
        "  provider: custom\n"
        f"  cloud_model: {MODEL_ID}\n"
        f"  api_base: {trial.EXPECTED_API_BASE}\n"
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


def _args(tmp_path: Path, *, fixture: Path | None = None, run_id: str = "f2-run-001") -> dict:
    fixture = fixture or create_implementation_fixtures.create_fixtures(
        tmp_path / "fixtures", "f2-fixture-001",
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

    def __init__(
        self, source_path: str, *, wrong_path: str | None = None,
        max_chars: int = 128, duplicate_read_calls: bool = False,
        final_answer: str = "The file is 75 bytes and its exact first line is ITEM: olive.",
    ) -> None:
        self.source_path = source_path
        self.request_tools: list[list[str]] = []
        self.calls = 0
        self.wrong_path = wrong_path
        self.max_chars = max_chars
        self.duplicate_read_calls = duplicate_read_calls
        self.final_answer = final_answer

    def _emit_mock_http_attempts(self) -> None:
        from src.telemetry.events import emit_event

        attempt_count = 2 if self.calls == 1 else 1
        for attempt in range(1, attempt_count + 1):
            span_id = f"fake-f2-{self.calls}-{attempt}"
            status = 429 if self.calls == 1 and attempt == 1 else 200
            emit_event(
                "llm.request_started", span_id=span_id, status="started",
                provider="custom", model=MODEL_ID, component="llm",
                attributes={"configured_provider": "custom", "configured_model": MODEL_ID},
            )
            emit_event(
                "llm.completed", span_id=span_id,
                status="error" if status == 429 else "ok",
                provider="custom", model=MODEL_ID, component="llm",
                attributes={"http_status": status},
            )
            if status == 429:
                emit_event(
                    "llm.retrying", span_id=f"fake-f2-retry-{self.calls}",
                    status="retrying", provider="custom", model=MODEL_ID,
                    component="llm",
                    attributes={"reason": "429", "attempt": 1, "max_attempts": 3},
                )

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        self.calls += 1
        self.request_tools.append([tool.name for tool in tools or []])
        self._emit_mock_http_attempts()
        if self.calls == 1:
            read_call = {
                "id": "f2-read-1",
                "function": {
                    "name": "read_file",
                    "arguments": {"path": self.wrong_path or self.source_path, "max_chars": self.max_chars},
                },
            }
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [read_call, dict(read_call, id="f2-read-2")]
                if self.duplicate_read_calls else [read_call],
            }
        return {
            "role": "assistant",
            "content": self.final_answer,
            "tool_calls": [],
        }

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


def test_valid_fixture_path_runs_one_bounded_read_and_records_private_oracle(tmp_path, monkeypatch):
    args = _args(tmp_path)
    source_path = str((args["fixture_dir"] / "source.txt").resolve())
    fake = _FakeProvider(source_path)
    constructed = _patch_provider(monkeypatch, fake)
    built = []

    def build_brain(config, exact_source_path, secrets, oracle):
        result = trial._build_brain(config, exact_source_path, secrets, oracle)
        built.append(result)
        return result

    record = trial.run_trial(**args, brain_builder=build_brain)

    assert fake.calls == 2
    assert fake.request_tools == [["read_file"], ["read_file"]]
    assert len(constructed) == 1
    assert constructed[0].llm.tool_free_model == ""
    assert constructed[0].computer_control.enabled is False
    assert constructed[0].computer_vision.enabled is False
    assert constructed[0].browser_navigation.enabled is False
    assert built[0][1].engine == "capture-only"
    assert built[0][1].spoken[-1:] == [record["brain_response_text"]]
    assert built[0][0].tool_free_llm_client is None
    assert built[0][0].llm_client._client is fake
    expected = json.loads((args["fixture_dir"] / "expected.json").read_text(encoding="utf-8"))
    assert record["prompt"] == expected["prompts"]["file_status"]
    assert record["oracle"]["status"] == "manual_review_required"
    assert record["oracle"]["objective_evidence_status"] == "pass"
    assert record["oracle"]["source_unchanged"] is True
    assert record["source_integrity"]["before"]["sha256"] == expected["oracles"]["terminal"]["source_sha256"]
    assert record["brain_response_text"] == "The file is 75 bytes and its exact first line is ITEM: olive."
    assert record["dispatch_trace"][0]["status"] == "returned"
    assert record["dispatch_trace"][0]["bounded_max_chars"] == 128
    assert "Bytes: 75" in record["dispatch_trace"][0]["result"]
    assert record["provider_tool_trace"][0]["name"] == "read_file"
    assert record["telemetry"]["logical_primary_client_chat_calls"] == 2
    assert record["telemetry"]["logical_primary_client_chat_call_limit"] == 2
    assert record["telemetry"]["http_attempts_observed"] == 3
    assert record["telemetry"]["http_completions_observed"] == 3
    assert record["telemetry"]["retry_events_observed"] == 1
    assert record["telemetry"]["http_statuses_observed"] == [429, 200, 200]
    assert record["telemetry"]["attempt_telemetry_status"] == "complete"
    assert record["route"]["operator_reported_metadata_check"]["source"] == "operator-reported external metadata check"
    assert record["route"]["operator_reported_metadata_check"]["runner_fetched_or_cryptographically_verified"] is False
    assert "synthetic-test-secret" not in args["output_path"].read_text(encoding="utf-8")
    assert args["output_path"].stat().st_mode & 0o777 == 0o600
    event_path = args["output_path"].parent / "f2-run-001.events.jsonl"
    assert event_path.stat().st_mode & 0o777 == 0o600


def test_wrong_model_path_is_refused_before_filesystem_dispatch(tmp_path, monkeypatch):
    fixture = create_implementation_fixtures.create_fixtures(
        tmp_path / "fixtures", "f2-fixture-wrong-path",
    )
    args = _args(tmp_path, fixture=fixture, run_id="f2-wrong-path")
    fake = _FakeProvider(str(fixture / "source.txt"), wrong_path=str(fixture / "document.txt"))
    _patch_provider(monkeypatch, fake)
    import src.llm.brain as brain_module

    reads = []

    def forbidden_read(*_args, **_kwargs):
        reads.append(True)
        raise AssertionError("wrong path must never reach filesystem.read_file")

    monkeypatch.setattr(brain_module, "read_file", forbidden_read)

    record = trial.run_trial(**args)

    assert reads == []
    assert record["dispatch_trace"][0]["status"] == "refused"
    assert "exactly match generated fixture" in record["dispatch_trace"][0]["reason"]
    assert record["oracle"]["status"] == "manual_review_required"
    assert record["oracle"]["objective_evidence_status"] == "fail"
    assert record["source_integrity"]["unchanged"] is True


def test_negated_answer_is_never_automatically_scored_as_a_pass(tmp_path, monkeypatch):
    args = _args(tmp_path)
    source_path = str((args["fixture_dir"] / "source.txt").resolve())
    wrong_answer = "It is not 75 bytes; its first line is not ITEM: olive."
    fake = _FakeProvider(source_path, final_answer=wrong_answer)
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert record["brain_response_text"] == wrong_answer
    assert record["oracle"]["status"] == "manual_review_required"
    assert record["oracle"]["objective_evidence_status"] == "pass"
    assert record["oracle"]["read_file_evidence_verified"] is True
    assert "answer_contains_byte_length" not in record["oracle"]
    assert "answer_contains_exact_first_line" not in record["oracle"]


def test_changed_generated_source_is_refused_before_provider_construction(tmp_path, monkeypatch):
    args = _args(tmp_path)
    source_path = args["fixture_dir"] / "source.txt"
    source_path.write_text("changed fixture content\n", encoding="utf-8")
    constructed = []
    import src.llm.brain as brain_module

    monkeypatch.setattr(brain_module, "UniversalLLMClient", lambda _config: constructed.append(True))

    with pytest.raises(trial.TrialError, match="expected source SHA-256 oracle"):
        trial.run_trial(**args)

    assert constructed == []
    assert not args["output_path"].exists()


def test_requires_both_route_and_live_inference_flags_before_provider_construction(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args["confirm_provider_inference"] = False
    constructed = []
    import src.llm.brain as brain_module

    monkeypatch.setattr(brain_module, "UniversalLLMClient", lambda _config: constructed.append(True))

    with pytest.raises(trial.TrialError, match="without --confirm-provider-inference"):
        trial.run_trial(**args)

    args["confirm_provider_inference"] = True
    args["route_ready_confirmed"] = False
    with pytest.raises(trial.TrialError, match="fresh external metadata check"):
        trial.run_trial(**args)

    assert constructed == []
    assert not args["output_path"].exists()


def test_fixture_prompt_must_match_its_exact_generated_source_path(tmp_path):
    args = _args(tmp_path)
    expected_path = args["fixture_dir"] / "expected.json"
    metadata = json.loads(expected_path.read_text(encoding="utf-8"))
    metadata["prompts"]["file_status"] += " changed"
    expected_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(trial.TrialError, match="does not exactly match generated source.txt"):
        trial._validate_fixture(args["fixture_dir"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("metadata_http_status", 503, "HTTP status 200"),
        ("metadata_model_id", "other/model", "exactly match the expected model"),
        ("metadata_endpoint_count", 0, "positive integer"),
        ("metadata_checked_at", "2020-01-01T00:00:00Z", "older than one hour"),
    ],
)
def test_invalid_route_attestation_is_refused_before_config_or_provider(
    tmp_path, monkeypatch, field, value, message,
):
    args = _args(tmp_path)
    args[field] = value
    loaded = []
    constructed = []
    monkeypatch.setattr(trial, "load_config", lambda _path: loaded.append(True))
    import src.llm.brain as brain_module

    monkeypatch.setattr(brain_module, "UniversalLLMClient", lambda _config: constructed.append(True))

    with pytest.raises(trial.TrialError, match=message):
        trial.run_trial(**args)

    assert loaded == []
    assert constructed == []
    assert not args["output_path"].exists()


def test_provider_fallbacks_must_be_disabled_before_provider_construction(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args["config_path"] = _config_file(tmp_path, fallbacks=True)
    constructed = []
    import src.llm.brain as brain_module

    monkeypatch.setattr(brain_module, "UniversalLLMClient", lambda _config: constructed.append(True))

    with pytest.raises(trial.TrialError, match="fallbacks must be explicitly disabled"):
        trial.run_trial(**args)

    assert constructed == []
    assert not args["output_path"].exists()


def test_out_of_bounds_read_chars_are_refused_before_filesystem_dispatch(tmp_path, monkeypatch):
    args = _args(tmp_path)
    source_path = str((args["fixture_dir"] / "source.txt").resolve())
    fake = _FakeProvider(source_path, max_chars=trial.MAX_READ_CHARS + 1)
    _patch_provider(monkeypatch, fake)
    import src.llm.brain as brain_module

    reads = []

    def forbidden_read(*_args, **_kwargs):
        reads.append(True)
        raise AssertionError("out-of-bounds read must not reach filesystem.read_file")

    monkeypatch.setattr(brain_module, "read_file", forbidden_read)
    record = trial.run_trial(**args)

    assert reads == []
    assert record["dispatch_trace"] == []
    assert record["normalized_provider_responses"][0]["tool_calls"]
    assert record["oracle"]["status"] == "manual_review_required"
    assert record["oracle"]["objective_evidence_status"] == "fail"


def test_only_one_read_file_dispatch_is_permitted_per_turn(tmp_path, monkeypatch):
    args = _args(tmp_path)
    source_path = str((args["fixture_dir"] / "source.txt").resolve())
    fake = _FakeProvider(source_path, duplicate_read_calls=True)
    _patch_provider(monkeypatch, fake)
    import src.llm.brain as brain_module

    reads = []
    original_read = brain_module.read_file

    def counted_read(*read_args, **read_kwargs):
        reads.append(True)
        return original_read(*read_args, **read_kwargs)

    monkeypatch.setattr(brain_module, "read_file", counted_read)
    record = trial.run_trial(**args)

    assert len(reads) == 1
    assert len(record["dispatch_trace"]) == 2
    assert record["dispatch_trace"][0]["status"] == "returned"
    assert record["dispatch_trace"][1]["status"] == "refused"
    assert "dispatch limit" in record["dispatch_trace"][1]["reason"]


def test_module_help_is_available_without_provider_or_runtime_startup():
    project_root = Path(trial.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "tools.run_implementation_file_status_trial", "--help"],
        cwd=project_root,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: run_implementation_file_status_trial.py" in result.stdout
    assert result.stderr == ""


def test_script_path_help_works_from_outside_repository_even_with_shadow_src(tmp_path):
    script_path = Path(trial.__file__).resolve()
    project_root = script_path.parents[1]
    shadow_src = tmp_path / "src"
    shadow_src.mkdir()
    (shadow_src / "__init__.py").write_text("", encoding="utf-8")
    (shadow_src / "config.py").write_text(
        "raise RuntimeError('shadowed src import')\n", encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(project_root)))
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: run_implementation_file_status_trial.py" in result.stdout
    assert result.stderr == ""
