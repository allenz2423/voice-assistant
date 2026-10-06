"""Offline safety and record tests for the headless F1 Brain runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_implementation_factual_trial as trial


def _config_file(
    tmp_path: Path,
    *,
    fallbacks: bool = False,
    api_base: str = trial.EXPECTED_API_BASE,
) -> Path:
    path = tmp_path / "protected-config.yaml"
    path.write_text(
        "llm:\n"
        "  provider: custom\n"
        "  cloud_model: stealth/space-bunny-alpha\n"
        f"  api_base: {api_base}\n"
        "  api_key: synthetic-test-secret\n"
        f"  allow_provider_fallbacks: {'true' if fallbacks else 'false'}\n"
        "  tool_free_model: different-tool-free-model\n"
        "  tool_free_provider_only: [different-provider]\n"
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


def _run_args(
    tmp_path: Path,
    *,
    run_id: str = "run-001",
    fallback: bool = False,
    api_base: str = trial.EXPECTED_API_BASE,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "private"
    if not output_dir.exists():
        _private_dir(output_dir)
    else:
        output_dir.chmod(0o700)
    return {
        "config_path": _config_file(tmp_path, fallbacks=fallback, api_base=api_base),
        "prompt": trial.FACTUAL_PROMPT,
        "fixture_run_id": "f1-series-001",
        "run_id": run_id,
        "config_id": "space-bunny-fallbacks-off-v1",
        "output_path": output_dir / "runs.jsonl",
        "route_ready_confirmed": True,
        "confirm_provider_inference": True,
    }


class _FakeProvider:
    provider = "custom"

    def __init__(self, *, tool_call: bool = False, emit_usage: bool = True) -> None:
        self.tool_call = tool_call
        self.emit_usage = emit_usage
        self.request_tools = []

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        from src.telemetry.events import emit_event

        self.request_tools.append(tools)
        emit_event(
            "llm.request_started", span_id="provider-span-1", status="started",
            provider="custom", model="stealth/space-bunny-alpha", component="provider",
        )
        attributes = {"accounting_status": "available"}
        if self.emit_usage:
            attributes["usage"] = {
                "input_tokens": 128,
                "output_tokens": 24,
                "provider_reported_cost": 0.001,
                "currency": "credits",
            }
        emit_event(
            "llm.completed", span_id="provider-span-1", status="ok",
            provider="custom", model="stealth/space-bunny-alpha", component="provider",
            attributes=attributes,
        )
        tool_calls = []
        if self.tool_call:
            tool_calls = [{
                "id": "tool-call-1",
                "function": {"name": "run_bash_command", "arguments": "{}"},
            }]
        return {
            "role": "assistant",
            "content": "Rigatoni has ridges and straight-cut ends; penne has angled ends and catches sauce in its hollow tubes.",
            "tool_calls": tool_calls,
        }


def _patch_provider(monkeypatch, fake: _FakeProvider) -> None:
    import src.llm.brain as brain_module

    configs = []

    def construct(config):
        configs.append(config)
        return fake

    monkeypatch.setattr(brain_module, "UniversalLLMClient", construct)
    return configs


def test_fake_provider_runs_brain_toolless_and_writes_private_trace_and_record(tmp_path, monkeypatch):
    args = _run_args(tmp_path)
    fake = _FakeProvider()
    constructed_configs = _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert fake.request_tools == [[]]
    assert len(constructed_configs) == 1
    assert constructed_configs[0].llm.provider == "custom"
    assert constructed_configs[0].llm.cloud_model == "stealth/space-bunny-alpha"
    assert constructed_configs[0].llm.tool_free_model == ""
    assert record["outcome"] == "not_scored"
    assert record["oracle_status"] == "manual_review_required"
    assert record["brain_response_text"].startswith("Rigatoni has ridges")
    assert record["route"] == {
        "provider": "custom",
        "model": "stealth/space-bunny-alpha",
        "api_base": "https://openrouter.ai/api/v1",
        "config_id": "space-bunny-fallbacks-off-v1",
        "provider_only": [],
        "allow_provider_fallbacks": False,
    }
    assert record["provider"]["provider_call_count"] == 1
    assert record["provider"]["retry_count"] == 0
    assert record["provider"]["http_429_response_count"] == 0
    assert record["provider"]["usage"] == {
        "input_tokens": 128,
        "output_tokens": 24,
        "reported_cost": 0.001,
        "currency": "credits",
        "accounting_status": "complete",
        "missing_reasons": {},
    }
    assert record["trace_id"] == "run-001"
    event_path = args["output_path"].parent / "run-001.events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert events
    assert {event["trace_id"] for event in events} == {"run-001"}
    assert args["output_path"].stat().st_mode & 0o777 == 0o600
    assert event_path.stat().st_mode & 0o777 == 0o600
    rows = [json.loads(line) for line in args["output_path"].read_text().splitlines()]
    assert rows == [record]
    assert "synthetic-test-secret" not in args["output_path"].read_text()


def test_tool_call_is_recorded_but_never_dispatched(tmp_path, monkeypatch):
    args = _run_args(tmp_path)
    fake = _FakeProvider(tool_call=True)
    _patch_provider(monkeypatch, fake)
    dispatched = []
    from src.llm.brain import AdamBrain

    async def record_dispatch(*_args, **_kwargs):
        dispatched.append(True)
        raise AssertionError("tool dispatch must stay unreachable")

    monkeypatch.setattr(AdamBrain, "_execute_tool", record_dispatch)

    record = trial.run_trial(**args)

    assert fake.request_tools == [[]]
    assert record["brain_execution_status"] == "error"
    assert record["unexpected_tool_call_stopped"] is True
    assert record["normalized_provider_responses"][0]["tool_names"] == ["run_bash_command"]
    assert record["errors"][0]["type"] == "_UnexpectedToolCall"
    assert dispatched == []


class _FailingProvider:
    provider = "custom"

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        from src.telemetry.events import emit_event

        emit_event("llm.request_started", span_id="failed-span", status="started", component="provider")
        emit_event(
            "llm.completed", span_id="failed-span", status="error", component="provider",
            attributes={"http_status": 429, "accounting_status": "http_error"},
        )
        emit_event(
            "llm.retrying", status="retrying", component="provider",
            attributes={"reason": "429", "attempt": 1},
        )
        raise RuntimeError(
            "Authorization: Bearer synthetic-test-secret "
            "https://user:password@example.invalid/path?api_key=query-secret"
        )


def test_provider_error_is_redacted_and_status_retry_evidence_is_kept(tmp_path, monkeypatch, capsys):
    args = _run_args(tmp_path)
    _patch_provider(monkeypatch, _FailingProvider())

    record = trial.run_trial(**args)
    serialized = args["output_path"].read_text()
    error = record["errors"][0]["message"]

    assert record["brain_execution_status"] == "error"
    assert record["provider"]["provider_call_count"] == 1
    assert record["provider"]["http_429_response_count"] == 1
    assert record["provider"]["http_429_retry_trigger_count"] == 1
    assert "synthetic-test-secret" not in error + serialized
    assert "password" not in error + serialized
    assert "query-secret" not in error + serialized
    assert "[REDACTED]" in error
    assert capsys.readouterr().out == ""


def test_no_start_event_does_not_discard_observed_429_or_retry():
    events = [
        {"trace_id": "r", "event": "llm.completed", "span_id": "s", "status": "error",
         "attributes": {"http_status": 429, "accounting_status": "http_error"}},
        {"trace_id": "r", "event": "llm.retrying", "span_id": "retry",
         "attributes": {"reason": "429"}},
    ]

    summary = trial._provider_event_summary(events, "r")

    assert summary["provider_call_count"] is None
    assert summary["http_429_response_count"] == 1
    assert summary["http_429_retry_trigger_count"] == 1
    assert summary["provider_errors"][0]["http_status"] == 429


def test_429_total_is_unknown_when_a_provider_start_has_no_completion():
    events = [
        {"trace_id": "r", "event": "llm.request_started", "span_id": "s1", "status": "started"},
        {"trace_id": "r", "event": "llm.completed", "span_id": "s1", "status": "error",
         "attributes": {"http_status": 429, "accounting_status": "http_error"}},
        {"trace_id": "r", "event": "llm.request_started", "span_id": "s2", "status": "started"},
    ]

    summary = trial._provider_event_summary(events, "r")

    assert summary["attempt_pairing_complete"] is False
    assert summary["http_429_response_count"] is None
    assert summary["http_429_response_count_missing_reason"]


def test_missing_usage_remains_null_with_reasons_and_brain_only_timing(tmp_path, monkeypatch):
    args = _run_args(tmp_path)
    _patch_provider(monkeypatch, _FakeProvider(emit_usage=False))

    record = trial.run_trial(**args)

    assert record["provider"]["usage"]["input_tokens"] is None
    assert record["provider"]["usage"]["reported_cost"] is None
    assert record["provider"]["usage"]["missing_reasons"]["input_tokens"]
    assert record["timing_ms"]["headless_brain_call"] is not None
    assert "headless_cli_elapsed" not in record["timing_ms"]
    assert set(record["event_clock_ns"]) == {
        "headless_brain_call_start", "headless_brain_call_return", "verified_completion",
    }
    assert record["timing_ms"]["request_end_to_verified_state"] is None
    assert record["timing_ms"]["request_end_to_ack"] is None


def test_refuses_changed_route_fallbacks_or_noncanonical_prompt_before_output(tmp_path):
    args = _run_args(tmp_path, fallback=True)
    with pytest.raises(trial.TrialError, match="fallbacks must be explicitly disabled"):
        trial.run_trial(**args)
    assert not args["output_path"].exists()


@pytest.mark.parametrize(
    "api_base",
    [
        "https://example.invalid/v1",
        "http://openrouter.ai/api/v1",
        "https://openrouter.ai.evil.test/api/v1",
        "https://openrouter.ai/api/v1/",
    ],
)
def test_refuses_api_base_outside_exact_pinned_https_route(tmp_path, api_base):
    args = _run_args(tmp_path, api_base=api_base)

    with pytest.raises(trial.TrialError, match="API base must remain pinned"):
        trial.run_trial(**args)

    assert not args["output_path"].exists()


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o666])
def test_config_must_not_be_group_or_world_accessible(tmp_path, mode):
    private_config = _config_file(tmp_path)
    private_config.chmod(mode)
    with pytest.raises(trial.TrialError, match="group or other users"):
        trial._load_existing_config(private_config)


def test_config_must_be_owned_by_current_user(tmp_path, monkeypatch):
    private_config = _config_file(tmp_path)
    private_config.chmod(0o600)
    original_euid = trial.os.geteuid()
    monkeypatch.setattr(trial.os, "geteuid", lambda: original_euid + 1)
    with pytest.raises(trial.TrialError, match="owned by the current user"):
        trial._load_existing_config(private_config)


def test_config_must_be_regular_file_and_symlinks_are_refused(tmp_path):
    directory = tmp_path / "config-dir"
    directory.mkdir(mode=0o700)
    with pytest.raises(trial.TrialError, match="regular file"):
        trial._load_existing_config(directory)

    target = _config_file(tmp_path)
    link = tmp_path / "config-link.yaml"
    link.symlink_to(target)
    with pytest.raises(trial.TrialError, match="symlinks and special files"):
        trial._load_existing_config(link)


def test_refuses_noncanonical_prompt_and_requires_explicit_live_flags(tmp_path):
    args = _run_args(tmp_path)
    args["prompt"] = "Different request that could disclose data"
    with pytest.raises(trial.TrialError, match="exact run-sheet factual prompt"):
        trial.run_trial(**args)
    assert not args["output_path"].exists()


def test_refuses_credential_in_serialized_config_id_before_creating_output(tmp_path):
    args = _run_args(tmp_path)
    args["config_id"] = "synthetic-test-secret-route"

    with pytest.raises(trial.TrialError, match="config ID must not contain a configured credential"):
        trial.run_trial(**args)

    assert not args["output_path"].exists()


def test_refuses_non_trial_jsonl_and_output_equal_to_config(tmp_path):
    args = _run_args(tmp_path)
    args["output_path"].write_text('{"run_id":"unrelated"}\n', encoding="utf-8")
    args["output_path"].chmod(0o600)

    with pytest.raises(trial.TrialError, match="not an F1 run record"):
        trial.run_trial(**args)

    private_config = args["output_path"].parent / "private-config.jsonl"
    private_config.write_text(
        "llm:\n  provider: custom\n  cloud_model: stealth/space-bunny-alpha\n"
        "  api_base: https://openrouter.ai/api/v1\n  api_key: synthetic-test-secret\n"
        "  allow_provider_fallbacks: false\n",
        encoding="utf-8",
    )
    private_config.chmod(0o600)
    args = _run_args(tmp_path / "same-config")
    args["config_path"] = private_config
    args["output_path"] = private_config

    with pytest.raises(trial.TrialError, match="may not replace or append to the config file"):
        trial.run_trial(**args)


def test_cli_missing_inference_attestation_never_enters_trial(tmp_path, capsys):
    args = _run_args(tmp_path)

    result = trial.main([
        "--config", str(args["config_path"]), "--prompt", args["prompt"],
        "--fixture-run-id", args["fixture_run_id"], "--run-id", args["run_id"],
        "--config-id", args["config_id"], "--output", str(args["output_path"]),
        "--route-ready-confirmed",
    ])

    assert result == 2
    assert "refusing live provider call" in capsys.readouterr().err
    assert not args["output_path"].exists()

    args = _run_args(tmp_path / "second", run_id="run-002")
    args["route_ready_confirmed"] = False
    with pytest.raises(trial.TrialError, match="serving endpoint"):
        trial.run_trial(**args)
    assert not args["output_path"].exists()
