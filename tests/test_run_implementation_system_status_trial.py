"""Offline safety tests for the bounded headless F3 Brain runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools import create_implementation_fixtures
from tools import run_implementation_system_status_trial as trial
from tools import run_implementation_file_status_trial as f2


MODEL_ID = "deepseek/deepseek-v4.1-flash"


def _config_file(tmp_path: Path, *, fallbacks: bool = False) -> Path:
    path = tmp_path / "protected-config.yaml"
    path.write_text(
        "llm:\n"
        "  provider: custom\n"
        f"  cloud_model: {MODEL_ID}\n"
        f"  api_base: {f2.EXPECTED_API_BASE}\n"
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


def _args(tmp_path: Path, *, fixture: Path | None = None, run_id: str = "f3-run-001") -> dict:
    fixture = fixture or create_implementation_fixtures.create_fixtures(
        tmp_path / "fixtures", "f3-fixture-001",
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

    def __init__(self, *, tool_name: str = "get_system_status", arguments=None) -> None:
        self.tool_name = tool_name
        self.arguments = {} if arguments is None else arguments
        self.calls = 0
        self.request_tools: list[list[str]] = []

    def _emit_mock_http_attempt(self) -> None:
        from src.telemetry.events import emit_event

        span_id = f"fake-f3-{self.calls}"
        emit_event(
            "llm.request_started", span_id=span_id, status="started",
            provider="custom", model=MODEL_ID, component="llm",
            attributes={"configured_provider": "custom", "configured_model": MODEL_ID},
        )
        emit_event(
            "llm.completed", span_id=span_id, status="ok",
            provider="custom", model=MODEL_ID, component="llm",
            attributes={
                "http_status": 200,
                "accounting_status": "available",
                "usage": {
                    "input_tokens": 31,
                    "output_tokens": 12,
                    "provider_reported_cost": 0.00001,
                    "currency": "credits",
                },
            },
        )

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        self.calls += 1
        self.request_tools.append([tool.name for tool in tools or []])
        self._emit_mock_http_attempt()
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "f3-status-1",
                    "function": {"name": self.tool_name, "arguments": self.arguments},
                }],
            }
        return {
            "role": "assistant",
            "content": "The system has 8 logical CPU cores and memory is 47 percent in use.",
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


def _snapshot(cpu: int, available: int, *, sampled_ns: int) -> dict:
    total = 1_000_000
    return {
        "sampled_at_utc": "2026-10-06T12:00:00.000000Z",
        "sampled_monotonic_ns": sampled_ns,
        "logical_cpu_count": cpu,
        "logical_cpu_count_source": "os.cpu_count",
        "mem_total_kib": total,
        "mem_available_kib": available,
        "memory_in_use_percent": round((total - available) * 100.0 / total, 4),
        "meminfo_error": None,
    }


def test_snapshot_comparison_rejects_conflicting_duplicate_claims():
    before = _snapshot(8, 500_000, sampled_ns=100)
    after = _snapshot(8, 500_000, sampled_ns=200)

    conflicting = trial._oracle_comparison(
        "CPU has 8 logical cores. CPU has 16 logical cores. "
        "Memory is 50 percent in use. Memory is 80 percent in use.",
        before,
        after,
    )
    assert conflicting["status"] == "not_assessable"
    assert conflicting["logical_cpu_count_match_before"] is False
    assert conflicting["memory_within_one_percentage_point"] is False
    assert conflicting["reasons"] == [
        "tool readback contains conflicting logical CPU counts",
        "tool readback contains conflicting memory-use percentages",
    ]

    repeated_agreeing = trial._oracle_comparison(
        "CPU has 8 logical cores. CPU has 8 logical cores. "
        "Memory is 50 percent in use. Memory is 50 percent in use.",
        before,
        after,
    )
    assert repeated_agreeing["status"] == "matches_snapshots"
    assert repeated_agreeing["logical_cpu_count_match_before"] is True
    assert repeated_agreeing["memory_within_one_percentage_point"] is True


def test_exact_f3_prompt_runs_one_empty_status_dispatch_and_captures_dynamic_evidence(tmp_path, monkeypatch):
    args = _args(tmp_path)
    fake = _FakeProvider()
    constructed = _patch_provider(monkeypatch, fake)
    snapshots = iter((_snapshot(8, 530_000, sampled_ns=100), _snapshot(8, 520_000, sampled_ns=200)))
    monkeypatch.setattr(trial, "_system_snapshot", lambda: next(snapshots))
    import src.llm.brain as brain_module

    tool_results = []

    def mocked_status():
        tool_results.append(True)
        return (
            "CPU has 8 logical cores with load average 0.10, 0.20, 0.30. "
            "Memory is 47 percent in use (470.0 gigabytes used out of 1000.0 gigabytes)."
        )

    monkeypatch.setattr(brain_module, "get_system_status", mocked_status)
    built = []

    def build_brain(config, secrets):
        result = trial._build_brain(config, secrets)
        built.append(result)
        return result

    from src.telemetry import events as telemetry_events

    prior_writer = telemetry_events.EventWriter(enabled=False, path=str(tmp_path / "prior-events.jsonl"))
    monkeypatch.setattr(telemetry_events, "_writer", prior_writer)
    outer_trace_token = telemetry_events.set_trace_id("outer-f3-test-context")
    try:
        record = trial.run_trial(**args, brain_builder=build_brain)
        assert telemetry_events._writer is prior_writer
        assert telemetry_events.get_trace_id() == "outer-f3-test-context"
    finally:
        telemetry_events.reset_trace_id(outer_trace_token)
    assert telemetry_events.get_trace_id() is None

    assert fake.calls == 2
    assert fake.request_tools == [["get_system_status"], []]
    assert len(constructed) == 1
    assert constructed[0].llm.tool_free_model == ""
    assert constructed[0].computer_control.enabled is False
    assert constructed[0].computer_vision.enabled is False
    assert constructed[0].browser_navigation.enabled is False
    assert built[0][1].engine == "capture-only"
    assert built[0][0].tool_free_llm_client is None
    assert tool_results == [True]
    assert record["prompt"] == trial.EXPECTED_PROMPT
    assert record["oracle"]["status"] == "manual_review_required"
    assert record["oracle"]["answer_review_status"] == "manual_review_required"
    assert record["oracle"]["tool_readback_count"] == 1
    assert record["oracle"]["tool_readback_returned"] is True
    assert record["oracle"]["snapshot_comparison"]["status"] == "matches_snapshots"
    assert record["oracle"]["snapshots_before_after"]["before"]["mem_total_kib"] == 1_000_000
    assert record["oracle"]["snapshots_before_after"]["after"]["mem_available_kib"] == 520_000
    assert record["dispatch_trace"][0]["arguments"] == {}
    assert "CPU has 8 logical cores" in record["dispatch_trace"][0]["result"]
    assert record["normalized_provider_responses"][-1]["content"] == record["brain_response_text"]
    assert record["telemetry"]["logical_primary_client_chat_calls"] == 2
    assert record["telemetry"]["logical_primary_client_chat_call_limit"] == 2
    assert record["telemetry"]["http_attempts_observed"] == 2
    assert record["telemetry"]["attempt_telemetry_status"] == "complete"
    assert record["telemetry"]["provider_usage"]["input_tokens"]["value"] == 62
    assert record["telemetry"]["provider_usage"]["output_tokens"]["value"] == 24
    assert record["telemetry"]["provider_usage"]["provider_reported_cost"]["total"] == {
        "amount": pytest.approx(0.00002), "currency": "credits",
    }
    assert record["fixture_integrity"]["unchanged"] is True
    assert "synthetic-test-secret" not in args["output_path"].read_text(encoding="utf-8")
    assert args["output_path"].stat().st_mode & 0o777 == 0o600
    assert (args["output_path"].parent / "f3-run-001.events.jsonl").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_reason"),
    [
        ("list_processes", {}, "only get_system_status is permitted"),
        ("get_system_status", {"unused": True}, "arguments must be an empty object"),
    ],
)
def test_unexpected_tool_or_nonempty_arguments_are_refused_before_host_status_call(
    tmp_path, monkeypatch, tool_name, arguments, expected_reason,
):
    args = _args(tmp_path, run_id="f3-refused-001")
    fake = _FakeProvider(tool_name=tool_name, arguments=arguments)
    _patch_provider(monkeypatch, fake)
    import src.llm.brain as brain_module

    status_calls = []
    monkeypatch.setattr(brain_module, "get_system_status", lambda: status_calls.append(True) or "unsafe")

    record = trial.run_trial(**args)

    assert status_calls == []
    assert fake.calls <= 2
    assert record["oracle"]["status"] == "manual_review_required"
    assert record["oracle"]["tool_readback_returned"] is False
    if tool_name == "get_system_status":
        assert record["dispatch_trace"][0]["status"] == "refused"
        assert expected_reason in record["dispatch_trace"][0]["reason"]
    else:
        assert record["dispatch_trace"] == []
        assert record["provider_tool_trace"][0]["name"] == "list_processes"


def test_route_confirmation_gate_rejects_before_provider_construction(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args["metadata_endpoint_count"] = 0
    fake = _FakeProvider()
    constructed = _patch_provider(monkeypatch, fake)

    with pytest.raises(f2.TrialError, match="positive integer"):
        trial.run_trial(**args)

    assert constructed == []
    assert fake.calls == 0
    assert not args["output_path"].exists()


def test_telemetry_writer_and_trace_are_restored_after_brain_construction_error(tmp_path, monkeypatch):
    args = _args(tmp_path, run_id="f3-builder-error-001")
    from src.telemetry import events as telemetry_events

    monkeypatch.setattr(telemetry_events, "_writer", None)
    outer_trace_token = telemetry_events.set_trace_id("outer-f3-error-context")

    def fail_to_build(_config, _secrets):
        raise RuntimeError("synthetic Brain construction failure")

    try:
        record = trial.run_trial(**args, brain_builder=fail_to_build)
        assert telemetry_events._writer is None
        assert telemetry_events.get_trace_id() == "outer-f3-error-context"
    finally:
        telemetry_events.reset_trace_id(outer_trace_token)

    assert telemetry_events.get_trace_id() is None
    assert record["brain_execution_status"] == "error"
    assert record["errors"][0]["type"] == "RuntimeError"
    assert record["oracle"]["status"] == "manual_review_required"


def test_event_log_read_failure_is_recorded_and_private_record_is_still_appended(tmp_path, monkeypatch):
    args = _args(tmp_path, run_id="f3-event-read-error-001")
    original_read_bytes = Path.read_bytes

    def fail_event_read(path: Path) -> bytes:
        if path.name == "f3-event-read-error-001.events.jsonl":
            raise OSError("synthetic telemetry read failure")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_event_read)

    record = trial.run_trial(**args, brain_builder=lambda _config, _secrets: (_ for _ in ()).throw(RuntimeError("synthetic builder failure")))

    assert record["telemetry"]["event_file_sha256"] is None
    assert record["telemetry"]["event_file_read_error"]["type"] == "OSError"
    assert record["errors"][0]["type"] == "RuntimeError"
    persisted = json.loads(args["output_path"].read_text(encoding="utf-8").splitlines()[-1])
    assert persisted["run_id"] == "f3-event-read-error-001"


def test_cli_reports_private_artifact_write_error_without_traceback(tmp_path, monkeypatch, capsys):
    def fail_record_write(**_kwargs):
        raise OSError("synthetic private file failure")

    monkeypatch.setattr(trial, "run_trial", fail_record_write)
    result = trial.main([
        "--fixture-dir", str(tmp_path / "fixture"),
        "--config", str(tmp_path / "config.yaml"),
        "--run-id", "f3-cli-error-001",
        "--config-id", "test-route",
        "--output", str(tmp_path / "runs.jsonl"),
        "--expected-model", MODEL_ID,
        "--metadata-http-status", "200",
        "--metadata-model-id", MODEL_ID,
        "--metadata-endpoint-count", "1",
        "--metadata-checked-at", "2026-10-06T12:00:00Z",
        "--route-ready-confirmed",
        "--confirm-provider-inference",
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert "private artifact (OSError)" in captured.err
    assert "Traceback" not in captured.err
