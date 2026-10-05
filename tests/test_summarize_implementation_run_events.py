from __future__ import annotations

import json

import pytest

from tools.summarize_implementation_run_events import (
    RunEventSummaryError,
    summarize_files,
)


TRACE_ID = "trace-eval-001"
BASE = 1_000_000_000_000


def _event(
    name: str,
    clock_ns: int,
    *,
    trace_id: str = TRACE_ID,
    span_id: str | None = None,
    status: str = "ok",
    attributes: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "span_id": span_id or f"span-{name}-{clock_ns}",
        "parent_span_id": None,
        "event": name,
        "clock_ns": clock_ns,
        "status": status,
        "provider": "openrouter",
        "model": "test-model",
        "component": "llm",
        "attempt": None,
        "attributes": attributes or {},
    }


def _provider_call(
    started_at: int,
    completed_at: int,
    *,
    span_id: str,
    status: str = "ok",
    http_status: int = 200,
    accounting_status: str = "available",
    usage: dict | None = None,
) -> list[dict]:
    return [
        _event("llm.request_started", started_at, span_id=span_id, status="started"),
        _event(
            "llm.completed",
            completed_at,
            span_id=span_id,
            status=status,
            attributes={
                "http_status": http_status,
                "accounting_status": accounting_status,
                "duration_ms": (completed_at - started_at) / 1_000_000,
                **({"usage": usage} if usage is not None else {}),
            },
        ),
    ]


def _playback_pair(
    started_at: int,
    *,
    span_id: str,
    speech_role: str | None,
    utterance_id: str | None,
    completion_status: str = "ok",
    completion_delay_ns: int = 10_000,
    extra_attributes: dict | None = None,
) -> list[dict]:
    attributes = {
        "speech_role": speech_role,
        "utterance_id": utterance_id,
        **(extra_attributes or {}),
    }
    return [
        _event(
            "playback.started", started_at, span_id=span_id, status="started",
            attributes=attributes,
        ),
        _event(
            "playback.completed", started_at + completion_delay_ns,
            span_id=span_id, status=completion_status, attributes=attributes,
        ),
    ]


def _write_inputs(
    tmp_path, events: list[dict], *, request_end: int = BASE,
    verified_completion: int = BASE + 4_000_000,
):
    record = {
        "run_id": "loop6-run-001",
        "trace_id": TRACE_ID,
        "scenario": "F2",
        "fixture_id": "fixture-001",
        "outcome": "provider_error",
        "failure_class": "provider_transport",
        "event_clock_ns": {
            "request_end": request_end,
            "verified_completion": verified_completion,
        },
    }
    record_path = tmp_path / "run.json"
    event_path = tmp_path / "events.jsonl"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    event_path.write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )
    return record_path, event_path


def test_join_isolates_trace_counts_provider_attempts_and_preserves_failure_evidence(tmp_path):
    events = [
        *_provider_call(
            BASE - 200_000,
            BASE - 100_000,
            span_id="stale-before-run",
            usage={
                "input_tokens": 999,
                "output_tokens": 999,
                "provider_reported_cost": 9.99,
                "currency": "credits",
            },
        ),
        _event("turn.started", BASE, span_id="turn-1", status="started"),
        _event("tool.started", BASE + 50_000, span_id="tool-1", status="started"),
        *_provider_call(
            BASE + 100_000,
            BASE + 200_000,
            span_id="provider-1",
            status="error",
            http_status=429,
            accounting_status="http_error",
        ),
        _event(
            "llm.retrying",
            BASE + 300_000,
            span_id="retry-1",
            status="retrying",
            attributes={"reason": 429, "attempt": 1, "max_attempts": 3},
        ),
        *_provider_call(
            BASE + 400_000,
            BASE + 500_000,
            span_id="provider-2",
            status="error",
            http_status=200,
            accounting_status="provider_error",
            usage={"private_prompt": "must not be retained"},
        ),
        _event(
            "llm.retrying",
            BASE + 600_000,
            span_id="retry-2",
            status="retrying",
            attributes={"reason": "429", "attempt": 2, "max_attempts": 3},
        ),
        *_provider_call(
            BASE + 700_000,
            BASE + 900_000,
            span_id="provider-3",
            usage={
                "input_tokens": 12,
                "output_tokens": 4,
                "provider_reported_cost": 0.002,
                "currency": "credits",
            },
        ),
        _event("playback.started", BASE + 1_500_000, span_id="play-1", status="started"),
        _event("turn.completed", BASE + 2_000_000, span_id="turn-1", status="ok"),
        *_provider_call(
            BASE + 4_500_000,
            BASE + 4_600_000,
            span_id="stale-after-run",
            usage={
                "input_tokens": 888,
                "output_tokens": 888,
                "provider_reported_cost": 8.88,
                "currency": "credits",
            },
        ),
        # This unrelated trace is deliberately out of chronological order and
        # contains a private field. Neither should affect the selected trace.
        _event(
            "playback.started",
            1,
            trace_id="another-run",
            attributes={"text": "unrelated private text"},
        ),
    ]
    record_path, event_path = _write_inputs(tmp_path, events)

    summary = summarize_files(record_path, event_path)

    provider = summary["provider"]
    assert summary["trace_id"] == TRACE_ID
    assert provider["request_started_count"] == 3
    assert provider["completed_count"] == 3
    assert provider["error_count"] == 2
    assert provider["retry_count"] == 2
    assert provider["http_429_response_count"] == 1
    assert provider["http_429_retry_trigger_count"] == 2
    assert provider["reported_cost"]["value"] is None
    assert provider["reported_cost"]["known_subtotals_by_currency"] == {"credits": 0.002}
    assert provider["reported_cost"]["complete"] is False
    assert any("provider_error" in reason for reason in provider["reported_cost"]["missing_reasons"])
    assert [row["attributes"]["http_status"] for row in summary["provider_error_evidence"]] == [
        429, 200,
    ]
    assert summary["first_playback"]["request_end_delta_ms"] == 1.5
    assert summary["first_tool_start"]["request_end_delta_ms"] == 0.05
    assert summary["event_clock_window"]["request_to_verified_completion_ms"] == 4.0
    assert summary["first_playback"]["meaningful_acknowledgment"] is None
    assert "unrelated private text" not in json.dumps(summary)
    assert "must not be retained" not in json.dumps(summary)
    assert summary["speech"]["first_acknowledgment"]["missing_reason"] == (
        "no successfully completed playback tagged 'acknowledgment' in the task window"
    )


def test_speech_summary_uses_tagged_roles_and_groups_clause_playback(tmp_path):
    events = [
        *_playback_pair(
            BASE + 500_000_000, span_id="earcon", speech_role=None,
            utterance_id="earcon-utterance",
        ),
        *_playback_pair(
            BASE + 900_000_000, span_id="failed-ack", speech_role="acknowledgment",
            utterance_id="failed-ack", completion_status="error",
        ),
        *_playback_pair(
            BASE + 1_000_000_000, span_id="ack-1", speech_role="acknowledgment",
            utterance_id="ack-utterance",
        ),
        *_playback_pair(
            BASE + 1_500_000_000, span_id="ack-2", speech_role="acknowledgment",
            utterance_id="ack-utterance",
        ),
        *_playback_pair(
            BASE + 12_000_000_000, span_id="progress-1a", speech_role="progress",
            utterance_id="progress-one",
        ),
        *_playback_pair(
            BASE + 12_100_000_000, span_id="progress-1b", speech_role="progress",
            utterance_id="progress-one",
            extra_attributes={"text": "private speech content must not be returned"},
        ),
        *_playback_pair(
            BASE + 28_000_000_000, span_id="progress-2", speech_role="progress",
            utterance_id="progress-two",
        ),
        *_playback_pair(
            BASE + 30_000_000_000, span_id="final-1", speech_role="final",
            utterance_id="final-utterance",
        ),
    ]
    record_path, event_path = _write_inputs(
        tmp_path, events, verified_completion=BASE + 40_000_000_000,
    )

    summary = summarize_files(record_path, event_path)
    speech = summary["speech"]

    assert summary["event_clock_window"]["request_to_verified_completion_ms"] == 40_000.0
    assert summary["first_playback"]["clock_ns"] == BASE + 500_000_000
    assert speech["first_acknowledgment"] == {
        "clock_ns": BASE + 1_000_000_000,
        "request_end_delta_ms": 1000.0,
        "missing_reason": None,
        "within_5s_target": True,
    }
    assert speech["progress"]["update_count"] == 2
    assert [row["request_end_delta_ms"] for row in speech["progress"]["updates"]] == [
        12_000.0, 28_000.0,
    ]
    assert speech["progress"]["measured_intervals_ms"] == [
        12_000.0, 16_000.0, 2_000.0,
    ]
    assert speech["progress"]["max_measured_interval_ms"] == 16_000.0
    assert speech["progress"]["within_10s_target"] is False
    assert speech["progress"]["end_reason"] == "first_final_speech"
    assert speech["progress"]["gap_to_end_ms"] == 2_000.0
    assert speech["first_final_speech"]["request_end_delta_ms"] == 30_000.0
    assert speech["unclassified_playback_count"] == 1
    assert speech["unconfirmed_playback_count"] == 1
    assert "private speech content" not in json.dumps(summary)


def test_unknown_and_identityless_speech_roles_remain_unavailable(tmp_path):
    events = [
        *_playback_pair(
            BASE + 500_000, span_id="ack-no-id", speech_role="acknowledgment",
            utterance_id=None,
        ),
        *_playback_pair(
            BASE + 1_000_000, span_id="progress-no-id", speech_role="progress",
            utterance_id=None,
        ),
        *_playback_pair(
            BASE + 1_500_000, span_id="unknown-role", speech_role="not-a-role",
            utterance_id="unknown",
        ),
    ]
    record_path, event_path = _write_inputs(tmp_path, events)

    speech = summarize_files(record_path, event_path)["speech"]

    assert speech["first_acknowledgment"]["clock_ns"] is None
    assert speech["first_acknowledgment"]["within_5s_target"] is None
    assert "lacked an utterance_id" in speech["first_acknowledgment"]["missing_reason"]
    assert speech["progress"]["update_count"] == 0
    assert speech["progress"]["missing_reason"] == (
        "1 progress playback event(s) lacked an utterance_id"
    )
    assert speech["unclassified_playback_count"] == 1
    assert speech["tagged_playback_missing_utterance_id_count"] == 2


def test_request_elapsed_rejects_completion_before_request_end(tmp_path):
    record_path, event_path = _write_inputs(
        tmp_path,
        [_event("turn.started", BASE, span_id="turn-1", status="started")],
        request_end=BASE + 2,
        verified_completion=BASE + 1,
    )

    with pytest.raises(RunEventSummaryError, match="verified completion precedes request end"):
        summarize_files(record_path, event_path)


def test_same_currency_complete_accounting_sums_usage_and_provider_durations(tmp_path):
    events = [
        _event("turn.started", BASE, span_id="turn-1", status="started"),
        *_provider_call(
            BASE + 100_000,
            BASE + 1_100_000,
            span_id="provider-1",
            usage={
                "input_tokens": 10,
                "output_tokens": 2,
                "provider_reported_cost": 0.001,
                "currency": "credits",
            },
        ),
        *_provider_call(
            BASE + 1_200_000,
            BASE + 3_200_000,
            span_id="provider-2",
            usage={
                "input_tokens": 20,
                "output_tokens": 3,
                "provider_reported_cost": 0.002,
                "currency": "credits",
            },
        ),
        _event("turn.completed", BASE + 3_500_000, span_id="turn-1", status="ok"),
    ]
    record_path, event_path = _write_inputs(tmp_path, events)

    summary = summarize_files(record_path, event_path)

    provider = summary["provider"]
    assert provider["reported_cost"] == {
        "value": 0.003,
        "currency": "credits",
        "complete": True,
        "known_subtotals_by_currency": {"credits": 0.003},
        "missing_reasons": [],
    }
    assert provider["input_tokens"]["value"] == 30
    assert provider["output_tokens"]["value"] == 5
    assert provider["call_durations"] == [
        {"span_id": "provider-1", "duration_ms": 1.0},
        {"span_id": "provider-2", "duration_ms": 2.0},
    ]
    assert summary["brain_turns"]["durations"] == [
        {"span_id": "turn-1", "duration_ms": 3.5, "status": "completed"},
    ]


def test_equal_provider_start_and_completion_counts_still_require_matching_spans(tmp_path):
    events = _provider_call(
        BASE + 100_000,
        BASE + 200_000,
        span_id="provider-completion",
        usage={
            "input_tokens": 10,
            "output_tokens": 2,
            "provider_reported_cost": 0.01,
            "currency": "credits",
        },
    )
    events[0]["span_id"] = "provider-start"
    record_path, event_path = _write_inputs(tmp_path, events)

    provider = summarize_files(record_path, event_path)["provider"]

    assert provider["request_started_count"] == provider["completed_count"] == 1
    assert provider["span_pairing"]["complete"] is False
    assert provider["unmatched_request_span_ids"] == ["provider-start"]
    assert provider["completion_without_request_span_ids"] == ["provider-completion"]
    assert provider["input_tokens"]["value"] is None
    assert provider["input_tokens"]["known_subtotal"] == 10
    assert provider["output_tokens"]["value"] is None
    assert provider["output_tokens"]["known_subtotal"] == 2
    assert provider["reported_cost"]["value"] is None
    assert provider["reported_cost"]["known_subtotals_by_currency"] == {"credits": 0.01}
    assert any("same span_id" in reason for reason in provider["reported_cost"]["missing_reasons"])


def test_mixed_currency_costs_are_not_added_together(tmp_path):
    events = [
        *_provider_call(
            BASE + 100_000,
            BASE + 200_000,
            span_id="provider-1",
            usage={"provider_reported_cost": 0.01, "currency": "USD"},
        ),
        *_provider_call(
            BASE + 300_000,
            BASE + 400_000,
            span_id="provider-2",
            usage={"provider_reported_cost": 0.02, "currency": "credits"},
        ),
    ]
    record_path, event_path = _write_inputs(tmp_path, events)

    cost = summarize_files(record_path, event_path)["provider"]["reported_cost"]

    assert cost["value"] is None
    assert cost["currency"] is None
    assert cost["complete"] is False
    assert cost["known_subtotals_by_currency"] == {"USD": 0.01, "credits": 0.02}
    assert any("more than one currency" in reason for reason in cost["missing_reasons"])


def test_missing_usage_keeps_accounting_reason_instead_of_zero_cost(tmp_path):
    events = _provider_call(
        BASE + 100_000,
        BASE + 200_000,
        span_id="provider-unsupported",
        accounting_status="unsupported",
    )
    record_path, event_path = _write_inputs(tmp_path, events)

    provider = summarize_files(record_path, event_path)["provider"]

    assert provider["input_tokens"]["value"] is None
    assert provider["input_tokens"]["missing_reasons"] == [
        "provider-unsupported: accounting_status=unsupported",
    ]
    assert provider["reported_cost"]["value"] is None
    assert provider["reported_cost"]["missing_reasons"] == [
        "provider-unsupported: accounting_status=unsupported",
    ]


def test_matching_trace_events_must_be_monotonic(tmp_path):
    events = [
        _event("turn.started", BASE, span_id="turn-1", status="started"),
        _event("turn.completed", BASE - 1, span_id="turn-1", status="ok"),
    ]
    record_path, event_path = _write_inputs(tmp_path, events)

    with pytest.raises(RunEventSummaryError, match="not monotonic"):
        summarize_files(record_path, event_path)


def test_trace_id_is_required_and_mismatched_event_sets_are_rejected(tmp_path):
    events = [_event("turn.started", BASE, trace_id="another-run")]
    record_path, event_path = _write_inputs(tmp_path, events)

    with pytest.raises(RunEventSummaryError, match="no telemetry events match"):
        summarize_files(record_path, event_path)
