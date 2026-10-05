#!/usr/bin/env python3
"""Join one private evaluation record to its privacy-conscious telemetry events.

The join uses the exact trace_id recorded by the run. It summarizes provider
attempts, retry signals, provider timing/accounting, Brain turn timing, and
playback timing. Telemetry does not identify playback purpose, so playback is
never interpreted as a meaningful acknowledgment.

Example:

    python tools/summarize_implementation_run_events.py \
        --record /private/run.json --events /private/events.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EVENT_SCHEMA_VERSION = 1


class RunEventSummaryError(ValueError):
    """Raised when a run record cannot be joined safely to telemetry events."""


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RunEventSummaryError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _matching_events(event_path: Path, trace_id: str) -> list[dict[str, Any]]:
    matching: list[dict[str, Any]] = []
    previous_clock_ns: int | None = None
    try:
        stream = event_path.open("r", encoding="utf-8")
    except OSError:
        raise

    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunEventSummaryError(
                    f"event line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise RunEventSummaryError(f"event line {line_number} must be an object")
            row_trace_id = row.get("trace_id")
            if not isinstance(row_trace_id, str) or not row_trace_id:
                raise RunEventSummaryError(
                    f"event line {line_number} is missing a trace_id"
                )
            if row_trace_id != trace_id:
                continue
            schema_version = _integer(
                row.get("schema_version"),
                f"event line {line_number} schema_version",
                minimum=1,
            )
            if schema_version != EVENT_SCHEMA_VERSION:
                raise RunEventSummaryError(
                    f"unsupported telemetry schema version {schema_version} "
                    f"on event line {line_number}"
                )
            event_name = row.get("event")
            if not isinstance(event_name, str) or not event_name:
                raise RunEventSummaryError(
                    f"event line {line_number} is missing an event name"
                )
            clock_ns = _integer(
                row.get("clock_ns"),
                f"event line {line_number} clock_ns",
            )
            if previous_clock_ns is not None and clock_ns < previous_clock_ns:
                raise RunEventSummaryError(
                    f"matching trace events are not monotonic at line {line_number}"
                )
            previous_clock_ns = clock_ns
            attributes = row.get("attributes", {})
            if not isinstance(attributes, dict):
                raise RunEventSummaryError(
                    f"event line {line_number} attributes must be an object"
                )
            matching.append({
                **row,
                "clock_ns": clock_ns,
                "attributes": attributes,
                "_line_number": line_number,
            })

    if not matching:
        raise RunEventSummaryError(f"no telemetry events match trace_id {trace_id!r}")
    return matching


def _event_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Project only bounded, operational fields; never copy arbitrary attributes."""
    attributes = row["attributes"]
    safe_attributes: dict[str, Any] = {}
    for key in (
        "reason",
        "attempt",
        "max_attempts",
        "http_status",
        "accounting_status",
        "duration_ms",
        "network_ms",
        "response_headers_ms",
        "response_body_ms",
        "configured_provider",
        "configured_model",
        "provider_request_id",
        "returned_model",
    ):
        value = attributes.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            if not isinstance(value, float) or math.isfinite(value):
                safe_attributes[key] = value
    usage = attributes.get("usage")
    if isinstance(usage, dict):
        safe_usage = {
            key: usage[key]
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "reasoning_tokens",
                "provider_reported_cost",
                "currency",
            )
            if key in usage
            and (
                (isinstance(usage[key], (int, float))
                 and not isinstance(usage[key], bool)
                 and (not isinstance(usage[key], float) or math.isfinite(usage[key])))
                or (key == "currency" and isinstance(usage[key], str)
                    and usage[key].isascii() and len(usage[key]) <= 24)
            )
        }
        if safe_usage:
            safe_attributes["usage"] = safe_usage

    result = {
        "line_number": row["_line_number"],
        "clock_ns": row["clock_ns"],
        "event": row["event"],
        "status": row.get("status"),
        "provider": row.get("provider"),
        "model": row.get("model"),
        "attributes": safe_attributes,
    }
    if isinstance(row.get("span_id"), str):
        result["span_id"] = row["span_id"]
    return result


def _counter_summary(
    completions: list[dict[str, Any]],
    started_count: int,
    field: str,
    *,
    integer_only: bool,
    span_pairing_complete: bool,
    span_pairing_missing_reasons: list[str],
) -> dict[str, Any]:
    observed: list[int | float] = []
    missing_reasons: list[str] = []
    if started_count == 0:
        missing_reasons.append("no provider requests were recorded")
    for row in completions:
        span_id = row.get("span_id") or f"line {row['_line_number']}"
        accounting_status = row["attributes"].get("accounting_status")
        if accounting_status != "available":
            missing_reasons.append(
                f"{span_id}: accounting_status={accounting_status or 'missing'}"
            )
            continue
        usage = row["attributes"].get("usage")
        value = usage.get(field) if isinstance(usage, dict) else None
        valid = (
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            if integer_only
            else _number(value) is not None
        )
        if valid:
            observed.append(value if integer_only else float(value))
            continue
        missing_reasons.append(f"{span_id}: usage field {field!r} is missing or invalid")

    if len(completions) < started_count:
        missing_reasons.append(
            f"{started_count - len(completions)} provider request(s) have no llm.completed event"
        )
    if len(completions) > started_count:
        missing_reasons.append(
            f"{len(completions) - started_count} llm.completed event(s) have no matching request start"
        )
    if not span_pairing_complete:
        missing_reasons.extend(span_pairing_missing_reasons)
    complete = (
        started_count > 0
        and len(completions) == started_count
        and len(observed) == len(completions)
        and span_pairing_complete
    )
    partial_sum: int | float | None
    if not observed:
        partial_sum = None
    elif integer_only:
        partial_sum = sum(int(value) for value in observed)
    else:
        partial_sum = math.fsum(float(value) for value in observed)
    return {
        "value": partial_sum if complete else None,
        "known_subtotal": partial_sum,
        "complete": complete,
        "missing_reasons": list(dict.fromkeys(missing_reasons)),
    }


def _cost_summary(
    completions: list[dict[str, Any]],
    started_count: int,
    *,
    span_pairing_complete: bool,
    span_pairing_missing_reasons: list[str],
) -> dict[str, Any]:
    subtotals: dict[str, list[float]] = {}
    missing_reasons: list[str] = []
    reported_count = 0
    if started_count == 0:
        missing_reasons.append("no provider requests were recorded")
    for row in completions:
        span_id = row.get("span_id") or f"line {row['_line_number']}"
        accounting_status = row["attributes"].get("accounting_status")
        usage = row["attributes"].get("usage")
        cost = _number(usage.get("provider_reported_cost")) if isinstance(usage, dict) else None
        currency = usage.get("currency") if isinstance(usage, dict) else None
        if accounting_status != "available":
            missing_reasons.append(
                f"{span_id}: accounting_status={accounting_status or 'missing'}"
            )
            continue
        if cost is None:
            missing_reasons.append(f"{span_id}: provider-reported cost is missing or invalid")
            continue
        if (
            not isinstance(currency, str)
            or not currency
            or len(currency) > 24
            or not currency.isascii()
        ):
            missing_reasons.append(f"{span_id}: cost currency is missing or invalid")
            continue
        reported_count += 1
        subtotals.setdefault(currency, []).append(cost)

    if len(completions) < started_count:
        missing_reasons.append(
            f"{started_count - len(completions)} provider request(s) have no llm.completed event"
        )
    if len(completions) > started_count:
        missing_reasons.append(
            f"{len(completions) - started_count} llm.completed event(s) have no matching request start"
        )
    if not span_pairing_complete:
        missing_reasons.extend(span_pairing_missing_reasons)

    known_subtotals = {
        currency: math.fsum(values)
        for currency, values in sorted(subtotals.items())
    }
    all_accounted = (
        started_count > 0
        and len(completions) == started_count
        and reported_count == len(completions)
        and span_pairing_complete
    )
    if len(known_subtotals) > 1:
        missing_reasons.append("provider calls reported costs in more than one currency")
    same_currency = len(known_subtotals) == 1
    complete = all_accounted and same_currency
    return {
        "value": next(iter(known_subtotals.values())) if complete else None,
        "currency": next(iter(known_subtotals)) if same_currency else None,
        "complete": complete,
        "known_subtotals_by_currency": known_subtotals,
        "missing_reasons": list(dict.fromkeys(missing_reasons)),
    }


def _provider_span_pairing(
    starts: list[dict[str, Any]], completions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require one unique completion span for every unique request-start span."""
    start_ids: list[str] = []
    completion_ids: list[str] = []
    missing_start_span_lines: list[int] = []
    missing_completion_span_lines: list[int] = []
    for row in starts:
        span_id = row.get("span_id")
        if isinstance(span_id, str) and span_id:
            start_ids.append(span_id)
        else:
            missing_start_span_lines.append(row["_line_number"])
    for row in completions:
        span_id = row.get("span_id")
        if isinstance(span_id, str) and span_id:
            completion_ids.append(span_id)
        else:
            missing_completion_span_lines.append(row["_line_number"])

    start_counts = Counter(start_ids)
    completion_counts = Counter(completion_ids)
    unmatched_start_ids = sorted(
        span_id for span_id, count in start_counts.items()
        if count > completion_counts.get(span_id, 0)
    )
    completion_without_start_ids = sorted(
        span_id for span_id, count in completion_counts.items()
        if count > start_counts.get(span_id, 0)
    )
    duplicate_start_ids = sorted(span_id for span_id, count in start_counts.items() if count > 1)
    duplicate_completion_ids = sorted(
        span_id for span_id, count in completion_counts.items() if count > 1
    )
    reasons = []
    if missing_start_span_lines:
        reasons.append(
            "provider request-start event(s) are missing span_id on line(s) "
            + ", ".join(map(str, missing_start_span_lines))
        )
    if missing_completion_span_lines:
        reasons.append(
            "provider completion event(s) are missing span_id on line(s) "
            + ", ".join(map(str, missing_completion_span_lines))
        )
    if unmatched_start_ids:
        reasons.append(
            "provider request-start span(s) have no completion with the same span_id: "
            + ", ".join(unmatched_start_ids)
        )
    if completion_without_start_ids:
        reasons.append(
            "provider completion span(s) have no request start with the same span_id: "
            + ", ".join(completion_without_start_ids)
        )
    if duplicate_start_ids:
        reasons.append("duplicate provider request-start span_id(s): " + ", ".join(duplicate_start_ids))
    if duplicate_completion_ids:
        reasons.append("duplicate provider completion span_id(s): " + ", ".join(duplicate_completion_ids))
    complete = (
        not missing_start_span_lines
        and not missing_completion_span_lines
        and not unmatched_start_ids
        and not completion_without_start_ids
        and not duplicate_start_ids
        and not duplicate_completion_ids
        and len(start_ids) == len(starts)
        and len(completion_ids) == len(completions)
    )
    return {
        "complete": complete,
        "matched_request_count": sum(
            min(count, completion_counts.get(span_id, 0))
            for span_id, count in start_counts.items()
        ),
        "unmatched_request_span_ids": unmatched_start_ids,
        "completion_without_request_span_ids": completion_without_start_ids,
        "duplicate_request_span_ids": duplicate_start_ids,
        "duplicate_completion_span_ids": duplicate_completion_ids,
        "missing_reasons": reasons,
    }


def _turn_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    starts: dict[str, int] = {}
    completions: dict[str, int] = {}
    for row in events:
        span_id = row.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            continue
        if row["event"] == "turn.started":
            starts[span_id] = row["clock_ns"]
        elif row["event"] in {"turn.completed", "turn.cancelled"}:
            completions[span_id] = row["clock_ns"]

    durations = []
    for span_id, start_clock in starts.items():
        end_clock = completions.get(span_id)
        if end_clock is not None:
            if end_clock < start_clock:
                raise RunEventSummaryError(
                    f"turn span {span_id!r} completes before it starts"
                )
            durations.append({
                "span_id": span_id,
                "duration_ms": round((end_clock - start_clock) / 1_000_000, 3),
                "status": "cancelled" if any(
                    row.get("span_id") == span_id and row["event"] == "turn.cancelled"
                    for row in events
                ) else "completed",
            })
    return {
        "started_count": len(starts),
        "completed_count": len(completions),
        "durations": durations,
        "incomplete_span_ids": sorted(set(starts) - set(completions)),
        "completion_without_start_span_ids": sorted(set(completions) - set(starts)),
    }


def summarize_run_events(
    run_record: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    trace_id = run_record.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        raise RunEventSummaryError("run record trace_id is required")
    run_id = run_record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RunEventSummaryError("run record run_id is required")
    clock_window = run_record.get("event_clock_ns")
    if not isinstance(clock_window, dict):
        raise RunEventSummaryError("run record event_clock_ns is required")
    request_end = _integer(
        clock_window.get("request_end"),
        "event_clock_ns.request_end",
        minimum=1,
    )
    verified_completion = _integer(
        clock_window.get("verified_completion"),
        "event_clock_ns.verified_completion",
        minimum=1,
    )
    if verified_completion < request_end:
        raise RunEventSummaryError("verified completion precedes request end")

    if not events:
        raise RunEventSummaryError("no matching telemetry events were supplied")
    if any(row.get("trace_id") != trace_id for row in events):
        raise RunEventSummaryError("event set contains a different trace_id")
    clocks = [row["clock_ns"] for row in events]
    if any(right < left for left, right in zip(clocks, clocks[1:])):
        raise RunEventSummaryError("matching trace events are not monotonic")
    events = [
        row for row in events
        if request_end <= row["clock_ns"] <= verified_completion
    ]
    if not events:
        raise RunEventSummaryError(
            "no matching telemetry events fall within the run's request-to-completion window"
        )
    clocks = [row["clock_ns"] for row in events]

    provider_starts = [row for row in events if row["event"] == "llm.request_started"]
    provider_completions = [row for row in events if row["event"] == "llm.completed"]
    provider_span_pairing = _provider_span_pairing(provider_starts, provider_completions)
    span_pairing_complete = provider_span_pairing["complete"]
    span_pairing_missing_reasons = provider_span_pairing["missing_reasons"]
    retries = [row for row in events if row["event"] == "llm.retrying"]
    http_429_responses = [
        row for row in provider_completions
        if row["attributes"].get("http_status") == 429
    ]
    http_429_retry_triggers = [
        row for row in retries
        if row["attributes"].get("reason") == 429
        or str(row["attributes"].get("reason", "")).strip() == "429"
    ]
    provider_errors = [
        row for row in provider_completions if row.get("status") != "ok"
    ]

    call_durations = []
    duration_missing_reasons = []
    for row in provider_completions:
        duration_ms = _number(row["attributes"].get("duration_ms"))
        if duration_ms is None:
            duration_missing_reasons.append(
                f"{row.get('span_id') or 'unknown span'}: duration_ms is missing or invalid"
            )
        else:
            call_durations.append({
                "span_id": row.get("span_id"),
                "duration_ms": duration_ms,
            })
    if len(provider_completions) < len(provider_starts):
        duration_missing_reasons.append(
            f"{len(provider_starts) - len(provider_completions)} provider request(s) "
            "have no llm.completed duration"
        )

    accounting_status_counts: dict[str, int] = {}
    for row in provider_completions:
        status = row["attributes"].get("accounting_status")
        normalized = str(status) if isinstance(status, str) and status else "missing"
        accounting_status_counts[normalized] = accounting_status_counts.get(normalized, 0) + 1

    playback_events = [
        row for row in events
        if row["event"] == "playback.started" and row["clock_ns"] >= request_end
    ]
    first_playback = min(playback_events, key=lambda row: row["clock_ns"]) if playback_events else None
    tool_start_events = [
        row for row in events
        if row["event"] == "tool.started" and row["clock_ns"] >= request_end
    ]
    first_tool_start = (
        min(tool_start_events, key=lambda row: row["clock_ns"])
        if tool_start_events else None
    )
    first_tool_start_summary = {
        "clock_ns": first_tool_start["clock_ns"] if first_tool_start else None,
        "request_end_delta_ms": (
            round((first_tool_start["clock_ns"] - request_end) / 1_000_000, 3)
            if first_tool_start else None
        ),
        "missing_reason": None if first_tool_start else "no tool.started event in the task window",
    }
    first_playback_summary = {
        "clock_ns": first_playback["clock_ns"] if first_playback else None,
        "request_end_delta_ms": (
            round((first_playback["clock_ns"] - request_end) / 1_000_000, 3)
            if first_playback else None
        ),
        "meaningful_acknowledgment": None,
        "meaningful_acknowledgment_missing_reason": (
            "playback events do not identify speech purpose or meaningfulness"
        ),
        "missing_reason": None if first_playback else "no playback.started event in the task window",
    }

    return {
        "run_id": run_id,
        "scenario": run_record.get("scenario"),
        "fixture_id": run_record.get("fixture_id"),
        "trace_id": trace_id,
        "outcome": run_record.get("outcome"),
        "failure_class": run_record.get("failure_class"),
        "event_count": len(events),
        "event_clock_window": {
            "request_end_clock_ns": request_end,
            "verified_completion_clock_ns": verified_completion,
            "first_clock_ns": clocks[0],
            "last_clock_ns": clocks[-1],
        },
        "brain_turns": _turn_summary(events),
        "provider": {
            "provider_call_count": len(provider_starts),
            "request_started_count": len(provider_starts),
            "completed_count": len(provider_completions),
            "span_pairing": provider_span_pairing,
            "unmatched_request_span_ids": provider_span_pairing["unmatched_request_span_ids"],
            "completion_without_request_span_ids": provider_span_pairing[
                "completion_without_request_span_ids"
            ],
            "error_count": len(provider_errors),
            "retry_count": len(retries),
            "http_429_response_count": len(http_429_responses),
            "http_429_retry_trigger_count": len(http_429_retry_triggers),
            "call_durations": call_durations,
            "duration_missing_reasons": list(dict.fromkeys(duration_missing_reasons)),
            "accounting_status_counts": dict(sorted(accounting_status_counts.items())),
            "input_tokens": _counter_summary(
                provider_completions, len(provider_starts), "input_tokens", integer_only=True,
                span_pairing_complete=span_pairing_complete,
                span_pairing_missing_reasons=span_pairing_missing_reasons,
            ),
            "output_tokens": _counter_summary(
                provider_completions, len(provider_starts), "output_tokens", integer_only=True,
                span_pairing_complete=span_pairing_complete,
                span_pairing_missing_reasons=span_pairing_missing_reasons,
            ),
            "reported_cost": _cost_summary(
                provider_completions,
                len(provider_starts),
                span_pairing_complete=span_pairing_complete,
                span_pairing_missing_reasons=span_pairing_missing_reasons,
            ),
        },
        "retry_evidence": [_event_evidence(row) for row in retries],
        "provider_error_evidence": [_event_evidence(row) for row in provider_errors],
        "first_tool_start": first_tool_start_summary,
        "first_playback": first_playback_summary,
        "interpretation": {
            "trace_join": (
                "Events were filtered by exact trace_id and the recorded request-end-to-verified-"
                "completion monotonic clock window."
            ),
            "http_429_counts": (
                "Response count uses llm.completed.http_status; retry-trigger count uses "
                "llm.retrying.reason. A provider error body can signal code 429 without HTTP 429."
            ),
            "cost": (
                "A total is reported only when every provider request has a completed event "
                "with available cost in the same currency."
            ),
            "first_playback": (
                "Playback timing is reported separately; it is not evidence of a meaningful acknowledgment."
            ),
        },
    }


def summarize_files(record_path: Path, event_path: Path) -> dict[str, Any]:
    try:
        record = json.loads(record_path.read_bytes())
    except json.JSONDecodeError:
        raise
    if not isinstance(record, dict):
        raise RunEventSummaryError("run record must be a JSON object")
    trace_id = record.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        raise RunEventSummaryError("run record trace_id is required")
    events = _matching_events(event_path, trace_id)
    return summarize_run_events(record, events)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path, help="Private per-run JSON record")
    parser.add_argument("--events", required=True, type=Path, help="Private telemetry JSONL event log")
    args = parser.parse_args()
    try:
        summary = summarize_files(args.record, args.events)
    except (OSError, UnicodeError, json.JSONDecodeError, RunEventSummaryError) as exc:
        print(f"run event summary failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
