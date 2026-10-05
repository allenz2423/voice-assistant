#!/usr/bin/env python3
"""Summarize comparable implementation runs without dropping raw outcomes.

Input is a private JSONL file containing one run record per line. Samples are
grouped by exact scenario, route, phase, and run-context identity (host, OS,
app, build, commit, and starting-state fingerprint). Elapsed percentiles use
the recorded request_end_to_verified_state metric; they are not voice-to-voice
latency measurements.

Example:

    python tools/summarize_implementation_run_series.py --records /private/runs.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


MIN_PERCENTILE_SAMPLES = 10
ROUTE_FIELDS = ("provider", "model", "config_id")
PHASES = frozenset({"cold", "warm", "long_wait", "recovery"})


class RunSeriesError(ValueError):
    """Raised when a run series cannot be compared safely."""


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunSeriesError(f"{name} must be a non-empty string")
    return value.strip()


def _count(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunSeriesError(f"{name} must be a non-negative integer or null")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        return None
    return result


def _elapsed_ms(record: dict[str, Any]) -> tuple[float | None, str | None]:
    timing = record.get("timing_ms")
    value = timing.get("request_end_to_verified_state") if isinstance(timing, dict) else None
    elapsed = _number(value)
    if elapsed is None:
        return None, "timing_ms.request_end_to_verified_state is missing or invalid"
    return elapsed, None


def _sample_row(record: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    outcome = _label(record.get("outcome"), f"run {run_id!r} outcome")
    failure_class = record.get("failure_class")
    if failure_class is not None and not isinstance(failure_class, str):
        raise RunSeriesError(f"run {run_id!r} failure_class must be a string or null")

    elapsed_ms, elapsed_missing_reason = _elapsed_ms(record)
    sample = {
        "run_id": run_id,
        "fixture_id": record.get("fixture_id") if isinstance(record.get("fixture_id"), str) else None,
        "outcome": outcome,
        "failure_class": failure_class,
        "request_end_to_verified_state_ms": elapsed_ms,
        "elapsed_missing_reason": elapsed_missing_reason,
    }
    missing_reasons: dict[str, str] = {}
    for source, target in (
        ("tool_model_rounds", "tool_model_rounds"),
        ("provider_call_count", "provider_call_count"),
        ("retries", "retries"),
        ("http_429_response_count", "http_429_response_count"),
        ("http_429_retry_trigger_count", "http_429_retry_trigger_count"),
    ):
        value = record.get(source)
        sample[target] = _count(value, f"run {run_id!r} {source}")
        if sample[target] is None:
            missing_reasons[target] = f"{source} is missing"
    sample["missing_reasons"] = missing_reasons

    usage = record.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    currency = usage.get("currency")
    accounting_status = usage.get("accounting_status")
    source_missing_reasons = usage.get("missing_reasons")
    if source_missing_reasons is None:
        source_missing_reasons = []
    elif not isinstance(source_missing_reasons, list):
        source_missing_reasons = ["usage.missing_reasons must be an array"]
    else:
        raw_source_reasons = source_missing_reasons
        source_missing_reasons = []
        invalid_reason_count = 0
        for reason in raw_source_reasons:
            if isinstance(reason, str) and reason.strip():
                source_missing_reasons.append(" ".join(reason.split())[:240])
            else:
                invalid_reason_count += 1
        if invalid_reason_count:
            source_missing_reasons.append(
                f"{invalid_reason_count} source missing-reason value(s) were invalid"
            )
        if len(source_missing_reasons) > 20:
            omitted_count = len(source_missing_reasons) - 20
            source_missing_reasons = source_missing_reasons[:20]
            source_missing_reasons.append(
                f"{omitted_count} additional source missing-reason value(s) omitted"
            )
    usage_field_missing_reasons: dict[str, list[str]] = {}
    reported_cost = _number(usage.get("reported_cost"))
    if reported_cost is None:
        reasons = ["usage.reported_cost is missing or invalid"]
        if isinstance(accounting_status, str) and accounting_status:
            reasons.append(f"accounting_status={accounting_status[:120]}")
        reasons.extend(source_missing_reasons)
        usage_field_missing_reasons["reported_cost"] = list(dict.fromkeys(reasons))
    normalized_currency = (
        currency.strip()
        if isinstance(currency, str) and currency.strip() and len(currency.strip()) <= 24
        else None
    )
    if normalized_currency is None:
        usage_field_missing_reasons["currency"] = ["usage.currency is missing or invalid"]
    normalized_accounting_status = (
        accounting_status.strip()
        if (
            isinstance(accounting_status, str)
            and accounting_status.strip()
            and len(accounting_status.strip()) <= 120
        )
        else None
    )
    if normalized_accounting_status is None:
        usage_field_missing_reasons["accounting_status"] = [
            "usage.accounting_status is missing or invalid"
        ]
    sample["usage"] = {
        "reported_cost": reported_cost,
        "currency": normalized_currency,
        "accounting_status": normalized_accounting_status,
        "missing_reasons": source_missing_reasons,
        "field_missing_reasons": usage_field_missing_reasons,
    }
    return sample


def _percentiles(
    values: list[float], *, context_missing_fields: list[str],
) -> dict[str, Any]:
    count = len(values)
    if context_missing_fields:
        return {
            "sample_count": count,
            "median_ms": None,
            "p95_ms": None,
            "missing_reason": (
                "matched context is missing: " + ", ".join(context_missing_fields)
            ),
        }
    if count < MIN_PERCENTILE_SAMPLES:
        return {
            "sample_count": count,
            "median_ms": None,
            "p95_ms": None,
            "missing_reason": (
                f"requires at least {MIN_PERCENTILE_SAMPLES} valid elapsed samples; found {count}"
            ),
        }

    ordered = sorted(values)
    middle = count // 2
    median = (
        ordered[middle]
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    nearest_rank = math.ceil(0.95 * count)
    return {
        "sample_count": count,
        "median_ms": round(median, 3),
        "p95_ms": round(ordered[nearest_rank - 1], 3),
        "p95_method": "nearest rank (ceil(0.95 * n))",
        "missing_reason": None,
    }


def _optional_label(record: dict[str, Any], field: str, run_id: str) -> str | None:
    value = record.get(field)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _label(value, f"run {run_id!r} {field}")


def _starting_state_digest(record: dict[str, Any], run_id: str) -> str | None:
    value = record.get("starting_state")
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RunSeriesError(f"run {run_id!r} starting_state is not JSON serializable") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    seen_run_ids: set[str] = set()

    for line_number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise RunSeriesError(f"run record {line_number} must be a JSON object")
        run_id = _label(record.get("run_id"), f"run record {line_number} run_id")
        if run_id in seen_run_ids:
            raise RunSeriesError(f"duplicate run_id {run_id!r}")
        seen_run_ids.add(run_id)

        scenario = _label(record.get("scenario"), f"run {run_id!r} scenario")
        phase = _label(record.get("phase"), f"run {run_id!r} phase")
        if phase not in PHASES:
            raise RunSeriesError(
                f"run {run_id!r} phase must be one of {', '.join(sorted(PHASES))}"
            )
        route = record.get("route")
        if not isinstance(route, dict):
            raise RunSeriesError(f"run {run_id!r} route must be an object")
        route_values = tuple(
            _label(route.get(field), f"run {run_id!r} route.{field}")
            for field in ROUTE_FIELDS
        )
        match_context = (
            _optional_label(record, "host", run_id),
            _optional_label(record, "os", run_id),
            _optional_label(record, "app", run_id),
            _optional_label(record, "build", run_id),
            _optional_label(record, "commit", run_id),
            _starting_state_digest(record, run_id),
        )
        key = (scenario, *route_values, phase, *match_context)
        groups.setdefault(key, []).append(_sample_row(record, run_id=run_id))

    summaries = []
    for key in sorted(groups, key=lambda value: json.dumps(value, ensure_ascii=False)):
        (
            scenario, provider, model, config_id, phase,
            host, os_name, app, build, commit, starting_state_sha256,
        ) = key
        samples = groups[key]
        elapsed = [
            row["request_end_to_verified_state_ms"]
            for row in samples
            if row["request_end_to_verified_state_ms"] is not None
        ]
        outcomes = Counter(row["outcome"] for row in samples)
        retries = [row["retries"] for row in samples if row["retries"] is not None]
        missing_retry_count = len(samples) - len(retries)
        context = {
            "host": host,
            "os": os_name,
            "app": app,
            "build": build,
            "commit": commit,
            "starting_state_sha256": starting_state_sha256,
        }
        missing_context_fields = [name for name, value in context.items() if value is None]
        summaries.append({
            "key": {
                "scenario": scenario,
                "route": {
                    "provider": provider,
                    "model": model,
                    "config_id": config_id,
                },
                "phase": phase,
                "match_context": context,
            },
            "sample_count": len(samples),
            "outcome_counts": dict(sorted(outcomes.items())),
            "retry_sample_count": sum(value > 0 for value in retries),
            "retry_observed_sample_count": len(retries),
            "known_retry_count_total": sum(retries),
            "missing_retry_count": missing_retry_count,
            "retry_count_complete": missing_retry_count == 0,
            "elapsed": {
                **_percentiles(
                    elapsed, context_missing_fields=missing_context_fields,
                ),
                "raw_sample_count": len(samples),
                "missing_elapsed_count": len(samples) - len(elapsed),
            },
            "samples": samples,
        })

    return {
        "group_count": len(summaries),
        "groups": summaries,
        "interpretation": (
            "Groups use exact scenario, provider/model/config_id route, phase, host, OS, app, build, "
            "commit, and a SHA-256 fingerprint of starting_state. All outcomes, including failures "
            "and retries, remain in samples. Elapsed percentiles use recorded "
            "request_end_to_verified_state values and are not voice-to-voice latency. Median and "
            "nearest-rank p95 are withheld until at least 10 valid elapsed samples are present and "
            "all matched context fields are recorded."
        ),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError:
        raise
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunSeriesError(f"run record line {line_number} is not valid JSON") from exc
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", required=True, type=Path,
        help="private JSONL file containing one run record per line",
    )
    args = parser.parse_args(argv)
    try:
        summary = summarize_records(load_records(args.records))
    except (OSError, RunSeriesError) as exc:
        print(f"run series summary failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
