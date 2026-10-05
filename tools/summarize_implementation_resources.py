#!/usr/bin/env python3
"""Join a Loop 6 run record to a private resource-sampler trace.

The summary uses only the versioned JSON emitted by
tools/sample_process_resources.py. Cgroup measurements describe the whole
cgroup and can include unrelated processes. memory.current maxima are sampled
observations, so short peaks between samples can be missed; the sampler's
cgroup memory.peak counter is not treated as task-attributed.

Example:

    python tools/summarize_implementation_resources.py \\
        --record /private/run.json --trace /private/resources.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


SAMPLER_SCHEMA_VERSION = 1
DEFAULT_RECOVERY_OFFSETS_SEC = (0, 30, 120)
DEFAULT_RECOVERY_TOLERANCE_SEC = 2.0
CGROUP_ATTRIBUTION_NOTE = (
    "Cgroup values describe the whole cgroup and may include unrelated processes."
)
PROCESS_TREE_NOTE = (
    "Process-tree membership is a best-effort snapshot; brief children between samples may be missed."
)
SAMPLED_PEAK_NOTE = (
    "Observed maximum of sampled cgroup memory.current values during the request; "
    "brief peaks between samples may be missed. The cgroup memory.peak counter "
    "may predate this request and is not treated as task-attributed."
)


class ResourceTraceError(ValueError):
    """Raised when run metadata and its resource trace cannot be joined safely."""


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResourceTraceError(f"{name} must be an integer >= {minimum}")
    return value


def _metric_value(sample: dict[str, Any], metric: str) -> int | None:
    if metric in {"process_tree_rss_bytes", "process_tree_pss_bytes"}:
        tree = sample.get("process_tree")
        if not isinstance(tree, dict):
            return None
        short_name = "rss" if metric.endswith("rss_bytes") else "pss"
        if tree.get(f"{short_name}_complete") is not True:
            return None
        value = tree.get(f"{short_name}_bytes")
    elif metric == "cgroup_memory_current_bytes":
        cgroup = sample.get("cgroup")
        value = cgroup.get("memory_current_bytes") if isinstance(cgroup, dict) else None
    elif metric == "cgroup_memory_swap_current_bytes":
        cgroup = sample.get("cgroup")
        value = cgroup.get("memory_swap_current_bytes") if isinstance(cgroup, dict) else None
    elif metric == "cgroup_cpu_usage_usec":
        cgroup = sample.get("cgroup")
        value = cgroup.get("cpu_usage_usec") if isinstance(cgroup, dict) else None
    elif metric in {"available_ram_bytes", "swap_used_bytes"}:
        host = sample.get("host_memory")
        if not isinstance(host, dict):
            return None
        value = host.get(metric)
    else:
        raise AssertionError(f"unknown resource metric: {metric}")

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _metric_missing_reason(sample: dict[str, Any], metric: str) -> str | None:
    if metric in {"process_tree_rss_bytes", "process_tree_pss_bytes"}:
        tree = sample.get("process_tree")
        if not isinstance(tree, dict):
            return "process-tree sample is missing"
        short_name = "rss" if metric.endswith("rss_bytes") else "pss"
        reasons = tree.get(f"{short_name}_missing_reasons") or []
        membership_reasons = tree.get("membership_missing_reasons") or []
        reason_text = "; ".join(dict.fromkeys(
            str(item.get("reason"))
            for item in [*reasons, *membership_reasons]
            if isinstance(item, dict) and item.get("reason")
        ))
        if reason_text:
            return reason_text
        if tree.get("root_missing_reason"):
            return str(tree["root_missing_reason"])
        if tree.get(f"{short_name}_complete") is not True:
            return f"process-tree {short_name.upper()} total is incomplete"
        return None

    if metric.startswith("cgroup_"):
        cgroup = sample.get("cgroup")
        if not isinstance(cgroup, dict):
            return "cgroup sample is missing"
        field = metric.removeprefix("cgroup_")
        reasons = cgroup.get("missing_reasons")
        if isinstance(reasons, dict):
            reason = reasons.get(field)
            if reason:
                return str(reason)
            if metric == "cgroup_cpu_usage_usec" and reasons.get("cpu_stat"):
                return str(reasons["cpu_stat"])
            if reasons.get("cgroup"):
                return str(reasons["cgroup"])
        return f"{field} is unavailable in the sample"

    host = sample.get("host_memory")
    if not isinstance(host, dict):
        return "host-memory sample is missing"
    reasons = host.get("missing_reasons")
    if isinstance(reasons, dict) and reasons.get(metric):
        return str(reasons[metric])
    return f"{metric} is unavailable in the sample"


def _summary_metric(
    samples: list[dict[str, Any]], metric: str, *, mode: str,
) -> dict[str, Any]:
    reasons = list(dict.fromkeys(
        reason
        for sample in samples
        if _metric_value(sample, metric) is None
        if (reason := _metric_missing_reason(sample, metric)) is not None
    ))
    observed = [
        (sample["clock_ns"], value)
        for sample in samples
        if (value := _metric_value(sample, metric)) is not None
    ]
    if not observed:
        if not reasons:
            reasons = ["no samples in the requested interval"]
        return {"value": None, "missing_reasons": reasons}

    if mode == "max":
        clock_ns, value = max(observed, key=lambda item: item[1])
    elif mode == "min":
        clock_ns, value = min(observed, key=lambda item: item[1])
    else:
        raise AssertionError(f"unknown summary mode: {mode}")
    result = {"value": value, "clock_ns": clock_ns}
    if reasons:
        result["missing_reasons"] = reasons
    return result


def _recovery_metrics(sample: dict[str, Any]) -> dict[str, Any]:
    modes = {
        "process_tree_rss_bytes": "max",
        "process_tree_pss_bytes": "max",
        "cgroup_memory_current_bytes": "max",
        "cgroup_memory_swap_current_bytes": "max",
        "cgroup_cpu_usage_usec": "max",
        "available_ram_bytes": "min",
        "swap_used_bytes": "max",
    }
    result: dict[str, Any] = {}
    for metric, mode in modes.items():
        item = _summary_metric([sample], metric, mode=mode)
        result[metric] = item["value"]
        if item["value"] is None:
            result.setdefault("missing_reasons", {})[metric] = item["missing_reasons"]
    return result


def _validate_and_get_samples(
    run_record: dict[str, Any], trace: dict[str, Any], trace_bytes: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
    for field in ("run_id", "scenario"):
        value = run_record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ResourceTraceError(f"run record {field} must be a nonempty string")

    binding = run_record.get("resource_trace")
    if not isinstance(binding, dict):
        raise ResourceTraceError("run record is missing resource_trace metadata")
    metadata = trace.get("metadata")
    if not isinstance(metadata, dict):
        raise ResourceTraceError("sampler trace is missing metadata")
    if not isinstance(metadata.get("interrupted"), bool):
        raise ResourceTraceError("trace.metadata.interrupted must be a boolean")

    expected_schema = _integer(
        binding.get("sampler_schema_version"),
        "resource_trace.sampler_schema_version",
        minimum=1,
    )
    actual_schema = _integer(trace.get("schema_version"), "trace.schema_version", minimum=1)
    if expected_schema != SAMPLER_SCHEMA_VERSION or actual_schema != expected_schema:
        raise ResourceTraceError(
            f"unsupported or mismatched sampler schema version: record={expected_schema}, trace={actual_schema}"
        )

    expected_hash = binding.get("sha256")
    actual_hash = hashlib.sha256(trace_bytes).hexdigest()
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ResourceTraceError("resource trace SHA-256 does not match the run record")

    target_pid = _integer(binding.get("target_pid"), "resource_trace.target_pid", minimum=1)
    target_start = _integer(
        binding.get("target_starttime_ticks"),
        "resource_trace.target_starttime_ticks",
    )
    trace_pid = _integer(metadata.get("target_pid"), "trace.metadata.target_pid", minimum=1)
    trace_start = _integer(
        metadata.get("target_starttime_ticks"),
        "trace.metadata.target_starttime_ticks",
    )
    if (trace_pid, trace_start) != (target_pid, target_start):
        raise ResourceTraceError(
            "resource trace target PID/start-time identity does not match the run record"
        )
    sampler_run_id = binding.get("sampler_run_id")
    if not isinstance(sampler_run_id, str) or not sampler_run_id:
        raise ResourceTraceError("resource_trace.sampler_run_id is required")
    if metadata.get("run_id") != sampler_run_id:
        raise ResourceTraceError("resource trace sampler run ID does not match the run record")

    event_clock = run_record.get("event_clock_ns")
    if not isinstance(event_clock, dict):
        raise ResourceTraceError("run record is missing event_clock_ns")
    request_end = _integer(event_clock.get("request_end"), "event_clock_ns.request_end", minimum=1)
    completion = _integer(
        event_clock.get("verified_completion"),
        "event_clock_ns.verified_completion",
        minimum=1,
    )
    if completion < request_end:
        raise ResourceTraceError("verified completion precedes request end")

    samples = trace.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ResourceTraceError("resource trace has no samples")
    count = _integer(metadata.get("sample_count"), "trace.metadata.sample_count", minimum=1)
    if count != len(samples):
        raise ResourceTraceError("trace metadata sample_count does not match samples array")

    previous_clock = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ResourceTraceError(f"trace sample {index} is not an object")
        sample_clock = _integer(
            sample.get("clock_ns"),
            f"trace.samples[{index}].clock_ns",
            minimum=1,
        )
        if sample_clock <= previous_clock:
            raise ResourceTraceError("resource sample clocks must be strictly increasing")
        previous_clock = sample_clock
        tree = sample.get("process_tree")
        if not isinstance(tree, dict):
            raise ResourceTraceError(f"trace sample {index} is missing process_tree")
        sample_root_pid = _integer(
            tree.get("root_pid"),
            f"trace.samples[{index}].process_tree.root_pid",
            minimum=1,
        )
        sample_root_start = _integer(
            tree.get("root_starttime_ticks"),
            f"trace.samples[{index}].process_tree.root_starttime_ticks",
        )
        if (sample_root_pid, sample_root_start) != (target_pid, target_start):
            raise ResourceTraceError(
                f"trace sample {index} process identity does not match the run record"
            )

    started = _integer(metadata.get("started_clock_ns"), "trace.metadata.started_clock_ns", minimum=1)
    ended = _integer(metadata.get("ended_clock_ns"), "trace.metadata.ended_clock_ns", minimum=1)
    if samples[0]["clock_ns"] < started or samples[-1]["clock_ns"] > ended or ended < started:
        raise ResourceTraceError("resource sample clocks fall outside sampler metadata bounds")
    return samples, metadata, request_end, completion


def summarize_resource_trace(
    run_record: dict[str, Any],
    trace: dict[str, Any],
    trace_bytes: bytes,
    *,
    recovery_offsets_sec: tuple[int, ...] = DEFAULT_RECOVERY_OFFSETS_SEC,
    recovery_tolerance_sec: float = DEFAULT_RECOVERY_TOLERANCE_SEC,
) -> dict[str, Any]:
    """Validate a run/trace join and summarize task-window and recovery samples."""
    if (
        isinstance(recovery_tolerance_sec, bool)
        or not isinstance(recovery_tolerance_sec, (int, float))
        or not math.isfinite(recovery_tolerance_sec)
        or recovery_tolerance_sec < 0
    ):
        raise ResourceTraceError("recovery tolerance must be finite and nonnegative")
    if not recovery_offsets_sec or recovery_offsets_sec[0] != 0:
        raise ResourceTraceError("recovery offsets must be nonempty and begin at 0 seconds")
    if any(
        isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
        for offset in recovery_offsets_sec
    ):
        raise ResourceTraceError("recovery offsets must be nonnegative integer seconds")
    if tuple(sorted(set(recovery_offsets_sec))) != recovery_offsets_sec:
        raise ResourceTraceError("recovery offsets must be strictly increasing")
    offset_gaps_sec = [
        later - earlier
        for earlier, later in zip(recovery_offsets_sec, recovery_offsets_sec[1:])
    ]
    if offset_gaps_sec and recovery_tolerance_sec >= min(offset_gaps_sec):
        raise ResourceTraceError(
            "recovery tolerance must be smaller than the smallest gap between recovery offsets"
        )
    tolerance_ns_float = recovery_tolerance_sec * 1_000_000_000
    if not math.isfinite(tolerance_ns_float):
        raise ResourceTraceError("recovery tolerance is too large")
    tolerance_ns = int(tolerance_ns_float)

    samples, metadata, request_end, completion = _validate_and_get_samples(
        run_record, trace, trace_bytes
    )
    in_task = [
        sample for sample in samples
        if request_end <= sample["clock_ns"] <= completion
    ]
    if not in_task:
        raise ResourceTraceError("resource trace has no samples within the request interval")

    modes = {
        "process_tree_rss_bytes": "max",
        "process_tree_pss_bytes": "max",
        "cgroup_memory_current_bytes": "max",
        "cgroup_memory_swap_current_bytes": "max",
        "available_ram_bytes": "min",
        "swap_used_bytes": "max",
    }
    interval_summary = {
        metric: _summary_metric(in_task, metric, mode=mode)
        for metric, mode in modes.items()
    }

    cpu_samples = [
        (sample["clock_ns"], _metric_value(sample, "cgroup_cpu_usage_usec"))
        for sample in in_task
    ]
    cpu_missing_reasons = list(dict.fromkeys(
        reason
        for sample in in_task
        if _metric_value(sample, "cgroup_cpu_usage_usec") is None
        if (reason := _metric_missing_reason(sample, "cgroup_cpu_usage_usec")) is not None
    ))
    cpu_samples = [(clock, value) for clock, value in cpu_samples if value is not None]
    if len(cpu_samples) < 2:
        cpu_delta: dict[str, Any] = {
            "value": None,
            "missing_reasons": cpu_missing_reasons or [
                "at least two valid in-task cgroup CPU samples are required"
            ],
        }
    elif any(
        current[1] < previous[1]
        for previous, current in zip(cpu_samples, cpu_samples[1:])
    ):
        cpu_delta = {
            "value": None,
            "missing_reasons": ["cgroup CPU counter decreased during the request interval"],
        }
    else:
        cpu_delta = {
            "value": cpu_samples[-1][1] - cpu_samples[0][1],
            "start_clock_ns": cpu_samples[0][0],
            "end_clock_ns": cpu_samples[-1][0],
            "method": "difference between first and last valid in-task samples",
        }
        if cpu_missing_reasons:
            cpu_delta["missing_reasons"] = cpu_missing_reasons

    recovery = []
    for offset_sec in recovery_offsets_sec:
        target_clock_ns = completion + offset_sec * 1_000_000_000
        after_target = [
            sample for sample in samples
            if target_clock_ns <= sample["clock_ns"] <= target_clock_ns + tolerance_ns
        ]
        if not after_target:
            raise ResourceTraceError(
                f"no resource sample at or after recovery +{offset_sec}s within "
                f"{recovery_tolerance_sec:g}s tolerance"
            )
        selected = min(after_target, key=lambda sample: sample["clock_ns"])
        recovery.append({
            "target_offset_sec": offset_sec,
            "target_clock_ns": target_clock_ns,
            "sample_clock_ns": selected["clock_ns"],
            "sample_offset_from_completion_ms": round(
                (selected["clock_ns"] - completion) / 1_000_000, 3
            ),
            "sample_delta_from_target_ms": round(
                (selected["clock_ns"] - target_clock_ns) / 1_000_000, 3
            ),
            "metrics": _recovery_metrics(selected),
        })

    trace_warnings = {
        "cgroup": metadata.get("cgroup_attribution_warning") or CGROUP_ATTRIBUTION_NOTE,
        "process_tree": metadata.get("process_snapshot_warning") or PROCESS_TREE_NOTE,
    }
    return {
        "run_id": run_record.get("run_id"),
        "scenario": run_record.get("scenario"),
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "sampler_run_id": metadata["run_id"],
        "target": {
            "pid": metadata["target_pid"],
            "starttime_ticks": metadata["target_starttime_ticks"],
        },
        "clock_window": {
            "request_end_clock_ns": request_end,
            "verified_completion_clock_ns": completion,
            "sampler_interrupted": metadata["interrupted"],
        },
        "in_task": {
            "sample_count": len(in_task),
            "observed_sampled_extrema": interval_summary,
            "cgroup_cpu_usage_delta_usec": cpu_delta,
        },
        "recovery": recovery,
        "interpretation": {
            "cgroup_memory_current": SAMPLED_PEAK_NOTE,
            "cgroup_attribution": trace_warnings["cgroup"],
            "process_tree": trace_warnings["process_tree"],
            "recovery_selection": (
                "First sample at or after each requested recovery offset, within the configured tolerance."
            ),
        },
    }


def summarize_files(
    record_path: Path,
    trace_path: Path,
    *,
    recovery_tolerance_sec: float = DEFAULT_RECOVERY_TOLERANCE_SEC,
) -> dict[str, Any]:
    record = json.loads(record_path.read_bytes())
    trace_bytes = trace_path.read_bytes()
    trace = json.loads(trace_bytes)
    if not isinstance(record, dict):
        raise ResourceTraceError("run record must be a JSON object")
    if not isinstance(trace, dict):
        raise ResourceTraceError("resource trace must be a JSON object")
    return summarize_resource_trace(
        record,
        trace,
        trace_bytes,
        recovery_tolerance_sec=recovery_tolerance_sec,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path, help="Private per-run JSON record")
    parser.add_argument("--trace", required=True, type=Path, help="Private sampler JSON trace")
    parser.add_argument(
        "--recovery-tolerance-sec",
        type=float,
        default=DEFAULT_RECOVERY_TOLERANCE_SEC,
        help=(
            "Maximum time after each recovery marker to accept a sample "
            f"(default: {DEFAULT_RECOVERY_TOLERANCE_SEC:g}s; must be less than 30s "
            "to keep the default recovery samples distinct)"
        ),
    )
    args = parser.parse_args()
    try:
        summary = summarize_files(
            args.record,
            args.trace,
            recovery_tolerance_sec=args.recovery_tolerance_sec,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ResourceTraceError) as exc:
        print(f"resource summary failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
