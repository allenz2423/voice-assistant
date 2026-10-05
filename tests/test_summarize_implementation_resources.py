from __future__ import annotations

import hashlib
import json
import sys

import pytest

from tools.summarize_implementation_resources import (
    ResourceTraceError,
    main,
    summarize_files,
    summarize_resource_trace,
)


def _sample(
    clock_ns: int,
    *,
    memory: int | None,
    cpu: int,
    pid: int = 1234,
    start: int = 9001,
    pss: int | None = 80,
    rss: int | None = 120,
    pss_complete: bool = True,
    rss_complete: bool = True,
    pss_reason: str = "PSS unavailable",
    cgroup_memory_reason: str | None = None,
    membership_missing_reasons: list[dict] | None = None,
) -> dict:
    cgroup_reasons = {}
    if cgroup_memory_reason:
        cgroup_reasons["memory_current_bytes"] = cgroup_memory_reason
    return {
        "clock_ns": clock_ns,
        "wall_time_utc": "2026-10-05T00:00:00.000Z",
        "process_tree": {
            "root_pid": pid,
            "root_starttime_ticks": start,
            "root_missing_reason": None,
            "membership_complete": not membership_missing_reasons,
            "membership_missing_reasons": membership_missing_reasons or [],
            "rss_bytes": rss if rss_complete else None,
            "pss_bytes": pss if pss_complete else None,
            "rss_complete": rss_complete,
            "pss_complete": pss_complete,
            "rss_missing_reasons": [] if rss_complete else [{"reason": "RSS permission denied"}],
            "pss_missing_reasons": [] if pss_complete else [{"reason": pss_reason}],
        },
        "cgroup": {
            "memory_current_bytes": memory,
            "memory_peak_bytes": memory + 50_000 if memory is not None else None,
            "memory_swap_current_bytes": 20,
            "cpu_usage_usec": cpu,
            "missing_reasons": cgroup_reasons,
            "attribution_warning": "whole cgroup; may include unrelated processes",
        },
        "host_memory": {
            "available_ram_bytes": 1_000_000 - memory if memory is not None else None,
            "swap_used_bytes": 0,
            "missing_reasons": {},
        },
    }


def _write_inputs(
    tmp_path,
    samples: list[dict],
    *,
    target_pid: int = 1234,
    target_start: int = 9001,
    started: int = 90_000_000_000,
    ended: int = 250_000_000_000,
) -> tuple:
    tmp_path.mkdir(parents=True, exist_ok=True)
    trace = {
        "schema_version": 1,
        "metadata": {
            "run_id": "sampler-trace-abc",
            "target_pid": target_pid,
            "target_starttime_ticks": target_start,
            "started_clock_ns": started,
            "ended_clock_ns": ended,
            "sample_count": len(samples),
            "interrupted": False,
            "cgroup_attribution_warning": "whole cgroup; may include unrelated processes",
            "process_snapshot_warning": "process-tree membership is best effort",
        },
        "samples": samples,
    }
    trace_bytes = (json.dumps(trace, separators=(",", ":")) + "\n").encode()
    record = {
        "run_id": "loop6-run-1",
        "scenario": "X1",
        "event_clock_ns": {
            "request_end": 100_000_000_000,
            "verified_completion": 110_000_000_000,
        },
        "resource_trace": {
            "sampler_schema_version": 1,
            "sampler_run_id": "sampler-trace-abc",
            "sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "target_pid": target_pid,
            "target_starttime_ticks": target_start,
        },
    }
    record_path = tmp_path / "run.json"
    trace_path = tmp_path / "resources.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    trace_path.write_bytes(trace_bytes)
    return record_path, trace_path, trace


def test_summarizer_reports_interval_sampled_extrema_and_cpu_delta(tmp_path):
    samples = [
        _sample(95_000_000_000, memory=100, cpu=10),
        _sample(100_500_000_000, memory=200, cpu=30),
        _sample(105_000_000_000, memory=900, cpu=60),
        _sample(109_500_000_000, memory=500, cpu=90),
        _sample(110_000_000_000, memory=450, cpu=95),
        _sample(140_500_000_000, memory=400, cpu=150),
        _sample(230_500_000_000, memory=300, cpu=220),
    ]
    record_path, trace_path, _ = _write_inputs(tmp_path, samples)

    summary = summarize_files(record_path, trace_path)

    in_task = summary["in_task"]
    assert in_task["sample_count"] == 4
    peak = in_task["observed_sampled_extrema"]["cgroup_memory_current_bytes"]
    assert peak == {"value": 900, "clock_ns": 105_000_000_000}
    assert in_task["observed_sampled_extrema"]["process_tree_pss_bytes"]["value"] == 80
    assert in_task["cgroup_cpu_usage_delta_usec"] == {
        "value": 65,
        "start_clock_ns": 100_500_000_000,
        "end_clock_ns": 110_000_000_000,
        "method": "difference between first and last valid in-task samples",
    }
    assert "sampled cgroup memory.current values" in summary["interpretation"]["cgroup_memory_current"]
    assert "unrelated processes" in summary["interpretation"]["cgroup_attribution"]
    # The sampler's cgroup high-water counter is deliberately not presented as
    # the request's peak: it may have accumulated before this request.
    assert "cgroup_memory_peak_bytes" not in in_task["observed_sampled_extrema"]


def test_summarizer_rejects_a_cpu_counter_drop_between_valid_samples(tmp_path):
    samples = [
        _sample(100_500_000_000, memory=200, cpu=100),
        _sample(105_000_000_000, memory=200, cpu=10),
        _sample(110_000_000_000, memory=200, cpu=200),
        _sample(140_500_000_000, memory=200, cpu=220),
        _sample(230_500_000_000, memory=200, cpu=240),
    ]
    record_path, trace_path, _ = _write_inputs(tmp_path, samples)

    summary = summarize_files(record_path, trace_path)

    assert summary["in_task"]["cgroup_cpu_usage_delta_usec"] == {
        "value": None,
        "missing_reasons": ["cgroup CPU counter decreased during the request interval"],
    }


def test_summarizer_selects_completion_and_30_120_second_recovery_samples(tmp_path):
    samples = [
        _sample(100_000_000_000, memory=300, cpu=10),
        _sample(110_000_000_000, memory=250, cpu=20),
        _sample(110_400_000_000, memory=240, cpu=21),
        _sample(140_800_000_000, memory=200, cpu=30),
        _sample(230_600_000_000, memory=100, cpu=40),
    ]
    record_path, trace_path, _ = _write_inputs(tmp_path, samples)

    summary = summarize_files(record_path, trace_path)

    recovery = summary["recovery"]
    assert [item["target_offset_sec"] for item in recovery] == [0, 30, 120]
    assert [item["sample_clock_ns"] for item in recovery] == [
        110_000_000_000,
        140_800_000_000,
        230_600_000_000,
    ]
    assert recovery[0]["metrics"]["cgroup_memory_current_bytes"] == 250
    assert recovery[1]["sample_delta_from_target_ms"] == 800.0
    assert recovery[1]["metrics"]["available_ram_bytes"] == 999_800
    assert recovery[2]["metrics"]["cgroup_cpu_usage_usec"] == 40


def test_summarizer_rejects_recovery_windows_that_can_reuse_a_sample(tmp_path):
    samples = [
        _sample(100_000_000_000, memory=300, cpu=10),
        _sample(110_000_000_000, memory=250, cpu=20),
        _sample(140_000_000_000, memory=200, cpu=30),
        _sample(230_000_000_000, memory=100, cpu=40),
    ]
    record_path, trace_path, _ = _write_inputs(tmp_path, samples)

    with pytest.raises(ResourceTraceError, match="smaller than the smallest gap"):
        summarize_files(record_path, trace_path, recovery_tolerance_sec=30.0)


def test_summarizer_rejects_recovery_tolerance_that_overflows_nanoseconds(tmp_path):
    samples = [
        _sample(100_000_000_000, memory=200, cpu=10),
        _sample(110_000_000_000, memory=200, cpu=20),
        _sample(140_000_000_000, memory=200, cpu=30),
        _sample(230_000_000_000, memory=200, cpu=40),
    ]
    record_path, trace_path, trace = _write_inputs(tmp_path, samples)
    record = json.loads(record_path.read_text(encoding="utf-8"))

    with pytest.raises(ResourceTraceError, match="tolerance is too large"):
        summarize_resource_trace(
            record,
            trace,
            trace_path.read_bytes(),
            recovery_offsets_sec=(0,),
            recovery_tolerance_sec=1e308,
        )


@pytest.mark.parametrize(
    ("location", "field", "value", "message"),
    [
        ("metadata", "target_pid", 4321, "PID/start-time identity"),
        ("metadata", "target_starttime_ticks", 9999, "PID/start-time identity"),
        ("sample", "root_pid", 4321, "trace sample 0 process identity"),
        ("sample", "root_starttime_ticks", 9999, "trace sample 0 process identity"),
    ],
)
def test_summarizer_rejects_metadata_and_sample_identity_mismatches(
    tmp_path, location, field, value, message,
):
    samples = [
        _sample(100_000_000_000, memory=200, cpu=10),
        _sample(110_000_000_000, memory=200, cpu=20),
        _sample(110_100_000_000, memory=200, cpu=21),
        _sample(140_100_000_000, memory=200, cpu=30),
        _sample(230_100_000_000, memory=200, cpu=40),
    ]
    record_path, trace_path, trace = _write_inputs(tmp_path, samples)
    if location == "metadata":
        trace["metadata"][field] = value
    else:
        trace["samples"][0]["process_tree"][field] = value
    raw = (json.dumps(trace, separators=(",", ":")) + "\n").encode()
    trace_path.write_bytes(raw)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["resource_trace"]["sha256"] = hashlib.sha256(raw).hexdigest()
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ResourceTraceError, match=message):
        summarize_files(record_path, trace_path)


@pytest.mark.parametrize("interrupted", ["false", 1, None])
def test_summarizer_rejects_non_boolean_interrupted_flag(tmp_path, interrupted):
    samples = [
        _sample(100_000_000_000, memory=200, cpu=10),
        _sample(110_000_000_000, memory=200, cpu=20),
        _sample(140_000_000_000, memory=200, cpu=30),
        _sample(230_000_000_000, memory=200, cpu=40),
    ]
    record_path, trace_path, trace = _write_inputs(tmp_path, samples)
    trace["metadata"]["interrupted"] = interrupted
    raw = (json.dumps(trace, separators=(",", ":")) + "\n").encode()
    trace_path.write_bytes(raw)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["resource_trace"]["sha256"] = hashlib.sha256(raw).hexdigest()
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ResourceTraceError, match="interrupted must be a boolean"):
        summarize_files(record_path, trace_path)


def test_summarizer_rejects_missing_120_second_recovery_coverage(tmp_path):
    samples = [
        _sample(100_000_000_000, memory=200, cpu=10),
        _sample(110_000_000_000, memory=200, cpu=20),
        _sample(110_100_000_000, memory=200, cpu=21),
        _sample(140_100_000_000, memory=200, cpu=30),
    ]
    record_path, trace_path, _ = _write_inputs(
        tmp_path, samples, ended=150_000_000_000,
    )

    with pytest.raises(ResourceTraceError, match=r"recovery \+120s"):
        summarize_files(record_path, trace_path)


def test_summarizer_preserves_reasons_for_unavailable_metrics(tmp_path):
    samples = [
        _sample(
            100_000_000_000,
            memory=None,
            cpu=10,
            pss_complete=False,
            pss_reason="PSS blocked by permissions",
            cgroup_memory_reason="memory.current permission denied",
            membership_missing_reasons=[{"reason": "process appeared during sampling"}],
        ),
        _sample(
            105_000_000_000,
            memory=None,
            cpu=20,
            pss_complete=False,
            pss_reason="PSS blocked by permissions",
            cgroup_memory_reason="memory.current permission denied",
        ),
        _sample(
            110_000_000_000,
            memory=200,
            cpu=30,
            pss_complete=False,
            pss_reason="PSS blocked by permissions",
        ),
        _sample(
            110_100_000_000,
            memory=200,
            cpu=31,
            pss_complete=False,
            pss_reason="PSS blocked by permissions",
        ),
        _sample(140_100_000_000, memory=200, cpu=40),
        _sample(230_100_000_000, memory=200, cpu=50),
    ]
    record_path, trace_path, _ = _write_inputs(tmp_path, samples)

    summary = summarize_files(record_path, trace_path)

    extrema = summary["in_task"]["observed_sampled_extrema"]
    assert extrema["process_tree_pss_bytes"]["value"] is None
    assert "PSS blocked by permissions" in extrema["process_tree_pss_bytes"]["missing_reasons"]
    assert any(
        "process appeared during sampling" in reason
        for reason in extrema["process_tree_pss_bytes"]["missing_reasons"]
    )
    assert extrema["cgroup_memory_current_bytes"]["value"] == 200
    assert "memory.current permission denied" in (
        extrema["cgroup_memory_current_bytes"]["missing_reasons"]
    )
    recovery_pss = summary["recovery"][0]["metrics"]
    assert recovery_pss["process_tree_pss_bytes"] is None
    assert recovery_pss["missing_reasons"]["process_tree_pss_bytes"] == [
        "PSS blocked by permissions",
    ]


def test_summarizer_rejects_trace_hash_mismatch(tmp_path):
    samples = [
        _sample(100_000_000_000, memory=200, cpu=10),
        _sample(110_000_000_000, memory=200, cpu=20),
        _sample(110_100_000_000, memory=200, cpu=21),
        _sample(140_100_000_000, memory=200, cpu=30),
        _sample(230_100_000_000, memory=200, cpu=40),
    ]
    record_path, trace_path, _ = _write_inputs(tmp_path, samples)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["resource_trace"]["sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ResourceTraceError, match="SHA-256"):
        summarize_files(record_path, trace_path)


@pytest.mark.parametrize("field", ["run_id", "scenario"])
def test_summarizer_requires_run_identity_fields(tmp_path, field):
    samples = [
        _sample(100_000_000_000, memory=200, cpu=10),
        _sample(110_000_000_000, memory=200, cpu=20),
        _sample(140_000_000_000, memory=200, cpu=30),
        _sample(230_000_000_000, memory=200, cpu=40),
    ]
    record_path, trace_path, _ = _write_inputs(tmp_path, samples)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record[field] = "  "
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ResourceTraceError, match=f"run record {field} must be a nonempty string"):
        summarize_files(record_path, trace_path)


def test_summarizer_rejects_non_monotonic_sample_clocks(tmp_path):
    samples = [
        _sample(100_000_000_000, memory=200, cpu=10),
        _sample(105_000_000_000, memory=200, cpu=20),
        _sample(104_000_000_000, memory=200, cpu=21),
        _sample(140_100_000_000, memory=200, cpu=30),
        _sample(230_100_000_000, memory=200, cpu=40),
    ]
    record_path, trace_path, _ = _write_inputs(tmp_path, samples)
    trace_bytes = trace_path.read_bytes()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["resource_trace"]["sha256"] = hashlib.sha256(trace_bytes).hexdigest()
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ResourceTraceError, match="strictly increasing"):
        summarize_files(record_path, trace_path)


def test_cli_emits_parseable_json_summary(tmp_path, monkeypatch, capsys):
    samples = [
        _sample(100_000_000_000, memory=300, cpu=10),
        _sample(110_000_000_000, memory=250, cpu=20),
        _sample(140_000_000_000, memory=200, cpu=30),
        _sample(230_000_000_000, memory=100, cpu=40),
    ]
    record_path, trace_path, _ = _write_inputs(tmp_path, samples)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_implementation_resources", "--record", str(record_path), "--trace", str(trace_path)],
    )

    assert main() == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["run_id"] == "loop6-run-1"
    assert summary["clock_window"]["sampler_interrupted"] is False
    assert captured.err == ""


def test_cli_reports_binding_error_without_partial_json(tmp_path, monkeypatch, capsys):
    samples = [
        _sample(100_000_000_000, memory=300, cpu=10),
        _sample(110_000_000_000, memory=250, cpu=20),
        _sample(140_000_000_000, memory=200, cpu=30),
        _sample(230_000_000_000, memory=100, cpu=40),
    ]
    record_path, trace_path, _ = _write_inputs(tmp_path, samples)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["resource_trace"]["sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_implementation_resources", "--record", str(record_path), "--trace", str(trace_path)],
    )

    assert main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SHA-256" in captured.err


def test_cli_reports_invalid_utf8_without_traceback(tmp_path, monkeypatch, capsys):
    record_path = tmp_path / "run.json"
    trace_path = tmp_path / "resources.json"
    record_path.write_bytes(b"\xff")
    trace_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_implementation_resources", "--record", str(record_path), "--trace", str(trace_path)],
    )

    assert main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "UnicodeDecodeError" in captured.err
