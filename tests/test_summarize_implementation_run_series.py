from __future__ import annotations

import hashlib
import json

import pytest

from tools.summarize_implementation_run_series import (
    RunSeriesError,
    main,
    summarize_records,
)


def _record(
    index: int,
    elapsed_ms: int | float | None,
    *,
    scenario: str = "F2",
    provider: str = "openrouter",
    model: str = "vendor/model-a",
    config_id: str = "config-a",
    phase: str = "warm",
    host: str = "laptop-a",
    os_name: str = "linux-test",
    app: str = "adam",
    build: str = "build-a",
    commit: str = "commit-a",
    starting_state: object = "known synthetic fixture state",
    outcome: str = "pass",
    failure_class: str | None = None,
    retries: int | None = 0,
) -> dict:
    return {
        "run_id": f"run-{index:02d}",
        "fixture_id": f"fixture-{index:02d}",
        "scenario": scenario,
        "route": {
            "provider": provider,
            "model": model,
            "config_id": config_id,
        },
        "phase": phase,
        "host": host,
        "os": os_name,
        "app": app,
        "build": build,
        "commit": commit,
        "starting_state": starting_state,
        "outcome": outcome,
        "failure_class": failure_class,
        "timing_ms": {
            "request_end_to_verified_state": elapsed_ms,
        },
        "tool_model_rounds": 2,
        "provider_call_count": 2,
        "retries": retries,
        "http_429_response_count": 0,
        "http_429_retry_trigger_count": 0,
        "usage": {
            "reported_cost": 0.001,
            "currency": "credits",
            "accounting_status": "available",
        },
        "observed_state": "do not copy arbitrary raw state",
    }


def _group_for(summary: dict, *, scenario: str = "F2", phase: str = "warm") -> dict:
    expected_state_sha256 = hashlib.sha256(
        json.dumps(
            "known synthetic fixture state",
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    for group in summary["groups"]:
        key = group["key"]
        context = key["match_context"]
        if (
            key["scenario"] == scenario
            and key["phase"] == phase
            and key["route"]["provider"] == "openrouter"
            and key["route"]["model"] == "vendor/model-a"
            and key["route"]["config_id"] == "config-a"
            and context["host"] == "laptop-a"
            and context["os"] == "linux-test"
            and context["app"] == "adam"
            and context["build"] == "build-a"
            and context["commit"] == "commit-a"
            and context["starting_state_sha256"] == expected_state_sha256
        ):
            return group
    raise AssertionError(f"group not found: {scenario=} {phase=}")


def test_ten_matched_runs_report_median_p95_and_preserve_failures_retries():
    records = [_record(index, index * 10) for index in range(1, 11)]
    records[2].update(outcome="fail", failure_class="action", retries=1)
    records[8].update(outcome="provider_error", failure_class="provider_transport", retries=2)
    records[8]["http_429_response_count"] = 1
    records[8]["http_429_retry_trigger_count"] = 1

    summary = summarize_records(records)
    group = _group_for(summary)

    assert group["sample_count"] == 10
    assert group["outcome_counts"] == {
        "fail": 1,
        "pass": 8,
        "provider_error": 1,
    }
    assert group["retry_sample_count"] == 2
    assert group["known_retry_count_total"] == 3
    assert group["elapsed"] == {
        "sample_count": 10,
        "median_ms": 55.0,
        "p95_ms": 100.0,
        "p95_method": "nearest rank (ceil(0.95 * n))",
        "missing_reason": None,
        "raw_sample_count": 10,
        "missing_elapsed_count": 0,
    }
    assert len(group["samples"]) == 10
    assert group["samples"][2]["outcome"] == "fail"
    assert group["samples"][2]["retries"] == 1
    assert group["samples"][8]["outcome"] == "provider_error"
    assert group["samples"][8]["http_429_response_count"] == 1
    assert group["samples"][8]["http_429_retry_trigger_count"] == 1
    assert "do not copy arbitrary raw state" not in json.dumps(summary)
    assert "known synthetic fixture state" not in json.dumps(summary)
    assert "voice-to-voice latency" in summary["interpretation"]


def test_groups_do_not_mix_scenario_route_phase_or_matching_context():
    baseline = [_record(index, 10) for index in range(10)]
    different_groups = [
        _record(20 + index, 20, scenario="F3") for index in range(10)
    ]
    different_groups += [
        _record(30 + index, 30, provider="local") for index in range(10)
    ]
    different_groups += [
        _record(40 + index, 40, phase="cold") for index in range(10)
    ]
    different_groups += [
        _record(50 + index, 50, host="laptop-b") for index in range(10)
    ]
    different_groups += [
        _record(60 + index, 60, starting_state="different fixture state")
        for index in range(10)
    ]
    different_groups += [
        _record(70 + index, 70, os_name="other-linux") for index in range(10)
    ]
    different_groups += [
        _record(80 + index, 80, app="adam-alt") for index in range(10)
    ]
    different_groups += [
        _record(90 + index, 90, build="build-b") for index in range(10)
    ]
    different_groups += [
        _record(100 + index, 100, commit="commit-b") for index in range(10)
    ]
    different_groups += [
        _record(110 + index, 110, model="vendor/model-b") for index in range(10)
    ]
    different_groups += [
        _record(120 + index, 120, config_id="config-b") for index in range(10)
    ]

    summary = summarize_records(baseline + different_groups)

    assert summary["group_count"] == 12
    assert _group_for(summary)["elapsed"]["median_ms"] == 10.0
    assert _group_for(summary)["elapsed"]["p95_ms"] == 10.0
    assert _group_for(summary, scenario="F3")["elapsed"]["median_ms"] == 20.0
    host_groups = [
        group for group in summary["groups"]
        if group["key"]["match_context"]["host"] == "laptop-b"
    ]
    assert len(host_groups) == 1
    assert host_groups[0]["elapsed"]["median_ms"] == 50.0
    for field, value, expected_elapsed in (
        ("os", "other-linux", 70.0),
        ("app", "adam-alt", 80.0),
        ("build", "build-b", 90.0),
        ("commit", "commit-b", 100.0),
    ):
        matches = [
            group for group in summary["groups"]
            if group["key"]["match_context"][field] == value
        ]
        assert len(matches) == 1
        assert matches[0]["elapsed"]["median_ms"] == expected_elapsed
    model_groups = [
        group for group in summary["groups"]
        if group["key"]["route"]["model"] == "vendor/model-b"
    ]
    config_groups = [
        group for group in summary["groups"]
        if group["key"]["route"]["config_id"] == "config-b"
    ]
    assert model_groups[0]["elapsed"]["median_ms"] == 110.0
    assert config_groups[0]["elapsed"]["median_ms"] == 120.0
    state_groups = [
        group for group in summary["groups"]
        if group["key"]["match_context"]["starting_state_sha256"]
        != _group_for(summary)["key"]["match_context"]["starting_state_sha256"]
    ]
    assert len(state_groups) == 1
    assert state_groups[0]["elapsed"]["median_ms"] == 60.0


def test_percentiles_are_withheld_below_ten_or_when_elapsed_is_missing():
    nine = summarize_records([_record(index, index) for index in range(1, 10)])
    nine_elapsed = _group_for(nine)["elapsed"]
    assert nine_elapsed["sample_count"] == 9
    assert nine_elapsed["median_ms"] is None
    assert nine_elapsed["p95_ms"] is None
    assert "at least 10" in nine_elapsed["missing_reason"]

    records = [_record(index, index) for index in range(1, 11)]
    records[9]["timing_ms"]["request_end_to_verified_state"] = None
    records[9]["retries"] = None
    ten_with_missing = summarize_records(records)
    group = _group_for(ten_with_missing)
    elapsed = group["elapsed"]
    assert elapsed["raw_sample_count"] == 10
    assert elapsed["sample_count"] == 9
    assert elapsed["missing_elapsed_count"] == 1
    assert elapsed["median_ms"] is None
    assert elapsed["p95_ms"] is None
    assert group["samples"][9]["elapsed_missing_reason"]
    assert group["samples"][9]["retries"] is None
    assert group["missing_retry_count"] == 1
    assert group["retry_count_complete"] is False


def test_missing_matched_context_withholds_percentiles():
    records = [_record(index, index) for index in range(1, 11)]
    for record in records:
        record.pop("build")

    elapsed = summarize_records(records)["groups"][0]["elapsed"]

    assert elapsed["sample_count"] == 10
    assert elapsed["median_ms"] is None
    assert elapsed["p95_ms"] is None
    assert elapsed["missing_reason"] == "matched context is missing: build"


def test_sample_preserves_accounting_reasons_for_missing_counters_and_cost():
    record = _record(1, 12)
    record["provider_call_count"] = None
    record["retries"] = None
    record["http_429_response_count"] = None
    record["usage"] = {
        "reported_cost": None,
        "currency": "credits",
        "accounting_status": "unsupported",
        "missing_reasons": ["endpoint does not report cost"],
    }

    sample = summarize_records([record])["groups"][0]["samples"][0]

    assert sample["provider_call_count"] is None
    assert sample["retries"] is None
    assert sample["http_429_response_count"] is None
    assert sample["missing_reasons"] == {
        "provider_call_count": "provider_call_count is missing",
        "retries": "retries is missing",
        "http_429_response_count": "http_429_response_count is missing",
    }
    assert sample["usage"]["accounting_status"] == "unsupported"
    assert sample["usage"]["missing_reasons"] == ["endpoint does not report cost"]
    assert sample["usage"]["field_missing_reasons"]["reported_cost"] == [
        "usage.reported_cost is missing or invalid",
        "accounting_status=unsupported",
        "endpoint does not report cost",
    ]


def test_invalid_cost_is_reported_and_invalid_counter_is_rejected():
    record = _record(1, 12)
    record["usage"].update(
        reported_cost="not-a-number",
        currency=17,
        accounting_status=None,
        missing_reasons=["provider usage payload was invalid"],
    )

    sample = summarize_records([record])["groups"][0]["samples"][0]

    assert sample["usage"]["reported_cost"] is None
    assert sample["usage"]["currency"] is None
    assert sample["usage"]["accounting_status"] is None
    assert sample["usage"]["missing_reasons"] == ["provider usage payload was invalid"]
    assert sample["usage"]["field_missing_reasons"]["reported_cost"] == [
        "usage.reported_cost is missing or invalid",
        "provider usage payload was invalid",
    ]
    assert sample["usage"]["field_missing_reasons"]["currency"] == [
        "usage.currency is missing or invalid",
    ]
    assert sample["usage"]["field_missing_reasons"]["accounting_status"] == [
        "usage.accounting_status is missing or invalid",
    ]

    invalid_count = _record(2, 24)
    invalid_count["retries"] = "1"
    with pytest.raises(RunSeriesError, match="retries must be a non-negative integer or null"):
        summarize_records([invalid_count])


def test_duplicate_run_id_and_missing_route_labels_are_rejected():
    record = _record(1, 4)
    duplicate = _record(1, 8)
    with pytest.raises(RunSeriesError, match="duplicate run_id"):
        summarize_records([record, duplicate])

    record["route"]["model"] = ""
    with pytest.raises(RunSeriesError, match="route.model must be a non-empty string"):
        summarize_records([record])


def test_cli_reads_private_jsonl_records_and_prints_summary(tmp_path, capsys):
    path = tmp_path / "private-runs.jsonl"
    path.write_text(
        "\n".join(json.dumps(_record(index, 10)) for index in range(1, 3)) + "\n",
        encoding="utf-8",
    )

    assert main(["--records", str(path)]) == 0
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert summary["group_count"] == 1
    assert summary["groups"][0]["sample_count"] == 2
