import argparse
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath

import pytest

from tools import sample_process_resources as sampler


def _proc_stat(pid: int, ppid: int, starttime: int, comm: str = "sample (worker)") -> str:
    # After comm, stat fields begin at field 3. starttime is field 22.
    rest = ["S", str(ppid)] + ["0"] * 17 + [str(starttime)]
    return f"{pid} ({comm}) " + " ".join(rest)


def _write_process(proc_root: Path, pid: int, ppid: int, starttime: int,
                   *, rss_kib: int = 10, pss_kib: int = 7,
                   smaps: bool = True) -> Path:
    directory = proc_root / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stat").write_text(_proc_stat(pid, ppid, starttime), encoding="utf-8")
    if smaps:
        (directory / "smaps_rollup").write_text(
            f"Rss: {rss_kib} kB\nPss: {pss_kib} kB\n", encoding="utf-8"
        )
    (directory / "status").write_text(f"VmRSS:\t{rss_kib} kB\n", encoding="utf-8")
    return directory


def _resource_fixture(tmp_path: Path) -> tuple[Path, Path, sampler.ProcessIdentity]:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    root = _write_process(proc_root, 100, 1, 1000, rss_kib=100, pss_kib=80)
    _write_process(proc_root, 101, 100, 1001, rss_kib=40, pss_kib=30)
    _write_process(proc_root, 102, 1, 1002, rss_kib=900, pss_kib=800)

    (root / "cgroup").write_text("0::/service.slice/adam.service\n", encoding="utf-8")
    cgroup_root = tmp_path / "sys" / "fs" / "cgroup"
    cgroup_dir = cgroup_root / "service.slice" / "adam.service"
    cgroup_dir.mkdir(parents=True)
    (cgroup_dir / "memory.current").write_text("4096\n", encoding="ascii")
    (cgroup_dir / "memory.peak").write_text("8192\n", encoding="ascii")
    (cgroup_dir / "memory.swap.current").write_text("512\n", encoding="ascii")
    (cgroup_dir / "cpu.stat").write_text("usage_usec 1234\nuser_usec 1000\nsystem_usec 234\n", encoding="ascii")
    self_dir = proc_root / "self"
    self_dir.mkdir()
    sampler_ns = self_dir / "ns"
    target_ns = root / "ns"
    sampler_ns.mkdir()
    target_ns.mkdir()
    for namespace in ("mnt", "cgroup"):
        os.symlink(f"{namespace}:[123]", sampler_ns / namespace)
        os.symlink(f"{namespace}:[123]", target_ns / namespace)
    (self_dir / "mountinfo").write_text(
        f"36 25 0:32 / {cgroup_root} rw,nosuid,nodev - cgroup2 cgroup rw\n", encoding="utf-8"
    )
    (proc_root / "meminfo").write_text(
        "MemAvailable: 2048 kB\nSwapTotal: 1024 kB\nSwapFree: 256 kB\n", encoding="utf-8"
    )
    return proc_root, cgroup_root, sampler.ProcessIdentity(100, 1, 1000)


def test_proc_stat_parser_handles_parentheses_and_uses_starttime_field_22():
    identity = sampler.parse_proc_stat(_proc_stat(321, 17, 9988, "a ) tricky (worker) name"))

    assert identity == sampler.ProcessIdentity(pid=321, ppid=17, starttime_ticks=9988)


def test_sample_once_sums_only_target_tree_and_reports_cgroup_and_host_values(tmp_path):
    proc_root, _, root = _resource_fixture(tmp_path)

    sample = sampler.sample_once(root, proc_root)

    tree = sample["process_tree"]
    assert tree["membership_complete"] is True
    assert tree["rss_bytes"] == (100 + 40) * 1024
    assert tree["pss_bytes"] == (80 + 30) * 1024
    assert {process["pid"] for process in tree["processes"]} == {100, 101}
    cgroup = sample["cgroup"]
    assert cgroup["memory_current_bytes"] == 4096
    assert cgroup["memory_peak_bytes"] == 8192
    assert cgroup["memory_swap_current_bytes"] == 512
    assert cgroup["cpu_usage_usec"] == 1234
    assert "entire reported cgroup" in cgroup["attribution_warning"]
    host = sample["host_memory"]
    assert host["available_ram_bytes"] == 2048 * 1024
    assert host["swap_total_bytes"] == 1024 * 1024
    assert host["swap_free_bytes"] == 256 * 1024
    assert host["swap_used_bytes"] == 768 * 1024


def test_unavailable_pss_has_reason_and_rss_uses_vm_rss_fallback(tmp_path):
    proc_root, _, root = _resource_fixture(tmp_path)
    child = proc_root / "101"
    (child / "smaps_rollup").unlink()

    tree = sampler.sample_process_tree(root, proc_root)

    process = next(item for item in tree["processes"] if item["pid"] == 101)
    assert process["rss_bytes"] == 40 * 1024
    assert process["rss_missing_reason"] is None
    assert process["pss_bytes"] is None
    assert "smaps_rollup" in process["pss_missing_reason"]
    assert tree["rss_bytes"] == 140 * 1024
    assert tree["pss_bytes"] is None
    assert tree["pss_missing_reasons"] == [
        {"pid": 101, "starttime_ticks": 1001, "reason": process["pss_missing_reason"]}
    ]


def test_pid_reuse_during_sampling_is_reported_and_excluded_from_tree_total(tmp_path, monkeypatch):
    proc_root, _, root = _resource_fixture(tmp_path)
    original_read = sampler.read_process_memory

    def reuse_child_after_read(identity, root_path):
        result = original_read(identity, root_path)
        if identity.pid == 101:
            (root_path / "101" / "stat").write_text(_proc_stat(101, 100, 9001), encoding="utf-8")
        return result

    monkeypatch.setattr(sampler, "read_process_memory", reuse_child_after_read)
    tree = sampler.sample_process_tree(root, proc_root)

    assert tree["membership_complete"] is False
    assert tree["rss_bytes"] is None
    old = next(item for item in tree["processes"] if item["pid"] == 101 and item["starttime_ticks"] == 1001)
    replacement = next(item for item in tree["processes"] if item["pid"] == 101 and item["starttime_ticks"] == 9001)
    assert old["membership_status"] == "pid_reused_or_replaced"
    assert replacement["membership_status"] == "appeared_during_sampling"


def test_vanished_child_is_marked_and_excluded_from_stable_sum(tmp_path, monkeypatch):
    proc_root, _, root = _resource_fixture(tmp_path)
    original_read = sampler.read_process_memory

    def remove_child_after_read(identity, root_path):
        result = original_read(identity, root_path)
        if identity.pid == 101:
            shutil.rmtree(root_path / "101")
        return result

    monkeypatch.setattr(sampler, "read_process_memory", remove_child_after_read)
    tree = sampler.sample_process_tree(root, proc_root)

    child = next(item for item in tree["processes"] if item["pid"] == 101)
    assert child["membership_status"] == "exited_during_sampling"
    assert "exited" in child["membership_missing_reason"]
    assert tree["membership_complete"] is False
    assert tree["rss_bytes"] is None
    assert tree["rss_known_sum_bytes"] == 100 * 1024


def test_unavailable_cgroup_counters_keep_field_specific_reasons(tmp_path):
    proc_root, cgroup_root, root = _resource_fixture(tmp_path)
    cgroup_dir = cgroup_root / "service.slice" / "adam.service"
    (cgroup_dir / "memory.current").unlink()
    (cgroup_dir / "cpu.stat").write_text("usage_usec invalid\n", encoding="ascii")

    cgroup = sampler.sample_cgroup_v2(root, proc_root)

    assert cgroup["memory_current_bytes"] is None
    assert "memory.current" in cgroup["missing_reasons"]["memory_current_bytes"]
    assert cgroup["cpu_usage_usec"] is None
    assert cgroup["missing_reasons"]["cpu_usage_usec"] == "cpu.stat value is not a nonnegative integer"


@pytest.mark.parametrize("namespace", ["mnt", "cgroup"])
def test_cgroup_metrics_are_unavailable_when_target_namespace_differs(tmp_path, namespace):
    proc_root, _, root = _resource_fixture(tmp_path)
    target_ns = proc_root / str(root.pid) / "ns" / namespace
    target_ns.unlink()
    os.symlink(f"{namespace}:[456]", target_ns)

    cgroup = sampler.sample_cgroup_v2(root, proc_root)

    assert cgroup["cgroup_path"] is None
    assert cgroup["memory_current_bytes"] is None
    assert cgroup["cpu_usage_usec"] is None
    assert f"different {namespace} namespace" in cgroup["missing_reasons"]["cgroup"]


def test_combined_sample_count_limit_boundary():
    maximum_duration = sampler.MAX_SAMPLES * sampler.MIN_INTERVAL_SEC
    over_limit_duration = (sampler.MAX_SAMPLES + 1) * sampler.MIN_INTERVAL_SEC

    assert sampler._validate_sample_count(maximum_duration, sampler.MIN_INTERVAL_SEC) == sampler.MAX_SAMPLES
    with pytest.raises(ValueError, match="maximum is 10000"):
        sampler._validate_sample_count(over_limit_duration, sampler.MIN_INTERVAL_SEC)


def test_over_limit_request_is_rejected_before_reading_target_or_allocating_samples(monkeypatch):
    def unexpected_target_read(*_args, **_kwargs):
        raise AssertionError("sample cap must be checked before reading /proc")

    monkeypatch.setattr(sampler, "_read_identity", unexpected_target_read)

    with pytest.raises(ValueError, match="maximum is 10000"):
        sampler.run_sampler(123, duration_sec=100.01, interval_sec=0.01)


def test_cgroup_mapping_respects_mount_root_and_rejects_traversal(tmp_path):
    mount = tmp_path / "cgroup"
    mounts = [(PurePosixPath("/delegated"), mount)]

    assert sampler._map_cgroup_path("/delegated/service/adam", mounts) == mount / "service" / "adam"
    assert sampler._map_cgroup_path("/delegated/../outside", mounts) is None
    assert sampler._map_cgroup_path("/other/adam", mounts) is None


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1", "0.001", "86401"])
def test_duration_cli_parser_rejects_nonfinite_and_out_of_bounds_values(value):
    parse = sampler._bounded_seconds(sampler.MIN_DURATION_SEC, sampler.MAX_DURATION_SEC, "duration")

    with pytest.raises(argparse.ArgumentTypeError):
        parse(value)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1", "0.001", "3601"])
def test_interval_cli_parser_rejects_nonfinite_and_out_of_bounds_values(value):
    parse = sampler._bounded_seconds(sampler.MIN_INTERVAL_SEC, sampler.MAX_INTERVAL_SEC, "interval")

    with pytest.raises(argparse.ArgumentTypeError):
        parse(value)


def test_sampler_writes_versioned_json_with_restricted_permissions(tmp_path):
    proc_root, _, _ = _resource_fixture(tmp_path)
    output = tmp_path / "private" / "resources.json"

    trace = sampler.run_sampler(100, duration_sec=0.01, interval_sec=0.01,
                                output_path=str(output), proc_root=proc_root)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert trace["schema_version"] == sampler.SCHEMA_VERSION == saved["schema_version"]
    assert saved["metadata"]["target_pid"] == 100
    assert saved["metadata"]["target_starttime_ticks"] == 1000
    assert saved["metadata"]["sample_count"] >= 1
    assert saved["samples"][0]["clock_ns"] > 0
    assert saved["samples"][0]["wall_time_utc"].endswith("Z")
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_private_trace_writer_rejects_symlink_without_touching_target(tmp_path):
    target = tmp_path / "sensitive-target.json"
    target.write_text("preserve this target\n", encoding="utf-8")
    output = tmp_path / "trace.json"
    output.symlink_to(target)

    with pytest.raises(ValueError, match="output symlink"):
        sampler._write_private_trace(output, '{"schema_version":1}\n')

    assert output.is_symlink()
    assert target.read_text(encoding="utf-8") == "preserve this target\n"
    assert list(tmp_path.glob(".trace.json.*.tmp")) == []


def test_private_trace_writer_atomically_replaces_file_and_cleans_temp_on_error(tmp_path, monkeypatch):
    output = tmp_path / "trace.json"
    output.write_text("old trace\n", encoding="utf-8")
    output.chmod(0o644)
    sampler._write_private_trace(output, '{"schema_version":2}\n')

    assert output.read_text(encoding="utf-8") == '{"schema_version":2}\n'
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert list(tmp_path.glob(".trace.json.*.tmp")) == []

    output.write_text("keep on replace error\n", encoding="utf-8")

    def fail_replace(*_args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(sampler.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        sampler._write_private_trace(output, '{"schema_version":3}\n')

    assert output.read_text(encoding="utf-8") == "keep on replace error\n"
    assert list(tmp_path.glob(".trace.json.*.tmp")) == []
