#!/usr/bin/env python3
"""Sample one Linux process tree and its cgroup into versioned JSON.

The sampler is read-only and uses only the Python standard library. Process
memory is sampled per PID/start-time identity; cgroup values describe the
whole cgroup and can include unrelated processes. A process tree is inherently
a best-effort snapshot, so short-lived children can start and exit between
observations. Cgroup paths are resolved using the sampler's mountinfo; if the
target is in a different mount or cgroup namespace, cgroup values are reported
unavailable instead of being resolved against a potentially unrelated mount.

Example::

    python tools/sample_process_resources.py --pid 1234 --duration 30 \
        --interval 0.5 --output resources.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional, Sequence


SCHEMA_VERSION = 1
DEFAULT_DURATION_SEC = 10.0
DEFAULT_INTERVAL_SEC = 1.0
MIN_DURATION_SEC = 0.01
MAX_DURATION_SEC = 86_400.0
MIN_INTERVAL_SEC = 0.01
MAX_INTERVAL_SEC = 3_600.0
MAX_SAMPLES = 10_000
PROC_ROOT = Path("/proc")
CGROUP_ATTRIBUTION_WARNING = (
    "Cgroup memory and CPU values cover the entire reported cgroup; they may "
    "include unrelated processes and are not attributed to this task or PID."
)
PROCESS_SNAPSHOT_WARNING = (
    "Process-tree membership is a best-effort snapshot; short-lived children "
    "between observations may not be counted."
)
CGROUP_NAMESPACE_NOTE = (
    "Cgroup paths use the sampler's mountinfo. Targets in a different mount or "
    "cgroup namespace have unavailable cgroup metrics rather than an inferred mapping."
)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    ppid: int
    starttime_ticks: int

    @property
    def key(self) -> tuple[int, int]:
        return self.pid, self.starttime_ticks


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _reason(exc: BaseException, operation: str) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"{operation}: process or kernel file disappeared"
    if isinstance(exc, PermissionError):
        return f"{operation}: permission denied"
    if isinstance(exc, UnicodeError):
        return f"{operation}: file was not valid text"
    if isinstance(exc, OSError):
        return f"{operation}: {type(exc).__name__}"
    return f"{operation}: {type(exc).__name__}"


def parse_proc_stat(text: str) -> ProcessIdentity:
    """Parse PID, PPID and start time from /proc/PID/stat."""
    left = text.find("(")
    right = text.rfind(")")
    if left <= 0 or right <= left:
        raise ValueError("malformed /proc stat record")
    try:
        pid = int(text[:left].strip())
        fields = text[right + 1 :].split()
        # fields[0] is stat field 3 (state); starttime is stat field 22.
        if len(fields) <= 19:
            raise ValueError("/proc stat record is missing starttime")
        ppid = int(fields[1])
        starttime = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise ValueError("malformed /proc stat identity fields") from exc
    if pid <= 0 or ppid < 0 or starttime < 0:
        raise ValueError("/proc stat identity fields are out of range")
    return ProcessIdentity(pid=pid, ppid=ppid, starttime_ticks=starttime)


def _read_identity(pid: int, proc_root: Path) -> tuple[Optional[ProcessIdentity], Optional[str]]:
    try:
        text = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return None, _reason(exc, "read process stat")
    try:
        identity = parse_proc_stat(text)
    except ValueError as exc:
        return None, str(exc)
    if identity.pid != pid:
        return None, "PID in /proc stat did not match proc directory"
    return identity, None


def scan_process_table(proc_root: Path = PROC_ROOT) -> tuple[dict[int, ProcessIdentity], list[dict[str, Any]]]:
    """Take one process-table snapshot, retaining non-transient scan failures."""
    identities: dict[int, ProcessIdentity] = {}
    errors: list[dict[str, Any]] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        return {}, [{"pid": None, "reason": _reason(exc, "scan proc directory")}]
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        identity, error = _read_identity(pid, proc_root)
        if identity is not None:
            identities[pid] = identity
        elif error and not error.startswith("read process stat: process or kernel file disappeared"):
            errors.append({"pid": pid, "reason": error})
    return identities, errors


def _tree_from_snapshot(
    identities: dict[int, ProcessIdentity], root: ProcessIdentity
) -> dict[int, ProcessIdentity]:
    current_root = identities.get(root.pid)
    if current_root is None or current_root.starttime_ticks != root.starttime_ticks:
        return {}
    children: dict[int, list[ProcessIdentity]] = {}
    for identity in identities.values():
        children.setdefault(identity.ppid, []).append(identity)
    tree: dict[int, ProcessIdentity] = {root.pid: current_root}
    pending = [root.pid]
    while pending:
        parent_pid = pending.pop()
        for child in children.get(parent_pid, ()):
            if child.pid not in tree:
                tree[child.pid] = child
                pending.append(child.pid)
    return tree


def _memory_kib_fields(text: str) -> tuple[dict[str, int], dict[str, str]]:
    wanted = {"Rss", "Pss"}
    values: dict[str, int] = {}
    errors: dict[str, str] = {}
    for line in text.splitlines():
        name = line.partition(":")[0]
        if name not in wanted:
            continue
        parts = line.split()
        try:
            if len(parts) < 3 or parts[2] != "kB":
                raise ValueError("expected a kB value")
            kib = int(parts[1])
            if kib < 0:
                raise ValueError("negative memory value")
            values[name] = kib * 1024
            errors.pop(name, None)
        except ValueError:
            values.pop(name, None)
            errors[name] = "malformed memory counter"
    return values, errors


def read_process_memory(
    identity: ProcessIdentity, proc_root: Path = PROC_ROOT
) -> dict[str, Any]:
    """Read RSS/PSS while checking that the PID still names the same process."""
    before, before_error = _read_identity(identity.pid, proc_root)
    if before is None:
        return _missing_process(identity, before_error or "process identity unavailable")
    if before.starttime_ticks != identity.starttime_ticks:
        return _missing_process(identity, "PID was reused before memory sampling")

    values: dict[str, int] = {}
    missing: dict[str, Optional[str]] = {"rss": None, "pss": None}
    smaps = proc_root / str(identity.pid) / "smaps_rollup"
    try:
        parsed, parse_errors = _memory_kib_fields(smaps.read_text(encoding="utf-8"))
        values.update({key.lower(): value for key, value in parsed.items()})
        for key, error in parse_errors.items():
            missing[key.lower()] = error
        if "rss" not in values and missing["rss"] is None:
            missing["rss"] = "Rss field missing from smaps_rollup"
        if "pss" not in values and missing["pss"] is None:
            missing["pss"] = "Pss field missing from smaps_rollup"
    except (OSError, UnicodeError) as exc:
        smaps_error = _reason(exc, "read smaps_rollup")
        missing["rss"] = smaps_error
        missing["pss"] = smaps_error

    # VmRSS is a useful, explicitly identified fallback when smaps is not
    # readable. It cannot provide PSS.
    if "rss" not in values:
        try:
            status = (proc_root / str(identity.pid) / "status").read_text(encoding="utf-8")
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 3 and parts[2] == "kB":
                        try:
                            kib = int(parts[1])
                            if kib >= 0:
                                values["rss"] = kib * 1024
                                missing["rss"] = None
                        except ValueError:
                            pass
                    if "rss" not in values:
                        missing["rss"] = "malformed VmRSS field in status"
                    break
        except (OSError, UnicodeError) as exc:
            # Preserve the smaps failure where it is more specific, while
            # making the failed fallback visible too.
            fallback_error = _reason(exc, "read status VmRSS")
            missing["rss"] = f"{missing['rss']}; {fallback_error}" if missing["rss"] else fallback_error

    after, after_error = _read_identity(identity.pid, proc_root)
    if after is None:
        return {
            "rss_bytes": values.get("rss"),
            "pss_bytes": values.get("pss"),
            "rss_missing_reason": missing["rss"],
            "pss_missing_reason": missing["pss"],
            "identity_status": "exited_during_read",
            "identity_missing_reason": after_error or "process exited while memory was read",
        }
    if after.starttime_ticks != identity.starttime_ticks:
        return _missing_process(identity, "PID was reused while memory was being read")
    return {
        "rss_bytes": values.get("rss"),
        "pss_bytes": values.get("pss"),
        "rss_missing_reason": missing["rss"],
        "pss_missing_reason": missing["pss"],
        "identity_status": "ok",
        "identity_missing_reason": None,
    }


def _missing_process(identity: ProcessIdentity, reason: Optional[str]) -> dict[str, Any]:
    message = reason or "process memory unavailable"
    return {
        "rss_bytes": None,
        "pss_bytes": None,
        "rss_missing_reason": message,
        "pss_missing_reason": message,
        "identity_status": "unavailable",
        "identity_missing_reason": message,
    }


def _aggregate_memory(
    process_records: list[dict[str, Any]], stable_keys: set[tuple[int, int]], field: str
) -> tuple[Optional[int], Optional[int], bool, list[dict[str, Any]]]:
    values: list[int] = []
    missing: list[dict[str, Any]] = []
    for item in process_records:
        key = (item["pid"], item["starttime_ticks"])
        if key not in stable_keys:
            continue
        value = item[field + "_bytes"]
        if value is None:
            missing.append({"pid": item["pid"], "starttime_ticks": item["starttime_ticks"],
                            "reason": item[field + "_missing_reason"] or f"{field.upper()} unavailable"})
        else:
            values.append(value)
    known_sum = sum(values) if values else None
    complete = bool(stable_keys) and len(values) == len(stable_keys) and not missing
    return (known_sum if complete else None), known_sum, complete, missing


def sample_process_tree(
    root: ProcessIdentity, proc_root: Path = PROC_ROOT
) -> dict[str, Any]:
    """Sample descendants, preserving PID/starttime identity and race reasons."""
    before, before_errors = scan_process_table(proc_root)
    before_tree = _tree_from_snapshot(before, root)
    records_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for identity in sorted(before_tree.values(), key=lambda item: item.pid):
        memory = read_process_memory(identity, proc_root)
        records_by_key[identity.key] = {
            "pid": identity.pid,
            "ppid": identity.ppid,
            "starttime_ticks": identity.starttime_ticks,
            **memory,
            "membership_status": "present_in_both_scans_pending",
            "membership_missing_reason": None,
        }

    after, after_errors = scan_process_table(proc_root)
    after_tree = _tree_from_snapshot(after, root)
    before_keys = {item.key for item in before_tree.values()}
    after_keys = {item.key for item in after_tree.values()}
    stable_keys = before_keys & after_keys
    membership_reasons: list[dict[str, Any]] = []

    for key, record in records_by_key.items():
        if key not in after_keys:
            replacement = after.get(key[0])
            if replacement is not None and replacement.starttime_ticks != key[1]:
                reason = "PID was reused or replaced during process-tree sampling"
                record["membership_status"] = "pid_reused_or_replaced"
            elif key[0] not in after:
                reason = "process exited during process-tree sampling"
                record["membership_status"] = "exited_during_sampling"
            else:
                reason = "process exited or left the selected process tree during sampling"
                record["membership_status"] = "left_process_tree"
            record["membership_missing_reason"] = reason
            membership_reasons.append({"pid": key[0], "starttime_ticks": key[1], "reason": reason})
        else:
            record["membership_status"] = "stable"

    for key in sorted(after_keys - before_keys):
        identity = after_tree[key[0]]
        replacement = key[0] in {old[0] for old in before_keys}
        reason = "PID was reused or replaced after process-tree discovery" if replacement else \
            "process appeared after process-tree discovery"
        records_by_key[key] = {
            "pid": identity.pid,
            "ppid": identity.ppid,
            "starttime_ticks": identity.starttime_ticks,
            "rss_bytes": None,
            "pss_bytes": None,
            "rss_missing_reason": reason,
            "pss_missing_reason": reason,
            "identity_status": "not_sampled",
            "identity_missing_reason": reason,
            "membership_status": "appeared_during_sampling",
            "membership_missing_reason": reason,
        }
        membership_reasons.append({"pid": key[0], "starttime_ticks": key[1], "reason": reason})

    scan_errors = before_errors + after_errors
    if scan_errors:
        membership_reasons.extend(scan_errors)
    membership_complete = before_keys == after_keys and not scan_errors and bool(root.pid in before_tree)
    process_records = sorted(records_by_key.values(), key=lambda item: (item["pid"], item["starttime_ticks"]))
    rss_total, rss_known_sum, rss_complete, rss_missing = _aggregate_memory(process_records, stable_keys, "rss")
    pss_total, pss_known_sum, pss_complete, pss_missing = _aggregate_memory(process_records, stable_keys, "pss")
    # Incomplete membership prevents a complete tree total, but retain the sum
    # for identities present in both observations as a clearly labelled value.
    rss_complete = rss_complete and membership_complete
    pss_complete = pss_complete and membership_complete
    tree_root_in_after = after.get(root.pid)
    root_missing_reason = None
    if tree_root_in_after is None:
        root_missing_reason = "target process exited during process-tree sampling"
    elif tree_root_in_after.starttime_ticks != root.starttime_ticks:
        root_missing_reason = "target PID was reused during process-tree sampling"
    return {
        "root_pid": root.pid,
        "root_starttime_ticks": root.starttime_ticks,
        "root_missing_reason": root_missing_reason,
        "process_count_before": len(before_tree),
        "process_count_after": len(after_tree),
        "stable_process_count": len(stable_keys),
        "membership_complete": membership_complete,
        "membership_missing_reasons": membership_reasons,
        "rss_bytes": rss_total if rss_complete else None,
        "rss_known_sum_bytes": rss_known_sum,
        "rss_complete": rss_complete,
        "rss_missing_reasons": rss_missing,
        "pss_bytes": pss_total if pss_complete else None,
        "pss_known_sum_bytes": pss_known_sum,
        "pss_complete": pss_complete,
        "pss_missing_reasons": pss_missing,
        "processes": process_records,
        "snapshot_warning": PROCESS_SNAPSHOT_WARNING,
    }


def _parse_meminfo(text: str) -> tuple[dict[str, int], dict[str, str]]:
    wanted = {"MemAvailable", "SwapTotal", "SwapFree"}
    values: dict[str, int] = {}
    missing: dict[str, str] = {}
    for line in text.splitlines():
        name = line.partition(":")[0]
        if name not in wanted:
            continue
        parts = line.split()
        try:
            if len(parts) < 3 or parts[2] != "kB":
                raise ValueError("expected a kB value")
            kib = int(parts[1])
            if kib < 0:
                raise ValueError("negative memory value")
            values[name] = kib * 1024
            missing.pop(name, None)
        except ValueError:
            values.pop(name, None)
            missing[name] = "malformed /proc/meminfo value"
    for name in wanted - values.keys() - missing.keys():
        missing[name] = f"{name} is not available in /proc/meminfo"
    return values, missing


def sample_host_memory(proc_root: Path = PROC_ROOT) -> dict[str, Any]:
    try:
        text = (proc_root / "meminfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        reason = _reason(exc, "read host meminfo")
        return {
            "available_ram_bytes": None,
            "swap_total_bytes": None,
            "swap_free_bytes": None,
            "swap_used_bytes": None,
            "missing_reasons": {"available_ram_bytes": reason, "swap": reason},
        }
    values, missing = _parse_meminfo(text)
    swap_used: Optional[int] = None
    if "SwapTotal" in values and "SwapFree" in values:
        if values["SwapFree"] <= values["SwapTotal"]:
            swap_used = values["SwapTotal"] - values["SwapFree"]
        else:
            missing["swap_used_bytes"] = "SwapFree exceeded SwapTotal"
    else:
        missing["swap_used_bytes"] = "SwapTotal or SwapFree is unavailable"
    return {
        "available_ram_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
        "swap_used_bytes": swap_used,
        "missing_reasons": {
            key: reason for key, reason in (
                ("available_ram_bytes", missing.get("MemAvailable")),
                ("swap_total_bytes", missing.get("SwapTotal")),
                ("swap_free_bytes", missing.get("SwapFree")),
                ("swap_used_bytes", missing.get("swap_used_bytes")),
            ) if reason is not None
        },
    }


def _unescape_mount_field(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _cgroup2_mounts(mountinfo: str) -> list[tuple[PurePosixPath, Path]]:
    mounts: list[tuple[PurePosixPath, Path]] = []
    for line in mountinfo.splitlines():
        fields = line.split(" - ", 1)
        if len(fields) != 2:
            continue
        before, after = fields
        post_fields = after.split()
        pre_fields = before.split()
        if not post_fields or post_fields[0] != "cgroup2" or len(pre_fields) < 5:
            continue
        root = PurePosixPath(_unescape_mount_field(pre_fields[3]))
        mountpoint = Path(_unescape_mount_field(pre_fields[4]))
        if root.is_absolute() and mountpoint.is_absolute():
            mounts.append((root, mountpoint))
    return mounts


def _check_target_namespaces(root: ProcessIdentity, proc_root: Path) -> Optional[str]:
    """Refuse to map a target cgroup using a different namespace's mountinfo."""
    for namespace in ("mnt", "cgroup"):
        sampler_link = proc_root / "self" / "ns" / namespace
        target_link = proc_root / str(root.pid) / "ns" / namespace
        try:
            sampler_identity = os.readlink(sampler_link)
            target_identity = os.readlink(target_link)
        except OSError as exc:
            return f"cannot verify target {namespace} namespace matches sampler: {type(exc).__name__}"
        if sampler_identity != target_identity:
            return (
                f"target is in a different {namespace} namespace; cgroup path cannot be safely "
                "mapped using sampler mountinfo"
            )
    return None


def _map_cgroup_path(cgroup_path: str, mounts: list[tuple[PurePosixPath, Path]]) -> Optional[Path]:
    path = PurePosixPath(cgroup_path)
    if not path.is_absolute() or ".." in path.parts:
        return None
    for mount_root, mountpoint in mounts:
        try:
            relative = path.relative_to(mount_root)
        except ValueError:
            continue
        mapped = mountpoint.joinpath(*relative.parts)
        return mapped
    return None


def _read_counter(path: Path, metric: str) -> tuple[Optional[int], Optional[str]]:
    try:
        raw = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        return None, _reason(exc, f"read cgroup {metric}")
    try:
        value = int(raw)
    except ValueError:
        return None, f"cgroup {metric} is not an integer"
    if value < 0:
        return None, f"cgroup {metric} is negative"
    return value, None


def _read_cpu_stat(path: Path) -> tuple[Optional[dict[str, int]], Optional[str], dict[str, str]]:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        return None, _reason(exc, "read cgroup cpu.stat"), {}
    values: dict[str, int] = {}
    errors: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0]:
            errors[f"line_{len(errors) + 1}"] = "malformed cpu.stat line"
            continue
        try:
            value = int(parts[1])
            if value < 0:
                raise ValueError
        except ValueError:
            errors[parts[0]] = "cpu.stat value is not a nonnegative integer"
            continue
        values[parts[0]] = value
    if not values:
        return None, "cgroup cpu.stat contained no valid counters", errors
    return values, None, errors


def sample_cgroup_v2(
    root: ProcessIdentity,
    proc_root: Path = PROC_ROOT,
    mountinfo_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Read cgroup v2 counters for the target, never attributing them to it."""
    default_metrics: dict[str, Any] = {
        "memory_current_bytes": None,
        "memory_peak_bytes": None,
        "memory_swap_current_bytes": None,
        "cpu_stat": None,
        "cpu_usage_usec": None,
        "missing_reasons": {},
    }
    current, error = _read_identity(root.pid, proc_root)
    if current is None:
        reason = error or "target process identity unavailable"
        return {**default_metrics, "cgroup_path": None, "attribution_warning": CGROUP_ATTRIBUTION_WARNING,
                "missing_reasons": {"cgroup": reason}}
    if current.starttime_ticks != root.starttime_ticks:
        return {**default_metrics, "cgroup_path": None, "attribution_warning": CGROUP_ATTRIBUTION_WARNING,
                "missing_reasons": {"cgroup": "target PID was reused"}}
    namespace_error = _check_target_namespaces(root, proc_root)
    if namespace_error:
        return {**default_metrics, "cgroup_path": None, "attribution_warning": CGROUP_ATTRIBUTION_WARNING,
                "missing_reasons": {"cgroup": namespace_error}}
    try:
        memberships = (proc_root / str(root.pid) / "cgroup").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        reason = _reason(exc, "read target cgroup membership")
        return {**default_metrics, "cgroup_path": None, "attribution_warning": CGROUP_ATTRIBUTION_WARNING,
                "missing_reasons": {"cgroup": reason}}
    cgroup_path: Optional[str] = None
    for line in memberships.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            cgroup_path = parts[2]
            break
    if cgroup_path is None:
        return {**default_metrics, "cgroup_path": None, "attribution_warning": CGROUP_ATTRIBUTION_WARNING,
                "missing_reasons": {"cgroup": "target has no unified cgroup v2 membership"}}
    mount_path = mountinfo_path or (proc_root / "self" / "mountinfo")
    try:
        mountinfo = mount_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        reason = _reason(exc, "read cgroup mountinfo")
        return {**default_metrics, "cgroup_path": cgroup_path, "attribution_warning": CGROUP_ATTRIBUTION_WARNING,
                "missing_reasons": {"cgroup": reason}}
    cgroup_dir = _map_cgroup_path(cgroup_path, _cgroup2_mounts(mountinfo))
    if cgroup_dir is None:
        return {**default_metrics, "cgroup_path": cgroup_path, "attribution_warning": CGROUP_ATTRIBUTION_WARNING,
                "missing_reasons": {"cgroup": "could not map target membership to a cgroup2 mount"}}
    after, after_error = _read_identity(root.pid, proc_root)
    if after is None or after.starttime_ticks != root.starttime_ticks:
        reason = after_error or "target PID was reused while resolving cgroup"
        return {**default_metrics, "cgroup_path": cgroup_path, "attribution_warning": CGROUP_ATTRIBUTION_WARNING,
                "missing_reasons": {"cgroup": reason}}

    result = {**default_metrics, "cgroup_path": cgroup_path, "attribution_warning": CGROUP_ATTRIBUTION_WARNING}
    reasons: dict[str, str] = {}
    for output_key, filename in (
        ("memory_current_bytes", "memory.current"),
        ("memory_peak_bytes", "memory.peak"),
        ("memory_swap_current_bytes", "memory.swap.current"),
    ):
        value, read_error = _read_counter(cgroup_dir / filename, filename)
        result[output_key] = value
        if read_error:
            reasons[output_key] = read_error
    cpu, cpu_error, cpu_parse_errors = _read_cpu_stat(cgroup_dir / "cpu.stat")
    result["cpu_stat"] = cpu
    result["cpu_usage_usec"] = cpu.get("usage_usec") if cpu else None
    if cpu_error:
        reasons["cpu_stat"] = cpu_error
        reasons["cpu_usage_usec"] = cpu_parse_errors.get("usage_usec", cpu_error)
    else:
        if "usage_usec" not in (cpu or {}):
            reasons["cpu_usage_usec"] = cpu_parse_errors.get(
                "usage_usec", "cgroup cpu.stat has no valid usage_usec counter"
            )
        if cpu_parse_errors:
            result["cpu_stat_parse_errors"] = cpu_parse_errors
    result["missing_reasons"] = reasons
    return result


def sample_once(root: ProcessIdentity, proc_root: Path = PROC_ROOT) -> dict[str, Any]:
    clock_ns = time.monotonic_ns()
    wall_time_utc = _utc_now()
    return {
        "clock_ns": clock_ns,
        "wall_time_utc": wall_time_utc,
        "process_tree": sample_process_tree(root, proc_root),
        "cgroup": sample_cgroup_v2(root, proc_root),
        "host_memory": sample_host_memory(proc_root),
    }


def _validate_duration_interval(duration_sec: float, interval_sec: float) -> None:
    if not math.isfinite(duration_sec) or not MIN_DURATION_SEC <= duration_sec <= MAX_DURATION_SEC:
        raise ValueError(f"duration must be finite and between {MIN_DURATION_SEC:g} and {MAX_DURATION_SEC:g} seconds")
    if not math.isfinite(interval_sec) or not MIN_INTERVAL_SEC <= interval_sec <= MAX_INTERVAL_SEC:
        raise ValueError(f"interval must be finite and between {MIN_INTERVAL_SEC:g} and {MAX_INTERVAL_SEC:g} seconds")


def _requested_sample_count(duration_sec: float, interval_sec: float) -> int:
    """Return the maximum scheduled sample count for the requested time window."""
    duration_ns = int(duration_sec * 1_000_000_000)
    interval_ns = max(1, int(interval_sec * 1_000_000_000))
    return (duration_ns + interval_ns - 1) // interval_ns


def _validate_sample_count(duration_sec: float, interval_sec: float) -> int:
    count = _requested_sample_count(duration_sec, interval_sec)
    if count > MAX_SAMPLES:
        raise ValueError(
            f"requested sampling schedule has {count} samples; the maximum is {MAX_SAMPLES} "
            "(increase --interval or reduce --duration)"
        )
    return count


def run_sampler(
    pid: int,
    duration_sec: float = DEFAULT_DURATION_SEC,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    output_path: Optional[str] = None,
    *,
    proc_root: Path = PROC_ROOT,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Collect a time series anchored to monotonic time and optionally save it."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be a positive integer")
    _validate_duration_interval(duration_sec, interval_sec)
    maximum_samples = _validate_sample_count(duration_sec, interval_sec)
    root, error = _read_identity(pid, proc_root)
    if root is None:
        raise ValueError(error or "target process was not found")
    start_clock_ns = time.monotonic_ns()
    start_wall = _utc_now()
    end_deadline_ns = start_clock_ns + int(duration_sec * 1_000_000_000)
    interval_ns = max(1, int(interval_sec * 1_000_000_000))
    samples: list[dict[str, Any]] = []
    interrupted = False
    deadline_ns = start_clock_ns
    try:
        while len(samples) < maximum_samples and (not samples or time.monotonic_ns() < end_deadline_ns):
            sample = sample_once(root, proc_root)
            sample["elapsed_sec"] = (sample["clock_ns"] - start_clock_ns) / 1_000_000_000
            sample["actual_interval_sec"] = (
                (sample["clock_ns"] - samples[-1]["clock_ns"]) / 1_000_000_000 if samples else None
            )
            samples.append(sample)
            deadline_ns += interval_ns
            now_ns = time.monotonic_ns()
            if now_ns >= end_deadline_ns:
                break
            while deadline_ns <= now_ns:
                deadline_ns += interval_ns
            sleep_ns = min(deadline_ns, end_deadline_ns) - now_ns
            if sleep_ns > 0:
                sleep_fn(sleep_ns / 1_000_000_000)
    except KeyboardInterrupt:
        interrupted = True
    end_clock_ns = time.monotonic_ns()
    trace = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "run_id": str(uuid.uuid4()),
            "target_pid": root.pid,
            "target_starttime_ticks": root.starttime_ticks,
            "requested_duration_sec": duration_sec,
            "requested_interval_sec": interval_sec,
            "maximum_sample_count": maximum_samples,
            "started_clock_ns": start_clock_ns,
            "started_at_utc": start_wall,
            "ended_clock_ns": end_clock_ns,
            "ended_at_utc": _utc_now(),
            "sample_count": len(samples),
            "interrupted": interrupted,
            "cgroup_attribution_warning": CGROUP_ATTRIBUTION_WARNING,
            "cgroup_namespace_note": CGROUP_NAMESPACE_NOTE,
            "process_snapshot_warning": PROCESS_SNAPSHOT_WARNING,
        },
        "samples": samples,
    }
    encoded = json.dumps(trace, indent=2, allow_nan=False) + "\n"
    if output_path:
        _write_private_trace(Path(output_path).expanduser(), encoded)
    return trace


def _write_private_trace(output: Path, encoded: str) -> None:
    """Atomically replace a regular output file without following its symlink."""
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.is_symlink():
        raise ValueError(f"refusing to write through output symlink: {output}")

    temporary_fd: Optional[int] = None
    temporary_path: Optional[Path] = None
    try:
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as output_file:
            temporary_fd = None  # fdopen owns and closes the descriptor now
            os.fchmod(output_file.fileno(), 0o600)
            output_file.write(encoded)
            output_file.flush()
            os.fsync(output_file.fileno())
        # A symlink created during the write is rejected too. os.replace itself
        # replaces a destination symlink rather than following its target.
        if output.is_symlink():
            raise ValueError(f"refusing to replace output symlink: {output}")
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _bounded_seconds(minimum: float, maximum: float, name: str) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be a number") from exc
        if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be finite and between {minimum:g} and {maximum:g} seconds"
            )
        return parsed
    return parse


def _positive_pid(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pid must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("pid must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample Linux process-tree memory, cgroup v2 counters, and host memory into versioned JSON.",
        epilog=(
            "Cgroup values describe the whole cgroup and may include unrelated processes. "
            "Cgroup paths use sampler mountinfo; targets in another mount or cgroup namespace "
            "are reported unavailable. The combined schedule is limited to 10,000 samples."
        ),
    )
    parser.add_argument("--pid", required=True, type=_positive_pid, help="PID whose current process tree to sample")
    parser.add_argument("--duration", type=_bounded_seconds(MIN_DURATION_SEC, MAX_DURATION_SEC, "duration"),
                        default=DEFAULT_DURATION_SEC, help="Sampling duration in seconds (default: 10)")
    parser.add_argument("--interval", type=_bounded_seconds(MIN_INTERVAL_SEC, MAX_INTERVAL_SEC, "interval"),
                        default=DEFAULT_INTERVAL_SEC, help="Requested sample interval in seconds (default: 1)")
    parser.add_argument("--output", help="Optional output JSON path; defaults to stdout")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        trace = run_sampler(args.pid, args.duration, args.interval, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not args.output:
        json.dump(trace, sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
