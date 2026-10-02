#!/usr/bin/env python3
"""Sample host-exposed power telemetry into a versioned JSON trace.

This tool does not measure whole-system AC power. Use an external wall meter
for that quantity. Samples use ``time.monotonic_ns()`` so ``clock_ns`` can be
correlated with application events from the same host boot.

Example:
    python tools/measure_power.py --duration 120 --interval 0.5 \
        --host-label workstation --power-source rapl+nvidia \
        --config-id vad-gate-off --workload-label vad-silence --output trace.json
"""

import argparse
import csv
import datetime as dt
import json
import math
import platform
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
NVIDIA_FIELDS = (
    "index,uuid,name,driver_version,power.draw,pstate,clocks.current.memory,"
    "clocks.current.graphics,temperature.gpu,display_active"
)
NVIDIA_QUERY_COMMAND = (
    "nvidia-smi",
    f"--query-gpu={NVIDIA_FIELDS}",
    "--format=csv,noheader,nounits",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _read_text(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        return path.read_text().strip(), None
    except (OSError, UnicodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip().lower() in {"", "n/a", "na", "not supported", "[not supported]", "unknown"}:
        return None
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_int(value: Optional[str]) -> Optional[int]:
    parsed = _parse_float(value)
    return int(parsed) if parsed is not None and parsed.is_integer() else None


def query_nvidia_smi() -> Dict[str, Any]:
    """Return per-GPU readings and explicit status for unavailable fields."""
    result: Dict[str, Any] = {
        "status": "unavailable",
        "error": None,
        "devices": [],
        "command": list(NVIDIA_QUERY_COMMAND),
    }
    try:
        proc = subprocess.run(NVIDIA_QUERY_COMMAND, capture_output=True, text=True, check=True, timeout=10)
    except FileNotFoundError as exc:
        result["error"] = f"nvidia-smi not found: {exc}"
        return result
    except subprocess.TimeoutExpired:
        result["status"] = "error"
        result["error"] = "nvidia-smi query timed out after 10 seconds"
        return result
    except subprocess.CalledProcessError as exc:
        result["status"] = "error"
        result["error"] = f"nvidia-smi exited {exc.returncode}: {exc.stderr.strip()[:500]}"
        return result
    except OSError as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if not proc.stdout.strip():
        result["status"] = "unavailable"
        result["error"] = "nvidia-smi returned no GPU rows"
        return result

    expected = NVIDIA_FIELDS.split(",")
    devices: List[Dict[str, Any]] = []
    parse_errors: List[str] = []
    try:
        rows = csv.reader(proc.stdout.splitlines(), skipinitialspace=True)
        for row_number, row in enumerate(rows, start=1):
            if not row:
                continue
            if len(row) != len(expected):
                parse_errors.append(f"row {row_number}: expected {len(expected)} columns, got {len(row)}")
            row += [""] * (len(expected) - len(row))
            values = dict(zip(expected, [part.strip() for part in row[:len(expected)]]))
            fields: Dict[str, Any] = {}
            unavailable: List[str] = []

            def set_value(key: str, value: Any) -> None:
                fields[key] = value
                if value is None:
                    unavailable.append(key)

            set_value("index", _parse_int(values.get("index")))
            set_value("uuid", values.get("uuid") or None)
            set_value("name", values.get("name") or None)
            set_value("driver_version", values.get("driver_version") or None)
            set_value("power_draw_w", _parse_float(values.get("power.draw")))
            set_value("pstate", values.get("pstate") or None)
            set_value("mem_clock_mhz", _parse_int(values.get("clocks.current.memory")))
            set_value("graphics_clock_mhz", _parse_int(values.get("clocks.current.graphics")))
            set_value("temperature_c", _parse_int(values.get("temperature.gpu")))
            display = values.get("display_active", "").strip().lower()
            set_value("display_active", True if display in {"enabled", "on", "true", "yes"} else
                      False if display in {"disabled", "off", "false", "no"} else None)
            fields["unavailable_fields"] = unavailable
            devices.append(fields)
    except csv.Error as exc:
        result["status"] = "error"
        result["error"] = f"Could not parse nvidia-smi CSV: {exc}"
        return result

    result["devices"] = devices
    if not devices:
        result["status"] = "unavailable"
        result["error"] = "nvidia-smi returned no parseable GPU rows"
    elif parse_errors or any(device["unavailable_fields"] for device in devices):
        result["status"] = "partial"
        result["error"] = "; ".join(parse_errors) if parse_errors else None
    else:
        result["status"] = "ok"
    return result


def query_nvidia_version() -> Dict[str, Any]:
    try:
        proc = subprocess.run(("nvidia-smi", "--version"), capture_output=True, text=True, check=True, timeout=10)
        return {"status": "ok", "text": proc.stdout.strip()[:2000], "error": None}
    except FileNotFoundError as exc:
        return {"status": "unavailable", "text": None, "error": f"nvidia-smi not found: {exc}"}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return {"status": "error", "text": None, "error": f"{type(exc).__name__}: {exc}"}


def query_laptop_battery() -> Dict[str, Any]:
    """Return raw battery sysfs values; do not infer discharge watts/sign."""
    root = Path("/sys/class/power_supply")
    result: Dict[str, Any] = {"status": "unavailable", "error": None, "devices": []}
    if not root.exists():
        result["error"] = f"{root} does not exist"
        return result

    devices: List[Dict[str, Any]] = []
    errors: List[str] = []
    try:
        batteries = sorted(root.glob("BAT*"))
    except OSError as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    for battery in batteries:
        item: Dict[str, Any] = {"name": battery.name, "sysfs_path": str(battery)}
        item_errors = []
        for field, filename in (
            ("status", "status"), ("capacity_raw", "capacity"),
            ("power_now_raw", "power_now"), ("current_now_raw", "current_now"),
            ("voltage_now_raw", "voltage_now"),
        ):
            path = battery / filename
            if path.exists():
                raw, error = _read_text(path)
            else:
                raw, error = None, None
            item[field] = raw
            item[field + "_path"] = str(path) if path.exists() else None
            if error:
                item_errors.append(f"{filename}: {error}")
        item["errors"] = item_errors
        if item_errors:
            errors.extend(f"{battery.name}: {message}" for message in item_errors)
        devices.append(item)

    result["devices"] = devices
    if not devices:
        result["status"] = "unavailable"
        result["error"] = "No BAT* devices found"
    elif errors:
        result["status"] = "partial"
        result["error"] = "; ".join(errors)
    else:
        result["status"] = "ok"
    return result


def query_rapl_energy() -> Dict[str, Any]:
    """Read raw Intel RAPL counters and wrap ranges for direct package zones."""
    rapl_base = Path("/sys/class/powercap/intel-rapl")
    result: Dict[str, Any] = {"status": "unavailable", "error": None, "zones": []}
    if not rapl_base.exists():
        result["error"] = f"{rapl_base} does not exist"
        return result

    zones: List[Dict[str, Any]] = []
    errors: List[str] = []
    try:
        packages = sorted(rapl_base.glob("intel-rapl:*"))
    except OSError as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    for package in packages:
        name, name_error = _read_text(package / "name")
        energy_text, energy_error = _read_text(package / "energy_uj")
        range_text, range_error = _read_text(package / "max_energy_range_uj")
        energy = _parse_int(energy_text)
        energy_range = _parse_int(range_text)
        zone_errors = [message for message in (
            f"name: {name_error}" if name_error else None,
            f"energy_uj: {energy_error}" if energy_error else None,
            f"max_energy_range_uj: {range_error}" if range_error else None,
            "energy_uj is not an integer" if energy_text is not None and energy is None else None,
            "max_energy_range_uj is not a positive integer" if range_text is not None and (energy_range is None or energy_range <= 0) else None,
        ) if message]
        zone = {
            "id": package.name,
            "name": name,
            "energy_uj": energy,
            "max_energy_range_uj": energy_range if energy_range and energy_range > 0 else None,
            "errors": zone_errors,
        }
        if not zone_errors and energy is not None and energy_range and energy_range > 0:
            zone["status"] = "ok"
        else:
            zone["status"] = "partial"
            errors.extend(f"{package.name}: {message}" for message in zone_errors)
        zones.append(zone)

    result["zones"] = zones
    if not zones:
        result["error"] = "No direct intel-rapl:* package zones found"
    elif errors:
        result["status"] = "partial"
        result["error"] = "; ".join(errors)
    else:
        result["status"] = "ok"
    return result


def _cpu_model() -> Optional[str]:
    text, _ = _read_text(Path("/proc/cpuinfo"))
    if not text:
        return None
    for line in text.splitlines():
        if line.lower().startswith(("model name", "hardware")) and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def get_cpu_stat() -> Dict[str, int]:
    """Read aggregate /proc/stat counters for callers that need raw CPU data.

    This helper is intentionally not used by the power trace: one snapshot is
    not CPU utilization, context switches, wakeups, or CPU energy.
    """
    try:
        with open("/proc/stat", encoding="utf-8") as stat_file:
            for line in stat_file:
                if line.startswith("cpu "):
                    parts = [int(value) for value in line.split()[1:]]
                    keys = ("user", "nice", "system", "idle", "iowait", "irq", "softirq")
                    return {key: parts[index] for index, key in enumerate(keys) if index < len(parts)}
    except (OSError, ValueError):
        return {}
    return {}


def _run_metadata(args: argparse.Namespace, started_clock_ns: int, started_wall_utc: str,
                  nvidia_version: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": str(uuid.uuid4()),
        "trace_id": args.trace_id,
        "host_label": args.host_label,
        "kernel": platform.release(),
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
        "gpu_inventory": [],
        "nvidia_smi_command": list(NVIDIA_QUERY_COMMAND),
        "nvidia_smi_version": nvidia_version,
        "power_source": args.power_source,
        "config_id": args.config_id,
        "workload_label": args.workload_label,
        "requested_duration_sec": args.duration,
        "requested_interval_sec": args.interval,
        "started_clock_ns": started_clock_ns,
        "started_at_utc": started_wall_utc,
        "ended_clock_ns": None,
        "ended_at_utc": None,
        "sample_count": 0,
        "interrupted": False,
    }


def run_benchmark(
    duration_sec: float,
    interval_sec: float,
    output_path: Optional[str] = None,
    *,
    host_label: Optional[str] = None,
    power_source: Optional[str] = None,
    config_id: Optional[str] = None,
    workload_label: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect samples on an anchored monotonic cadence and write a JSON trace."""
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("duration_sec must be a finite positive number")
    if not math.isfinite(interval_sec) or interval_sec <= 0:
        raise ValueError("interval_sec must be a finite positive number")
    nvidia_version = query_nvidia_version()
    start_clock_ns = time.monotonic_ns()
    start_wall_utc = _utc_now()
    metadata = _run_metadata(argparse.Namespace(
        trace_id=trace_id, host_label=host_label, power_source=power_source,
        config_id=config_id, workload_label=workload_label,
        duration=duration_sec, interval=interval_sec,
    ), start_clock_ns, start_wall_utc, nvidia_version)
    samples: List[Dict[str, Any]] = []
    previous_clock_ns: Optional[int] = None
    previous_rapl: Dict[str, Dict[str, Any]] = {}
    interrupted = False
    interval_ns = max(1, int(interval_sec * 1_000_000_000))
    deadline_ns = start_clock_ns
    end_deadline_ns = start_clock_ns + int(duration_sec * 1_000_000_000)

    print("=== Adam Power Telemetry Sampler (not a wall-power meter) ===")
    print(f"Duration: {duration_sec}s | Requested interval: {interval_sec}s")

    try:
        while True:
            sample_clock_ns = time.monotonic_ns()
            if sample_clock_ns >= end_deadline_ns:
                break
            wall_time_utc = _utc_now()
            nvidia = query_nvidia_smi()
            battery = query_laptop_battery()
            rapl = query_rapl_energy()
            after_queries_ns = time.monotonic_ns()
            actual_interval = ((sample_clock_ns - previous_clock_ns) / 1e9) if previous_clock_ns is not None else None
            rapl_intervals: List[Dict[str, Any]] = []
            current_rapl: Dict[str, Dict[str, Any]] = {}
            rapl_dt = actual_interval
            for zone in rapl["zones"]:
                zone_id = zone["id"]
                current = {
                    "name": zone["name"],
                    "energy_uj": zone["energy_uj"],
                    "max_energy_range_uj": zone["max_energy_range_uj"],
                }
                current_rapl[zone_id] = current
                prior = previous_rapl.get(zone_id)
                energy_delta = None
                average_power = None
                interval_status = "unavailable"
                error = None
                if zone["status"] != "ok":
                    error = "current RAPL counter or range is unavailable"
                elif prior is None:
                    error = "no prior valid sample for this zone"
                elif (prior["name"] != current["name"] or
                      prior["max_energy_range_uj"] != current["max_energy_range_uj"]):
                    error = "RAPL zone identity or counter range changed"
                elif rapl_dt is None or rapl_dt <= 0:
                    error = "elapsed sample interval is not positive"
                else:
                    raw_delta = current["energy_uj"] - prior["energy_uj"]
                    if raw_delta < 0:
                        # A modulo delta is consistent with a wrap, but sysfs alone
                        # cannot distinguish that from a counter reset. Preserve the
                        # candidate for review and avoid presenting inferred watts.
                        energy_delta = raw_delta % current["max_energy_range_uj"]
                        error = "counter decreased; wrap versus reset is ambiguous"
                        interval_status = "ambiguous"
                    else:
                        energy_delta = raw_delta
                        average_power = (energy_delta / 1_000_000.0) / rapl_dt
                        interval_status = "ok"
                rapl_intervals.append({
                    "id": zone_id,
                    "name": zone["name"],
                    "energy_delta_uj": energy_delta,
                    "average_power_w": average_power,
                    "elapsed_sec": rapl_dt,
                    "status": interval_status,
                    "error": error,
                })
            # A missing/invalid sample breaks continuity; do not bridge its gap.
            previous_rapl = current_rapl if rapl["status"] in {"ok", "partial"} else {}

            source_errors = {
                "nvidia": nvidia["error"],
                "battery": battery["error"],
                "rapl": rapl["error"],
            }
            gpus = nvidia["devices"]
            for gpu in gpus:
                if gpu.get("name") or gpu.get("uuid"):
                    entry = {"uuid": gpu.get("uuid"), "name": gpu.get("name"),
                             "driver_version": gpu.get("driver_version")}
                    if entry not in metadata["gpu_inventory"]:
                        metadata["gpu_inventory"].append(entry)

            sample = {
                "clock_ns": sample_clock_ns,
                "wall_time_utc": wall_time_utc,
                "elapsed_sec": (sample_clock_ns - start_clock_ns) / 1e9,
                "actual_interval_sec": actual_interval,
                "query_duration_sec": (after_queries_ns - sample_clock_ns) / 1e9,
                "sources": {
                    "nvidia": {"status": nvidia["status"], "error": nvidia["error"]},
                    "battery": {"status": battery["status"], "error": battery["error"]},
                    "rapl": {"status": rapl["status"], "error": rapl["error"]},
                },
                "gpus": gpus,
                "battery": battery["devices"],
                "rapl_zones": rapl["zones"],
                "rapl_intervals": rapl_intervals,
                "source_errors": source_errors,
            }
            samples.append(sample)
            previous_clock_ns = sample_clock_ns
            gpu_summary = " | ".join(
                f"GPU {gpu.get('index')} {gpu.get('name') or 'unknown'}: "
                f"{gpu['power_draw_w']:.1f} W"
                for gpu in gpus if gpu.get("power_draw_w") is not None
            ) or f"NVIDIA {nvidia['status']}"
            rapl_summary = " | ".join(
                f"{zone['id']}: {zone['average_power_w']:.2f} W"
                for zone in rapl_intervals if zone["average_power_w"] is not None
            ) or f"RAPL {rapl['status']}"
            print(f"[{sample['elapsed_sec']:7.2f}s] {gpu_summary} | "
                  f"Battery {battery['status']} (raw) | {rapl_summary}")

            deadline_ns += interval_ns
            now_ns = time.monotonic_ns()
            while deadline_ns <= now_ns:
                deadline_ns += interval_ns
            sleep_ns = min(deadline_ns, end_deadline_ns) - now_ns
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
    except KeyboardInterrupt:
        interrupted = True
        print("\nMeasurement interrupted by user.")
    finally:
        end_clock_ns = time.monotonic_ns()
        metadata["ended_clock_ns"] = end_clock_ns
        metadata["ended_at_utc"] = _utc_now()
        metadata["sample_count"] = len(samples)
        metadata["interrupted"] = interrupted

    trace = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "samples": samples,
    }
    gpu_power: Dict[str, List[float]] = {}
    for sample in samples:
        for gpu in sample["gpus"]:
            if gpu.get("power_draw_w") is not None:
                key = str(gpu.get("uuid") or gpu.get("index"))
                gpu_power.setdefault(key, []).append(gpu["power_draw_w"])
    if gpu_power:
        print("=== NVIDIA sampled board-power summary (not integrated energy) ===")
        for gpu_id, values in gpu_power.items():
            print(f"GPU {gpu_id}: average={sum(values) / len(values):.2f} W "
                  f"min={min(values):.2f} W max={max(values):.2f} W n={len(values)}")
    if output_path:
        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        output.write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
        output.chmod(0o600)
        print(f"Trace saved to {output}")
    else:
        print("No --output path supplied; trace was not saved.")
    return trace


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _optional_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty when supplied")
    if len(stripped) > 200:
        raise argparse.ArgumentTypeError("must be at most 200 characters")
    return stripped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample host power telemetry into versioned JSON (not wall power).")
    parser.add_argument("--duration", type=_positive_float, default=10.0, help="Sampling duration in seconds (default: 10)")
    parser.add_argument("--interval", type=_positive_float, default=1.0, help="Requested sample interval in seconds (default: 1)")
    parser.add_argument("--output", help="Optional output versioned JSON trace")
    parser.add_argument("--host-label", type=_optional_label, help="Operator-supplied host label; hostname is not collected")
    parser.add_argument("--power-source", type=_optional_label, help="Operator label for intended measurement source; does not validate it")
    parser.add_argument("--config-id", type=_optional_label, help="Non-secret configuration/command identifier")
    parser.add_argument("--workload-label", type=_optional_label, help="Non-content workload label")
    parser.add_argument("--trace-id", type=_optional_label, help="Optional opaque app trace ID for a single correlated run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_benchmark(
            args.duration, args.interval, args.output,
            host_label=args.host_label,
            power_source=args.power_source,
            config_id=args.config_id,
            workload_label=args.workload_label,
            trace_id=args.trace_id,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
