#!/usr/bin/env python3
"""Run one bounded, headless F3 system-status task through AdamBrain.

Provider inference is available only after separate route and inference
confirmations. The runner accepts an owner-private generated fixture, offers
only get_system_status, permits one empty-argument dispatch, and captures TTS
text without opening an audio device or playing sound. Answers always require
independent review.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

# Keep absolute-script invocation working, matching the F2 runner.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STRING = str(PROJECT_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != PROJECT_ROOT_STRING]
sys.path.insert(0, PROJECT_ROOT_STRING)
if __name__ == "__main__":
    from importlib.machinery import PathFinder
    from importlib.util import module_from_spec

    _src_spec = PathFinder.find_spec("src", [PROJECT_ROOT_STRING])
    if _src_spec is None or _src_spec.submodule_search_locations is None:
        raise ImportError("repository-local 'src' package could not be found")
    _src_package = module_from_spec(_src_spec)
    if _src_spec.loader is not None:
        _src_spec.loader.exec_module(_src_package)
    sys.modules["src"] = _src_package

from tools import run_implementation_file_status_trial as f2
from src.llm.tools import CanonicalTool
from src.telemetry.events import configure_telemetry, reset_trace_id, set_trace_id

EXPECTED_PROMPT = (
    "According to a fresh system status check, how many logical CPU cores does this system have "
    "and what percentage of memory is in use?"
)
FIXTURE_VERSION = 4
MAX_TOOL_DISPATCHES = 1
MAX_LOGICAL_CHAT_CALLS = 2
MAX_RESPONSE_CHARS = 16_000
MAX_TOOL_TRACE_CHARS = 8_192
_CPU_COUNT = re.compile(r"\bCPU has (\d+) logical cores\b", re.IGNORECASE)
_MEMORY_PERCENT = re.compile(r"\bMemory is ([0-9]+(?:\.[0-9]+)?) percent in use\b", re.IGNORECASE)
_SYSTEM_STATUS_TOOL = CanonicalTool(
    name="get_system_status",
    description=(
        "Read current system status once. Use no arguments and report only the CPU core count and memory use requested."
    ),
    parameters={"type": "object", "properties": {}},
)


class TrialError(ValueError):
    """Raised when the requested F3 trial is unsafe or malformed."""


def _private_record_fd(path: Path, run_id: str) -> int:
    """Open a private append-only record, accepting only F3 records already there."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
        ):
            raise TrialError("run JSONL must be an owned private single-link regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            stream = path.open("r", encoding="utf-8")
        except FileNotFoundError:
            stream = None
        if stream is not None:
            with stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise TrialError(f"existing output JSONL is invalid on line {number}") from exc
                    if (
                        not isinstance(record, dict)
                        or record.get("app") != "AdamBrain headless F3 system-status run"
                        or record.get("schema_version") != 1
                    ):
                        raise TrialError(f"existing output JSONL line {number} is not an F3 run record")
                    if record.get("run_id") == run_id:
                        raise TrialError(f"run ID {run_id!r} already exists in output JSONL")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _validate_fixture(fixture_dir: Path) -> dict[str, Any]:
    supplied = fixture_dir.expanduser()
    if not supplied.is_absolute():
        raise TrialError("--fixture-dir must be an absolute generated fixture directory")
    try:
        resolved = supplied.resolve(strict=True)
        info = supplied.lstat()
    except OSError as exc:
        raise TrialError("--fixture-dir must name an existing generated fixture directory") from exc
    if resolved != supplied.absolute() or not stat.S_ISDIR(info.st_mode):
        raise TrialError("--fixture-dir must not be a symlink or traverse symlinked directories")
    if info.st_uid != os.geteuid() or info.st_mode & 0o077 or info.st_mode & 0o700 != 0o700:
        raise TrialError("fixture directory must be owned by the current user and mode 0700")
    expected_bytes = f2._read_owned_regular(resolved / "expected.json", max_bytes=128_000)
    try:
        metadata = json.loads(expected_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrialError("fixture expected.json is not valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("fixture_version") != FIXTURE_VERSION:
        raise TrialError("fixture must use the generated implementation fixture format")
    fixture_id = metadata.get("run_id")
    f2._validate_id(fixture_id, "fixture run ID")
    if fixture_id != resolved.name:
        raise TrialError("fixture run ID must match its generated directory name")
    paths = metadata.get("paths")
    prompts = metadata.get("prompts")
    if not isinstance(paths, dict) or paths.get("root") != str(resolved):
        raise TrialError("fixture root metadata does not match --fixture-dir")
    if not isinstance(prompts, dict) or prompts.get("system_status") != EXPECTED_PROMPT:
        raise TrialError("fixture system-status prompt does not exactly match the generated F3 prompt")
    return {
        "fixture_id": fixture_id,
        "fixture_dir": resolved,
        "prompt": EXPECTED_PROMPT,
        "expected_json_sha256_before": hashlib.sha256(expected_bytes).hexdigest(),
    }


def _system_snapshot() -> dict[str, Any]:
    """Capture only the contemporaneous CPU-count and MemAvailable oracle inputs."""
    sampled_mono_ns = time.monotonic_ns()
    sampled_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    cpu_count = os.cpu_count()
    cpu_source = "os.cpu_count"
    if cpu_count is None:
        try:
            cpu_count = multiprocessing.cpu_count()
            cpu_source = "multiprocessing.cpu_count"
        except (NotImplementedError, OSError):
            cpu_count = None
            cpu_source = "unavailable"

    mem_total_kib = None
    mem_available_kib = None
    mem_error = None
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as stream:
            for line in stream:
                key, separator, rest = line.partition(":")
                if separator and key in {"MemTotal", "MemAvailable"}:
                    fields = rest.strip().split()
                    if not fields:
                        raise ValueError(f"{key} value is empty")
                    values[key] = int(fields[0])
        mem_total_kib = values.get("MemTotal")
        mem_available_kib = values.get("MemAvailable")
        if (
            mem_total_kib is None or mem_available_kib is None
            or mem_total_kib <= 0 or not 0 <= mem_available_kib <= mem_total_kib
        ):
            raise ValueError("MemTotal or MemAvailable is missing or invalid")
    except (OSError, UnicodeError, ValueError) as exc:
        mem_error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    memory_used_percent = None
    if mem_total_kib and mem_available_kib is not None:
        memory_used_percent = (mem_total_kib - mem_available_kib) * 100.0 / mem_total_kib
    return {
        "sampled_at_utc": sampled_utc,
        "sampled_monotonic_ns": sampled_mono_ns,
        "logical_cpu_count": cpu_count,
        "logical_cpu_count_source": cpu_source,
        "mem_total_kib": mem_total_kib,
        "mem_available_kib": mem_available_kib,
        "memory_in_use_percent": round(memory_used_percent, 4) if memory_used_percent is not None else None,
        "meminfo_error": mem_error,
    }


def _oracle_comparison(result_text: str, before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    cpu_match = None
    memory_matches: list[bool] = []
    cpu_match_detail = None
    cpu_values = [int(value) for value in _CPU_COUNT.findall(result_text)]
    memory_values = [float(value) for value in _MEMORY_PERCENT.findall(result_text)]
    cpu_conflict = len(set(cpu_values)) > 1
    memory_conflict = len(set(memory_values)) > 1
    reasons = []

    if cpu_conflict:
        reasons.append("tool readback contains conflicting logical CPU counts")
    if memory_conflict:
        reasons.append("tool readback contains conflicting memory-use percentages")
    if before and isinstance(before.get("logical_cpu_count"), int) and cpu_values:
        cpu_match = not cpu_conflict and cpu_values[0] == before["logical_cpu_count"]
        cpu_match_detail = {
            "reported_values": cpu_values,
            "claims_agree": not cpu_conflict,
            "snapshot_value": before["logical_cpu_count"],
        }
    elif cpu_conflict:
        cpu_match_detail = {"reported_values": cpu_values, "claims_agree": False, "snapshot_value": None}

    memory_snapshot_details = []
    if memory_values and not memory_conflict:
        reported_percent = memory_values[0]
        for label, snapshot in (("before", before), ("after", after)):
            if snapshot is None or not isinstance(snapshot.get("memory_in_use_percent"), (int, float)):
                continue
            oracle = float(snapshot["memory_in_use_percent"])
            match = abs(reported_percent - oracle) <= 1.0
            memory_matches.append(match)
            memory_snapshot_details.append({
                "snapshot": label,
                "reported_percent": reported_percent,
                "snapshot_percent": oracle,
                "within_one_percentage_point": match,
            })
    elif memory_conflict:
        for label, snapshot in (("before", before), ("after", after)):
            if snapshot is not None and isinstance(snapshot.get("memory_in_use_percent"), (int, float)):
                memory_matches.append(False)
                memory_snapshot_details.append({
                    "snapshot": label,
                    "reported_values": memory_values,
                    "claims_agree": False,
                    "snapshot_percent": float(snapshot["memory_in_use_percent"]),
                    "within_one_percentage_point": False,
                })
    assessment_possible = cpu_match is not None and bool(memory_matches)
    conflicting_claims = cpu_conflict or memory_conflict
    return {
        "status": (
            "not_assessable" if conflicting_claims
            else "matches_snapshots" if assessment_possible and cpu_match and all(memory_matches)
            else "differs_from_snapshot" if assessment_possible
            else "not_assessable"
        ),
        "logical_cpu_count_match_before": cpu_match,
        "logical_cpu_count_detail": cpu_match_detail,
        "memory_within_one_percentage_point": all(memory_matches) if memory_matches else None,
        "memory_comparison_details": memory_snapshot_details,
        "reasons": reasons,
        "note": "Objective comparison is evidence for reviewer use; it never changes manual_review_required.",
    }


class _CaptureOnlyTTS:
    engine = "capture-only"
    pending_barge_in_text = None

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak_async(self, text: str) -> None:
        self.spoken.append(str(text))


class _NoMemory:
    def retrieve_context(self, _utterance: str) -> None:
        return None


class _NoSkills:
    def get_startup_context(self) -> str:
        return ""

    def get_matched_skill_context(self, _text: str) -> None:
        return None


class _NoCustomTools:
    tools: dict[str, Any] = {}

    def get_canonical_tools(self) -> list[Any]:
        return []

    def has_tool(self, _name: str) -> bool:
        return False


class _CapturingClient:
    """Capture normalized provider outputs and enforce the F3 single-tool boundary."""

    def __init__(self, client: Any, secrets: list[str]) -> None:
        self._client = client
        self.provider = getattr(client, "provider", None)
        self.secrets = secrets
        self.responses: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def chat(self, messages: list[dict], tools=None, max_tokens=None, think=None) -> dict:
        if len(self.requests) >= MAX_LOGICAL_CHAT_CALLS:
            raise TrialError("F3 runner stopped after its two logical primary chat-call bound")
        offered = list(tools or [])
        names = [getattr(tool, "name", None) for tool in offered]
        # Brain removes get_system_status from the second hop after a status
        # tool call. A tool-free second hop is therefore expected; no other
        # schema may be offered on either hop.
        allowed = names == ["get_system_status"] if not self.requests else names in (["get_system_status"], [])
        if not allowed:
            raise TrialError("F3 runner refused a tool schema other than its single get_system_status schema")
        self.requests.append({"call_number": len(self.requests) + 1, "offered_tool_names": names})
        response = await self._client.chat(messages, tools=offered, max_tokens=max_tokens, think=think)
        if not isinstance(response, dict):
            self.responses.append({"response_type": type(response).__name__, "content": None, "tool_calls": []})
            return response
        content = response.get("content")
        safe_content, truncated = f2._scrub_text(
            content if isinstance(content, str) else "", self.secrets, limit=MAX_RESPONSE_CHARS,
        )
        calls = []
        raw_calls = response.get("tool_calls")
        if isinstance(raw_calls, list):
            for call in raw_calls[:8]:
                function = call.get("function") if isinstance(call, dict) else None
                function = function if isinstance(function, dict) else {}
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    raw_arguments = arguments[:4000]
                else:
                    try:
                        raw_arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))[:4000]
                    except (TypeError, ValueError):
                        raw_arguments = f"<{type(arguments).__name__}>"
                calls.append({
                    "id": str(call.get("id", ""))[:100] if isinstance(call, dict) else None,
                    "name": str(function.get("name", ""))[:100],
                    "arguments": f2._scrub_text(raw_arguments, self.secrets, limit=4000)[0],
                })
        self.responses.append({
            "role": response.get("role") if isinstance(response.get("role"), str) else None,
            "content": safe_content,
            "content_truncated": truncated,
            "tool_calls": calls,
            "provider_error": response.get("provider_error") is True,
        })
        return response


def _build_brain(config, secrets: list[str]):
    from src.llm.brain import AdamBrain

    config = config.model_copy(deep=True)
    config.llm.tool_free_model = ""
    config.llm.tool_free_provider_only = []
    config.llm.tool_free_allow_provider_fallbacks = False
    if config.computer_control is not None:
        config.computer_control.enabled = False
        config.computer_control.ocr_only = False
    if config.computer_vision is not None:
        config.computer_vision.enabled = False
    if config.browser_navigation is not None:
        config.browser_navigation.enabled = False
    tts = _CaptureOnlyTTS()
    with patch("src.llm.brain.CustomToolManager", return_value=_NoCustomTools()), \
            patch("src.llm.brain.SkillManager", _NoSkills):
        brain = AdamBrain(
            config, supervisor=None, probe=None, confirmation_mgr=None,
            tts_engine=tts, memory_mgr=_NoMemory(),
        )
    brain.get_tools = lambda: [_SYSTEM_STATUS_TOOL]
    brain.tool_free_llm_client = None
    brain.max_tool_rounds = MAX_LOGICAL_CHAT_CALLS
    brain.llm_client = _CapturingClient(brain.llm_client, secrets)
    dispatch_trace: list[dict[str, Any]] = []
    original_execute = brain._execute_tool
    dispatch_lock = asyncio.Lock()

    async def restricted_execute_locked(name: str, args: dict) -> str:
        entry: dict[str, Any] = {
            "tool_name": str(name)[:100],
            "arguments": f2._private_value(args, secrets, string_limit=2000),
        }
        dispatch_trace.append(entry)
        if len(dispatch_trace) > MAX_TOOL_DISPATCHES:
            entry["status"] = "refused"
            entry["reason"] = "dispatch limit reached"
            raise TrialError("F3 runner permits only one tool dispatch")
        if name != "get_system_status":
            entry["status"] = "refused"
            entry["reason"] = "only get_system_status is permitted"
            raise TrialError("F3 runner permits only get_system_status")
        if not isinstance(args, dict) or args:
            entry["status"] = "refused"
            entry["reason"] = "get_system_status arguments must be an empty object"
            raise TrialError("F3 runner refused non-empty get_system_status arguments")
        try:
            before = _system_snapshot()
        except Exception as exc:
            before = None
            entry["oracle_snapshot_before_error"] = f2._safe_error(exc, secrets)
        entry["oracle_snapshot_before"] = before
        entry["status"] = "dispatched"
        try:
            result = await original_execute("get_system_status", {})
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f2._safe_error(exc, secrets)
            raise
        finally:
            try:
                after = _system_snapshot()
            except Exception as exc:
                after = None
                entry["oracle_snapshot_after_error"] = f2._safe_error(exc, secrets)
            entry["oracle_snapshot_after"] = after
        result_text = result if isinstance(result, str) else str(result)
        entry["status"] = "returned"
        entry["result"] = f2._scrub_text(result_text, secrets, limit=MAX_TOOL_TRACE_CHARS)[0]
        entry["result_truncated"] = len(result_text) > MAX_TOOL_TRACE_CHARS
        entry["objective_comparison"] = _oracle_comparison(result_text, before, after)
        return result_text

    async def restricted_execute(name: str, args: dict) -> str:
        async with dispatch_lock:
            return await restricted_execute_locked(name, args)

    brain._f3_dispatch_trace = dispatch_trace
    brain._execute_tool = restricted_execute
    return brain, tts


def _usage_summary(event_bytes: bytes, trace_id: str) -> dict[str, Any]:
    """Summarize only provider usage fields already allowlisted by telemetry."""
    by_field: dict[str, list[int | float]] = {"input_tokens": [], "output_tokens": []}
    costs: dict[str, list[float]] = {}
    missing: dict[str, list[str]] = {"input_tokens": [], "output_tokens": [], "provider_reported_cost": []}
    starts = []
    completions = []
    parse_errors = 0
    for line_number, line in enumerate(event_bytes.decode("utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if isinstance(event, dict) and event.get("trace_id") == trace_id:
            if event.get("event") == "llm.request_started":
                starts.append((line_number, event))
            elif event.get("event") == "llm.completed":
                completions.append((line_number, event))
    start_spans = [event.get("span_id") for _line, event in starts]
    completion_spans = [event.get("span_id") for _line, event in completions]
    spans_pair = (
        bool(starts)
        and len(start_spans) == len(starts)
        and len(completion_spans) == len(completions)
        and all(isinstance(span, str) and span for span in start_spans + completion_spans)
        and Counter(start_spans) == Counter(completion_spans)
        and len(starts) == len(completions)
        and parse_errors == 0
    )
    if not starts:
        for reasons in missing.values():
            reasons.append("no provider request starts were recorded")
    elif len(starts) != len(completions):
        reason = f"provider event pairing incomplete: {len(starts)} request start(s), {len(completions)} completion(s)"
        for reasons in missing.values():
            reasons.append(reason)
    elif not spans_pair:
        for reasons in missing.values():
            reasons.append("provider request-start and completion span IDs did not pair exactly")
    for line_number, event in completions:
        attributes = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
        usage = attributes.get("usage") if isinstance(attributes.get("usage"), dict) else {}
        accounting = attributes.get("accounting_status")
        span = event.get("span_id") or f"line {line_number}"
        for field in ("input_tokens", "output_tokens"):
            value = usage.get(field)
            if accounting == "available" and isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                by_field[field].append(value)
            else:
                missing[field].append(f"{span}: unavailable or invalid provider usage")
        cost = usage.get("provider_reported_cost")
        currency = usage.get("currency")
        if accounting == "available" and isinstance(cost, (int, float)) and not isinstance(cost, bool) and isinstance(currency, str):
            costs.setdefault(currency, []).append(float(cost))
        else:
            missing["provider_reported_cost"].append(f"{span}: unavailable or invalid provider-reported cost")
    if parse_errors:
        for reasons in missing.values():
            reasons.append(f"{parse_errors} telemetry line(s) could not be parsed")
    def field_result(field: str) -> dict[str, Any]:
        values = by_field[field]
        return {
            "value": sum(values) if spans_pair and len(values) == len(completions) else None,
            "known_subtotal": sum(values) if values else None,
            "missing_reasons": list(dict.fromkeys(missing[field])) or ([] if spans_pair and len(values) == len(completions) else ["no provider completion events recorded or usage was incomplete"]),
        }
    cost_total = None
    if spans_pair and completions and not missing["provider_reported_cost"] and len(costs) == 1:
        cost_total = {"amount": sum(next(iter(costs.values()))), "currency": next(iter(costs))}
    elif len(costs) > 1:
        missing["provider_reported_cost"].append("provider costs used more than one currency")
    return {
        "provider_completion_count": len(completions),
        "input_tokens": field_result("input_tokens"),
        "output_tokens": field_result("output_tokens"),
        "provider_reported_cost": {
            "total": cost_total,
            "known_subtotals": {currency: sum(values) for currency, values in costs.items()},
            "missing_reasons": list(dict.fromkeys(missing["provider_reported_cost"])) or ([] if cost_total is not None else ["no provider completion events recorded or cost usage was incomplete"]),
        },
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if re.fullmatch(r"[0-9a-fA-F]{7,64}", value) else None


def _restore_telemetry_writer(installed_writer, previous_writer) -> None:
    """Restore the prior process-wide sink if this run still owns the slot."""
    from src.telemetry import events as telemetry_events

    # The telemetry module has no public writer getter/context manager. Restore
    # its prior object under the same lock used for listener updates, and do not
    # overwrite a different writer installed by another component meanwhile.
    if installed_writer is None:
        return
    with telemetry_events._writer_lock:
        if telemetry_events._writer is installed_writer:
            telemetry_events._writer = previous_writer


def run_trial(
    *, fixture_dir: Path, config_path: Path, run_id: str, config_id: str,
    output_path: Path, route_ready_confirmed: bool, expected_model: str,
    metadata_http_status: int | None, metadata_model_id: str | None,
    metadata_endpoint_count: int | None, metadata_checked_at: str | None,
    confirm_provider_inference: bool,
    brain_builder: Callable[..., tuple[Any, _CaptureOnlyTTS]] = _build_brain,
) -> dict[str, Any]:
    fixture = _validate_fixture(fixture_dir)
    run_id = f2._validate_id(run_id, "opaque run ID")
    config_id = f2._validate_id(config_id, "config ID")
    if not route_ready_confirmed:
        raise TrialError("refusing inference until the operator confirms a fresh external metadata check")
    attestation = f2._metadata_attestation(
        expected_model=expected_model, http_status=metadata_http_status,
        model_id=metadata_model_id, endpoint_count=metadata_endpoint_count,
        checked_at=metadata_checked_at,
    )
    if not confirm_provider_inference:
        raise TrialError("refusing live provider call without --confirm-provider-inference")
    config_path = config_path.expanduser()
    config = f2._load_config(config_path)
    route = f2._validate_route(config, config_id=config_id, expected_model=expected_model)
    if attestation["model_id"] != route["model"]:
        raise TrialError("metadata model ID must exactly match the configured model")
    route["operator_reported_metadata_check"] = attestation
    secret_values = f2._secret_values(config)
    for label, value in (
        ("fixture run ID", fixture["fixture_id"]), ("opaque run ID", run_id),
        ("config ID", config_id), ("model ID", route["model"]),
        *[("provider route", item) for item in route["provider_only"]],
    ):
        if any(len(secret) >= 4 and secret.casefold() in value.casefold() for secret in secret_values):
            raise TrialError(f"{label} must not contain a configured credential")
    output_path, event_path = f2._private_output_paths(output_path, run_id, fixture["fixture_dir"])
    config_realpath = config_path.resolve(strict=True)
    if config_realpath in {output_path.resolve(strict=False), event_path.resolve(strict=False)}:
        raise TrialError("run output may not replace or append to the config file")
    record_fd = _private_record_fd(output_path, run_id)
    try:
        event_fd = f2._create_private_file(event_path)
    except Exception:
        os.close(record_fd)
        raise
    os.close(event_fd)

    from src.telemetry import events as telemetry_events

    previous_writer = telemetry_events._writer
    installed_writer = None
    trace_token = None
    brain = None
    client_capture = None
    start_ns = end_ns = None
    execution_status = "error"
    error_record = None
    suppressed_stdout = io.StringIO()
    suppressed_stderr = io.StringIO()
    try:
        installed_writer = configure_telemetry(SimpleNamespace(enabled=True, path=str(event_path)))
        trace_token = set_trace_id(run_id)
        with redirect_stdout(suppressed_stdout), redirect_stderr(suppressed_stderr), \
                patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
            brain, _tts = brain_builder(config, secret_values)
            client_capture = brain.llm_client

            async def measured_turn() -> None:
                nonlocal start_ns, end_ns
                start_ns = time.monotonic_ns()
                try:
                    await brain.process_user_utterance(fixture["prompt"])
                finally:
                    end_ns = time.monotonic_ns()

            asyncio.run(measured_turn())
            execution_status = "returned"
    except Exception as exc:
        error_record = f2._safe_error(exc, secret_values)
    finally:
        if brain is not None:
            try:
                brain.close()
            except Exception as exc:
                error_record = error_record or f2._safe_error(exc, secret_values)
        try:
            if trace_token is not None:
                reset_trace_id(trace_token)
        finally:
            _restore_telemetry_writer(installed_writer, previous_writer)

    try:
        expected_after = f2._read_owned_regular(fixture["fixture_dir"] / "expected.json", max_bytes=128_000)
        fixture_after_hash = hashlib.sha256(expected_after).hexdigest()
        fixture_unchanged = fixture_after_hash == fixture["expected_json_sha256_before"]
        fixture_after_error = None
    except Exception as exc:
        fixture_after_hash = None
        fixture_unchanged = False
        fixture_after_error = f2._safe_error(exc, secret_values)
    dispatch_trace = getattr(brain, "_f3_dispatch_trace", []) if brain is not None else []
    final_answer = None
    if brain is not None:
        final_answer = next(
            (str(item.get("content") or "") for item in reversed(brain.messages)
             if item.get("role") == "assistant"),
            None,
        )
    safe_answer, answer_truncated = f2._scrub_text(final_answer or "", secret_values, limit=MAX_RESPONSE_CHARS)
    status_dispatches = [entry for entry in dispatch_trace if entry.get("tool_name") == "get_system_status"]
    oracle_reasons = (
        ["Independently compare the captured answer with the contemporaneous CPU and /proc/meminfo snapshots; natural-language text is never automatically scored."]
        if final_answer is not None else ["Brain did not append a final assistant answer"]
    )
    event_read_error = None
    try:
        event_bytes = event_path.read_bytes()
        event_digest = hashlib.sha256(event_bytes).hexdigest()
    except OSError as exc:
        event_bytes = b""
        event_digest = None
        event_read_error = f2._safe_error(exc, secret_values)
        error_record = error_record or {
            "type": "TrialError",
            "message": "private telemetry event log could not be read",
        }
    provider_attempt_summary = f2._provider_attempt_summary(event_bytes, run_id)
    try:
        event_info = event_path.stat()
        if not stat.S_ISREG(event_info.st_mode) or event_info.st_mode & 0o077:
            error_record = error_record or {"type": "TrialError", "message": "event log did not remain private"}
    except OSError as exc:
        event_read_error = event_read_error or f2._safe_error(exc, secret_values)
        error_record = error_record or {
            "type": "TrialError",
            "message": "private telemetry event log could not be inspected",
        }
    responses = client_capture.responses if isinstance(client_capture, _CapturingClient) else []
    logical_calls = len(client_capture.requests) if isinstance(client_capture, _CapturingClient) else 0
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "trace_id": run_id,
        "fixture_id": fixture["fixture_id"],
        "commit": _git_commit(),
        "host": platform.node() or None,
        "os": platform.platform(),
        "app": "AdamBrain headless F3 system-status run",
        "build": _git_commit(),
        "route": route,
        "scenario": "F3",
        "phase": "cold",
        "phase_scope": "fresh headless Brain call; no daemon, microphone, TTS playback, desktop, or browser",
        "starting_state": {
            "conversation": "empty",
            "memory": "disabled",
            "skills": "disabled",
            "custom_tools": "disabled",
            "model_tools": ["get_system_status"],
            "audio": "capture-only TTS; no playback",
            "desktop": "disabled",
            "browser": "disabled",
        },
        "prompt_id": "prompts.system_status",
        "prompt": fixture["prompt"],
        "oracle_id": "dynamic logical CPU count and contemporaneous /proc/meminfo snapshots; manual answer review",
        "oracle": {
            "status": "manual_review_required",
            "answer_review_status": "manual_review_required" if final_answer is not None else "not_assessable",
            "reasons": oracle_reasons,
            "tool_readback_count": len(status_dispatches),
            "tool_readback_returned": bool(status_dispatches and status_dispatches[0].get("status") == "returned"),
            "snapshot_comparison": status_dispatches[0].get("objective_comparison") if status_dispatches else None,
            "snapshots_before_after": {
                "before": status_dispatches[0].get("oracle_snapshot_before") if status_dispatches else None,
                "after": status_dispatches[0].get("oracle_snapshot_after") if status_dispatches else None,
            },
        },
        "brain_execution_status": execution_status,
        "brain_response_text": safe_answer if final_answer is not None else None,
        "brain_response_text_truncated": answer_truncated,
        "normalized_provider_responses": responses,
        "provider_tool_trace": [call for response in responses for call in response.get("tool_calls", [])],
        "dispatch_trace": dispatch_trace,
        "errors": [error_record] if error_record else [],
        "timing_ms": {
            "headless_brain_call": round((end_ns - start_ns) / 1_000_000, 3)
            if start_ns is not None and end_ns is not None else None,
            "request_end_to_verified_state": None,
            "request_end_to_verified_state_missing_reason": "headless F3 capture does not measure user request end or independently verified system state",
            "request_end_to_ack": None,
            "request_end_to_ack_missing_reason": "capture-only TTS does not measure audible playback",
        },
        "event_clock_ns": {"headless_brain_call_start": start_ns, "headless_brain_call_return": end_ns},
        "fixture_integrity": {
            "expected_json_sha256_before": fixture["expected_json_sha256_before"],
            "expected_json_sha256_after": fixture_after_hash,
            "unchanged": fixture_unchanged,
            "after_error": fixture_after_error,
        },
        "telemetry": {
            "event_path": str(event_path),
            "event_file_sha256": event_digest,
            "event_file_read_error": event_read_error,
            "matching_trace_id": run_id,
            "logical_primary_client_chat_calls": logical_calls,
            "logical_primary_client_chat_call_limit": MAX_LOGICAL_CHAT_CALLS,
            **provider_attempt_summary,
            "provider_usage": _usage_summary(event_bytes, run_id),
            "attempt_scope_note": "HTTP attempts and retry events are counts observed in Adam's client telemetry; provider-side rerouting or retries beyond this client are not observable.",
        },
        "provider_raw_http_body_available": False,
        "provider_raw_http_body_missing_reason": "UniversalLLMClient exposes normalized responses and allowlisted telemetry, not the raw HTTP body",
        "private_output_paths": {"record": str(output_path), "events": str(event_path)},
    }
    try:
        f2._append_record(record_fd, record)
    finally:
        os.close(record_fd)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path, help="absolute owner-private generated fixture directory")
    parser.add_argument("--config", required=True, type=Path, help="existing protected Adam config; read-only")
    parser.add_argument("--run-id", required=True, help="unique opaque ID; also used as telemetry trace ID")
    parser.add_argument("--config-id", required=True, help="non-secret stable label for this route/config")
    parser.add_argument("--output", required=True, type=Path, help="private .jsonl series file under an existing mode-0700 directory")
    parser.add_argument("--expected-model", required=True, help="exact configured model ID to run and attest")
    parser.add_argument("--metadata-http-status", required=True, type=int, help="operator-reported external metadata-only HTTP status; must be 200")
    parser.add_argument("--metadata-model-id", required=True, help="operator-reported exact model ID from external metadata check")
    parser.add_argument("--metadata-endpoint-count", required=True, type=int, help="positive endpoint count from external metadata check")
    parser.add_argument("--metadata-checked-at", required=True, help="external check time as ISO 8601 UTC ending in Z")
    parser.add_argument("--route-ready-confirmed", action="store_true", help="confirm fresh external metadata check was performed")
    parser.add_argument("--confirm-provider-inference", action="store_true", help="separately authorize live provider inference")
    args = parser.parse_args(argv)
    try:
        record = run_trial(
            fixture_dir=args.fixture_dir, config_path=args.config, run_id=args.run_id,
            config_id=args.config_id, output_path=args.output,
            route_ready_confirmed=args.route_ready_confirmed,
            expected_model=args.expected_model, metadata_http_status=args.metadata_http_status,
            metadata_model_id=args.metadata_model_id,
            metadata_endpoint_count=args.metadata_endpoint_count,
            metadata_checked_at=args.metadata_checked_at,
            confirm_provider_inference=args.confirm_provider_inference,
        )
    except (TrialError, f2.TrialError) as exc:
        print(f"F3 trial refused: {exc}", file=sys.stderr)
        return 2
    except OSError:
        print("F3 trial could not read or write a private artifact (OSError).", file=sys.stderr)
        return 2
    except (TypeError, ValueError) as exc:
        # Do not dump a serialization traceback that might include captured
        # provider output or private paths.
        print(f"F3 trial could not finalize its record ({type(exc).__name__}).", file=sys.stderr)
        return 2
    print(json.dumps({
        "run_id": record["run_id"],
        "fixture_id": record["fixture_id"],
        "scenario": record["scenario"],
        "outcome": record["oracle"]["status"],
        "brain_execution_status": record["brain_execution_status"],
        "logical_primary_client_chat_calls": record["telemetry"]["logical_primary_client_chat_calls"],
        "http_attempts_observed": record["telemetry"]["http_attempts_observed"],
        "private_output_paths": record["private_output_paths"],
    }, ensure_ascii=False))
    return 0 if record["brain_execution_status"] == "returned" else 1


if __name__ == "__main__":
    raise SystemExit(main())
