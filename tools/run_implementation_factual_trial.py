#!/usr/bin/env python3
"""Run one headless, tool-less F1 turn through AdamBrain.

This is a live-provider CLI when explicitly confirmed. It never starts Adam's
audio daemon and never executes model-requested tools. The raw provider HTTP
body is not exposed by UniversalLLMClient; the record retains its normalized
response and the private operational event trace instead.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import io
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
import json
import math
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

# Keep imports working when invoked as `python tools/run_implementation_factual_trial.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STRING = str(PROJECT_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != PROJECT_ROOT_STRING]
sys.path.insert(0, PROJECT_ROOT_STRING)

# `src` is a namespace package, so a later regular package named `src` can
# override it even when the checkout root is first on sys.path. Anchor the
# standalone CLI's source package to this checkout before importing it.
if __name__ == "__main__":
    _src_spec = PathFinder.find_spec("src", [PROJECT_ROOT_STRING])
    if _src_spec is None or _src_spec.submodule_search_locations is None:
        raise ImportError("repository-local 'src' package could not be found")
    _src_package = module_from_spec(_src_spec)
    if _src_spec.loader is not None:
        _src_spec.loader.exec_module(_src_package)
    sys.modules["src"] = _src_package

from src.config import load_config
from src.telemetry.events import configure_telemetry, reset_trace_id, set_trace_id


FACTUAL_PROMPT = "Compare rigatoni and penne in two short sentences."
EXPECTED_PROVIDER = "custom"
EXPECTED_API_BASE = "https://openrouter.ai/api/v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
_SAFE_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/-]{0,199}\Z")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_MAX_RESPONSE_CHARS = 32_000
_MAX_METADATA_AGE = timedelta(hours=1)
_MAX_METADATA_FUTURE_SKEW = timedelta(minutes=5)


class TrialError(ValueError):
    """Raised when the requested trial is not safe or well-formed."""


class _UnexpectedToolCall(RuntimeError):
    """The model requested a tool during a deliberately tool-less F1 run."""


class _CaptureOnlyTTS:
    """Capture Brain's response text without opening or playing audio."""

    engine = "capture-only"
    pending_barge_in_text = None

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.submitted_at_ns: list[int] = []

    async def speak_async(self, text: str) -> None:
        self.submitted_at_ns.append(time.monotonic_ns())
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


class _CapturingToollessClient:
    """Capture normalized provider responses and prevent any tool dispatch."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.provider = getattr(client, "provider", None)
        self.responses: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def chat(self, messages: list[dict], tools=None, max_tokens=None, think=None) -> dict:
        if tools:
            raise TrialError("F1 runner refused a non-empty tool schema")
        response = await self._client.chat(
            messages, tools=[], max_tokens=max_tokens, think=think,
        )
        safe_response = _safe_normalized_response(response)
        self.responses.append(safe_response)
        tool_calls = response.get("tool_calls") if isinstance(response, dict) else None
        if tool_calls:
            names = safe_response.get("tool_names") or []
            raise _UnexpectedToolCall(
                "model returned a tool call during the tool-less F1 trial"
                + (f" (names: {', '.join(names)})" if names else "")
            )
        return response


def _safe_normalized_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"response_type": type(response).__name__, "content": None}
    content = response.get("content")
    safe_content = content if isinstance(content, str) else None
    truncated = bool(safe_content and len(safe_content) > _MAX_RESPONSE_CHARS)
    if truncated:
        safe_content = safe_content[:_MAX_RESPONSE_CHARS]
    raw_calls = response.get("tool_calls")
    names: list[str] = []
    if isinstance(raw_calls, list):
        for call in raw_calls[:32]:
            function = call.get("function") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str):
                names.append(name[:100])
    result = {
        "role": response.get("role") if isinstance(response.get("role"), str) else None,
        "content": safe_content,
        "content_truncated": truncated,
        "tool_call_count": len(raw_calls) if isinstance(raw_calls, list) else 0,
        "tool_names": names,
        "provider_error": response.get("provider_error") is True,
    }
    return result


def _validate_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise TrialError(f"{name} must be 1–80 safe opaque characters [A-Za-z0-9._-]")
    return value


def _load_existing_config(config_path: Path):
    try:
        config_stat = config_path.lstat()
    except OSError as exc:
        raise TrialError("--config must name an existing protected regular file") from exc
    if not stat.S_ISREG(config_stat.st_mode):
        raise TrialError("--config must be a regular file; symlinks and special files are refused")
    if config_stat.st_uid != os.geteuid():
        raise TrialError("--config must be owned by the current user")
    if config_stat.st_mode & 0o077:
        raise TrialError("--config must not be accessible by group or other users (mode 0600 or stricter)")
    return load_config(str(config_path))


def _validate_model_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_MODEL_ID.fullmatch(value)
        or "/" not in value
        or "//" in value
        or value.endswith("/")
    ):
        raise TrialError(f"{name} must be a safe namespaced OpenRouter model ID")
    return value


def _validate_metadata_attestation(
    *,
    expected_model: str,
    metadata_http_status: int | None,
    metadata_model_id: str | None,
    metadata_endpoint_count: int | None,
    metadata_checked_at: str | None,
) -> dict[str, Any]:
    expected_model = _validate_model_id(expected_model, "expected model")
    if metadata_http_status != 200:
        raise TrialError("route metadata must attest HTTP status 200")
    if metadata_model_id is None or _validate_model_id(metadata_model_id, "metadata model ID") != expected_model:
        raise TrialError("metadata model ID must exactly match the expected model")
    if (
        isinstance(metadata_endpoint_count, bool)
        or not isinstance(metadata_endpoint_count, int)
        or metadata_endpoint_count < 1
    ):
        raise TrialError("metadata endpoint count must be a positive integer")
    if not isinstance(metadata_checked_at, str) or not _UTC_TIMESTAMP.fullmatch(metadata_checked_at):
        raise TrialError("metadata check time must be an ISO 8601 UTC timestamp ending in Z")
    try:
        checked_at = datetime.fromisoformat(metadata_checked_at[:-1] + "+00:00")
    except ValueError as exc:
        raise TrialError("metadata check time must be an ISO 8601 UTC timestamp ending in Z") from exc
    if checked_at.tzinfo is None or checked_at.utcoffset() != timedelta(0):
        raise TrialError("metadata check time must be in UTC")
    age = datetime.now(timezone.utc) - checked_at
    if age < -_MAX_METADATA_FUTURE_SKEW:
        raise TrialError("metadata check time is too far in the future")
    if age > _MAX_METADATA_AGE:
        raise TrialError("route metadata is older than one hour; refresh the metadata check")
    return {
        "http_status": 200,
        "model_id": metadata_model_id,
        "endpoint_count": metadata_endpoint_count,
        "checked_at_utc": checked_at.isoformat().replace("+00:00", "Z"),
    }


def _validate_config(config, *, config_id: str, expected_model: str) -> dict[str, Any]:
    llm = config.llm
    provider = str(getattr(llm, "provider", "")).lower()
    model = str(getattr(llm, "cloud_model", ""))
    api_base = str(getattr(llm, "api_base", ""))
    fallbacks = getattr(llm, "allow_provider_fallbacks", None)
    expected_model = _validate_model_id(expected_model, "expected model")
    if provider != EXPECTED_PROVIDER:
        raise TrialError(f"configured provider must remain {EXPECTED_PROVIDER!r}")
    if model != expected_model:
        raise TrialError("configured model must exactly match --expected-model")
    if api_base not in {EXPECTED_API_BASE, f"{EXPECTED_API_BASE}/"}:
        raise TrialError(f"configured API base must remain pinned to {EXPECTED_API_BASE!r}")
    if fallbacks is not False:
        raise TrialError("provider fallbacks must be explicitly disabled for this matched route")
    provider_only = getattr(llm, "provider_only", None)
    if provider_only is not None and (
        not isinstance(provider_only, list)
        or any(not isinstance(item, str) for item in provider_only)
    ):
        raise TrialError("provider_only must be a list of provider labels")
    return {
        "provider": provider,
        "model": model,
        "api_base": EXPECTED_API_BASE,
        "config_id": config_id,
        "provider_only": list(provider_only or []),
        "allow_provider_fallbacks": False,
    }


def _config_secret_values(config) -> list[str]:
    values = []
    for section_name in ("llm", "tts", "stt"):
        section = getattr(config, section_name, None)
        secret = getattr(section, "api_key", None) if section is not None else None
        if isinstance(secret, str) and secret:
            values.append(secret)
    for name, value in os.environ.items():
        if any(marker in name.upper() for marker in ("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "SECRET")):
            if value:
                values.append(value)
    return list(dict.fromkeys(values))


def _private_output_paths(output: Path, run_id: str) -> tuple[Path, Path]:
    output = output.expanduser()
    if not output.is_absolute():
        raise TrialError("--output must be an absolute path inside an existing private directory")
    if output.suffix != ".jsonl":
        raise TrialError("--output must use the .jsonl extension")
    parent = output.parent
    try:
        resolved_parent = parent.resolve(strict=True)
        parent_stat = parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise TrialError("--output parent must be an existing private directory") from exc
    if resolved_parent != parent.absolute():
        raise TrialError("--output path may not traverse symlinked directories")
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise TrialError("--output parent must be a directory")
    if parent_stat.st_uid != os.geteuid() or parent_stat.st_mode & 0o077:
        raise TrialError("--output parent must be owned by the current user and mode 0700 or stricter")
    event_path = output.with_name(f"{run_id}.events.jsonl")
    try:
        output_stat = output.lstat()
    except FileNotFoundError:
        output_stat = None
    if output_stat is not None and (
        not stat.S_ISREG(output_stat.st_mode)
        or output_stat.st_uid != os.geteuid()
        or output_stat.st_mode & 0o077
        or output_stat.st_nlink != 1
    ):
        raise TrialError("existing --output must be an owned private regular file")
    try:
        event_path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise TrialError("per-run event log must not already exist")
    return output, event_path


def _open_private_jsonl(path: Path, *, run_id: str) -> int:
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        opened_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_uid != os.geteuid()
            or opened_stat.st_mode & 0o077
            or opened_stat.st_nlink != 1
        ):
            raise TrialError("run JSONL must be an owned private single-link regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _reject_duplicate_or_invalid_jsonl(path, run_id)
        return descriptor
    except TrialError:
        try:
            os.close(descriptor)
        except (UnboundLocalError, OSError):
            pass
        raise
    except OSError as exc:
        try:
            os.close(descriptor)
        except (UnboundLocalError, OSError):
            pass
        raise TrialError(f"could not safely open private run JSONL ({type(exc).__name__})") from exc


def _reject_duplicate_or_invalid_jsonl(path: Path, run_id: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrialError(f"existing output JSONL has invalid JSON on line {line_number}") from exc
                if not isinstance(row, dict):
                    raise TrialError(f"existing output JSONL line {line_number} is not an object")
                if row.get("schema_version") != 1 or row.get("app") != "AdamBrain headless F1 text run":
                    raise TrialError(f"existing output JSONL line {line_number} is not an F1 run record")
                if row.get("run_id") == run_id:
                    raise TrialError(f"run ID {run_id!r} already exists in output JSONL")
    finally:
        os.close(descriptor)


def _create_private_file(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        return descriptor
    except OSError as exc:
        raise TrialError(f"could not create private output file ({type(exc).__name__})") from exc


def _append_jsonl_fd(descriptor: int, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    remaining = memoryview(encoded)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short append to private run JSONL")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _safe_error(exc: BaseException, secrets: list[str]) -> dict[str, str]:
    message = str(exc)[:2000]
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s&,;]+", r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", message)
    message = re.sub(r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key)=)[^&#\s]+", r"\1[REDACTED]", message)
    return {"type": type(exc).__name__, "message": message}


def _provider_event_summary(events: list[dict[str, Any]], trace_id: str) -> dict[str, Any]:
    matching = [event for event in events if event.get("trace_id") == trace_id]
    starts = [event for event in matching if event.get("event") == "llm.request_started"]
    completions = [event for event in matching if event.get("event") == "llm.completed"]
    retries = [event for event in matching if event.get("event") == "llm.retrying"]
    if not starts:
        statuses = [
            (event.get("attributes") or {}).get("http_status")
            for event in completions if isinstance(event.get("attributes"), dict)
        ]
        retry_reasons = Counter(
            (event.get("attributes") or {}).get("reason")[:80]
            for event in retries
            if isinstance((event.get("attributes") or {}).get("reason"), str)
        )
        retry_429 = sum(
            reason.strip().casefold() in {"429", "http 429", "status 429"}
            for reason in [
                (event.get("attributes") or {}).get("reason")
                for event in retries
                if isinstance((event.get("attributes") or {}).get("reason"), str)
            ]
        )
        error_events = []
        for event in completions:
            if event.get("status") == "ok":
                continue
            attributes = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
            error_events.append({
                "span_id": event.get("span_id"),
                "status": event.get("status"),
                "http_status": attributes.get("http_status"),
                "accounting_status": attributes.get("accounting_status"),
            })
        return {
            "trace_event_count": len(matching),
            "provider_call_count": None,
            "provider_call_count_missing_reason": "no matching llm.request_started event was observed",
            "provider_completion_count": len(completions),
            "retry_count": len(retries) if matching else None,
            "retry_count_missing_reason": None if matching else "no matching trace events were observed",
            "retry_reason_counts": dict(sorted(retry_reasons.items())),
            "http_429_response_count": sum(status == 429 for status in statuses) if completions else None,
            "http_429_response_count_missing_reason": None if completions else "no matching provider completion was observed",
            "http_429_retry_trigger_count": retry_429 if retries else None,
            "http_429_retry_trigger_count_missing_reason": None if retries else "no matching retry event was observed",
            "usage": _missing_usage("no complete provider attempt accounting was observed"),
            "provider_errors": error_events,
            "provider_error_details_missing_reason": (
                "telemetry records status/accounting metadata only; UniversalLLMClient does not expose raw HTTP error bodies"
                if error_events else None
            ),
        }

    start_spans = Counter(event.get("span_id") for event in starts)
    completion_spans = Counter(event.get("span_id") for event in completions)
    pairing_complete = (
        all(span_id and start_spans[span_id] == 1 and completion_spans[span_id] == 1
            for span_id in start_spans)
        and all(span_id and completion_spans[span_id] == 1 and start_spans[span_id] == 1
                for span_id in completion_spans)
        and len(starts) == len(completions)
    )
    error_events = []
    for event in completions:
        if event.get("status") == "ok":
            continue
        attributes = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
        error_events.append({
            "span_id": event.get("span_id"),
            "status": event.get("status"),
            "http_status": attributes.get("http_status"),
            "accounting_status": attributes.get("accounting_status"),
        })
    retry_429 = 0
    retry_reasons = Counter()
    for event in retries:
        attributes = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
        reason = attributes.get("reason")
        if isinstance(reason, str):
            retry_reasons[reason[:80]] += 1
            if reason.strip().casefold() in {"429", "http 429", "status 429"}:
                retry_429 += 1

    statuses = [
        (event.get("attributes") or {}).get("http_status")
        for event in completions if isinstance(event.get("attributes"), dict)
    ]
    usage_rows = []
    for event in completions:
        attributes = event.get("attributes")
        if not isinstance(attributes, dict):
            continue
        usage_value = attributes.get("usage")
        if isinstance(usage_value, dict):
            usage_value = {
                **usage_value,
                "_event_accounting_status": attributes.get("accounting_status"),
            }
        usage_rows.append(usage_value)
    usage = _aggregate_usage(usage_rows, complete=pairing_complete)
    return {
        "trace_event_count": len(matching),
        "provider_call_count": len(starts),
        "provider_completion_count": len(completions),
        "attempt_pairing_complete": pairing_complete,
        "attempt_pairing_missing_reason": None if pairing_complete else "provider start/completion spans do not pair one-to-one",
        "retry_count": len(retries),
        "retry_reason_counts": dict(sorted(retry_reasons.items())),
        "http_429_response_count": sum(status == 429 for status in statuses) if pairing_complete else None,
        "http_429_response_count_missing_reason": (
            None if pairing_complete else "provider starts and completions did not pair one-to-one; total 429 count is incomplete"
        ),
        "http_429_retry_trigger_count": retry_429,
        "usage": usage,
        "provider_errors": error_events,
        "provider_error_details_missing_reason": (
            "telemetry records status/accounting metadata only; UniversalLLMClient does not expose raw HTTP error bodies"
            if error_events else None
        ),
    }


def _missing_usage(reason: str) -> dict[str, Any]:
    return {
        "input_tokens": None,
        "output_tokens": None,
        "reported_cost": None,
        "currency": None,
        "accounting_status": "missing",
        "missing_reasons": {field: [reason] for field in (
            "input_tokens", "output_tokens", "reported_cost", "currency",
        )},
    }


def _aggregate_usage(rows: list[Any], *, complete: bool) -> dict[str, Any]:
    if not rows:
        return _missing_usage("provider completion events contained no usage object")
    mappings = [row for row in rows if isinstance(row, dict)]
    reasons: dict[str, list[str]] = {}
    result: dict[str, Any] = {}
    for field in ("input_tokens", "output_tokens"):
        values = [row.get(field) for row in mappings]
        if complete and len(mappings) == len(rows) and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in values
        ):
            result[field] = sum(values)
        else:
            result[field] = None
            reasons[field] = ["missing or unpaired provider usage field"]

    cost_values: list[float] = []
    currencies: list[str] = []
    for row in mappings:
        value = row.get("provider_reported_cost")
        currency = row.get("currency")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            cost_values = []
            break
        if not isinstance(currency, str) or not currency.strip():
            cost_values = []
            break
        cost_values.append(float(value))
        currencies.append(currency.strip())
    if complete and len(mappings) == len(rows) and len(cost_values) == len(rows) and len(set(currencies)) == 1:
        result["reported_cost"] = round(sum(cost_values), 12)
        result["currency"] = currencies[0]
    else:
        result["reported_cost"] = None
        result["currency"] = None
        reasons["reported_cost"] = ["cost is missing, unpaired, or spans mixed currencies"]
        reasons["currency"] = ["currency is missing, unpaired, or spans mixed currencies"]

    statuses = [
        row.get("_event_accounting_status")
        for row in mappings
        if isinstance(row.get("_event_accounting_status"), str)
    ]
    result["accounting_status"] = "complete" if complete and len(mappings) == len(rows) and all(
        status in {"available", "complete"} for status in statuses
    ) and len(statuses) == len(rows) else "partial_or_missing"
    if not complete:
        for field in ("input_tokens", "output_tokens", "reported_cost", "currency"):
            reasons.setdefault(field, ["provider attempts did not pair one-to-one"])
    result["missing_reasons"] = reasons
    return result


def _read_event_log(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    events = []
    errors = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"event line {number} was invalid JSON")
                continue
            if isinstance(value, dict):
                events.append(value)
            else:
                errors.append(f"event line {number} was not a JSON object")
    return events, errors


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if re.fullmatch(r"[0-9a-fA-F]{7,64}", value) else None


def _build_brain(config):
    from src.llm.brain import AdamBrain

    # F1 must not read user skills, custom tools, or memory, and must never
    # acquire a desktop backend. Disable optional components only in this
    # in-memory config copy; the user's config file is read-only.
    config = config.model_copy(deep=True)
    # The normal Brain can construct a second client for its optional tool-free
    # route. This single-route trial always uses the configured primary client.
    config.llm.tool_free_model = ""
    config.llm.tool_free_provider_only = []
    config.llm.tool_free_allow_provider_fallbacks = False
    if config.computer_control is not None:
        config.computer_control.enabled = False
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
    brain.get_tools = lambda: []
    brain.tool_free_llm_client = None
    brain.llm_client = _CapturingToollessClient(brain.llm_client)
    return brain, tts


def run_trial(
    *,
    config_path: Path,
    prompt: str,
    fixture_run_id: str,
    run_id: str,
    config_id: str,
    output_path: Path,
    route_ready_confirmed: bool,
    expected_model: str,
    metadata_http_status: int | None,
    metadata_model_id: str | None,
    metadata_endpoint_count: int | None,
    metadata_checked_at: str | None,
    confirm_provider_inference: bool,
    brain_builder: Callable[[Any], tuple[Any, _CaptureOnlyTTS]] = _build_brain,
) -> dict[str, Any]:
    if prompt != FACTUAL_PROMPT:
        raise TrialError("F1 runner accepts only the exact run-sheet factual prompt")
    fixture_run_id = _validate_id(fixture_run_id, "fixture run ID")
    run_id = _validate_id(run_id, "opaque run ID")
    config_id = _validate_id(config_id, "config ID")
    if not route_ready_confirmed:
        raise TrialError("refusing inference until an operator confirms the supplied route metadata attestation")
    metadata_attestation = _validate_metadata_attestation(
        expected_model=expected_model,
        metadata_http_status=metadata_http_status,
        metadata_model_id=metadata_model_id,
        metadata_endpoint_count=metadata_endpoint_count,
        metadata_checked_at=metadata_checked_at,
    )
    if not confirm_provider_inference:
        raise TrialError("refusing live provider call without --confirm-provider-inference")

    config = _load_existing_config(config_path.expanduser())
    route = _validate_config(config, config_id=config_id, expected_model=expected_model)
    if metadata_attestation["model_id"] != route["model"]:
        raise TrialError("metadata model ID must exactly match the configured model")
    route["metadata_attestation"] = metadata_attestation
    secret_values = _config_secret_values(config)
    serialized_identity = [
        ("fixture run ID", fixture_run_id), ("opaque run ID", run_id),
        ("config ID", config_id), ("provider", route["provider"]),
        ("model", route["model"]), *[("provider route", item) for item in route["provider_only"]],
    ]
    for label, value in serialized_identity:
        if any(len(secret) >= 8 and secret.casefold() in value.casefold() for secret in secret_values):
            raise TrialError(f"{label} must not contain a configured credential")
    output_path, event_path = _private_output_paths(output_path, run_id)
    config_realpath = config_path.expanduser().resolve(strict=True)
    if output_path.resolve(strict=False) == config_realpath or event_path.resolve(strict=False) == config_realpath:
        raise TrialError("run output may not replace or append to the config file")
    record_fd = _open_private_jsonl(output_path, run_id=run_id)
    try:
        event_fd = _create_private_file(event_path)
    except Exception:
        os.close(record_fd)
        raise
    os.close(event_fd)

    # Never reuse any telemetry path configured by the user. EventWriter only
    # receives this run's already-created mode-0600 companion file.
    configure_telemetry(SimpleNamespace(enabled=True, path=str(event_path)))
    trace_token = set_trace_id(run_id)
    brain = None
    tts = None
    provider_capture = None
    suppressed_stdout = io.StringIO()
    suppressed_stderr = io.StringIO()
    brain_turn_start_ns = None
    brain_turn_end_ns = None
    execution_status = "error"
    exception_record = None
    try:
        with redirect_stdout(suppressed_stdout), redirect_stderr(suppressed_stderr), \
                patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
            brain, tts = brain_builder(config)
            provider_capture = brain.llm_client

            async def measured_brain_turn() -> None:
                nonlocal brain_turn_start_ns, brain_turn_end_ns
                brain_turn_start_ns = time.monotonic_ns()
                try:
                    await brain.process_user_utterance(prompt)
                finally:
                    brain_turn_end_ns = time.monotonic_ns()

            asyncio.run(measured_brain_turn())
            execution_status = "returned"
    except Exception as exc:
        exception_record = _safe_error(exc, secret_values)
    finally:
        if brain is not None:
            try:
                brain.close()
            except Exception as exc:
                exception_record = exception_record or _safe_error(exc, secret_values)
        reset_trace_id(trace_token)
    events, event_parse_errors = _read_event_log(event_path)
    provider_events = _provider_event_summary(events, run_id)
    normalized_responses = (
        list(provider_capture.responses)
        if isinstance(provider_capture, _CapturingToollessClient)
        else []
    )
    final_message = None
    if brain is not None:
        final_message = next(
            (str(item.get("content") or "") for item in reversed(brain.messages)
             if item.get("role") == "assistant"),
            None,
        )
        if final_message is not None and len(final_message) > _MAX_RESPONSE_CHARS:
            final_message = final_message[:_MAX_RESPONSE_CHARS]
    call_tts_ns = tts.submitted_at_ns[-1] if tts and tts.submitted_at_ns else None
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "trace_id": run_id,
        "fixture_id": fixture_run_id,
        "commit": _git_commit(),
        "host": platform.node() or None,
        "os": platform.platform(),
        "app": "AdamBrain headless F1 text run",
        "build": _git_commit(),
        "route": route,
        "scenario": "F1",
        "phase": "cold",
        "phase_scope": "fresh headless Brain process; no daemon, microphone, TTS playback, or desktop",
        "starting_state": {
            "conversation": "empty",
            "memory": "disabled",
            "skills": "disabled",
            "custom_tools": "disabled",
            "model_tools": "disabled",
            "audio": "capture-only TTS; no playback",
            "desktop": "disabled",
        },
        "prompt_id": "prompts.factual",
        "prompt": prompt,
        "oracle_id": "F1 shape and sauce/texture rubric; manual score required",
        "outcome": "not_scored",
        "oracle_status": "manual_review_required",
        "brain_execution_status": execution_status,
        "brain_response_text": final_message,
        "brain_response_text_missing_reason": None if final_message is not None else "Brain did not append a final assistant message",
        "normalized_provider_responses": normalized_responses,
        "normalized_provider_responses_missing_reason": (
            None if normalized_responses else "provider client did not return a normalized response"
        ),
        "unexpected_tool_call_stopped": any(
            response.get("tool_call_count", 0) > 0 for response in normalized_responses
        ),
        "errors": [exception_record] if exception_record else [],
        "timing_ms": {
            "headless_brain_call": (
                round((brain_turn_end_ns - brain_turn_start_ns) / 1_000_000, 3)
                if brain_turn_start_ns is not None and brain_turn_end_ns is not None else None
            ),
            "headless_brain_call_missing_reason": (
                None if brain_turn_start_ns is not None and brain_turn_end_ns is not None
                else "Brain turn did not reach a measured call boundary"
            ),
            "brain_final_text_submission": (
                round((call_tts_ns - brain_turn_start_ns) / 1_000_000, 3)
                if call_tts_ns is not None and brain_turn_start_ns is not None else None
            ),
            "request_end_to_verified_state": None,
            "request_end_to_verified_state_missing_reason": "F1 response has not been independently scored; this CLI has no user request-end or audible timestamp",
            "request_end_to_ack": None,
            "request_end_to_ack_missing_reason": "headless capture-only TTS does not measure audible playback",
        },
        "event_clock_ns": {
            "headless_brain_call_start": brain_turn_start_ns,
            "headless_brain_call_return": brain_turn_end_ns,
            "verified_completion": None,
        },
        "provider": provider_events,
        "telemetry": {
            "event_path": str(event_path),
            "event_file_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
            "matching_trace_id": run_id,
            "event_parse_errors": event_parse_errors,
        },
        "resource_trace": {
            "status": "not_collected",
            "missing_reason": "this narrow Brain CLI does not start the Adam service or resource sampler",
        },
        "capture": {"status": "not_applicable", "reason": "F1 is text-only"},
        "provider_raw_http_body_available": False,
        "provider_raw_http_body_missing_reason": "UniversalLLMClient returns a normalized assistant response and allowlisted telemetry, not the raw HTTP body",
        "private_output_paths": {
            "record": str(output_path),
            "events": str(event_path),
        },
    }
    try:
        _append_jsonl_fd(record_fd, record)
    finally:
        os.close(record_fd)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="existing protected Adam config; read-only")
    parser.add_argument("--prompt", required=True, help="must exactly match the F1 prompt in the run sheet")
    parser.add_argument("--fixture-run-id", required=True, help="opaque fixture/run-series ID")
    parser.add_argument("--run-id", required=True, help="unique opaque ID; also used as telemetry trace ID")
    parser.add_argument("--config-id", required=True, help="non-secret stable label for this route/config")
    parser.add_argument("--output", required=True, type=Path, help="private .jsonl series file under an existing mode-0700 directory")
    parser.add_argument("--expected-model", required=True, help="exact configured model ID to run and attest")
    parser.add_argument("--metadata-http-status", type=int, help="HTTP status from the metadata-only model check; must be 200")
    parser.add_argument("--metadata-model-id", help="exact model ID returned by the metadata-only check")
    parser.add_argument("--metadata-endpoint-count", type=int, help="number of serving endpoints in the metadata response")
    parser.add_argument("--metadata-checked-at", help="metadata check time as an ISO 8601 UTC timestamp ending in Z")
    parser.add_argument(
        "--route-ready-confirmed", action="store_true",
        help="confirm the supplied recent metadata status, model ID, endpoint count, and check time",
    )
    parser.add_argument(
        "--confirm-provider-inference", action="store_true",
        help="explicitly authorize one live provider-backed F1 turn",
    )
    args = parser.parse_args(argv)
    try:
        record = run_trial(
            config_path=args.config,
            prompt=args.prompt,
            fixture_run_id=args.fixture_run_id,
            run_id=args.run_id,
            config_id=args.config_id,
            output_path=args.output,
            route_ready_confirmed=args.route_ready_confirmed,
            expected_model=args.expected_model,
            metadata_http_status=args.metadata_http_status,
            metadata_model_id=args.metadata_model_id,
            metadata_endpoint_count=args.metadata_endpoint_count,
            metadata_checked_at=args.metadata_checked_at,
            confirm_provider_inference=args.confirm_provider_inference,
        )
    except TrialError as exc:
        print(f"F1 trial refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"F1 trial failed ({type(exc).__name__}); inspect the private record if created.", file=sys.stderr)
        return 1
    print(
        "F1 headless turn recorded; outcome remains not_scored. "
        f"Private record: {record['private_output_paths']['record']}"
    )
    return 0 if record["brain_execution_status"] == "returned" else 1


if __name__ == "__main__":
    raise SystemExit(main())
