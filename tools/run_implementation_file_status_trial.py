#!/usr/bin/env python3
"""Run one bounded, headless F2 file-status task through AdamBrain.

Provider inference is live only when both route attestation and inference are
explicitly confirmed. The runner accepts a generated fixture directory, offers
only read_file, limits dispatch to its source.txt, and captures TTS text without
opening an audio device or playing sound.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
import fcntl
import hashlib
import io
import json
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

# Keep imports working when called by absolute script path from outside the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STRING = str(PROJECT_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != PROJECT_ROOT_STRING]
sys.path.insert(0, PROJECT_ROOT_STRING)
if __name__ == "__main__":
    _src_spec = PathFinder.find_spec("src", [PROJECT_ROOT_STRING])
    if _src_spec is None or _src_spec.submodule_search_locations is None:
        raise ImportError("repository-local 'src' package could not be found")
    _src_package = module_from_spec(_src_spec)
    if _src_spec.loader is not None:
        _src_spec.loader.exec_module(_src_package)
    sys.modules["src"] = _src_package

from src.config import load_config
from src.llm.tools import CanonicalTool
from src.telemetry.events import configure_telemetry, reset_trace_id, set_trace_id

EXPECTED_PROVIDER = "custom"
EXPECTED_API_BASE = "https://openrouter.ai/api/v1"
FIXTURE_VERSION = 4
MAX_READ_CHARS = 512
MAX_TOOL_DISPATCHES = 1
MAX_LOGICAL_CHAT_CALLS = 2
MAX_RESPONSE_CHARS = 16_000
MAX_TOOL_TRACE_CHARS = 8_192
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
_SAFE_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/-]{0,199}\Z")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_READ_FILE_TOOL = CanonicalTool(
    name="read_file",
    description=(
        "Read the generated F2 source.txt fixture to report its UTF-8 byte size and exact first line. "
        "Use the exact absolute path from the request and no other path."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Exact absolute fixture source.txt path."},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": MAX_READ_CHARS},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)


class TrialError(ValueError):
    """Raised when the requested trial is unsafe or does not match F2."""


class _CaptureOnlyTTS:
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


def _validate_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise TrialError(f"{name} must be 1–80 safe opaque characters [A-Za-z0-9._-]")
    return value


def _validate_model_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_MODEL_ID.fullmatch(value)
        or "/" not in value
        or "//" in value
        or value.endswith("/")
    ):
        raise TrialError(f"{name} must be a safe namespaced model ID")
    return value


def _metadata_attestation(
    *, expected_model: str, http_status: int | None,
    model_id: str | None, endpoint_count: int | None, checked_at: str | None,
) -> dict[str, Any]:
    expected_model = _validate_model_id(expected_model, "expected model")
    if http_status != 200:
        raise TrialError("route metadata must attest HTTP status 200")
    if model_id is None or _validate_model_id(model_id, "metadata model ID") != expected_model:
        raise TrialError("metadata model ID must exactly match the expected model")
    if isinstance(endpoint_count, bool) or not isinstance(endpoint_count, int) or endpoint_count < 1:
        raise TrialError("metadata endpoint count must be a positive integer")
    if not isinstance(checked_at, str) or not _UTC_TIMESTAMP.fullmatch(checked_at):
        raise TrialError("metadata check time must be an ISO 8601 UTC timestamp ending in Z")
    try:
        checked = datetime.fromisoformat(checked_at[:-1] + "+00:00")
    except ValueError as exc:
        raise TrialError("metadata check time must be a valid UTC timestamp") from exc
    age = datetime.now(timezone.utc) - checked
    if age < -timedelta(minutes=5):
        raise TrialError("route metadata check time is too far in the future")
    if age > timedelta(hours=1):
        raise TrialError("route metadata is older than one hour; refresh the metadata check")
    return {
        "http_status": 200,
        "model_id": model_id,
        "endpoint_count": endpoint_count,
        "checked_at_utc": checked.isoformat().replace("+00:00", "Z"),
        "source": "operator-reported external metadata check",
        "runner_fetched_or_cryptographically_verified": False,
    }


def _load_config(path: Path):
    try:
        info = path.lstat()
    except OSError as exc:
        raise TrialError("--config must name an existing protected regular file") from exc
    if not stat.S_ISREG(info.st_mode):
        raise TrialError("--config must be a regular file; symlinks and special files are refused")
    if info.st_uid != os.geteuid():
        raise TrialError("--config must be owned by the current user")
    if info.st_mode & 0o077:
        raise TrialError("--config must not be accessible by group or other users")
    return load_config(str(path))


def _validate_route(config, *, config_id: str, expected_model: str) -> dict[str, Any]:
    _validate_id(config_id, "config ID")
    expected_model = _validate_model_id(expected_model, "expected model")
    llm = config.llm
    provider = str(getattr(llm, "provider", "")).lower()
    model = str(getattr(llm, "cloud_model", ""))
    api_base = str(getattr(llm, "api_base", ""))
    fallbacks = getattr(llm, "allow_provider_fallbacks", None)
    if provider != EXPECTED_PROVIDER:
        raise TrialError(f"configured provider must remain {EXPECTED_PROVIDER!r}")
    if model != expected_model:
        raise TrialError("configured model must exactly match --expected-model")
    if api_base not in {EXPECTED_API_BASE, EXPECTED_API_BASE + "/"}:
        raise TrialError(f"configured API base must remain pinned to {EXPECTED_API_BASE!r}")
    if fallbacks is not False:
        raise TrialError("provider fallbacks must be explicitly disabled for this matched route")
    provider_only = getattr(llm, "provider_only", None)
    if provider_only is not None and (
        not isinstance(provider_only, list) or any(not isinstance(item, str) for item in provider_only)
    ):
        raise TrialError("provider_only must be a list of provider labels")
    return {
        "provider": provider,
        "model": model,
        "api_base": EXPECTED_API_BASE,
        "config_id": config_id,
        "provider_only": list(provider_only or []),
        "allow_provider_fallbacks": False,
        "secondary_tool_free_route": "disabled for this trial",
    }


def _secret_values(config) -> list[str]:
    values: list[str] = []
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


def _scrub_text(value: str, secrets: list[str], *, limit: int) -> tuple[str, bool]:
    safe = str(value)
    for secret in secrets:
        safe = safe.replace(secret, "[REDACTED]")
    safe = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", safe)
    safe = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s&,;]+", r"\1[REDACTED]", safe)
    safe = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", safe)
    safe = re.sub(r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key)=)[^&#\s]+", r"\1[REDACTED]", safe)
    truncated = len(safe) > limit
    return safe[:limit], truncated


def _safe_error(exc: BaseException, secrets: list[str]) -> dict[str, str]:
    message, _ = _scrub_text(str(exc), secrets, limit=2000)
    return {"type": type(exc).__name__, "message": message}


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
    expected_path = resolved / "expected.json"
    source_path = resolved / "source.txt"
    expected = _read_owned_regular(expected_path, max_bytes=128_000)
    source_bytes = _read_owned_regular(source_path, max_bytes=16_384)
    try:
        metadata = json.loads(expected.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrialError("fixture expected.json is not valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("fixture_version") != FIXTURE_VERSION:
        raise TrialError("fixture must use the generated implementation fixture format")
    fixture_id = metadata.get("run_id")
    _validate_id(fixture_id, "fixture run ID")
    if fixture_id != resolved.name:
        raise TrialError("fixture run ID must match its generated directory name")
    paths = metadata.get("paths")
    prompts = metadata.get("prompts")
    oracles = metadata.get("oracles")
    expected_prompt = f'Read "{source_path}" and tell me its byte size and exact first line.'
    if not isinstance(paths, dict) or paths.get("root") != str(resolved):
        raise TrialError("fixture root metadata does not match --fixture-dir")
    if paths.get("source_file") != str(source_path):
        raise TrialError("fixture source path metadata does not match generated source.txt")
    if not isinstance(prompts, dict) or prompts.get("file_status") != expected_prompt:
        raise TrialError("fixture file-status prompt does not exactly match generated source.txt")
    file_oracle = oracles.get("file_status") if isinstance(oracles, dict) else None
    terminal_oracle = oracles.get("terminal") if isinstance(oracles, dict) else None
    if (
        not isinstance(file_oracle, dict)
        or isinstance(file_oracle.get("byte_length"), bool)
        or not isinstance(file_oracle.get("byte_length"), int)
        or file_oracle["byte_length"] < 1
        or not isinstance(file_oracle.get("first_line"), str)
        or not file_oracle["first_line"]
        or "\n" in file_oracle["first_line"]
        or "\r" in file_oracle["first_line"]
    ):
        raise TrialError("fixture file-status oracle is malformed")
    expected_source_hash = (
        terminal_oracle.get("source_sha256") if isinstance(terminal_oracle, dict) else None
    )
    if not isinstance(expected_source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_source_hash):
        raise TrialError("fixture source SHA-256 oracle is malformed")
    source_before = _snapshot_bytes(source_bytes)
    if source_before["sha256"] != expected_source_hash:
        raise TrialError("fixture source.txt does not match the expected source SHA-256 oracle")
    if (
        source_before["byte_length"] != file_oracle["byte_length"]
        or source_before["first_line"] != file_oracle["first_line"]
    ):
        raise TrialError("fixture file-status oracle does not match the generated source")
    return {
        "fixture_id": fixture_id,
        "fixture_dir": resolved,
        "source_path": source_path,
        "prompt": prompts["file_status"],
        "oracle": {
            "byte_length": file_oracle["byte_length"],
            "first_line": file_oracle["first_line"],
        },
        "source_before": source_before,
    }


def _read_owned_regular(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TrialError(f"fixture file {path.name} must be a regular non-symlink file") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_size > max_bytes
        ):
            raise TrialError(f"fixture file {path.name} must be an owned, bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise TrialError(f"fixture file {path.name} exceeds its size bound")
        return data
    finally:
        os.close(descriptor)


def _snapshot_bytes(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    first_line = text.splitlines()[0] if text.splitlines() else ""
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_length": len(data),
        "first_line": first_line,
    }


def _source_snapshot(path: Path) -> dict[str, Any]:
    return _snapshot_bytes(_read_owned_regular(path, max_bytes=16_384))


def _private_output_paths(output: Path, run_id: str, fixture_dir: Path) -> tuple[Path, Path]:
    output = output.expanduser()
    if not output.is_absolute() or output.suffix != ".jsonl":
        raise TrialError("--output must be an absolute .jsonl path in an existing private directory")
    try:
        parent_real = output.parent.resolve(strict=True)
        parent_info = output.parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise TrialError("--output parent must be an existing private directory") from exc
    if parent_real != output.parent.absolute():
        raise TrialError("--output path may not traverse symlinked directories")
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.geteuid()
        or parent_info.st_mode & 0o077
    ):
        raise TrialError("--output parent must be owned by the current user and mode 0700 or stricter")
    event_path = output.with_name(f"{run_id}.events.jsonl")
    for candidate in (output, event_path):
        resolved_candidate = candidate.resolve(strict=False)
        if resolved_candidate == fixture_dir or fixture_dir in resolved_candidate.parents:
            raise TrialError("run outputs must be outside the immutable fixture directory")
    try:
        info = output.lstat()
    except FileNotFoundError:
        info = None
    if info is not None and (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
    ):
        raise TrialError("existing --output must be an owned private single-link regular file")
    try:
        event_path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise TrialError("per-run event log must not already exist")
    return output, event_path


def _open_record(path: Path, run_id: str) -> int:
    descriptor = None
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
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
        _validate_existing_records(path, run_id)
        return descriptor
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _validate_existing_records(path: Path, run_id: str) -> None:
    try:
        stream = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return
    with stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrialError(f"existing output JSONL is invalid on line {number}") from exc
            if not isinstance(record, dict) or record.get("app") != "AdamBrain headless F2 file-status run":
                raise TrialError(f"existing output JSONL line {number} is not an F2 run record")
            if record.get("schema_version") != 1:
                raise TrialError(f"existing output JSONL line {number} has an unsupported schema")
            if record.get("run_id") == run_id:
                raise TrialError(f"run ID {run_id!r} already exists in output JSONL")


def _create_private_file(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        return descriptor
    except OSError as exc:
        raise TrialError(f"could not create private event log ({type(exc).__name__})") from exc


def _append_record(descriptor: int, record: dict[str, Any]) -> None:
    payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short append to private run JSONL")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _private_value(value: Any, secrets: list[str], *, string_limit: int = MAX_RESPONSE_CHARS) -> Any:
    if isinstance(value, str):
        return _scrub_text(value, secrets, limit=string_limit)[0]
    if isinstance(value, list):
        return [_private_value(item, secrets, string_limit=string_limit) for item in value[:64]]
    if isinstance(value, dict):
        return {
            str(key)[:100]: _private_value(item, secrets, string_limit=string_limit)
            for key, item in list(value.items())[:64]
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


def _provider_attempt_summary(event_bytes: bytes, trace_id: str) -> dict[str, Any]:
    """Count client HTTP attempts and retries from the allowlisted event trace."""
    events: list[dict[str, Any]] = []
    parse_errors = 0
    try:
        lines = event_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        lines = []
        parse_errors += 1
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if not isinstance(event, dict):
            parse_errors += 1
            continue
        if event.get("trace_id") == trace_id:
            events.append(event)

    starts = [event for event in events if event.get("event") == "llm.request_started"]
    completions = [event for event in events if event.get("event") == "llm.completed"]
    retries = [event for event in events if event.get("event") == "llm.retrying"]
    start_spans = [event.get("span_id") for event in starts if isinstance(event.get("span_id"), str)]
    completion_spans = [event.get("span_id") for event in completions if isinstance(event.get("span_id"), str)]
    attempt_telemetry_complete = (
        bool(starts)
        and not parse_errors
        and len(start_spans) == len(starts) == len(completions) == len(completion_spans)
        and Counter(start_spans) == Counter(completion_spans)
    )
    statuses = []
    for event in completions:
        attributes = event.get("attributes")
        status = attributes.get("http_status") if isinstance(attributes, dict) else None
        if isinstance(status, int) and not isinstance(status, bool):
            statuses.append(status)

    return {
        "http_attempts_observed": len(starts),
        "http_completions_observed": len(completions),
        "http_statuses_observed": statuses,
        "retry_events_observed": len(retries),
        "attempt_telemetry_status": (
            "complete" if attempt_telemetry_complete
            else "incomplete" if starts or completions or parse_errors
            else "unavailable"
        ),
        "event_parse_error_count": parse_errors,
    }


class _CapturingClient:
    """Keep normalized provider output and enforce one read_file-only schema."""

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
            raise TrialError("F2 runner stopped after its two logical chat-call bound")
        offered = list(tools or [])
        names = [getattr(tool, "name", None) for tool in offered]
        if any(name != "read_file" for name in names) or len(names) > 1:
            raise TrialError("F2 runner refused a tool schema other than its single read_file schema")
        self.requests.append({"call_number": len(self.requests) + 1, "offered_tool_names": names})
        response = await self._client.chat(messages, tools=offered, max_tokens=max_tokens, think=think)
        if not isinstance(response, dict):
            self.responses.append({"response_type": type(response).__name__, "content": None, "tool_calls": []})
            return response
        content = response.get("content")
        safe_content, truncated = _scrub_text(
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
                raw_arguments = _scrub_text(raw_arguments, self.secrets, limit=4000)[0]
                calls.append({
                    "id": str(call.get("id", ""))[:100] if isinstance(call, dict) else None,
                    "name": str(function.get("name", ""))[:100],
                    "arguments": raw_arguments,
                })
        self.responses.append({
            "role": response.get("role") if isinstance(response.get("role"), str) else None,
            "content": safe_content,
            "content_truncated": truncated,
            "tool_calls": calls,
            "provider_error": response.get("provider_error") is True,
        })
        return response


def _build_brain(config, source_path: Path, secrets: list[str], oracle: dict[str, Any]):
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
    brain.get_tools = lambda: [_READ_FILE_TOOL]
    brain.tool_free_llm_client = None
    brain.max_tool_rounds = MAX_LOGICAL_CHAT_CALLS
    brain.llm_client = _CapturingClient(brain.llm_client, secrets)
    dispatch_trace: list[dict[str, Any]] = []
    original_execute = brain._execute_tool
    dispatch_lock = asyncio.Lock()

    async def restricted_execute_locked(name: str, args: dict) -> str:
        entry: dict[str, Any] = {
            "tool_name": str(name)[:100],
            "arguments": _private_value(args, secrets, string_limit=2000),
        }
        dispatch_trace.append(entry)
        if len(dispatch_trace) > MAX_TOOL_DISPATCHES:
            entry["status"] = "refused"
            entry["reason"] = "dispatch limit reached"
            raise TrialError("F2 runner permits only one tool dispatch")
        if name != "read_file":
            entry["status"] = "refused"
            entry["reason"] = "only read_file is permitted"
            raise TrialError("F2 runner permits only read_file")
        if not isinstance(args, dict) or set(args) - {"path", "max_chars"}:
            entry["status"] = "refused"
            entry["reason"] = "unexpected read_file arguments"
            raise TrialError("F2 runner refused unexpected read_file arguments")
        if args.get("path") != str(source_path):
            entry["status"] = "refused"
            entry["reason"] = "path did not exactly match generated fixture source.txt"
            raise TrialError("F2 runner refused a path outside its exact fixture source.txt")
        requested_chars = args.get("max_chars", MAX_READ_CHARS)
        if (
            isinstance(requested_chars, bool)
            or not isinstance(requested_chars, int)
            or not 1 <= requested_chars <= MAX_READ_CHARS
        ):
            entry["status"] = "refused"
            entry["reason"] = "max_chars must be an integer from 1 through 512"
            raise TrialError("F2 runner refused an out-of-bounds read_file max_chars value")
        bounded_args = {"path": str(source_path), "max_chars": requested_chars}
        entry["status"] = "dispatched"
        entry["bounded_max_chars"] = requested_chars
        try:
            result = await original_execute("read_file", bounded_args)
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = _safe_error(exc, secrets)
            raise
        result_text = result if isinstance(result, str) else str(result)
        evidence_verified = False
        try:
            tool_payload = json.loads(result_text)
            readback = tool_payload.get("readback") if isinstance(tool_payload, dict) else None
            expected_header = (
                f"File: {source_path}\nBytes: {oracle['byte_length']}\nText:\n"
            )
            evidence_verified = (
                isinstance(tool_payload, dict)
                and tool_payload.get("ok") is True
                and isinstance(readback, str)
                and readback.startswith(expected_header)
                and readback[len(expected_header):].splitlines()[:1] == [oracle["first_line"]]
            )
        except (TypeError, ValueError):
            pass
        safe_result, truncated = _scrub_text(result_text, secrets, limit=MAX_TOOL_TRACE_CHARS)
        entry["status"] = "returned"
        entry["evidence_verified"] = evidence_verified
        entry["result"] = safe_result
        entry["result_truncated"] = truncated
        return result_text

    async def restricted_execute(name: str, args: dict) -> str:
        # Brain may run a batch of read-only tool calls concurrently. Serialize
        # this guard so its dispatch count and trace stay correct under that path.
        async with dispatch_lock:
            return await restricted_execute_locked(name, args)

    brain._f2_dispatch_trace = dispatch_trace
    brain._execute_tool = restricted_execute
    return brain, tts


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


def run_trial(
    *, fixture_dir: Path, config_path: Path, run_id: str, config_id: str,
    output_path: Path, route_ready_confirmed: bool, expected_model: str,
    metadata_http_status: int | None, metadata_model_id: str | None,
    metadata_endpoint_count: int | None, metadata_checked_at: str | None,
    confirm_provider_inference: bool,
    brain_builder: Callable[..., tuple[Any, _CaptureOnlyTTS]] = _build_brain,
) -> dict[str, Any]:
    fixture = _validate_fixture(fixture_dir)
    run_id = _validate_id(run_id, "opaque run ID")
    config_id = _validate_id(config_id, "config ID")
    if not route_ready_confirmed:
        raise TrialError("refusing inference until the operator confirms a fresh external metadata check")
    attestation = _metadata_attestation(
        expected_model=expected_model, http_status=metadata_http_status,
        model_id=metadata_model_id, endpoint_count=metadata_endpoint_count,
        checked_at=metadata_checked_at,
    )
    if not confirm_provider_inference:
        raise TrialError("refusing live provider call without --confirm-provider-inference")

    config_path = config_path.expanduser()
    config = _load_config(config_path)
    route = _validate_route(config, config_id=config_id, expected_model=expected_model)
    if attestation["model_id"] != route["model"]:
        raise TrialError("metadata model ID must exactly match the configured model")
    route["operator_reported_metadata_check"] = attestation
    secret_values = _secret_values(config)
    for label, value in (
        ("fixture run ID", fixture["fixture_id"]), ("opaque run ID", run_id),
        ("config ID", config_id), ("model ID", route["model"]),
        *[("provider route", item) for item in route["provider_only"]],
    ):
        if any(len(secret) >= 4 and secret.casefold() in value.casefold() for secret in secret_values):
            raise TrialError(f"{label} must not contain a configured credential")
    output_path, event_path = _private_output_paths(output_path, run_id, fixture["fixture_dir"])
    config_realpath = config_path.resolve(strict=True)
    if config_realpath in {output_path.resolve(strict=False), event_path.resolve(strict=False)}:
        raise TrialError("run output may not replace or append to the config file")
    record_fd = _open_record(output_path, run_id)
    try:
        event_fd = _create_private_file(event_path)
    except Exception:
        os.close(record_fd)
        raise
    os.close(event_fd)

    configure_telemetry(SimpleNamespace(enabled=True, path=str(event_path)))
    trace_token = set_trace_id(run_id)
    brain = None
    tts = None
    client_capture = None
    suppressed_stdout = io.StringIO()
    suppressed_stderr = io.StringIO()
    start_ns = end_ns = None
    execution_status = "error"
    error_record = None
    try:
        with redirect_stdout(suppressed_stdout), redirect_stderr(suppressed_stderr), \
                patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
            brain, tts = brain_builder(
                config, fixture["source_path"], secret_values, fixture["oracle"],
            )
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
        error_record = _safe_error(exc, secret_values)
    finally:
        if brain is not None:
            try:
                brain.close()
            except Exception as exc:
                error_record = error_record or _safe_error(exc, secret_values)
        reset_trace_id(trace_token)

    try:
        source_after = _source_snapshot(fixture["source_path"])
        source_error = None
    except Exception as exc:
        source_after = None
        source_error = _safe_error(exc, secret_values)
    source_before = fixture["source_before"]
    source_unchanged = source_after == source_before
    dispatch_trace = getattr(brain, "_f2_dispatch_trace", []) if brain is not None else []
    final_answer = None
    if brain is not None:
        final_answer = next(
            (str(item.get("content") or "") for item in reversed(brain.messages)
             if item.get("role") == "assistant"),
            None,
        )
    safe_answer, answer_truncated = _scrub_text(
        final_answer or "", secret_values, limit=MAX_RESPONSE_CHARS,
    )
    read_file_evidence_verified = any(
        entry.get("tool_name") == "read_file"
        and entry.get("status") == "returned"
        and entry.get("evidence_verified") is True
        for entry in dispatch_trace
    ) if brain is not None else False
    objective_evidence_reasons = []
    if not read_file_evidence_verified:
        objective_evidence_reasons.append("no successful read_file result verified the fixture byte length and first line")
    if not source_unchanged:
        objective_evidence_reasons.append("source.txt SHA-256, byte length, or first line changed")
    objective_evidence_status = "pass" if not objective_evidence_reasons else "fail"
    answer_review_status = "manual_review_required" if final_answer is not None else "not_assessable"
    oracle_reasons = (
        ["Independently review the captured answer for an affirmative, exact byte-size and first-line claim; natural-language text is not automatically scored."]
        if final_answer is not None else ["Brain did not append a final assistant answer"]
    )

    event_bytes = event_path.read_bytes()
    event_digest = hashlib.sha256(event_bytes).hexdigest()
    provider_attempt_summary = _provider_attempt_summary(event_bytes, run_id)
    event_info = event_path.stat()
    if not stat.S_ISREG(event_info.st_mode) or event_info.st_mode & 0o077:
        error_record = error_record or {"type": "TrialError", "message": "event log did not remain private"}
    responses = client_capture.responses if isinstance(client_capture, _CapturingClient) else []
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "trace_id": run_id,
        "fixture_id": fixture["fixture_id"],
        "commit": _git_commit(),
        "host": platform.node() or None,
        "os": platform.platform(),
        "app": "AdamBrain headless F2 file-status run",
        "build": _git_commit(),
        "route": route,
        "scenario": "F2",
        "phase": "cold",
        "phase_scope": "fresh headless Brain call; no daemon, microphone, TTS playback, or desktop capture",
        "starting_state": {
            "conversation": "empty",
            "memory": "disabled",
            "skills": "disabled",
            "custom_tools": "disabled",
            "model_tools": ["read_file"],
            "audio": "capture-only TTS; no playback",
            "desktop": "disabled",
            "browser": "disabled",
        },
        "prompt_id": "prompts.file_status",
        "prompt": fixture["prompt"],
        "oracle_id": "oracles.file_status; objective source/readback checks plus independent answer review",
        "oracle": {
            "status": answer_review_status,
            "answer_review_status": answer_review_status,
            "objective_evidence_status": objective_evidence_status,
            "objective_evidence_reasons": objective_evidence_reasons,
            "reasons": oracle_reasons,
            "read_file_evidence_verified": read_file_evidence_verified,
            "source_unchanged": source_unchanged,
            "expected": fixture["oracle"],
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
            "request_end_to_verified_state_missing_reason": (
                "capture-only headless F2 run does not measure user request end or independent filesystem verification"
            ),
            "request_end_to_ack": None,
            "request_end_to_ack_missing_reason": "capture-only TTS does not measure audible playback",
        },
        "event_clock_ns": {"headless_brain_call_start": start_ns, "headless_brain_call_return": end_ns},
        "source_integrity": {
            "before": source_before,
            "after": source_after,
            "unchanged": source_unchanged,
            "after_error": source_error,
        },
        "telemetry": {
            "event_path": str(event_path),
            "event_file_sha256": event_digest,
            "matching_trace_id": run_id,
            "logical_primary_client_chat_calls": (
                len(client_capture.requests) if isinstance(client_capture, _CapturingClient) else 0
            ),
            "logical_primary_client_chat_call_limit": MAX_LOGICAL_CHAT_CALLS,
            **provider_attempt_summary,
            "attempt_scope_note": (
                "HTTP attempts and retry events are counts observed in Adam's client telemetry; "
                "provider-side rerouting or retries beyond this client are not observable."
            ),
        },
        "provider_raw_http_body_available": False,
        "provider_raw_http_body_missing_reason": (
            "UniversalLLMClient exposes normalized responses and allowlisted telemetry, not the raw HTTP body"
        ),
        "private_output_paths": {"record": str(output_path), "events": str(event_path)},
    }
    try:
        _append_record(record_fd, record)
    finally:
        os.close(record_fd)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path, help="absolute owner-private fixture generated by create_implementation_fixtures.py")
    parser.add_argument("--config", required=True, type=Path, help="existing protected Adam config; read-only")
    parser.add_argument("--run-id", required=True, help="unique opaque ID; also used as telemetry trace ID")
    parser.add_argument("--config-id", required=True, help="non-secret stable label for this route/config")
    parser.add_argument("--output", required=True, type=Path, help="private .jsonl series file under an existing mode-0700 directory")
    parser.add_argument("--expected-model", required=True, help="exact configured model ID to run and attest")
    parser.add_argument("--metadata-http-status", required=True, type=int, help="operator-reported HTTP status from an external metadata-only check; must be 200")
    parser.add_argument("--metadata-model-id", required=True, help="operator-reported exact model ID from the external metadata-only check")
    parser.add_argument("--metadata-endpoint-count", required=True, type=int, help="operator-reported serving endpoint count from the external metadata response")
    parser.add_argument("--metadata-checked-at", required=True, help="operator-reported external check time as ISO 8601 UTC ending in Z")
    parser.add_argument("--route-ready-confirmed", action="store_true", help="confirm a fresh external metadata-only check; the runner validates supplied fields but does not fetch or authenticate them")
    parser.add_argument("--confirm-provider-inference", action="store_true", help="explicitly authorize one live F2 provider-backed turn")
    args = parser.parse_args(argv)
    try:
        record = run_trial(
            fixture_dir=args.fixture_dir,
            config_path=args.config,
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
        safe_message, _ = _scrub_text(str(exc), [], limit=500)
        print(f"F2 trial refused: {safe_message}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"F2 trial failed ({type(exc).__name__}); inspect the private record if created.", file=sys.stderr)
        return 1
    print(
        f"F2 headless run recorded with oracle status {record['oracle']['status']}. "
        f"Private record: {record['private_output_paths']['record']}"
    )
    return 0 if record["brain_execution_status"] == "returned" else 1


if __name__ == "__main__":
    raise SystemExit(main())
