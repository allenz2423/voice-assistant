#!/usr/bin/env python3
"""Run one bounded, headless D1 document task through AdamBrain.

The runner accepts only the generated implementation fixture and exposes only
its exact ``document.txt`` through guarded, directory-descriptor-bound tool
handlers. It does not delegate fixture I/O to the normal filesystem handlers.
Provider inference is live only when the operator confirms both a fresh route
attestation and one inference turn. Desktop, browser, microphone, and audio
playback are disabled.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stderr, redirect_stdout
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
import secrets
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

from tools import run_implementation_file_status_trial as f2
from tools import run_implementation_system_status_trial as f3
from src.llm.tools import CanonicalTool
from src.telemetry.events import configure_telemetry, reset_trace_id, set_trace_id

APP_ID = "AdamBrain headless D1 document run"
SCENARIO = "D1"
FIXTURE_VERSION = 4
MAX_DOCUMENT_BYTES = 4096
MAX_READ_CHARS = 512
MAX_TOOL_DISPATCHES = 3
MAX_LOGICAL_CHAT_CALLS = 4
MAX_RESPONSE_CHARS = 16_000
MAX_TOOL_TRACE_CHARS = 8_192

_READ_FILE_TOOL = CanonicalTool(
    name="read_file",
    description="Read UTF-8 text from the exact generated D1 document.txt path requested by the user.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Exact absolute generated document.txt path."},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": MAX_READ_CHARS},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)
_WRITE_FILE_TOOL = CanonicalTool(
    name="write_file",
    description="Write UTF-8 text to the exact generated D1 document.txt path requested by the user.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Exact absolute generated document.txt path."},
            "content": {"type": "string", "maxLength": MAX_DOCUMENT_BYTES},
            "overwrite": {"type": "boolean", "enum": [True]},
        },
        "required": ["path", "content", "overwrite"],
        "additionalProperties": False,
    },
)


class TrialError(ValueError):
    """Raised when the requested D1 trial is unsafe or malformed."""


def _open_absolute_directory_nofollow(path: Path) -> int:
    """Open an absolute directory by walking each component without following symlinks."""
    if not path.is_absolute():
        raise TrialError("fixture directory must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor or "/", flags)
    try:
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise TrialError("fixture directory contains an unsafe path component")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_fixture_entry(fixture_fd: int, name: str, *, max_bytes: int) -> bytes:
    """Read one regular fixture entry relative to its pinned directory descriptor."""
    if name not in {"expected.json", "document.txt"}:
        raise TrialError("internal fixture entry is not allowlisted")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=fixture_fd)
    except OSError as exc:
        raise TrialError(f"fixture file {name} must be a regular non-symlink file") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_size > max_bytes
        ):
            raise TrialError(f"fixture file {name} must be an owned, bounded regular file")
        data = os.read(descriptor, max_bytes + 1)
        if len(data) > max_bytes:
            raise TrialError(f"fixture file {name} exceeds its size bound")
        current = os.stat(name, dir_fd=fixture_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise TrialError(f"fixture file {name} changed during the read")
        return data
    finally:
        os.close(descriptor)


def _write_fixture_document(
    fixture_fd: int, *, initial_bytes: bytes, final_bytes: bytes,
) -> None:
    """Atomically update document.txt using only names relative to the pinned fixture dir."""
    current = _read_fixture_entry(fixture_fd, "document.txt", max_bytes=MAX_DOCUMENT_BYTES)
    if current != initial_bytes:
        raise TrialError("D1 runner refused an unexpected document state before save")

    temporary_name = f".document.txt.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary_name, flags, 0o600, dir_fd=fixture_fd)
    try:
        remaining = memoryview(final_bytes)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write to fixture temporary file")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=fixture_fd)
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)

    try:
        current = _read_fixture_entry(fixture_fd, "document.txt", max_bytes=MAX_DOCUMENT_BYTES)
        if current != initial_bytes:
            raise TrialError("D1 runner refused a document changed before the atomic save")
        # renameat replaces a destination symlink itself; it never follows that link.
        # Both names are fixed allowlisted entries relative to the pinned directory fd.
        os.replace(
            temporary_name, "document.txt",
            src_dir_fd=fixture_fd, dst_dir_fd=fixture_fd,
        )
        os.fsync(fixture_fd)
    except Exception:
        try:
            os.unlink(temporary_name, dir_fd=fixture_fd)
        except OSError:
            pass
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

    fixture_fd = _open_absolute_directory_nofollow(resolved)
    try:
        directory_info = os.fstat(fixture_fd)
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or directory_info.st_uid != os.geteuid()
            or directory_info.st_mode & 0o077
            or directory_info.st_mode & 0o700 != 0o700
        ):
            raise TrialError("fixture directory must be owned by the current user and mode 0700")

        expected_path = resolved / "expected.json"
        document_path = resolved / "document.txt"
        expected_bytes = _read_fixture_entry(fixture_fd, "expected.json", max_bytes=128_000)
        document_bytes = _read_fixture_entry(fixture_fd, "document.txt", max_bytes=MAX_DOCUMENT_BYTES)
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
        oracles = metadata.get("oracles")
        expected_prompt = (
            f'In "{document_path}", change DRAFT to FINAL and save it. '
            "Leave every other character unchanged."
        )
        if not isinstance(paths, dict) or paths.get("root") != str(resolved):
            raise TrialError("fixture root metadata does not match --fixture-dir")
        if paths.get("document_file") != str(document_path):
            raise TrialError("fixture document path metadata does not match generated document.txt")
        if not isinstance(prompts, dict) or prompts.get("document") != expected_prompt:
            raise TrialError("fixture D1 prompt does not exactly match generated document.txt")

        document_oracle = oracles.get("document") if isinstance(oracles, dict) else None
        initial_text = "STATUS: DRAFT\nOWNER: Adam\n"
        final_text = "STATUS: FINAL\nOWNER: Adam\n"
        if document_oracle != {"initial_text": initial_text, "final_text": final_text}:
            raise TrialError("fixture D1 document oracle does not match the generated task")
        initial_bytes = initial_text.encode("utf-8")
        final_bytes = final_text.encode("utf-8")
        if document_bytes != initial_bytes:
            raise TrialError("document.txt does not match the generated D1 starting bytes")

        return {
            "fixture_id": fixture_id,
            "fixture_dir": resolved,
            "fixture_fd": fixture_fd,
            "expected_path": expected_path,
            "document_path": document_path,
            "expected_json_sha256_before": hashlib.sha256(expected_bytes).hexdigest(),
            "prompt": expected_prompt,
            "initial_text": initial_text,
            "final_text": final_text,
            "initial_bytes": initial_bytes,
            "final_bytes": final_bytes,
            "document_before": f2._snapshot_bytes(document_bytes),
        }
    except Exception:
        os.close(fixture_fd)
        raise


def _read_fixture_document(fixture: dict[str, Any], max_chars: int) -> tuple[str, str, bool]:
    """Return a read_file-compatible envelope from the pinned fixture dirfd."""
    raw = _read_fixture_entry(
        fixture["fixture_fd"], "document.txt", max_bytes=MAX_DOCUMENT_BYTES,
    )
    if raw == fixture["initial_bytes"]:
        state = "initial"
    elif raw == fixture["final_bytes"]:
        state = "final"
    else:
        raise TrialError("D1 runner refused an unexpected document state during read")
    text_bytes = raw[:max_chars]
    text = text_bytes.decode("utf-8", errors="replace")
    truncated = len(raw) > max_chars
    suffix = "\n[truncated]" if truncated else ""
    readback = f"File: {fixture['document_path']}\nBytes: {len(raw)}\nText:\n{text}{suffix}"
    return json.dumps({"ok": True, "readback": readback}, ensure_ascii=False), state, not truncated


def _open_document_record(path: Path, run_id: str) -> int:
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
                        or record.get("app") != APP_ID
                        or record.get("schema_version") != 1
                    ):
                        raise TrialError(f"existing output JSONL line {number} is not a D1 run record")
                    if record.get("run_id") == run_id:
                        raise TrialError(f"run ID {run_id!r} already exists in output JSONL")
        return descriptor
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


class _CapturingClient:
    """Capture provider output while enforcing the exact two D1 schemas."""

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
            raise TrialError("D1 runner stopped after its four logical chat-call bound")
        offered = list(tools or [])
        names = [getattr(tool, "name", None) for tool in offered]
        if names != ["read_file", "write_file"]:
            raise TrialError("D1 runner refused a tool schema other than read_file and write_file")
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
                raw_arguments = f2._scrub_text(raw_arguments, self.secrets, limit=4000)[0]
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


def _build_brain(config, document_path: Path, fixture: dict[str, Any], secrets: list[str]):
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
    tts = f2._CaptureOnlyTTS()
    with patch("src.llm.brain.CustomToolManager", return_value=f2._NoCustomTools()), \
            patch("src.llm.brain.SkillManager", f2._NoSkills):
        brain = AdamBrain(
            config, supervisor=None, probe=None, confirmation_mgr=None,
            tts_engine=tts, memory_mgr=f2._NoMemory(),
        )
    brain.get_tools = lambda: [_READ_FILE_TOOL, _WRITE_FILE_TOOL]
    brain.tool_free_llm_client = None
    brain.max_tool_rounds = MAX_LOGICAL_CHAT_CALLS
    brain.llm_client = _CapturingClient(brain.llm_client, secrets)
    dispatch_trace: list[dict[str, Any]] = []
    dispatch_lock = asyncio.Lock()
    state = {"read_dispatches": 0, "write_returned": False}

    async def restricted_execute_locked(name: str, args: dict) -> str:
        entry: dict[str, Any] = {
            "tool_name": str(name)[:100],
            "arguments": f2._private_value(args, secrets, string_limit=2000),
        }
        dispatch_trace.append(entry)
        if len(dispatch_trace) > MAX_TOOL_DISPATCHES:
            entry["status"] = "refused"
            entry["reason"] = "dispatch limit reached"
            raise TrialError("D1 runner permits at most three tool dispatches")
        if name not in {"read_file", "write_file"}:
            entry["status"] = "refused"
            entry["reason"] = "only read_file and write_file are permitted"
            raise TrialError("D1 runner permits only read_file and write_file")
        if not isinstance(args, dict):
            entry["status"] = "refused"
            entry["reason"] = "tool arguments must be an object"
            raise TrialError("D1 runner refused non-object tool arguments")
        if args.get("path") != str(document_path):
            entry["status"] = "refused"
            entry["reason"] = "path did not exactly match generated fixture document.txt"
            raise TrialError("D1 runner refused a path outside its exact fixture document.txt")

        if name == "read_file":
            if set(args) - {"path", "max_chars"}:
                entry["status"] = "refused"
                entry["reason"] = "unexpected read_file arguments"
                raise TrialError("D1 runner refused unexpected read_file arguments")
            requested_chars = args.get("max_chars", MAX_READ_CHARS)
            if (
                isinstance(requested_chars, bool)
                or not isinstance(requested_chars, int)
                or not 1 <= requested_chars <= MAX_READ_CHARS
            ):
                entry["status"] = "refused"
                entry["reason"] = "max_chars is outside the D1 document bound"
                raise TrialError("D1 runner refused an out-of-bounds read_file max_chars value")
            if state["read_dispatches"] >= 2:
                entry["status"] = "refused"
                entry["reason"] = "read dispatch limit reached"
                raise TrialError("D1 runner permits at most two read_file dispatches")
            entry["status"] = "dispatched"
            try:
                result_text, read_state, complete_read = _read_fixture_document(fixture, requested_chars)
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f2._safe_error(exc, secrets)
                raise
            safe_result, truncated = f2._scrub_text(result_text, secrets, limit=MAX_TOOL_TRACE_CHARS)
            entry["status"] = "returned"
            entry["read_state"] = read_state
            entry["evidence_verified"] = complete_read
            entry["result"] = safe_result
            entry["result_truncated"] = truncated
            entry["read_dispatch"] = state["read_dispatches"] + 1
            state["read_dispatches"] += 1
            return result_text

        if set(args) != {"path", "content", "overwrite"}:
            entry["status"] = "refused"
            entry["reason"] = "write_file requires only path, content, and overwrite"
            raise TrialError("D1 runner refused unexpected write_file arguments")
        if state["write_returned"]:
            entry["status"] = "refused"
            entry["reason"] = "write dispatch limit reached"
            raise TrialError("D1 runner permits at most one write_file dispatch")
        if args.get("overwrite") is not True or args.get("content") != fixture["final_text"]:
            entry["status"] = "refused"
            entry["reason"] = "write must use overwrite=true and the exact D1 final text"
            raise TrialError("D1 runner refused content or overwrite arguments outside the exact oracle")
        entry["status"] = "dispatched"
        entry["authorized_fixture_overwrite"] = True
        try:
            _write_fixture_document(
                fixture["fixture_fd"],
                initial_bytes=fixture["initial_bytes"],
                final_bytes=fixture["final_bytes"],
            )
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f2._safe_error(exc, secrets)
            raise
        result_text = f"Updated {document_path} ({len(fixture['final_bytes'])} bytes)."
        safe_result, truncated = f2._scrub_text(result_text, secrets, limit=MAX_TOOL_TRACE_CHARS)
        entry["status"] = "returned"
        entry["result"] = safe_result
        entry["result_truncated"] = truncated
        state["write_returned"] = True
        return result_text

    async def restricted_execute(name: str, args: dict) -> str:
        async with dispatch_lock:
            return await restricted_execute_locked(name, args)

    brain._d1_dispatch_trace = dispatch_trace
    brain._d1_dispatch_state = state
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
    brain_builder: Callable[..., tuple[Any, Any]] = _build_brain,
) -> dict[str, Any]:
    fixture = _validate_fixture(fixture_dir)
    try:
        return _run_validated_fixture_trial(
            fixture=fixture,
            config_path=config_path,
            run_id=run_id,
            config_id=config_id,
            output_path=output_path,
            route_ready_confirmed=route_ready_confirmed,
            expected_model=expected_model,
            metadata_http_status=metadata_http_status,
            metadata_model_id=metadata_model_id,
            metadata_endpoint_count=metadata_endpoint_count,
            metadata_checked_at=metadata_checked_at,
            confirm_provider_inference=confirm_provider_inference,
            brain_builder=brain_builder,
        )
    finally:
        os.close(fixture["fixture_fd"])


def _run_validated_fixture_trial(
    *, fixture: dict[str, Any], config_path: Path, run_id: str, config_id: str,
    output_path: Path, route_ready_confirmed: bool, expected_model: str,
    metadata_http_status: int | None, metadata_model_id: str | None,
    metadata_endpoint_count: int | None, metadata_checked_at: str | None,
    confirm_provider_inference: bool,
    brain_builder: Callable[..., tuple[Any, Any]],
) -> dict[str, Any]:
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
    record_fd = _open_document_record(output_path, run_id)
    try:
        event_fd = f2._create_private_file(event_path)
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
    start_ns = end_ns = verification_ns = None
    execution_status = "error"
    error_record = None
    try:
        with redirect_stdout(suppressed_stdout), redirect_stderr(suppressed_stderr), \
                patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
            brain, tts = brain_builder(
                config, fixture["document_path"], fixture, secret_values,
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
        error_record = f2._safe_error(exc, secret_values)
    finally:
        if brain is not None:
            try:
                brain.close()
            except Exception as exc:
                error_record = error_record or f2._safe_error(exc, secret_values)
        reset_trace_id(trace_token)

    try:
        document_after_bytes = _read_fixture_entry(
            fixture["fixture_fd"], "document.txt", max_bytes=MAX_DOCUMENT_BYTES,
        )
        document_after = f2._snapshot_bytes(document_after_bytes)
        document_after_error = None
    except Exception as exc:
        document_after_bytes = None
        document_after = None
        document_after_error = f2._safe_error(exc, secret_values)
    try:
        expected_after = _read_fixture_entry(
            fixture["fixture_fd"], "expected.json", max_bytes=128_000,
        )
        expected_unchanged = (
            hashlib.sha256(expected_after).hexdigest() == fixture["expected_json_sha256_before"]
        )
        expected_after_error = None
    except Exception as exc:
        expected_unchanged = False
        expected_after_error = f2._safe_error(exc, secret_values)

    verification_ns = time.monotonic_ns()
    dispatch_trace = getattr(brain, "_d1_dispatch_trace", []) if brain is not None else []
    final_answer = None
    if brain is not None:
        final_answer = next(
            (str(item.get("content") or "") for item in reversed(brain.messages)
             if item.get("role") == "assistant"),
            None,
        )
    safe_answer, answer_truncated = f2._scrub_text(
        final_answer or "", secret_values, limit=MAX_RESPONSE_CHARS,
    )
    write_returned = any(
        entry.get("tool_name") == "write_file" and entry.get("status") == "returned"
        for entry in dispatch_trace
    )
    initial_readback_verified = any(
        entry.get("tool_name") == "read_file"
        and entry.get("read_state") == "initial"
        and entry.get("status") == "returned"
        and entry.get("evidence_verified") is True
        for entry in dispatch_trace
    )
    final_readback_verified = any(
        entry.get("tool_name") == "read_file"
        and entry.get("read_state") == "final"
        and entry.get("status") == "returned"
        and entry.get("evidence_verified") is True
        for entry in dispatch_trace
    )
    final_bytes_match = document_after_bytes == fixture["final_bytes"]
    objective_reasons = []
    if not final_bytes_match:
        objective_reasons.append("independent raw-byte readback does not equal the D1 final-text oracle")
    if not expected_unchanged:
        objective_reasons.append("fixture expected.json changed during the run")
    objective_status = "pass" if not objective_reasons else "fail"
    answer_status = "manual_review_required" if final_answer is not None else "not_assessable"

    event_bytes = event_path.read_bytes()
    provider_attempt_summary = f2._provider_attempt_summary(event_bytes, run_id)
    provider_usage_summary = f3._usage_summary(event_bytes, run_id)
    event_digest = hashlib.sha256(event_bytes).hexdigest()
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
        "app": APP_ID,
        "build": _git_commit(),
        "route": route,
        "scenario": SCENARIO,
        "phase": "cold",
        "phase_scope": "fresh headless Brain call; no daemon, microphone, TTS playback, desktop capture, or browser",
        "starting_state": {
            "conversation": "empty",
            "fixture_document_sha256": fixture["document_before"]["sha256"],
            "fixture_document_byte_length": fixture["document_before"]["byte_length"],
            "memory": "disabled",
            "skills": "disabled",
            "custom_tools": "disabled",
            "model_tools": ["read_file", "write_file"],
            "audio": "capture-only TTS; no playback",
            "desktop": "disabled",
            "browser": "disabled",
        },
        "prompt_id": "prompts.document",
        "prompt": fixture["prompt"],
        "oracle_id": "oracles.document; exact independent raw-byte check",
        "outcome": objective_status,
        "oracle": {
            "status": answer_status,
            "answer_review_status": answer_status,
            "objective_evidence_status": objective_status,
            "objective_evidence_reasons": objective_reasons,
            "initial_readback_observed": initial_readback_verified,
            "write_tool_returned": write_returned,
            "final_readback_observed": final_readback_verified,
            "independent_final_bytes_match": final_bytes_match,
            "expected_json_unchanged": expected_unchanged,
            "expected": {
                "initial_text": fixture["initial_text"],
                "final_text": fixture["final_text"],
            },
            "answer_review_reason": (
                "Review the final prose separately; the runner never treats Adam's claim as task-state evidence."
                if final_answer is not None else "Brain did not append a final assistant answer"
            ),
        },
        "brain_execution_status": execution_status,
        "brain_response_text": safe_answer if final_answer is not None else None,
        "brain_response_text_truncated": answer_truncated,
        "normalized_provider_responses": responses,
        "provider_tool_trace": [call for response in responses for call in response.get("tool_calls", [])],
        "dispatch_trace": dispatch_trace,
        "tool_dispatch_implementation": (
            "runner-guarded in-process dirfd handlers; fixture entries opened with O_NOFOLLOW; "
            "normal filesystem handlers are not delegated"
        ),
        "errors": [error_record] if error_record else [],
        "timing_ms": {
            "headless_brain_call": round((end_ns - start_ns) / 1_000_000, 3)
            if start_ns is not None and end_ns is not None else None,
            "request_end_to_verified_state": None,
            "request_end_to_verified_state_missing_reason": (
                "headless D1 does not capture user request-end; independent filesystem verification follows the Brain call"
            ),
            "request_end_to_ack": None,
            "request_end_to_ack_missing_reason": "capture-only TTS does not measure audible playback",
        },
        "event_clock_ns": {
            "headless_brain_call_start": start_ns,
            "headless_brain_call_return": end_ns,
            "independent_byte_verification": verification_ns,
        },
        "document_integrity": {
            "before": fixture["document_before"],
            "after": document_after,
            "matches_final_text": final_bytes_match,
            "after_error": document_after_error,
            "expected_json_unchanged": expected_unchanged,
            "expected_json_after_error": expected_after_error,
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
            "provider_usage": provider_usage_summary,
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
        f2._append_record(record_fd, record)
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
    parser.add_argument("--confirm-provider-inference", action="store_true", help="explicitly authorize one live D1 provider-backed turn")
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
        safe_message, _ = f2._scrub_text(str(exc), [], limit=500)
        print(f"D1 trial refused: {safe_message}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"D1 trial failed ({type(exc).__name__}); inspect the private record if created.", file=sys.stderr)
        return 1
    print(
        f"D1 headless run recorded with objective evidence {record['oracle']['objective_evidence_status']} "
        f"and prose status {record['oracle']['answer_review_status']}. "
        f"Private record: {record['private_output_paths']['record']}"
    )
    return 0 if record["brain_execution_status"] == "returned" else 1


if __name__ == "__main__":
    raise SystemExit(main())
