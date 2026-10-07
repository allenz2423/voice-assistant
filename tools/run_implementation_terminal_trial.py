#!/usr/bin/env python3
"""Run one bounded, headless T1 terminal task through AdamBrain.

This runner exposes only the T1 command and interprets that exact command in
process against the generated fixture. It never sends model text to a shell.
Provider inference requires separate route and inference confirmations.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import stat
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

# Keep absolute-script invocation anchored to this checkout's src package.
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

FIXTURE_VERSION = 4
MAX_TOOL_DISPATCHES = 1
MAX_LOGICAL_CHAT_CALLS = 2
MAX_RESPONSE_CHARS = 16_000
MAX_TOOL_TRACE_CHARS = 8_192
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


class TrialError(ValueError):
    """Raised when the requested T1 run is unsafe or malformed."""


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


class _CapturingClient:
    """Capture normalized responses and reject every schema outside T1."""

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
            raise TrialError("T1 runner stopped after its two logical primary chat-call bound")
        offered = list(tools or [])
        names = [getattr(tool, "name", None) for tool in offered]
        allowed = names == ["run_bash_command"] if not self.requests else names in (["run_bash_command"], [])
        if not allowed:
            raise TrialError("T1 runner refused a tool schema other than its single terminal command schema")
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
                    raw_arguments = arguments[:4_000]
                else:
                    try:
                        raw_arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))[:4_000]
                    except (TypeError, ValueError):
                        raw_arguments = f"<{type(arguments).__name__}>"
                calls.append({
                    "id": str(call.get("id", ""))[:100] if isinstance(call, dict) else None,
                    "name": str(function.get("name", ""))[:100],
                    "arguments": f2._scrub_text(raw_arguments, self.secrets, limit=4_000)[0],
                })
        self.responses.append({
            "role": response.get("role") if isinstance(response.get("role"), str) else None,
            "content": safe_content,
            "content_truncated": truncated,
            "tool_calls": calls,
            "provider_error": response.get("provider_error") is True,
        })
        return response


def _validate_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise TrialError(f"{name} must be 1–80 safe opaque characters [A-Za-z0-9._-]")
    return value


def _read_owned_regular(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TrialError(f"fixture file {path.name} must be an owned regular non-symlink file") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_size > max_bytes
        ):
            raise TrialError(f"fixture file {path.name} must be an owned, bounded single-link regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise TrialError(f"fixture file {path.name} exceeds its size bound")
        return data
    finally:
        os.close(descriptor)


def _snapshot(data: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(data).hexdigest(), "byte_length": len(data)}


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
    answer_path = resolved / "answer.txt"
    expected_bytes = _read_owned_regular(expected_path, max_bytes=128_000)
    source_bytes = _read_owned_regular(source_path, max_bytes=16_384)
    try:
        answer_path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise TrialError("answer.txt must not exist; use a fresh generated T1 fixture")
    try:
        metadata = json.loads(expected_bytes.decode("utf-8"))
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
    terminal_oracle = oracles.get("terminal") if isinstance(oracles, dict) else None
    source_oracle = oracles.get("file_status") if isinstance(oracles, dict) else None
    if (
        not isinstance(paths, dict)
        or paths.get("root") != str(resolved)
        or paths.get("source_file") != str(source_path)
    ):
        raise TrialError("fixture path metadata does not match the generated T1 fixture")
    expected_prompt = (
        f'In "{resolved}", count lines in source.txt beginning with ITEM: and write only the count plus a newline '
        "to answer.txt. Do not change source.txt."
    )
    if not isinstance(prompts, dict) or prompts.get("terminal") != expected_prompt:
        raise TrialError("fixture T1 prompt does not exactly match the generated terminal task")
    if not isinstance(terminal_oracle, dict):
        raise TrialError("fixture terminal oracle is malformed")
    if (
        not isinstance(source_oracle, dict)
        or isinstance(source_oracle.get("byte_length"), bool)
        or not isinstance(source_oracle.get("byte_length"), int)
        or source_oracle["byte_length"] < 1
    ):
        raise TrialError("fixture source byte-length oracle is malformed")
    expected_hash = terminal_oracle.get("source_sha256")
    expected_answer = terminal_oracle.get("answer_text")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise TrialError("fixture source SHA-256 oracle is malformed")
    if not isinstance(expected_answer, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)\n", expected_answer):
        raise TrialError("fixture answer oracle must be a decimal count followed by one newline")
    source_before = _snapshot(source_bytes)
    if source_before["sha256"] != expected_hash:
        raise TrialError("fixture source.txt does not match the expected source SHA-256 oracle")
    if source_before["byte_length"] != source_oracle["byte_length"]:
        raise TrialError("fixture source.txt does not match the expected byte-length oracle")
    actual_count = sum(line.startswith(b"ITEM:") for line in source_bytes.split(b"\n"))
    if expected_answer.encode("ascii") != f"{actual_count}\n".encode("ascii"):
        raise TrialError("fixture terminal answer oracle does not match source.txt")
    return {
        "fixture_id": fixture_id,
        "fixture_dir": resolved,
        "expected_path": expected_path,
        "source_path": source_path,
        "answer_path": answer_path,
        "prompt": expected_prompt,
        "source_before": source_before,
        "source_expected_byte_length": source_oracle["byte_length"],
        "expected_json_sha256_before": hashlib.sha256(expected_bytes).hexdigest(),
        "oracle_answer_bytes": expected_answer.encode("ascii"),
        "line_count": actual_count,
    }


def _expected_command(source_path: Path, answer_path: Path) -> str:
    return f"grep -c '^ITEM:' {shlex.quote(str(source_path))} > {shlex.quote(str(answer_path))}"


def _write_answer(answer_path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(answer_path, flags, 0o600)
    except OSError as exc:
        raise TrialError("runner could not create the fixture answer.txt exclusively") from exc
    try:
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise TrialError("runner-created answer.txt did not remain a private regular file")
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write to fixture answer.txt")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_tool() -> CanonicalTool:
    return CanonicalTool(
        name="run_bash_command",
        description=(
            "T1 synthetic terminal capability. In the generated fixture directory named in the user request, "
            "count lines in source.txt that begin with ITEM: and write only the decimal count followed by one "
            "newline to answer.txt. Use only those two fixture files. The runner accepts only the exact command "
            "for this operation, validates it against the fixture paths, and interprets it in process without "
            "starting a shell."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "maxLength": 2_000, "description": "A terminal command for the requested fixture-only operation."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )


def _build_brain(config, fixture: dict[str, Any], secrets: list[str]):
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
    command = _expected_command(fixture["source_path"], fixture["answer_path"])
    with patch("src.llm.brain.CustomToolManager", return_value=_NoCustomTools()), \
            patch("src.llm.brain.SkillManager", _NoSkills):
        brain = AdamBrain(
            config, supervisor=None, probe=None, confirmation_mgr=None,
            tts_engine=tts, memory_mgr=_NoMemory(),
        )
    brain.get_tools = lambda: [_make_tool()]
    brain.tool_free_llm_client = None
    brain.max_tool_rounds = MAX_LOGICAL_CHAT_CALLS
    brain.llm_client = _CapturingClient(brain.llm_client, secrets)
    dispatch_trace: list[dict[str, Any]] = []
    dispatch_lock = asyncio.Lock()

    async def restricted_execute_locked(name: str, args: dict) -> str:
        entry: dict[str, Any] = {
            "tool_name": str(name)[:100],
            "arguments": f2._private_value(args, secrets, string_limit=2_000),
        }
        dispatch_trace.append(entry)
        if len(dispatch_trace) > MAX_TOOL_DISPATCHES:
            entry["status"] = "refused"
            entry["reason"] = "dispatch limit reached"
            raise TrialError("T1 runner permits only one terminal dispatch")
        if name != "run_bash_command":
            entry["status"] = "refused"
            entry["reason"] = "only run_bash_command is permitted"
            raise TrialError("T1 runner permits only run_bash_command")
        if not isinstance(args, dict) or set(args) != {"command"}:
            entry["status"] = "refused"
            entry["reason"] = "arguments must contain only command"
            raise TrialError("T1 runner refused unexpected terminal arguments")
        if not isinstance(args.get("command"), str) or args["command"] != command:
            entry["status"] = "refused"
            entry["reason"] = "command did not exactly match the generated fixture operation"
            raise TrialError("T1 runner refused a command outside the exact fixture operation")
        source_now = _read_owned_regular(fixture["source_path"], max_bytes=16_384)
        if _snapshot(source_now) != fixture["source_before"]:
            entry["status"] = "refused"
            entry["reason"] = "source.txt changed after fixture validation"
            raise TrialError("T1 runner refused a changed source.txt")
        try:
            fixture["answer_path"].lstat()
        except FileNotFoundError:
            pass
        else:
            entry["status"] = "refused"
            entry["reason"] = "answer.txt appeared after fixture validation"
            raise TrialError("T1 runner refused to replace an existing answer.txt")
        entry["status"] = "dispatched"
        entry["interpreted_in_process"] = True
        entry["shell_started"] = False
        try:
            count = sum(line.startswith(b"ITEM:") for line in source_now.split(b"\n"))
            result_bytes = f"{count}\n".encode("ascii")
            _write_answer(fixture["answer_path"], result_bytes)
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f2._safe_error(exc, secrets)
            raise
        result_text = result_bytes.decode("ascii")
        entry["status"] = "returned"
        entry["result"] = result_text
        entry["answer_bytes_written"] = len(result_bytes)
        entry["answer_sha256"] = hashlib.sha256(result_bytes).hexdigest()
        entry["source_sha256_at_dispatch"] = hashlib.sha256(source_now).hexdigest()
        return result_text

    async def restricted_execute(name: str, args: dict) -> str:
        async with dispatch_lock:
            return await restricted_execute_locked(name, args)

    brain._t1_dispatch_trace = dispatch_trace
    brain._execute_tool = restricted_execute
    return brain, tts


def _private_record_fd(path: Path, run_id: str) -> int:
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
        with path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrialError(f"existing output JSONL is invalid on line {number}") from exc
                if not isinstance(record, dict) or record.get("app") != "AdamBrain headless T1 terminal run":
                    raise TrialError(f"existing output JSONL line {number} is not a T1 run record")
                if record.get("schema_version") != 1:
                    raise TrialError(f"existing output JSONL line {number} has an unsupported schema")
                if record.get("run_id") == run_id:
                    raise TrialError(f"run ID {run_id!r} already exists in output JSONL")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _restore_telemetry_writer(installed_writer, previous_writer) -> None:
    from src.telemetry import events as telemetry_events

    if installed_writer is None:
        return
    with telemetry_events._writer_lock:
        if telemetry_events._writer is installed_writer:
            telemetry_events._writer = previous_writer


def _git_commit() -> str | None:
    return f2._git_commit()


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
    secrets = f2._secret_values(config)
    for label, value in (
        ("fixture run ID", fixture["fixture_id"]), ("opaque run ID", run_id),
        ("config ID", config_id), ("model ID", route["model"]),
        *[("provider route", item) for item in route["provider_only"]],
    ):
        if any(len(secret) >= 4 and secret.casefold() in value.casefold() for secret in secrets):
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
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        installed_writer = configure_telemetry(SimpleNamespace(enabled=True, path=str(event_path)))
        trace_token = set_trace_id(run_id)
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture), \
                patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
            brain, _tts = brain_builder(config, fixture, secrets)
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
        error_record = f2._safe_error(exc, secrets)
    finally:
        if brain is not None:
            try:
                brain.close()
            except Exception as exc:
                error_record = error_record or f2._safe_error(exc, secrets)
        try:
            if trace_token is not None:
                reset_trace_id(trace_token)
        finally:
            _restore_telemetry_writer(installed_writer, previous_writer)

    try:
        source_after_bytes = _read_owned_regular(fixture["source_path"], max_bytes=16_384)
        source_after = _snapshot(source_after_bytes)
        source_error = None
    except Exception as exc:
        source_after = None
        source_error = f2._safe_error(exc, secrets)
    try:
        expected_after_bytes = _read_owned_regular(fixture["expected_path"], max_bytes=128_000)
        expected_after_hash = hashlib.sha256(expected_after_bytes).hexdigest()
        fixture_metadata_unchanged = expected_after_hash == fixture["expected_json_sha256_before"]
        metadata_error = None
    except Exception as exc:
        expected_after_hash = None
        fixture_metadata_unchanged = False
        metadata_error = f2._safe_error(exc, secrets)
    try:
        answer_after_bytes = _read_owned_regular(fixture["answer_path"], max_bytes=1_024)
        answer_error = None
    except Exception as exc:
        answer_after_bytes = None
        answer_error = f2._safe_error(exc, secrets)
    answer_matches = answer_after_bytes == fixture["oracle_answer_bytes"]
    source_unchanged = source_after == fixture["source_before"]
    dispatch_trace = getattr(brain, "_t1_dispatch_trace", []) if brain is not None else []
    terminal_returned = any(
        entry.get("tool_name") == "run_bash_command" and entry.get("status") == "returned"
        for entry in dispatch_trace
    )
    final_answer = None
    if brain is not None:
        final_answer = next(
            (str(item.get("content") or "") for item in reversed(brain.messages)
             if item.get("role") == "assistant"),
            None,
        )
    safe_answer, answer_truncated = f2._scrub_text(final_answer or "", secrets, limit=MAX_RESPONSE_CHARS)
    objective_reasons = []
    if not terminal_returned:
        objective_reasons.append("no permitted terminal operation returned successfully")
    if len(dispatch_trace) != 1:
        objective_reasons.append("the model attempted more than one terminal dispatch")
    if not answer_matches:
        objective_reasons.append("answer.txt bytes did not exactly match the generated answer oracle")
    if not source_unchanged:
        objective_reasons.append("source.txt SHA-256 or byte length changed")
    if not fixture_metadata_unchanged:
        objective_reasons.append("expected.json changed after fixture validation")
    objective_status = "pass" if not objective_reasons else "fail"
    answer_review_status = "manual_review_required" if final_answer is not None else "not_assessable"
    event_bytes = event_path.read_bytes()
    event_digest = hashlib.sha256(event_bytes).hexdigest()
    attempt_summary = f2._provider_attempt_summary(event_bytes, run_id)
    responses = client_capture.responses if isinstance(client_capture, _CapturingClient) else []
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "trace_id": run_id,
        "fixture_id": fixture["fixture_id"],
        "commit": _git_commit(),
        "host": platform.node() or None,
        "os": platform.platform(),
        "app": "AdamBrain headless T1 terminal run",
        "build": _git_commit(),
        "route": route,
        "scenario": "T1",
        "phase": "cold",
        "phase_scope": "fresh headless Brain call; exact fixture command interpreted in process; no daemon, microphone, TTS playback, desktop, or browser",
        "starting_state": {
            "conversation": "empty",
            "memory": "disabled",
            "skills": "disabled",
            "custom_tools": "disabled",
            "model_tools": ["run_bash_command"],
            "audio": "capture-only TTS; no playback",
            "desktop": "disabled",
            "browser": "disabled",
        },
        "prompt_id": "prompts.terminal",
        "prompt": fixture["prompt"],
        "oracle_id": "oracles.terminal.answer_text exact bytes plus source SHA-256/length immutability",
        "oracle": {
            "status": objective_status,
            "objective_evidence_status": objective_status,
            "objective_evidence_reasons": objective_reasons,
            "answer_review_status": answer_review_status,
            "terminal_dispatch_returned": terminal_returned,
            "answer_file_matches_exact_bytes": answer_matches,
            "source_unchanged": source_unchanged,
            "fixture_metadata_unchanged": fixture_metadata_unchanged,
            "expected_answer_byte_length": len(fixture["oracle_answer_bytes"]),
            "expected_line_count": fixture["line_count"],
            "expected_source_byte_length": fixture["source_expected_byte_length"],
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
            "request_end_to_verified_state_missing_reason": "headless T1 capture does not measure user request end",
            "request_end_to_ack": None,
            "request_end_to_ack_missing_reason": "capture-only TTS does not measure audible playback",
        },
        "event_clock_ns": {"headless_brain_call_start": start_ns, "headless_brain_call_return": end_ns},
        "source_integrity": {"before": fixture["source_before"], "after": source_after, "unchanged": source_unchanged, "after_error": source_error},
        "answer_integrity": {
            "path": str(fixture["answer_path"]),
            "sha256": hashlib.sha256(answer_after_bytes).hexdigest() if answer_after_bytes is not None else None,
            "byte_length": len(answer_after_bytes) if answer_after_bytes is not None else None,
            "matches_expected_bytes": answer_matches,
            "read_error": answer_error,
        },
        "fixture_integrity": {
            "expected_json_sha256_before": fixture["expected_json_sha256_before"],
            "expected_json_sha256_after": expected_after_hash,
            "metadata_unchanged": fixture_metadata_unchanged,
            "after_error": metadata_error,
        },
        "telemetry": {
            "event_path": str(event_path),
            "event_file_sha256": event_digest,
            "matching_trace_id": run_id,
            "logical_primary_client_chat_calls": len(client_capture.requests) if isinstance(client_capture, _CapturingClient) else 0,
            "logical_primary_client_chat_call_limit": MAX_LOGICAL_CHAT_CALLS,
            **attempt_summary,
            "attempt_scope_note": "HTTP attempts and retry events are Adam client telemetry; provider-side routing or retries are not observable.",
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
    parser.add_argument("--fixture-dir", required=True, type=Path, help="absolute owner-private generated T1 fixture directory")
    parser.add_argument("--config", required=True, type=Path, help="existing protected Adam config; read-only")
    parser.add_argument("--run-id", required=True, help="unique opaque ID; also used as telemetry trace ID")
    parser.add_argument("--config-id", required=True, help="non-secret stable label for this route/config")
    parser.add_argument("--output", required=True, type=Path, help="private .jsonl series file under an existing mode-0700 directory")
    parser.add_argument("--expected-model", required=True, help="exact configured model ID to run and attest")
    parser.add_argument("--metadata-http-status", required=True, type=int, help="operator-reported external metadata-only status; must be 200")
    parser.add_argument("--metadata-model-id", required=True, help="exact model ID from the external metadata-only check")
    parser.add_argument("--metadata-endpoint-count", required=True, type=int, help="positive endpoint count from external metadata")
    parser.add_argument("--metadata-checked-at", required=True, help="external check time as ISO 8601 UTC ending in Z")
    parser.add_argument("--route-ready-confirmed", action="store_true", help="confirm that a fresh external metadata-only check was performed")
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
        print(f"T1 trial refused: {exc}", file=sys.stderr)
        return 2
    except OSError:
        print("T1 trial could not read or write a private artifact (OSError).", file=sys.stderr)
        return 2
    except (TypeError, ValueError) as exc:
        print(f"T1 trial could not finalize its record ({type(exc).__name__}).", file=sys.stderr)
        return 2
    print(json.dumps({
        "run_id": record["run_id"],
        "fixture_id": record["fixture_id"],
        "scenario": record["scenario"],
        "outcome": record["oracle"]["status"],
        "brain_execution_status": record["brain_execution_status"],
        "logical_primary_client_chat_calls": record["telemetry"]["logical_primary_client_chat_calls"],
        "private_output_paths": record["private_output_paths"],
    }, ensure_ascii=False))
    return 0 if record["brain_execution_status"] == "returned" else 1


if __name__ == "__main__":
    raise SystemExit(main())
