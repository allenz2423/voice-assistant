#!/usr/bin/env python3
"""Run one provider-backed, headless M1 memory-recall turn through AdamBrain.

Inference requires both a fresh operator-reported route attestation and a
separate explicit confirmation. The runner seeds a private temporary store
with only the generated M1 records, disables every tool and optional input
surface, captures TTS text, and removes the temporary store after the turn.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Keep imports working when called by absolute script path from outside the repo.
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

from src.memory.temporal import local_timezone_name
from src.telemetry.events import configure_telemetry, reset_trace_id, set_trace_id
from tools import run_implementation_factual_trial as _f1

EXPECTED_PROVIDER = _f1.EXPECTED_PROVIDER
EXPECTED_API_BASE = _f1.EXPECTED_API_BASE
FIXTURE_VERSION = 4
PROMPT_IDS = ("day", "week", "year", "timezone")
MAX_RESPONSE_CHARS = 32_000
MAX_METADATA_AGE = timedelta(hours=1)
MAX_FIXTURE_AGE = timedelta(hours=24)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
_SAFE_MODEL_ID = _f1._SAFE_MODEL_ID
_UTC_TIMESTAMP = _f1._UTC_TIMESTAMP


class TrialError(ValueError):
    """Raised when an M1 trial input or boundary is invalid."""


class _UnexpectedToolCall(RuntimeError):
    """The model requested a tool in a deliberately tool-less M1 turn."""


class _CapturingMemoryOnlyClient:
    """Capture normalized output and stop before any model tool can dispatch."""

    def __init__(self, client: Any, secrets: list[str]) -> None:
        self._client = client
        self._secrets = secrets
        self.provider = getattr(client, "provider", None)
        self.responses: list[dict[str, Any]] = []
        self.chat_call_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def chat(self, messages: list[dict], tools=None, max_tokens=None, think=None) -> dict:
        if tools:
            raise TrialError("M1 runner refused a non-empty tool schema")
        if self.chat_call_count >= 1:
            raise TrialError("M1 runner permits one primary-client chat call")
        self.chat_call_count += 1
        response = await self._client.chat(
            messages, tools=[], max_tokens=max_tokens, think=think,
        )
        normalized = _f1._safe_normalized_response(response)
        self.responses.append(_scrub_value(normalized, self._secrets))
        tool_calls = response.get("tool_calls") if isinstance(response, dict) else None
        if tool_calls:
            names = normalized.get("tool_names") or []
            raise _UnexpectedToolCall(
                "model returned a tool call during the tool-less M1 trial"
                + (f" (names: {', '.join(names)})" if names else "")
            )
        return response


def _scrub_text(value: str, secrets: list[str], *, limit: int = MAX_RESPONSE_CHARS) -> tuple[str, bool]:
    """Redact configured credentials without shortening ordinary answers."""
    safe = str(value)
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, "[REDACTED]")
    safe = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", safe)
    safe = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s&,;]+", r"\1[REDACTED]", safe)
    safe = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", safe)
    safe = re.sub(r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key)=)[^&#\s]+", r"\1[REDACTED]", safe)
    truncated = len(safe) > limit
    return safe[:limit], truncated


def _scrub_value(value: Any, secrets: list[str]) -> Any:
    """Redact configured credentials from any string serialized by this CLI."""
    if isinstance(value, str):
        return _scrub_text(value, secrets)[0]
    if isinstance(value, list):
        return [_scrub_value(item, secrets) for item in value]
    if isinstance(value, dict):
        return {str(key): _scrub_value(item, secrets) for key, item in value.items()}
    return value


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


def _expected_memory_spec(fixture_dir: Path) -> dict[str, Any]:
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

    raw_expected = _read_owned_regular(resolved / "expected.json", max_bytes=128_000)
    try:
        metadata = json.loads(raw_expected.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrialError("fixture expected.json is not valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("fixture_version") != FIXTURE_VERSION:
        raise TrialError("fixture must use the generated implementation fixture format")
    try:
        fixture_id = _f1._validate_id(metadata.get("run_id"), "fixture run ID")
    except _f1.TrialError as exc:
        raise TrialError(str(exc)) from exc
    if fixture_id != resolved.name:
        raise TrialError("fixture run ID must match its generated directory name")
    paths = metadata.get("paths")
    if not isinstance(paths, dict) or paths.get("root") != str(resolved):
        raise TrialError("fixture root metadata does not match --fixture-dir")

    timezone_name = metadata.get("timezone")
    try:
        zone = ZoneInfo(timezone_name) if isinstance(timezone_name, str) else None
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise TrialError("fixture must name a valid IANA timezone") from exc
    if zone is None:
        raise TrialError("fixture must name a valid IANA timezone")
    expected_local_zone = local_timezone_name() or "UTC"
    if timezone_name != expected_local_zone:
        raise TrialError("fixture timezone must match the current local timezone used by MemoryManager")

    created_value = metadata.get("created_at_local")
    try:
        created_at = datetime.fromisoformat(created_value) if isinstance(created_value, str) else None
    except ValueError as exc:
        raise TrialError("fixture creation time must be a local ISO 8601 timestamp") from exc
    if created_at is None or created_at.tzinfo is None or created_at.utcoffset() is None:
        raise TrialError("fixture creation time must include its local UTC offset")
    oracles = metadata.get("oracles")
    memory = oracles.get("memory") if isinstance(oracles, dict) else None
    if not isinstance(memory, dict):
        raise TrialError("fixture is missing its generated M1 memory oracle")
    run_date_value = memory.get("run_date")
    try:
        run_date = date.fromisoformat(run_date_value) if isinstance(run_date_value, str) else None
    except ValueError as exc:
        raise TrialError("fixture memory run_date must be an ISO date") from exc
    if run_date is None or created_at.date() != run_date:
        raise TrialError("fixture memory date must match its creation date")
    now = datetime.now(zone)
    age = now.astimezone(timezone.utc) - created_at.astimezone(timezone.utc)
    if now.date() != run_date or age < -timedelta(minutes=5) or age > MAX_FIXTURE_AGE:
        raise TrialError("M1 requires a fresh fixture generated today in the current local timezone")

    # Verify the generated M1 data and all four prompts against the checked-in
    # fixture generator's deterministic date layout. This prevents a modified
    # expected.json from silently changing what the runner seeds or asks.
    previous_week_monday = run_date - timedelta(days=run_date.weekday() + 7)
    first_date = previous_week_monday + timedelta(days=1)
    second_date = previous_week_monday + timedelta(days=4)
    prior_year_date = date(run_date.year - 1, 1, 15)
    if first_date.year != run_date.year or second_date.year != run_date.year:
        raise TrialError("M1 fixture dates cross a year boundary and cannot isolate the last-year oracle")
    if memory.get("timezone") != timezone_name or memory.get("run_date") != run_date.isoformat():
        raise TrialError("fixture memory timezone or run date is malformed")
    records = [
        f"I worked on the Juniper migration on {first_date.isoformat()} from 9:30 to 10:15am.",
        f"I worked on the Juniper invoice export on {second_date.isoformat()} from 3:00 to 4:00pm.",
        f"I worked on the Juniper migration on {prior_year_date.isoformat()} from 11:00am to noon.",
    ]
    record_dates = [first_date.isoformat(), second_date.isoformat(), prior_year_date.isoformat()]
    prompts = {
        "day": f"What did I work on {first_date.isoformat()}, and what local time?",
        "week": "What Juniper work did I log last week?",
        "year": "What Juniper work did I log last year?",
        "timezone": f"What timezone did I use for the {first_date.isoformat()} work entry?",
    }
    if memory.get("records") != records or memory.get("expected_record_dates") != record_dates:
        raise TrialError("fixture memory records do not match the generated M1 oracle")
    if memory.get("prompts") != prompts:
        raise TrialError("fixture memory prompts do not match the generated M1 prompts")
    top_prompts = metadata.get("prompts")
    if not isinstance(top_prompts, dict) or top_prompts.get("memory") != prompts:
        raise TrialError("fixture memory prompts do not match the generated prompt list")

    return {
        "fixture_id": fixture_id,
        "fixture_dir": resolved,
        "fixture_created_at_local": created_at.isoformat(),
        "run_date": run_date.isoformat(),
        "timezone": timezone_name,
        "records": records,
        "expected_record_dates": record_dates,
        "prompts": prompts,
    }


def _validate_prompt_id(prompt_id: str, fixture: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if prompt_id not in PROMPT_IDS:
        raise TrialError("--prompt-id must be one of: day, week, year, timezone")
    prompts = fixture["prompts"]
    criteria: dict[str, Any] = {
        "day": {
            "must_report_date": fixture["expected_record_dates"][0],
            "must_report_local_interval": "9:30–10:15 a.m.",
        },
        "week": {
            "must_recall_records": fixture["records"][:2],
            "must_exclude_records": [fixture["records"][2]],
        },
        "year": {
            "must_recall_records": [fixture["records"][2]],
            "must_exclude_records": fixture["records"][:2],
        },
        "timezone": {"must_name_iana_timezone": fixture["timezone"]},
    }
    return f"prompts.memory.{prompt_id}", prompts[prompt_id], criteria[prompt_id]


def _metadata_attestation(**kwargs) -> dict[str, Any]:
    try:
        attestation = _f1._validate_metadata_attestation(**kwargs)
    except _f1.TrialError as exc:
        raise TrialError(str(exc)) from exc
    attestation["source"] = "operator-reported external metadata-only check"
    attestation["runner_fetched_or_cryptographically_verified"] = False
    return attestation


def _load_config(config_path: Path):
    try:
        return _f1._load_existing_config(config_path.expanduser())
    except _f1.TrialError as exc:
        raise TrialError(str(exc)) from exc


def _validate_route(config, *, config_id: str, expected_model: str) -> dict[str, Any]:
    try:
        return _f1._validate_config(config, config_id=config_id, expected_model=expected_model)
    except _f1.TrialError as exc:
        raise TrialError(str(exc)) from exc


def _private_output_paths(output: Path, run_id: str, fixture_dir: Path) -> tuple[Path, Path]:
    try:
        output_path, event_path = _f1._private_output_paths(output, run_id)
    except _f1.TrialError as exc:
        raise TrialError(str(exc)) from exc
    for candidate in (output_path, event_path):
        resolved = candidate.resolve(strict=False)
        if resolved == fixture_dir or fixture_dir in resolved.parents:
            raise TrialError("run outputs must be outside the immutable fixture directory")
    return output_path, event_path


def _open_private_jsonl(path: Path, *, run_id: str) -> int:
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
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
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        read_descriptor = os.open(path, read_flags)
        try:
            with os.fdopen(read_descriptor, "r", encoding="utf-8", closefd=False) as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise TrialError(f"existing output JSONL has invalid JSON on line {line_number}") from exc
                    if not isinstance(row, dict) or row.get("schema_version") != 1 or row.get("app") != "AdamBrain headless M1 memory trial":
                        raise TrialError(f"existing output JSONL line {line_number} is not an M1 run record")
                    if row.get("scenario") != "M1":
                        raise TrialError(f"existing output JSONL line {line_number} is not an M1 run record")
                    if row.get("run_id") == run_id:
                        raise TrialError(f"run ID {run_id!r} already exists in output JSONL")
        finally:
            os.close(read_descriptor)
        return descriptor
    except TrialError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise TrialError(f"could not safely open private run JSONL ({type(exc).__name__})") from exc


def _build_brain(config, memory_mgr, secrets: list[str]):
    from src.llm.brain import AdamBrain

    # Preserve the user's config file; limit only the in-memory trial copy.
    config = config.model_copy(deep=True)
    config.llm.tool_free_model = ""
    config.llm.tool_free_provider_only = []
    config.llm.tool_free_allow_provider_fallbacks = False
    if config.computer_control is not None:
        config.computer_control.enabled = False
    if config.computer_vision is not None:
        config.computer_vision.enabled = False
    if config.browser_navigation is not None:
        config.browser_navigation.enabled = False

    tts = _f1._CaptureOnlyTTS()
    with patch("src.llm.brain.CustomToolManager", return_value=_f1._NoCustomTools()), \
            patch("src.llm.brain.SkillManager", _f1._NoSkills):
        brain = AdamBrain(
            config, supervisor=None, probe=None, confirmation_mgr=None,
            tts_engine=tts, memory_mgr=memory_mgr,
        )
    brain.get_tools = lambda: []
    brain.tool_free_llm_client = None
    brain.llm_client = _CapturingMemoryOnlyClient(brain.llm_client, secrets)
    return brain, tts


def _seed_memory(fixture: dict[str, Any], storage_path: Path):
    from src.memory.embedder import MemoryEmbedder
    from src.memory.manager import MemoryManager

    if storage_path.exists() or storage_path.is_symlink():
        raise TrialError("private per-run memory store path unexpectedly already exists")
    if not storage_path.parent.is_dir() or any(storage_path.parent.iterdir()):
        raise TrialError("private per-run memory directory must be new and empty")
    manager = MemoryManager(
        storage_path=storage_path,
        embedder=MemoryEmbedder(disabled=True),
    )
    if manager.storage_path != storage_path:
        raise TrialError("MemoryManager did not retain the explicit per-run storage path")
    for text in fixture["records"]:
        manager.save(text)
    saved_texts = [record.text for record in manager._memories]
    if saved_texts != fixture["records"]:
        raise TrialError("isolated memory store did not contain exactly the fixture records")
    for index, record in enumerate(manager._memories):
        if record.event_date_start != fixture["expected_record_dates"][index]:
            raise TrialError("MemoryManager did not preserve a fixture record date")
        if record.event_timezone != fixture["timezone"]:
            raise TrialError("MemoryManager did not preserve the fixture IANA timezone")
    if manager._memories[0].event_start_at is None or manager._memories[0].event_end_at is None:
        raise TrialError("MemoryManager did not resolve the fixture's local work interval")
    if "T09:30:00" not in manager._memories[0].event_start_at or "T10:15:00" not in manager._memories[0].event_end_at:
        raise TrialError("MemoryManager changed the fixture's 9:30–10:15 a.m. interval")
    try:
        persisted = json.loads(storage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrialError("isolated memory store was not persisted as valid JSON") from exc
    persisted_texts = [item.get("text") for item in persisted.get("memories", []) if isinstance(item, dict)]
    if persisted_texts != fixture["records"]:
        raise TrialError("persisted private memory store does not contain only the fixture records")
    return manager


def _git_commit() -> str | None:
    return _f1._git_commit()


def run_trial(
    *,
    fixture_dir: Path,
    prompt_id: str,
    config_path: Path,
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
) -> dict[str, Any]:
    try:
        run_id = _f1._validate_id(run_id, "opaque run ID")
        config_id = _f1._validate_id(config_id, "config ID")
    except _f1.TrialError as exc:
        raise TrialError(str(exc)) from exc
    if not route_ready_confirmed:
        raise TrialError("refusing inference until an operator confirms the supplied route metadata attestation")
    if not confirm_provider_inference:
        raise TrialError("refusing live provider call without --confirm-provider-inference")

    fixture = _expected_memory_spec(fixture_dir)
    prompt_key, prompt, prompt_oracle = _validate_prompt_id(prompt_id, fixture)
    metadata_attestation = _metadata_attestation(
        expected_model=expected_model,
        metadata_http_status=metadata_http_status,
        metadata_model_id=metadata_model_id,
        metadata_endpoint_count=metadata_endpoint_count,
        metadata_checked_at=metadata_checked_at,
    )
    config = _load_config(config_path)
    route = _validate_route(config, config_id=config_id, expected_model=expected_model)
    if metadata_attestation["model_id"] != route["model"]:
        raise TrialError("metadata model ID must exactly match the configured model")
    route["metadata_attestation"] = metadata_attestation
    secrets = _f1._config_secret_values(config)
    serialized_identity = [
        ("fixture run ID", fixture["fixture_id"]), ("opaque run ID", run_id),
        ("config ID", config_id), ("provider", route["provider"]),
        ("model", route["model"]), ("fixture timezone", fixture["timezone"]),
        ("output path", str(output_path)),
        *[("provider route", item) for item in route["provider_only"]],
    ]
    for label, value in serialized_identity:
        if any(len(secret) >= 8 and secret.casefold() in value.casefold() for secret in secrets):
            raise TrialError(f"{label} must not contain a configured credential")

    output_path, event_path = _private_output_paths(output_path, run_id, fixture["fixture_dir"])
    config_realpath = config_path.expanduser().resolve(strict=True)
    if output_path.resolve(strict=False) == config_realpath or event_path.resolve(strict=False) == config_realpath:
        raise TrialError("run output may not replace or append to the config file")
    record_fd = _open_private_jsonl(output_path, run_id=run_id)
    try:
        event_fd = _f1._create_private_file(event_path)
    except Exception:
        os.close(record_fd)
        raise
    os.close(event_fd)

    configure_telemetry(SimpleNamespace(enabled=True, path=str(event_path)))
    trace_token = set_trace_id(run_id)
    brain = None
    tts = None
    provider_capture = None
    memory_manager = None
    memory_dir: Path | None = None
    memory_path: Path | None = None
    retrievals: list[dict[str, Any]] = []
    retrieval_original = None
    output_capture = io.StringIO()
    error_record = None
    brain_start_ns = None
    brain_end_ns = None
    execution_status = "error"
    store_integrity: dict[str, Any] = {
        "explicit_per_run_path": True,
        "preexisting_store_path": False,
        "seeded_record_count": 0,
        "seeded_texts_match_fixture": False,
        "post_turn_texts_match_fixture": None,
        "default_memory_store_touched": False,
        "cleanup_status": "not_started",
    }
    try:
        with redirect_stdout(output_capture), redirect_stderr(output_capture), \
                patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
            memory_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.memory-", dir=output_path.parent))
            os.chmod(memory_dir, 0o700)
            memory_path = memory_dir / "memory.json"
            memory_manager = _seed_memory(fixture, memory_path)
            store_integrity["seeded_record_count"] = len(memory_manager._memories)
            store_integrity["seeded_texts_match_fixture"] = [r.text for r in memory_manager._memories] == fixture["records"]

            retrieval_original = memory_manager.retrieve_context

            def capture_retrieval(utterance: str, limit: int = 3):
                context = retrieval_original(utterance, limit=limit)
                retrievals.append({"query": str(utterance), "context": context})
                return context

            memory_manager.retrieve_context = capture_retrieval
            brain, tts = _build_brain(config, memory_manager, secrets)
            if brain.memory_mgr is not memory_manager:
                raise TrialError("Brain did not use the isolated per-run memory store")
            provider_capture = brain.llm_client

            async def measured_brain_turn() -> None:
                nonlocal brain_start_ns, brain_end_ns
                brain_start_ns = time.monotonic_ns()
                try:
                    await brain.process_user_utterance(prompt)
                finally:
                    brain_end_ns = time.monotonic_ns()

            asyncio.run(measured_brain_turn())
            execution_status = "returned"
    except Exception as exc:
        error_record = _f1._safe_error(exc, secrets)
    finally:
        if memory_manager is not None:
            store_integrity["post_turn_texts_match_fixture"] = (
                [r.text for r in memory_manager._memories] == fixture["records"]
            )
        if brain is not None:
            try:
                brain.close()
            except Exception as exc:
                error_record = error_record or _f1._safe_error(exc, secrets)
        if memory_dir is not None:
            try:
                shutil.rmtree(memory_dir)
                store_integrity["cleanup_status"] = "removed"
            except OSError as exc:
                store_integrity["cleanup_status"] = "failed"
                error_record = error_record or _f1._safe_error(exc, secrets)
        store_integrity["store_path_absent_after_cleanup"] = bool(
            memory_dir is None or not memory_dir.exists()
        )
        reset_trace_id(trace_token)
        configure_telemetry(SimpleNamespace(enabled=False, path=""))

    events, event_parse_errors = _f1._read_event_log(event_path)
    provider_events = _f1._provider_event_summary(events, run_id)
    normalized_responses = (
        list(provider_capture.responses)
        if isinstance(provider_capture, _CapturingMemoryOnlyClient)
        else []
    )
    final_message = None
    final_message_truncated = False
    if brain is not None:
        final_message_raw = next(
            (str(item.get("content") or "") for item in reversed(brain.messages)
             if item.get("role") == "assistant"),
            None,
        )
        if final_message_raw is not None:
            final_message, final_message_truncated = _scrub_text(final_message_raw, secrets)
    retrievals_safe = _scrub_value(retrievals, secrets)
    tts_ns = tts.submitted_at_ns[-1] if tts and tts.submitted_at_ns else None
    tts_text = _scrub_text(tts.spoken[-1], secrets)[0] if tts and tts.spoken else None
    cleanup_ok = store_integrity["cleanup_status"] == "removed" and store_integrity["store_path_absent_after_cleanup"]
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "trace_id": run_id,
        "fixture_id": fixture["fixture_id"],
        "fixture_version": FIXTURE_VERSION,
        "fixture_created_at_local": fixture["fixture_created_at_local"],
        "commit": _git_commit(),
        "host": platform.node() or None,
        "os": platform.platform(),
        "app": "AdamBrain headless M1 memory trial",
        "build": _git_commit(),
        "route": route,
        "scenario": "M1",
        "phase": "cold",
        "phase_scope": "fresh headless Brain per prompt; no daemon, microphone, TTS playback, or desktop",
        "starting_state": {
            "conversation": "empty",
            "memory": "new private store seeded only from generated M1 records; embeddings disabled",
            "skills": "disabled",
            "custom_tools": "disabled",
            "model_tools": "disabled",
            "tool_free_client": "disabled",
            "audio": "capture-only TTS; no playback",
            "desktop": "disabled",
            "browser": "disabled",
        },
        "prompt_id": prompt_key,
        "prompt": prompt,
        "oracle": {
            "oracle_id": "M1 generated memory recall criteria; manual review required",
            "status": "manual_review_required",
            "criteria": prompt_oracle,
            "fixture_timezone": fixture["timezone"],
            "expected_record_dates": fixture["expected_record_dates"],
            "fixture_records": fixture["records"],
            "automatically_scored": False,
        },
        "outcome": "not_scored",
        "oracle_status": "manual_review_required",
        "brain_execution_status": execution_status,
        "brain_response_text": final_message,
        "brain_response_text_truncated": final_message_truncated,
        "brain_response_text_missing_reason": None if final_message is not None else "Brain did not append a final assistant message",
        "capture_only_tts_text": tts_text,
        "capture_only_tts_text_missing_reason": None if tts_text is not None else "Brain did not submit text to capture-only TTS",
        "normalized_provider_responses": normalized_responses,
        "normalized_provider_responses_missing_reason": None if normalized_responses else "provider client did not return a normalized response",
        "unexpected_tool_call_stopped": any(response.get("tool_call_count", 0) > 0 for response in normalized_responses),
        "primary_client_chat_call_count": provider_capture.chat_call_count if isinstance(provider_capture, _CapturingMemoryOnlyClient) else 0,
        "primary_client_chat_call_limit": 1,
        "memory_retrieval": {
            "observations": retrievals_safe,
            "call_count": len(retrievals_safe),
            "expected_prompt_only": all(item.get("query") == prompt for item in retrievals_safe),
            "independent_answer_review_required": True,
        },
        "memory_store_integrity": store_integrity,
        "errors": [error_record] if error_record else [],
        "timing_ms": {
            "headless_brain_call": round((brain_end_ns - brain_start_ns) / 1_000_000, 3)
            if brain_start_ns is not None and brain_end_ns is not None else None,
            "headless_brain_call_missing_reason": None if brain_start_ns is not None and brain_end_ns is not None else "Brain turn did not reach a measured call boundary",
            "brain_final_text_submission": round((tts_ns - brain_start_ns) / 1_000_000, 3)
            if tts_ns is not None and brain_start_ns is not None else None,
            "request_end_to_verified_state": None,
            "request_end_to_verified_state_missing_reason": "M1 response requires independent review; this CLI has no user request-end or audible timestamp",
            "request_end_to_ack": None,
            "request_end_to_ack_missing_reason": "headless capture-only TTS does not measure audible playback",
        },
        "event_clock_ns": {
            "headless_brain_call_start": brain_start_ns,
            "headless_brain_call_return": brain_end_ns,
            "verified_completion": None,
        },
        "provider": provider_events,
        "telemetry": {
            "event_path": str(event_path),
            "event_file_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
            "matching_trace_id": run_id,
            "event_parse_errors": event_parse_errors,
            "attempt_scope_note": "Provider call/retry counts come from Adam's client telemetry; provider-side routing or retries beyond this client are not observable.",
        },
        "provider_raw_http_body_available": False,
        "provider_raw_http_body_missing_reason": "UniversalLLMClient exposes normalized responses and allowlisted telemetry, not the raw HTTP body",
        "private_output_paths": {"record": str(output_path), "events": str(event_path)},
    }
    try:
        _f1._append_jsonl_fd(record_fd, record)
    finally:
        os.close(record_fd)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path, help="absolute fresh owner-private fixture from create_implementation_fixtures.py")
    parser.add_argument("--prompt-id", required=True, choices=PROMPT_IDS, help="one generated M1 prompt: day, week, year, or timezone")
    parser.add_argument("--config", required=True, type=Path, help="existing protected Adam config; read-only")
    parser.add_argument("--run-id", required=True, help="unique opaque ID; also used as the telemetry trace ID")
    parser.add_argument("--config-id", required=True, help="non-secret stable label for this route/config")
    parser.add_argument("--output", required=True, type=Path, help="append-only .jsonl file in an existing owner-private directory")
    parser.add_argument("--expected-model", required=True, help="exact configured model ID to run and attest")
    parser.add_argument("--metadata-http-status", required=True, type=int, help="operator-reported status from a fresh external metadata-only check; must be 200")
    parser.add_argument("--metadata-model-id", required=True, help="exact operator-reported model ID from the metadata check")
    parser.add_argument("--metadata-endpoint-count", required=True, type=int, help="operator-reported positive serving-endpoint count")
    parser.add_argument("--metadata-checked-at", required=True, help="operator-reported metadata check time as ISO 8601 UTC ending in Z")
    parser.add_argument("--route-ready-confirmed", action="store_true", help="confirm the supplied recent metadata attestation; the runner does not fetch or authenticate it")
    parser.add_argument("--confirm-provider-inference", action="store_true", help="separately authorize one live M1 provider-backed turn")
    args = parser.parse_args(argv)
    try:
        record = run_trial(
            fixture_dir=args.fixture_dir,
            prompt_id=args.prompt_id,
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
        print(f"M1 trial refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"M1 trial failed ({type(exc).__name__}); inspect the private record if created.", file=sys.stderr)
        return 1
    if record["brain_execution_status"] != "returned" or record["memory_store_integrity"]["cleanup_status"] != "removed":
        print("M1 headless turn recorded with an execution or cleanup error; outcome remains not_scored.", file=sys.stderr)
        return 1
    print(
        "M1 headless turn recorded; outcome remains not_scored and requires independent review. "
        f"Private record: {record['private_output_paths']['record']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
