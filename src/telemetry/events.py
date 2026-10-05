"""Small, privacy-conscious JSONL event sink for latency diagnostics.

The sink is disabled unless explicitly enabled in application config. Event
attributes must be operational metadata; transcript and request payloads are
never valid attributes.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
import json
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable


_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "adam_telemetry_trace_id", default=None
)
_speech_role: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "adam_telemetry_speech_role", default=None
)
_SPEECH_ROLES = frozenset({"acknowledgment", "progress", "final"})
_writer_lock = threading.Lock()
_writer: "EventWriter | None" = None
_reported_write_failure = False
_event_listeners: set[Callable[[dict[str, Any]], None]] = set()


def subscribe_events(listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
    """Subscribe to operational metadata in memory, independently of disk logging."""
    with _writer_lock:
        _event_listeners.add(listener)

    def unsubscribe():
        with _writer_lock:
            _event_listeners.discard(listener)

    return unsubscribe

_SENSITIVE_KEY_PARTS = (
    "transcript", "prompt", "text", "audio", "image", "argument", "result",
    "content", "secret", "api_key", "credential", "authorization",
)
_USAGE_KEYS = {
    "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
    "cache_write_tokens", "cost", "cost_usd", "credit_cost", "credits",
    "provider_reported_cost", "reasoning_tokens", "currency",
}


def get_trace_id() -> str | None:
    """Return the correlation ID active in this async context, if any."""
    return _trace_id.get()


def new_span_id() -> str:
    """Create an opaque span ID for matching start and completion events."""
    return str(uuid.uuid4())


def set_trace_id(trace_id: str | None):
    """Set the active trace ID; retain the token and pass it to reset_trace_id."""
    return _trace_id.set(trace_id)


def reset_trace_id(token) -> None:
    """Restore the trace context that was active before ``set_trace_id``."""
    _trace_id.reset(token)


@contextmanager
def speech_role_scope(role: str):
    """Mark TTS playback purpose without retaining the speech content."""
    if not isinstance(role, str) or role not in _SPEECH_ROLES:
        raise ValueError(f"unsupported speech role: {role!r}")
    token = _speech_role.set(role)
    try:
        yield
    finally:
        _speech_role.reset(token)


def get_speech_role() -> str | None:
    """Return the purpose label active for playback in this async context."""
    return _speech_role.get()


def _safe_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}

    safe: dict[str, Any] = {}
    for raw_key, value in attributes.items():
        key = str(raw_key)[:64]
        key_lower = key.lower()
        if key_lower == "usage" and isinstance(value, dict):
            usage = {}
            for usage_key, usage_value in value.items():
                normalized = str(usage_key)[:64].lower()
                if (
                    normalized in _USAGE_KEYS
                    and (
                        (isinstance(usage_value, (int, float))
                         and not isinstance(usage_value, bool)
                         and (not isinstance(usage_value, float) or math.isfinite(usage_value)))
                        or (normalized == "currency" and isinstance(usage_value, str)
                            and len(usage_value) <= 24 and usage_value.isascii())
                    )
                ):
                    usage[normalized] = usage_value
            if usage:
                safe["usage"] = usage
            continue
        if any(part in key_lower for part in _SENSITIVE_KEY_PARTS) or "token" in key_lower:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
            safe[key] = value
        elif isinstance(value, str):
            safe[key] = value[:160]
    return safe


class EventWriter:
    """Append schema-versioned event rows to a local JSONL file."""

    def __init__(self, enabled: bool = False, path: str = "~/.local/state/adam/telemetry/events.jsonl"):
        self.enabled = bool(enabled)
        self.path = Path(os.path.expandvars(os.path.expanduser(str(path))))
        self._lock = threading.Lock()

    def emit(
        self,
        event: str,
        *,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        status: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        component: str | None = None,
        attempt: int | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        row: dict[str, Any] = {
            "schema_version": 1,
            "trace_id": trace_id or get_trace_id() or str(uuid.uuid4()),
            "span_id": span_id or str(uuid.uuid4()),
            "parent_span_id": parent_span_id,
            "event": str(event)[:96],
            "clock_ns": time.monotonic_ns(),
            "status": status,
            "provider": provider[:80] if isinstance(provider, str) else None,
            "model": model[:120] if isinstance(model, str) else None,
            "component": component[:80] if isinstance(component, str) else None,
            "attempt": attempt,
            "attributes": _safe_attributes(attributes),
        }
        try:
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self._lock:
                descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
                try:
                    os.fchmod(descriptor, 0o600)
                    remaining = memoryview((encoded + "\n").encode("utf-8"))
                    while remaining:
                        written = os.write(descriptor, remaining)
                        if written <= 0:
                            raise OSError("short JSONL event write")
                        remaining = remaining[written:]
                finally:
                    os.close(descriptor)
            return row
        except (OSError, TypeError, ValueError) as exc:
            global _reported_write_failure
            with _writer_lock:
                if not _reported_write_failure:
                    print(f"[Telemetry] Event write failed ({type(exc).__name__}).", file=sys.stderr, flush=True)
                    _reported_write_failure = True
            return None


def configure_telemetry(config) -> EventWriter:
    """Install the process-wide event writer from ``TelemetryConfig`` or equivalent."""
    global _writer
    _writer = EventWriter(
        enabled=getattr(config, "enabled", False),
        path=getattr(config, "path", "~/.local/state/adam/telemetry/events.jsonl"),
    )
    return _writer


def emit_event(event: str, **kwargs) -> dict[str, Any] | None:
    """Emit one event through the configured writer; safe before configuration."""
    writer = _writer
    row = writer.emit(event, **kwargs) if writer is not None else None
    with _writer_lock:
        listeners = tuple(_event_listeners)
    if listeners:
        live_row = {
            "event": str(event)[:96], "span_id": kwargs.get("span_id"),
            "status": kwargs.get("status"), "clock_ns": time.monotonic_ns(),
            "attributes": _safe_attributes(kwargs.get("attributes")),
        }
        for listener in listeners:
            try:
                listener(live_row)
            except Exception:
                # A display subscriber must never prevent execution.
                pass
    return row
