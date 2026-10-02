"""Privacy-safe timing logs for desktop observation stages."""

from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager

from src.telemetry.events import get_trace_id


_operation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "desktop_timing_operation_id", default=None
)


@contextmanager
def timing_operation(name: str):
    """Group related desktop stage logs without recording screen content."""
    current = _operation_id.get()
    token = None
    if current is None:
        current = get_trace_id() or uuid.uuid4().hex[:12]
        token = _operation_id.set(current)
    started = time.perf_counter()
    status = "ok"
    try:
        yield current
    except BaseException as exc:
        status = "error"
        error_type = type(exc).__name__
        raise
    finally:
        log_duration(name, started, status=status, **({"error_type": error_type} if status == "error" else {}))
        if token is not None:
            _operation_id.reset(token)


@contextmanager
def timed_stage(name: str, **attributes):
    started = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException as exc:
        status = "error"
        attributes["error_type"] = type(exc).__name__
        raise
    finally:
        log_duration(name, started, status=status, **attributes)


def log_duration(name: str, started: float, *, status: str = "ok", **attributes) -> None:
    elapsed_ms = (time.perf_counter() - started) * 1000
    log_elapsed(name, elapsed_ms, status=status, **attributes)


def log_elapsed(name: str, elapsed_ms: float, *, status: str = "ok", **attributes) -> None:
    safe = " ".join(
        f"{key}={value}"
        for key, value in attributes.items()
        if isinstance(value, (str, int, float, bool)) and value is not None
    )
    operation_id = _operation_id.get() or get_trace_id() or "untraced"
    suffix = f" {safe}" if safe else ""
    print(
        f"[DesktopTiming] op={operation_id} stage={name} "
        f"duration_ms={elapsed_ms:.1f} status={status}{suffix}",
        flush=True,
    )
