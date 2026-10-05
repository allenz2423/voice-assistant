"""Runtime bridge abstraction for Adam's WebUI sidecar.

Wires browser chat and honest runtime telemetry to Adam's ReAct engine
while explicitly reporting unavailable features (such as durable task
checkpoints and multi-channel remote approvals) where the runtime lacks
safe APIs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import re
from typing import Any, Callable, Awaitable
from src.config import AppConfig, load_config


class RuntimeBridge:
    """Base interface for bridging between WebUI sidecar and Adam runtime."""

    def __init__(self) -> None:
        self._listeners: set[Callable[[dict[str, Any]], Awaitable[None]]] = set()

    async def get_status(self) -> dict[str, Any]:
        raise NotImplementedError

    async def start_live_events(self) -> None:
        pass

    async def stop_live_events(self) -> None:
        pass

    async def get_history(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def handle_user_message(self, text: str) -> dict[str, Any]:
        raise NotImplementedError

    async def register_listener(self, callback: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._listeners.add(callback)

    async def unregister_listener(self, callback: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._listeners.discard(callback)

    async def broadcast_state(self, payload: dict[str, Any]) -> None:
        if not self._listeners:
            return
        coros = []
        for cb in list(self._listeners):
            try:
                coros.append(cb(payload))
            except Exception:
                pass
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)


class DaemonBridge(RuntimeBridge):
    """Bridge connected directly to an active AdamDaemon instance."""

    def __init__(self, daemon: Any) -> None:
        super().__init__()
        self.daemon = daemon
        self._lock = asyncio.Lock()
        self._event_task = None
        self._unsubscribe_events = None

    async def start_live_events(self) -> None:
        if self._event_task is not None:
            return
        from src.telemetry.events import subscribe_events
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=256)
        starts = {}

        def enqueue(row):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(row)

        def receive(row):
            if row.get("event") in {"tool.started", "tool.completed"}:
                loop.call_soon_threadsafe(enqueue, row)

        async def deliver():
            while True:
                row = await queue.get()
                span = row.get("span_id")
                attrs = row.get("attributes", {})
                started = row["event"] == "tool.started"
                duration = None
                if started:
                    if len(starts) >= 256:
                        starts.pop(next(iter(starts)))
                    starts[span] = row["clock_ns"]
                else:
                    clock = starts.pop(span, None)
                    if clock is not None:
                        duration = round((row["clock_ns"] - clock) / 1_000_000)
                await self.broadcast_state({
                    "type": "tool_activity", "span_id": span,
                    "tool": attrs.get("tool_name", "tool"),
                    "phase": "started" if started else "finished",
                    "outcome": attrs.get("outcome") or row.get("status"),
                    "duration_ms": duration,
                })

        self._unsubscribe_events = subscribe_events(receive)
        self._event_task = asyncio.create_task(deliver())

    async def stop_live_events(self) -> None:
        if self._unsubscribe_events is not None:
            self._unsubscribe_events()
            self._unsubscribe_events = None
        if self._event_task is not None:
            self._event_task.cancel()
            await asyncio.gather(self._event_task, return_exceptions=True)
            self._event_task = None

    async def get_status(self) -> dict[str, Any]:
        state = "UNKNOWN"
        arbiter = getattr(self.daemon, "arbiter", None)
        if arbiter is not None:
            curr = getattr(arbiter, "current_state", None)
            state = getattr(curr, "value", str(curr)) if curr else "IDLE_LISTENING"

        brain = getattr(self.daemon, "brain", None)
        model_name = "default"
        tools_count = 0
        if brain is not None:
            llm_client = getattr(brain, "llm_client", None)
            if llm_client is not None:
                model_name = getattr(llm_client, "model", "default")
            tools = brain.get_tools() if hasattr(brain, "get_tools") else []
            tools_count = len(tools)

        active_tool = None
        if brain is not None:
            active_task = getattr(brain, "_active_tool_task", None)
            if active_task is not None and not active_task.done():
                active_tool = "executing_tool"

        return {
            "connected": True,
            "mode": "daemon",
            "system_state": state,
            "voice_default": True,
            "model": model_name,
            "tools_count": tools_count,
            "active_tool": active_tool,
            "features": {
                "chat": {
                    "available": True,
                    "description": "Uses the daemon turn executor and microphone interruption monitor",
                },
                "conversation_history": {
                    "available": True,
                    "description": "Active session message log",
                },
                "long_running_tasks": {
                    "available": False,
                    "reason": "Runtime lacks durable task checkpoint store (P3 gate in progress per proposed-codebase-fixes.md)",
                },
                "run_controls": {
                    "available": False,
                    "reason": "Runtime lacks safe pause/resume/stop API for long runs",
                },
                "remote_approvals": {
                    "available": False,
                    "reason": "Verbal confirmation via microphone is Adam's only active confirmation path. Multi-channel remote approval API is not yet implemented.",
                },
            },
        }

    async def get_history(self) -> list[dict[str, Any]]:
        brain = getattr(self.daemon, "brain", None)
        if brain is None or not hasattr(brain, "messages"):
            return []

        history: list[dict[str, Any]] = []
        # messages[0] is system prompt; messages[1:] contain conversational turns
        for msg in list(brain.messages)[1:]:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if role == "user" and isinstance(content, str):
                content = re.sub(r"^\[Current Desktop State\].*?\[Local Time:[^\]]+\]\s*\n", "", content, flags=re.S).strip()
            tool_calls = msg.get("tool_calls")
            formatted_tc = None
            if tool_calls:
                formatted_tc = []
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                    name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "tool")
                    formatted_tc.append(name)

            if content or formatted_tc:
                history.append({
                    "role": role,
                    "content": content,
                    "tool_calls": formatted_tc,
                    "timestamp": msg.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                })
        return history

    async def handle_user_message(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            return {"status": "error", "error": "Message cannot be empty."}

        brain = getattr(self.daemon, "brain", None)
        if brain is None:
            return {"status": "error", "error": "Brain component not initialized."}
        execute_turn = getattr(self.daemon, "_execute_turn", None)
        if not callable(execute_turn):
            return {"status": "error", "error": "Daemon turn execution is unavailable."}

        arbiter = getattr(self.daemon, "arbiter", None)

        async with self._lock:
            # Check if currently busy or awaiting confirmation
            if arbiter is not None:
                from src.arbiter.arbiter import SystemState
                if arbiter.current_state == SystemState.AWAITING_CONFIRMATION:
                    return {
                        "status": "busy",
                        "error": "Adam is currently awaiting verbal confirmation for an existing action. Please answer via microphone.",
                    }
                if arbiter.current_state != SystemState.IDLE_LISTENING:
                    curr_val = getattr(arbiter.current_state, "value", str(arbiter.current_state))
                    return {
                        "status": "busy",
                        "error": f"Adam is currently busy with a voice interaction ({curr_val}). Please wait for it to complete.",
                    }
            await self.broadcast_state({"type": "state", "system_state": "PROCESSING_REACT"})

            prev_len = len(brain.messages)
            try:
                # Memory retrieval integration if memory manager is active
                mem_ctx = None
                mem_mgr = getattr(self.daemon, "memory_manager", None)
                if mem_mgr is not None and hasattr(mem_mgr, "retrieve_context"):
                    mem_ctx = mem_mgr.retrieve_context(text)

                # Use the same daemon execution path as voice requests. This
                # preserves its microphone interruption monitoring and turn
                # cleanup instead of leaving WebUI turns uninterruptible.
                brain._turn_message_start = prev_len
                await execute_turn(text, memory_context=mem_ctx)

                turn_start = getattr(brain, "_turn_message_start", prev_len)
                new_messages = brain.messages[turn_start:]
                final_response = ""
                tool_calls_executed = []

                for m in new_messages:
                    if m.get("role") == "assistant":
                        if m.get("content"):
                            final_response = m.get("content", "")
                        for tc in m.get("tool_calls") or []:
                            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "tool")
                            tool_calls_executed.append(name)

                if not str(final_response or "").strip():
                    return {
                        "status": "error",
                        "error": "Adam's turn ended without a text reply. It may have been interrupted or failed; no completion was confirmed.",
                        "tool_calls": tool_calls_executed,
                    }

                return {
                    "status": "completed",
                    "response": final_response,
                    "tool_calls": tool_calls_executed,
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "error": f"Execution error: {type(exc).__name__}: {exc}",
                }
            finally:
                if arbiter is not None:
                    curr_state = getattr(arbiter.current_state, "value", str(arbiter.current_state))
                else:
                    curr_state = "IDLE_LISTENING"
                await self.broadcast_state({"type": "state", "system_state": curr_state})


class DisconnectedBridge(RuntimeBridge):
    """Bridge representing a disconnected or unreachable daemon.

    Truthfully reports that the daemon is offline and rejects messages without
    faking connection or instantiating a secondary AdamBrain.
    """

    def __init__(self, reason: str | None = None) -> None:
        super().__init__()
        self.reason = reason or (
            "Adam daemon is not connected. To use the WebUI with live Adam voice "
            "and tools, start the daemon with WebUI enabled (e.g. `webui.enabled: true` "
            "in config or `python -m src.main --webui`). Standalone sidecar processes "
            "do not instantiate duplicate brain engines."
        )

    async def get_status(self) -> dict[str, Any]:
        return {
            "connected": False,
            "mode": "disconnected",
            "system_state": "DISCONNECTED",
            "voice_default": True,
            "model": "unavailable",
            "tools_count": 0,
            "active_tool": None,
            "features": {
                "chat": {
                    "available": False,
                    "reason": self.reason,
                },
                "conversation_history": {
                    "available": False,
                    "reason": "Daemon disconnected",
                },
                "long_running_tasks": {
                    "available": False,
                    "reason": "Runtime lacks durable task checkpoint store (P3 gate in progress per proposed-codebase-fixes.md)",
                },
                "run_controls": {
                    "available": False,
                    "reason": "Runtime lacks safe pause/resume/stop API for long runs",
                },
                "remote_approvals": {
                    "available": False,
                    "reason": "Verbal confirmation via microphone is Adam's only active confirmation path. Multi-channel remote approval API is not yet implemented.",
                },
            },
        }

    async def get_history(self) -> list[dict[str, Any]]:
        return []

    async def handle_user_message(self, text: str) -> dict[str, Any]:
        return {
            "status": "error",
            "error": "Cannot send message: Adam daemon is disconnected. Please launch the Adam daemon with WebUI enabled.",
        }


class StandaloneBridge(DisconnectedBridge):
    """Bridge for standalone sidecar execution.

    A standalone sidecar process must not instantiate a second AdamBrain or claim
    a fake live connection to the daemon. It truthfully presents a disconnected UI
    documenting how to run the WebUI wired into the live daemon.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__(
            reason=(
                "Standalone sidecar running without live daemon connection. "
                "Standalone sidecar does not instantiate a second AdamBrain. "
                "To connect WebUI to Adam, enable `webui.enabled: true` or run "
                "`python -m src.main --webui`."
            )
        )
        self.config = config
