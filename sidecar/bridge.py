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

    async def handle_user_message(
        self, text: str, allowed_tools: list[str] | None = None,
    ) -> dict[str, Any]:
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


def _explicit_silent_mode_action(text: str) -> str | None:
    """Return the explicit local silent-mode action, matching Brain routing."""
    negated_mode_request = re.search(
        r"\b(?:don't|do not|never|shouldn't|should not)\b[^.!?\n]{0,50}"
        r"\b(?:turn|switch|set|put|enable|activate)\b[^.!?\n]{0,40}"
        r"\b(?:silent|notification)\s+mode\b",
        text,
        re.IGNORECASE,
    )
    if negated_mode_request:
        return None

    disable_silent = re.search(
        r"\b(?:turn|switch)\s+off\b[^.!?\n]{0,30}\b(?:silent|notification)\s+mode\b|"
        r"\b(?:disable|deactivate|leave|exit)\b[^.!?\n]{0,30}\b(?:silent|notification)\s+mode\b",
        text,
        re.IGNORECASE,
    )
    if disable_silent:
        return "disable"

    enable_silent = re.search(
        r"\b(?:turn|switch|set|put|enable|activate)\b[^.!?\n]{0,30}"
        r"\b(?:silent|notification)\s+mode\b|"
        r"\b(?:silent|notification)\s+mode\s+on\b",
        text,
        re.IGNORECASE,
    )
    return "enable" if enable_silent else None


class DaemonBridge(RuntimeBridge):
    """Bridge connected directly to an active AdamDaemon instance."""

    def __init__(self, daemon: Any) -> None:
        super().__init__()
        self.daemon = daemon
        self._lock = asyncio.Lock()
        self._event_task = None
        self._unsubscribe_events = None
        # Browser-only local quickpaths belong in the displayed transcript, not
        # in AdamBrain's model context.
        self._display_only_exchanges: list[dict[str, Any]] = []

    @staticmethod
    def _display_exchange_anchor(
        exchange: dict[str, Any], messages: list[dict[str, Any]]
    ) -> int:
        """Resolve a local exchange's insertion point across Brain history compaction."""
        before_ids = exchange["before_message_ids"]
        if (
            len(messages) >= len(before_ids)
            and all(id(messages[index]) == message_id for index, message_id in enumerate(before_ids))
        ):
            return min(exchange["after_message_index"], len(messages))

        # Brain compaction rebuilds messages as fresh dictionaries and has no
        # timestamps or stable IDs. Text signatures can match a later repeated
        # turn, so fail safe by placing the local exchange before the retained
        # history instead of moving it past newer Brain messages.
        return 1

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
            if row.get("event") in {
                "turn.started", "llm.request_started", "llm.retrying",
                "tool.started", "tool.completed",
                "brain.tool_recovery", "brain.empty_completion_recovery",
                "brain.capability_recovery",
            }:
                loop.call_soon_threadsafe(enqueue, row)

        async def deliver():
            while True:
                row = await queue.get()
                span = row.get("span_id")
                attrs = row.get("attributes", {})
                if row["event"].startswith("tool."):
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
                    continue

                progress = {"type": "task_progress", "event": row["event"]}
                for field in ("attempt", "max_attempts", "reason"):
                    if field in attrs:
                        progress[field] = attrs[field]
                await self.broadcast_state(progress)

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
        tool_free_model_name = None
        tools_count = 0
        if brain is not None:
            llm_client = getattr(brain, "llm_client", None)
            if llm_client is not None:
                model_field = "local_model" if getattr(llm_client, "provider", None) == "local" else "cloud_model"
                configured_model = getattr(llm_client, model_field, None)
                model_name = configured_model if isinstance(configured_model, str) else getattr(llm_client, "model", "default")
            tool_free_client = getattr(brain, "tool_free_llm_client", None)
            if tool_free_client is not None:
                model_field = "local_model" if getattr(tool_free_client, "provider", None) == "local" else "cloud_model"
                configured_model = getattr(tool_free_client, model_field, None)
                if isinstance(configured_model, str):
                    tool_free_model_name = configured_model
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
            "tool_free_model": tool_free_model_name,
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
        messages = list(getattr(brain, "messages", []) or []) if brain is not None else []

        history: list[dict[str, Any]] = []
        local_exchanges = sorted(
            (
                (self._display_exchange_anchor(exchange, messages), exchange)
                for exchange in self._display_only_exchanges
            ),
            key=lambda anchored: anchored[0],
        )
        local_index = 0

        def append_local_exchanges_through(message_index: int) -> None:
            nonlocal local_index
            while (
                local_index < len(local_exchanges)
                and local_exchanges[local_index][0] <= message_index
            ):
                exchange = local_exchanges[local_index][1]
                history.extend((
                    {
                        "role": "user",
                        "content": exchange["user"],
                        "tool_calls": None,
                        "timestamp": exchange["timestamp"],
                    },
                    {
                        "role": "assistant",
                        "content": exchange["assistant"],
                        "tool_calls": None,
                        "timestamp": exchange["timestamp"],
                    },
                ))
                local_index += 1

        # messages[0] is system prompt; messages[1:] contain conversational turns
        for message_index, msg in enumerate(messages[1:], start=1):
            append_local_exchanges_through(message_index)
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if role == "user" and isinstance(content, str):
                content = re.sub(r"^\[Current Desktop State\].*?\[Local Time:[^\]]+\]\s*\n", "", content, flags=re.S).strip()
                context_headers = (
                    "[Current Desktop State]",
                    "[Current User Request",
                    "[Retrieved user memory]",
                    "[Relevant Specialized Skill Context]",
                    "[Fresh Desktop Screenshot]",
                    "[Fresh Adam Browser Snapshot]",
                )
                # Brain stores request context (retrieved skills, memory, and
                # desktop observations) alongside the user's request. Keep that
                # context in the model history, but expose only the request in
                # the WebUI conversation transcript.
                request = re.search(r"\[Current User Request[^\]]*\]\s*", content)
                if request:
                    end_marker = "[End Current User Request]"
                    closing_positions = []
                    search_from = request.end()
                    while (position := content.find(end_marker, search_from)) >= 0:
                        closing_positions.append(position)
                        search_from = position + len(end_marker)

                    closing_position = None
                    for position in closing_positions:
                        suffix = content[position + len(end_marker):].lstrip()
                        if not suffix or suffix.startswith(context_headers[2:]):
                            closing_position = position
                            break
                    compacted_truncation = content.endswith("...")
                    if closing_position is None and closing_positions and not compacted_truncation:
                        # Keep malformed or older wrapped messages readable; the
                        # final marker is less likely to be part of user text.
                        closing_position = closing_positions[-1]
                    if closing_position is not None:
                        content = content[request.end():closing_position].strip()
                    else:
                        # Brain compaction can cut off the wrapper's closing
                        # marker. In that case only the retained request prefix
                        # is safe to display; discard any context-looking suffix.
                        request_content = content[request.end():]
                        context_positions = [
                            position
                            for header in context_headers[2:]
                            if (position := request_content.find(header)) >= 0
                        ]
                        if context_positions:
                            request_content = request_content[:min(context_positions)]
                        request_content = request_content.strip()
                        if compacted_truncation and request_content.endswith("..."):
                            request_content = request_content[:-3].rstrip()
                        content = request_content
                else:
                    # A compacted prompt can end before the request header is
                    # reached. Never render the remaining automatic context as
                    # if it were user-authored text.
                    if content.startswith(context_headers):
                        content = ""
                    else:
                        context_positions = [
                            position
                            for header in context_headers
                            if (position := content.find(header)) >= 0
                        ]
                        if context_positions:
                            content = content[:min(context_positions)].strip()
            tool_calls = msg.get("tool_calls")
            formatted_tc = None
            if tool_calls:
                formatted_tc = []
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                    name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "tool")
                    formatted_tc.append(name)

            if (content or formatted_tc) and not (
                role == "user" and not str(content or "").strip()
            ):
                history.append({
                    "role": role,
                    "content": content,
                    "tool_calls": formatted_tc,
                    "timestamp": msg.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                })
        append_local_exchanges_through(len(messages))
        return history

    async def handle_user_message(
        self, text: str, allowed_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        text = text.strip()
        if not text:
            return {"status": "error", "error": "Message cannot be empty."}

        brain = getattr(self.daemon, "brain", None)
        if brain is None:
            return {"status": "error", "error": "Brain component not initialized."}
        execute_turn = getattr(self.daemon, "_execute_turn", None)
        if not callable(execute_turn):
            return {"status": "error", "error": "Daemon turn execution is unavailable."}

        if allowed_tools is not None:
            if (
                not isinstance(allowed_tools, list)
                or any(not isinstance(name, str) or not name.strip() for name in allowed_tools)
                or len(set(allowed_tools)) != len(allowed_tools)
            ):
                return {
                    "status": "error",
                    "error": "allowed_tools must be a list of unique non-empty tool names.",
                }
            get_tools = getattr(brain, "get_tools", None)
            if not callable(get_tools):
                return {"status": "error", "error": "Tool scope cannot be validated by this brain."}
            available_names = {
                str(getattr(tool, "name", "")) for tool in get_tools()
                if str(getattr(tool, "name", ""))
            }
            unknown_tools = [name for name in allowed_tools if name not in available_names]
            if unknown_tools:
                return {
                    "status": "error",
                    "error": f"Unknown or unavailable tool name(s): {', '.join(unknown_tools)}",
                }

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

            brain_messages_before_turn = list(brain.messages)
            prev_len = len(brain_messages_before_turn)
            silent_mode_action = _explicit_silent_mode_action(text)
            if allowed_tools is not None and silent_mode_action is not None:
                required_tool = (
                    "enable_silent_mode"
                    if silent_mode_action == "enable"
                    else "disable_silent_mode"
                )
                if required_tool not in allowed_tools:
                    silent_mode_action = None
            tts = getattr(brain, "tts", None)
            if tts is None:
                tts = getattr(self.daemon, "tts", None)
            previous_tts_engine = getattr(tts, "engine", None)
            turn_events: list[dict[str, Any]] = []
            unsubscribe_turn_events = None
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
                from src.telemetry.events import subscribe_events

                def capture_current_brain_turn(event: dict[str, Any]) -> None:
                    if event.get("event") not in {"turn.started", "turn.completed", "turn.cancelled"}:
                        return
                    try:
                        current_task = asyncio.current_task()
                    except RuntimeError:
                        return
                    # Brain emits these events from its active ReAct task. The
                    # identity check avoids treating another runtime turn as
                    # evidence that this WebUI request completed.
                    if current_task is getattr(brain, "_active_react_task", None):
                        turn_events.append(event)

                unsubscribe_turn_events = subscribe_events(capture_current_brain_turn)
                if allowed_tools is None:
                    turn_response = await execute_turn(text, memory_context=mem_ctx)
                else:
                    turn_response = await execute_turn(
                        text, memory_context=mem_ctx, allowed_tools=allowed_tools
                    )

                turn_start = getattr(brain, "_turn_message_start", prev_len)
                new_messages = brain.messages[turn_start:]
                # Some daemon-handled controls (such as meeting-mode start/stop)
                # return a local reply without adding synthetic LLM history.
                final_response = turn_response.strip() if isinstance(turn_response, str) else ""
                fallback_response = ""
                tool_calls_executed = []

                for m in new_messages:
                    if m.get("role") == "assistant":
                        content = m.get("content")
                        if content and not (m.get("tool_calls") or []):
                            if str(content).strip():
                                fallback_response = content
                        for tc in m.get("tool_calls") or []:
                            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "tool")
                            tool_calls_executed.append(name)

                if not final_response:
                    final_response = fallback_response

                if (
                    not str(final_response or "").strip()
                    and not new_messages
                    and silent_mode_action is not None
                    and self._successful_brain_turn(turn_events)
                ):
                    current_engine = getattr(tts, "engine", None)
                    if silent_mode_action == "enable" and current_engine == "silent":
                        final_response = (
                            "Silent mode is already enabled."
                            if previous_tts_engine == "silent"
                            else "Silent mode enabled. Future responses will appear as desktop notifications."
                        )
                    elif silent_mode_action == "disable":
                        config = getattr(self.daemon, "config", None)
                        restore_engine = str(
                            getattr(getattr(config, "tts", None), "silent_restore_engine", "kokoro")
                        ).lower()
                        if restore_engine == "silent" and current_engine == "silent":
                            final_response = "Silent mode is configured as the default and cannot be turned off."
                        elif restore_engine != "silent" and current_engine == restore_engine:
                            final_response = "Silent mode disabled. Spoken responses are restored."

                if not str(final_response or "").strip():
                    return {
                        "status": "error",
                        "error": "Adam's turn ended without a text reply. It may have been interrupted or failed; no completion was confirmed.",
                        "tool_calls": tool_calls_executed,
                    }

                if isinstance(turn_response, str) and not any(
                    m.get("role") in ("user", "assistant") for m in new_messages
                ):
                    self._display_only_exchanges.append({
                        "after_message_index": prev_len,
                        "before_message_ids": tuple(id(message) for message in brain_messages_before_turn),
                        "user": text,
                        "assistant": str(final_response).strip(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

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
                if unsubscribe_turn_events is not None:
                    unsubscribe_turn_events()
                if arbiter is not None:
                    curr_state = getattr(arbiter.current_state, "value", str(arbiter.current_state))
                else:
                    curr_state = "IDLE_LISTENING"
                await self.broadcast_state({"type": "state", "system_state": curr_state})

    @staticmethod
    def _successful_brain_turn(events: list[dict[str, Any]]) -> bool:
        """Require the request's first Brain turn to complete successfully."""
        started_span: str | None = None
        for event in events:
            span_id = event.get("span_id")
            if not isinstance(span_id, str):
                continue
            if started_span is None:
                if event.get("event") == "turn.started":
                    started_span = span_id
                continue
            if span_id == started_span and event.get("event") in {"turn.completed", "turn.cancelled"}:
                return event.get("event") == "turn.completed" and event.get("status") == "ok"
        return False


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
