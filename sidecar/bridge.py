"""Runtime bridge abstraction for Adam's WebUI sidecar.

Wires browser chat and honest runtime telemetry to Adam's ReAct engine
while explicitly reporting unavailable features (such as durable task
checkpoints and multi-channel remote approvals) where the runtime lacks
safe APIs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
import json
import math
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
        request_id: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def cancel_user_message(self, request_id: str) -> bool:
        """Request cancellation of one active bridge-owned chat turn."""
        return False

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


_FRESH_DESKTOP_OBSERVATION_PROMPT = (
    "Fresh desktop observation after the preceding action. Treat this as current state, "
    "compare it with the user's requested outcome, and continue or answer only when the "
    "requested result is supported by evidence. If the user asks what text or value was "
    "saved, entered, or selected, read back the exact visible text or value when it is "
    "legible; do not replace a requested readback with a generic completion claim. Say "
    "you cannot verify it only when the relevant evidence in this observation is actually "
    "unreadable or ambiguous."
)

_EMPTY_COMPLETION_RETRY_PROMPT = (
    "The previous model turn contained no answer and no action. Continue the current user request "
    "using the available evidence and tools. Take an action only if it is needed to reach the "
    "requested outcome; otherwise give a concise answer or state the blocker."
)


def _is_internal_orchestration_prompt(content: str) -> bool:
    """Identify the synthetic user-role prompts added by Brain between tool hops."""
    if content in {_FRESH_DESKTOP_OBSERVATION_PROMPT, _EMPTY_COMPLETION_RETRY_PROMPT}:
        return True
    return re.fullmatch(
        r"Tool recovery [1-3]/3: .+ failed\. "
        r"Use the failure details in the tool results to correct the approach or arguments\. "
        r"Continue the original request, preserving successful steps\. For a timed-out or "
        r"partially dispatched action, inspect its current effects before repeating it\. "
        r"Do not repeat a write, click, or other action blindly; do not expand authorization "
        r"or claim completion\. If no permitted recovery exists, explain the blocker\."
        r"(?: Use a fresh desktop observation before any further input\. Treat successful "
        r"sequence steps reported in the result as completed and omit them from any new call\. "
        r"Choose and dispatch only the next action grounded in that fresh observation; if no "
        r"fresh observation is available, inspect first\.)?",
        content,
    ) is not None


class DaemonBridge(RuntimeBridge):
    """Bridge connected directly to an active AdamDaemon instance."""

    def __init__(self, daemon: Any) -> None:
        super().__init__()
        self.daemon = daemon
        self._lock = asyncio.Lock()
        self._event_task = None
        self._unsubscribe_events = None
        self._active_webui_request: dict[str, Any] | None = None
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
                "brain.capability_recovery", "brain.long_task_progress",
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
                        event_duration = attrs.get("duration_ms")
                        if (
                            isinstance(event_duration, (int, float))
                            and not isinstance(event_duration, bool)
                            and event_duration >= 0
                            and (isinstance(event_duration, int) or math.isfinite(event_duration))
                        ):
                            duration = round(event_duration)
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

    def register_user_message(self, request_id: str) -> bool:
        """Register an incoming WebUI request before its task can start running."""
        if self._active_webui_request is not None:
            return False
        self._active_webui_request = {
            "request_id": request_id,
            "cancel_requested": False,
            "react_task": None,
        }
        return True

    def _release_user_message(self, request_id: str | None) -> None:
        active = self._active_webui_request
        if active is not None and active.get("request_id") == request_id:
            self._active_webui_request = None

    async def cancel_user_message(self, request_id: str) -> bool:
        """Cancel only the ReAct task owned by this WebUI request."""
        active = self._active_webui_request
        if active is None or active.get("request_id") != request_id:
            return False
        react_task = active.get("react_task")
        if react_task is not None and react_task.done():
            return False

        confirmation = getattr(self.daemon, "confirmation", None)
        pending_action = getattr(confirmation, "pending_action", None)
        confirmation_owned = getattr(
            confirmation, "is_confirmation_action_owned_by", None
        )
        if (
            react_task is not None
            and callable(confirmation_owned)
            and confirmation_owned(react_task)
        ):
            # A microphone answer owns this confirmation transition, or has
            # just affirmed it and may be dispatching the action. Do not cancel
            # its owner; the UI must report Stop rejected.
            return False

        active["cancel_requested"] = True
        brain = getattr(self.daemon, "brain", None)
        # Before ReAct starts, the flag is checked before dispatch and again
        # from turn.started. Never cancel an unrelated voice task.
        if (
            brain is not None
            and react_task is not None
            and react_task is getattr(brain, "_active_react_task", None)
        ):
            cancel_active = getattr(brain, "cancel_active_execution", None)
            if callable(cancel_active):
                cancel_active()
        # The registered task belongs to this WebUI request even if it just
        # finished and cleared Brain's active-task pointer. The confirmation
        # manager also checks action and task identity before clearing anything,
        # so a microphone-owned confirmation cannot be affected.
        if react_task is not None:
            cancel_confirmation = getattr(
                confirmation, "cancel_pending_confirmation_silently", None
            )
            if pending_action is not None and callable(cancel_confirmation):
                await cancel_confirmation(pending_action, react_task)
        return True

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

        meeting_session_active = None
        meeting_session = getattr(self.daemon, "meeting_session", None)
        if meeting_session is not None:
            try:
                capture_state = getattr(meeting_session, "active", None)
            except Exception:
                capture_state = None
            if isinstance(capture_state, bool):
                meeting_session_active = capture_state

        return {
            "connected": True,
            "mode": "daemon",
            "system_state": state,
            "meeting_session_active": meeting_session_active,
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
                    "available": True,
                    "description": "An active WebUI chat request can be stopped by its request ID. Durable background jobs do not yet have pause or resume controls.",
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
            if (
                role == "user"
                and isinstance(content, str)
                and _is_internal_orchestration_prompt(content)
            ):
                continue
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
                    "timestamp": msg.get("timestamp"),
                })
        append_local_exchanges_through(len(messages))
        return history

    async def handle_user_message(
        self, text: str, allowed_tools: list[str] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        text = text.strip()
        if not text:
            self._release_user_message(request_id)
            return {"status": "error", "error": "Message cannot be empty."}

        brain = getattr(self.daemon, "brain", None)
        if brain is None:
            self._release_user_message(request_id)
            return {"status": "error", "error": "Brain component not initialized."}
        execute_turn = getattr(self.daemon, "_execute_turn", None)
        if not callable(execute_turn):
            self._release_user_message(request_id)
            return {"status": "error", "error": "Daemon turn execution is unavailable."}

        if allowed_tools is not None:
            if (
                not isinstance(allowed_tools, list)
                or any(not isinstance(name, str) or not name.strip() for name in allowed_tools)
                or len(set(allowed_tools)) != len(allowed_tools)
            ):
                self._release_user_message(request_id)
                return {
                    "status": "error",
                    "error": "allowed_tools must be a list of unique non-empty tool names.",
                }
            get_tools = getattr(brain, "get_tools", None)
            if not callable(get_tools):
                self._release_user_message(request_id)
                return {"status": "error", "error": "Tool scope cannot be validated by this brain."}
            available_names = {
                str(getattr(tool, "name", "")) for tool in get_tools()
                if str(getattr(tool, "name", ""))
            }
            unknown_tools = [name for name in allowed_tools if name not in available_names]
            if unknown_tools:
                self._release_user_message(request_id)
                return {
                    "status": "error",
                    "error": f"Unknown or unavailable tool name(s): {', '.join(unknown_tools)}",
                }

        arbiter = getattr(self.daemon, "arbiter", None)

        async with self._lock:
            request_state = None
            if request_id is not None:
                request_state = self._active_webui_request
                if request_state is None:
                    self.register_user_message(request_id)
                    request_state = self._active_webui_request
                if request_state is None or request_state.get("request_id") != request_id:
                    return {"status": "busy", "error": "Adam is already working on a WebUI request."}
                if request_state.get("cancel_requested"):
                    self._active_webui_request = None
                    return {
                        "status": "cancelled",
                        "response": "Stop requested before Adam began the request. Any dispatched action may have taken effect; check the current state before retrying.",
                        "tool_calls": [],
                    }

            # Check if currently busy or awaiting confirmation
            if arbiter is not None:
                from src.arbiter.arbiter import SystemState
                busy_error = None
                state_lock = getattr(arbiter, "_state_lock", None)
                if request_state is not None and state_lock is not None:
                    # Reserve the daemon turn atomically before memory lookup or
                    # any other await, so a microphone turn cannot start and be
                    # mistaken for this WebUI request during the ReAct-start gap.
                    async with state_lock:
                        current_state = arbiter.current_state
                        if current_state == SystemState.AWAITING_CONFIRMATION:
                            busy_error = "Adam is currently awaiting verbal confirmation for an existing action. Please answer via microphone."
                        elif current_state != SystemState.IDLE_LISTENING:
                            curr_val = getattr(current_state, "value", str(current_state))
                            busy_error = f"Adam is currently busy with a voice interaction ({curr_val}). Please wait for it to complete."
                        else:
                            arbiter.current_state = SystemState.PROCESSING_REACT
                            request_state["reservation_acquired"] = True
                elif arbiter.current_state == SystemState.AWAITING_CONFIRMATION:
                    busy_error = "Adam is currently awaiting verbal confirmation for an existing action. Please answer via microphone."
                elif arbiter.current_state != SystemState.IDLE_LISTENING:
                    curr_val = getattr(arbiter.current_state, "value", str(arbiter.current_state))
                    busy_error = f"Adam is currently busy with a voice interaction ({curr_val}). Please wait for it to complete."
                if busy_error:
                    if request_state is not None and self._active_webui_request is request_state:
                        self._active_webui_request = None
                    return {
                        "status": "busy",
                        "error": busy_error,
                    }
            try:
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
            except BaseException:
                if (
                    request_state is not None
                    and request_state.get("reservation_acquired")
                    and arbiter is not None
                ):
                    from src.arbiter.arbiter import SystemState
                    if arbiter.current_state == SystemState.PROCESSING_REACT:
                        await arbiter.set_state("IDLE_LISTENING")
                self._release_user_message(request_id)
                raise
            try:
                if request_state is not None and request_state.get("cancel_requested"):
                    return {
                        "status": "cancelled",
                        "response": "Stop requested before Adam began the request. Any dispatched action may have taken effect; check the current state before retrying.",
                        "tool_calls": [],
                    }
                # Memory retrieval integration if memory manager is active
                mem_ctx = None
                mem_mgr = getattr(self.daemon, "memory_manager", None)
                if mem_mgr is not None and hasattr(mem_mgr, "retrieve_context"):
                    retrieve_memory_context = getattr(
                        self.daemon, "_retrieve_memory_context", None
                    )
                    if callable(retrieve_memory_context):
                        mem_ctx = await retrieve_memory_context(text)
                    else:
                        mem_ctx = await asyncio.to_thread(mem_mgr.retrieve_context, text)

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
                        if request_state is not None and event.get("event") == "turn.started":
                            request_state["react_task"] = current_task
                            if request_state.get("cancel_requested"):
                                cancel_active = getattr(brain, "cancel_active_execution", None)
                                if callable(cancel_active):
                                    cancel_active()

                unsubscribe_turn_events = subscribe_events(capture_current_brain_turn)
                execute_kwargs = {"memory_context": mem_ctx}
                # Real AdamDaemon exposes the lookup-complete marker. Keep
                # compatibility with injected executors and older adapters.
                try:
                    if "memory_context_resolved" in inspect.signature(execute_turn).parameters:
                        execute_kwargs["memory_context_resolved"] = True
                except (TypeError, ValueError):
                    pass
                # A stop can arrive while memory retrieval is still running.
                # Check again at the last point before handing control to Adam.
                if request_state is not None and request_state.get("cancel_requested"):
                    return {
                        "status": "cancelled",
                        "response": "Stop requested before execution. Any desktop action already sent may have taken effect; check the current screen before retrying.",
                        "tool_calls": [],
                    }
                if allowed_tools is None:
                    turn_response = await execute_turn(text, **execute_kwargs)
                else:
                    turn_response = await execute_turn(
                        text, **execute_kwargs, allowed_tools=allowed_tools
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

                # ReAct may have produced partial text before a stop landed,
                # including while TTS was finishing. A stop accepted for this
                # request always gets an explicit cancelled result.
                if request_state is not None and request_state.get("cancel_requested"):
                    return {
                        "status": "cancelled",
                        "response": "Stop requested. Any dispatched action may have taken effect; check the current state before retrying.",
                        "tool_calls": tool_calls_executed,
                    }

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
                if request_state is not None and self._active_webui_request is request_state:
                    self._active_webui_request = None
                if (
                    request_state is not None
                    and request_state.get("reservation_acquired")
                    and request_state.get("react_task") is None
                    and arbiter is not None
                ):
                    from src.arbiter.arbiter import SystemState
                    if arbiter.current_state == SystemState.PROCESSING_REACT:
                        await arbiter.set_state("IDLE_LISTENING")
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
