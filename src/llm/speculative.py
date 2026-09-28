"""Speculative Pre-flight Execution Router for Adam Voice Assistant.

Analyzes volatile streaming transcripts while the user is still speaking.
Prefetches and caches idempotent, read-only tools (web search, weather, time, telemetry)
in the background, dropping perceived tool latency to near zero upon utterance completion.
"""

from __future__ import annotations

import re
import time
import asyncio
from typing import Any, Callable


class SpeculativeCacheEntry:
    def __init__(self, tool_name: str, args: dict[str, Any], task: asyncio.Task, timestamp: float):
        self.tool_name = tool_name
        self.args = args
        self.task = task
        self.timestamp = timestamp

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > 12.0


class SpeculativeRouter:
    """Speculatively runs read-only tools on partial transcripts."""

    TIME_REGEX = re.compile(
        r"\b(?:what(?:'s|\s+is)\s+the\s+time|what\s+time\s+is\s+it)(?:\s+in\s+([a-zA-Z\s]+))?",
        re.IGNORECASE,
    )
    WEATHER_REGEX = re.compile(
        r"\b(?:what(?:'s|\s+is)\s+the\s+weather|weather\s+forecast|how(?:'s|\s+is)\s+the\s+weather)(?:\s+(?:in|for|at)\s+([a-zA-Z\s]+))?",
        re.IGNORECASE,
    )
    SEARCH_REGEX = re.compile(
        r"\b(?:could\s+you\s+)?(?:search\s+(?:the\s+web\s+|up\s+|for\s+)?|google\s+|look\s+up\s+)(.+)",
        re.IGNORECASE,
    )
    BATTERY_REGEX = re.compile(
        r"\b(?:how\s+much\s+battery|battery\s+(?:level|status|percentage|left)|system\s+(?:status|telemetry|health|specs|stats)|cpu\s+usage|ram\s+usage|memory\s+usage)\b",
        re.IGNORECASE,
    )

    def __init__(self, executor_map: dict[str, Callable] | None = None):
        self.executor_map = executor_map or {}
        self.active_entry: SpeculativeCacheEntry | None = None
        self._lock = asyncio.Lock()

    def set_executor(self, tool_name: str, fn: Callable):
        self.executor_map[tool_name] = fn

    def parse_partial_intent(self, partial_text: str, wake_word: str = "") -> tuple[str, dict[str, Any]] | None:
        """Extracts candidate read-only tool and args from a partial transcript."""
        if not partial_text:
            return None

        clean = partial_text.strip()
        lower = clean.lower()

        if wake_word:
            w = wake_word.strip().lower()
            if clean.lower().startswith(w):
                clean = clean[len(w):].lstrip(" ,:;!-.")
            elif w.startswith("hey ") and clean.lower().startswith(w[4:]):
                clean = clean[len(w[4:]):].lstrip(" ,:;!-.")

        clean = re.sub(r"[.!?。！？]+$", "", clean).strip()

        # 1. Time query
        m_time = self.TIME_REGEX.search(clean)
        if m_time:
            loc = (m_time.group(1) or "local").strip()
            return "get_current_time", {"location": loc}

        # 2. Weather query
        m_weather = self.WEATHER_REGEX.search(clean)
        if m_weather:
            loc = (m_weather.group(1) or "local").strip()
            return "get_weather", {"location": loc}

        # 3. Web search query (require at least 4 characters in query)
        m_search = self.SEARCH_REGEX.search(clean)
        if m_search:
            query = m_search.group(1).strip()
            # Don't prefetch on incomplete single short word
            if len(query) >= 4 and not query.lower().endswith(("for", "to", "in", "the", "a", "up")):
                return "web_search", {"query": query}

        # 4. Battery / System status query
        if self.BATTERY_REGEX.search(clean):
            return "get_system_status", {}

        return None

    async def preflight(self, partial_text: str, wake_word: str = "") -> bool:
        """Evaluates partial text and launches speculative background task if read-only intent detected."""
        intent = self.parse_partial_intent(partial_text, wake_word=wake_word)
        if not intent:
            return False

        tool_name, args = intent
        executor = self.executor_map.get(tool_name)
        if not executor and tool_name == "get_system_status":
            executor = self.executor_map.get("battery_status")
        if not executor:
            return False

        async with self._lock:
            # If current active speculative task already matches, let it keep running
            if self.active_entry and not self.active_entry.is_expired:
                if self.active_entry.tool_name == tool_name and self.active_entry.args == args:
                    return True
                # User changed parameters/intent; cancel previous speculative task
                if not self.active_entry.task.done():
                    self.active_entry.task.cancel()

            # Launch speculative task
            async def _run():
                try:
                    if asyncio.iscoroutinefunction(executor):
                        return await executor(**args)
                    return await asyncio.to_thread(executor, **args)
                except asyncio.CancelledError:
                    return None
                except Exception as e:
                    return None

            task = asyncio.create_task(_run())
            self.active_entry = SpeculativeCacheEntry(tool_name, args, task, time.time())
            return True

    async def consume_speculative_result(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, Any]:
        """Checks if a matching speculative result is available. Returns (hit, result)."""
        async with self._lock:
            if not self.active_entry or self.active_entry.is_expired:
                return False, None

            # Check matching tool and arguments (supporting battery_status / get_system_status alias)
            tool_a = self.active_entry.tool_name
            tool_b = tool_name
            alias_map = {"battery_status": "get_system_status"}
            if alias_map.get(tool_a, tool_a) != alias_map.get(tool_b, tool_b):
                return False, None

            # Case-insensitive argument matching
            entry_args = self.active_entry.args
            matched = True
            for k, v in args.items():
                ev = entry_args.get(k)
                if isinstance(v, str) and isinstance(ev, str):
                    if v.strip().lower() != ev.strip().lower():
                        matched = False
                        break
                elif ev != v:
                    matched = False
                    break

            if not matched:
                return False, None

            task = self.active_entry.task
            try:
                # Wait up to 1.5s for speculative task to complete if still in flight
                result = await asyncio.wait_for(task, timeout=1.5)
                if result is not None:
                    print(f"[Speculative] Cache HIT for {tool_name}({args})! Zero network latency.", flush=True)
                    self.active_entry = None
                    return True, result
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass

            self.active_entry = None
            return False, None

    def cancel_active(self):
        """Cancels any in-flight speculative task."""
        if self.active_entry and not self.active_entry.task.done():
            self.active_entry.task.cancel()
        self.active_entry = None
