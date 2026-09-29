import os
import re
import glob
import subprocess
import asyncio
import re
from pathlib import Path
from src.llm.provider import UniversalLLMClient
from src.llm.tools import ADAM_TOOLS
from src.tools.weather import get_weather_report
from src.tools.desktop import (
    list_applications,
    launch_application,
    close_application,
    list_windows,
    focus_window,
    swap_windows,
    workspace_control,
    manage_clipboard,
    get_open_windows_prompt_context,
    capture_screenshot,
    is_tool_enabled,
    execute_desktop_macro,
    list_desktop_macros,
    open_in_browser,
    close_browser_tab,
    show_desktop_notification,
)
from src.tools.system_telemetry import (
    get_system_status,
    list_processes,
    kill_process,
    list_audio_devices,
    volume_control
)
from src.tools.media import get_now_playing, media_control
from src.tools.web import web_search, fetch_webpage
from src.tools.timers import TimerManager
from src.tools.reminders import ReminderManager
from src.tools.noctalia import open_noctalia_calendar
from src.tools.noctalia_calendar import NoctaliaCalendar
from src.tools.waynote import (
    create_waynote,
    append_waynote,
    list_waynotes,
    manage_waynote,
)
from src.tools.financial import get_financial_quote
from src.tools.math_calc import calculate_math
from src.tools.dev_sys import (
    check_system_updates,
    manage_service,
    git_repo_status,
    docker_container_status
)
from src.skills import SkillManager
from src.tools.custom import CustomToolManager

TEXT_FALLBACK_READ_ONLY_TOOLS = {
    "find_files", "get_current_time", "get_weather", "list_applications",
    "list_windows", "get_system_status", "list_processes", "list_audio_devices",
    "get_now_playing", "web_search", "list_timers", "list_reminders",
    "get_financial_quote", "calculate_math", "list_skills", "get_skill_context",
    "fetch_webpage",
}

SYSTEM_PROMPT = """You are Adam, a voice-first Linux assistant with desktop, system, web, and productivity tools.

Speak naturally and briefly in plain text, without markdown or filler. After an action, say "Done." Never invent results.

Use tools when needed: weather and time require their tools; use web search for current facts and fetch_webpage for a specific URL. Use desktop tools for apps, windows, workspaces, browser tabs, clipboard, media, volume, and screenshots. Inspect visible content with a screenshot. Use injected desktop state for windows and resolve "it" from the previous turn. Retrieve desktop skill details with get_skill_context when needed.

Use tools for reminders, timers, calendars, notes, files, math, finance, system status, services, processes, and background jobs. Clarify ambiguous reminder times. Use Noctalia tools for Noctalia events and Remind tools only when asked for the Remind calendar. Gather unknown targets before acting; never guess.

Confirm destructive actions, including deleting files, killing processes, terminating apps, rebooting, or mass edits using ask_user_confirmation with command set to the exact shell command to run on confirmation (e.g. 'systemctl reboot' for restart, 'systemctl poweroff' for shutdown). Keep spoken confirmations under 10 words; put paths, IDs, addresses, and technical details in notification details, never in speech. If the user message starts with 'User confirmed:', do not ask confirmation again; execute the action immediately.

Use open_in_browser to search or open URLs and close_browser_tab to close a tab. Change silent mode only on explicit requests. For visual inspection, report what the screenshot shows rather than saying only "Done."""


def _is_visual_inspection_request(text: str) -> bool:
    """Recognize common requests to inspect visible app/window contents."""
    text = " ".join(text.lower().split())
    phrases = (
        "what's in", "what is in", "what's happening", "what is happening",
        "what's going on", "what is going on", "what does it say", "what does that say",
        "what are they saying", "what's in the chat", "what is in the chat",
        "read the chat", "read this chat", "read my chat", "summarize the chat",
        "summarize this chat", "what's on my screen", "what is on my screen",
        "what's on the screen", "what is on the screen", "inspect the screen",
    )
    return any(phrase in text for phrase in phrases)


def _format_visual_spoken_answer(text: str) -> str:
    """Keep screenshot answers brief and safe for speech, even if the model ignores formatting guidance."""
    text = re.sub(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", text or "")
    text = re.sub(r"[`*_#~]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Don't feed foreign-script display names into the English TTS voice; they
    # can trigger the wrong language path and sound like spelling/letter names.
    text = re.sub(r'["“][^"”]*[^\x00-\x7f][^"”]*["”]', "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")
    # Screenshot replies are spoken summaries, not transcripts or reports.
    match = re.search(r"^(.+?[.!?])(?:\s|$)", text)
    if match:
        text = match.group(1)
    words = text.split()
    if len(words) > 24:
        text = " ".join(words[:24]).rstrip(" ,;:-") + "."
    return text

class AdamBrain:
    """The central ReAct autonomous agent loop driving tool execution and conversation."""
    def __init__(self, config, supervisor, probe, confirmation_mgr, tts_engine, arbiter=None, speculative_router=None):
        self.config = config
        self.supervisor = supervisor
        self.probe = probe
        self.confirmation = confirmation_mgr
        self.tts = tts_engine
        self.arbiter = arbiter
        self.speculative_router = speculative_router
        self.timer_mgr = TimerManager(
            tts_engine=self.tts,
            earcon_engine=getattr(self.arbiter, "earcon", None) if self.arbiter else None
        )
        self.reminder_mgr = ReminderManager()
        self.noctalia_calendar = NoctaliaCalendar()
        self.llm_client = UniversalLLMClient(config)
        self.skill_manager = SkillManager()
        self.custom_tool_mgr = CustomToolManager(config_path="config.yaml")
        self.system_prompt = self._build_system_prompt()
        self.messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        self._pending_screenshot: bytes | None = None

    def get_tools(self) -> list:
        """Returns canonical built-in tools (filtered by active desktop capabilities) plus user-defined custom tools."""
        supported_tools = [tool for tool in ADAM_TOOLS if is_tool_enabled(tool.name)]
        return supported_tools + self.custom_tool_mgr.get_canonical_tools()

    def _build_system_prompt(self) -> str:
        """Keep desktop skill details on demand instead of in every request."""
        return SYSTEM_PROMPT

    def _compact_history_for_new_turn(self):
        """Keep a few short dialogue turns; discard old tool payloads and desktop snapshots."""
        recent = []
        for message in self.messages[1:]:
            role = message.get("role")
            if role not in {"user", "assistant"} or message.get("tool_calls") or message.get("images"):
                continue
            content = str(message.get("content") or "").strip()
            if role == "user":
                content = re.sub(r"^\[Current Desktop State\].*?\[Local Time:[^\]]+\]\s*\n", "", content, flags=re.S)
            if not content:
                continue
            if len(content) > 500:
                content = content[:497].rstrip() + "..."
            recent.append({"role": role, "content": content})
        self.messages = [{"role": "system", "content": self.system_prompt}, *recent[-6:]]

    async def warmup(self):
        """Warms up the underlying LLM client."""
        await self.llm_client.warmup()

    async def process_user_utterance(self, user_text: str):
        """Processes a transcribed user prompt through the autonomous ReAct cycle."""
        print(f"\n[Adam] User said: \"{user_text}\"")
        import datetime
        now_local = datetime.datetime.now().astimezone()
        now_str = now_local.strftime("%I:%M %p %Z (UTC%z) on %A, %B %d, %Y")

        # Remove old tool payloads and desktop snapshots before adding fresh state.
        self._compact_history_for_new_turn()

        # Real-time desktop state prompt injection
        desktop_state = get_open_windows_prompt_context()
        user_prompt_content = f"[Current Desktop State]\n{desktop_state}\n\n[Local Time: {now_str}]\n{user_text}"
        self.messages.append({"role": "user", "content": user_prompt_content})

        # Run ReAct iteration loop (up to 4 tool hops)
        executed_calls = set()
        turn_completed_with_speech = False
        last_tool_output: str | None = None
        for hop in range(4):
            response = await self.llm_client.chat(self.messages, tools=self.get_tools())
            content = response.get("content", "")
            tool_calls = response.get("tool_calls") or []

            # 1. Speak assistant commentary if present (only when no tool calls are being dispatched)
            has_speech_tool = any(
                (tc.get("function", {}).get("name") if hasattr(tc, "get") else getattr(getattr(tc, "function", None), "name", "")) in ["ask_user_confirmation", "speak"]
                for tc in tool_calls
            )
            if content and not tool_calls and not has_speech_tool:
                if any(message.get("images") for message in self.messages):
                    # Vision answers tend to over-explain or emit markdown. Ask
                    # Qwen to rewrite the observation specifically for speech.
                    concise_messages = [*self.messages, {
                        "role": "user",
                        "content": (
                            "Rewrite your screenshot answer as one short spoken sentence of at most 18 words. "
                            "Plain text only: no markdown, bullets, lists, or introductory phrase. "
                            "Do not say usernames or foreign-script display names; summarize what the messages mean."
                        ),
                    }]
                    concise_response = await self.llm_client.chat(concise_messages, tools=[])
                    concise_content = concise_response.get("content", "")
                    content = _format_visual_spoken_answer(concise_content or content)
                print(f"[Adam] Response: {content}")
                await self.tts.speak_async(content)
                turn_completed_with_speech = True

            # 2. Record the assistant's turn in conversation history
            # (Crucial: must record tool_calls so LLMs understand subsequent tool results!)
            assistant_msg = {"role": "assistant", "content": content or ""}
            if tool_calls:
                formatted_calls = []
                for idx, tc in enumerate(tool_calls):
                    fn = tc.get("function", {}) if hasattr(tc, "get") else getattr(tc, "function", {})
                    fn_name = fn.get("name") if hasattr(fn, "get") else getattr(fn, "name", "")
                    fn_args = fn.get("arguments", {}) if hasattr(fn, "get") else getattr(fn, "arguments", {})
                    tc_id = tc.get("id", f"call_{hop}_{idx}") if hasattr(tc, "get") else getattr(tc, "id", f"call_{hop}_{idx}")
                    formatted_calls.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": (
                                json.dumps(fn_args)
                                if tc.get("_origin") == "text_fallback"
                                and self.llm_client.provider not in {"local", "ollama"}
                                and isinstance(fn_args, dict)
                                else fn_args
                            )
                        }
                    })
                assistant_msg["tool_calls"] = formatted_calls
            self.messages.append(assistant_msg)

            # 3. If no tools were called, the turn is complete
            if not tool_calls:
                break

            # Repetition detection: avoid re-running the exact same tool calls in consecutive hops
            import json
            call_signatures = []
            for tc in tool_calls:
                fn = tc.get("function", {}) if hasattr(tc, "get") else getattr(tc, "function", {})
                fn_name = fn.get("name") if hasattr(fn, "get") else getattr(fn, "name", "")
                fn_args = fn.get("arguments", {}) if hasattr(fn, "get") else getattr(fn, "arguments", {})
                arg_str = json.dumps(fn_args, sort_keys=True) if isinstance(fn_args, dict) else str(fn_args)
                call_signatures.append((fn_name, arg_str))

            if all(sig in executed_calls for sig in call_signatures):
                # Repetition loop detected! Break to synthesize final response
                break

            for sig in call_signatures:
                executed_calls.add(sig)

            # 4. Execute all tool calls
            executed_hop_results = []
            for idx, tc in enumerate(tool_calls):
                fn = tc.get("function", {}) if hasattr(tc, "get") else getattr(tc, "function", {})
                name = fn.get("name") if hasattr(fn, "get") else getattr(fn, "name", "")
                args = fn.get("arguments", {}) if hasattr(fn, "get") else getattr(fn, "arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}

                origin = tc.get("_origin", "native")
                print(f"[Adam] Tool call ({origin}): {name}({args})")
                if tc.get("_origin") == "text_fallback" and name not in TEXT_FALLBACK_READ_ONLY_TOOLS:
                    tool_output = (
                        f"Rejected text-form tool call '{name}': fallback tool calls are limited "
                        "to read-only tools. Please retry using the provider's structured tool-call format."
                    )
                    print(f"[Adam] {tool_output}")
                else:
                    tool_output = await self._execute_tool(name, args)
                last_tool_output = str(tool_output)
                print(f"[Adam] Tool result: {tool_output}")
                # For an inspection question, focusing a window is only setup.
                # Capture it immediately so the next model turn can answer from
                # pixels instead of incorrectly treating focus as completion.
                if (
                    name == "focus_window"
                    and _is_visual_inspection_request(user_text)
                    and str(tool_output).lower().startswith("focused ")
                ):
                    try:
                        self._pending_screenshot = await asyncio.to_thread(capture_screenshot)
                    except Exception as e:
                        print(f"[Adam] Screenshot after focusing window failed: {e}")
                if name in ["speak", "ask_user_confirmation"]:
                    return

                if self.custom_tool_mgr.has_tool(name) and self.custom_tool_mgr.tools[name].background:
                    turn_completed_with_speech = True
                    return

                executed_hop_results.append((name, args, tool_output))

                tc_id = tc.get("id", f"call_{hop}_{idx}") if hasattr(tc, "get") else getattr(tc, "id", f"call_{hop}_{idx}")
                # Format tool output for subsequent turns
                tool_resp_msg = self.llm_client.format_tool_response(
                    tool_call_id=tc_id,
                    tool_name=name,
                    result=str(tool_output)
                )
                self.messages.append(tool_resp_msg)

            if self._pending_screenshot is not None:
                self.messages.append({
                    "role": "user",
                    "content": (
                        "The current desktop screenshot is attached. Inspect it to answer the user's request. "
                        "Keep the answer to one brief spoken sentence in plain text. Do not use markdown, "
                        "lists, introductory phrases, or read out usernames/display names."
                    ),
                    "images": [self._pending_screenshot],
                })
                self._pending_screenshot = None

            # Deterministic fast-path for operational action tools: respond with "Done." without a second LLM hop
            ACTION_TOOLS = {
                "focus_window",
                "swap_windows",
                "workspace_control",
                "launch_application",
                "close_application",
                "media_control",
                "volume_control",
                "desktop_macro",
                "create_waynote",
                "append_waynote",
                "manage_waynote",
                "open_in_browser",
                "close_browser_tab",
            }
            def _is_successful_action(act_name: str, act_args: dict, act_output: str) -> bool:
                out_lower = act_output.lower()
                failure_keywords = [
                    "error", "failed", "could not", "not found",
                    "please specify", "not supported", "not available", "unknown",
                    "not installed", "unavailable", "rejected", "refused",
                    "permission denied", "timed out",
                ]
                if any(k in out_lower for k in failure_keywords):
                    return False
                if act_name in ACTION_TOOLS:
                    return True
                if act_name == "manage_clipboard" and act_args.get("action") == "write":
                    return True
                return False

            all_are_actions = executed_hop_results and all(
                (n in ACTION_TOOLS or (n == "manage_clipboard" and a.get("action") == "write"))
                for n, a, _ in executed_hop_results
            )
            if (
                all_are_actions
                and not _is_visual_inspection_request(user_text)
                and all(_is_successful_action(n, a, o) for n, a, o in executed_hop_results)
            ):
                is_user_ja = any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in user_text)
                fast_response = "完了しました。" if is_user_ja else "Done."
                print(f"[Adam] Response: {fast_response}")
                self.messages.append({"role": "assistant", "content": fast_response})
                await self.tts.speak_async(fast_response)
                turn_completed_with_speech = True
                break
            elif all_are_actions and not _is_visual_inspection_request(user_text):
                failure_names = ", ".join(
                    n for n, a, o in executed_hop_results
                    if not _is_successful_action(n, a, o)
                )
                response = "I couldn't complete that action."
                if failure_names:
                    response = f"I couldn't complete {failure_names.replace('_', ' ')}."
                print(f"[Adam] Response: {response}")
                self.messages.append({"role": "assistant", "content": response})
                await self.tts.speak_async(response)
                turn_completed_with_speech = True
                break

        # If turn finished without any spoken response, ask model for concise spoken answer
        if not turn_completed_with_speech:
            summary_response = await self.llm_client.chat(self.messages, tools=[])
            final_content = summary_response.get("content", "")
            if not final_content or not final_content.strip():
                # Qwen can occasionally return only a tool result/thinking with no
                # user-facing text. Give it one explicit, tool-free synthesis retry.
                retry_messages = [*self.messages, {
                    "role": "user",
                    "content": (
                        "Your previous answer was empty. Now respond to the user's original request "
                        "using the tool results above. Give only a concise, natural spoken answer; "
                        "do not repeat raw tool output or call another tool."
                    ),
                }]
                retry_response = await self.llm_client.chat(retry_messages, tools=[])
                final_content = retry_response.get("content", "")

            if not final_content or not final_content.strip():
                final_content = (
                    "I got the result, but couldn't summarize it just now."
                    if last_tool_output is not None
                    else "I couldn't generate a response just now."
                )

            print(f"[Adam] Response: {final_content}")
            self.messages.append({"role": "assistant", "content": final_content})
            await self.tts.speak_async(final_content)

    async def _execute_tool(self, name: str, args: dict) -> str:
        """Executes the requested tool action."""
        if self.speculative_router:
            hit, cached_result = await self.speculative_router.consume_speculative_result(name, args)
            if hit:
                return str(cached_result)

        if self.custom_tool_mgr.has_tool(name):
            return await self.custom_tool_mgr.execute(
                name=name,
                args=args,
                arbiter=self.arbiter,
                tts=self.tts,
                confirmation=self.confirmation
            )

        if name == "transcode_video":
            pattern = args.get("file_pattern", "*")
            codec = args.get("target_codec", "av1")
            return await self._handle_transcode(pattern, codec)

        elif name == "capture_screenshot":
            try:
                screenshot = await asyncio.to_thread(capture_screenshot)
            except Exception as e:
                return f"Could not capture a screenshot: {e}"
            for message in self.messages:
                message.pop("images", None)
            self._pending_screenshot = screenshot
            return "Captured the current desktop screenshot."

        elif name == "run_bash_command":
            cmd = args.get("command", "").strip()
            # Guard against invalid / conversational English phrases in ping
            if re.match(r"^ping\b", cmd):
                if "-c" not in cmd:
                    cmd = f"{cmd} -c 4"
                import shlex
                try:
                    tokens = shlex.split(cmd)
                    targets = []
                    i = 1
                    while i < len(tokens):
                        if tokens[i].startswith("-"):
                            if tokens[i] in ["-c", "-i", "-w", "-W", "-s", "-t", "-m", "-I", "-p"]:
                                i += 2
                                continue
                            else:
                                i += 1
                                continue
                        targets.append(tokens[i])
                        i += 1
                    if len(targets) > 1 or (targets and (not re.match(r"^[a-zA-Z0-9.-]+$", targets[0]) or ("." not in targets[0] and targets[0] != "localhost"))):
                        invalid_str = " ".join(targets)
                        return f"Error: '{invalid_str}' is not a valid domain name or IP address. Please provide a valid hostname (e.g., 'heise.de' or 'de.wikipedia.org') or IP address."
                except Exception:
                    pass

            # Auto-bound traceroute / tracepath to prevent hanging timeouts
            if re.match(r"^traceroute\b", cmd) and "-m" not in cmd:
                cmd = f"{cmd} -m 15 -w 1 -q 1"
            elif re.match(r"^tracepath\b", cmd) and "-m" not in cmd:
                cmd = f"{cmd} -m 10 -n"
            try:
                import signal
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True  # Put shell in dedicated process group (PGID == proc.pid)
                )
                buffer = bytearray()
                max_bytes = 64 * 1024  # 64 KB maximum memory cap prevents OOM on unbounded streams
                deadline = asyncio.get_event_loop().time() + 15.0

                while True:
                    time_left = deadline - asyncio.get_event_loop().time()
                    if time_left <= 0:
                        raise asyncio.TimeoutError()
                    try:
                        chunk = await asyncio.wait_for(proc.stdout.read(4096), timeout=min(time_left, 1.0))
                    except asyncio.TimeoutError:
                        if proc.returncode is not None:
                            break
                        if asyncio.get_event_loop().time() >= deadline:
                            raise
                        continue
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    if len(buffer) >= max_bytes:
                        # Output hit 64KB cap: stop reading and terminate process group immediately
                        buffer = buffer[:max_bytes]
                        buffer.extend(b"\n[... output capped at 64KB ...]")
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except (ProcessLookupError, PermissionError):
                            pass
                        break

                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.5)
                except asyncio.TimeoutError:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    await proc.wait()

                out_text = buffer.decode("utf-8", errors="replace")
                if len(out_text) > 12000:
                    out_text = out_text[:12000] + "\n[... output truncated ...]"
                return out_text
            except asyncio.TimeoutError:
                # Kill entire process group to ensure no orphaned pipeline descendants
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        proc.kill()
                    except Exception:
                        pass
                if hasattr(proc, "_transport") and proc._transport:
                    proc._transport.close()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except Exception:
                    pass
                return "Command timed out after 15 seconds."
            except Exception as e:
                return f"Error executing command: {e}"

        elif name == "start_background_job":
            import shlex
            job_name = args.get("job_name", "job")
            cmd_str = args.get("command", "")
            try:
                cmd = shlex.split(cmd_str)
            except Exception:
                cmd = cmd_str.split()
            pid = await self.supervisor.start_sandboxed_job(job_name, cmd, arbiter=self.arbiter)
            return f"Started background job '{job_name}' with PID {pid}."

        elif name == "find_files":
            import fnmatch
            from collections import Counter
            directory = Path(args.get("directory", "~/Downloads")).expanduser()
            pattern = args.get("pattern", "*").strip()
            if not directory.exists():
                return f"Directory '{directory}' does not exist."

            video_exts = {".mkv", ".mp4", ".avi", ".webm", ".mov", ".flv", ".m4v"}
            pat_lower = pattern.lower()
            if "video" in pat_lower or pat_lower in ["*.mkv", "*.mp4"]:
                matched = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in video_exts]
                by_dir = Counter([str(p.parent.relative_to(directory)) for p in matched])
                sample = [p.name for p in matched[:10]]
                return f"Found {len(matched)} video files across {len(by_dir)} folders (e.g., Wistoria: {by_dir.get('Wistoria', 0)} files). Sample files: {sample}"

            elif pat_lower in ["*", "*.*", "all", ""]:
                top_files = [p for p in directory.iterdir() if p.is_file()]
                top_dirs = [p for p in directory.iterdir() if p.is_dir()]
                all_videos = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in video_exts]
                top_files_sorted = sorted(top_files, key=lambda p: p.stat().st_mtime, reverse=True)
                recent_names = [p.name for p in top_files_sorted[:10]]
                return (
                    f"Directory '{directory.name}' contains {len(top_files)} top-level files and {len(top_dirs)} subfolders "
                    f"(including {len(all_videos)} video files in subfolders like 'Wistoria'). "
                    f"Top subfolders: {[p.name for p in top_dirs[:6]]}. "
                    f"Most recent files: {recent_names}"
                )
            else:
                clean_p = pattern.strip("*").lower()
                # Recursive case-insensitive match (supporting both substring and wildcard glob matching)
                matched = [
                    p for p in directory.rglob("*")
                    if p.is_file() and (clean_p in p.name.lower() or fnmatch.fnmatch(p.name.lower(), pat_lower))
                ]
                sample = [p.name for p in matched[:15]]
                return f"Found {len(matched)} files matching '{pattern}' (case-insensitive): {sample}"

        elif name == "media_control":
            action = args.get("action", "play-pause")
            return media_control(action)

        elif name == "get_weather":
            return await get_weather_report(args.get("location", ""))

        elif name == "list_applications":
            return list_applications(args.get("query", ""))

        elif name == "launch_application":
            return launch_application(args.get("app_name", ""), args.get("args", ""))

        elif name == "close_application":
            return close_application(args.get("app_name", ""))

        elif name == "list_windows":
            return list_windows()

        elif name == "focus_window":
            return focus_window(args.get("target", ""))

        elif name == "swap_windows":
            return swap_windows(args.get("window_one", ""), args.get("window_two", ""))

        elif name == "workspace_control":
            return workspace_control(
                action=args.get("action", "switch"),
                workspace=args.get("workspace", ""),
                target=args.get("target", "")
            )

        elif name == "desktop_macro":
            return execute_desktop_macro(args.get("macro", ""))

        elif name == "get_system_status":
            return get_system_status()

        elif name == "list_processes":
            return list_processes(args.get("sort_by", "cpu"), args.get("limit", 5))

        elif name == "kill_process":
            return kill_process(args.get("target", ""), args.get("force", False))

        elif name == "list_audio_devices":
            return list_audio_devices()

        elif name == "volume_control":
            return volume_control(args.get("action", "up"), args.get("level"))

        elif name == "get_now_playing":
            return get_now_playing()

        elif name == "web_search":
            return web_search(args.get("query", ""))

        elif name == "fetch_webpage":
            return fetch_webpage(args.get("url", ""))

        elif name == "manage_clipboard":
            return manage_clipboard(args.get("action", "read"), args.get("text", ""))

        elif name == "set_timer":
            return self.timer_mgr.set_timer(args.get("duration_seconds", 60), args.get("label", "timer"))

        elif name == "list_timers":
            return self.timer_mgr.list_timers()

        elif name == "cancel_timer":
            return self.timer_mgr.cancel_timer(args.get("timer_id", ""))

        elif name == "create_reminder":
            return self.reminder_mgr.create_reminder(
                args.get("message", ""), args.get("when", ""), args.get("repeat", "none")
            )

        elif name == "list_reminders":
            return self.reminder_mgr.list_reminders()

        elif name == "cancel_reminder":
            return self.reminder_mgr.cancel_reminder(args.get("reminder_id", ""))

        elif name == "show_calendar":
            return self.reminder_mgr.show_calendar(args.get("months", 1))

        elif name == "open_noctalia_calendar":
            return open_noctalia_calendar()

        elif name == "create_noctalia_event":
            return self.noctalia_calendar.create_event(
                args.get("title", ""), args.get("when", ""),
                args.get("duration_minutes", 15), args.get("alarm_minutes", 10),
                args.get("repeat", "none"),
            )

        elif name == "list_noctalia_events":
            return self.noctalia_calendar.list_events()

        elif name == "cancel_noctalia_event":
            return self.noctalia_calendar.cancel_event(args.get("event_id", ""))

        elif name == "create_waynote":
            return create_waynote(
                content=args.get("content", ""),
                title=args.get("title", ""),
                color=args.get("color", "yellow")
            )

        elif name == "append_waynote":
            return append_waynote(
                content=args.get("content", ""),
                target=args.get("target", "")
            )

        elif name == "list_waynotes":
            return list_waynotes()

        elif name == "manage_waynote":
            return manage_waynote(args.get("action", "show-all"))

        elif name == "open_in_browser":
            return open_in_browser(args.get("query_or_url", ""))

        elif name == "close_browser_tab":
            return close_browser_tab(args.get("target", "browser"))


        elif name == "get_financial_quote":
            return get_financial_quote(args.get("symbol", ""))

        elif name == "calculate_math":
            return calculate_math(args.get("expression", ""))

        elif name == "check_system_updates":
            return check_system_updates()

        elif name == "manage_service":
            return manage_service(args.get("action", "status"), args.get("service_name", ""))

        elif name == "git_repo_status":
            return git_repo_status(args.get("repo_path", ""))

        elif name == "docker_container_status":
            return docker_container_status()

        elif name == "ask_user_confirmation":
            question = args.get("question", "Confirm action?")
            summary = args.get("summary", "")
            details = args.get("details", "")
            cmd = args.get("command", "")
            await self.confirmation.request_confirmation({
                "type": "command" if cmd else "general",
                "summary": summary,
                "details": details,
                "command": cmd
            }, question)
            return "Confirmation requested from user. Execution is paused waiting for user's verbal confirmation."

        elif name == "show_desktop_notification":
            title = args.get("title", "Adam")
            message = args.get("message", "")
            urgency = args.get("urgency", "normal")
            return await asyncio.to_thread(show_desktop_notification, str(title), str(message), urgency)

        elif name == "speak":
            msg = args.get("message", "")
            await self.tts.speak_async(msg)
            return "Message spoken."

        elif name == "enable_silent_mode":
            self.tts.engine = "silent"
            print("[TTS] Silent mode enabled by voice command.", flush=True)
            return "Silent mode enabled. Future responses will appear as desktop notifications."

        elif name == "disable_silent_mode":
            configured_engine = getattr(getattr(self.config, "tts", None), "engine", "silent")
            if str(configured_engine).lower() == "silent":
                return "Silent mode is the configured default, so it cannot be disabled by voice command."
            self.tts.engine = configured_engine
            print(f"[TTS] Silent mode disabled; restored configured engine '{configured_engine}'.", flush=True)
            return "Silent mode disabled. Spoken responses are restored."

        elif name == "get_current_time":
            loc = args.get("location", "local") if args else "local"
            return self._resolve_time(loc)

        elif name == "organize_files":
            return await self._handle_organize_files(args)

        elif name == "list_skills":
            skills = self.skill_manager.list_skills()
            active = self.skill_manager.detect_desktop_environment()
            names = [f"{s['id']}{' (active)' if s['is_active'] else ''}" for s in skills]
            return f"Installed desktop skills ({len(skills)} total, active: {active}): {', '.join(names)}."

        elif name == "get_skill_context":
            s_name = args.get("skill_name", "active")
            if not s_name or s_name.lower() in ["active", "current"]:
                return self.skill_manager.get_active_de_context()
            content = self.skill_manager.load_skill(s_name)
            if content:
                return f"Skill '{s_name}':\n{content[:1500]}"
            return f"Skill '{s_name}' not found. Available skills: {[s['id'] for s in self.skill_manager.list_skills()]}."

        return f"Unknown tool: {name}"

    async def _handle_transcode(self, pattern: str, target_codec: str) -> str:
        """Handles video transcode requests, file matching, and background spawning."""
        downloads = Path(self.config.execution.downloads_dir).expanduser()
        clean_pat = pattern.strip("*").lower()
        matches = []
        if clean_pat:
            matches = [p for p in downloads.rglob("*") if p.is_file() and clean_pat in p.name.lower()]

        # If no specific matches, check common video extensions recursively
        if not matches:
            for ext in [".mkv", ".mp4", ".avi", ".webm"]:
                matches.extend([p for p in downloads.rglob("*") if p.is_file() and p.suffix.lower() == ext])

        if not matches:
            msg = f"No video files matching '{pattern}' were found in Downloads."
            await self.tts.speak_async(msg)
            return msg

        # Select encoder parameters using hardware probe
        params = await self.probe.select_transcode_parameters(target_codec)
        encoder_args = params["args"]

        output_dir = Path(self.config.execution.workspace_dir).expanduser() / "transcoded"
        output_dir.mkdir(parents=True, exist_ok=True)

        count = len(matches)
        prompt_text = f"Found {count} files matching {pattern} in Downloads. Ready to transcode to {target_codec} using {params['encoder']}. Should I proceed?"

        # Ask confirmation for multi-file transcoding
        await self.confirmation.request_confirmation(
            action_payload={
                "type": "batch_transcode",
                "files": [str(p) for p in matches],
                "encoder_args": encoder_args,
                "output_dir": str(output_dir),
                "summary": f"Transcoding {count} files to {target_codec}"
            },
            prompt_text=prompt_text
        )

        return f"Found {count} files. Asked user confirmation before starting background transcode."

    def _resolve_time(self, location: str = "local") -> str:
        """Resolves current date/time for local system or any world location/timezone."""
        import datetime
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        if not location or location.lower() in ["local", "here", "system", "mine"]:
            now = datetime.datetime.now().astimezone()
            return f"Current local time: {now.strftime('%I:%M %p %Z on %A, %B %d, %Y')} ({now.strftime('%H:%M:%S')} 24h, UTC{now.strftime('%z')})."

        loc_clean = location.strip().lower()

        # Common city/country/region to IANA timezone aliases
        tz_aliases = {
            "japan": "Asia/Tokyo",
            "tokyo": "Asia/Tokyo",
            "uk": "Europe/London",
            "london": "Europe/London",
            "england": "Europe/London",
            "britain": "Europe/London",
            "france": "Europe/Paris",
            "paris": "Europe/Paris",
            "germany": "Europe/Berlin",
            "berlin": "Europe/Berlin",
            "italy": "Europe/Rome",
            "rome": "Europe/Rome",
            "spain": "Europe/Madrid",
            "madrid": "Europe/Madrid",
            "china": "Asia/Shanghai",
            "beijing": "Asia/Shanghai",
            "shanghai": "Asia/Shanghai",
            "hong kong": "Asia/Hong_Kong",
            "taiwan": "Asia/Taipei",
            "taipei": "Asia/Taipei",
            "korea": "Asia/Seoul",
            "south korea": "Asia/Seoul",
            "seoul": "Asia/Seoul",
            "india": "Asia/Kolkata",
            "delhi": "Asia/Kolkata",
            "mumbai": "Asia/Kolkata",
            "singapore": "Asia/Singapore",
            "australia": "Australia/Sydney",
            "sydney": "Australia/Sydney",
            "melbourne": "Australia/Melbourne",
            "brisbane": "Australia/Brisbane",
            "new zealand": "Pacific/Auckland",
            "auckland": "Pacific/Auckland",
            "california": "America/Los_Angeles",
            "los angeles": "America/Los_Angeles",
            "san francisco": "America/Los_Angeles",
            "seattle": "America/Los_Angeles",
            "pacific": "America/Los_Angeles",
            "pst": "America/Los_Angeles",
            "pdt": "America/Los_Angeles",
            "new york": "America/New_York",
            "eastern": "America/New_York",
            "est": "America/New_York",
            "edt": "America/New_York",
            "chicago": "America/Chicago",
            "central": "America/Chicago",
            "cst": "America/Chicago",
            "cdt": "America/Chicago",
            "denver": "America/Denver",
            "mountain": "America/Denver",
            "mst": "America/Denver",
            "mdt": "America/Denver",
            "hawaii": "Pacific/Honolulu",
            "alaska": "America/Anchorage",
            "utc": "UTC",
            "gmt": "GMT",
            "toronto": "America/Toronto",
            "vancouver": "America/Vancouver",
            "canada": "America/Toronto",
            "brazil": "America/Sao_Paulo",
            "sao paulo": "America/Sao_Paulo",
            "dubai": "Asia/Dubai",
            "uae": "Asia/Dubai"
        }

        target_tz_str = tz_aliases.get(loc_clean)
        if not target_tz_str:
            import zoneinfo
            available = zoneinfo.available_timezones()
            for tz in available:
                if tz.lower() == loc_clean or tz.lower().endswith(f"/{loc_clean}"):
                    target_tz_str = tz
                    break

        if not target_tz_str:
            target_tz_str = location.strip()

        try:
            tz = ZoneInfo(target_tz_str)
            now = datetime.datetime.now(tz)
            loc_display = location.title() if len(location) > 3 else location.upper()
            return f"Current time in {loc_display} ({target_tz_str}): {now.strftime('%I:%M %p %Z on %A, %B %d, %Y')} ({now.strftime('%H:%M:%S')} 24h)."
        except ZoneInfoNotFoundError:
            now = datetime.datetime.now().astimezone()
            return f"Could not find timezone for '{location}'. Local time is {now.strftime('%I:%M %p %Z')}. You can specify a city or IANA timezone like 'Asia/Tokyo' or 'America/New_York'."

    async def _handle_organize_files(self, args: dict) -> str:
        """Organizes files in a directory into clean subfolders by show name, extension, or type."""
        import shutil
        directory = Path(args.get("directory", "~/Downloads")).expanduser()
        group_by = args.get("group_by", "show")
        show_filter = args.get("show_name", "").strip().lower()
        dry_run = args.get("dry_run", False)

        if not directory.exists() or not directory.is_dir():
            return f"Directory '{directory}' does not exist or is not a folder."

        # Collect top-level files in directory
        files = [p for p in directory.iterdir() if p.is_file()]
        if not files:
            subdirs = [p for p in directory.iterdir() if p.is_dir()]
            return f"No files found directly in '{directory.name}'. Found {len(subdirs)} subfolders: {[p.name for p in subdirs[:5]]}."

        groups = {}
        for f in files:
            if group_by == "show":
                stem = f.stem
                stem = re.sub(r"^\s*\[[^\]]+\]\s*", "", stem)
                stem = re.sub(r"\s*\[[^\]]+\]\s*", "", stem)
                stem = re.sub(r"\s*\([^)]*\)\s*", "", stem)
                m = re.split(r"(\s+-\s+\d+|\s+S\d+|\s+Season\s+\d+|\.S\d+E\d+|\s+\d+th\s+Season|\s+2nd\s+Season|\s+3rd\s+Season|\s+1st\s+Season|_S\d+E\d+)", stem, flags=re.IGNORECASE)
                s = m[0].strip(" -_") if m and len(m) > 1 else stem.strip()
                if "." in s and " " not in s: s = s.replace(".", " ")
                if "_" in s and " " not in s: s = s.replace("_", " ")

                canonical_shows = [
                    "Link Click", "Mushoku Tensei", "Tensei Shitara Slime Datta Ken",
                    "Tsue to Tsurugi no Wistoria", "Quanzhi Fashi", "Rick and Morty", "City Of God"
                ]
                norm_s = s
                for c in canonical_shows:
                    if c.lower() in s.lower():
                        norm_s = c
                        break
                folder_name = norm_s
            elif group_by == "type":
                ext = f.suffix.lower()
                if ext in [".mkv", ".mp4", ".avi", ".webm", ".mov"]:
                    folder_name = "Videos"
                elif ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]:
                    folder_name = "Audio"
                elif ext in [".pdf", ".docx", ".txt", ".md", ".csv"]:
                    folder_name = "Documents"
                elif ext in [".zip", ".tar.gz", ".tar", ".gz", ".7z", ".rar"]:
                    folder_name = "Archives"
                elif ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                    folder_name = "Images"
                else:
                    folder_name = "Other"
            else:
                ext = f.suffix.lstrip(".").upper() or "Unknown"
                folder_name = ext

            if show_filter and show_filter not in folder_name.lower():
                continue

            # Ensure folder_name is valid, non-empty, and sanitized
            folder_name = folder_name.strip(" -_.")
            if not folder_name:
                folder_name = "Unsorted"

            # Avoid collision if an existing file in the root has the exact same name as the target folder
            target_candidate = directory / folder_name
            if target_candidate.exists() and not target_candidate.is_dir():
                folder_name = f"{folder_name}_folder"

            groups.setdefault(folder_name, []).append(f)

        if not groups:
            return f"No matching files found to organize in '{directory.name}'."

        total_files = sum(len(flist) for flist in groups.values())

        if dry_run:
            summary_list = [f"{name} ({len(flist)} files)" for name, flist in groups.items()]
            return f"Dry-run organization plan for '{directory.name}': {', '.join(summary_list)}."

        plan_dict = {str(directory / folder_name): [str(p) for p in flist] for folder_name, flist in groups.items()}
        shows_desc = ", ".join(list(groups.keys())[:3])
        if len(groups) > 3:
            shows_desc += f", and {len(groups) - 3} more"

        question = f"Found {total_files} files across {len(groups)} categories ({shows_desc}) in {directory.name}. Should I proceed to move them into organized subfolders?"
        summary = f"Organize {total_files} files into {len(groups)} folders in {directory.name}"

        await self.confirmation.request_confirmation(
            action_payload={
                "type": "organize_files",
                "plan": plan_dict,
                "summary": summary
            },
            prompt_text=question
        )
        return "Confirmation requested from user. Execution is paused waiting for user's verbal confirmation."
