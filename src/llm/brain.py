import os
import re
import glob
import subprocess
import asyncio
import re
from pathlib import Path
from src.llm.provider import UniversalLLMClient
from src.llm.tools import SHIN_TOOLS
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

SYSTEM_PROMPT = """You are Shin, an autonomous local voice-activated personal assistant on Linux.
You have full access to the user's system to launch apps, manage windows, control media and volume, check system status, monitor processes, check weather, search the web, manage timers and persistent reminders/calendar entries, and interact via natural voice.

RULES:
1. PLAIN TEXT ONLY - ABSOLUTELY NO MARKDOWN:
   - Output purely plain conversational text.
   - NEVER use ANY markdown formatting whatsoever: no asterisks (* or **), no bold, no italics, no bullet points (- or *), no numbered lists (1.), no headers or hashtags (#), no backticks (` or ```), no code fences, no markdown links, no tables, and no emojis.
   - All responses must sound fluid and natural when read aloud by a text-to-speech engine. Format answers in standard, clean conversational sentences rather than lists or code snippets.
2. Never output simulated XML or pseudotags like <tool_response>, <tool_call>, or code blocks.
3. ULTRA-CONCISE RESPONSES (NO YAPPING):
   - For operational action commands (such as focusing a window, moving a window, switching workspaces, launching or closing an app, pausing/playing media, changing volume, copying to clipboard): respond with ONLY 'Done.'
   - Never repeat the user's command or add conversational filler like 'I have moved Spotify to Workspace 1 for you' or 'Sure, focusing Edge now'.
   - For observational remarks, side comments, or acknowledgments where no action or information is requested: respond with at most 1–3 words (e.g. 'Got it.' or 'Understood.') or remain silent. NEVER output conversational filler like 'Okay, I see. Let me know if you need anything else moved or managed' or 'How can I help you today?'.
4. Real-Time Weather:
   - When asked about weather (locally or for any city/location), ALWAYS use the `get_weather` tool. Never claim that you lack real-time weather information.
5. Applications & Desktop Management:
   - To search something in the browser or open a website/URL (e.g. 'search for X in my browser', 'use my browser to search X', 'open youtube in browser'): ALWAYS use `open_in_browser(query_or_url='...')`. Do not just focus the browser window.
   - To close the current browser tab (e.g. 'close this tab', 'close browser tab', 'close the tab in Edge'): ALWAYS use `close_browser_tab(target='...')`. This focuses the browser and triggers Ctrl+W.
   - To launch apps or games (e.g. Discord, Firefox, Spotify, Steam, terminal, Zed, games): Use `launch_application`.
   - To list installed apps: Use `list_applications`.
   - To list or manage open windows: Use `list_windows`, `focus_window`, or `close_application`.
   - To switch workspaces or move windows to workspaces:
     - Switch workspace: `workspace_control(action='switch', workspace='<id>')` (or `workspace='back'` / `'previous'`)
     - Move window to workspace: `workspace_control(action='move', workspace='<id>', target='<app or window>')`
     - Move to current workspace: `workspace_control(action='move', workspace='current', target='<app or window>')`
     - Move window back / undo: `workspace_control(action='move', workspace='back', target='<app or window>')`
   - To read or write clipboard: Use `manage_clipboard`.
6. Media & Audio:
   - To identify current song/podcast: Use `get_now_playing`.
   - To control playback: Use `media_control`.
   - To adjust volume or mute: Use `volume_control`.
   - To list audio devices: Use `list_audio_devices`.
7. Hardware & System Health:
   - To check CPU, RAM, disk, and GPU status: Use `get_system_status`.
   - To check or kill processes: Use `list_processes` or `kill_process`.
8. Web Search & Information:
   - When asked facts, news, definitions, or general questions: Use `web_search`.
   - To fetch and read the content of a specific webpage, article, or URL: Use `fetch_webpage(url='...')`.
9. Timers & Time:
   - To set, list, or cancel timers: Use `set_timer`, `list_timers`, and `cancel_timer`.
   - For Google Calendar questions, use the config-defined `google_calendar_agenda` tool; add events with `google_calendar_quick_add`.
   - `show_calendar` is specifically the legacy Remind month-grid. Use it only when the user explicitly asks for the Remind calendar. For Shin reminders, use `create_reminder`, `list_reminders`, and `cancel_reminder`.
   - For an event or reminder that should appear in Noctalia, use `create_noctalia_event`; manage these with `list_noctalia_events` and `cancel_noctalia_event`. Use `open_noctalia_calendar` to show the calendar.
   - Reminders use local system time. If the date or time is ambiguous, clarify before creating the reminder.
   - When asked for the time or date: Use `get_current_time`.
   - For desktop sticky notes (Waynote):
     - To create a note with text, a task, or a checklist: Use `create_waynote(content='...', title='...', color='...')`. Colors: yellow, green, pink, purple, blue, orange, gray.
     - To append to an existing note: Use `append_waynote(content='...', target='...')`.
     - To list or read notes: Use `list_waynotes()`.
     - To toggle, show, or hide notes: Use `manage_waynote(action='show-all' | 'hide-all' | 'toggle' | 'new')`.
10. File Management:
    - Search files recursively with `find_files`. Organize media and series with `organize_files`.
11. Shell & Background Execution:
    - Run quick shell inspection with `run_bash_command`.
    - For heavy background processing (like compiling or transcoding), use `start_background_job` or `transcode_video`.
12. Financial & Math:
    - To check crypto (Bitcoin, Ethereum, etc.) or stock prices (NVDA, AAPL, etc.): Use `get_financial_quote`.
    - To perform calculations, arithmetic, or unit conversions: Use `calculate_math`.
13. Development & System Services:
    - To check pending OS/package updates: Use `check_system_updates`.
    - To check or restart systemd services: Use `manage_service`.
    - To check git repositories: Use `git_repo_status`.
    - To check Docker containers: Use `docker_container_status`.
14. Destructive Tasks:
    - For destructive actions (file deletion, process termination), call `ask_user_confirmation`.
15. Spoken Output:
    - Return concise natural plain text without any markdown formatting, or use the `speak` tool.
16. Modular Desktop Skills:
    - You have modular skills detailing capabilities and CLI dispatchers for various desktop environments and window managers (Hyprland, Sway, i3, KDE Plasma, GNOME, COSMIC).
    - When executing desktop commands, use the exact CLI syntax and dispatchers detailed in the active desktop environment skill.
    - You can inspect or search other desktop skills using `list_skills` and `get_skill_context`.
17. Pronoun & Context Resolution:
    - When the user refers to 'it', 'that', 'the app', or 'the window' (e.g., 'move it back', 'close it', 'focus on it'), 'it' refers to the application or window acted on or mentioned in the immediate previous turn.
    - To reorder two tiled windows on the current workspace, use `swap_windows` with both app/window names when supported by the active compositor. This swaps their positions; it does not move them between workspaces.
    - To trigger desktop actions or compositor macros (such as overview, show desktop, grid, toggle floating, fullscreen, night mode, or lock), use `desktop_macro` with the macro name.
18. Real-Time Injected Desktop State:
    - You are continuously provided with [Current Desktop State] containing the active workspace, currently focused window, and every open window with its workspace ID, application class, and address.
    - When asked what apps or windows are open, answer immediately using this injected state without needing to call `list_windows`.
    - When asked to close or focus an app, use the [Current Desktop State] to know whether it is open, which workspace it is on, and its exact class or address.
    - If the user says 'close it' or 'close this', close the currently focused window shown in [Current Desktop State]. If an app the user wants to close is not open, state that it is not open.
    - Distinguish app control from app inspection. If the user asks what is in a chat, what a message says, what is happening in an app, or asks you to read/summarize visible content, that is an inspection request—not a request merely to focus the window. Focus the named app if needed, then inspect a screenshot and answer what you can actually see. Never answer only 'Done.' for an inspection request.
19. Sequential Multi-Step Tool Execution:
    - You can and should execute tools in sequence across multiple hops!
    - If a task requires information you do not have (e.g. finding a specific server, IP address, file path, process ID, or web URL before acting on it):
      1. First call an informational tool (`web_search`, `find_files`, `list_processes`, or `run_bash_command` with `curl`/`dig`) to find the required target or data.
      2. In the next turn, you will receive the tool result in the conversation history.
      3. Then immediately call the subsequent operational tool (e.g. `run_bash_command` with `ping`, `kill_process`, `manage_service`, etc.) using the exact target found!
      4. Never guess fake local IPs (like 127.0.0.1, 192.168.1.1, or 8.8.8.8) when asked for specific external entities; look them up first.
20. Visual Desktop Inspection:
    - When the user asks what is visible on screen or asks you to inspect the current UI, call `capture_screenshot` and base your answer on the attached image. For app-specific inspection, focus that app first if needed, then capture the screenshot. The screenshot is sent to the configured local vision model.
"""


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

class ShinBrain:
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
        supported_tools = [tool for tool in SHIN_TOOLS if is_tool_enabled(tool.name)]
        return supported_tools + self.custom_tool_mgr.get_canonical_tools()

    def _build_system_prompt(self) -> str:
        """Injects active desktop environment skill context into the core system prompt."""
        active_skill_context = self.skill_manager.get_active_de_context()
        if active_skill_context:
            return f"{SYSTEM_PROMPT}\n\n{active_skill_context}"
        return SYSTEM_PROMPT

    async def warmup(self):
        """Warms up the underlying LLM client."""
        await self.llm_client.warmup()

    async def process_user_utterance(self, user_text: str):
        """Processes a transcribed user prompt through the autonomous ReAct cycle."""
        print(f"\n[Shin] User said: \"{user_text}\"")
        import datetime
        now_local = datetime.datetime.now().astimezone()
        now_str = now_local.strftime("%I:%M %p %Z (UTC%z) on %A, %B %d, %Y")

        # Real-time desktop state prompt injection
        desktop_state = get_open_windows_prompt_context()
        user_prompt_content = f"[Current Desktop State]\n{desktop_state}\n\n[Local Time: {now_str}]\n{user_text}"
        self.messages.append({"role": "user", "content": user_prompt_content})

        # Bound context history (expanded to 100 messages to match 16k context window)
        if len(self.messages) > 100:
            self.messages = [self.messages[0]] + self.messages[-99:]

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
                print(f"[Shin] Response: {content}")
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
                            "arguments": fn_args
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

                print(f"[Shin] Tool call: {name}({args})")
                tool_output = await self._execute_tool(name, args)
                last_tool_output = str(tool_output)
                print(f"[Shin] Tool result: {tool_output}")
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
                        print(f"[Shin] Screenshot after focusing window failed: {e}")
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
                    "please specify", "not supported", "not available", "unknown"
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
                print(f"[Shin] Response: {fast_response}")
                self.messages.append({"role": "assistant", "content": fast_response})
                await self.tts.speak_async(fast_response)
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

            print(f"[Shin] Response: {final_content}")
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
            cmd = args.get("command", "")
            await self.confirmation.request_confirmation({
                "type": "command" if cmd else "general",
                "summary": summary,
                "command": cmd
            }, question)
            return "Confirmation requested from user. Execution is paused waiting for user's verbal confirmation."

        elif name == "speak":
            msg = args.get("message", "")
            await self.tts.speak_async(msg)
            return "Message spoken."

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
