from pydantic import BaseModel, Field
from typing import Any

class CanonicalTool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict:
        """For Ollama, Groq, and OpenAI compatible endpoints."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def to_gemini(self) -> dict:
        """For Google Gemini function declarations."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    def to_anthropic(self) -> dict:
        """For Anthropic Claude tools."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }

# Define the standard Adam tools
ADAM_TOOLS: list[CanonicalTool] = [
    CanonicalTool(
        name="capture_screenshot",
        description="Captures the current desktop and attaches the screenshot so you can inspect visible applications, text, and UI. Use when the user asks what is on screen or asks you to inspect a screenshot.",
        parameters={"type": "object", "properties": {}}
    ),
    CanonicalTool(
        name="run_bash_command",
        description="Executes a quick, synchronous shell command inside an isolated container (e.g. ls, du, file inspection).",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The exact shell command line string to run"}
            },
            "required": ["command"]
        }
    ),
    CanonicalTool(
        name="start_background_job",
        description="Spawns a long-running background task in an isolated process session (e.g. ffmpeg transcode, make, heavy processing).",
        parameters={
            "type": "object",
            "properties": {
                "job_name": {"type": "string", "description": "Short identifier for the job, e.g. 'transcode_slime'"},
                "command": {"type": "string", "description": "The command line string to execute in the background"}
            },
            "required": ["job_name", "command"]
        }
    ),
    CanonicalTool(
        name="transcode_video",
        description="Transcodes one or more video files in the Downloads folder using hardware or optimized SVT-AV1 CPU encoding.",
        parameters={
            "type": "object",
            "properties": {
                "file_pattern": {"type": "string", "description": "Filename pattern to match, e.g. '*slime*tensei*' or full path"},
                "target_codec": {"type": "string", "description": "Target video codec, e.g. 'av1' or 'hevc'"}
            },
            "required": ["file_pattern"]
        }
    ),
    CanonicalTool(
        name="find_files",
        description="Finds files in a directory matching a glob or substring pattern (case-insensitive and recursive).",
        parameters={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Target directory (e.g. '~/Downloads' or '~/workspace')"},
                "pattern": {"type": "string", "description": "Pattern or keyword to match case-insensitively (e.g. '*slime*', '*.pdf', or 'delilah')"}
            },
            "required": ["directory", "pattern"]
        }
    ),
    CanonicalTool(
        name="media_control",
        description="Controls media playback using playerctl (play, pause, next, previous).",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["play-pause", "play", "pause", "next", "previous", "volume-up", "volume-down"],
                    "description": "Playback control action"
                }
            },
            "required": ["action"]
        }
    ),
    CanonicalTool(
        name="ask_user_confirmation",
        description="Stops execution and asks the user for explicit verbal confirmation before running heavy or destructive tasks (like file deletion or killing processes). Spoken question MUST be ultra-concise (under 10 words, no technical dumps or hex addresses). Technical details, process lists, or file paths are flashed on screen as a desktop notification instead of spoken.",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Ultra-concise spoken verbal question under 10 words (e.g. 'Kill 5 Alacritty processes?'). NEVER include hex addresses (0x...), PIDs, or technical dumps in the spoken question."},
                "summary": {"type": "string", "description": "Brief summary of the pending action (e.g. 'Kill all Alacritty terminal processes')"},
                "details": {"type": "string", "description": "Optional technical details, hex addresses, process lists, or file paths to flash on screen as a desktop notification instead of speaking aloud"},
                "command": {"type": "string", "description": "The exact shell command to run automatically once the user says yes (e.g. 'kill_process target=\"Alacritty\" force=true')"}
            },
            "required": ["question", "summary"]
        }
    ),
    CanonicalTool(
        name="show_desktop_notification",
        description="Displays a visual desktop notification popup on the user's screen using notify-send. Use to flash alerts, technical details, status updates, or information on screen without speaking it out loud.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Notification title"},
                "message": {"type": "string", "description": "Notification body text to flash on the screen"},
                "urgency": {
                    "type": "string",
                    "enum": ["low", "normal", "critical"],
                    "description": "Urgency level of the notification (default: 'normal')"
                }
            },
            "required": ["title", "message"]
        }
    ),
    CanonicalTool(
        name="speak",
        description="Speaks a conversational message to the user. The message must be strictly plain text with no markdown, asterisks, bullet points, or formatting.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The plain text message to speak (no markdown, no bullets, no code blocks)"}
            },
            "required": ["message"]
        }
    ),
    CanonicalTool(
        name="enable_silent_mode",
        description="Switches Adam's spoken responses to desktop notifications. Call only when the user explicitly asks for 'silent mode' or 'notification mode'; do not call for 'shut up', 'quiet', or other requests to stop the current speech.",
        parameters={"type": "object", "properties": {}}
    ),
    CanonicalTool(
        name="disable_silent_mode",
        description="Restores the configured silent_restore_engine after silent mode. Call only when the user explicitly asks to disable or leave silent mode.",
        parameters={"type": "object", "properties": {}}
    ),
    CanonicalTool(
        name="get_current_time",
        description="Gets the current date and time for any city, country, region, timezone (e.g. 'Japan', 'Tokyo', 'London', 'Paris', 'New York', 'California'), or local system time.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City, country, or timezone name (e.g. 'Japan', 'Tokyo', 'London', 'US/Pacific', 'UTC', or 'local'). Defaults to 'local'."
                }
            }
        }
    ),
    CanonicalTool(
        name="organize_files",
        description="Organizes files in a folder into subfolders grouped by TV/anime show name, category/extension, or moves a specific show into its own folder.",
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The directory containing the files to organize (e.g. '~/Downloads/Wistoria' or '~/Downloads')"
                },
                "group_by": {
                    "type": "string",
                    "enum": ["show", "type", "extension"],
                    "description": "How to group files: 'show' (by parsed TV/anime show name), 'type' (Videos, Documents, Audio, etc.), or 'extension'."
                },
                "show_name": {
                    "type": "string",
                    "description": "Optional specific show name to organize (e.g. 'Link Click', 'Slime', 'Mushoku Tensei'). If specified, only moves files for this show."
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, only previews the planned groupings without moving files. Defaults to false."
                }
            },
            "required": ["directory"]
        }
    ),
    CanonicalTool(
        name="get_weather",
        description="Gets real-time weather, current temperature, feels like, humidity, wind, and today's forecast for any city or current local area.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or location name (e.g. 'Tokyo', 'London', 'Chicago', 'Paris', or empty for local area)"
                }
            }
        }
    ),
    CanonicalTool(
        name="list_applications",
        description="Lists installed desktop applications, software, and games across the system, optionally filtered by keyword.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional search term to filter applications (e.g. 'browser', 'game', 'discord', 'editor')"
                }
            }
        }
    ),
    CanonicalTool(
        name="launch_application",
        description="Launches any installed application, desktop tool, or game (e.g. Discord, Firefox, Spotify, Steam, terminal, Zed, games) by name.",
        parameters={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "The application or game name to launch (e.g. 'Discord', 'Firefox', 'Spotify', 'Steam', 'Alacritty', 'Zed', 'Cyberpunk 2077')"
                },
                "args": {
                    "type": "string",
                    "description": "Optional command line arguments or URLs to pass to the application"
                }
            },
            "required": ["app_name"]
        }
    ),
    CanonicalTool(
        name="close_application",
        description="Closes an application or its window cleanly.",
        parameters={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name of the application or window to close (e.g. 'Spotify', 'Firefox', 'Discord')"
                }
            },
            "required": ["app_name"]
        }
    ),
    CanonicalTool(
        name="list_windows",
        description="Lists all currently open desktop windows, their titles, applications, and workspace numbers.",
        parameters={
            "type": "object",
            "properties": {}
        }
    ),
    CanonicalTool(
        name="focus_window",
        description="Brings an open window to the front by matching its title or application name.",
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Window title or application class to focus (e.g. 'Firefox', 'Discord', 'Alacritty')"
                }
            },
            "required": ["target"]
        }
    ),
    CanonicalTool(
        name="swap_windows",
        description="Swaps the on-screen positions of two open windows on the current workspace. Use this to reorder tiled windows, such as putting the browser where the terminal is. This swaps positions; it does not move windows between workspaces.",
        parameters={
            "type": "object",
            "properties": {
                "window_one": {"type": "string", "description": "First window title or application name"},
                "window_two": {"type": "string", "description": "Second window title or application name"},
            },
            "required": ["window_one", "window_two"],
        }
    ),
    CanonicalTool(
        name="desktop_macro",
        description="Executes a desktop environment or window manager macro or action (e.g. 'overview', 'show_desktop', 'grid', 'toggle_floating', 'fullscreen', 'lock', 'night_mode', or user-defined custom macros).",
        parameters={
            "type": "object",
            "properties": {
                "macro": {
                    "type": "string",
                    "description": "Name of the desktop macro to execute (e.g. 'overview', 'show_desktop', 'grid', 'toggle_floating', 'fullscreen', 'lock', 'night_mode', or user-defined macro name)"
                }
            },
            "required": ["macro"]
        }
    ),
    CanonicalTool(
        name="workspace_control",
        description="Switches the active workspace or moves a specific window or active window to a workspace.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["switch", "move"],
                    "description": "Whether to 'switch' active workspace or 'move' a window to workspace"
                },
                "workspace": {
                    "type": "string",
                    "description": "Target workspace number (e.g. '1', '2', '3'), workspace name, or 'current'/'active' to move a window to the user's currently focused workspace."
                },
                "target": {
                    "type": "string",
                    "description": "Optional window title, application name, or class to move (e.g. 'Spotify', 'Discord', 'browser'). If omitted, moves the currently active window."
                }
            },
            "required": ["action", "workspace"]
        }
    ),
    CanonicalTool(
        name="get_system_status",
        description="Queries real-time hardware health: CPU load, RAM usage, root disk space, and all detected GPU temperatures and VRAM usage.",
        parameters={
            "type": "object",
            "properties": {}
        }
    ),
    CanonicalTool(
        name="list_processes",
        description="Lists top running processes sorted by CPU or memory consumption.",
        parameters={
            "type": "object",
            "properties": {
                "sort_by": {
                    "type": "string",
                    "enum": ["cpu", "memory"],
                    "description": "Sort criteria: 'cpu' or 'memory'. Defaults to 'cpu'."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of processes to list (default 5)"
                }
            }
        }
    ),
    CanonicalTool(
        name="kill_process",
        description="Terminates a process by PID or process name.",
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "PID (e.g. '12345') or process name (e.g. 'firefox')"
                },
                "force": {
                    "type": "boolean",
                    "description": "If true, sends SIGKILL (-9). Defaults to false (SIGTERM)."
                }
            },
            "required": ["target"]
        }
    ),
    CanonicalTool(
        name="list_audio_devices",
        description="Lists all available audio output sinks and microphone capture sources on the system.",
        parameters={
            "type": "object",
            "properties": {}
        }
    ),
    CanonicalTool(
        name="volume_control",
        description="Controls audio playback volume, microphone mute, or speaker mute.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["up", "down", "set", "mute", "unmute", "toggle-mute", "mute-mic", "unmute-mic", "toggle-mic"],
                    "description": "Volume action to perform"
                },
                "level": {
                    "type": "integer",
                    "description": "Volume level percentage (0 to 100) when action is 'set', or step percentage for up/down"
                }
            },
            "required": ["action"]
        }
    ),
    CanonicalTool(
        name="get_now_playing",
        description="Gets the title, artist, album, and status of the currently playing music, podcast, or media across all players.",
        parameters={
            "type": "object",
            "properties": {}
        }
    ),
    CanonicalTool(
        name="web_search",
        description="Searches the web and Wikipedia for factual answers, news, definitions, and summaries.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, topic, or question to look up"
                }
            },
            "required": ["query"]
        }
    ),
    CanonicalTool(
        name="manage_clipboard",
        description="Reads the current system clipboard text or copies new text to the clipboard.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write"],
                    "description": "Whether to 'read' the clipboard or 'write' text to it"
                },
                "text": {
                    "type": "string",
                    "description": "The text to copy to the clipboard (required when action is 'write')"
                }
            },
            "required": ["action"]
        }
    ),
    CanonicalTool(
        name="set_timer",
        description="Sets an asynchronous countdown timer that plays an alert chime and speaks when completed.",
        parameters={
            "type": "object",
            "properties": {
                "duration_seconds": {
                    "type": "integer",
                    "description": "Timer duration in seconds (e.g. 60 for 1 minute, 900 for 15 minutes)"
                },
                "label": {
                    "type": "string",
                    "description": "Optional label or description for the timer (e.g. 'pizza', 'tea', 'laundry')"
                }
            },
            "required": ["duration_seconds"]
        }
    ),
    CanonicalTool(
        name="list_timers",
        description="Lists all currently active countdown timers and their remaining time.",
        parameters={
            "type": "object",
            "properties": {}
        }
    ),
    CanonicalTool(
        name="cancel_timer",
        description="Cancels an active timer by ID or matching label.",
        parameters={
            "type": "object",
            "properties": {
                "timer_id": {
                    "type": "string",
                    "description": "The timer ID or label keyword to cancel"
                }
            },
            "required": ["timer_id"]
        }
    ),
    CanonicalTool(
        name="create_reminder",
        description="Creates a persistent timed reminder in the user's editable Remind config. Use the user's local time and ISO format for when.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "What the reminder should say"},
                "when": {"type": "string", "description": "Local date/time in ISO format, e.g. 2026-09-28T09:00"},
                "repeat": {"type": "string", "enum": ["none", "daily", "weekly", "monthly", "yearly"], "description": "Repeat cadence; defaults to none"}
            },
            "required": ["message", "when"]
        }
    ),
    CanonicalTool(
        name="list_reminders",
        description="Lists upcoming reminders from all Remind files in the user's config directory.",
        parameters={"type": "object", "properties": {}}
    ),
    CanonicalTool(
        name="cancel_reminder",
        description="Cancels a Adam-created reminder using its ID. Reminders entered manually in Remind config files are left untouched.",
        parameters={
            "type": "object",
            "properties": {"reminder_id": {"type": "string", "description": "Reminder ID returned when it was created"}},
            "required": ["reminder_id"]
        }
    ),
    CanonicalTool(
        name="show_calendar",
        description="Shows the Remind calendar for the next one to three months, including entries from the user's editable Remind files.",
        parameters={
            "type": "object",
            "properties": {"months": {"type": "integer", "description": "Number of months to show, from 1 to 3"}}
        }
    ),
    CanonicalTool(
        name="open_noctalia_calendar",
        description="Opens the Calendar tab in Noctalia's Control Center using Noctalia IPC.",
        parameters={"type": "object", "properties": {}}
    ),
    CanonicalTool(
        name="create_noctalia_event",
        description="Creates a timed event in the Adam local calendar shown by Noctalia. Noctalia handles its event alarm.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title or reminder text"},
                "when": {"type": "string", "description": "Local date/time in ISO format, e.g. 2026-09-28T09:00"},
                "duration_minutes": {"type": "integer", "description": "Event length in minutes; defaults to 15"},
                "alarm_minutes": {"type": "integer", "description": "Minutes before the event to notify; defaults to 10; 0 means at event time"},
                "repeat": {"type": "string", "enum": ["none", "daily", "weekly", "monthly", "yearly"], "description": "Repeat cadence; defaults to none"}
            },
            "required": ["title", "when"]
        }
    ),
    CanonicalTool(
        name="list_noctalia_events",
        description="Lists upcoming Adam-created events in the Noctalia calendar.",
        parameters={"type": "object", "properties": {}}
    ),
    CanonicalTool(
        name="cancel_noctalia_event",
        description="Removes a Adam-created event from the Noctalia calendar by event ID.",
        parameters={
            "type": "object",
            "properties": {"event_id": {"type": "string", "description": "Event ID returned when it was created"}},
            "required": ["event_id"]
        }
    ),
    CanonicalTool(
        name="get_financial_quote",
        description="Gets real-time price and 24-hour change for cryptocurrencies (Bitcoin, Ethereum, Solana, etc.) or stock market equities/tickers (NVDA, AAPL, MSFT, etc.).",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Cryptocurrency name or stock ticker symbol (e.g. 'Bitcoin', 'BTC', 'NVDA', 'AAPL')"
                }
            },
            "required": ["symbol"]
        }
    ),
    CanonicalTool(
        name="calculate_math",
        description="Evaluates mathematical expressions, arithmetic, percentages, powers, square roots, trigonometric functions, or unit conversions.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g. '25 * 14', 'sqrt(144)', '15% of 250')"
                }
            },
            "required": ["expression"]
        }
    ),
    CanonicalTool(
        name="check_system_updates",
        description="Checks for available system and package updates on the distribution without acquiring database locks.",
        parameters={
            "type": "object",
            "properties": {}
        }
    ),
    CanonicalTool(
        name="manage_service",
        description="Checks the status or controls (restart, start, stop) a systemd user or system service (e.g. 'adam', 'pipewire', 'docker', 'sunadame').",
        parameters={
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "Name of the service (e.g. 'adam', 'pipewire', 'docker')"
                },
                "action": {
                    "type": "string",
                    "enum": ["status", "restart", "start", "stop"],
                    "description": "Action to perform (default 'status')"
                }
            },
            "required": ["service_name"]
        }
    ),
    CanonicalTool(
        name="git_repo_status",
        description="Summarizes the current git branch, uncommitted modifications, and status of a git repository.",
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Optional repository directory path (defaults to current directory)"
                }
            }
        }
    ),
    CanonicalTool(
        name="docker_container_status",
        description="Inspects status and health of local Docker containers.",
        parameters={
            "type": "object",
            "properties": {}
        }
    ),
    CanonicalTool(
        name="list_skills",
        description="Lists all modular desktop environment and window manager skills installed on the system (e.g. hyprland, sway, i3, kde_plasma, gnome, cosmic, generic_desktop).",
        parameters={
            "type": "object",
            "properties": {}
        }
    ),
    CanonicalTool(
        name="get_skill_context",
        description="Reads the detailed markdown skill guide and CLI reference for a specific desktop environment or window manager (e.g. 'hyprland', 'sway', 'i3', 'kde_plasma', 'gnome', 'cosmic', or 'active' for the current session).",
        parameters={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to read (e.g. 'hyprland', 'sway', 'i3', 'kde_plasma', 'gnome', 'cosmic', or 'active' for the current session)"
                }
            },
            "required": ["skill_name"]
        }
    ),
    CanonicalTool(
        name="create_waynote",
        description="Creates a Waynote desktop sticky note with specified text, title, and color. Renders directly on screen.",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text, checklist, or body content to write into the sticky note"
                },
                "title": {
                    "type": "string",
                    "description": "Optional title for the note"
                },
                "color": {
                    "type": "string",
                    "description": "Color of the sticky note",
                    "enum": ["yellow", "green", "pink", "purple", "blue", "orange", "gray"]
                }
            },
            "required": ["content"]
        }
    ),
    CanonicalTool(
        name="append_waynote",
        description="Appends a task, checklist item, or text to an existing sticky note (defaults to the most recent note).",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Item or text to append"
                },
                "target": {
                    "type": "string",
                    "description": "Optional title or keyword of the note to append to. If omitted, appends to the most recent note."
                }
            },
            "required": ["content"]
        }
    ),
    CanonicalTool(
        name="list_waynotes",
        description="Lists all desktop sticky notes and their contents so you can read them or tell the user what is on their notes.",
        parameters={
            "type": "object",
            "properties": {}
        }
    ),
    CanonicalTool(
        name="manage_waynote",
        description="Toggles or changes the display state of desktop sticky notes ('show-all', 'hide-all', 'toggle', 'new').",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["show-all", "hide-all", "toggle", "new"]
                }
            },
            "required": ["action"]
        }
    ),
    CanonicalTool(
        name="open_in_browser",
        description="Opens a website URL or performs a web search in the user's default browser (e.g. searching 'Santal 33', opening 'youtube.com', or any URL). Use whenever the user asks to search for something in their browser, look something up on the web in their browser, or open a website.",
        parameters={
            "type": "object",
            "properties": {
                "query_or_url": {
                    "type": "string",
                    "description": "Search query (e.g. 'Santal 33', 'RTX 5090 specs') or URL (e.g. 'https://youtube.com', 'reddit.com') to open in the default browser"
                }
            },
            "required": ["query_or_url"]
        }
    ),
    CanonicalTool(
        name="close_browser_tab",
        description="Focuses the browser and closes the current active tab using the Ctrl+W shortcut. Use whenever the user asks to close a tab, close the current tab, or close the browser tab.",
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Browser application to target (e.g. 'browser', 'edge', 'chrome', 'firefox'). Defaults to 'browser'."
                }
            }
        }
    ),
    CanonicalTool(
        name="fetch_webpage",
        description="Fetches and extracts clean, readable text from any webpage URL or article. Use whenever you need to read a webpage, inspect an article, read documentation, or get information from a specific link.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to fetch and read (e.g. 'https://en.wikipedia.org/wiki/AMD' or 'theverge.com')"
                }
            },
            "required": ["url"]
        }
    )
]
