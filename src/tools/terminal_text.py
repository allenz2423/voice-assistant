"""Read the focused terminal's text buffer without injecting input or changing focus.

Terminal buffers are private to their emulator. This module uses a native
read-only API when one is available, tmux capture for panes running under tmux,
and the desktop accessibility text interface as a generic fallback.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_TERMINAL_NAMES = (
    "alacritty", "kitty", "wezterm", "ghostty", "ptyxis", "kgx",
    "gnome-console", "gnome-terminal", "konsole", "blackbox-terminal", "blackbox", "deepin-terminal",
    "lxterminal", "mate-terminal", "qterminal", "xfce4-terminal", "tilix",
    "foot", "xterm", "urxvt", "rxvt", "terminator", "sakura", "st",
    "terminology", "tilda", "guake", "yakuake", "contour", "rio", "tabby",
    "iterm2", "terminal.app", "windows terminal",
)
_MAX_TOOL_CHARS = 24_000


@dataclass
class FocusedWindow:
    title: str
    app: str
    pid: int
    source: str


@dataclass
class TerminalDetection:
    ok: bool
    terminal: str | None
    app: str | None
    title: str | None
    pid: int | None
    backend: str | None
    available: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(args: list[str], *, timeout: float = 2.0, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _walk_focused_tree(node: dict[str, Any]) -> dict[str, Any] | None:
    if node.get("focused") is True and node.get("pid"):
        return node
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        if isinstance(child, dict):
            found = _walk_focused_tree(child)
            if found:
                return found
    return None


def _focused_window() -> FocusedWindow | None:
    """Find the currently focused top-level app without using keyboard input."""
    # Hyprland reports the app class and owning PID directly, including on Wayland.
    result = _run(["hyprctl", "activewindow", "-j"])
    if result and result.returncode == 0:
        try:
            item = json.loads(result.stdout or "{}")
            pid = int(item.get("pid") or 0)
            if pid:
                return FocusedWindow(
                    str(item.get("title") or ""),
                    str(item.get("class") or item.get("initialClass") or ""),
                    pid,
                    "Hyprland",
                )
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    # Sway and i3 expose the focused node in their tree.
    for command, source in (("swaymsg", "Sway"), ("i3-msg", "i3")):
        result = _run([command, "-t", "get_tree"])
        if not result or result.returncode != 0:
            continue
        try:
            focused = _walk_focused_tree(json.loads(result.stdout))
            if focused and int(focused.get("pid") or 0):
                return FocusedWindow(
                    str(focused.get("name") or ""),
                    str(focused.get("app_id") or focused.get("window_properties", {}).get("class") or ""),
                    int(focused["pid"]),
                    source,
                )
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    # X11 and XWayland fallback.
    result = _run(["xdotool", "getactivewindow"])
    if result and result.returncode == 0 and result.stdout.strip().isdigit():
        window_id = result.stdout.strip()
        pid_result = _run(["xdotool", "getwindowpid", window_id])
        title_result = _run(["xdotool", "getwindowname", window_id])
        app_result = _run(["xdotool", "getwindowclassname", window_id])
        try:
            pid = int(pid_result.stdout.strip()) if pid_result and pid_result.returncode == 0 else 0
        except ValueError:
            pid = 0
        if pid:
            return FocusedWindow(
                title_result.stdout.strip() if title_result else "",
                app_result.stdout.strip() if app_result else "",
                pid,
                "X11",
            )

    # GNOME Wayland and other sessions that do not expose a generic active-
    # window CLI can still identify a focused terminal through AT-SPI.
    helper = Path(__file__).with_name("terminal_atspi_reader.py")
    python = os.environ.get("ADAM_ATSPI_PYTHON", "/usr/bin/python3")
    if not Path(python).exists():
        python = shutil.which("python3") or "python3"
    payload = json.dumps({"find_focused_terminal": True})
    try:
        result = subprocess.run([python, str(helper), payload], capture_output=True, text=True, timeout=3.0, check=False)
        detected = json.loads(result.stdout or "{}") if result.returncode == 0 else {}
        if detected.get("ok") and int(detected.get("pid") or 0):
            return FocusedWindow(
                str(detected.get("title") or ""),
                str(detected.get("app") or ""),
                int(detected["pid"]),
                "AT-SPI focused terminal",
            )
    except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError):
        pass
    return None


def _process_snapshot() -> tuple[dict[int, dict[str, Any]], dict[int, list[int]]]:
    """Read only process metadata needed to associate a terminal with its child pane."""
    processes: dict[int, dict[str, Any]] = {}
    children: dict[int, list[int]] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return processes, children
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            raw_stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            close = raw_stat.rfind(")")
            fields = raw_stat[close + 2 :].split()
            ppid = int(fields[1])
            comm = raw_stat[raw_stat.find("(") + 1 : close]
            cmdline = (entry / "cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()
            processes[pid] = {"pid": pid, "ppid": ppid, "comm": comm, "cmdline": cmdline}
            children.setdefault(ppid, []).append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return processes, children


def _descendants(pid: int, children: dict[int, list[int]], limit: int = 512) -> list[int]:
    found: list[int] = []
    queue = list(children.get(pid, []))
    while queue and len(found) < limit:
        child = queue.pop(0)
        found.append(child)
        queue.extend(children.get(child, []))
    return found


def _proc_environment(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace")
    except OSError:
        return {}
    output: dict[str, str] = {}
    for item in raw.split("\0"):
        key, sep, value = item.partition("=")
        if sep and key in {"TMUX", "TMUX_PANE", "KITTY_WINDOW_ID", "KITTY_PID", "KITTY_LISTEN_ON", "WEZTERM_PANE"}:
            output[key] = value
    return output


def _environment_for_focused_terminal(
    focused: FocusedWindow,
    processes: dict[int, dict[str, Any]],
    children: dict[int, list[int]],
) -> dict[str, str]:
    """Collect terminal identity variables from the focused app and its descendants."""
    pids = [focused.pid, *_descendants(focused.pid, children)]
    # Some window managers report a helper process; include its ancestors too.
    current = focused.pid
    for _ in range(8):
        info = processes.get(current)
        if not info:
            break
        current = int(info["ppid"])
        if current <= 1:
            break
        pids.append(current)
    result: dict[str, str] = {}
    for pid in pids:
        result.update(_proc_environment(pid))
    return result


def _matches_terminal(value: str) -> str | None:
    normalized = re.sub(r"[_.-]+", " ", value or "").casefold()
    for name in _TERMINAL_NAMES:
        normalized_name = re.sub(r"[_.-]+", " ", name).casefold()
        if normalized_name in normalized:
            return name
    return None


def _tmux_client_for_focused_window(
    focused: FocusedWindow,
    processes: dict[int, dict[str, Any]],
    children: dict[int, list[int]],
) -> tuple[str | None, str | None]:
    """Return the active tmux client and pane IDs only when tied to this window."""
    tmux = shutil.which("tmux")
    if not tmux:
        return None, None
    descendants = set(_descendants(focused.pid, children))
    clients = _run([tmux, "list-clients", "-F", "#{client_pid}\t#{client_name}"], timeout=2.0)
    if not clients or clients.returncode != 0:
        return None, None
    for line in clients.stdout.splitlines():
        client_pid_text, sep, client_name = line.partition("\t")
        if not sep or not client_pid_text.isdigit() or int(client_pid_text) not in descendants:
            continue
        pane = _run([tmux, "display-message", "-p", "-c", client_name, "#{pane_id}"], timeout=2.0)
        pane_id = pane.stdout.strip() if pane and pane.returncode == 0 else ""
        if pane_id.startswith("%"):
            return client_name, pane_id
    return None, None


def _detect_terminal(*, probe_accessibility: bool) -> dict[str, Any]:
    """Detect the focused terminal and the safest available buffer reader."""
    focused = _focused_window()
    if focused is None:
        return TerminalDetection(
            False, None, None, None, None, None, False,
            "Could not identify the focused application through the active window manager.",
        ).to_dict()

    processes, children = _process_snapshot()
    process = processes.get(focused.pid, {})
    # Window titles are user-controlled (a shell can set them), so identify
    # the emulator from the window class or owning process only.
    terminal = (
        _matches_terminal(focused.app)
        or _matches_terminal(process.get("comm", ""))
        or _matches_terminal(process.get("cmdline", ""))
    )
    if not terminal:
        return TerminalDetection(
            False, None, focused.app or None, focused.title or None, focused.pid,
            None, False,
            "The focused application is not a recognized terminal emulator. Focus the terminal window and retry.",
        ).to_dict()

    client_name, pane_id = _tmux_client_for_focused_window(focused, processes, children)
    if pane_id:
        return TerminalDetection(True, terminal, focused.app or None, focused.title or None,
                                 focused.pid, "tmux", True,
                                 f"Focused tmux client and pane {pane_id} detected.").to_dict() | {
                                     "pane_id": pane_id, "client_name": client_name
                                 }

    env = _environment_for_focused_terminal(focused, processes, children)
    if terminal == "kitty":
        same_kitty_process = (
            env.get("KITTY_PID")
            and str(focused.pid) == str(env.get("KITTY_PID"))
        )
        if shutil.which("kitten") and (env.get("KITTY_LISTEN_ON") or (same_kitty_process and os.environ.get("KITTY_LISTEN_ON"))):
            return TerminalDetection(True, terminal, focused.app or None, focused.title or None,
                                     focused.pid, "kitty", True,
                                     "Kitty remote-control socket is available; buffer reads are read-only.").to_dict()
        reason = "Kitty is detected, but no remote-control socket is available to this process."
    elif terminal == "wezterm":
        pane = env.get("WEZTERM_PANE")
        if shutil.which("wezterm") and pane:
            return TerminalDetection(True, terminal, focused.app or None, focused.title or None,
                                     focused.pid, "wezterm", True,
                                     "WezTerm pane ID is available for a read-only text capture.").to_dict() | {
                                         "pane_id": pane
                                     }
        reason = "WezTerm is detected, but its CLI or focused pane ID is unavailable."
    elif terminal == "iterm2":
        reason = "iTerm2 is detected; its scripting API adapter is not available in this runtime yet."
    else:
        reason = "Using the generic AT-SPI text interface if this terminal exposes one."

    result = TerminalDetection(
        True, terminal, focused.app or None, focused.title or None, focused.pid,
        "atspi", not probe_accessibility,
        reason + " Checking accessibility without changing focus.",
    ).to_dict()
    if probe_accessibility:
        available = _probe_atspi(result)
        result["available"] = available
        result["backend"] = "atspi" if available else None
        result["reason"] = (
            "An accessible terminal text widget is available; reading it will not move focus."
            if available else reason + " No AT-SPI terminal text widget was detected."
        )
    return result


def detect_terminal() -> dict[str, Any]:
    """Detect the focused terminal and verify that a read-only path exists."""
    return _detect_terminal(probe_accessibility=True)


def _bounded_text(text: str, max_chars: int) -> tuple[str, bool]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(normalized) <= max_chars:
        return normalized, False
    start = len(normalized) - max_chars
    tail = normalized[start:]
    if start and normalized[start - 1] != "\n" and "\n" in tail:
        tail = tail[tail.find("\n") + 1 :]
    return tail, True


def _read_tmux(detection: dict[str, Any], scope: str, max_chars: int) -> dict[str, Any]:
    tmux = shutil.which("tmux")
    pane_id = detection.get("pane_id")
    if not tmux or not pane_id:
        return {"ok": False, "error": "The focused tmux pane could not be identified."}
    command = [tmux, "capture-pane", "-p", "-t", str(pane_id)]
    if scope == "recent":
        command.extend(["-S", "-200", "-E", "-"])
    elif scope == "all":
        command.extend(["-S", "-", "-E", "-"])
    result = _run(command, timeout=3.0)
    if not result or result.returncode != 0:
        error = result.stderr.strip() if result else "tmux capture timed out or could not be started"
        return {"ok": False, "error": f"Could not capture the focused tmux pane: {error[:300]}"}
    text, truncated = _bounded_text(result.stdout, max_chars)
    return {"ok": True, "terminal": detection["terminal"], "backend": "tmux",
            "scope": scope, "pane_id": pane_id, "text": text, "truncated": truncated,
            "text_is_untrusted": True}


def _read_kitty(detection: dict[str, Any], scope: str, max_chars: int) -> dict[str, Any] | None:
    kitten = shutil.which("kitten")
    focused = _focused_window()
    socket = ""
    if focused:
        processes, children = _process_snapshot()
        env = _environment_for_focused_terminal(focused, processes, children)
        socket = env.get("KITTY_LISTEN_ON", "")
        if not socket and env.get("KITTY_PID") == str(focused.pid):
            socket = os.environ.get("KITTY_LISTEN_ON", "")
    if not kitten or not socket:
        return None
    extent = {"screen": "screen", "recent": "all", "all": "all"}[scope]
    command = [
        kitten, "@", "--to", socket, "--use-password=never",
        "get-text", "--match", "state:focused", "--extent", extent,
    ]
    result = _run(command, timeout=3.0)
    if not result or result.returncode != 0:
        return None
    text = result.stdout
    truncated_by_lines = False
    if scope == "recent":
        lines = text.splitlines()
        truncated_by_lines = len(lines) > 200
        text = "\n".join(lines[-200:])
    text, truncated_by_chars = _bounded_text(text, max_chars)
    return {"ok": True, "terminal": detection["terminal"], "backend": "kitty",
            "scope": scope, "text": text, "truncated": truncated_by_lines or truncated_by_chars,
            "text_is_untrusted": True}


def _read_wezterm(detection: dict[str, Any], scope: str, max_chars: int) -> dict[str, Any] | None:
    wezterm = shutil.which("wezterm")
    pane = detection.get("pane_id") or os.environ.get("WEZTERM_PANE")
    if not wezterm or not pane:
        return None
    command = [wezterm, "cli", "get-text", "--pane-id", str(pane)]
    if scope == "recent":
        command.extend(["--start-line", "-200"])
    elif scope == "all":
        command.extend(["--start-line", "-1000000"])
    result = _run(command, timeout=3.0)
    if not result or result.returncode != 0:
        return None
    text, truncated = _bounded_text(result.stdout, max_chars)
    return {"ok": True, "terminal": detection["terminal"], "backend": "wezterm",
            "scope": scope, "pane_id": pane, "text": text, "truncated": truncated,
            "text_is_untrusted": True}


def _read_atspi(detection: dict[str, Any], scope: str, max_chars: int) -> dict[str, Any] | None:
    helper = Path(__file__).with_name("terminal_atspi_reader.py")
    python = os.environ.get("ADAM_ATSPI_PYTHON", "/usr/bin/python3")
    if not Path(python).exists():
        python = shutil.which("python3") or "python3"
    payload = json.dumps({
        "pid": detection.get("pid"), "title": detection.get("title") or "",
        "scope": scope, "max_chars": max_chars,
    })
    try:
        result = subprocess.run([python, str(helper), payload], capture_output=True, text=True, timeout=5.0, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not output.get("ok"):
        return None
    return {
        "ok": True, "terminal": detection.get("terminal"), "backend": "atspi",
        "scope": scope, "text": output.get("text", ""),
        "truncated": bool(output.get("truncated")),
        "scope_note": "The accessibility interface does not standardize the exact viewport/scrollback boundary.",
        "accessible_role": output.get("role"),
        "text_is_untrusted": True,
    }


def _probe_atspi(detection: dict[str, Any]) -> bool:
    """Check for a terminal text widget without returning its contents."""
    helper = Path(__file__).with_name("terminal_atspi_reader.py")
    python = os.environ.get("ADAM_ATSPI_PYTHON", "/usr/bin/python3")
    if not Path(python).exists():
        python = shutil.which("python3") or "python3"
    payload = json.dumps({
        "pid": detection.get("pid"), "title": detection.get("title") or "",
        "probe_only": True,
    })
    try:
        result = subprocess.run([python, str(helper), payload], capture_output=True, text=True, timeout=3.0, check=False)
        return bool(result.returncode == 0 and json.loads(result.stdout or "{}").get("ok"))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return False


def read_terminal_text(scope: str = "screen", max_chars: int = 12_000) -> dict[str, Any]:
    """Read text from the focused terminal without changing its state.

    `screen` reads the visible viewport where the backend supports it;
    `recent` includes up to 200 previous lines; `all` requests retained history.
    Output is always bounded and labelled as untrusted terminal data.
    """
    scope = str(scope or "screen").strip().lower()
    if scope not in {"screen", "recent", "all"}:
        return {"ok": False, "error": "scope must be screen, recent, or all."}
    try:
        max_chars = int(max_chars)
    except (TypeError, ValueError):
        return {"ok": False, "error": "max_chars must be an integer."}
    if not 1 <= max_chars <= _MAX_TOOL_CHARS:
        return {"ok": False, "error": f"max_chars must be between 1 and {_MAX_TOOL_CHARS}."}

    detection = _detect_terminal(probe_accessibility=False)
    if not detection.get("ok"):
        return {"ok": False, "error": detection.get("reason"), "detection": detection}

    backend = detection.get("backend")
    if backend == "tmux":
        return _read_tmux(detection, scope, max_chars)
    if backend == "kitty":
        output = _read_kitty(detection, scope, max_chars)
        if output:
            return output
    elif backend == "wezterm":
        output = _read_wezterm(detection, scope, max_chars)
        if output:
            return output

    output = _read_atspi(detection, scope, max_chars)
    if output:
        return output

    terminal = detection.get("terminal") or "terminal"
    return {
        "ok": False,
        "terminal": terminal,
        "backend": backend,
        "error": (
            f"{terminal} is focused, but no read-only text-buffer interface is available. "
            "The terminal did not expose an enabled native API or an AT-SPI Text interface. "
            "For this session, a tmux pane or an emulator buffer API must be enabled; "
            "no input was sent and focus was not changed."
        ),
        "text_is_untrusted": True,
    }
