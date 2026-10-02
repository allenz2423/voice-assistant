"""Small read-only observer that combines available desktop interfaces."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from src.tools.desktop_timing import timed_stage, timing_operation


@dataclass
class DesktopObservation:
    message: str
    screenshot: bytes | None


def _windows() -> tuple[list[dict], str | None]:
    """Collect window metadata from Hyprland, or just the X11 active window."""
    try:
        result = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, timeout=2)
        clients = json.loads(result.stdout) if result.returncode == 0 else []
        if clients:
            result = subprocess.run(["hyprctl", "activewindow", "-j"], capture_output=True, text=True, timeout=2)
            active = json.loads(result.stdout or "{}")
            windows = [{
                "address": str(c.get("address") or ""),
                "pid": int(c.get("pid") or 0), "title": c.get("title") or "",
                "app": c.get("class") or c.get("initialClass") or "Application",
                "workspace": (c.get("workspace") or {}).get("id"),
            } for c in clients if c.get("pid")]
            windows.sort(key=lambda w: (w["address"] != str(active.get("address") or ""), w["workspace"] or 0))
            return windows, "Hyprland"
    except Exception:
        pass

    try:
        active_id = subprocess.run(["xdotool", "getactivewindow"], capture_output=True, text=True, timeout=2, check=True).stdout.strip()
        pid = subprocess.run(["xdotool", "getwindowpid", active_id], capture_output=True, text=True, timeout=2, check=True).stdout.strip()
        title = subprocess.run(["xdotool", "getwindowname", active_id], capture_output=True, text=True, timeout=2).stdout.strip()
        return [{"pid": int(pid), "title": title, "app": "Application", "workspace": None}], "X11"
    except Exception:
        return [], None


def _read_atspi(windows: list[dict]) -> list[dict]:
    if not windows:
        return []
    helper = Path(__file__).with_name("atspi_reader.py")
    python = os.environ.get("ADAM_ATSPI_PYTHON", "/usr/bin/python3")
    payload = json.dumps(windows[:30], ensure_ascii=False)
    try:
        result = subprocess.run([python, str(helper), payload], capture_output=True, text=True, timeout=8)
        data = json.loads(result.stdout) if result.stdout.strip() else []
        return data if isinstance(data, list) else []
    except Exception as exc:
        return [{"ok": False, "error": str(exc)}]


def _read_browser_dom(active: dict) -> tuple[str | None, str]:
    """Read an explicitly configured or browser-advertised local CDP endpoint."""
    app = str(active.get("app") or "").casefold()
    if not any(browser in app for browser in ("edge", "chrome", "chromium", "brave", "vivaldi", "opera")):
        return None, "Active window is not a Chromium browser"

    endpoint = os.environ.get("ADAM_CDP_ENDPOINT", "").strip()
    if not endpoint:
        # Only attach when the active browser itself advertises a debugging port.
        # Never scan for ports or start a browser/profile during passive observation.
        try:
            command = Path(f"/proc/{int(active['pid'])}/cmdline").read_bytes().decode(errors="ignore").split("\0")
            port = next((arg.partition("=")[2] for arg in command if arg.startswith("--remote-debugging-port=")), "")
            address = next((arg.partition("=")[2] for arg in command if arg.startswith("--remote-debugging-address=")), "127.0.0.1")
            if port and address in ("127.0.0.1", "localhost", "::1"):
                host = "[::1]" if address == "::1" else address
                endpoint = f"http://{host}:{port}"
        except Exception:
            pass
    if not endpoint:
        return None, "No local CDP endpoint advertised/configured"
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        return None, "CDP endpoint must be a local HTTP(S) address"
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint, timeout=1800)
            pages = [page for context in browser.contexts for page in context.pages if not page.is_closed()]
            active_title = str(active.get("title") or "")
            page = next((p for p in pages if p.title().casefold() in active_title.casefold() or active_title.casefold() in p.title().casefold()), None)
            if page is None:
                return None, "No CDP page matched the active window"
            title, url = page.title(), page.url
            body = page.locator("body").inner_text(timeout=1800)
            return f"Title: {title}\nURL: {url}\nPage text (untrusted):\n{body[:3500]}", "CDP"
    except ImportError:
        return None, "Playwright is not installed"
    except Exception as exc:
        return None, f"CDP unavailable: {str(exc)[:180]}"


def _screenshot(scope: str = "monitor") -> tuple[bytes | None, str]:
    """Bound capture time so a stuck compositor tool cannot block the observer."""
    code = (
        "from src.tools.desktop import capture_screenshot; import sys; "
        f"sys.stdout.buffer.write(capture_screenshot({scope!r}))"
    )
    try:
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], capture_output=True, timeout=8)
        if result.returncode == 0 and result.stdout:
            return result.stdout, "captured"
        return None, (result.stderr.decode(errors="replace") or "capture returned no image")[:180]
    except Exception as exc:
        return None, str(exc)[:180]


def _observe_desktop_impl(scope: str = "monitor", include_screenshot: bool = True) -> DesktopObservation:
    """Read desktop state; window scope limits structured reads and capture to the focused window."""
    scope = (scope or "monitor").strip().lower()
    if scope not in {"monitor", "window"}:
        return DesktopObservation("Screenshot scope must be 'monitor' or 'window'.", None)
    with timed_stage("observer.window_enumeration"):
        windows, wm = _windows()
    if not windows:
        message = "Window metadata unavailable."
        screenshot = None
        if include_screenshot:
            with timed_stage("observer.screenshot_subprocess"):
                screenshot, status = _screenshot(scope)
            message += " Screenshot attached." if screenshot else f" Screenshot unavailable: {status}."
        return DesktopObservation(message, screenshot)

    active = windows[0]
    inspected_windows = [active] if scope == "window" else windows[:30]
    with timed_stage("observer.atspi", window_count=len(inspected_windows)):
        accessibility = _read_atspi(inspected_windows)
    if scope == "window":
        sections = [f"Window manager: {wm}. Focused window only: {active['app']} — {active['title'] or '(untitled)'}."]
    else:
        sections = [f"Window manager: {wm}. Windows found: {len(windows)} (up to 30 inspected)."]
    for index, window in enumerate(inspected_windows):
        label = f"{window['app']} — {window['title'] or '(untitled)'}"
        if window.get("workspace") is not None:
            label += f" [workspace {window['workspace']}]"
        row = accessibility[index] if index < len(accessibility) else {"ok": False, "error": "no AT-SPI result"}
        sections.append(f"\n{label}\nAT-SPI: {row.get('tree') if row.get('ok') else row.get('error', 'unavailable')}")

    with timed_stage("observer.browser_dom"):
        dom, dom_status = _read_browser_dom(active)
    sections.append(f"\nBrowser DOM: {dom_status}.")
    if dom:
        sections.append(dom)
    screenshot = None
    if include_screenshot:
        with timed_stage("observer.screenshot_subprocess"):
            screenshot, screenshot_status = _screenshot(scope)
        sections.append("Screenshot attached." if screenshot else f"Screenshot unavailable: {screenshot_status}.")
    return DesktopObservation("\n".join(sections), screenshot)


def observe_desktop(scope: str = "monitor", include_screenshot: bool = True) -> DesktopObservation:
    """Read desktop state while emitting stage timings without screen contents."""
    with timing_operation("observer.operation"):
        return _observe_desktop_impl(scope, include_screenshot)
