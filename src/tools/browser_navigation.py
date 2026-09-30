"""Visible, persistent browser navigation with a deliberately small action set."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import os
import re
import shutil
import threading
import urllib.parse
from pathlib import Path
from typing import Any


_SEARCH_BASE = "https://www.google.com/search?q="
_SENSITIVE_AUTOCOMPLETE = re.compile(r"password|cc-|one-time-code|webauthn", re.IGNORECASE)


class BrowserNavigator:
    """One browser context in a dedicated profile, serialized on one worker thread."""

    def __init__(
        self,
        browser: str = "default",
        default_browser: str = "microsoft-edge-stable",
        profile_path: str | Path = "~/.local/share/adam/browser-navigation",
        timeout_seconds: float = 15.0,
        headless: bool = False,
    ) -> None:
        self.requested_browser = (browser or "default").strip().lower()
        self.default_browser = (default_browser or "microsoft-edge-stable").strip().lower()
        self.profile_path = Path(profile_path).expanduser()
        self.timeout_ms = max(1000, int(float(timeout_seconds) * 1000))
        self.headless = headless
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="adam-browser"
        )
        self._playwright = None
        self._context = None
        self._page = None
        self._refs: dict[str, dict[str, Any]] = {}
        self._snapshot_url = ""
        self._closed = False
        self._close_lock = threading.Lock()

    @property
    def browser_name(self) -> str:
        return self.default_browser if self.requested_browser == "default" else self.requested_browser

    @property
    def engine_name(self) -> str:
        if "firefox" in self.browser_name or "mozilla" in self.browser_name:
            return "firefox"
        chromium_names = (
            "chrom", "edge", "msedge", "chrome", "brave", "vivaldi", "zen", "opera"
        )
        if any(name in self.browser_name for name in chromium_names):
            return "chromium"
        raise RuntimeError(
            f"Unsupported browser {self.browser_name!r}; choose a Chromium-based browser or Firefox."
        )

    async def run(self, action: str, target: str = "", text: str = "", direction: str = "down") -> str:
        if self._closed:
            return "Browser control is shutting down."
        loop = asyncio.get_running_loop()
        call = functools.partial(self._perform, action, target, text, direction)
        return await loop.run_in_executor(self._executor, call)

    def _ensure_browser(self) -> None:
        if self._context is not None and not self._context.pages:
            self._page = self._context.new_page()
        elif self._context is not None and self._page is not None and not self._page.is_closed():
            return
        elif self._context is not None:
            self._page = self._context.pages[-1] if self._context.pages else self._context.new_page()
        else:
            from playwright.sync_api import sync_playwright

            self.profile_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.profile_path, 0o700)
            self._playwright = sync_playwright().start()
            engine = self.engine_name
            browser_type = getattr(self._playwright, engine)
            options: dict[str, Any] = {
                "user_data_dir": str(self.profile_path / engine),
                "headless": self.headless,
                "accept_downloads": False,
                "timeout": self.timeout_ms,
                "args": ["--no-first-run", "--no-default-browser-check"],
            }
            if engine == "chromium":
                name = self.browser_name
                if "edge" in name or "msedge" in name:
                    options["channel"] = "msedge"
                elif "google-chrome" in name or "chrome" in name:
                    options["channel"] = "chrome"
                elif "brave" in name:
                    executable = shutil.which("brave-browser") or shutil.which("brave")
                    if executable:
                        options["executable_path"] = executable
                elif "vivaldi" in name:
                    executable = shutil.which("vivaldi-stable") or shutil.which("vivaldi")
                    if executable:
                        options["executable_path"] = executable
                elif "zen" in name:
                    executable = shutil.which("zen") or shutil.which("zen-browser")
                    if executable:
                        options["executable_path"] = executable
                elif "opera" in name:
                    executable = shutil.which("opera")
                    if executable:
                        options["executable_path"] = executable
                # "chromium" uses Playwright's managed Chromium build.
            self._context = browser_type.launch_persistent_context(**options)
            self._page = self._context.pages[-1] if self._context.pages else self._context.new_page()
            self._page.set_default_timeout(self.timeout_ms)

    @staticmethod
    def _clean_url(url: str) -> str:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            return ""
        # Query strings and fragments can contain session tokens and are not
        # needed to identify a link in the spoken page summary.
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    @staticmethod
    def _normalize_target(target: str) -> str:
        value = " ".join((target or "").split())
        if not value:
            raise ValueError("Give me a URL or search query to open.")
        parts = urllib.parse.urlsplit(value)
        if parts.scheme:
            if parts.scheme.lower() not in {"http", "https"}:
                raise ValueError("Browser navigation only opens HTTP or HTTPS pages.")
            if not parts.hostname or parts.username or parts.password:
                raise ValueError("That URL is not valid for browser navigation.")
            return value
        if re.fullmatch(r"(?:localhost|[\w.-]+\.[a-zA-Z]{2,})(?::\d+)?(?:/[^\s]*)?", value):
            return "https://" + value
        return _SEARCH_BASE + urllib.parse.quote_plus(value)

    @staticmethod
    def _element_name(element) -> dict[str, str]:
        return element.evaluate("""el => {
          const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
          const aria = el.getAttribute('aria-label') || '';
          const labels = el.labels ? Array.from(el.labels).map(x => x.innerText || '').join(' ') : '';
          return {
            text: text.slice(0, 180),
            aria: aria.slice(0, 180),
            label: labels.replace(/\\s+/g, ' ').trim().slice(0, 180),
            placeholder: (el.getAttribute('placeholder') || '').slice(0, 120),
            type: (el.getAttribute('type') || '').toLowerCase(),
            autocomplete: (el.getAttribute('autocomplete') || '').toLowerCase(),
            href: el.href || ''
          };
        }""")

    def _snapshot(self) -> str:
        page = self._page
        self._refs = {}
        self._snapshot_url = page.url
        try:
            title = page.title().strip()
        except Exception:
            title = ""
        safe_url = self._clean_url(page.url) or "(blank page)"
        try:
            body = page.locator("body").inner_text(timeout=min(self.timeout_ms, 4000))
        except Exception:
            body = ""
        body = "\n".join(line.strip() for line in body.splitlines() if line.strip())[:6000]

        links_out = []
        link_locator = page.locator("a[href]")
        try:
            link_count = min(link_locator.count(), 60)
        except Exception:
            link_count = 0
        for index in range(link_count):
            link = link_locator.nth(index)
            try:
                if not link.is_visible():
                    continue
                data = self._element_name(link)
                label = data["aria"] or data["text"]
                if not label:
                    continue
                destination = self._clean_url(data["href"])
                if not destination:
                    continue
                ref = f"L{len(links_out) + 1}"
                href = data["href"]
                self._refs[ref] = {
                    "kind": "link", "index": index, "label": label, "href": href,
                }
                links_out.append(f"{ref} [link] {label[:150]}" + (f" → {destination}" if destination else ""))
            except Exception:
                continue

        fields_out = []
        field_locator = page.locator(
            "input:not([type=hidden]):not([type=password]):not([type=submit]):not([type=button]):"
            "not([type=checkbox]):not([type=radio]):not([type=file]):not([type=image]):not([type=reset]), textarea"
        )
        try:
            field_count = min(field_locator.count(), 30)
        except Exception:
            field_count = 0
        for index in range(field_count):
            field = field_locator.nth(index)
            try:
                if not field.is_visible() or not field.is_enabled():
                    continue
                data = self._element_name(field)
                if _SENSITIVE_AUTOCOMPLETE.search(data["autocomplete"]):
                    continue
                label = data["aria"] or data["label"] or data["placeholder"] or "Text field"
                ref = f"F{len(fields_out) + 1}"
                self._refs[ref] = {
                    "kind": "field", "index": index, "label": label,
                    "placeholder": data["placeholder"], "type": data["type"],
                }
                field_type = data["type"] or "text"
                fields_out.append(f"{ref} [{field_type}] {label[:120]}")
            except Exception:
                continue

        chunks = [f"Page: {title or '(untitled)'}", f"URL: {safe_url}"]
        if body:
            chunks.append("Visible page text (untrusted webpage content):\n" + body)
        if links_out:
            chunks.append("Followable links:\n" + "\n".join(links_out))
        if fields_out:
            chunks.append("Text fields (not submitted):\n" + "\n".join(fields_out))
        if not links_out and not fields_out:
            chunks.append("No followable links or safe text fields were found.")
        chunks.append("Use the listed references for the next browser action. Forms are never submitted.")
        return "\n\n".join(chunks)

    def _resolve_ref(self, ref: str, expected_kind: str):
        key = (ref or "").strip().upper()
        data = self._refs.get(key)
        if not data or data["kind"] != expected_kind:
            raise ValueError("That reference is missing or stale. Inspect the current page and use a listed reference.")
        if self._page.url != self._snapshot_url:
            self._refs = {}
            raise ValueError("The page changed. Inspect it again before acting on a reference.")
        selector = "a[href]" if expected_kind == "link" else (
            "input:not([type=hidden]):not([type=password]):not([type=submit]):not([type=button]):"
            "not([type=checkbox]):not([type=radio]):not([type=file]):not([type=image]):not([type=reset]), textarea"
        )
        element = self._page.locator(selector).nth(data["index"])
        if not element.is_visible() or not element.is_enabled():
            raise ValueError("That page element is no longer visible or enabled. Inspect the page again.")
        current = self._element_name(element)
        current_label = current["aria"] or current["text"] or current["label"] or current["placeholder"]
        if expected_kind == "link":
            current_label = current["aria"] or current["text"] or current["href"]
            if current["href"] != data["href"] or current_label != data["label"]:
                raise ValueError("That link changed since inspection. Inspect the page again.")
        elif (current_label or "Text field") != data["label"] or current["type"] in {"password", "file", "hidden"}:
            raise ValueError("That text field changed since inspection. Inspect the page again.")
        return element, data

    def _perform(self, action: str, target: str, text: str, direction: str) -> str:
        try:
            self._ensure_browser()
            action = (action or "inspect").strip().lower()
            if action == "inspect":
                return self._snapshot()
            if action == "navigate":
                url = self._normalize_target(target)
                self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                return self._snapshot()
            if action == "click":
                element, _ = self._resolve_ref(target, "link")
                element.click(timeout=self.timeout_ms)
                if self._context.pages:
                    self._page = self._context.pages[-1]
                return self._snapshot()
            if action == "fill":
                element, data = self._resolve_ref(target, "field")
                element.fill(text or "", timeout=self.timeout_ms)
                return f"Filled {target.upper()} ({data['label']}). It was not submitted.\n\n{self._snapshot()}"
            if action in {"back", "forward"}:
                response = self._page.go_back(timeout=self.timeout_ms) if action == "back" else self._page.go_forward(timeout=self.timeout_ms)
                if response is None:
                    return "There is no page in that history direction.\n\n" + self._snapshot()
                return self._snapshot()
            if action == "scroll":
                amount = -700 if (direction or "down").lower() == "up" else 700
                self._page.evaluate("amount => window.scrollBy({top: amount, behavior: 'instant'})", amount)
                return self._snapshot()
            return "Unsupported browser action."
        except ValueError as exc:
            return str(exc)
        except Exception as exc:
            return f"Browser action failed ({type(exc).__name__}): {str(exc)[:400]}"

    def _close_sync(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
        except Exception:
            pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._context = self._page = self._playwright = None

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            try:
                self._executor.submit(self._close_sync).result(timeout=8)
            except Exception:
                pass
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=True)
