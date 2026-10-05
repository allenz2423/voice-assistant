"""Browser navigation coverage against a local, deterministic test page."""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.tools.browser_navigation import BrowserNavigator


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None,
    reason="Install the optional browser-control dependencies to run browser tests",
)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/next"):
            body = "<title>Next page</title><p>Reached the next page.</p><a href='/'>Home</a>"
        else:
            body = """<title>Adam browser test</title>
              <h1>Local browser fixture</h1>
              <label for='query'>Search words</label><input id='query' type='text'>
              <label for='secret'>Password</label><input id='secret' type='password'>
              <a href='/next'>Continue to next page</a>
              <a href='javascript:alert(1)'>Unsafe link</a>
              <button type='submit'>Submit</button>"""
        data = f"<!doctype html><html><body>{body}</body></html>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


@pytest.fixture
def local_page():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.mark.parametrize(
    "browser",
    [
        "chromium",
        "firefox",
        pytest.param(
            "microsoft-edge-stable",
            id="microsoft-edge",
            marks=pytest.mark.skipif(
                shutil.which("microsoft-edge-stable") is None,
                reason="Microsoft Edge is not installed",
            ),
        ),
    ],
)
def test_browser_navigation_flow(browser, local_page, tmp_path):
    navigator = BrowserNavigator(
        browser=browser,
        profile_path=tmp_path / "adam-profile",
        timeout_seconds=10,
        headless=True,
    )

    async def exercise():
        try:
            page = await navigator.run("navigate", target=local_page)
            assert "Local browser fixture" in page
            assert "L1 [link] Continue to next page" in page
            assert "F1 [text] Search words" in page
            assert "F2 [password]" not in page
            followable_links = page.split("Followable links:\n", 1)[1].split("\n\n", 1)[0]
            assert "Unsafe link" not in followable_links

            filled = await navigator.run("fill", target="F1", text="hello Adam")
            assert "It was not submitted" in filled
            field_value = await asyncio.get_running_loop().run_in_executor(
                navigator._executor,
                lambda: navigator._page.locator("#query").input_value(),
            )
            assert field_value == "hello Adam"

            next_page = await navigator.run("click", target="L1")
            assert "Reached the next page" in next_page
            home = await navigator.run("back")
            assert "Local browser fixture" in home
            rejected = await navigator.run("navigate", target="javascript:alert(1)")
            assert "only opens HTTP or HTTPS" in rejected

            await navigator.release_browser()
            assert navigator._context is None
            stale_ref = await navigator.run("click", target="L1")
            assert "reference is missing or stale" in stale_ref
            restored = await navigator.run("inspect")
            assert "Local browser fixture" in restored
            assert navigator._restore_url == ""
        finally:
            navigator.close()

    asyncio.run(exercise())


def test_adam_brain_dispatches_browser_tool():
    from src.llm.brain import AdamBrain

    class CustomTools:
        @staticmethod
        def has_tool(_name):
            return False

    class Navigator:
        async def run(self, **kwargs):
            return f"browser result: {kwargs['action']} {kwargs['target']}"

    brain = object.__new__(AdamBrain)
    brain.speculative_router = None
    brain.custom_tool_mgr = CustomTools()
    brain.browser_navigator = Navigator()
    result = asyncio.run(
        brain._execute_tool(
            "browser_navigation",
            {"action": "navigate", "target": "https://example.org"},
        )
    )
    assert result == "browser result: navigate https://example.org"
