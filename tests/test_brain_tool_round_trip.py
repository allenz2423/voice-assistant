import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _DummyTTS:
    def __init__(self):
        self.spoken = []

    async def speak_async(self, text):
        self.spoken.append(text)


class _DummyClient:
    def __init__(self, provider, responses):
        self.provider = provider
        self.responses = list(responses)
        self.requests = []
        self.request_tools = []

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        self.requests.append([dict(message) for message in messages])
        self.request_tools.append(tools)
        self.request_max_tokens = getattr(self, "request_max_tokens", [])
        self.request_max_tokens.append(max_tokens)
        self.request_think = getattr(self, "request_think", [])
        self.request_think.append(think)
        return self.responses.pop(0)

    def format_tool_response(self, tool_call_id, tool_name, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}


def _config(provider="local"):
    return SimpleNamespace(llm=SimpleNamespace(
        provider=provider,
        local_model="test-model",
        cloud_model="test-model",
        ollama_host="http://localhost:11434",
        api_base="https://example.test/v1",
        api_key="test-key",
        temperature=0,
        num_ctx=8192,
    ))


@pytest.mark.asyncio
async def test_confirmation_pause_correlates_every_call_in_the_tool_batch():
    from src.llm.brain import AdamBrain

    class Confirmation:
        async def request_confirmation(self, payload, question):
            self.payload = payload
            self.question = question

    tts = _DummyTTS()
    confirmation = Confirmation()
    brain = AdamBrain(_config(), None, None, confirmation, tts)
    brain.llm_client = _DummyClient("local", [{
        "content": "",
        "tool_calls": [
            {
                "id": "confirm-call-7",
                "function": {"name": "ask_user_confirmation", "arguments": {
                    "question": "Restart now?", "summary": "Restart system", "command": "systemctl reboot"
                }},
            },
            {
                "id": "time-call-9",
                "function": {"name": "get_current_time", "arguments": {"location": "local"}},
            },
        ],
    }])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Restart after telling me the time")

    assistant = next(message for message in brain.messages if message.get("tool_calls"))
    result_ids = {message.get("tool_call_id") for message in brain.messages if message.get("role") == "tool"}
    assert {call["id"] for call in assistant["tool_calls"]} == result_ids
    results = [json.loads(message["content"]) for message in brain.messages if message.get("role") == "tool"]
    assert results[0]["status"] == "returned"
    assert results[1]["status"] == "cancelled"
    assert results[1]["effect_status"] == "not_dispatched"


@pytest.mark.asyncio
async def test_current_request_is_active_task_and_prior_dialogue_is_context_only():
    from src.llm.brain import AdamBrain

    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.messages.extend([
        {"role": "user", "content": "Play a video and pause it."},
        {"role": "assistant", "content": "I could not confirm playback."},
    ])
    brain.llm_client = _DummyClient("local", [
        {"content": "Spotify is open.", "tool_calls": []},
    ])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Open Spotify")

    prompt = next(
        message["content"] for message in brain.llm_client.requests[0]
        if message.get("role") == "user" and "Current User Request" in message.get("content", "")
    )
    assert "[Current User Request — active task for this run]\nOpen Spotify" in prompt
    system_prompt = brain.llm_client.requests[0][0]["content"]
    assert "Earlier dialogue is context for resolving references, not a queue of unfinished work" in system_prompt
    assert brain.tts.spoken == ["Spotify is open."]


@pytest.mark.asyncio
async def test_empty_model_turn_gets_three_recoveries_without_dispatching_actions():
    from src.llm.brain import AdamBrain

    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {"content": "", "tool_calls": []},
        {"content": "", "tool_calls": []},
        {"content": "", "tool_calls": []},
        {"content": "", "tool_calls": []},
    ])
    executed = []
    brain._execute_tool = lambda name, args: executed.append((name, args))

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Open Spotify")

    assert brain.llm_client.requests.__len__() == 4
    assert executed == []
    assert "three recovery attempts" in brain.tts.spoken[0]


@pytest.mark.asyncio
async def test_false_filesystem_access_refusal_reprompts_when_read_tool_is_available():
    from unittest.mock import AsyncMock
    from src.llm.brain import AdamBrain

    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {
            "content": "I can't see your files, so I can't tell you directly. But if you're asking, it's probably feeling a bit chaotic!",
            "tool_calls": [],
        },
        {"content": "", "tool_calls": [{
            "id": "read-file",
            "function": {"name": "read_file", "arguments": {"path": "/tmp/synthetic-note.txt"}},
        }]},
        {"content": "The synthetic note says hello.", "tool_calls": []},
    ])
    brain._execute_tool = AsyncMock(return_value=json.dumps({"ok": True, "readback": "hello"}))

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Check my Downloads folder")

    assert len(brain.llm_client.requests) == 3
    assert "relevant tool is available: find_files, read_file" in brain.llm_client.requests[1][-1]["content"]
    brain._execute_tool.assert_awaited_once_with("read_file", {"path": "/tmp/synthetic-note.txt"})
    assert brain.tts.spoken[-1] == "The synthetic note says hello."


@pytest.mark.asyncio
async def test_optional_tool_free_model_receives_only_the_current_standalone_question():
    from src.llm.brain import AdamBrain

    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.messages.extend([
        {"role": "user", "content": "My private draft says PROJECT-EMBER-731."},
        {"role": "assistant", "content": "I can help revise it."},
    ])
    main_client = _DummyClient("custom", [{"content": "Main route", "tool_calls": []}])
    tool_free_client = _DummyClient("custom", [{
        "content": "Photosynthesis lets plants turn light into stored chemical energy.",
        "tool_calls": [],
    }])
    brain.llm_client = main_client
    brain.tool_free_llm_client = tool_free_client

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Explain photosynthesis in one sentence.")

    assert main_client.requests == []
    assert len(tool_free_client.requests) == 1
    routed_messages = tool_free_client.requests[0]
    assert len(routed_messages) == 2
    assert all("PROJECT-EMBER-731" not in message.get("content", "") for message in routed_messages)
    assert "Explain photosynthesis in one sentence." in routed_messages[-1]["content"]
    assert tool_free_client.request_tools == [[]]


@pytest.mark.asyncio
async def test_file_management_refusal_reprompts_with_confirmed_organizer_tool():
    from unittest.mock import AsyncMock
    from src.llm.brain import AdamBrain

    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {
            "content": "I can't actually move or delete files for you; I don't have file management tools hooked up.",
            "tool_calls": [],
        },
        {"content": "", "tool_calls": [{
            "id": "organize-downloads",
            "function": {
                "name": "organize_files",
                "arguments": {"directory": "~/Downloads", "group_by": "show"},
            },
        }]},
        {"content": "The organization request is waiting for your confirmation.", "tool_calls": []},
        {"content": "The organization request is waiting for your confirmation.", "tool_calls": []},
    ])
    brain._execute_tool = AsyncMock(
        return_value="Confirmation requested from user. Execution is paused waiting for user's verbal confirmation."
    )

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Organize my Downloads folder by show")

    assert "organize_files" in brain.llm_client.requests[1][-1]["content"]
    brain._execute_tool.assert_awaited_once_with(
        "organize_files", {"directory": "~/Downloads", "group_by": "show"}
    )
    assert "confirmation" in brain.tts.spoken[-1].lower()


@pytest.mark.asyncio
async def test_find_files_missing_directory_is_retried_as_a_tool_failure(tmp_path):
    from src.llm.brain import AdamBrain

    good_directory = tmp_path / "downloads"
    good_directory.mkdir()
    (good_directory / "synthetic.txt").write_text("synthetic", encoding="utf-8")
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {"content": "", "tool_calls": [{
            "id": "bad-folder",
            "function": {"name": "find_files", "arguments": {
                "directory": str(tmp_path / "missing"), "pattern": "*",
            }},
        }]},
        {"content": "", "tool_calls": [{
            "id": "good-folder",
            "function": {"name": "find_files", "arguments": {
                "directory": str(good_directory), "pattern": "*",
            }},
        }]},
        {"content": "I found the synthetic file.", "tool_calls": []},
        {"content": "I found the synthetic file.", "tool_calls": []},
    ])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Check the test Downloads folder")

    assert len(brain.llm_client.requests) == 3
    assert "Tool recovery 1/3" in brain.llm_client.requests[1][-1]["content"]
    assert "does not exist" in brain.llm_client.requests[1][-2]["content"]
    assert "synthetic file" in brain.tts.spoken[-1]


@pytest.mark.asyncio
async def test_capability_refusal_recovery_stops_after_three_reprompts():
    from src.llm.brain import AdamBrain

    refusal = {"content": "I can't access your file system.", "tool_calls": []}
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [refusal.copy() for _ in range(4)])
    executed = []
    brain._execute_tool = lambda name, args: executed.append((name, args))

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Check my Downloads folder")

    assert len(brain.llm_client.requests) == 4
    assert executed == []
    assert "after three recovery attempts" in brain.tts.spoken[-1]
    assert "haven't inspected or changed anything" in brain.tts.spoken[-1]


@pytest.mark.asyncio
async def test_explicit_adam_browser_request_prefetches_untrusted_page_text():
    from unittest.mock import patch
    from src.llm.brain import AdamBrain
    from src.llm.tools import ADAM_TOOLS

    class Browser:
        def __init__(self):
            self.actions = []
            self.released = 0

        async def run(self, **kwargs):
            self.actions.append(kwargs)
            return "Page: Invoice Register\nVisible page text (untrusted webpage content):\nOVERDUE $533.41"

        async def release_browser(self):
            self.released += 1

    class Controller:
        available = False
        drag_active = False
        coordinate_mode = "pixels"

        @staticmethod
        def invalidate_snapshot():
            pass

    brain = AdamBrain(
        _config(), None, None, None, _DummyTTS(),
        memory_mgr=type("M", (), {"retrieve_context": lambda *_: None})(),
    )
    browser = Browser()
    browser_tool = next(tool for tool in ADAM_TOOLS if tool.name == "browser_navigation")
    brain.browser_navigator = browser
    brain.config.browser_navigation = SimpleNamespace(headless=True)
    brain.computer_controller = Controller()
    brain.get_tools = lambda: [browser_tool]
    brain.llm_client = _DummyClient("local", [{"content": "The total is $533.41.", "tool_calls": []}])
    from src.telemetry.events import subscribe_events
    events = []
    unsubscribe = subscribe_events(events.append)

    try:
        with patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
            await brain.process_user_utterance(
                "In Adam's isolated browser, read the report and find the total."
            )
    finally:
        unsubscribe()

    assert browser.actions == [{"action": "inspect"}]
    assert browser.released == 1
    assert brain.llm_client.request_tools[0] == []
    first_user_message = next(
        message for message in brain.llm_client.requests[0] if message.get("role") == "user"
    )
    assert "Fresh Adam Browser Snapshot" in first_user_message["content"]
    assert "untrusted webpage content" in first_user_message["content"]
    assert "OVERDUE $533.41" in first_user_message["content"]
    assert not first_user_message.get("images")
    browser_events = [
        event for event in events
        if event.get("event") in {"tool.started", "tool.completed"}
    ]
    assert [event["event"] for event in browser_events] == ["tool.started", "tool.completed"]
    assert all(event["attributes"]["tool_name"] == "browser_navigation" for event in browser_events)


def test_browser_navigator_is_constructed_and_exposed_only_when_enabled(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.llm.brain import AdamBrain

    class Browser:
        def __init__(self, **kwargs):
            self.options = kwargs

        def close(self):
            pass

    monkeypatch.setattr("src.tools.browser_navigation.BrowserNavigator", Browser)
    config = _config()
    config.desktop = SimpleNamespace(default_browser="microsoft-edge-stable")
    config.browser_navigation = SimpleNamespace(
        enabled=True,
        browser="default",
        profile_path=str(tmp_path / "isolated-profile"),
        timeout_seconds=8,
        headless=True,
    )
    brain = AdamBrain(
        config, None, None, None, _DummyTTS(),
        memory_mgr=type("M", (), {"retrieve_context": lambda *_: None})(),
    )
    assert brain.browser_navigator.options == {
        "browser": "default",
        "default_browser": "microsoft-edge-stable",
        "profile_path": str(tmp_path / "isolated-profile"),
        "timeout_seconds": 8,
        "headless": True,
    }
    assert "browser_navigation" in {tool.name for tool in brain.get_tools()}
    brain.close()

    default_brain = AdamBrain(
        _config(), None, None, None, _DummyTTS(),
        memory_mgr=type("M", (), {"retrieve_context": lambda *_: None})(),
    )
    assert "browser_navigation" not in {tool.name for tool in default_brain.get_tools()}
    default_brain.close()


def test_adam_browser_scope_requires_an_explicit_profile_reference():
    from src.llm.brain import (
        _explicit_adam_browser_request,
        _filter_tools_for_dedicated_desktop_navigation,
        _is_dedicated_adam_browser_request,
        _is_read_only_adam_browser_request,
    )
    from src.llm.tools import ADAM_TOOLS

    assert _explicit_adam_browser_request("Use Adam's isolated browser to read this page.")
    assert _explicit_adam_browser_request("Open the Adam browser profile.")
    assert not _explicit_adam_browser_request("Read the current page in my browser.")
    assert _is_dedicated_adam_browser_request(
        "In Adam's isolated browser, read this webpage and summarize it."
    )
    assert not _is_dedicated_adam_browser_request(
        "In Adam's isolated browser, read the report and save the result to memory."
    )
    assert _is_read_only_adam_browser_request(
        "In Adam's isolated browser, read the report; do not navigate or change anything."
    )
    assert not _is_read_only_adam_browser_request(
        "In Adam's isolated browser, make a restaurant reservation."
    )
    assert not _is_read_only_adam_browser_request(
        "In Adam's isolated browser, open the booking page."
    )
    assert not _is_read_only_adam_browser_request(
        "In Adam's isolated browser, handle this for me."
    )
    assert not _is_read_only_adam_browser_request(
        "In Adam's browser, read the report and click its Next link."
    )
    assert not _is_read_only_adam_browser_request(
        "In Adam's browser, read the report; do not click Next, but fill the search field."
    )
    assert not _is_read_only_adam_browser_request(
        "Read the report in my regular browser."
    )
    selected = _filter_tools_for_dedicated_desktop_navigation(
        ADAM_TOOLS,
        "In Adam's isolated browser, read the report and save the result to memory.",
    )
    selected_names = {tool.name for tool in selected}
    assert "browser_navigation" in selected_names
    assert "manage_memory" in selected_names
    browser_only = _filter_tools_for_dedicated_desktop_navigation(
        ADAM_TOOLS,
        "In Adam's isolated browser, read this webpage and summarize it.",
    )
    assert {tool.name for tool in browser_only} == {"browser_navigation"}


@pytest.mark.asyncio
async def test_independent_native_web_reads_run_concurrently_and_keep_call_order():
    import asyncio
    from src.llm.brain import AdamBrain

    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {"content": "", "tool_calls": [
            {"id": "search-a", "function": {"name": "web_search", "arguments": {"query": "alpha"}}},
            {"id": "fetch-b", "function": {"name": "fetch_webpage", "arguments": {"url": "https://example.test/b"}}},
        ]},
        {"content": "Both lookups completed.", "tool_calls": []},
    ])
    active = 0
    peak_active = 0

    async def fake_execute(name, args):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return f"{name}:{args.get('query', args.get('url'))}"

    brain._execute_tool = fake_execute
    with patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
        await brain.process_user_utterance("Search alpha and read this page")

    tool_messages = [message for message in brain.messages if message.get("role") == "tool"]
    assert peak_active == 2
    assert [message["tool_call_id"] for message in tool_messages] == ["search-a", "fetch-b"]
    assert [json.loads(message["content"])["data"] for message in tool_messages] == [
        "web_search:alpha", "fetch_webpage:https://example.test/b",
    ]


@pytest.mark.asyncio
async def test_sync_web_tools_are_offloaded_so_parallel_batching_does_not_block():
    import asyncio
    import time
    from src.llm.brain import AdamBrain

    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    active = 0
    peak_active = 0

    def fake_search(query):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        time.sleep(0.05)
        active -= 1
        return query

    with patch("src.llm.brain.web_search", side_effect=fake_search):
        results = await asyncio.gather(
            brain._execute_tool("web_search", {"query": "alpha"}),
            brain._execute_tool("web_search", {"query": "beta"}),
        )

    assert peak_active == 2
    assert results == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_long_wait_speaks_progress_but_short_wait_does_not(monkeypatch):
    import asyncio
    import src.llm.brain as brain_module
    from src.llm.brain import AdamBrain

    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    monkeypatch.setattr(brain_module, "LONG_TASK_PROGRESS_INTERVAL_SECONDS", 0.005)

    quick = await brain._await_with_progress(asyncio.sleep(0, result="quick"))
    assert quick == "quick"
    assert brain.tts.spoken == []

    slow = await brain._await_with_progress(asyncio.sleep(0.02, result="done"))
    assert slow == "done"
    assert brain.tts.spoken == ["I’m still working through your request."]


@pytest.mark.asyncio
async def test_explicit_screen_read_uses_ocr_evidence_without_tools_or_image():
    from src.llm.brain import AdamBrain, _is_explicit_screen_read_request

    class Controller:
        def __init__(self):
            self.kwargs = None

        def run(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                status="ok",
                ocr_regions=[SimpleNamespace(
                    text="Quarterly revenue: $1.2M", top=100, left=200,
                )],
                message=(
                    "Current monitor capture is 1920x1080.\n"
                    "Extracted screen text:\n"
                    "R1 text='Quarterly revenue: $1.2M' confidence=0.98"
                ),
            )

    request = "What is on my screen, can you read it to me?"
    assert _is_explicit_screen_read_request(request)
    assert not _is_explicit_screen_read_request("What's on my screen right now?")

    brain = AdamBrain.__new__(AdamBrain)
    brain.system_prompt = "System"
    brain.messages = [{"role": "system", "content": "System"}]
    brain.screen_ocr = object()
    brain.computer_controller = Controller()
    brain.config = SimpleNamespace(computer_control=SimpleNamespace(screenshot_delay_seconds=0.25))
    brain.tts = _DummyTTS()
    brain._is_interrupted = False
    brain.llm_client = _DummyClient("custom", [])

    await brain._process_user_utterance_impl(request)

    assert brain.computer_controller.kwargs["include_ocr"] is True
    assert brain.computer_controller.kwargs["include_visual_grounding"] is False
    assert brain.llm_client.requests == []
    assert brain.tts.spoken == ["The readable text on the screen is: Quarterly revenue: $1.2M"]


@pytest.mark.asyncio
async def test_explicit_screen_read_reports_capture_failure_without_calling_llm():
    from src.llm.brain import AdamBrain

    class Controller:
        def run(self, **kwargs):
            raise RuntimeError("display server unavailable")

    brain = AdamBrain.__new__(AdamBrain)
    brain.system_prompt = "System"
    brain.messages = [{"role": "system", "content": "System"}]
    brain.screen_ocr = object()
    brain.computer_controller = Controller()
    brain.config = SimpleNamespace(computer_control=SimpleNamespace(screenshot_delay_seconds=0.25))
    brain.tts = _DummyTTS()
    brain._is_interrupted = False
    brain.llm_client = _DummyClient("custom", [])

    await brain._process_user_utterance_impl("Please read my screen to me")

    assert brain.llm_client.requests == []
    assert "couldn't capture" in brain.tts.spoken[0]


@pytest.mark.asyncio
async def test_assistant_envelope_returned_by_tool_is_not_treated_as_tool_success():
    from src.llm.brain import AdamBrain

    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {"content": "", "tool_calls": [{
            "id": "malformed-tool-result",
            "function": {"name": "get_now_playing", "arguments": {}},
        }]},
        {"content": "The observation path returned an invalid tool result.", "tool_calls": []},
    ])

    async def malformed_result(_name, _args):
        return {"role": "assistant", "content": "", "tool_calls": [{"id": "nested"}]}

    brain._execute_tool = malformed_result
    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Check the current playback state")

    result = next(message for message in brain.messages if message.get("role") == "tool")
    payload = json.loads(result["content"])
    assert payload["status"] == "failed"
    assert "assistant completion envelope" in payload["data"]


@pytest.mark.asyncio
async def test_openai_compatible_history_serializes_native_argument_objects():
    from src.llm.brain import AdamBrain

    tts = _DummyTTS()
    brain = AdamBrain(_config("custom"), None, None, None, tts)
    brain.llm_client = _DummyClient("custom", [
        {
            "content": "",
            "tool_calls": [{
                "id": "math-call-4",
                "function": {"name": "calculate_math", "arguments": {"expression": "6 * 7"}},
            }],
        },
        {"content": "The answer is 42.", "tool_calls": []},
    ])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"), patch(
        "src.llm.brain.calculate_math", return_value="42"
    ):
        await brain.process_user_utterance("What is six times seven?")

    assistant_call = next(message for message in brain.llm_client.requests[1] if message.get("tool_calls"))
    tool_call = assistant_call["tool_calls"][0]
    assert tool_call["id"] == "math-call-4"
    assert tool_call["function"]["arguments"] == '{"expression": "6 * 7"}'
    tool_result = next(message for message in brain.llm_client.requests[1] if message.get("role") == "tool")
    assert tool_result["tool_call_id"] == "math-call-4"


@pytest.mark.asyncio
async def test_agent_loop_supports_file_creation_then_readback(tmp_path):
    from src.llm.brain import AdamBrain

    target = tmp_path / "meeting-notes.txt"
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {
            "content": "",
            "tool_calls": [{
                "id": "create-notes",
                "function": {"name": "create_file", "arguments": {
                    "path": str(target), "content": "Decisions\n- Ship the prototype\n"
                }},
            }],
        },
        {
            "content": "",
            "tool_calls": [{
                "id": "read-notes",
                "function": {"name": "read_file", "arguments": {"path": str(target)}},
            }],
        },
        {"content": "I created the notes and confirmed their contents.", "tool_calls": []},
    ])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Create meeting notes and check what you wrote")

    assert target.read_text() == "Decisions\n- Ship the prototype\n"
    results = [message for message in brain.messages if message.get("role") == "tool"]
    assert [message["tool_call_id"] for message in results] == ["create-notes", "read-notes"]
    assert "Decisions" in results[1]["content"]


@pytest.mark.asyncio
async def test_agent_loop_reobserves_between_desktop_sequence_and_goal_assessment():
    from src.llm.brain import AdamBrain
    from src.tools.computer_control import ComputerControlResult

    screenshot = b"fresh-screen"
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())

    class Controller:
        available = True
        coordinate_mode = "pixels"

        def __init__(self):
            self.calls = []

        def run(self, action, **kwargs):
            self.calls.append((action, kwargs))
            return ComputerControlResult(
                "Inspected the current application. Snapshot ID: snap-1",
                screenshot=screenshot,
                status="ok",
                dispatched=False,
                snapshot_id="snap-1",
            )

        def run_sequence(self, **kwargs):
            self.calls.append(("sequence", kwargs))
            return ComputerControlResult(
                "Typed the requested search query. Snapshot ID: snap-2",
                screenshot=screenshot,
                status="ok",
                dispatched=True,
                snapshot_id="snap-2",
            )

        def invalidate_snapshot(self):
            return None

    controller = Controller()
    brain.computer_controller = controller
    brain.llm_client = _DummyClient("local", [
        {
            "content": "",
            "tool_calls": [{"id": "observe", "function": {
                "name": "computer_control",
                "arguments": {"action": "inspect", "scope": "window", "snapshot_id": ""},
            }}],
        },
        {
            "content": "",
            "tool_calls": [{"id": "input", "function": {
                "name": "computer_control",
                "arguments": {
                    "action": "sequence",
                    "snapshot_id_snap-1": "true",
                    "include_ocr_after": "true",
                    "actions": [
                        {"action": "click", "x": 210, "y": 80},
                        {"action": "type", "text": "requested search"},
                    ],
                },
            }}],
        },
        {"content": "The query is entered in the search field.", "tool_calls": []},
    ])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Enter the requested search text")

    assert [action for action, _ in controller.calls] == ["inspect", "sequence"]
    sequence_call = controller.calls[1][1]
    assert sequence_call["snapshot_id"] == "snap-1"
    assert sequence_call["include_ocr"] is True
    assert sequence_call["actions"][1]["text"] == "requested search"
    assert any(message.get("images") == [screenshot] for message in brain.llm_client.requests[1])
    assert any(message.get("images") == [screenshot] for message in brain.llm_client.requests[2])
    assert brain.tts.spoken[-1] == "The query is entered in the search field."


@pytest.mark.asyncio
async def test_standalone_desktop_wait_becomes_bounded_fresh_inspection():
    from unittest.mock import Mock
    from src.llm.brain import AdamBrain

    brain = AdamBrain.__new__(AdamBrain)
    brain.speculative_router = None
    brain.custom_tool_mgr = SimpleNamespace(has_tool=lambda _name: False)
    brain.computer_controller = SimpleNamespace(
        available=True,
        run=Mock(return_value=SimpleNamespace(message="Inspected screen", screenshot=b"image")),
    )
    brain.messages = [{"role": "system", "content": "system"}]
    brain._pending_screenshot = None

    result = await brain._execute_tool_impl(
        "computer_control", {"action": "wait", "seconds": 5}
    )

    assert result.message == "Inspected screen"
    brain.computer_controller.run.assert_called_once()
    call = brain.computer_controller.run.call_args.kwargs
    assert call["action"] == "inspect"
    assert call["screenshot_delay_seconds"] == 5
    assert brain._pending_screenshot == b"image"


@pytest.mark.asyncio
async def test_file_write_failure_returns_failed_execution_status(tmp_path):
    from src.llm.brain import AdamBrain

    target = tmp_path / "existing.txt"
    target.write_text("keep existing content")
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {
            "content": "",
            "tool_calls": [{
                "id": "protected-write",
                "function": {"name": "write_file", "arguments": {
                    "path": str(target), "content": "replacement"
                }},
            }],
        },
        {"content": "The file already exists, so I left it unchanged.", "tool_calls": []},
    ])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Update the existing note")

    result = next(message for message in brain.llm_client.requests[1] if message.get("role") == "tool")
    payload = json.loads(result["content"])
    assert payload["status"] == "failed"
    assert payload["effect_status"] == "unknown"
    assert "refusing to overwrite" in payload["data"]
    assert target.read_text() == "keep existing content"


@pytest.mark.asyncio
async def test_process_status_turn_keeps_the_dedicated_tools_for_model_selection():
    from src.llm.brain import AdamBrain

    memory = SimpleNamespace(retrieve_context=lambda _text: None)
    brain = AdamBrain(_config("custom"), None, None, None, _DummyTTS(), memory_mgr=memory)
    brain.llm_client = _DummyClient("custom", [
        {"content": "I can check that.", "tool_calls": []},
    ])

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance(
            "Report current CPU and memory status, list processes, and explain what the results suggest."
        )

    sent_tools = brain.llm_client.request_tools[0]
    assert [tool.name for tool in sent_tools] == ["get_system_status", "list_processes"]


@pytest.mark.asyncio
async def test_simple_system_status_skips_model_and_speaks_validated_tool_result():
    from src.llm.brain import AdamBrain

    status = "CPU load average is 0.10. Memory is 40 percent in use. GPU utilization is 2 percent."
    brain = AdamBrain(_config("custom"), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("custom", [])

    async def return_status(_name, _args):
        return status

    brain._execute_tool = return_status
    with patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
        await brain.process_user_utterance("Report current CPU, RAM, and GPU utilization.")

    assert len(brain.llm_client.requests) == 0
    assert brain.tts.spoken == [status]
    assert brain.messages[-1] == {"role": "assistant", "content": status}

@pytest.mark.asyncio
async def test_failed_tool_reprompts_model_to_correct_path_without_replaying_success():
    from unittest.mock import AsyncMock
    from src.llm.brain import AdamBrain
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    def call(path):
        return {'content':'', 'tool_calls':[{'function':{'name':'read_file','arguments':{'path':path}}}]}
    brain.llm_client = _DummyClient('local', [call('/wrong/note.md'), call('/right/note.md'),
                                            {'content':'Read the note.', 'tool_calls':[]},
                                            {'content':'Read the note.', 'tool_calls':[]}])
    brain._execute_tool = AsyncMock(side_effect=[FileNotFoundError('wrong path'), 'Verified note'])
    with patch('src.llm.brain.get_open_windows_prompt_context',return_value='Desktop'):
        await brain.process_user_utterance('Read my note file')
    assert [c.args[1]['path'] for c in brain._execute_tool.await_args_list] == ['/wrong/note.md','/right/note.md']
    request = brain.llm_client.requests[1]
    assert any('Tool recovery 1/3' in str(m.get('content')) for m in request)
    assert any('FileNotFoundError' in str(m.get('content')) for m in request)
    assert brain.tts.spoken[-1] == 'Read the note.'


@pytest.mark.parametrize("status", ["partial", "uncertain"])
@pytest.mark.asyncio
async def test_unconfirmed_tool_outcome_gets_a_focused_recovery_prompt(status):
    from unittest.mock import AsyncMock
    from src.llm.brain import AdamBrain
    from src.tools.computer_control import ComputerControlResult

    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient("local", [
        {"content": "", "tool_calls": [{"function": {
            "name": "read_file", "arguments": {"path": "/tmp/probe.txt"},
        }}]},
        {"content": "The read did not confirm the requested result.", "tool_calls": []},
        {"content": "The read did not confirm the requested result.", "tool_calls": []},
    ])
    brain._execute_tool = AsyncMock(return_value=ComputerControlResult(
        "The operation may have run, but its result was not confirmed.", status=status,
    ))

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Desktop"):
        await brain.process_user_utterance("Read my probe file")

    recovery_request = brain.llm_client.requests[1]
    assert any("Tool recovery 1/3" in str(message.get("content")) for message in recovery_request)
    assert any("inspect its current effects before repeating it" in str(message.get("content"))
               for message in recovery_request)


@pytest.mark.asyncio
async def test_failed_tools_stop_after_three_correction_prompts():
    from unittest.mock import AsyncMock
    from src.llm.brain import AdamBrain
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient('local', [
        {'content':'', 'tool_calls':[{'function':{'name':'read_file','arguments':{'path':f'/missing/{i}'}}}]}
        for i in range(4)])
    brain._execute_tool = AsyncMock(return_value='{"ok": false, "error": "File unavailable"}')
    with patch('src.llm.brain.get_open_windows_prompt_context',return_value='Desktop'):
        await brain.process_user_utterance('Read my note file')
    assert brain._execute_tool.await_count == 4
    assert len(brain.llm_client.requests) == 4
    assert 'three recovery prompts' in brain.tts.spoken[-1]
    assert not any('completed' in str(m.get('content','')).lower() for m in brain.messages[-1:])


@pytest.mark.asyncio
async def test_exhausted_provider_error_does_not_trigger_another_synthesis_or_tool():
    from unittest.mock import AsyncMock
    from src.llm.brain import AdamBrain
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient('local', [{'content':'Provider unavailable', 'provider_error':True}])
    brain._execute_tool = AsyncMock()
    with patch('src.llm.brain.get_open_windows_prompt_context',return_value='Desktop'):
        await brain.process_user_utterance('Read my note file')
    assert len(brain.llm_client.requests) == 1
    brain._execute_tool.assert_not_awaited()
    assert brain.tts.spoken == ['Provider unavailable']


@pytest.mark.asyncio
async def test_nonzero_shell_exit_is_a_failure_even_with_output():
    from src.llm.brain import AdamBrain
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    output = await brain._execute_tool('run_bash_command', {'command':'printf test-output; exit 2'})
    result = json.loads(output)
    assert result == {'ok':False, 'exit_code':2, 'output':'test-output'}

@pytest.mark.asyncio
async def test_cancelled_tool_is_not_reprompted():
    import asyncio
    from unittest.mock import AsyncMock
    from src.llm.brain import AdamBrain
    brain = AdamBrain(_config(), None, None, None, _DummyTTS())
    brain.llm_client = _DummyClient('local', [
        {'content':'', 'tool_calls':[{'function':{'name':'read_file','arguments':{'path':'/tmp/a'}}}]}])
    brain._execute_tool = AsyncMock(side_effect=asyncio.CancelledError)
    with patch('src.llm.brain.get_open_windows_prompt_context',return_value='Desktop'):
        with pytest.raises(asyncio.CancelledError):
            await brain.process_user_utterance('Read my note file')
    assert len(brain.llm_client.requests) == 1
    assert brain._execute_tool.await_count == 1
