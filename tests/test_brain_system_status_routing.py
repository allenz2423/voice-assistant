from types import SimpleNamespace
import json

import pytest
import asyncio
from unittest.mock import patch

from src.llm.brain import (
    AdamBrain,
    COMPACT_CONVERSATION_SYSTEM_PROMPT,
    _can_direct_dispatch_system_status,
    _can_answer_without_tools,
    _desktop_no_progress_repeats,
    _desktop_unchanged_screen_count,
    _desktop_screenshot_signature,
    _explicitly_requests_expanded_capture,
    _capture_scope_validation_error,
    _desktop_screens_match,
    _desktop_action_signature,
    _application_launch_key,
    _should_block_unverified_application_launch,
    _filter_tools_for_dedicated_desktop_navigation,
    _is_dedicated_desktop_navigation_request,
    _is_bounded_visible_form_request,
    _filter_tools_for_system_status,
    _is_memory_only_recall_request,
    _is_desktop_context_request,
    _desktop_task_acknowledgment,
    _should_acknowledge_desktop_task,
    _should_use_initial_ocr_for_desktop_request,
    _summarize_desktop_readback_if_generic,
    _desktop_tool_evidence_for_final_answer,
    _is_dedicated_system_status_request,
    _is_browser_app,
    _is_explicit_short_comparison_request,
    _has_two_short_sentence_shape,
    _should_review_short_comparison,
    _should_use_compact_conversation_prompt,
    _can_route_to_tool_free_model,
)


@pytest.mark.parametrize("prompt", [
    "Which works better with chunky tomato sauce, rigatoni or penne?",
    "Explain entropy in plain language.",
    "Explain time complexity in plain language.",
    "What's time complexity?",
    "What is time complexity?",
    "Write a concise thank-you note for my neighbor.",
    "Write a concise thank-you note for a neighbor who watered my plants.",
    "Tell me a joke.",
    "Could you tell me a joke?",
    "List three ways to make a paragraph clearer.",
    "Could you pick a number between 1 and 10?",
])
def test_ordinary_conversation_can_skip_irrelevant_tool_schemas(prompt):
    assert _can_answer_without_tools(prompt)


@pytest.mark.parametrize("prompt", [
    "What's the weather in Boston today?",
    "What time is it?",
    "What's the current date?",
    "What time is it in Tokyo?",
    "What day is it?",
    "Tell me the time.",
    "Search for the latest NVIDIA news.",
    "Check my CPU and RAM usage.",
    "What did I say about my work hours last week?",
    "What's my favorite food?",
    "Could you tell me what my favorite food is?",
    "Which of my writing drafts is better?",
    "Set a timer for ten minutes.",
    "Read https://example.com and summarize it.",
    "Write this answer into a file.",
    "Write this thank-you note into my notes.",
    "can you check my downloads folder?",
    "Is my downloads folder a mess?",
    "How organized are the folders in Documents?",
    "Are there duplicates in ~/Downloads?",
    "No, the other terminal.",
    "Show the files in my Documents directory.",
    "What's in my Downloads?",
    "List the folders at ~/workspace.",
])
def test_tool_or_live_information_intent_keeps_tools_available(prompt):
    assert not _can_answer_without_tools(prompt)


def test_compact_conversation_prompt_requires_no_external_context():
    assert _should_use_compact_conversation_prompt("Explain entropy in plain language.")
    assert not _should_use_compact_conversation_prompt("What time is it?")
    assert not _should_use_compact_conversation_prompt("My processor is pegged.")
    assert not _should_use_compact_conversation_prompt(
        "Explain entropy in plain language.", memory_context="The user studies physics."
    )
    assert not _should_use_compact_conversation_prompt(
        "What do you think?", needs_desktop_context=True
    )
    assert not _should_use_compact_conversation_prompt(
        "Explain entropy in plain language.", skill_context="Specialized guidance"
    )


def test_compact_conversation_prompt_defaults_to_short_plain_comparisons():
    assert "one or two short sentences by default" in COMPACT_CONVERSATION_SYSTEM_PROMPT
    assert "state the main difference first" in COMPACT_CONVERSATION_SYSTEM_PROMPT
    assert "avoid tables and lists unless requested" in COMPACT_CONVERSATION_SYSTEM_PROMPT
    assert "When the user asks for detail, examples, or a list, provide them" in COMPACT_CONVERSATION_SYSTEM_PROMPT


@pytest.mark.parametrize("prompt", [
    "Compare rigatoni and penne in two short sentences.",
    "Compare penguins and birds in two short sentences.",
    "Compare iron and copper in two short sentences.",
])
def test_experimental_comparison_trigger_is_entity_agnostic_and_covers_f1(prompt):
    assert _is_explicit_short_comparison_request(prompt)
    assert _should_review_short_comparison(prompt, enabled=True)


def test_experimental_comparison_trigger_does_not_rely_on_plain_chat_classifier():
    prompt = "Compare rigatoni and penne in two short sentences."
    assert not _can_answer_without_tools(prompt)
    assert _should_review_short_comparison(prompt, enabled=True)


@pytest.mark.parametrize("prompt", [
    "Compare rigatoni and penne in two short sentences using this source.",
    "Compare rigatoni and penne in two short sentences, and search the web.",
    "Compare my notes and your notes in two short sentences.",
    "Compare rigatoni and penne in two short sentences, then write the answer to a file.",
])
def test_experimental_comparison_trigger_excludes_source_action_and_personal_context(prompt):
    assert not _is_explicit_short_comparison_request(prompt)
    assert not _should_review_short_comparison(prompt, enabled=True)


@pytest.mark.parametrize("context", [
    {"memory_context": "A prior user fact."},
    {"skill_context": "Specialized guidance."},
    {"has_image": True},
    {"needs_desktop_context": True},
    {"has_task_evidence": True},
    {"has_prior_context": True},
    {"tools_executed": True},
])
def test_experimental_comparison_review_requires_clean_standalone_turn(context):
    assert not _should_review_short_comparison(
        "Compare rigatoni and penne in two short sentences.",
        enabled=True,
        **context,
    )


def test_experimental_comparison_review_is_disabled_by_default():
    from src.config import LLMConfig

    assert LLMConfig().experimental_factual_comparison_review is False


@pytest.mark.parametrize("answer", [
    "Rigatoni is a ridged tube; penne has angled ends. Both are pasta.",
    "Penguins are birds, but many other birds can fly. Iron conducts electricity, and copper does too.",
    "Iron conducts electricity. Copper conducts heat.",
])
def test_review_output_validator_accepts_two_short_sentence_spans(answer):
    assert _has_two_short_sentence_shape(answer)


@pytest.mark.parametrize("answer", [
    "Rigatoni is a ridged tube.",
    "Rigatoni is a ridged tube. Penne has angled ends. Both are pasta.",
    "Rigatoni is a ridged tube Penne has angled ends.",
    "Iron conducts electricity. .",
    (
        "Rigatoni is a ridged tube with a long and compacted shape that can hold sauce in many different "
        "ways across a wide variety of dishes while remaining sturdy during cooking and serving to large "
        "families. Penne has angled ends."
    ),
])
def test_review_output_validator_rejects_wrong_count_empty_span_or_long_answer(answer):
    assert not _has_two_short_sentence_shape(answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("review_result, expected_text", [
    (
        {"content": "Rigatoni is a ridged tube; penne has angled ends. Both are pasta.", "tool_calls": []},
        "Rigatoni is a ridged tube; penne has angled ends. Both are pasta.",
    ),
    ({"content": "", "tool_calls": []}, "Draft answer."),
    ({"content": "", "provider_error": True, "tool_calls": []}, "Draft answer."),
    ({"content": "Unexpected tool request.", "tool_calls": [{"id": "review-tool"}]}, "Draft answer."),
    ({"content": "Rigatoni is a ridged tube.", "tool_calls": []}, "Draft answer."),
    (
        {"content": "Rigatoni is ridged. Penne has angled ends. Both are pasta.", "tool_calls": []},
        "Draft answer.",
    ),
    ({"content": "Iron conducts electricity. .", "tool_calls": []}, "Draft answer."),
    (RuntimeError("synthetic review failure"), "Draft answer."),
])
async def test_experimental_comparison_review_is_bounded_and_preserves_draft_on_failure(
    review_result, expected_text,
):
    class SilentTTS:
        engine = "silent"
        pending_barge_in_text = None

        def __init__(self):
            self.spoken = []

        async def speak_async(self, text):
            self.spoken.append(text)

    class Model:
        provider = "custom"

        def __init__(self):
            self.calls = []

        async def chat(self, messages, tools=None, max_tokens=None, **_kwargs):
            self.calls.append({
                "messages": [dict(message) for message in messages],
                "tools": tools,
                "max_tokens": max_tokens,
            })
            if len(self.calls) == 1:
                return {"content": "Draft answer.", "tool_calls": []}
            if isinstance(review_result, Exception):
                raise review_result
            return review_result

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="test", cloud_model="test",
        ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
        api_key="", temperature=0, num_ctx=8192, max_tool_rounds=4,
        experimental_factual_comparison_review=True,
    ))
    tts = SilentTTS()
    brain = AdamBrain(config, None, None, None, tts)
    model = Model()
    brain.llm_client = model
    brain.skill_manager.get_matched_skill_context = lambda _query: None
    tool_schema = SimpleNamespace(name="search_web")
    brain.get_tools = lambda: [tool_schema]

    with patch(
        "src.llm.brain.get_open_windows_prompt_context",
        return_value="Open window: Mozilla Firefox.",
    ):
        await brain.process_user_utterance("Compare rigatoni and penne in two short sentences.")

    assert len(model.calls) == 2
    assert model.calls[0]["max_tokens"] is None
    assert model.calls[1]["max_tokens"] == 192
    assert model.calls[0]["tools"] == [tool_schema]
    assert model.calls[1]["tools"] == []
    assert "Mozilla Firefox" in model.calls[0]["messages"][-1]["content"]
    assert "Draft answer." in model.calls[1]["messages"][1]["content"]
    assert tts.spoken == [expected_text]


@pytest.mark.asyncio
async def test_comparison_review_does_not_change_tool_bearing_request_route():
    class SilentTTS:
        engine = "silent"
        pending_barge_in_text = None

        async def speak_async(self, _text):
            pass

    class Model:
        async def chat(self, messages, tools=None, max_tokens=None, **_kwargs):
            self.calls = getattr(self, "calls", [])
            self.calls.append({"tools": tools, "max_tokens": max_tokens})
            return {"content": "A source-backed comparison.", "tool_calls": []}

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="test", cloud_model="test",
        ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
        api_key="", temperature=0, num_ctx=8192, max_tool_rounds=4,
        experimental_factual_comparison_review=True,
    ))
    brain = AdamBrain(config, None, None, None, SilentTTS())
    model = Model()
    brain.llm_client = model
    brain.skill_manager.get_matched_skill_context = lambda _query: None
    tool_schema = SimpleNamespace(name="search_web")
    brain.get_tools = lambda: [tool_schema]

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
        await brain.process_user_utterance(
            "Compare rigatoni and penne in two short sentences, and search the web for sources."
        )

    assert len(model.calls) == 1
    assert model.calls[0]["tools"] == [tool_schema]


def test_optional_tool_free_model_route_accepts_only_standalone_generic_chat():
    assert _can_route_to_tool_free_model(
        "Explain photosynthesis in one sentence.",
        compact_conversation=True,
        has_image=False,
        memory_only_query=False,
    )
    for prompt, has_image, memory_only_query in (
        ("What did I say about work last week?", False, False),
        ("Why is it blue?", False, False),
        ("Explain photosynthesis.", True, False),
        ("Explain photosynthesis.", False, True),
    ):
        assert not _can_route_to_tool_free_model(
            prompt,
            compact_conversation=True,
            has_image=has_image,
            memory_only_query=memory_only_query,
        )


def test_browser_name_detection_does_not_match_part_of_an_unrelated_app_name():
    assert _is_browser_app("Zen Browser")
    assert _is_browser_app("Mozilla Firefox")
    assert not _is_browser_app("SHENZHEN I/O")
    assert not _is_browser_app("Zenith Image Editor")
    assert not _is_browser_app("Edgecase Editor")


def test_desktop_no_progress_breaker_stops_after_one_unchanged_repeat():
    same_screen_click = ('{"action":"click","x":400,"y":300}', "unchanged-screen")
    repeats, tripped = _desktop_no_progress_repeats(None, same_screen_click, 0)
    assert (repeats, tripped) == (0, False)
    repeats, tripped = _desktop_no_progress_repeats(same_screen_click, same_screen_click, repeats)
    assert (repeats, tripped) == (1, True)
    repeats, tripped = _desktop_no_progress_repeats(
        same_screen_click, ("different-action", "new-screen"), repeats
    )
    assert (repeats, tripped) == (0, False)


def test_desktop_stagnation_breaker_stops_different_actions_after_three_unchanged_screens():
    count, tripped = _desktop_unchanged_screen_count(None, "same-screen", 0)
    assert (count, tripped) == (0, False)
    count, tripped = _desktop_unchanged_screen_count("same-screen", "same-screen", count)
    assert (count, tripped) == (1, False)
    count, tripped = _desktop_unchanged_screen_count("same-screen", "same-screen", count)
    assert (count, tripped) == (2, False)
    count, tripped = _desktop_unchanged_screen_count("same-screen", "same-screen", count)
    assert (count, tripped) == (3, True)


def test_desktop_stagnation_counter_resets_when_screen_changes():
    count, tripped = _desktop_unchanged_screen_count("old-screen", "new-screen", 2)
    assert (count, tripped) == (0, False)


def test_expanded_capture_requires_explicit_user_scope():
    assert not _explicitly_requests_expanded_capture(
        'In Writer, type "Laptop UI smoke test" and tell me the exact text.'
    )
    assert _explicitly_requests_expanded_capture("Capture the entire desktop.")
    assert _explicitly_requests_expanded_capture("Show me a screenshot of both monitors.")


@pytest.mark.parametrize("tool_name", ["computer_control", "capture_screenshot", "observe_desktop"])
def test_screenshot_tools_reject_unrequested_expanded_scope(tool_name):
    prompt = 'In Writer, type "Laptop UI smoke test" and tell me the exact text.'
    assert _capture_scope_validation_error(tool_name, {"scope": "monitor"}, prompt)
    assert _capture_scope_validation_error(tool_name, {"scope": "desktop"}, prompt)
    assert _capture_scope_validation_error(
        tool_name, {"scope": "monitor"}, "Take a screenshot of the entire monitor."
    ) is None
    assert _capture_scope_validation_error(tool_name, {"scope": "window"}, prompt) is None


def test_generic_desktop_ack_reads_back_explicitly_requested_visible_text():
    prompt = 'In Writer, type "Laptop UI smoke test" and tell me the exact text.'
    evidence = "O1 text='Laptop UI smoke test' center=(329,471) box=(184,450,475,492) confidence=1.00"
    assert _summarize_desktop_readback_if_generic(prompt, "Done.", evidence) == (
        "The text shown is: Laptop UI smoke test"
    )
    unrelated = "O1 text='Different screen text' center=(329,471) box=(184,450,475,492) confidence=1.00"
    assert _summarize_desktop_readback_if_generic(prompt, "Done.", unrelated) is None


def test_stale_desktop_ocr_is_compacted_but_recent_screen_evidence_is_preserved():
    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="test", cloud_model="test",
        ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
        api_key="", temperature=0, num_ctx=8192, max_tool_rounds=4,
    ))
    brain = AdamBrain(config, None, None, None, None)
    older = {
        "role": "tool", "name": "computer_control",
        "content": json.dumps({
            "status": "ok", "data": "Clicked Next.\nOCR text regions (coordinates use pixels):\n"
            "O1 text='Welcome' center=(1,2)\nO2 text='Next' center=(3,4)"
        }),
    }
    current = {
        "role": "tool", "name": "computer_control",
        "content": json.dumps({
            "status": "ok", "data": "Typed requested text.\nOCR text regions (coordinates use pixels):\n"
            "O1 text='Laptop UI smoke test' center=(5,6)"
        }),
    }
    brain.messages = [older, current]

    assert brain._compact_stale_desktop_ocr() == 1
    old_data = json.loads(older["content"])
    current_data = json.loads(current["content"])
    assert old_data["status"] == "ok"
    assert "Clicked Next." in old_data["data"]
    assert "Welcome" not in old_data["data"]
    assert "Laptop UI smoke test" in current_data["data"]


def test_unverified_application_launch_blocks_only_the_same_target_for_current_request():
    failed = {"shenzhen i/o"}
    assert _application_launch_key(
        "launch_application", {"app_name": "  SHENZHEN   I/O "}
    ) == "shenzhen i/o"
    assert _should_block_unverified_application_launch(
        "launch_application", {"app_name": "Shenzhen I/O"}, failed
    )
    assert not _should_block_unverified_application_launch(
        "launch_application", {"app_name": "Steam"}, failed
    )
    assert not _should_block_unverified_application_launch(
        "computer_control", {"app_name": "Shenzhen I/O"}, failed
    )


def test_desktop_no_progress_breaker_ignores_small_live_screen_changes():
    from PIL import Image
    from io import BytesIO

    def png(edit=None):
        image = Image.new("RGB", (320, 180), (20, 20, 20))
        if edit:
            for x, y, color in edit:
                image.putpixel((x, y), color)
        output = BytesIO()
        image.save(output, format="PNG")
        return _desktop_screenshot_signature(output.getvalue())

    baseline = png()
    one_animated_pixel = png([(160, 90, (255, 255, 255))])
    major_scene_change = png([(x, y, (255, 255, 255)) for x in range(160) for y in range(90)])

    assert _desktop_screens_match(baseline, one_animated_pixel)
    assert not _desktop_screens_match(baseline, major_scene_change)


def test_ocr_text_change_counts_as_desktop_progress_when_pixels_look_unchanged():
    from src.tools.ocr import OCRRegion
    from PIL import Image
    from io import BytesIO

    image = Image.new("RGB", (320, 180), (20, 20, 20))
    output = BytesIO()
    image.save(output, format="PNG")
    screenshot = output.getvalue()
    changed_text = [OCRRegion("O1", "Quiet", 0.99, 100, 80, 150, 110)]
    same_text_repositioned = [OCRRegion("O2", "quiet", 0.80, 105, 85, 155, 115)]
    original_text = [OCRRegion("O1", "Normal", 0.99, 100, 80, 150, 110)]

    before = _desktop_screenshot_signature(screenshot, original_text)
    changed = _desktop_screenshot_signature(screenshot, changed_text)
    repositioned = _desktop_screenshot_signature(screenshot, same_text_repositioned)

    assert not _desktop_screens_match(before, changed)
    assert _desktop_screens_match(changed, repositioned)


def test_desktop_action_signature_matches_sequence_first_step_to_standalone_action():
    direct = {"action": "click", "x": 224, "y": 545, "include_ocr": True}
    sequence = {
        "action": "sequence",
        "actions": [
            {"action": "click", "x": 224, "y": 545},
            {"action": "click", "x": 224, "y": 545},
        ],
        "snapshot_id": "old-snapshot",
    }

    assert _desktop_action_signature("computer_control", direct) == _desktop_action_signature(
        "computer_control", sequence
    )


def test_desktop_action_signature_treats_small_click_jitter_as_same_target():
    first = {"action": "click", "x": 224, "y": 535}
    jittered = {"action": "click", "x": 228, "y": 525}
    other_control = {"action": "click", "x": 300, "y": 535}

    assert _desktop_action_signature("computer_control", first) == _desktop_action_signature(
        "computer_control", jittered
    )
    assert _desktop_action_signature("computer_control", first) != _desktop_action_signature(
        "computer_control", other_control
    )


def test_memory_only_recall_requires_matched_context_and_no_external_action():
    context = "- I worked on a synthetic project on Monday at 3:45 PM."
    assert _is_memory_only_recall_request("What times did I work this week?", context)
    assert _is_memory_only_recall_request("What's my favorite food?", context)
    assert not _is_memory_only_recall_request("What times did I work this week?", None)
    assert not _is_memory_only_recall_request(
        "What times did I work this week, and add them to my calendar?", context
    )


@pytest.mark.asyncio
async def test_personal_question_keeps_automatic_memory_retrieval():
    class SilentTTS:
        engine = "silent"
        pending_barge_in_text = None

        async def speak_async(self, text):
            self.last_text = text

    class Memory:
        def __init__(self):
            self.queries = []

        def retrieve_context(self, query):
            self.queries.append(query)
            return "Favorite food: rigatoni."

    class Model:
        async def chat(self, messages, tools=None, **_kwargs):
            self.messages = messages
            self.tools = tools
            return {"content": "You told me your favorite food is rigatoni.", "tool_calls": []}

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="test", cloud_model="test",
        ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
        api_key="", temperature=0, num_ctx=8192, max_tool_rounds=4,
    ))
    memory = Memory()
    tts = SilentTTS()
    brain = AdamBrain(config, None, None, None, tts, memory_mgr=memory)
    brain.llm_client = Model()
    brain.skill_manager.get_matched_skill_context = lambda _query: None

    window_reads = []
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "src.llm.brain.get_open_windows_prompt_context",
            lambda: window_reads.append(True) or "",
        )
        await brain.process_user_utterance("What's my favorite food?")

    assert memory.queries == ["What's my favorite food?"]
    user_message = next(
        message for message in brain.llm_client.messages
        if message.get("role") == "user"
    )
    assert "Favorite food: rigatoni." in user_message["content"]
    assert brain.llm_client.tools == []
    assert window_reads == []
    assert "Answer this personal-history or current-state question from the retrieved user memory" in (
        brain.llm_client.messages[0]["content"]
    )
    assert "rigatoni" in tts.last_text


@pytest.mark.asyncio
async def test_dedicated_desktop_action_gets_fresh_screen_on_first_model_turn():
    class SilentTTS:
        engine = "silent"
        pending_barge_in_text = None

        async def speak_async(self, _text):
            pass

    class Model:
        async def chat(self, messages, tools=None, **_kwargs):
            self.messages = messages
            self.tools = tools
            return {"content": "The screen is ready.", "tool_calls": []}

    class Memory:
        def retrieve_context(self, _query):
            return None

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="test", cloud_model="test",
        ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
        api_key="", temperature=0, num_ctx=8192, max_tool_rounds=4,
    ))
    brain = AdamBrain(config, None, None, None, SilentTTS(), memory_mgr=Memory())
    brain.llm_client = Model()
    brain.skill_manager.get_matched_skill_context = lambda _query: None
    captures = []

    def inspect(**kwargs):
        captures.append(kwargs)
        return SimpleNamespace(
            status="ok", screenshot=b"synthetic screenshot",
            message="Snapshot ID: initial123. Active window bounds: x=0..100, y=0..100.",
        )

    brain.computer_controller = SimpleNamespace(
        available=True, coordinate_mode="pixels", drag_active=False, run=inspect
    )
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "src.llm.brain.get_open_windows_prompt_context",
            lambda: "Test dialog is open.",
        )
        await brain.process_user_utterance("Select 5:00 PM in the local scheduler.")

    assert captures and captures[0]["action"] == "inspect"
    assert captures[0]["include_ocr"] is True
    user_message = next(
        message for message in brain.llm_client.messages
        if message.get("role") == "user"
    )
    assert user_message["images"] == [b"synthetic screenshot"]
    assert "Snapshot ID: initial123" in user_message["content"]


def test_initial_ocr_is_limited_to_desktop_tasks_with_explicit_text_values():
    assert _should_use_initial_ocr_for_desktop_request(
        "Select 5:00 PM in the local scheduler."
    )
    assert _should_use_initial_ocr_for_desktop_request(
        'Click "Save changes" in the current dialog.'
    )
    assert not _should_use_initial_ocr_for_desktop_request(
        "Click the visible settings button."
    )
    assert not _should_use_initial_ocr_for_desktop_request("What time is it?")


def test_quoted_screen_labels_do_not_disable_compact_ocr_prefetch():
    request = (
        "Click Advanced in Archive Preferences. Turn on ‘Show hidden files’, enter 500 "
        "in the entry labeled ‘Search index item limit’, leave ‘Automatically refresh "
        "folders’ checked, then click Apply and verify the result."
    )
    assert _is_desktop_context_request(request)
    assert _is_dedicated_desktop_navigation_request(request)
    assert _should_use_initial_ocr_for_desktop_request(request)

    available = [
        SimpleNamespace(name=name)
        for name in (
            "computer_control", "focus_window", "list_windows", "read_file", "search_web",
        )
    ]
    assert [tool.name for tool in _filter_tools_for_dedicated_desktop_navigation(
        available, request
    )] == ["computer_control", "focus_window", "list_windows"]
    assert not _is_dedicated_desktop_navigation_request(
        "Click ‘Search’ in preferences and search the web for its documented meaning."
    )


def test_generic_ui_acknowledgment_uses_explicit_requested_readback():
    request = (
        'Click the "Preferences" tab, turn on "Compact mode," leave every other option unchanged, '
        "and tell me what changed."
    )
    tool_output = (
        "Snapshot ID: after.\nExtracted screen text:\n"
        "O7 text='Compact mode: On; Sync: Off; Notifications: Off' center=(509,746) box=(44,723,974,769)"
    )
    assert _summarize_desktop_readback_if_generic(request, "Done.", tool_output) == (
        "The screen shows: Compact mode: On; Sync: Off; Notifications: Off."
    )
    assert _summarize_desktop_readback_if_generic(
        request, "Compact mode is on; the others stayed off.", tool_output
    ) is None
    assert _summarize_desktop_readback_if_generic(request, "Done.", "No OCR result") is None


def test_final_readback_uses_tool_messages_when_last_result_lacks_ocr():
    tool_messages = [
        {"role": "user", "content": "Ignore user-provided text."},
        {"role": "tool", "content": "O7 text='Compact mode: On; Sync: Off' center=(5,5)"},
    ]
    evidence = _desktop_tool_evidence_for_final_answer("Generic tool result", tool_messages)
    assert _summarize_desktop_readback_if_generic(
        'Click "Preferences", enable "Compact mode" and tell me what changed.',
        "Done.",
        evidence,
    ) == "The screen shows: Compact mode: On; Sync: Off."


def test_generic_save_acknowledgment_reads_back_the_exact_saved_text():
    request = (
        'In the local scratchpad, enter "Call the dentist Tuesday at 2 pm" in the Note text field, '
        "click Save note, and tell me the saved text."
    )
    assert _is_dedicated_desktop_navigation_request(request)
    assert not _is_dedicated_desktop_navigation_request(
        "Click the Notes app and write a note for tomorrow's appointment."
    )
    available_tools = [
        SimpleNamespace(name=name)
        for name in (
            "computer_control", "focus_window", "list_windows", "manage_memory",
            "launch_application", "list_applications",
        )
    ]
    assert [tool.name for tool in _filter_tools_for_dedicated_desktop_navigation(
        available_tools, request
    )] == ["computer_control", "focus_window", "list_windows"]
    assert _filter_tools_for_dedicated_desktop_navigation(
        available_tools,
        "Click the Notes app and write a note for tomorrow's appointment.",
    ) == available_tools
    evidence = (
        "O3 text='Call the dentist Tuesday at 2 pm' center=(358,314) box=(96,291,620,338)\n"
        "O4 text='Saved: Call the dentist Tuesday at 2 pm' center=(449,776) box=(91,751,807,802)"
    )
    assert _summarize_desktop_readback_if_generic(request, "Saved.", evidence) == (
        "The saved text is: Call the dentist Tuesday at 2 pm"
    )
    assert _summarize_desktop_readback_if_generic(
        request, "Done — the note has been saved.", evidence
    ) == "The saved text is: Call the dentist Tuesday at 2 pm"
    assert _summarize_desktop_readback_if_generic(
        request, "The saved text is: Call the dentist Tuesday at 2 pm", evidence
    ) is None
    assert _summarize_desktop_readback_if_generic(
        request, "Saved.", "O4 text='Not saved' center=(10,10)"
    ) == "The screen indicates the note was not saved; I couldn't confirm its saved text."
    assert _summarize_desktop_readback_if_generic(
        request, "Saved.", "No OCR result"
    ) == "I couldn't verify from the screen that the note was saved, so I can't confirm the saved text."
    punctuated_evidence = "O4 text='Saved: Review the draft.' center=(10,10)"
    punctuated_request = 'In the local scratchpad note field, type "Review the draft." and tell me the saved text.'
    assert _summarize_desktop_readback_if_generic(
        punctuated_request, "Saved.", punctuated_evidence
    ) == "The saved text is: Review the draft."


@pytest.mark.parametrize(("tool_evidence", "expected_response"), [
    (
        "O3 text='Call the dentist Tuesday at 2 pm' center=(358,314) box=(96,291,620,338)\n"
        "O4 text='Saved: Call the dentist Tuesday at 2 pm' center=(449,776) box=(91,751,807,802)",
        "The saved text is: Call the dentist Tuesday at 2 pm",
    ),
    (
        "O4 text='Not saved' center=(449,776) box=(91,751,807,802)",
        "The screen indicates the note was not saved; I couldn't confirm its saved text.",
    ),
    (
        "O4 text='Saved: Call the dentist Tuesday at 2 pm' center=(449,776) box=(91,751,807,802)\n"
        "O5 text='Not saved' center=(449,810) box=(91,790,807,835)",
        "The screen indicates the note was not saved; I couldn't confirm its saved text.",
    ),
    (
        "O3 text='Call the dentist Tuesday at 2 pm' center=(358,314) box=(96,291,620,338)",
        "I couldn't verify from the screen that the note was saved, so I can't confirm the saved text.",
    ),
])
@pytest.mark.asyncio
async def test_scratchpad_save_round_trip_uses_compact_tools_and_verified_readback(
    tool_evidence, expected_response,
):
    events = []

    class SilentTTS:
        engine = "silent"
        pending_barge_in_text = None

        def __init__(self):
            self.spoken = []

        async def speak_async(self, text):
            self.spoken.append(text)
            events.append(("speech", text))

    class Memory:
        def retrieve_context(self, _query):
            return None

    class Model:
        def __init__(self):
            self.responses = [
                {
                    "content": "",
                    "tool_calls": [{
                        "id": "save-note",
                        "function": {
                            "name": "computer_control",
                            "arguments": {
                                "action": "sequence",
                                "snapshot_id": "synthetic-scratchpad-1",
                                "actions": [
                                    {"action": "click", "target_text": "Note text field"},
                                    {"action": "type", "text": "Call the dentist Tuesday at 2 pm"},
                                    {"action": "click", "target_text": "Save note"},
                                ],
                            },
                        },
                    }],
                },
                {"content": "Saved.", "tool_calls": []},
            ]
            self.requested_tool_names = []

        async def chat(self, _messages, tools=None, **_kwargs):
            self.requested_tool_names.append([tool.name for tool in (tools or [])])
            return self.responses.pop(0)

        def format_tool_response(self, tool_call_id, tool_name, result):
            return {
                "role": "tool", "tool_call_id": tool_call_id,
                "name": tool_name, "content": result,
            }

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="test", cloud_model="test",
        ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
        api_key="", temperature=0, num_ctx=8192, max_tool_rounds=4,
    ))
    tts = SilentTTS()
    brain = AdamBrain(config, None, None, None, tts, memory_mgr=Memory())
    model = Model()
    brain.llm_client = model
    brain.skill_manager.get_matched_skill_context = lambda _query: None

    def inspect_fixture(**_kwargs):
        return SimpleNamespace(
            status="ok", screenshot=None,
            message="Synthetic local scratchpad inspection.",
            snapshot_id="synthetic-scratchpad-1",
        )

    brain.computer_controller = SimpleNamespace(
        available=True, coordinate_mode="pixels", drag_active=False,
        run=inspect_fixture,
    )
    calls = []

    async def execute_tool(name, args):
        calls.append((name, args))
        events.append(("tool", name))
        return tool_evidence

    brain._execute_tool = execute_tool
    request = (
        'In the local scratchpad, enter "Call the dentist Tuesday at 2 pm" in the Note text field, '
        "click Save note, and tell me the saved text."
    )
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "src.llm.brain.get_open_windows_prompt_context",
            lambda: "Synthetic scratchpad fixture.",
        )
        await brain.process_user_utterance(request)

    assert model.requested_tool_names == [[
        "computer_control", "list_windows", "focus_window",
    ], ["computer_control", "list_windows", "focus_window"]]
    assert len(calls) == 1
    assert calls[0][0] == "computer_control"
    assert calls[0][1]["actions"][1]["text"] == "Call the dentist Tuesday at 2 pm"
    assert events[0] == ("speech", "I’ll save the note and check the saved text.")
    assert events[1] == ("tool", "computer_control")
    assert tts.spoken[-1] == expected_response


def test_hardware_usage_question_uses_dedicated_status_routing():
    assert _is_dedicated_system_status_request(
        "Report current CPU, RAM, and GPU utilization."
    )


def test_explicit_shell_request_keeps_shell_tool_available():
    assert not _is_dedicated_system_status_request(
        "Run nvidia-smi and tell me what it reports."
    )


def test_general_hardware_question_does_not_trigger_status_routing():
    assert not _is_dedicated_system_status_request(
        "Explain how a CPU processes instructions."
    )
    assert not _is_dedicated_system_status_request(
        "What have I told you about my memory?"
    )


def test_compound_file_and_status_request_keeps_its_non_status_tools_available():
    prompt = (
        'Find the supplier in the local fixture report. Write one line to "task-output.txt". '
        "Then report current CPU core count and memory-use percentage."
    )
    tools = [
        SimpleNamespace(name="read_file"),
        SimpleNamespace(name="create_file"),
        SimpleNamespace(name="get_system_status"),
        SimpleNamespace(name="list_processes"),
    ]

    assert not _is_dedicated_system_status_request(prompt)
    assert _filter_tools_for_system_status(tools, prompt) == tools


def test_only_normalized_explicit_status_phrases_use_the_quickpath():
    for prompt in (
        "system status", "SYSTEM STATUS!", "System-status?",
        "cpu usage", "CPU USAGE!!!", "memory usage?", "disk-usage", "GPU usage.",
    ):
        assert _can_direct_dispatch_system_status(prompt), prompt

    for prompt in (
        "What's my CPU usage?",
        "Check current CPU usage.",
        "Report current CPU, RAM, and GPU utilization.",
        "Why is my CPU usage so high?",
        "My CPU usage is too high, ugh.",
        "Check system status and list running processes.",
    ):
        assert not _can_direct_dispatch_system_status(prompt), prompt


@pytest.mark.parametrize("prompt", [
    "Why is my CPU usage so high?",
    "My CPU usage is too high, ugh.",
    "Why is my CPU working so hard?",
    "My processor is pegged.",
    "My RAM is filling up; what's doing that?",
    "cpu usage eso",
])
def test_nonliteral_cpu_diagnostics_do_not_take_factual_status_fast_path(prompt):
    tools = [
        SimpleNamespace(name="get_system_status"),
        SimpleNamespace(name="list_processes"),
        SimpleNamespace(name="run_bash_command"),
    ]

    assert _is_dedicated_system_status_request(prompt)
    assert not _can_direct_dispatch_system_status(prompt)
    assert [tool.name for tool in _filter_tools_for_system_status(tools, prompt)] == [
        "get_system_status", "list_processes"
    ]


@pytest.mark.parametrize(("prompt", "expected", "omitted"), [
    (
        "CPU usage!!!",
        "CPU utilization was 19 percent during this status sample.",
        ("load average", "Memory is", "Root storage", "GPU 0"),
    ),
    (
        "MEMORY usage?",
        "Memory is 42 percent in use (6.7 gigabytes used out of 16.0 gigabytes).",
        ("CPU", "Root storage", "GPU 0"),
    ),
    (
        "disk-usage",
        "Root storage has 78.3 gigabytes free out of 101.5 gigabytes.",
        ("CPU", "Memory is", "GPU 0"),
    ),
    (
        "GPU usage.",
        "GPU 0 (NVIDIA RTX) utilization is 12 percent.",
        ("CPU", "Memory is", "Root storage", "temperature", "VRAM"),
    ),
    (
        "System status?",
        "CPU 19%; memory 42%; disk 78.3 GB free; GPU 0 12%.",
        ("load average", "logical cores", "temperature", "VRAM"),
    ),
])
@pytest.mark.asyncio
async def test_factual_status_quickpath_speaks_only_requested_metrics(prompt, expected, omitted):
    status = (
        "CPU has 8 logical cores with load average 0.29, 0.24, 0.30. "
        "Memory is 42 percent in use (6.7 gigabytes used out of 16.0 gigabytes). "
        "Root storage has 78.3 gigabytes free out of 101.5 gigabytes. "
        "GPU 0 (NVIDIA RTX) utilization is 12 percent, temperature is 43 degrees Celsius, "
        "and 2.0 of 8.0 gigabytes VRAM is in use. "
        "CPU utilization was 19 percent during this status sample."
    )

    class SilentTTS:
        engine = "silent"
        pending_barge_in_text = None

        async def speak_async(self, text):
            self.last_text = text

    class NoModelCalls:
        def __init__(self):
            self.chat_calls = 0

        async def chat(self, *_args, **_kwargs):
            self.chat_calls += 1
            raise AssertionError("a factual status quickpath must not call the model")

        def format_tool_response(self, tool_call_id, tool_name, result):
            return {
                "role": "tool", "tool_call_id": tool_call_id,
                "name": tool_name, "content": result,
            }

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="test", cloud_model="test",
        ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
        api_key="", temperature=0, num_ctx=8192, max_tool_rounds=4,
    ))
    tts = SilentTTS()
    brain = AdamBrain(
        config, None, None, None, tts,
        memory_mgr=SimpleNamespace(retrieve_context=lambda _text: None),
    )
    model = NoModelCalls()
    brain.llm_client = model

    async def fake_execute(name, _args):
        assert name == "get_system_status"
        return status

    brain._execute_tool = fake_execute
    with patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
        await brain.process_user_utterance(prompt)

    assert tts.last_text == expected
    assert all(fragment not in tts.last_text for fragment in omitted)
    assert model.chat_calls == 0


@pytest.mark.asyncio
async def test_factual_cpu_quickpath_reports_unavailable_utilization_without_substituting_load():
    class SilentTTS:
        engine = "silent"
        pending_barge_in_text = None

        async def speak_async(self, text):
            self.last_text = text

    class NoModelCalls:
        def format_tool_response(self, tool_call_id, tool_name, result):
            return {
                "role": "tool", "tool_call_id": tool_call_id,
                "name": tool_name, "content": result,
            }

        async def chat(self, *_args, **_kwargs):
            raise AssertionError("a factual status quickpath must not call the model")

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="test", cloud_model="test",
        ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
        api_key="", temperature=0, num_ctx=8192, max_tool_rounds=4,
    ))
    tts = SilentTTS()
    brain = AdamBrain(
        config, None, None, None, tts,
        memory_mgr=SimpleNamespace(retrieve_context=lambda _text: None),
    )
    brain.llm_client = NoModelCalls()

    async def fake_execute(name, _args):
        assert name == "get_system_status"
        return (
            "CPU has 8 logical cores with load average 0.29, 0.24, 0.30. "
            "Memory is 42 percent in use (6.7 gigabytes used out of 16.0 gigabytes). "
            "CPU utilization percentage is unavailable; CPU load average is reported above."
        )

    brain._execute_tool = fake_execute
    with patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
            await brain.process_user_utterance("CPU usage")

    assert tts.last_text == "CPU utilization is unavailable in the status result."
    assert "load average" not in tts.last_text


@pytest.mark.asyncio
async def test_natural_cpu_question_uses_model_to_summarize_rich_status_result():
    class SilentTTS:
        engine = "silent"
        pending_barge_in_text = None

        async def speak_async(self, text):
            self.last_text = text

    class Model:
        def __init__(self):
            self.responses = [
                {"content": "", "tool_calls": [{
                    "id": "status-1",
                    "function": {"name": "get_system_status", "arguments": {}},
                }]},
                {"content": "CPU utilization was 19 percent during the status sample.", "tool_calls": []},
            ]
            self.requested_tools = []
            self.tool_result_contents = []

        async def chat(self, messages, tools=None, **_kwargs):
            self.requested_tools.append([tool.name for tool in (tools or [])])
            self.tool_result_contents.extend(
                str(message.get("content", ""))
                for message in messages if message.get("role") == "tool"
            )
            return self.responses.pop(0)

        def format_tool_response(self, tool_call_id, tool_name, result):
            return {
                "role": "tool", "tool_call_id": tool_call_id,
                "name": tool_name, "content": result,
            }

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="test", cloud_model="test",
        ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
        api_key="", temperature=0, num_ctx=8192, max_tool_rounds=4,
    ))
    tts = SilentTTS()
    brain = AdamBrain(
        config, None, None, None, tts,
        memory_mgr=SimpleNamespace(retrieve_context=lambda _text: None),
    )
    model = Model()
    brain.llm_client = model
    brain.skill_manager.get_matched_skill_context = lambda _query: None
    rich_status = (
        "CPU has 8 logical cores with load average 0.29, 0.24, 0.30. "
        "Memory is 42 percent in use. Root storage has 78 gigabytes free. "
        "GPU telemetry is unavailable. CPU utilization was 19 percent during this status sample."
    )

    async def fake_execute(name, _args):
        assert name == "get_system_status"
        return rich_status

    brain._execute_tool = fake_execute
    with patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
        await brain.process_user_utterance("What's my CPU usage?")

    assert _can_direct_dispatch_system_status("What's my CPU usage?") is False
    assert model.requested_tools[0] == ["get_system_status"]
    assert "get_system_status" not in model.requested_tools[1]
    assert len(model.requested_tools) == 2
    assert any("Memory is 42 percent" in content for content in model.tool_result_contents)
    assert "CPU utilization was 19 percent" in tts.last_text
    assert "Memory is" not in tts.last_text
    assert "GPU telemetry" not in tts.last_text


@pytest.mark.parametrize("prompt", [
    "Why is my CPU usage so high?",
    "My processor is pegged.",
])
@pytest.mark.asyncio
async def test_causal_cpu_question_uses_process_evidence_then_model_explanation(prompt):
    class SilentTTS:
        engine = "silent"
        pending_barge_in_text = None

        async def speak_async(self, text):
            self.last_text = text

    class Model:
        def __init__(self):
            self.responses = [
                {"content": "", "tool_calls": [{
                    "id": "status-1",
                    "function": {"name": "get_system_status", "arguments": {}},
                }]},
                {"content": "", "tool_calls": [{
                    "id": "processes-1",
                    "function": {
                        "name": "list_processes",
                        "arguments": {"sort_by": "cpu", "limit": 5},
                    },
                }]},
                {"content": "The process sample points to python as a likely contributor, but it does not prove the cause of the full CPU reading.", "tool_calls": []},
            ]
            self.requested_tool_names = []
            self.system_prompts = []

        async def chat(self, messages, tools=None, **_kwargs):
            self.requested_tool_names.append([tool.name for tool in (tools or [])])
            self.system_prompts.append(next(
                message["content"] for message in messages if message["role"] == "system"
            ))
            return self.responses.pop(0)

        def format_tool_response(self, tool_call_id, tool_name, result):
            return {
                "role": "tool", "tool_call_id": tool_call_id,
                "name": tool_name, "content": result,
            }

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="test", cloud_model="test",
        ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
        api_key="", temperature=0, num_ctx=8192, max_tool_rounds=4,
    ))
    tts = SilentTTS()
    brain = AdamBrain(
        config, None, None, None, tts,
        memory_mgr=SimpleNamespace(retrieve_context=lambda _text: None),
    )
    model = Model()
    brain.llm_client = model
    brain.skill_manager.get_matched_skill_context = lambda _query: None

    called = []

    async def fake_execute(name, args):
        called.append((name, args))
        if name == "get_system_status":
            return "CPU utilization was 20 percent during this status sample."
        if name == "list_processes":
            return "Top processes by cpu: python (PID 7): 80% CPU, 1% RAM"
        raise AssertionError(f"Unexpected tool call: {name}")

    brain._execute_tool = fake_execute
    with patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
        await brain.process_user_utterance(prompt)

    assert [name for name, _args in called] == ["get_system_status", "list_processes"]
    assert model.requested_tool_names[0] == ["get_system_status", "list_processes"]
    assert "inspect the relevant top processes" in model.system_prompts[0]
    assert "get_system_status" not in model.requested_tool_names[1]
    assert "list_processes" in model.requested_tool_names[1]
    assert len(model.requested_tool_names) == 3
    assert "20 percent during this status sample" not in tts.last_text
    assert "likely contributor" in tts.last_text


def test_status_turn_only_exposes_the_dedicated_read_only_tool():
    tools = [
        SimpleNamespace(name="get_system_status"),
        SimpleNamespace(name="run_bash_command"),
        SimpleNamespace(name="start_background_job"),
    ]

    filtered = _filter_tools_for_system_status(
        tools, "Report current CPU, RAM, and GPU utilization."
    )

    assert [tool.name for tool in filtered] == ["get_system_status"]


def test_compound_status_and_action_keeps_the_full_tool_set():
    tools = [
        SimpleNamespace(name="get_system_status"),
        SimpleNamespace(name="restart_service"),
    ]

    assert _filter_tools_for_system_status(
        tools, "Report my RAM usage and restart Adam if it is over 90%."
    ) == tools


def test_single_domain_desktop_action_gets_compact_desktop_tools():
    tools = [
        SimpleNamespace(name="computer_control"),
        SimpleNamespace(name="capture_screenshot"),
        SimpleNamespace(name="observe_desktop"),
        SimpleNamespace(name="focus_window"),
        SimpleNamespace(name="list_windows"),
        SimpleNamespace(name="run_bash_command"),
        SimpleNamespace(name="search_web"),
    ]

    filtered = _filter_tools_for_dedicated_desktop_navigation(
        tools, "Click the OK button in the dialog, then confirm it closed."
    )

    assert [tool.name for tool in filtered] == [
        "computer_control", "focus_window", "list_windows",
    ]


@pytest.mark.parametrize("prompt", [
    "Launch SHENZHEN I/O and beat one level.",
    "Open GIMP and edit the current image.",
    "Start LibreOffice and create a new document.",
])
def test_self_contained_app_task_keeps_launch_and_visual_interaction_tools(prompt):
    tools = [
        SimpleNamespace(name="launch_application"),
        SimpleNamespace(name="list_applications"),
        SimpleNamespace(name="computer_control"),
        SimpleNamespace(name="capture_screenshot"),
        SimpleNamespace(name="observe_desktop"),
        SimpleNamespace(name="focus_window"),
        SimpleNamespace(name="list_windows"),
        SimpleNamespace(name="run_bash_command"),
        SimpleNamespace(name="search_web"),
        SimpleNamespace(name="get_weather"),
    ]

    filtered = _filter_tools_for_dedicated_desktop_navigation(
        tools, prompt
    )

    assert [tool.name for tool in filtered] == [
        "launch_application", "list_applications", "computer_control",
        "capture_screenshot", "observe_desktop", "focus_window", "list_windows",
    ]


def test_desktop_control_label_start_minimized_does_not_disable_single_domain_route():
    request = (
        "In Notification Preferences, enable Always on top, change Notification level "
        "from Normal to Quiet, leave Start minimized unchanged, click Apply, and verify."
    )
    assert _is_dedicated_desktop_navigation_request(request)
    assert not _is_dedicated_desktop_navigation_request(
        "Click Save and start Firefox."
    )


def test_desktop_navigation_keeps_full_tools_when_another_domain_is_requested():
    tools = [
        SimpleNamespace(name="computer_control"),
        SimpleNamespace(name="capture_screenshot"),
        SimpleNamespace(name="run_bash_command"),
    ]
    assert _filter_tools_for_dedicated_desktop_navigation(
        tools, "Click the button and search the web for the matching instructions."
    ) == tools


def test_status_and_other_capability_keeps_the_full_tool_set():
    tools = [
        SimpleNamespace(name="get_system_status"),
        SimpleNamespace(name="get_weather"),
        SimpleNamespace(name="search_web"),
    ]

    assert _filter_tools_for_system_status(
        tools, "Report CPU usage and check tomorrow's weather."
    ) == tools


def test_desktop_request_gets_desktop_context():
    assert _is_desktop_context_request("Click the green button in the browser.")
    assert _is_desktop_context_request("Click it, then press Enter.")
    assert _is_desktop_context_request("Help me beat this game level.")
    assert _is_desktop_context_request("Read the text on my screen.")


def test_scheduler_slot_selection_routes_to_visual_desktop_tools():
    request = (
        "Select 5:00 PM in the local scheduler, but stop before confirming. "
        "Tell me which time is selected."
    )
    assert _is_desktop_context_request(request)
    assert _is_dedicated_desktop_navigation_request(request)
    assert not _can_answer_without_tools(request)
    assert _should_acknowledge_desktop_task(request)


def test_document_editing_in_a_named_app_routes_to_computer_use():
    request = (
        'Create a two-column table in Writer with headers "Item" and "Count" and rows '
        '"apples" / "3" and "oranges" / "5". Make the header row bold, save it, and report the rows.'
    )
    assert _is_desktop_context_request(request)
    assert _is_dedicated_desktop_navigation_request(request)
    assert not _can_answer_without_tools(request)
    assert not _is_dedicated_desktop_navigation_request(
        "Create a two-column document table with apples and oranges."
    )
    assert not _is_dedicated_desktop_navigation_request(
        "Create a table in Writer and email it to Sam."
    )


def test_local_visible_form_that_stops_before_commit_uses_compact_gui_tools():
    request = (
        "On the local reservation preview, choose tomorrow at 5:00 PM for two people. "
        "Fill the form and stop before placing it. Tell me the status."
    )
    assert _is_bounded_visible_form_request(request)
    assert _is_dedicated_desktop_navigation_request(request)
    assert not _can_answer_without_tools(request)
    available = [
        SimpleNamespace(name="computer_control"),
        SimpleNamespace(name="observe_desktop"),
        SimpleNamespace(name="capture_screenshot"),
        SimpleNamespace(name="focus_window"),
        SimpleNamespace(name="list_windows"),
        SimpleNamespace(name="web_search"),
    ]
    assert [tool.name for tool in _filter_tools_for_dedicated_desktop_navigation(available, request)] == [
        "computer_control", "focus_window", "list_windows",
    ]


@pytest.mark.parametrize("additional_request", [
    "Also check tomorrow's weather.",
    "Also check my calendar for that time.",
    "Also search the web for the venue's hours.",
])
def test_mixed_bounded_visible_form_keeps_tools_for_every_requested_domain(additional_request):
    request = (
        "On the local reservation preview, choose tomorrow at 5:00 PM for two people. "
        "Fill the form and stop before placing it. Tell me the status. "
        f"{additional_request}"
    )
    assert _is_bounded_visible_form_request(request)
    available = [
        SimpleNamespace(name=name)
        for name in (
            "computer_control", "focus_window", "list_windows", "get_weather",
            "calendar", "search_web", "read_file",
        )
    ]

    assert not _is_dedicated_desktop_navigation_request(request)
    assert _filter_tools_for_dedicated_desktop_navigation(available, request) == available


def test_mixed_app_launch_and_edit_keeps_general_tools():
    request = "Open GIMP and edit the current image, then check tomorrow's weather."
    available = [
        SimpleNamespace(name=name)
        for name in ("launch_application", "list_applications", "computer_control", "get_weather")
    ]

    assert _filter_tools_for_dedicated_desktop_navigation(available, request) == available


@pytest.mark.asyncio
async def test_desktop_task_stops_without_model_retry_when_display_is_dpms_off():
    class DummyClient:
        def __init__(self):
            self.chat_calls = 0

        async def chat(self, *_args, **_kwargs):
            self.chat_calls += 1
            return {"content": "", "tool_calls": []}

    class DummyTTS:
        def __init__(self):
            self.spoken = []

        async def speak_async(self, text):
            self.spoken.append(text)

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="local", local_model="qwen", cloud_model="", ollama_host="",
        temperature=0.3, num_ctx=4096,
    ))
    tts = DummyTTS()
    brain = AdamBrain(config, None, None, None, tts)
    client = DummyClient()
    brain.llm_client = client
    brain.computer_controller = SimpleNamespace(
        available=True,
        coordinate_mode="pixels",
        run=lambda **_kwargs: SimpleNamespace(
            status="failed",
            screenshot=None,
            message="Could not capture: focused monitor is powered off (DPMS).",
        ),
    )

    await brain.process_user_utterance(
        "Click the blue button in the browser and tell me what happens."
    )

    assert client.chat_calls == 0
    assert tts.spoken == [
        "I’ve got the request. I’m checking the screen now.",
        "The display is powered off, so I couldn't inspect the screen or make changes. Wake the display and I can continue.",
    ]


def test_visible_desktop_work_gets_an_immediate_status_cue():
    assert _should_acknowledge_desktop_task("Click Reveal code in the Safe UI Pilot.")
    assert _should_acknowledge_desktop_task("What is the Discord chat window showing?")
    assert _should_acknowledge_desktop_task("Could you move Spotify to Workspace One?")
    assert not _should_acknowledge_desktop_task("What's on my browser right now?")
    assert not _should_acknowledge_desktop_task("Explain what a browser report is.")


def test_task_acknowledgment_names_explicit_workflows_without_repeating_note_content():
    g2_request = (
        'In the local scratchpad, enter "Call the dentist Tuesday at 2 pm" in the Note text field, '
        "click Save note, and tell me the saved text."
    )
    w1_request = (
        "In the visible browser report at http://127.0.0.1:8765/invoice-report.html, find the supplier "
        "with the largest overdue invoice total and all its invoice IDs. In the open work order document, "
        "replace the TBD supplier, invoice IDs, and total with those values; change REVIEW from Pending "
        "to Complete. Leave every other character unchanged, save the document, reopen it, and report "
        "the saved values."
    )

    assert _should_acknowledge_desktop_task(g2_request)
    assert _desktop_task_acknowledgment(g2_request) == "I’ll save the note and check the saved text."
    assert "Call the dentist" not in _desktop_task_acknowledgment(g2_request)
    assert _desktop_task_acknowledgment(w1_request) == (
        "I’ll compare the overdue invoices, update the work order, "
        "and reopen it to verify the saved file."
    )
    assert _desktop_task_acknowledgment("Click Reveal code in the Safe UI Pilot.") == (
        "I’ve got the request. I’m checking the screen now."
    )


@pytest.mark.parametrize("prompt", [
    (
        "In the visible browser report, find the supplier with the largest overdue invoice total. "
        "In the open work order, do not update, save, or reopen the file. "
        "Explain the numbers without changing it."
    ),
    (
        "In the visible browser report, find the largest overdue invoice total. "
        "Don't modify, save, or reopen the work order; just explain the numbers."
    ),
    (
        "In the visible browser report, find the largest overdue invoice total. "
        "Keep the work order read-only; do not update, save, or reopen it."
    ),
    (
        "In the visible browser report, find the largest overdue invoice total. "
        "Leave the work order unchanged; do not save it or reopen it."
    ),
])
def test_work_order_acknowledgment_respects_negative_and_read_only_instructions(prompt):
    assert _should_acknowledge_desktop_task(prompt)
    assert _desktop_task_acknowledgment(prompt) == (
        "I’ve got the request. I’m checking the screen now."
    )


@pytest.mark.parametrize("negation", [
    "I won't update, save, or reopen the work order.",
    "I cannot update, save, or reopen the work order.",
    "I’m not going to update, save, or reopen the work order.",
])
def test_work_order_acknowledgment_does_not_override_explicit_negation(negation):
    prompt = (
        "In the visible browser report, find the supplier with the largest overdue invoice total. "
        "In the open work order, replace the TBD supplier and change REVIEW to Complete. "
        f"{negation} Explain the numbers."
    )

    assert _should_acknowledge_desktop_task(prompt)
    assert _desktop_task_acknowledgment(prompt) == (
        "I’ve got the request. I’m checking the screen now."
    )


@pytest.mark.parametrize(("prompt", "eligible_for_acknowledgment"), [
    (
        'In the local scratchpad, enter "Call the dentist Tuesday at 2 pm" in the Note text field, '
        "do not click Save note, and tell me the saved text.",
        True,
    ),
    (
        'In the local scratchpad, enter "Call the dentist Tuesday at 2 pm" in the Note text field, '
        "don't save the note or click Save note, and tell me the saved text.",
        False,
    ),
])
def test_scratchpad_acknowledgment_does_not_promise_a_forbidden_save(
    prompt, eligible_for_acknowledgment,
):
    assert _should_acknowledge_desktop_task(prompt) is eligible_for_acknowledgment
    assert _desktop_task_acknowledgment(prompt) == (
        "I’ve got the request. I’m checking the screen now."
    )


@pytest.mark.asyncio
async def test_multistep_browser_to_document_task_acknowledges_before_tool_work(tmp_path, monkeypatch):
    from tools.create_implementation_fixtures import create_fixtures

    fixture = create_fixtures(tmp_path, "w1-acknowledgment")
    spec = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    request = spec["prompts"]["work_order_template"].replace(
        "<LOCAL_REPORT_URL>", "http://127.0.0.1:8765/invoice-report.html"
    )
    events = []
    speech_roles = []

    class DummyTTS:
        engine = "silent"
        pending_barge_in_text = None

        async def speak_async(self, text):
            from src.telemetry.events import get_speech_role

            events.append(("speech", text))
            speech_roles.append(get_speech_role())

    class DummyClient:
        def __init__(self):
            self.calls = 0

        async def chat(self, _messages, tools=None, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                assert any(tool.name == "list_windows" for tool in tools)
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "windows-1",
                        "function": {"name": "list_windows", "arguments": {}},
                    }],
                }
            return {"content": "The work order still needs editing.", "tool_calls": []}

        def format_tool_response(self, tool_call_id, tool_name, result):
            return {
                "role": "tool", "tool_call_id": tool_call_id,
                "name": tool_name, "content": result,
            }

    config = SimpleNamespace(llm=SimpleNamespace(
        provider="local", local_model="qwen", cloud_model="", ollama_host="",
        temperature=0.3, num_ctx=4096, max_tool_rounds=3,
    ))
    brain = AdamBrain(
        config, None, None, None, DummyTTS(),
        memory_mgr=SimpleNamespace(retrieve_context=lambda _text: None),
    )
    brain.llm_client = DummyClient()
    brain.computer_controller = SimpleNamespace(
        available=True, coordinate_mode="pixels", drag_active=False,
    )
    monkeypatch.setattr("src.llm.brain.get_open_windows_prompt_context", lambda: "Synthetic fixture windows.")

    async def execute_tool(name, _args):
        events.append(("tool", name))
        return "Synthetic fixture windows only."

    brain._execute_tool = execute_tool
    await brain.process_user_utterance(request)

    assert events[0] == (
        "speech",
        "I’ll compare the overdue invoices, update the work order, and reopen it to verify the saved file.",
    )
    assert events[1] == ("tool", "list_windows")
    assert speech_roles == ["acknowledgment", "final"]


def test_general_question_skips_desktop_context():
    assert not _is_desktop_context_request(
        "In two sentences, compare rigatoni and penne for a chunky tomato sauce."
    )
    assert not _is_desktop_context_request("What time is it in Tokyo?")
    assert not _is_dedicated_desktop_navigation_request("Pick a movie to watch.")


def test_system_prompt_can_omit_or_include_startup_desktop_guidance():
    brain = SimpleNamespace(
        ocr_only=False,
        computer_controller=SimpleNamespace(coordinate_mode="pixels"),
        skill_manager=SimpleNamespace(get_startup_context=lambda: "DESKTOP SETUP GUIDANCE"),
    )

    concise = AdamBrain._build_system_prompt(brain, include_startup_context=False)
    desktop = AdamBrain._build_system_prompt(brain, include_startup_context=True)

    assert "DESKTOP SETUP GUIDANCE" not in concise
    assert "DESKTOP SETUP GUIDANCE" in desktop
