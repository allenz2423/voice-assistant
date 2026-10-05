"""Offline Brain-to-controller plumbing for screenshot visual grounding."""

import json
import re
import subprocess
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw


def _png(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (120, 90), color).save(output, format="PNG")
    return output.getvalue()


def _annotated_png(bounds: tuple[int, int, int, int], color: tuple[int, int, int]) -> bytes:
    with Image.open(BytesIO(RAW_SCREENSHOT)) as captured:
        image = captured.convert("RGB")
    ImageDraw.Draw(image).rectangle(bounds, fill=color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


RAW_SCREENSHOT = _png((21, 42, 63))
ANNOTATED_SCREENSHOT = _png((63, 42, 21))
REGION_ANNOTATION = "Visual region VG-42 center=(73,41) box=(67,35,79,47)."


class _DummyTTS:
    def __init__(self):
        self.spoken = []
        self.pending_barge_in_text = None

    async def speak_async(self, text):
        self.spoken.append(text)


class _GroundedReplayModel:
    """Script inspect, then derive the bounded click from the returned evidence."""

    provider = "local"

    def __init__(self):
        self.requests = []
        self.inspect_snapshot_id = ""
        self.click_arguments = None

    async def chat(self, messages, tools=None, **_kwargs):
        request = [dict(message) for message in messages]
        self.requests.append(request)
        call_number = len(self.requests)
        if call_number == 1:
            return {
                "content": "",
                "tool_calls": [{
                    "id": "inspect-grounded-window",
                    "function": {
                        "name": "computer_control",
                        "arguments": {
                            "action": "inspect",
                            "include_ocr": False,
                            "include_visual_grounding": True,
                            "screenshot_delay_seconds": 0,
                        },
                    },
                }],
            }

        if call_number == 2:
            tool_data = [
                json.loads(message["content"])["data"]
                for message in request
                if message.get("role") == "tool"
                and isinstance(message.get("content"), str)
                and message["content"].startswith("{")
            ]
            grounded_result = next(data for data in tool_data if REGION_ANNOTATION in data)
            self.inspect_snapshot_id = re.search(
                r"Snapshot ID: ([A-Za-z0-9_-]+)", grounded_result
            ).group(1)
            match = re.search(r"VG-42 center=\((\d+),(\d+)\)", grounded_result)
            assert match is not None
            self.click_arguments = {
                "action": "click",
                "snapshot_id": self.inspect_snapshot_id,
                "x": int(match.group(1)),
                "y": int(match.group(2)),
                "screenshot_delay_seconds": 0,
            }
            return {
                "content": "",
                "tool_calls": [{
                    "id": "click-grounded-region",
                    "function": {
                        "name": "computer_control",
                        "arguments": dict(self.click_arguments),
                    },
                }],
            }

        return {"content": "The grounded icon was clicked.", "tool_calls": []}

    def format_tool_response(self, tool_call_id, tool_name, result):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }


class _RepeatedClickReplayModel(_GroundedReplayModel):
    """Repeat the same grounded click once using each returned fresh token."""

    async def chat(self, messages, tools=None, **kwargs):
        call_number = len(self.requests) + 1
        if call_number <= 2:
            return await super().chat(messages, tools=tools, **kwargs)
        if call_number > 3:
            raise AssertionError("Brain requested another model turn after the repeat-click limit.")

        request = [dict(message) for message in messages]
        self.requests.append(request)
        tool_data = [
            json.loads(message["content"])["data"]
            for message in reversed(request)
            if message.get("role") == "tool"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("{")
        ]
        latest_result = next(data for data in tool_data if "Snapshot ID:" in data)
        snapshot_id = re.search(r"Snapshot ID: ([A-Za-z0-9_-]+)", latest_result).group(1)
        self.click_arguments = {
            "action": "click",
            "snapshot_id": snapshot_id,
            "x": 73,
            "y": 41,
            "screenshot_delay_seconds": 0,
        }
        return {
            "content": "",
            "tool_calls": [{
                "id": "repeat-grounded-click",
                "function": {
                    "name": "computer_control",
                    "arguments": dict(self.click_arguments),
                },
            }],
        }


def _config():
    return SimpleNamespace(llm=SimpleNamespace(
        provider="local",
        local_model="fixture-model",
        cloud_model="fixture-model",
        ollama_host="http://localhost:11434",
        api_base="https://example.test/v1",
        api_key="offline-test-key",
        temperature=0,
        num_ctx=8192,
    ))


@pytest.mark.asyncio
async def test_visual_grounding_evidence_reaches_model_and_fresh_action(monkeypatch):
    from src.llm.brain import AdamBrain
    from src.tools.computer_control import ComputerController

    input_calls = []

    def fake_runner(args, **_kwargs):
        input_calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    grounder_images = []

    def grounder(_image):
        # Simulate detector-overlay variation over identical raw screenshots.
        # The controller should compare raw frames for visual-change checks.
        annotated = ANNOTATED_SCREENSHOT if not grounder_images else _png((7, 8, 9))
        grounder_images.append(annotated)
        return annotated, REGION_ANNOTATION

    controller = ComputerController(
        screenshot_fn=lambda: RAW_SCREENSHOT,
        runner=fake_runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":fixture"},
        visual_grounder=grounder,
        screenshot_delay_seconds=0,
    )
    controller._read_active_window_state = lambda: ("fixture-window", (0, 0, 120, 90))

    # The controller has only a synthetic screenshot and a recording runner.
    # Mark its capability available so Brain publishes the normal tool schema.
    monkeypatch.setattr(ComputerController, "available", property(lambda _self: True))

    model = _GroundedReplayModel()
    tts = _DummyTTS()
    brain = AdamBrain(
        _config(), None, None, None, tts,
        memory_mgr=SimpleNamespace(retrieve_context=lambda _text: None),
    )
    brain.skill_manager = SimpleNamespace(get_matched_skill_context=lambda _text: None)
    brain.computer_controller = controller
    brain.llm_client = model

    from unittest.mock import patch

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Synthetic desktop"):
        await brain.process_user_utterance("Run a visual-grounding replay on the fixture.")

    assert len(model.requests) == 3
    followup_request = model.requests[1]
    tool_result = next(
        json.loads(message["content"])["data"]
        for message in followup_request
        if message.get("role") == "tool"
        and isinstance(message.get("content"), str)
        and message["content"].startswith("{")
        and REGION_ANNOTATION in json.loads(message["content"]).get("data", "")
    )
    assert REGION_ANNOTATION in tool_result
    assert f"Snapshot ID: {model.inspect_snapshot_id}" in tool_result
    image_message = next(
        message for message in followup_request
        if message.get("role") == "user" and message.get("images")
    )
    assert image_message["images"] == [ANNOTATED_SCREENSHOT]
    assert controller._last_screenshot == RAW_SCREENSHOT
    assert grounder_images == [ANNOTATED_SCREENSHOT, _png((7, 8, 9))]
    click_tool_reply = next(
        message for message in model.requests[2]
        if message.get("role") == "tool"
        and message.get("tool_call_id") == "click-grounded-region"
    )
    click_result = json.loads(click_tool_reply["content"])["data"]
    assert "screen appears unchanged" in click_result

    assert model.click_arguments == {
        "action": "click",
        "snapshot_id": model.inspect_snapshot_id,
        "x": 73,
        "y": 41,
        "screenshot_delay_seconds": 0,
    }
    assert controller.snapshot_id
    assert controller.snapshot_id != model.inspect_snapshot_id
    assert ["xdotool", "mousemove", "73", "41"] in input_calls
    assert ["xdotool", "click", "--delay", "80", "1"] in input_calls

    calls_before_stale_attempt = len(input_calls)
    stale_result = controller.run(
        action="click",
        snapshot_id=model.inspect_snapshot_id,
        x=73,
        y=41,
        screenshot_delay_seconds=0,
    )
    assert stale_result.status == "invalid_input"
    assert stale_result.dispatched is False
    assert "not the controller's latest observation" in stale_result.message
    assert len(input_calls) == calls_before_stale_attempt


@pytest.mark.asyncio
async def test_brain_repeat_action_guard_uses_raw_frames_when_overlays_change(monkeypatch):
    from unittest.mock import patch

    from src.llm.brain import AdamBrain
    from src.tools.computer_control import ComputerController

    input_calls = []

    def fake_runner(args, **_kwargs):
        input_calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    overlay_images = [
        _annotated_png((4, 4, 52, 68), (240, 230, 220)),
        _annotated_png((67, 8, 116, 84), (230, 220, 210)),
        _annotated_png((28, 16, 91, 75), (220, 210, 200)),
    ]
    grounder_calls = []

    def grounder(_image):
        annotated = overlay_images[min(len(grounder_calls), len(overlay_images) - 1)]
        grounder_calls.append(annotated)
        return annotated, REGION_ANNOTATION

    controller = ComputerController(
        screenshot_fn=lambda: RAW_SCREENSHOT,
        runner=fake_runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":fixture"},
        visual_grounder=grounder,
        screenshot_delay_seconds=0,
    )
    controller._read_active_window_state = lambda: ("fixture-window", (0, 0, 120, 90))
    monkeypatch.setattr(ComputerController, "available", property(lambda _self: True))

    model = _RepeatedClickReplayModel()
    tts = _DummyTTS()
    brain = AdamBrain(
        _config(), None, None, None, tts,
        memory_mgr=SimpleNamespace(retrieve_context=lambda _text: None),
    )
    brain.skill_manager = SimpleNamespace(get_matched_skill_context=lambda _text: None)
    brain.computer_controller = controller
    brain.llm_client = model

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Synthetic desktop"):
        await brain.process_user_utterance("Run a visual-grounding replay on the fixture.")

    # The raw frame is identical after both clicks. A changed detector overlay
    # must not reset Brain's repeat-action guard or trigger another model turn.
    assert grounder_calls == overlay_images
    assert controller._last_screenshot == RAW_SCREENSHOT
    assert len(model.requests) == 3  # inspect, first click, one repeated click
    assert model.click_arguments["snapshot_id"]
    assert model.click_arguments["snapshot_id"] != model.inspect_snapshot_id
    assert len([call for call in input_calls if call[:2] == ["xdotool", "click"]]) == 2
    assert len([call for call in input_calls if call[:2] == ["xdotool", "mousemove"]]) == 2
    assert "same desktop action left the screen unchanged" in tts.spoken[-1].lower()
