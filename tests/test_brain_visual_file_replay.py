"""Offline Brain replay combining visual control with file readback."""

from __future__ import annotations

import hashlib
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


RAW_BEFORE = _png((22, 44, 66))
RAW_AFTER = _png((33, 55, 77))
REGION_NOTE = "Visual region VG-7 center=(72,40) box=(66,34,78,46)."


def _annotate(image_bytes: bytes) -> tuple[bytes, str]:
    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    ImageDraw.Draw(image).rectangle((66, 34, 78, 46), outline=(255, 40, 20), width=2)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), REGION_NOTE


class _SilentTTS:
    def __init__(self):
        self.spoken: list[str] = []
        self.pending_barge_in_text = None

    async def speak_async(self, text: str) -> None:
        self.spoken.append(text)


def _tool_result(request: list[dict], tool_name: str) -> str:
    for message in reversed(request):
        if message.get("role") != "tool" or message.get("name") != tool_name:
            continue
        envelope = json.loads(message["content"])
        data = envelope.get("data", "")
        if isinstance(data, str):
            return data
    raise AssertionError(f"No {tool_name} result was present in the model request.")


class _VisualFileReplayModel:
    """A deterministic fake model that follows the observed region and readback."""

    provider = "local"

    def __init__(self, target_path: str):
        self.target_path = target_path
        self.requests: list[list[dict]] = []
        self.inspect_snapshot_id = ""
        self.click_snapshot_id = ""
        self.post_click_snapshot_id = ""
        self.readback = ""

    async def chat(self, messages, tools=None, **_kwargs):
        request = [dict(message) for message in messages]
        self.requests.append(request)
        call_number = len(self.requests)
        tool_names = {tool.name for tool in (tools or [])}
        assert "computer_control" in tool_names
        assert "read_file" in tool_names

        if call_number == 1:
            return {
                "content": "",
                "tool_calls": [{
                    "id": "inspect-fixture-window",
                    "function": {"name": "computer_control", "arguments": {
                        "action": "inspect",
                        "scope": "window",
                        "include_ocr": False,
                        "include_visual_grounding": True,
                        "screenshot_delay_seconds": 0,
                    }},
                }],
            }

        if call_number == 2:
            inspection = _tool_result(request, "computer_control")
            assert REGION_NOTE in inspection
            match = re.search(r"Snapshot ID: ([A-Za-z0-9_-]+)", inspection)
            assert match is not None
            self.inspect_snapshot_id = match.group(1)
            image_message = next(
                message for message in request
                if message.get("role") == "user" and message.get("images")
            )
            assert image_message["images"] == [_annotate(RAW_BEFORE)[0]]
            region = re.search(r"VG-7 center=\((\d+),(\d+)\)", inspection)
            assert region is not None
            args = {
                "action": "click",
                "snapshot_id": self.inspect_snapshot_id,
                "x": int(region.group(1)),
                "y": int(region.group(2)),
                "screenshot_delay_seconds": 0,
            }
            self.click_snapshot_id = args["snapshot_id"]
            return {"content": "", "tool_calls": [{
                "id": "click-fixture-region",
                "function": {"name": "computer_control", "arguments": args},
            }]}

        if call_number == 3:
            click_result = _tool_result(request, "computer_control")
            assert "Clicked left at (72, 40)." in click_result
            match = re.search(r"Snapshot ID: ([A-Za-z0-9_-]+)", click_result)
            assert match is not None
            self.post_click_snapshot_id = match.group(1)
            assert self.post_click_snapshot_id != self.inspect_snapshot_id
            image_message = next(
                message for message in request
                if message.get("role") == "user" and message.get("images")
            )
            assert image_message["images"] == [_annotate(RAW_AFTER)[0]]
            return {"content": "", "tool_calls": [{
                "id": "read-fixture-file",
                "function": {"name": "read_file", "arguments": {"path": self.target_path}},
            }]}

        if call_number == 4:
            # Ground the final claim in the actual Brain read_file result.
            raw_read = _tool_result(request, "read_file")
            read_envelope = json.loads(raw_read)
            assert read_envelope["ok"] is True
            self.readback = read_envelope["readback"]
            content_match = re.search(r"\nText:\n(.*)", self.readback, re.DOTALL)
            assert content_match is not None
            content_bytes = content_match.group(1).encode("utf-8")
            digest = hashlib.sha256(content_bytes).hexdigest()
            return {
                "content": (
                    f"The fixture record is FIX-731; the readback is {len(content_bytes)} bytes "
                    f"with SHA-256 {digest}."
                ),
                "tool_calls": [],
            }

        raise AssertionError("The mixed replay exceeded its scripted model turns.")

    def format_tool_response(self, tool_call_id, tool_name, result):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
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
        max_tool_rounds=8,
    ))


@pytest.mark.asyncio
async def test_brain_mixes_fresh_visual_action_and_exact_file_readback(tmp_path, monkeypatch):
    from unittest.mock import patch

    from src.llm.brain import AdamBrain
    from src.tools.computer_control import ComputerController

    payload = b"fixture=FIX-731\nstate=ready\nunits=4\n"
    expected_hash = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "mixed-fixture.txt"
    target.write_bytes(payload)

    runner_calls: list[list[str]] = []
    action_dispatched = False

    def fake_runner(args, **_kwargs):
        nonlocal action_dispatched
        call = list(args)
        runner_calls.append(call)
        if call[:2] == ["xdotool", "click"]:
            action_dispatched = True
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    controller = ComputerController(
        screenshot_fn=lambda: RAW_AFTER if action_dispatched else RAW_BEFORE,
        runner=fake_runner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":fixture"},
        visual_grounder=_annotate,
        screenshot_delay_seconds=0,
    )
    controller._read_active_window_state = lambda: ("fixture-window", (0, 0, 120, 90))
    monkeypatch.setattr(ComputerController, "available", property(lambda _self: True))

    tts = _SilentTTS()
    brain = AdamBrain(
        _config(), None, None, None, tts,
        memory_mgr=SimpleNamespace(retrieve_context=lambda _text: None),
    )
    brain.computer_controller = controller
    model = _VisualFileReplayModel(str(target))
    brain.llm_client = model

    with patch("src.llm.brain.get_open_windows_prompt_context", return_value="Synthetic fixture window"):
        await brain.process_user_utterance(
            f"Use the highlighted fixture control, then read the fixture file at {target} "
            "and report its record and digest."
        )

    assert model.inspect_snapshot_id
    assert model.click_snapshot_id == model.inspect_snapshot_id
    assert model.post_click_snapshot_id
    assert model.post_click_snapshot_id != model.inspect_snapshot_id
    assert controller.snapshot_id == model.post_click_snapshot_id
    assert controller._last_screenshot == RAW_AFTER
    assert [call for call in runner_calls if call[:2] == ["xdotool", "mousemove"]] == [
        ["xdotool", "mousemove", "72", "40"]
    ]
    assert [call for call in runner_calls if call[:2] == ["xdotool", "click"]] == [
        ["xdotool", "click", "--delay", "80", "1"]
    ]

    # Once the click has refreshed the controller state, the prior observation
    # cannot authorize another action and must not reach the injected runner.
    calls_before_stale_action = len(runner_calls)
    stale = controller.run(
        action="click",
        snapshot_id=model.inspect_snapshot_id,
        x=72,
        y=40,
        screenshot_delay_seconds=0,
    )
    assert stale.status == "invalid_input"
    assert stale.dispatched is False
    assert "not the controller's latest observation" in stale.message
    assert len(runner_calls) == calls_before_stale_action

    assert target.read_bytes() == payload
    actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    assert actual_hash == expected_hash
    assert f"Bytes: {len(payload)}" in model.readback
    assert "Text:\nfixture=FIX-731\nstate=ready\nunits=4\n" in model.readback
    assert model.readback.endswith(payload.decode("utf-8"))
    assert tts.spoken[-1] == (
        f"The fixture record is FIX-731; the readback is {len(payload)} bytes "
        f"with SHA-256 {expected_hash}."
    )
