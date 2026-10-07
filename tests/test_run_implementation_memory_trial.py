"""Offline safety and record tests for the provider-backed headless M1 runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pytest

from tools import create_implementation_fixtures
from tools import run_implementation_memory_trial as trial

MODEL_ID = "deepseek/deepseek-v4.1-flash"


def test_module_path_help_exposes_only_generated_prompt_ids():
    project_root = Path(trial.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "tools.run_implementation_memory_trial", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "--prompt-id {day,week,year,timezone}" in result.stdout
    assert "--confirm-provider-inference" in result.stdout
    assert result.stderr == ""


def _config_file(tmp_path: Path, *, fallbacks: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "protected-config.yaml"
    path.write_text(
        "llm:\n"
        "  provider: custom\n"
        f"  cloud_model: {MODEL_ID}\n"
        f"  api_base: {trial.EXPECTED_API_BASE}\n"
        "  api_key: synthetic-test-secret\n"
        f"  allow_provider_fallbacks: {'true' if fallbacks else 'false'}\n"
        "  provider_only: [openrouter]\n"
        "  tool_free_model: should-be-disabled-for-trial\n"
        "  tool_free_provider_only: [other-provider]\n"
        "  tool_free_allow_provider_fallbacks: true\n"
        "computer_control:\n  enabled: true\n"
        "computer_vision:\n  enabled: true\n"
        "browser_navigation:\n  enabled: true\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700)
    path.chmod(0o700)


def _run_args(tmp_path: Path, *, prompt_id: str = "day", run_id: str = "m1-run-001", fallbacks: bool = False):
    fixture = create_implementation_fixtures.create_fixtures(tmp_path / "fixtures", f"m1-fixture-{run_id}")
    output_dir = tmp_path / "private-output"
    _private_dir(output_dir)
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "fixture_dir": fixture,
        "prompt_id": prompt_id,
        "config_path": _config_file(tmp_path, fallbacks=fallbacks),
        "run_id": run_id,
        "config_id": "deepseek-openrouter-fallbacks-off-v1",
        "output_path": output_dir / "runs.jsonl",
        "route_ready_confirmed": True,
        "expected_model": MODEL_ID,
        "metadata_http_status": 200,
        "metadata_model_id": MODEL_ID,
        "metadata_endpoint_count": 31,
        "metadata_checked_at": checked_at,
        "confirm_provider_inference": True,
    }


class _FakeProvider:
    provider = "custom"

    def __init__(self, *, tool_call: bool = False) -> None:
        self.tool_call = tool_call
        self.request_tools = []
        self.messages = []
        self.model = MODEL_ID

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        from src.telemetry.events import emit_event

        self.request_tools.append(tools)
        self.messages.append(messages)
        emit_event(
            "llm.request_started", span_id="m1-provider-span", status="started",
            provider="custom", model=self.model, component="provider",
        )
        emit_event(
            "llm.completed", span_id="m1-provider-span", status="ok",
            provider="custom", model=self.model, component="provider",
            attributes={
                "accounting_status": "available",
                "usage": {
                    "input_tokens": 150,
                    "output_tokens": 20,
                    "provider_reported_cost": 0.002,
                    "currency": "credits",
                },
            },
        )
        tool_calls = []
        if self.tool_call:
            tool_calls = [{
                "id": "unexpected-tool-call",
                "function": {"name": "run_bash_command", "arguments": "{}"},
            }]
        return {
            "role": "assistant",
            "content": "Synthetic fixture answer; independent review required.",
            "tool_calls": tool_calls,
        }


def _patch_provider(monkeypatch, fake: _FakeProvider):
    import src.llm.brain as brain_module

    configs = []

    def construct(config):
        configs.append(config)
        fake.model = config.llm.cloud_model
        return fake

    monkeypatch.setattr(brain_module, "UniversalLLMClient", construct)
    return configs


@pytest.mark.parametrize("prompt_id", trial.PROMPT_IDS)
def test_provider_mock_exercises_real_memory_only_brain_and_cleans_isolated_store(tmp_path, monkeypatch, prompt_id):
    args = _run_args(tmp_path, prompt_id=prompt_id, run_id=f"m1-{prompt_id}-001")
    fixture_metadata = json.loads((args["fixture_dir"] / "expected.json").read_text(encoding="utf-8"))
    memory_oracle = fixture_metadata["oracles"]["memory"]
    fake = _FakeProvider()
    constructed_configs = _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert fake.request_tools == [[]]
    assert len(fake.messages) == 1
    assert len(constructed_configs) == 1
    trial_config = constructed_configs[0]
    assert trial_config.llm.provider == "custom"
    assert trial_config.llm.cloud_model == MODEL_ID
    assert trial_config.llm.provider_only == ["openrouter"]
    assert trial_config.llm.tool_free_model == ""
    assert trial_config.computer_control.enabled is False
    assert trial_config.computer_vision.enabled is False
    assert trial_config.browser_navigation.enabled is False

    prompt = memory_oracle["prompts"][prompt_id]
    user_message = next(message["content"] for message in fake.messages[0] if message.get("role") == "user")
    assert prompt in user_message
    assert "[Retrieved user memory]" in user_message
    assert record["prompt"] == prompt
    assert record["prompt_id"] == f"prompts.memory.{prompt_id}"
    assert record["outcome"] == "not_scored"
    assert record["oracle_status"] == "manual_review_required"
    assert record["oracle"]["automatically_scored"] is False
    assert record["brain_response_text"] == "Synthetic fixture answer; independent review required."
    assert record["capture_only_tts_text"] == record["brain_response_text"]
    assert record["normalized_provider_responses"][0]["content"] == record["brain_response_text"]
    assert record["primary_client_chat_call_count"] == 1
    assert record["memory_store_integrity"] == {
        "explicit_per_run_path": True,
        "preexisting_store_path": False,
        "seeded_record_count": 3,
        "seeded_texts_match_fixture": True,
        "post_turn_texts_match_fixture": True,
        "default_memory_store_touched": False,
        "cleanup_status": "removed",
        "store_path_absent_after_cleanup": True,
    }
    assert record["memory_retrieval"]["call_count"] == 1
    context = record["memory_retrieval"]["observations"][0]["context"]
    assert context
    for index, text in enumerate(memory_oracle["records"]):
        if prompt_id == "day":
            assert (text in context) is (index == 0)
        elif prompt_id == "week":
            assert (text in context) is (index in {0, 1})
        elif prompt_id == "year":
            assert (text in context) is (index == 2)
        else:
            assert (text in context) is (index == 0)
    if prompt_id == "day":
        assert "T09:30:00" in context and "T10:15:00" in context
    if prompt_id == "timezone":
        assert f"[event timezone: {memory_oracle['timezone']}]" in context

    route_meta = record["route"]["metadata_attestation"]
    assert route_meta["source"] == "operator-reported external metadata-only check"
    assert route_meta["runner_fetched_or_cryptographically_verified"] is False
    output_text = args["output_path"].read_text(encoding="utf-8")
    assert "synthetic-test-secret" not in output_text
    assert args["output_path"].stat().st_mode & 0o777 == 0o600
    event_path = args["output_path"].parent / f"{args['run_id']}.events.jsonl"
    assert event_path.stat().st_mode & 0o777 == 0o600
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert events and {event["trace_id"] for event in events} == {args["run_id"]}
    assert not list(args["output_path"].parent.glob(f".{args['run_id']}.memory-*"))


def test_unexpected_model_tool_call_is_recorded_and_never_dispatched(tmp_path, monkeypatch):
    args = _run_args(tmp_path, run_id="m1-tool-call-001")
    fake = _FakeProvider(tool_call=True)
    _patch_provider(monkeypatch, fake)
    from src.llm.brain import AdamBrain

    dispatched = []

    async def record_dispatch(*_args, **_kwargs):
        dispatched.append(True)
        raise AssertionError("tool dispatch must stay unreachable")

    monkeypatch.setattr(AdamBrain, "_execute_tool", record_dispatch)
    record = trial.run_trial(**args)

    assert fake.request_tools == [[]]
    assert dispatched == []
    assert record["brain_execution_status"] == "error"
    assert record["unexpected_tool_call_stopped"] is True
    assert record["normalized_provider_responses"][0]["tool_names"] == ["run_bash_command"]
    assert record["memory_store_integrity"]["cleanup_status"] == "removed"
    assert record["outcome"] == "not_scored"


def test_fresh_generated_fixture_is_the_only_prompt_and_oracle_source(tmp_path, monkeypatch):
    args = _run_args(tmp_path, run_id="m1-tampered-fixture-001")
    expected_path = args["fixture_dir"] / "expected.json"
    metadata = json.loads(expected_path.read_text(encoding="utf-8"))
    metadata["oracles"]["memory"]["records"][0] = "unexpected user memory"
    expected_path.write_text(json.dumps(metadata), encoding="utf-8")
    fake = _FakeProvider()
    _patch_provider(monkeypatch, fake)

    with pytest.raises(trial.TrialError, match="records do not match the generated M1 oracle"):
        trial.run_trial(**args)

    assert fake.request_tools == []
    assert not args["output_path"].exists()


def test_inference_confirmation_and_protected_route_are_required(tmp_path, monkeypatch):
    args = _run_args(tmp_path, run_id="m1-refusal-001", fallbacks=True)
    fake = _FakeProvider()
    _patch_provider(monkeypatch, fake)

    with pytest.raises(trial.TrialError, match="provider fallbacks must be explicitly disabled"):
        trial.run_trial(**args)
    assert fake.request_tools == []
    assert not args["output_path"].exists()

    args["config_path"] = _config_file(tmp_path / "no-fallback-config", fallbacks=False)
    args["route_ready_confirmed"] = False
    with pytest.raises(trial.TrialError, match="operator confirms"):
        trial.run_trial(**args)
    assert fake.request_tools == []
