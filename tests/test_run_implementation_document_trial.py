"""Offline safety tests for the bounded headless D1 Brain runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path
import stat

import pytest

from tools import create_implementation_fixtures
from tools import run_implementation_document_trial as trial


MODEL_ID = "deepseek/deepseek-v4.1-flash"


def _config_file(tmp_path: Path, *, fallbacks: bool = False) -> Path:
    path = tmp_path / "protected-config.yaml"
    path.write_text(
        "llm:\n"
        "  provider: custom\n"
        f"  cloud_model: {MODEL_ID}\n"
        f"  api_base: {trial.f2.EXPECTED_API_BASE}\n"
        "  api_key: synthetic-test-secret\n"
        f"  allow_provider_fallbacks: {'true' if fallbacks else 'false'}\n"
        "  tool_free_model: secondary/model\n"
        "  tool_free_provider_only: [secondary-route]\n"
        "  tool_free_allow_provider_fallbacks: true\n"
        "computer_control:\n  enabled: false\n"
        "computer_vision:\n  enabled: false\n"
        "browser_navigation:\n  enabled: false\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700)
    path.chmod(0o700)


def _args(tmp_path: Path, *, fixture: Path | None = None, run_id: str = "d1-run-001") -> dict:
    fixture = fixture or create_implementation_fixtures.create_fixtures(
        tmp_path / "fixtures", "d1-fixture-001",
    )
    output_dir = tmp_path / "private-runs"
    if not output_dir.exists():
        _private_dir(output_dir)
    return {
        "fixture_dir": fixture,
        "config_path": _config_file(tmp_path),
        "run_id": run_id,
        "config_id": "deepseek-openrouter-fallbacks-off-v1",
        "output_path": output_dir / "runs.jsonl",
        "route_ready_confirmed": True,
        "expected_model": MODEL_ID,
        "metadata_http_status": 200,
        "metadata_model_id": MODEL_ID,
        "metadata_endpoint_count": 31,
        "metadata_checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "confirm_provider_inference": True,
    }


class _FakeProvider:
    provider = "custom"

    def __init__(
        self, document_path: str, *, write_path: str | None = None,
        actions: list[tuple[str, dict]] | None = None, usage_available: bool = True,
    ) -> None:
        self.document_path = document_path
        self.write_path = write_path or document_path
        self.request_tools: list[list[str]] = []
        self.request_descriptions: list[list[str]] = []
        self.calls = 0
        self.usage_available = usage_available
        self.actions = actions or [
            ("read_file", {"path": document_path, "max_chars": trial.MAX_READ_CHARS}),
            ("write_file", {
                "path": self.write_path,
                "content": "STATUS: FINAL\nOWNER: Adam\n",
                "overwrite": True,
            }),
            ("read_file", {"path": document_path, "max_chars": trial.MAX_READ_CHARS}),
        ]

    def _emit_mock_provider_events(self) -> None:
        from src.telemetry.events import emit_event

        span_id = f"d1-provider-{self.calls}"
        emit_event(
            "llm.request_started", span_id=span_id, status="started",
            provider="custom", model=MODEL_ID, component="llm",
            attributes={"configured_provider": "custom", "configured_model": MODEL_ID},
        )
        attributes = {"http_status": 200}
        if self.usage_available:
            attributes.update({
                "accounting_status": "available",
                "usage": {
                    "input_tokens": 62,
                    "output_tokens": 24,
                    "provider_reported_cost": 0.00002,
                    "currency": "credits",
                },
            })
        else:
            attributes["accounting_status"] = "unavailable"
        emit_event(
            "llm.completed", span_id=span_id, status="ok",
            provider="custom", model=MODEL_ID, component="llm",
            attributes=attributes,
        )

    async def chat(self, messages, tools=None, max_tokens=None, think=None):
        self.calls += 1
        self.request_tools.append([tool.name for tool in tools or []])
        self.request_descriptions.append([tool.description for tool in tools or []])
        self._emit_mock_provider_events()
        if self.calls <= len(self.actions):
            tool_name, arguments = self.actions[self.calls - 1]
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"d1-tool-{self.calls}",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments,
                    },
                }],
            }
        return {
            "role": "assistant",
            "content": "Saved and verified the document.",
            "tool_calls": [],
        }

    def format_tool_response(self, tool_call_id, tool_name, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}


def _patch_provider(monkeypatch, fake: _FakeProvider) -> list:
    import src.llm.brain as brain_module

    constructed = []

    def construct(config):
        constructed.append(config)
        return fake

    monkeypatch.setattr(brain_module, "UniversalLLMClient", construct)
    return constructed


def test_valid_fixture_allows_optional_reads_and_records_private_oracle(tmp_path, monkeypatch):
    args = _args(tmp_path)
    document_path = str((args["fixture_dir"] / "document.txt").resolve())
    fake = _FakeProvider(document_path)
    constructed = _patch_provider(monkeypatch, fake)
    built = []

    def build_brain(config, exact_document_path, fixture, secrets):
        result = trial._build_brain(config, exact_document_path, fixture, secrets)
        built.append(result)
        return result

    record = trial.run_trial(**args, brain_builder=build_brain)

    assert fake.calls == 4
    assert fake.request_tools == [["read_file", "write_file"]] * 4
    assert all("before" not in description.lower() and "after" not in description.lower()
               for request in fake.request_descriptions for description in request)
    assert len(constructed) == 1
    assert constructed[0].llm.tool_free_model == ""
    assert constructed[0].computer_control.enabled is False
    assert constructed[0].computer_vision.enabled is False
    assert constructed[0].browser_navigation.enabled is False
    assert built[0][1].engine == "capture-only"
    assert built[0][1].spoken[-1:] == [record["brain_response_text"]]
    assert built[0][0].tool_free_llm_client is None

    fixture = json.loads((args["fixture_dir"] / "expected.json").read_text(encoding="utf-8"))
    assert record["prompt"] == fixture["prompts"]["document"]
    assert record["oracle"]["status"] == "manual_review_required"
    assert record["oracle"]["objective_evidence_status"] == "pass"
    assert record["oracle"]["initial_readback_observed"] is True
    assert record["oracle"]["write_tool_returned"] is True
    assert record["oracle"]["final_readback_observed"] is True
    assert record["oracle"]["independent_final_bytes_match"] is True
    assert record["document_integrity"]["before"]["sha256"] != record["document_integrity"]["after"]["sha256"]
    assert (args["fixture_dir"] / "document.txt").read_bytes() == b"STATUS: FINAL\nOWNER: Adam\n"
    assert [entry["tool_name"] for entry in record["dispatch_trace"]] == [
        "read_file", "write_file", "read_file",
    ]
    assert record["dispatch_trace"][1]["authorized_fixture_overwrite"] is True
    assert record["tool_dispatch_implementation"].startswith("runner-guarded in-process dirfd handlers")
    assert record["telemetry"]["logical_primary_client_chat_calls"] == 4
    assert record["telemetry"]["provider_usage"]["input_tokens"]["value"] == 248
    assert record["telemetry"]["provider_usage"]["output_tokens"]["value"] == 96
    assert record["telemetry"]["provider_usage"]["provider_reported_cost"]["total"] == {
        "amount": pytest.approx(0.00008), "currency": "credits",
    }
    assert record["timing_ms"]["request_end_to_verified_state"] is None
    assert "synthetic-test-secret" not in args["output_path"].read_text(encoding="utf-8")
    assert args["output_path"].stat().st_mode & 0o777 == 0o600
    event_path = args["output_path"].parent / "d1-run-001.events.jsonl"
    assert event_path.stat().st_mode & 0o777 == 0o600


def test_exact_final_bytes_pass_without_model_readback_calls(tmp_path, monkeypatch):
    args = _args(tmp_path, run_id="d1-write-only")
    document_path = str((args["fixture_dir"] / "document.txt").resolve())
    fake = _FakeProvider(document_path, actions=[("write_file", {
        "path": document_path,
        "content": "STATUS: FINAL\nOWNER: Adam\n",
        "overwrite": True,
    })])
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    assert fake.calls == 2
    assert record["oracle"]["objective_evidence_status"] == "pass"
    assert record["oracle"]["initial_readback_observed"] is False
    assert record["oracle"]["final_readback_observed"] is False
    assert record["oracle"]["independent_final_bytes_match"] is True
    assert record["telemetry"]["provider_usage"]["input_tokens"]["value"] == 124


def test_missing_provider_accounting_has_explicit_reasons(tmp_path, monkeypatch):
    args = _args(tmp_path, run_id="d1-missing-usage")
    document_path = str((args["fixture_dir"] / "document.txt").resolve())
    fake = _FakeProvider(document_path, usage_available=False, actions=[("write_file", {
        "path": document_path,
        "content": "STATUS: FINAL\nOWNER: Adam\n",
        "overwrite": True,
    })])
    _patch_provider(monkeypatch, fake)

    record = trial.run_trial(**args)

    provider_usage = record["telemetry"]["provider_usage"]
    assert provider_usage["input_tokens"]["value"] is None
    assert provider_usage["input_tokens"]["missing_reasons"]
    assert provider_usage["output_tokens"]["missing_reasons"]
    assert provider_usage["provider_reported_cost"]["total"] is None
    assert provider_usage["provider_reported_cost"]["missing_reasons"]


def test_wrong_write_path_is_refused_before_filesystem_dispatch(tmp_path, monkeypatch):
    args = _args(tmp_path, run_id="d1-wrong-path")
    document_path = str((args["fixture_dir"] / "document.txt").resolve())
    outside_path = str((tmp_path / "outside.txt").resolve())
    fake = _FakeProvider(document_path, write_path=outside_path)
    _patch_provider(monkeypatch, fake)
    import src.llm.brain as brain_module

    actual_writes = []
    original_write = brain_module.write_file

    def capture_write(*write_args, **write_kwargs):
        actual_writes.append((write_args, write_kwargs))
        return original_write(*write_args, **write_kwargs)

    monkeypatch.setattr(brain_module, "write_file", capture_write)
    record = trial.run_trial(**args)

    assert actual_writes == []
    assert (args["fixture_dir"] / "document.txt").read_bytes() == b"STATUS: DRAFT\nOWNER: Adam\n"
    assert not Path(outside_path).exists()
    assert record["oracle"]["objective_evidence_status"] == "fail"
    assert record["oracle"]["write_tool_returned"] is False
    assert "normal filesystem handlers are not delegated" in record["tool_dispatch_implementation"]
    assert any(
        entry.get("tool_name") == "write_file"
        and entry.get("status") == "refused"
        and "path" in entry.get("reason", "")
        for entry in record["dispatch_trace"]
    )


def test_unexpected_initial_document_state_is_refused_before_provider_construction(tmp_path, monkeypatch):
    args = _args(tmp_path, run_id="d1-bad-start-state")
    document = args["fixture_dir"] / "document.txt"
    document.write_bytes(b"STATUS: FINAL\nOWNER: someone else\n")
    fake = _FakeProvider(str(document.resolve()))
    constructed = _patch_provider(monkeypatch, fake)

    with pytest.raises(trial.TrialError, match="starting bytes"):
        trial.run_trial(**args)

    assert fake.calls == 0
    assert constructed == []
    assert not args["output_path"].exists()


def test_fixture_symlink_is_rejected_before_provider_construction(tmp_path, monkeypatch):
    args = _args(tmp_path, run_id="d1-fixture-symlink")
    document = args["fixture_dir"] / "document.txt"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"private outside file\n")
    document.unlink()
    document.symlink_to(outside)
    fake = _FakeProvider(str(document.resolve()))
    constructed = _patch_provider(monkeypatch, fake)

    with pytest.raises(trial.TrialError, match="non-symlink"):
        trial.run_trial(**args)

    assert outside.read_bytes() == b"private outside file\n"
    assert fake.calls == 0
    assert constructed == []
    assert not args["output_path"].exists()


def test_dirfd_atomic_write_does_not_follow_symlink_swapped_at_rename(tmp_path, monkeypatch):
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir(mode=0o700)
    fixture_dir.chmod(0o700)
    document = fixture_dir / "document.txt"
    document.write_bytes(b"STATUS: DRAFT\nOWNER: Adam\n")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside file must stay unchanged\n")
    fixture_fd = trial._open_absolute_directory_nofollow(fixture_dir.resolve())
    original_replace = os.replace

    def swap_in_symlink_then_replace(source, destination, **kwargs):
        os.unlink("document.txt", dir_fd=fixture_fd)
        os.symlink(str(outside), "document.txt", dir_fd=fixture_fd)
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(trial.os, "replace", swap_in_symlink_then_replace)
    try:
        trial._write_fixture_document(
            fixture_fd,
            initial_bytes=b"STATUS: DRAFT\nOWNER: Adam\n",
            final_bytes=b"STATUS: FINAL\nOWNER: Adam\n",
        )
        final_info = os.stat("document.txt", dir_fd=fixture_fd, follow_symlinks=False)
        final_bytes = trial._read_fixture_entry(fixture_fd, "document.txt", max_bytes=trial.MAX_DOCUMENT_BYTES)
    finally:
        os.close(fixture_fd)

    assert stat.S_ISREG(final_info.st_mode)
    assert final_bytes == b"STATUS: FINAL\nOWNER: Adam\n"
    assert outside.read_bytes() == b"outside file must stay unchanged\n"


def test_requires_separate_provider_inference_confirmation(tmp_path, monkeypatch):
    args = _args(tmp_path, run_id="d1-no-inference")
    args["confirm_provider_inference"] = False
    fake = _FakeProvider(str((args["fixture_dir"] / "document.txt").resolve()))
    constructed = _patch_provider(monkeypatch, fake)

    with pytest.raises(trial.TrialError, match="without --confirm-provider-inference"):
        trial.run_trial(**args)

    assert fake.calls == 0
    assert constructed == []
    assert not args["output_path"].exists()
