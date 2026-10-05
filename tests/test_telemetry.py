import json
from types import SimpleNamespace

import src.telemetry.events as telemetry
from src.telemetry.events import EventWriter
from src.llm.provider import _openrouter_usage_fields


def test_llm_config_rejects_conflicting_tool_free_reasoning_options():
    import pytest
    from src.config import LLMConfig

    with pytest.raises(ValueError, match="cannot both be configured"):
        LLMConfig(
            disable_reasoning_for_tool_free=True,
            tool_free_reasoning_effort="low",
        )


def test_event_writer_is_disabled_until_configured(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path=str(path))

    assert writer.emit("turn.started") is None
    assert not path.exists()


def test_event_writer_filters_sensitive_fields_and_restricts_permissions(tmp_path):
    path = tmp_path / "nested" / "events.jsonl"
    writer = EventWriter(enabled=True, path=str(path))
    row = writer.emit(
        "llm.completed",
        trace_id="trace-a",
        span_id="span-a",
        attributes={
            "safe_count": 3,
            "transcript": "private sentinel",
            "authorization": "Bearer fake-key",
            "usage": {
                "input_tokens": 12,
                "provider_reported_cost": 0.02,
                "currency": "credits",
                "unreviewed_field": 1,
            },
        },
    )

    assert row["trace_id"] == "trace-a"
    assert row["attributes"] == {
        "safe_count": 3,
        "usage": {
            "input_tokens": 12,
            "provider_reported_cost": 0.02,
            "currency": "credits",
        },
    }
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    raw = path.read_text()
    assert "private sentinel" not in raw
    assert "fake-key" not in raw
    assert json.loads(raw)["clock_ns"] > 0


def test_configured_writer_uses_current_trace_context(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr(telemetry, "_writer", None)
    writer = telemetry.configure_telemetry(SimpleNamespace(enabled=True, path=str(path)))
    token = telemetry.set_trace_id("trace-context")
    try:
        event = telemetry.emit_event("stt.started", status="started")
    finally:
        telemetry.reset_trace_id(token)
    assert event["trace_id"] == "trace-context"
    assert writer.enabled


def test_openrouter_usage_maps_only_finite_nonnegative_documented_fields():
    usage, status = _openrouter_usage_fields({
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 1},
            "cost": 0.004,
            "private": "do not retain",
        },
    })
    assert status == "available"
    assert usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "cached_input_tokens": 3,
        "cache_write_tokens": 1,
        "provider_reported_cost": 0.004,
        "currency": "credits",
    }

    empty, status = _openrouter_usage_fields({
        "usage": {"prompt_tokens": -1, "completion_tokens": float("nan"), "cost": True},
    })
    assert empty == {}
    assert status == "missing"


def test_openrouter_request_emits_correlated_content_free_events(tmp_path, monkeypatch):
    import aiohttp
    import asyncio
    import src.llm.provider as provider_module

    response_data = {
        "id": "request-123",
        "model": "vendor/model",
        "choices": [{"message": {"role": "assistant", "content": "Hello."}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "cost": 0.001},
    }

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return response_data

    class FakeSession:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            self.payload = kwargs.get("json")
            return FakeResponse()

    sessions = []
    original_session_init = FakeSession.__init__

    def record_session(self, **kwargs):
        original_session_init(self, **kwargs)
        sessions.append(self)

    FakeSession.__init__ = record_session

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(telemetry, "_writer", None)
    telemetry.configure_telemetry(SimpleNamespace(enabled=True, path=str(tmp_path / "events.jsonl")))
    token = telemetry.set_trace_id("trace-provider")
    try:
        config = SimpleNamespace(llm=SimpleNamespace(
            provider="custom", local_model="local", cloud_model="requested/model",
            ollama_host="http://localhost:11434", api_base="https://openrouter.ai/api/v1",
            api_key="fake-key", temperature=0.0, num_ctx=4096, think=True,
            disable_reasoning_for_tool_free=True,
            provider_only=["route"], allow_provider_fallbacks=False,
        ))
        result = asyncio.run(provider_module.UniversalLLMClient(config).chat(
            [{"role": "user", "content": "sentinel-private-prompt"}], tools=[],
            max_tokens=384,
        ))
    finally:
        telemetry.reset_trace_id(token)

    assert result["content"] == "Hello."
    assert sessions[0].payload["max_tokens"] == 384
    assert sessions[0].payload["reasoning"] == {"enabled": False}
    assert sessions[0].payload["provider"]["require_parameters"] is True
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["llm.request_started", "llm.completed"]
    assert {row["trace_id"] for row in rows} == {"trace-provider"}
    assert rows[0]["span_id"] == rows[1]["span_id"]
    assert rows[1]["status"] == "ok"
    assert rows[1]["attributes"]["usage"]["currency"] == "credits"
    assert rows[1]["attributes"]["provider_request_id"] == "request-123"
    assert "sentinel-private-prompt" not in (tmp_path / "events.jsonl").read_text()
    assert "fake-key" not in (tmp_path / "events.jsonl").read_text()


def test_openrouter_keeps_reasoning_available_when_tools_are_present(monkeypatch):
    import aiohttp
    import asyncio
    import src.llm.provider as provider_module

    class FakeResponse:
        status = 200
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"choices": [{"message": {"content": "Done."}}]}

    class FakeSession:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            self.payload = kwargs.get("json")
            payloads.append(self.payload)
            return FakeResponse()

    payloads = []
    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="local", cloud_model="requested/model",
        ollama_host="http://localhost:11434", api_base="https://openrouter.ai/api/v1",
        api_key="fake-key", temperature=0.0, num_ctx=4096, think=True,
        disable_reasoning_for_tool_free=True,
        provider_only=[], allow_provider_fallbacks=False,
    ))
    tool = SimpleNamespace(to_openai=lambda: {"type": "function", "function": {"name": "test"}})

    asyncio.run(provider_module.UniversalLLMClient(config).chat(
        [{"role": "user", "content": "Do the task"}], tools=[tool]
    ))

    assert "reasoning_effort" not in payloads[0]
    assert payloads[0]["reasoning"] == {"enabled": True}
    assert "provider" not in payloads[0]


def test_openrouter_uses_configured_effort_only_for_tool_free_calls(monkeypatch):
    import aiohttp
    import asyncio
    import src.llm.provider as provider_module

    class FakeResponse:
        status = 200
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"choices": [{"message": {"content": "Entropy measures disorder."}}]}

    payloads = []

    class FakeSession:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            payloads.append(kwargs.get("json"))
            return FakeResponse()

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="local", cloud_model="requested/model",
        ollama_host="http://localhost:11434", api_base="https://openrouter.ai/api/v1",
        api_key="fake-key", temperature=0.0, num_ctx=4096, think=True,
        disable_reasoning_for_tool_free=False,
        tool_free_reasoning_effort="none",
        provider_only=[], allow_provider_fallbacks=False,
    ))

    asyncio.run(provider_module.UniversalLLMClient(config).chat(
        [{"role": "user", "content": "Explain entropy."}], tools=[]
    ))
    assert payloads[0]["reasoning"] == {"effort": "none"}
    assert payloads[0]["provider"]["require_parameters"] is True


def test_openrouter_timeout_with_missing_usage_returns_fallback_without_logging_crash(monkeypatch):
    import aiohttp
    import asyncio
    import src.llm.provider as provider_module

    class TimeoutResponse:
        async def __aenter__(self):
            raise asyncio.TimeoutError

        async def __aexit__(self, *args):
            return False

    class TimeoutSession:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return TimeoutResponse()

    monkeypatch.setattr(aiohttp, "ClientSession", TimeoutSession)
    config = SimpleNamespace(llm=SimpleNamespace(
        provider="custom", local_model="local", cloud_model="requested/model",
        ollama_host="http://localhost:11434", api_base="https://openrouter.ai/api/v1",
        api_key="fake-key", temperature=0.0, num_ctx=4096, think=False,
        provider_only=[], allow_provider_fallbacks=False,
    ))

    result = asyncio.run(provider_module.UniversalLLMClient(config).chat(
        [{"role": "user", "content": "read my screen"}], tools=[]
    ))

    assert result["provider_error"] is True
    assert "couldn't reach the language model" in result["content"]


def test_power_parsers_keep_na_unavailable_and_reject_nonfinite_values():
    from tools.measure_power import _parse_float, _parse_int

    assert _parse_float("N/A") is None
    assert _parse_float("not supported") is None
    assert _parse_float("nan") is None
    assert _parse_float("inf") is None
    assert _parse_float("12.5") == 12.5
    assert _parse_int("12") == 12
    assert _parse_int("12.5") is None


def test_power_sampler_marks_decreasing_rapl_counter_ambiguous(tmp_path, monkeypatch):
    import tools.measure_power as power

    rapl_reads = iter((990, 10, 20, 30, 40))

    monkeypatch.setattr(power, "query_nvidia_version", lambda: {"status": "unavailable"})
    monkeypatch.setattr(power, "query_nvidia_smi", lambda: {
        "status": "unavailable", "error": "not available", "devices": [],
    })
    monkeypatch.setattr(power, "query_laptop_battery", lambda: {
        "status": "unavailable", "error": "not available", "devices": [],
    })
    monkeypatch.setattr(power, "query_rapl_energy", lambda: {
        "status": "ok", "error": None, "zones": [{
            "id": "intel-rapl:0", "name": "package-0",
            "energy_uj": next(rapl_reads), "max_energy_range_uj": 1000,
            "status": "ok", "errors": [],
        }],
    })

    trace = power.run_benchmark(0.025, 0.005, str(tmp_path / "power.json"))
    assert trace["schema_version"] == power.SCHEMA_VERSION
    assert len(trace["samples"]) >= 2
    interval = trace["samples"][1]["rapl_intervals"][0]
    assert interval["status"] == "ambiguous"
    assert interval["error"] == "counter decreased; wrap versus reset is ambiguous"
    assert interval["energy_delta_uj"] == 20
    assert interval["average_power_w"] is None
    assert (tmp_path / "power.json").stat().st_mode & 0o777 == 0o600


def test_partial_endpoint_callback_applies_analyzer_timeout():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import numpy as np
    from src.audio.endpoint import EndpointState
    from src.main import AdamDaemon

    async def exercise():
        daemon = AdamDaemon.__new__(AdamDaemon)
        daemon.config = SimpleNamespace(stt=SimpleNamespace(provider="local"))
        daemon.stream = SimpleNamespace(sample_rate=16000)
        daemon.speaker_verifier = None
        daemon._transcribe_stt = lambda _audio, kind="final": "open the browser"
        daemon.endpointer = SimpleNamespace(
            analyze=lambda *_args, **_kwargs: (EndpointState.INCOMPLETE, 0.37)
        )
        daemon.speculative_router = SimpleNamespace(preflight=AsyncMock())
        callback = daemon._create_partial_callback(asyncio.get_running_loop())

        selected_timeout = callback(np.zeros(8000, dtype=np.float32))
        await asyncio.sleep(0)
        return selected_timeout

    assert asyncio.run(exercise()) == 0.37


def test_wake_stage_stt_events_share_the_capture_trace(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from src.main import AdamDaemon

    monkeypatch.setattr(telemetry, "_writer", None)
    telemetry.configure_telemetry(SimpleNamespace(enabled=True, path=str(tmp_path / "stt.jsonl")))
    daemon = AdamDaemon.__new__(AdamDaemon)
    daemon._active_capture_trace_id = "capture-trace"
    daemon.config = SimpleNamespace(stt=SimpleNamespace(
        provider="local", model_size="small.en", cloud_model=""
    ))
    daemon._stt_lock = threading.Lock()
    daemon.stt = SimpleNamespace(transcribe=lambda _audio: "hey adam")

    assert daemon._transcribe_stt(object(), kind="wake") == "hey adam"
    rows = [json.loads(line) for line in (tmp_path / "stt.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["stt.started", "stt.completed"]
    assert {row["trace_id"] for row in rows} == {"capture-trace"}
