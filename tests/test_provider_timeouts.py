import math
import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.config import LLMConfig
from src.llm.provider import _request_timeout_seconds


def test_text_and_vision_requests_get_separate_bounded_defaults():
    config = LLMConfig()
    assert _request_timeout_seconds(config, has_image=False) == 45.0
    assert _request_timeout_seconds(config, has_image=True) == 90.0


def test_request_timeout_uses_configured_value_and_hard_ceiling():
    config = SimpleNamespace(request_timeout_seconds=20, vision_request_timeout_seconds=300)
    assert _request_timeout_seconds(config, has_image=False) == 20.0
    assert _request_timeout_seconds(config, has_image=True) == 120.0


@pytest.mark.parametrize("value", [None, "invalid", math.inf, -math.inf, math.nan])
def test_invalid_request_timeout_falls_back_to_a_bounded_default(value):
    config = SimpleNamespace(request_timeout_seconds=value)
    assert _request_timeout_seconds(config, has_image=False) == 45.0


def test_config_rejects_unbounded_provider_timeouts():
    with pytest.raises(ValidationError):
        LLMConfig(vision_request_timeout_seconds=600)


def test_openai_compatible_requests_apply_text_and_vision_timeouts(monkeypatch):
    import aiohttp
    import src.llm.provider as provider_module

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    sessions = []

    class FakeSession:
        def __init__(self, *, timeout):
            self.timeout = timeout
            sessions.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
    config = SimpleNamespace(llm=LLMConfig(
        provider="custom",
        api_base="https://provider.example/v1",
        cloud_model="vendor/model",
        request_timeout_seconds=17,
        vision_request_timeout_seconds=73,
    ))
    client = provider_module.UniversalLLMClient(config)

    async def run_requests():
        await client.chat([{"role": "user", "content": "text"}], tools=[])
        await client.chat([{"role": "user", "content": "visual", "images": [b"not-an-image"]}], tools=[])

    asyncio.run(run_requests())

    assert [session.timeout.total for session in sessions] == [17, 73]
