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

@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['json', 'shape', 'choices', 'message', '429', 'provider429', 'timeout', 'payload'])
async def test_provider_failure_recovers_with_three_retries_and_preserves_progress(monkeypatch, failure):
    import aiohttp
    from unittest.mock import AsyncMock
    from src.llm.provider import UniversalLLMClient
    requests = []

    class Response:
        headers = {}
        def __init__(self, attempt):
            self.attempt = attempt
            self.status = 429 if failure == '429' and attempt < 4 else 200
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return False
        async def json(self):
            if self.attempt == 4:
                return {'choices': [{'message': {'content': 'Recovered', 'tool_calls': []}}]}
            if failure == 'json': raise ValueError('malformed JSON')
            if failure == 'timeout': raise asyncio.TimeoutError()
            if failure == 'payload': raise aiohttp.ClientPayloadError('truncated body')
            if failure == 'shape': return []
            if failure == 'choices': return {'choices': []}
            if failure == 'provider429': return {'error': {'code': 429}}
            return {'choices': [{'message': None}]}

    class Session:
        def __init__(self, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return False
        def post(self, *_args, **kwargs):
            requests.append(kwargs['json'])
            return Response(len(requests))

    monkeypatch.setattr(aiohttp, 'ClientSession', Session)
    sleep = AsyncMock()
    monkeypatch.setattr('src.llm.provider.asyncio.sleep', sleep)
    client = UniversalLLMClient(SimpleNamespace(llm=LLMConfig(provider='custom', api_base='https://example.test/v1')))
    messages = [{'role': 'user', 'content': 'Continue the task'},
                {'role': 'assistant', 'tool_calls': [{'id':'done', 'type':'function', 'function':{'name':'read_file', 'arguments':'{}'}}]},
                {'role':'tool', 'tool_call_id':'done', 'content':'Already inspected the file'}]
    result = await client.chat(messages, tools=[])
    assert result['content'] == 'Recovered'
    assert len(requests) == 4
    assert sleep.await_count == 3
    assert all(any(m.get('tool_call_id') == 'done' for m in r['messages']) for r in requests)
    assert len(messages) == 3
    if failure not in {'429', 'provider429', 'timeout', 'payload'}:
        assert 'do not repeat completed actions' in requests[-1]['messages'][-1]['content']
    else:
        assert requests[-1]['messages'] == requests[0]['messages']


@pytest.mark.asyncio
async def test_provider_retries_stop_at_three_and_do_not_retry_permanent_errors(monkeypatch):
    from unittest.mock import AsyncMock
    from src.llm.provider import UniversalLLMClient
    from src.telemetry.events import subscribe_events
    client = UniversalLLMClient(SimpleNamespace(llm=LLMConfig()))
    monkeypatch.setattr('src.llm.provider.asyncio.sleep', AsyncMock())
    request = AsyncMock(side_effect=lambda *args: {'provider_error':True, '_retry_status':429})
    events = []
    unsubscribe = subscribe_events(events.append)
    try:
        result = await client._chat_with_retries(request, [], [], None, None)
    finally:
        unsubscribe()
    assert result['provider_error'] is True
    assert request.await_count == 4
    retries = [event for event in events if event['event'] == 'llm.retrying']
    assert [event['attributes']['attempt'] for event in retries] == [1, 2, 3]
    assert all(event['attributes']['max_attempts'] == 3 for event in retries)
    permanent = AsyncMock(return_value={'provider_error':True})
    await client._chat_with_retries(permanent, [], [], None, None)
    assert permanent.await_count == 1
    cancelled = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await client._chat_with_retries(cancelled, [], [], None, None)
    assert cancelled.await_count == 1
