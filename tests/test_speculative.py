import pytest
import asyncio
from src.llm.speculative import SpeculativeRouter


@pytest.mark.asyncio
async def test_parse_partial_intent():
    router = SpeculativeRouter()

    # Time
    intent = router.parse_partial_intent("hey adam what time is it in Tokyo", wake_word="hey adam")
    assert intent == ("get_current_time", {"location": "Tokyo"})

    # Weather
    intent = router.parse_partial_intent("what's the weather in Paris")
    assert intent == ("get_weather", {"location": "Paris"})

    # Web search
    intent = router.parse_partial_intent("hey adam could you search for AMD Ryzen 9000", wake_word="hey adam")
    assert intent == ("web_search", {"query": "AMD Ryzen 9000"})

    # Battery / System
    intent = router.parse_partial_intent("how much battery is left")
    assert intent == ("get_system_status", {})

    # Incomplete search (should not fire yet on dangling connector)
    intent = router.parse_partial_intent("search for")
    assert intent is None


@pytest.mark.asyncio
async def test_speculative_cache_hit():
    mock_weather_data = "Paris: 18°C, Partly Cloudy"

    async def fake_weather(location="local"):
        await asyncio.sleep(0.05)
        return f"{location}: 18°C, Partly Cloudy"

    router = SpeculativeRouter(executor_map={"get_weather": fake_weather})

    # 1. Partial transcript arrives while speaking
    launched = await router.preflight("what's the weather in Paris")
    assert launched is True

    # 2. Utterance finishes and brain calls get_weather
    hit, result = await router.consume_speculative_result("get_weather", {"location": "Paris"})
    assert hit is True
    assert result == "Paris: 18°C, Partly Cloudy"


@pytest.mark.asyncio
async def test_speculative_cancellation_on_intent_change():
    called = []

    async def slow_weather(location="local"):
        try:
            await asyncio.sleep(1.0)
            called.append("weather")
            return "weather"
        except asyncio.CancelledError:
            called.append("cancelled")
            raise

    router = SpeculativeRouter(executor_map={"get_weather": slow_weather})

    await router.preflight("what's the weather in Paris")
    assert router.active_entry is not None
    await asyncio.sleep(0.01)

    # User changed their mind and asked for time instead
    router.cancel_active()
    assert router.active_entry is None
    await asyncio.sleep(0.02)
    assert "cancelled" in called
