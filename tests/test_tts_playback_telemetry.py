import asyncio
import json
from types import SimpleNamespace

import pytest

import src.telemetry.events as telemetry


@pytest.mark.parametrize("role", ["acknowledgment", None])
def test_playback_events_carry_only_role_and_utterance_metadata(role):
    from src.tts.streaming import StreamingVoiceSynthesizer

    events = []
    unsubscribe = telemetry.subscribe_events(events.append)

    def fake_playback(_private_speech, *, playback_span_id, playback_state):
        playback_state["submitted"] = True

    async def run_playback():
        if role is None:
            await StreamingVoiceSynthesizer._play_interruptibly(
                SimpleNamespace(current_epoch=4), fake_playback, "private speech sentinel",
                epoch=4, utterance_id="utterance-1",
            )
            return
        with telemetry.speech_role_scope(role):
            await StreamingVoiceSynthesizer._play_interruptibly(
                SimpleNamespace(current_epoch=4), fake_playback, "private speech sentinel",
                epoch=4, utterance_id="utterance-1",
            )

    try:
        asyncio.run(run_playback())
    finally:
        unsubscribe()

    assert [event["event"] for event in events] == [
        "playback.started", "playback.completed",
    ]
    assert [event["attributes"] for event in events] == [
        {"speech_role": role, "utterance_id": "utterance-1"},
        {"speech_role": role, "utterance_id": "utterance-1"},
    ]
    assert "private speech sentinel" not in json.dumps(events)
    assert telemetry.get_speech_role() is None


def test_speech_role_scope_rejects_unrecognized_labels():
    with pytest.raises(ValueError, match="unsupported speech role"):
        with telemetry.speech_role_scope("transcript"):
            pass


def test_epoch_change_in_worker_completion_window_is_not_counted_as_success(monkeypatch):
    from src.tts.streaming import StreamingVoiceSynthesizer

    events = []
    unsubscribe = telemetry.subscribe_events(events.append)
    synth = SimpleNamespace(current_epoch=4)

    def fake_playback(_private_speech, *, playback_span_id, playback_state):
        playback_state["submitted"] = True
        # Model a barge-in racing with the worker's final return after a buffer
        # was submitted. The awaiting coroutine resumes only after the worker.
        synth.current_epoch += 1

    async def immediate_to_thread(function, *args, **kwargs):
        function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    try:
        asyncio.run(StreamingVoiceSynthesizer._play_interruptibly(
            synth, fake_playback, "private speech sentinel", epoch=4,
            utterance_id="interrupted-utterance",
        ))
    finally:
        unsubscribe()

    assert [event["event"] for event in events] == [
        "playback.started", "playback.completed",
    ]
    assert events[-1]["status"] == "cancelled"
    assert events[-1]["attributes"] == {
        "speech_role": None,
        "utterance_id": "interrupted-utterance",
    }
