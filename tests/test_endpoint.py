from src.audio.endpoint import SemanticEndpointer, EndpointState


def test_complete_standalone_commands():
    ep = SemanticEndpointer()
    for phrase in [
        "pause",
        "stop",
        "resume",
        "close this tab",
        "what time is it",
        "what's the weather",
        "yes",
        "cancel",
        "list windows",
    ]:
        state, duration = ep.analyze(phrase)
        assert state == EndpointState.COMPLETE, f"Expected {phrase} to be COMPLETE, got {state}"
        assert duration == ep.complete_silence


def test_incomplete_dangling_phrases():
    ep = SemanticEndpointer()
    for phrase in [
        "could you search for",
        "remind me to",
        "set a timer for",
        "what is the weather in",
        "open the",
        "turn on",
        "because of",
        "and",
        "uh",
        "hey adam",  # Only wake word
    ]:
        state, duration = ep.analyze(phrase, wake_word="hey adam")
        assert state == EndpointState.INCOMPLETE, f"Expected '{phrase}' to be INCOMPLETE, got {state}"
        assert duration == ep.incomplete_silence


def test_wake_word_prefix_stripping():
    ep = SemanticEndpointer()
    state, duration = ep.analyze("Hey Adam, what time is it?", wake_word="hey adam")
    assert state == EndpointState.COMPLETE
    assert duration == ep.complete_silence

    state, duration = ep.analyze("Hey Adam, search for", wake_word="hey adam")
    assert state == EndpointState.INCOMPLETE
    assert duration == ep.incomplete_silence


def test_punctuated_sentences():
    ep = SemanticEndpointer()
    state, duration = ep.analyze("Could you open Microsoft Edge.")
    assert state == EndpointState.COMPLETE
    assert duration == ep.complete_silence


def test_record_utterance_partial_callback():
    from unittest.mock import MagicMock
    import queue
    import numpy as np
    from src.audio.stream import AudioStreamManager

    manager = AudioStreamManager.__new__(AudioStreamManager)
    manager.quench_until = 0.0
    manager.is_assistant_speaking = False
    manager.ref_monitor = None
    manager.sample_rate = 16000
    manager.chunk_size = 512
    manager.audio_queue = queue.Queue()
    manager.vad = MagicMock()
    manager.vad.reset = MagicMock()
    manager.flush = MagicMock()

    # Mock VAD: speech for 10 chunks, then silence
    manager.vad.is_speech.side_effect = [(True, 0.9)] * 10 + [(False, 0.0)] * 50

    # Put chunks into audio_queue
    for _ in range(25):
        manager.audio_queue.put(np.ones(512, dtype=np.float32) * 0.1)

    partial_calls = []

    def on_partial(audio):
        partial_calls.append(len(audio))
        return 0.05  # shorten silence duration dynamically

    audio = manager.record_utterance(
        silence_duration=1.0,
        max_duration=1.0,
        on_partial_audio=on_partial,
        partial_interval_s=0.0,
    )
    assert len(partial_calls) > 0
    assert len(audio) > 0
