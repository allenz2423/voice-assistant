import numpy as np
import pytest

from src.stt.enroll import record_sample
from src.stt.speaker import SAMPLE_RATE


class FakeAudioStream:
    def __init__(self, audio):
        self.audio = audio
        self.kwargs = None

    def record_utterance(self, **kwargs):
        self.kwargs = kwargs
        return self.audio


def test_enrollment_captures_variable_length_utterance_until_silence():
    audio = np.ones(8 * SAMPLE_RATE, dtype=np.float32)
    stream = FakeAudioStream(audio)

    result = record_sample(stream, silence_duration=1.4, max_duration=15.0)

    assert len(result) == len(audio)
    assert stream.kwargs["silence_duration"] == 1.4
    assert stream.kwargs["max_duration"] == 15.0


def test_enrollment_retries_when_capture_is_too_short():
    stream = FakeAudioStream(np.ones(SAMPLE_RATE, dtype=np.float32))

    with pytest.raises(ValueError, match="capture enough speech"):
        record_sample(stream)
