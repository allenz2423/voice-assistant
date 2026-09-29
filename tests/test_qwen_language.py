from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from src.stt.transcriber import Qwen3Transcriber


def test_qwen_forwards_language_hint_to_transcribe_cpp():
    calls = []

    class Session:
        def run(self, audio, *, language=None):
            calls.append((audio, language))
            return SimpleNamespace(text="Hello there.")

    class Model:
        @contextmanager
        def session(self):
            yield Session()

    transcriber = Qwen3Transcriber.__new__(Qwen3Transcriber)
    transcriber.model = Model()
    transcriber.language = "English"

    audio = np.ones(1600, dtype=np.float32)
    assert transcriber.transcribe(audio) == "Hello there."
    assert calls == [(audio, "English")]


def test_qwen_can_use_automatic_language_detection():
    calls = []

    class Session:
        def run(self, _audio, *, language=None):
            calls.append(language)
            return SimpleNamespace(text="Transcript")

    class Model:
        @contextmanager
        def session(self):
            yield Session()

    transcriber = Qwen3Transcriber.__new__(Qwen3Transcriber)
    transcriber.model = Model()
    transcriber.language = ""

    assert transcriber.transcribe(np.ones(1600, dtype=np.float32)) == "Transcript"
    assert calls == [None]
