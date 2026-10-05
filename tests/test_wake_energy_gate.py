import numpy as np

from src.config import AudioConfig
from src.wake.engine import WakeWordDetector


class FakeWakeModel:
    def __init__(self):
        self.prediction_calls = []
        self.preprocessor_calls = []
        self.reset_calls = 0

    def predict(self, chunk):
        self.prediction_calls.append(np.asarray(chunk).copy())
        return {"hey_jarvis": 0.9}

    def preprocessor(self, chunk):
        self.preprocessor_calls.append(np.asarray(chunk).copy())

    def reset(self):
        self.reset_calls += 1


def _detector(monkeypatch, energy_floor=0.0005):
    model = FakeWakeModel()

    def load(self):
        self.model = model
        self.model_key = "hey_jarvis"
        self.is_custom_mode = False

    monkeypatch.setattr(WakeWordDetector, "_load_model", load)
    return WakeWordDetector("hey jarvis", energy_floor=energy_floor), model


def test_wake_detector_skips_quiet_inference_and_replays_recent_context(monkeypatch):
    detector, model = _detector(monkeypatch)
    quiet = np.zeros(1280, dtype=np.float32)
    active = np.full(1280, 0.01, dtype=np.float32)

    for _ in range(5):
        detector.predict(quiet)
    for _ in range(35):
        assert detector.predict(quiet) == (False, 0.0)

    triggered, score = detector.predict(active)

    assert triggered and score == 0.9
    assert len(model.prediction_calls) == 6  # Five startup frames plus the active frame.
    assert len(model.preprocessor_calls) == detector._quiet_context_max_samples // 1280
    assert all(np.array_equal(chunk, quiet.astype(np.int16)) for chunk in model.preprocessor_calls)
    assert detector._quiet_context_samples == 0


def test_zero_wake_energy_floor_disables_gating(monkeypatch):
    detector, model = _detector(monkeypatch, energy_floor=0)
    quiet = np.zeros(1280, dtype=np.float32)

    for _ in range(10):
        detector.predict(quiet)

    assert len(model.prediction_calls) == 10
    assert not model.preprocessor_calls


def test_empty_audio_does_not_poison_cached_wake_context(monkeypatch):
    detector, model = _detector(monkeypatch)
    quiet = np.zeros(1280, dtype=np.float32)
    for _ in range(5):
        detector.predict(quiet)

    assert detector.predict(np.array([], dtype=np.float32)) == (False, 0.0)
    detector.predict(quiet)
    triggered, _ = detector.predict(np.full(1280, 0.01, dtype=np.float32))

    assert triggered
    assert len(model.preprocessor_calls) == 1


def test_wake_energy_floor_default_is_quiet_speech_conservative():
    assert AudioConfig().wake_energy_floor == 0.0005


def test_wake_reset_restores_model_warmup_and_clears_silence(monkeypatch):
    detector, model = _detector(monkeypatch)
    quiet = np.zeros(1280, dtype=np.float32)

    for _ in range(5):
        detector.predict(quiet)
    detector.predict(quiet)
    assert detector._quiet_context_samples == 1280

    detector.reset()
    for _ in range(5):
        detector.predict(quiet)

    assert model.reset_calls == 1
    assert len(model.prediction_calls) == 10
    assert detector._quiet_context_samples == 0


def test_custom_wake_recovery_hint_uses_distinctive_tokens(monkeypatch):
    def load_custom(detector):
        detector.is_custom_mode = True
        detector.custom_regex = detector._build_wake_regex(
            detector.raw_wake_word, detector.aliases
        )

    monkeypatch.setattr(WakeWordDetector, "_load_model", load_custom)
    detector = WakeWordDetector("Hey Adam", aliases=["Yo Adam"])

    assert detector.may_contain_custom_wake_word("Adam, check the weather")
    assert detector.may_contain_custom_wake_word("Yo Adam, check the weather")
    assert not detector.may_contain_custom_wake_word("Hey, I had a busy day")
    assert not detector.may_contain_custom_wake_word("I use a computer every day")


def test_speaker_isolation_requires_a_wake_hint_or_pretrained_detection(monkeypatch):
    from src.main import _should_attempt_speaker_isolation

    def load_custom(detector):
        detector.is_custom_mode = True
        detector.custom_regex = detector._build_wake_regex(
            detector.raw_wake_word, detector.aliases
        )

    monkeypatch.setattr(WakeWordDetector, "_load_model", load_custom)
    detector = WakeWordDetector("Hey Adam")

    assert not _should_attempt_speaker_isolation("Can you schedule a meeting?", detector)
    assert _should_attempt_speaker_isolation("Adam can you schedule a meeting?", detector)
    assert _should_attempt_speaker_isolation(
        "Can you schedule a meeting?", detector, pretrained_wake_triggered=True
    )
    assert _should_attempt_speaker_isolation(
        "I need to finish writing the quarterly report.", detector,
        transcript_confidence=-0.8, speaker_similarity=0.18, speaker_threshold=0.25,
    )
    assert not _should_attempt_speaker_isolation(
        "I need to finish writing the quarterly report.", detector,
        transcript_confidence=-0.4, speaker_similarity=0.8, speaker_threshold=0.25,
    )
    assert not _should_attempt_speaker_isolation(
        "I need to finish writing the quarterly report.", detector,
        transcript_confidence=-0.8, speaker_similarity=0.14, speaker_threshold=0.25,
    )


def test_whisper_confidence_transcription_preserves_plain_text_api():
    from types import SimpleNamespace
    from src.stt.transcriber import WhisperTranscriber

    class FakeModel:
        def transcribe(self, _audio, **_kwargs):
            return iter([
                SimpleNamespace(text=" Hey Adam", avg_logprob=-0.7),
                SimpleNamespace(text=" check the weather", avg_logprob=-0.5),
            ]), None

    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.model = FakeModel()
    audio = np.ones(1600, dtype=np.float32)

    assert transcriber.transcribe_with_confidence(audio) == ("Hey Adam check the weather", -0.6)
    assert transcriber.transcribe(audio) == "Hey Adam check the weather"
