import sys
import types

import numpy as np
import pytest

from src.stt.diarizer import NemotronDiarizer, SpeakerSpan
from src.main import AdamDaemon


def test_exclusive_speaker_audio_drops_overlapping_frames():
    audio = np.arange(20000, dtype=np.float32)
    spans = [SpeakerSpan("a", 0.0, 0.75), SpeakerSpan("b", 0.5, 1.25)]

    exclusive = NemotronDiarizer.exclusive_speaker_audio(audio, spans, sample_rate=16000)

    assert set(exclusive) == {"a", "b"}
    np.testing.assert_array_equal(exclusive["a"], audio[:8000])
    np.testing.assert_array_equal(exclusive["b"], audio[12000:20000])


def test_speaker_turns_keep_original_mix_and_mark_overlap():
    audio = np.linspace(-0.8, 0.8, 32000, dtype=np.float32)
    spans = [SpeakerSpan("a", 0.0, 1.2), SpeakerSpan("b", 0.6, 1.8)]

    turns = NemotronDiarizer.speaker_turns(
        audio, spans, sample_rate=16000, min_seconds=0.1, merge_gap=0.0
    )

    assert [turn.speakers for turn in turns] == [("a",), ("a", "b"), ("b",)]
    np.testing.assert_array_equal(turns[0].audio, audio[:9600])
    np.testing.assert_array_equal(turns[1].audio, audio[9600:19200])
    np.testing.assert_array_equal(turns[2].audio, audio[19200:28800])


@pytest.mark.parametrize("device", ["vulkan:0", "auto"])
def test_diarize_runs_transformers_model_and_converts_segments(monkeypatch, device):
    class FakeBatch(dict):
        attention_mask = "mask"

        def to(self, _device, dtype=None):
            return self

    class FakeProcessor:
        feature_extractor = types.SimpleNamespace(sampling_rate=16000)

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert model == "nvidia/Nemotron-3-Diarization"
            return cls()

        def __call__(self, audio, sampling_rate):
            assert len(audio) == 16000
            assert sampling_rate == 16000
            return FakeBatch()

        def extract_speaker_dict(self, logits, attention_mask):
            assert logits == "logits"
            assert attention_mask == "mask"
            return [[{"Speaker": 1, "Start": 0.125, "End": 0.75}]]

    class FakeModel:
        device, dtype = "cpu", "float32"

        @classmethod
        def from_pretrained(cls, model, dtype, **kwargs):
            assert model == "nvidia/Nemotron-3-Diarization"
            assert dtype == "float32"
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            pass

        def __call__(self, **inputs):
            return types.SimpleNamespace(logits="logits")

    fake_torch = types.SimpleNamespace(
        # Even with CUDA available, auto and legacy Vulkan must not pick a GPU.
        cuda=types.SimpleNamespace(is_available=lambda: True),
        float16="float16", float32="float32", inference_mode=lambda: __import__("contextlib").nullcontext(),
    )
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.AutoModelForAudioFrameClassification = FakeModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    # Old setup wrote vulkan:0; the new PyTorch path migrates it to available CPU.
    spans = NemotronDiarizer(device=device).diarize(np.zeros(16000, dtype=np.float32))

    assert spans == [SpeakerSpan("speaker_1", 0.125, 0.75)]


def test_empty_audio_does_not_load_model():
    assert NemotronDiarizer().diarize(np.array([], dtype=np.float32)) == []


def test_cuda_support_check_rejects_gpu_arch_missing_from_torch_build():
    fake_cuda = types.SimpleNamespace(
        get_device_capability=lambda index: (6, 1),
        get_arch_list=lambda: ["sm_75", "sm_80", "sm_86"],
    )
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)

    supported, reason = NemotronDiarizer._cuda_supports_device(fake_torch, "cuda:0")

    assert not supported
    assert "sm_61" in reason


def test_registered_speaker_diarization_uses_context_but_returns_utterance_only():
    context = np.ones(16000, dtype=np.float32)
    utterance = np.full(32000, 2.0, dtype=np.float32)

    class FakeDiarizer:
        def diarize(self, audio):
            np.testing.assert_array_equal(audio, np.concatenate((context, utterance)))
            return [
                SpeakerSpan("speaker_0", 0.0, 0.8),
                SpeakerSpan("speaker_1", 1.1, 1.8),
            ]

        def exclusive_speaker_audio(self, audio, spans, sample_rate):
            np.testing.assert_array_equal(audio, utterance)
            assert sample_rate == 16000
            assert len(spans) == 1
            assert spans[0].speaker == "speaker_1"
            assert spans[0].start == pytest.approx(0.1)
            assert spans[0].end == pytest.approx(0.8)
            return {"speaker_1": audio[:12000]}

    class FakeVerifier:
        def verify(self, audio):
            assert len(audio) == 12000
            return True, 0.4

    daemon = AdamDaemon.__new__(AdamDaemon)
    daemon.speaker_diarizer = FakeDiarizer()
    daemon.speaker_verifier = FakeVerifier()
    daemon.stream = types.SimpleNamespace(sample_rate=16000)

    result = daemon._registered_speaker_audio(utterance, context)

    np.testing.assert_array_equal(result, utterance[:12000])


def test_supports_a_downloaded_local_transformers_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "processor_config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    assert NemotronDiarizer(model=str(tmp_path)).supports_model()
