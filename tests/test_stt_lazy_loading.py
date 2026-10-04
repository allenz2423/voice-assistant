import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.stt.transcriber import OpenAITranscriber, WhisperTranscriber, _load_whisper_model_class
from src.main import AdamDaemon


def test_cpu_whisper_does_not_preload_cuda_libraries(monkeypatch):
    import sys
    from types import ModuleType

    fake_runtime = ModuleType("faster_whisper")
    fake_runtime.WhisperModel = object()
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_runtime)
    monkeypatch.setattr("src.stt.transcriber.glob.glob", lambda _pattern: ["/fake/libcuda.so"])
    monkeypatch.setattr(
        "src.stt.transcriber.ctypes.CDLL",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("CPU path loaded a CUDA library")),
    )

    assert _load_whisper_model_class("cpu") is fake_runtime.WhisperModel


def test_cpu_whisper_model_uses_bounded_thread_count(monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, model_size, **kwargs):
            calls.append((model_size, kwargs))

    monkeypatch.setenv("ADAM_CPU_THREADS", "3")
    monkeypatch.setattr("src.stt.transcriber._load_whisper_model_class", lambda _device: FakeWhisperModel)

    transcriber = WhisperTranscriber(
        model_size="base.en",
        device="cpu",
        device_index=0,
        compute_type="int8",
    )

    assert transcriber.cpu_threads == 3
    assert calls == [("base.en", {
        "device": "cpu",
        "device_index": 0,
        "compute_type": "int8",
        "cpu_threads": 3,
    })]


def test_cloud_stt_does_not_import_local_asr_runtime():
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import sys
from types import SimpleNamespace
from src.stt.transcriber import create_transcriber

config = SimpleNamespace(
    provider="cloud",
    model_size="small.en",
    cloud_model="openai/whisper-large-v3-turbo",
    cloud_url="https://openrouter.ai/api/v1/audio/transcriptions",
    fallback_model="base.en",
    fallback_device="cpu",
    fallback_compute_type="int8",
)
transcriber = create_transcriber(config, shared_api_key="test-key")
assert transcriber._fallback is None
assert transcriber.model == "openai/whisper-large-v3-turbo"
assert transcriber.endpoint == "https://openrouter.ai/api/v1/audio/transcriptions"
assert transcriber.api_key == "test-key"
assert "faster_whisper" not in sys.modules
assert "ctranslate2" not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_cloud_stt_retains_only_reported_cost(monkeypatch):
    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self, **_kwargs):
            return {
                "text": "The test phrase was recognized.",
                "usage": {"cost": 0.000012, "seconds": 2.5, "transcript": "must not be retained"},
            }

    class FakeSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return FakeResponse()

    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
    transcriber = OpenAITranscriber(api_key="test-key")
    transcript = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert transcript == "The test phrase was recognized."
    assert transcriber.last_usage == {
        "provider_reported_cost": 0.000012,
        "currency": "USD",
    }


def test_cloud_fallback_reuses_compatible_wake_spotter():
    fallback = SimpleNamespace(
        model_size="base.en",
        device="cpu",
        device_index=0,
        compute_type="int8",
        transcribe=MagicMock(return_value="local transcript"),
    )
    transcriber = OpenAITranscriber(
        api_key="",
        fallback_model="base.en",
        fallback_device="cpu",
        fallback_compute_type="int8",
        fallback_device_index=0,
    )

    assert transcriber.attach_local_fallback(fallback)
    assert transcriber.transcribe(np.zeros(16000, dtype=np.float32)) == "local transcript"
    fallback.transcribe.assert_called_once()


def test_cloud_fallback_rejects_incompatible_wake_spotter():
    fallback = SimpleNamespace(
        model_size="small.en",
        device="cpu",
        device_index=0,
        compute_type="int8",
        transcribe=MagicMock(),
    )
    transcriber = OpenAITranscriber(api_key="", fallback_model="base.en")
    assert not transcriber.attach_local_fallback(fallback)
    assert transcriber._fallback is None


def test_cloud_wake_spotting_stays_on_local_fallback():
    app = object.__new__(AdamDaemon)
    app.config = SimpleNamespace(
        stt=SimpleNamespace(
            provider="cloud",
            fallback_model="base.en",
            fallback_device="cpu",
            fallback_compute_type="int8",
            device_index=0,
        )
    )
    app.wake_spotter = None
    app.stt = None
    app._stt_lock = threading.Lock()
    wake_model = MagicMock()
    wake_model.transcribe.return_value = "hey adam"

    with patch("src.main.WhisperTranscriber", return_value=wake_model) as load_model:
        transcript = app._transcribe_wake_candidate(np.zeros(16000, dtype=np.float32))

    assert transcript == "hey adam"
    load_model.assert_called_once_with(
        model_size="base.en",
        device="cpu",
        device_index=0,
        compute_type="int8",
    )
    wake_model.transcribe.assert_called_once()
