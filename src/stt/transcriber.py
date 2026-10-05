import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import glob
import ctypes
import asyncio
import io
import wave
import threading
import numpy as np

def _load_whisper_model_class(device: str = "cuda"):
    """Load CTranslate2 and optional CUDA libraries only when local ASR is used."""
    if str(device).lower() != "cpu":
        try:
            for lib_pattern in (
                ".venv/lib/python*/site-packages/nvidia/cublas/lib/libcublas.so.12",
                ".venv/lib/python*/site-packages/nvidia/cudnn/lib/libcudnn*.so.9",
            ):
                for lib_path in glob.glob(lib_pattern):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
        except Exception:
            pass
    from faster_whisper import WhisperModel
    return WhisperModel


CLOUD_STT_PROVIDERS = frozenset({"cloud", "openai", "openrouter", "custom"})

class WhisperTranscriber:
    """Hardware-accelerated Speech-to-Text engine using faster-whisper on CUDA or CPU."""
    def __init__(self, model_size="distil-large-v3", device="cuda", device_index=0, compute_type="int8_float32"):
        self.model_size = model_size
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        try:
            self.cpu_threads = max(1, min(256, int(os.environ.get("ADAM_CPU_THREADS", "4"))))
        except (TypeError, ValueError):
            self.cpu_threads = 4
        self.model = None
        self._load_model()

    def _load_model(self):
        WhisperModel = _load_whisper_model_class(self.device)
        try:
            device_label = f"{self.device}:{self.device_index}" if self.device != "cpu" else "cpu"
            print(f"[STT] Loading faster-whisper '{self.model_size}' on {device_label} ({self.compute_type})...")
            model_options = dict(
                device=self.device,
                device_index=self.device_index,
                compute_type=self.compute_type
            )
            if self.device == "cpu":
                model_options["cpu_threads"] = self.cpu_threads
            self.model = WhisperModel(self.model_size, **model_options)
            print(f"[STT] faster-whisper loaded successfully on {self.device}.")
        except Exception as e:
            print(f"[STT] Could not load faster-whisper on {self.device} ({e}); falling back to CPU int8...")
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.cpu_threads,
            )
            self.device = "cpu"
            print("[STT] faster-whisper loaded successfully on CPU (int8).")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribes 16kHz float32 mono audio array to text."""
        text, _confidence = self.transcribe_with_confidence(audio_data)
        return text

    def transcribe_with_confidence(self, audio_data: np.ndarray) -> tuple[str, float | None]:
        """Return text and mean decoder log probability for wake overlap checks."""
        if audio_data is None or len(audio_data) < 1600:  # < 100ms
            return "", None

        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_norm = audio_data / max_val
        else:
            audio_norm = audio_data

        try:
            segments, _ = self.model.transcribe(
                audio_norm,
                beam_size=1,
                language="en",
                vad_filter=False
            )
            segments = list(segments)
            text = " ".join(str(s.text).strip() for s in segments if s.text).strip()
            log_probs = [
                float(s.avg_logprob) for s in segments
                if getattr(s, "avg_logprob", None) is not None
            ]
            confidence = sum(log_probs) / len(log_probs) if log_probs else None
            return text, confidence
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return "", None

class Qwen3Transcriber:
    """Hardware-accelerated Speech-to-Text engine using Qwen3-ASR-1.7B via transcribe-cpp."""
    def __init__(self, model_size="qwen3-asr-1.7b", device="Vulkan0", language=""):
        import transcribe_cpp
        from huggingface_hub import hf_hub_download

        quant = "Q4_K_M" if "q4" in model_size.lower() else "Q8_0"
        filename = f"Qwen3-ASR-1.7B-{quant}.gguf"
        repo_id = "handy-computer/Qwen3-ASR-1.7B-gguf"

        print(f"[STT] Loading Qwen3-ASR ({filename}) from HuggingFace...")
        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.language = str(language or "").strip()
        if self.language and self.language.lower() != "auto":
            print(
                "[STT] This transcribe.cpp Qwen3-ASR runtime only supports automatic language detection; "
                f"ignoring the configured hint {self.language!r}."
            )

        backends = transcribe_cpp.backends()
        selected_dev = next((d for d in backends if d.name.lower() == str(device).lower()), None)
        if not selected_dev:
            # Look for Vulkan GPU or first available GPU device
            selected_dev = next((d for d in backends if "vulkan" in d.kind.lower() and getattr(d, "device_type", "") == "gpu"), None)
            if not selected_dev:
                selected_dev = next((d for d in backends if getattr(d, "device_type", "") == "gpu"), None)
            if not selected_dev:
                selected_dev = backends[0] if backends else None

        dev_desc = f"{selected_dev.name} ({selected_dev.description})" if selected_dev else "CPU"
        print(f"[STT] Initializing Qwen3-ASR-1.7B on {dev_desc}...")
        self.model = transcribe_cpp.Model(self.model_path, device=selected_dev)
        print("[STT] Qwen3-ASR-1.7B loaded successfully!")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribes 16kHz float32 mono audio array to text."""
        if audio_data is None or len(audio_data) < 1600:
            return ""
        try:
            with self.model.session() as session:
                # transcribe.cpp's Qwen3-ASR family currently rejects every
                # explicit language value; omit it to use model auto-detection.
                res = session.run(audio_data)
                return res.text.strip()
        except Exception as e:
            print(f"[STT] Qwen3-ASR transcription error: {e}")
            return ""


class OpenAITranscriber:
    """OpenAI-style cloud transcription with a lazy Faster-Whisper fallback."""
    def __init__(
        self,
        model="gpt-transcribe",
        api_key="",
        endpoint="https://api.openai.com/v1/audio/transcriptions",
        fallback_model="base.en",
        fallback_device="cpu",
        fallback_compute_type="int8",
        fallback_device_index=0,
        shared_api_key="",
    ):
        self.model = model
        self.api_key = api_key or shared_api_key
        self.endpoint = endpoint or "https://api.openai.com/v1/audio/transcriptions"
        self.fallback_model = fallback_model
        self.fallback_device = fallback_device
        self.fallback_compute_type = fallback_compute_type
        self.fallback_device_index = fallback_device_index
        self._fallback = None
        self._fallback_lock = threading.Lock()
        self.last_usage: dict[str, int | float | str] = {}

    def attach_local_fallback(self, transcriber: WhisperTranscriber) -> bool:
        """Reuse an already-loaded local model when its fallback settings match."""
        if (
            getattr(transcriber, "model_size", None) != self.fallback_model
            or getattr(transcriber, "device", None) != self.fallback_device
            or getattr(transcriber, "device_index", None) != self.fallback_device_index
            or getattr(transcriber, "compute_type", None) != self.fallback_compute_type
        ):
            return False
        with self._fallback_lock:
            if self._fallback is not None and self._fallback is not transcriber:
                return False
            self._fallback = transcriber
        return True

    def _transcribe_fallback(self, audio_data: np.ndarray, reason: str) -> str:
        try:
            with self._fallback_lock:
                if self._fallback is None:
                    print(f"[STT] Cloud transcription unavailable ({reason}); loading Faster-Whisper fallback.", flush=True)
                    self._fallback = WhisperTranscriber(
                        model_size=self.fallback_model,
                        device=self.fallback_device,
                        device_index=self.fallback_device_index,
                        compute_type=self.fallback_compute_type,
                    )
            return self._fallback.transcribe(audio_data)
        except Exception as e:
            print(f"[STT] Faster-Whisper fallback failed: {e}", flush=True)
            return ""

    def transcribe(self, audio_data: np.ndarray) -> str:
        if audio_data is None or len(audio_data) < 1600:
            return ""
        self.last_usage = {}
        if not self.api_key:
            return self._transcribe_fallback(audio_data, "no API key configured")

        pcm = np.clip(audio_data, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm.tobytes())

        async def _request():
            import aiohttp
            form = aiohttp.FormData()
            form.add_field("model", self.model)
            form.add_field("file", wav_io.getvalue(), filename="utterance.wav", content_type="audio/wav")
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=form,
                ) as response:
                    payload = await response.json(content_type=None)
                    if response.status != 200:
                        raise RuntimeError(f"OpenAI transcription failed ({response.status}): {payload}")
                    return payload

        try:
            payload = asyncio.run(_request())
            transcript = str(payload.get("text", "")).strip()
            usage = payload.get("usage")
            if isinstance(usage, dict):
                cost = usage.get("cost")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    self.last_usage = {
                        "provider_reported_cost": float(cost),
                        "currency": "USD",
                    }
            if transcript:
                return transcript
            return self._transcribe_fallback(audio_data, "cloud response contained no transcript")
        except Exception as e:
            return self._transcribe_fallback(audio_data, str(e))

def create_transcriber(config, shared_api_key=""):
    """Factory creating either Qwen3Transcriber or WhisperTranscriber based on config."""
    provider = getattr(config, "provider", "local").lower()
    if provider in CLOUD_STT_PROVIDERS:
        if provider == "openai":
            env_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider == "openrouter":
            env_key = os.environ.get("OPENROUTER_API_KEY", "")
        else:
            env_key = os.environ.get("CLOUD_STT_API_KEY", "")
        return OpenAITranscriber(
            model=getattr(config, "cloud_model", "gpt-transcribe"),
            api_key=getattr(config, "api_key", "") or env_key,
            endpoint=getattr(config, "cloud_url", "https://api.openai.com/v1/audio/transcriptions"),
            fallback_model=getattr(config, "fallback_model", "base.en"),
            fallback_device=getattr(config, "fallback_device", "cpu"),
            fallback_compute_type=getattr(config, "fallback_compute_type", "int8"),
            fallback_device_index=getattr(config, "device_index", 0),
            shared_api_key=shared_api_key if provider in ("cloud", "openrouter", "custom") else "",
        )
    model_name = getattr(config, "model_size", "distil-large-v3").lower()
    if "qwen" in model_name:
        device = getattr(config, "device", "Vulkan0")
        return Qwen3Transcriber(
            model_size=model_name,
            device=device,
            language=getattr(config, "language", ""),
        )
    else:
        return WhisperTranscriber(
            model_size=config.model_size,
            device=config.device,
            device_index=config.device_index,
            compute_type=config.compute_type
        )
