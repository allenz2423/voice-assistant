import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import glob
import ctypes
import numpy as np

# Auto-preload NVIDIA CUDA 12 libraries from venv if present
try:
    for lib_pattern in [
        ".venv/lib/python*/site-packages/nvidia/cublas/lib/libcublas.so.12",
        ".venv/lib/python*/site-packages/nvidia/cudnn/lib/libcudnn*.so.9"
    ]:
        for lib_path in glob.glob(lib_pattern):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
except Exception:
    pass

from faster_whisper import WhisperModel

class WhisperTranscriber:
    """Hardware-accelerated Speech-to-Text engine using faster-whisper on CUDA or CPU."""
    def __init__(self, model_size="distil-large-v3", device="cuda", device_index=0, compute_type="int8_float32"):
        self.model_size = model_size
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"[STT] Loading faster-whisper '{self.model_size}' on {self.device}:{self.device_index} ({self.compute_type})...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                device_index=self.device_index,
                compute_type=self.compute_type
            )
            print("[STT] faster-whisper loaded successfully on CUDA!")
        except Exception as e:
            print(f"[STT] CUDA initialization failed ({e}), falling back to CPU int8...")
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8"
            )

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribes 16kHz float32 mono audio array to text."""
        if audio_data is None or len(audio_data) < 1600:  # < 100ms
            return ""

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
            text = " ".join([s.text for s in segments]).strip()
            return text
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return ""

class Qwen3Transcriber:
    """Hardware-accelerated Speech-to-Text engine using Qwen3-ASR-1.7B via transcribe-cpp."""
    def __init__(self, model_size="qwen3-asr-1.7b", device="Vulkan0"):
        import transcribe_cpp
        from huggingface_hub import hf_hub_download

        quant = "Q4_K_M" if "q4" in model_size.lower() else "Q8_0"
        filename = f"Qwen3-ASR-1.7B-{quant}.gguf"
        repo_id = "handy-computer/Qwen3-ASR-1.7B-gguf"

        print(f"[STT] Loading Qwen3-ASR ({filename}) from HuggingFace...")
        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)

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
                res = session.run(audio_data)
                return res.text.strip()
        except Exception as e:
            print(f"[STT] Qwen3-ASR transcription error: {e}")
            return ""

def create_transcriber(config):
    """Factory creating either Qwen3Transcriber or WhisperTranscriber based on config."""
    model_name = getattr(config, "model_size", "distil-large-v3").lower()
    if "qwen" in model_name:
        device = getattr(config, "device", "Vulkan0")
        return Qwen3Transcriber(model_size=model_name, device=device)
    else:
        return WhisperTranscriber(
            model_size=config.model_size,
            device=config.device,
            device_index=config.device_index,
            compute_type=config.compute_type
        )
