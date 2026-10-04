from collections import deque
from importlib.metadata import distribution
from pathlib import Path

import numpy as np


class _NumpySileroOnnx:
    """Minimal NumPy/ONNX Silero wrapper that avoids importing PyTorch."""

    def __init__(self, sample_rate: int):
        import onnxruntime as ort

        model_path = Path(
            distribution("silero-vad").locate_file("silero_vad/data/silero_vad.onnx")
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"Silero ONNX model was not found: {model_path}")

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        self.sample_rate = sample_rate
        self.reset_states()

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 0), dtype=np.float32)

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float32).reshape(1, -1)
        if sample_rate != 16000 and sample_rate % 16000 == 0:
            values = values[:, :: sample_rate // 16000]
            sample_rate = 16000
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Unsupported Silero sampling rate: {sample_rate} Hz")
        expected_samples = 512 if sample_rate == 16000 else 256
        if values.shape[1] != expected_samples:
            raise ValueError(f"Silero expects {expected_samples} samples per frame")

        if not self._context.shape[1]:
            self._context = np.zeros((1, 64 if sample_rate == 16000 else 32), dtype=np.float32)
        model_input = np.concatenate((self._context, values), axis=1)
        output, self._state = self.session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": np.asarray(sample_rate, dtype=np.int64),
            },
        )
        self._context = model_input[:, -self._context.shape[1]:].copy()
        return output

class SileroVAD:
    """Silero VAD v5 on CPU ONNX with optional acoustic energy gating."""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model = _NumpySileroOnnx(sample_rate)
        self._quiet_context = deque(maxlen=8)
        self._energy_gated_silence = False
        self.reset()

    def reset(self):
        self.model.reset_states()
        self._quiet_context.clear()
        self._energy_gated_silence = False

    def is_speech(self, audio_chunk_16k: np.ndarray, threshold: float = 0.25, use_energy_floor: bool = True, energy_floor: float = 0.0025) -> tuple[bool, float]:
        """Calculates speech probability for 16kHz float32 audio frame with adaptive energy gating."""
        if len(audio_chunk_16k) == 0:
            return False, 0.0

        # Quiet chunks cannot pass the energy floor. Avoid running the neural
        # model on them. Keep a small ring of recent quiet frames, then replay
        # it once when speech energy returns so Silero retains onset context.
        if use_energy_floor:
            energy = float(np.sqrt(np.mean(audio_chunk_16k**2)))
            if energy < energy_floor:
                if not self._energy_gated_silence:
                    self.model.reset_states()
                    self._quiet_context.clear()
                    self._energy_gated_silence = True
                self._quiet_context.append(np.asarray(audio_chunk_16k, dtype=np.float32).copy())
                return False, 0.0

        if self._energy_gated_silence:
            for quiet_chunk in self._quiet_context:
                self._run_model(quiet_chunk)
            self._quiet_context.clear()
            self._energy_gated_silence = False

        # Subframe in 512-sample blocks (32ms at 16kHz)
        max_prob = self._run_model(audio_chunk_16k)

        return max_prob >= threshold, max_prob

    def _run_model(self, audio_chunk_16k: np.ndarray) -> float:
        """Run Silero across one chunk and return its maximum block score."""
        max_prob = 0.0
        for i in range(0, len(audio_chunk_16k) - 256, 512):
            block = audio_chunk_16k[i:i + 512]
            if len(block) < 512:
                block = np.pad(block, (0, 512 - len(block)))
            try:
                prob = float(np.asarray(self.model(block, self.sample_rate)).item())
                if prob > max_prob:
                    max_prob = prob
            except Exception:
                pass
        return max_prob
