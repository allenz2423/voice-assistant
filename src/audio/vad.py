import numpy as np
import torch
from silero_vad import load_silero_vad

# NNPACK reports unsupported hardware repeatedly for some otherwise capable
# CPUs (including this machine). PyTorch falls back to its other CPU kernels;
# disable the optional backend to keep each audio inference from flooding logs.
torch.backends.nnpack.set_flags(False)

class SileroVAD:
    """Production Silero VAD (v5) ONNX wrapper with neural inference and acoustic energy floor."""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model = load_silero_vad(onnx=True)
        self.reset()

    def reset(self):
        self.model.reset_states()

    def is_speech(self, audio_chunk_16k: np.ndarray, threshold: float = 0.25, use_energy_floor: bool = True, energy_floor: float = 0.0025) -> tuple[bool, float]:
        """Calculates speech probability for 16kHz float32 audio frame with adaptive energy gating."""
        if len(audio_chunk_16k) == 0:
            return False, 0.0

        # Subframe in 512-sample blocks (32ms at 16kHz)
        max_prob = 0.0
        for i in range(0, len(audio_chunk_16k) - 256, 512):
            block = audio_chunk_16k[i:i + 512]
            if len(block) < 512:
                block = np.pad(block, (0, 512 - len(block)))
            tensor = torch.from_numpy(block)
            try:
                prob = self.model(tensor, self.sample_rate).item()
                if prob > max_prob:
                    max_prob = prob
            except Exception:
                pass

        # Acoustic noise gate: Typical ambient room noise is ~0.0004 - 0.002 RMS
        # Normal spoken voice is ~0.015 - 0.100 RMS
        # Suppress sub-threshold ambient noise or speaker bleed
        if use_energy_floor:
            energy = float(np.sqrt(np.mean(audio_chunk_16k**2)))
            if energy < energy_floor:
                return False, 0.0

        return max_prob >= threshold, max_prob
