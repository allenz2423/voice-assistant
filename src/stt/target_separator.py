"""CPU SepFormer inference for selecting the enrolled speaker from two sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class TargetSpeakerResult:
    audio: np.ndarray
    score: float
    source_index: int
    source_tracks: tuple[np.ndarray, ...]


class TargetSpeakerSeparator:
    """Separate two speech sources and choose the one matching the voice profile.

    The SpeechBrain Libri2Mix checkpoint is intentionally pinned to CPU. It is a
    fallback for command isolation, not an unrestricted multi-speaker meeting
    diarizer; meeting labels come from Nemotron plus speaker embeddings.
    """

    def __init__(self, verifier, model_source: str = "speechbrain/sepformer-libri2mix", model_dir: str | None = None):
        self.verifier = verifier
        self.model_source = model_source
        self.model_dir = model_dir
        self._separator = None

    def _load(self):
        if self._separator is None:
            import torch
            from speechbrain.inference.separation import SepformerSeparation

            torch.set_num_threads(min(4, torch.get_num_threads()))
            options = {"source": self.model_source, "run_opts": {"device": "cpu"}}
            if self.model_dir:
                options["savedir"] = self.model_dir
            self._separator = SepformerSeparation.from_hparams(**options)
        return self._separator

    def separate(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[np.ndarray, ...]:
        """Return source tracks at the input sample rate without PCM clipping."""
        import torch

        source = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            raise ValueError("TargetSpeakerSeparator expects 16 kHz mono audio")
        if source.size < int(0.5 * sample_rate):
            return ()
        # SpeechBrain's Libri2Mix checkpoint expects 8 kHz audio.
        mix_8k = resample_poly(source, 1, 2).astype(np.float32, copy=False)
        with torch.inference_mode():
            separated = self._load().separate_batch(torch.from_numpy(mix_8k).unsqueeze(0))
        tracks = []
        for index in range(separated.shape[-1]):
            track_8k = separated[0, :, index].detach().cpu().numpy().astype(np.float32, copy=False)
            tracks.append(resample_poly(track_8k, 2, 1).astype(np.float32, copy=False))
        return tuple(tracks)

    def isolate(self, audio: np.ndarray, sample_rate: int = 16000) -> TargetSpeakerResult | None:
        """Return the highest-scoring source only when it clears profile threshold."""
        if self.verifier is None or not self.verifier.enrolled:
            return None
        tracks = self.separate(audio, sample_rate=sample_rate)
        eligible = []
        for index, track in enumerate(tracks):
            if len(track) < int(0.5 * sample_rate):
                continue
            try:
                _, score = self.verifier.verify(track)
            except Exception:
                continue
            eligible.append((float(score), index))
        if not eligible:
            return None
        score, index = max(eligible)
        if score < self.verifier.threshold:
            return None
        target = tracks[index]
        peak = float(np.max(np.abs(target))) if target.size else 0.0
        if peak > 1e-8:
            # The separator's internal normalization can produce floats outside
            # [-1, 1]. Scale cleanly for ASR rather than clipping the waveform.
            target = target * (0.95 / peak)
        return TargetSpeakerResult(target.astype(np.float32, copy=False), score, index, tracks)
