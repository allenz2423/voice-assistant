"""Standalone CPU proof of concept: separate two speech sources, select the user by voice profile.

This does not import or modify Adam's runtime. It uses SpeechBrain's Apache-2.0
SepFormer Libri2Mix checkpoint and Adam's existing ECAPA profile only to rank
the two separated outputs.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import soxr
import torch
from speechbrain.inference.separation import SepformerSeparation

from src.stt.speaker import SpeakerVerifier


ROOT = Path(__file__).resolve().parent
INPUT_DIR = Path.home() / ".local/state/adam/heard-captures"
PROFILE_PATH = Path.home() / ".local/state/adam/speaker-profile.npz"
MODEL_DIR = ROOT / "speechbrain-model"
OUTPUT_DIR = ROOT / "outputs"
SAMPLE_RATE = 16000
SEPARATION_RATE = 8000


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError(f"Expected mono PCM16 WAV: {path}")
        rate = wav_file.getframerate()
        audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0, rate


def write_wav(path: Path, audio: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm.tobytes())


def main() -> None:
    torch.set_num_threads(4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-libri2mix",
        savedir=str(MODEL_DIR),
        run_opts={"device": "cpu"},
    )
    verifier = SpeakerVerifier(profile_path=PROFILE_PATH, threshold=0.25)
    if not verifier.enrolled:
        raise RuntimeError(f"No enrolled voice profile found at {PROFILE_PATH}")

    report = []
    torch.set_grad_enabled(False)
    for mixture_path in sorted(INPUT_DIR.glob("*.wav")):
        mixture, rate = read_wav(mixture_path)
        if rate != SAMPLE_RATE:
            mixture = soxr.resample(mixture, rate, SAMPLE_RATE, quality="HQ")
        mixture_8k = soxr.resample(mixture, SAMPLE_RATE, SEPARATION_RATE, quality="HQ")
        separated = separator.separate_batch(torch.from_numpy(mixture_8k).unsqueeze(0))
        source_tracks = [
            soxr.resample(separated[0, :, i].detach().cpu().numpy(), SEPARATION_RATE, SAMPLE_RATE, quality="HQ")
            for i in range(separated.shape[-1])
        ]

        scores = []
        for track in source_tracks:
            if len(track) < int(0.5 * SAMPLE_RATE):
                scores.append(None)
            else:
                _, score = verifier.verify(track)
                scores.append(float(score))
        eligible = [i for i, score in enumerate(scores) if score is not None]
        best_candidate = max(eligible, key=lambda i: scores[i]) if eligible else None
        selected = (
            best_candidate
            if best_candidate is not None and scores[best_candidate] >= verifier.threshold
            else None
        )

        basename = mixture_path.stem
        for i, track in enumerate(source_tracks):
            write_wav(OUTPUT_DIR / f"{basename}-source-{i}.wav", track)
        if selected is not None:
            write_wav(OUTPUT_DIR / f"{basename}-target.wav", source_tracks[selected])
            write_wav(
                OUTPUT_DIR / f"{basename}-other.wav",
                source_tracks[1 - selected] if len(source_tracks) == 2 else np.zeros_like(source_tracks[selected]),
            )

        report.append({
            "input": mixture_path.name,
            "duration_seconds": round(len(mixture) / SAMPLE_RATE, 3),
            "target_profile_scores": [None if score is None else round(score, 4) for score in scores],
            "best_candidate": best_candidate,
            "selected_source": selected,
            "output_target": f"{basename}-target.wav" if selected is not None else None,
        })
        print(json.dumps(report[-1]), flush=True)

    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
