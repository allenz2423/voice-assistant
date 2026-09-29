"""Optional adapter for NVIDIA's local NeMo-Speech.cpp diarization CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SpeakerSpan:
    speaker: str
    start: float
    end: float


class NemotronDiarizer:
    """Diarize a completed mono 16 kHz utterance using nemo-speech."""

    def __init__(self, executable: str = "nemo-speech", model: str = "nvidia/Nemotron-3-Diarization",
                 device: str = "vulkan:0", timeout: float = 45.0):
        self.executable = executable
        self.model = model
        self.device = device
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def supports_model(self) -> bool:
        """Check whether this installed CLI's pinned model index contains the checkpoint."""
        if not self.available:
            return False
        if Path(self.model).is_file():
            return True
        try:
            result = subprocess.run(
                [self.executable, "--json", "model", "list"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            payload = json.loads(result.stdout)
            return result.returncode == 0 and any(
                entry.get("repo", "").lower() == self.model.lower()
                for entry in payload.get("models", [])
            )
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            return False

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray) -> None:
        pcm = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm16.tobytes())

    @staticmethod
    def parse_rttm(contents: str) -> list[SpeakerSpan]:
        spans: list[SpeakerSpan] = []
        for line in contents.splitlines():
            fields = line.split()
            if not fields or fields[0] != "SPEAKER" or len(fields) < 8:
                continue
            try:
                start, duration = float(fields[3]), float(fields[4])
                spans.append(SpeakerSpan(fields[7], start, start + duration))
            except (ValueError, IndexError):
                continue
        return spans

    def diarize(self, audio: np.ndarray) -> list[SpeakerSpan]:
        if not self.available:
            raise FileNotFoundError(
                f"{self.executable!r} not found. Install NVIDIA NeMo-Speech.cpp and enable its Vulkan backend."
            )
        with tempfile.TemporaryDirectory(prefix="adam-diarize-") as temp_dir:
            wav_path = Path(temp_dir) / "utterance.wav"
            rttm_path = Path(temp_dir) / "utterance.rttm"
            self._write_wav(wav_path, audio)
            command = [
                self.executable, "diarize", str(wav_path),
                "--model", self.model,
                "--device", self.device,
                "--format", "rttm",
                "--output", str(rttm_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout, check=False)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(f"Nemotron diarization failed ({result.returncode}): {detail[-1200:]}")
            if not rttm_path.exists():
                raise RuntimeError("nemo-speech completed without creating the requested RTTM file")
            return self.parse_rttm(rttm_path.read_text(encoding="utf-8"))

    @staticmethod
    def exclusive_speaker_audio(audio: np.ndarray, spans: list[SpeakerSpan], sample_rate: int,
                                min_seconds: float = 0.5) -> dict[str, np.ndarray]:
        """Return audio where each speaker is active alone; overlapping frames are omitted."""
        source = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            raise ValueError("Nemotron diarization expects 16 kHz audio")
        frame_count = len(source)
        activity: dict[str, np.ndarray] = {}
        for span in spans:
            start = max(0, min(frame_count, int(span.start * sample_rate)))
            end = max(start, min(frame_count, int(span.end * sample_rate)))
            if end > start:
                activity.setdefault(span.speaker, np.zeros(frame_count, dtype=bool))[start:end] = True
        if not activity:
            return {}
        active_count = np.sum(np.stack(list(activity.values())), axis=0)
        result: dict[str, np.ndarray] = {}
        minimum = int(min_seconds * sample_rate)
        for speaker, mask in activity.items():
            selected = source[mask & (active_count == 1)]
            if len(selected) >= minimum:
                result[speaker] = selected
        return result
