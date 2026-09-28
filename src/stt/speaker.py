"""Local, opt-in speaker verification for the command audio captured by Shin.

Only the averaged speaker embedding is persisted; enrollment audio stays in RAM.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
MIN_VERIFY_SECONDS = 0.5


def default_profile_path() -> Path:
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "shin" / "speaker-profile.npz"


class SpeakerVerifier:
    """Compare utterances against a locally enrolled voice embedding."""

    def __init__(
        self,
        profile_path: str | Path | None = None,
        threshold: float = 0.25,
        classifier=None,
    ):
        self.profile_path = Path(profile_path).expanduser() if profile_path else default_profile_path()
        self.threshold = threshold
        self.classifier = classifier
        self.profile = self._load_profile()

    @property
    def enrolled(self) -> bool:
        return self.profile is not None

    def _load_profile(self) -> np.ndarray | None:
        if not self.profile_path.exists():
            return None
        try:
            with np.load(self.profile_path, allow_pickle=False) as data:
                if str(data["model_id"].item()) != MODEL_ID:
                    raise ValueError("Speaker profile was created by a different model.")
                embedding = np.asarray(data["embedding"], dtype=np.float32).reshape(-1)
            if embedding.size == 0 or not np.all(np.isfinite(embedding)):
                raise ValueError("Speaker profile embedding is invalid.")
            return embedding / (np.linalg.norm(embedding) + 1e-12)
        except Exception as exc:
            raise RuntimeError(f"Could not load speaker profile at {self.profile_path}: {exc}") from exc

    def _get_classifier(self):
        if self.classifier is None:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ImportError:
                from speechbrain.inference.speaker import EncoderClassifier

            model_dir = self.profile_path.parent / "speaker-model"
            self.classifier = EncoderClassifier.from_hparams(
                source=MODEL_ID,
                savedir=str(model_dir),
                run_opts={"device": "cpu"},
            )
        return self.classifier

    def embedding(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < int(MIN_VERIFY_SECONDS * SAMPLE_RATE):
            raise ValueError("Need at least 0.5 seconds of speech to check the speaker.")
        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio contains invalid samples.")

        import torch

        signal = torch.from_numpy(audio).unsqueeze(0)
        encoded = self._get_classifier().encode_batch(signal).detach().cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(encoded)) or np.linalg.norm(encoded) < 1e-8:
            raise RuntimeError("Speaker model returned an invalid voice embedding.")
        return encoded.astype(np.float32) / np.linalg.norm(encoded)

    def verify(self, audio: np.ndarray) -> tuple[bool, float]:
        """Return (accepted, cosine similarity); short or failed checks fail closed."""
        if not self.enrolled:
            raise RuntimeError("No speaker profile is enrolled.")
        try:
            candidate = self.embedding(audio)
        except ValueError:
            return False, -1.0
        score = float(np.dot(self.profile, candidate))
        return score >= self.threshold, score

    def enroll(self, samples: list[np.ndarray]) -> None:
        if len(samples) < 2:
            raise ValueError("Enroll with at least two separate speech samples.")
        embeddings = np.stack([self.embedding(sample) for sample in samples])
        mean_embedding = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(mean_embedding))
        if norm < 1e-8:
            raise RuntimeError("Could not build a stable speaker profile from those samples.")
        profile = mean_embedding / norm

        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".npz", prefix=".speaker-profile-",
                dir=self.profile_path.parent, delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                np.savez_compressed(temp_file, model_id=np.array(MODEL_ID), embedding=profile)
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.profile_path)
            os.chmod(self.profile_path, 0o600)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()
        self.profile = profile.astype(np.float32)
