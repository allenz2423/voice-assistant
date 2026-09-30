"""Local, opt-in speaker verification for the command audio captured by Adam.

Each named user's voice embeddings are persisted as separate examples; the
enrollment audio itself stays in RAM and is discarded after enrollment.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path

import numpy as np


MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
MIN_VERIFY_SECONDS = 0.5


def default_profile_path() -> Path:
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "adam" / "speaker-profile.npz"


def default_user_profile_path(name: str, profile_name: str = "default") -> Path:
    user_slug = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    profile_slug = re.sub(r"[^a-z0-9]+", "-", str(profile_name).strip().lower()).strip("-")
    if not user_slug or not profile_slug:
        raise ValueError("Speaker user name must contain letters or numbers.")
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    root = state_home / "adam" / "speaker-profiles"
    if profile_slug == "default":
        return root / f"{user_slug}.npz"
    return root / user_slug / f"{profile_slug}.npz"


class SpeakerVerifier:
    """Identify configured users from locally enrolled voice examples."""

    def __init__(
        self,
        profile_path: str | Path | None = None,
        threshold: float = 0.25,
        classifier=None,
        user_profiles: dict[str, str | Path | dict[str, str | Path]] | None = None,
    ):
        self.profile_path = Path(profile_path).expanduser() if profile_path else default_profile_path()
        self.threshold = threshold
        self.classifier = classifier
        self._classifier_lock = threading.Lock()
        configured_profiles = user_profiles if user_profiles is not None else {"You": self.profile_path}
        self.user_profile_paths: dict[str, dict[str, Path]] = {}
        self.profiles: dict[str, dict[str, np.ndarray]] = {}
        for name, paths in configured_profiles.items():
            profile_paths = paths if isinstance(paths, dict) else {"default": paths}
            self.user_profile_paths[name] = {
                profile_name: Path(path).expanduser()
                for profile_name, path in profile_paths.items()
            }
            loaded = {
                profile_name: profile
                for profile_name, path in self.user_profile_paths[name].items()
                if (profile := self._load_profile(path)) is not None
            }
            if loaded:
                self.profiles[name] = loaded

    @property
    def enrolled(self) -> bool:
        return bool(self.profiles)

    @property
    def profile(self) -> np.ndarray | None:
        """Compatibility access to the default/first profile."""
        user_profiles = self.profiles.get("You", next(iter(self.profiles.values()), {}))
        return user_profiles.get("default", next(iter(user_profiles.values()), None))

    def _load_profile(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["model_id"].item()) != MODEL_ID:
                    raise ValueError("Speaker profile was created by a different model.")
                # Keep old single-centroid profiles readable.
                key = "embeddings" if "embeddings" in data else "embedding"
                embeddings = np.asarray(data[key], dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
                raise ValueError("Speaker profile embedding is invalid.")
            if not np.all(np.isfinite(embeddings)):
                raise ValueError("Speaker profile embedding is invalid.")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if np.any(norms < 1e-8):
                raise ValueError("Speaker profile contains an empty embedding.")
            return embeddings / norms
        except Exception as exc:
            raise RuntimeError(f"Could not load speaker profile at {path}: {exc}") from exc

    def _get_classifier(self):
        if self.classifier is None:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ImportError:
                from speechbrain.inference.speaker import EncoderClassifier

            model_dir = default_profile_path().parent / "speaker-model"
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
        with self._classifier_lock:
            encoded = self._get_classifier().encode_batch(signal).detach().cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(encoded)) or np.linalg.norm(encoded) < 1e-8:
            raise RuntimeError("Speaker model returned an invalid voice embedding.")
        return encoded.astype(np.float32) / np.linalg.norm(encoded)

    def identify(self, audio: np.ndarray) -> tuple[str | None, float]:
        """Return the best configured user above threshold and their cosine score."""
        if not self.enrolled:
            raise RuntimeError("No speaker profile is enrolled.")
        try:
            candidate = self.embedding(audio)
        except ValueError:
            return None, -1.0
        best_user = None
        best_score = float("-inf")
        for name, profiles in self.profiles.items():
            # A max over a user's separate distance profiles and examples finds
            # that user without blending different users into one voice.
            score = max(float(np.max(examples @ candidate)) for examples in profiles.values())
            if score > best_score:
                best_user, best_score = name, score
        return (best_user if best_score >= self.threshold else None), best_score

    def verify(self, audio: np.ndarray) -> tuple[bool, float]:
        """Return (accepted, best cosine similarity) for any configured user."""
        user, score = self.identify(audio)
        return user is not None, score

    def enroll(
        self,
        samples: list[np.ndarray],
        *,
        append: bool = False,
        user_name: str = "You",
        profile_name: str = "default",
    ) -> None:
        if len(samples) < 2:
            raise ValueError("Enroll with at least two separate speech samples.")
        matched_name = next(
            (name for name in self.user_profile_paths if name.casefold() == user_name.casefold()),
            None,
        )
        if matched_name is None:
            matched_name = user_name.strip()
            self.user_profile_paths[matched_name] = default_user_profile_path(matched_name)
        user_name = matched_name
        embeddings = np.stack([self.embedding(sample) for sample in samples])
        profile_name = profile_name.strip()
        if not profile_name:
            raise ValueError("Speaker profile name cannot be empty.")
        user_paths = self.user_profile_paths.setdefault(user_name, {})
        profile_path = user_paths.setdefault(
            profile_name,
            default_user_profile_path(user_name, profile_name),
        )
        existing = self.profiles.get(user_name, {}).get(profile_name)
        if append and existing is not None:
            if existing.shape[1] != embeddings.shape[1]:
                raise RuntimeError("New speaker embeddings do not match the existing profile dimensions.")
            profile = np.concatenate((existing, embeddings), axis=0)
        else:
            profile = embeddings

        profile_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".npz", prefix=".speaker-profile-",
                dir=profile_path.parent, delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                np.savez_compressed(temp_file, model_id=np.array(MODEL_ID), embeddings=profile)
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, profile_path)
            os.chmod(profile_path, 0o600)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()
        self.profiles.setdefault(user_name, {})[profile_name] = profile.astype(np.float32)
