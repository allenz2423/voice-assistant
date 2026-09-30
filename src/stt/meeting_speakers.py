"""Stable anonymous meeting speaker labels backed by ECAPA embeddings."""

from __future__ import annotations

import threading

import numpy as np


class MeetingSpeakerRegistry:
    """Map local diarizer turns onto You and stable Speaker N labels."""

    def __init__(self, voice_encoder, enrolled_verifier=None, similarity_threshold: float = 0.55):
        self.voice_encoder = voice_encoder
        self.enrolled_verifier = enrolled_verifier
        self.similarity_threshold = float(similarity_threshold)
        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def label(self, audio: np.ndarray) -> tuple[str, float | None]:
        """Return a stable speaker label and the best relevant cosine score."""
        if self.enrolled_verifier is not None and self.enrolled_verifier.enrolled:
            try:
                if hasattr(self.enrolled_verifier, "identify"):
                    user_name, score = self.enrolled_verifier.identify(audio)
                    if user_name is not None:
                        return str(user_name), float(score)
                else:
                    matched, score = self.enrolled_verifier.verify(audio)
                    if matched:
                        return "You", float(score)
            except Exception:
                pass
        try:
            embedding = np.asarray(self.voice_encoder.embedding(audio), dtype=np.float32).reshape(-1)
            embedding /= np.linalg.norm(embedding) + 1e-12
        except Exception:
            return "Unknown", None

        with self._lock:
            best_label = None
            best_score = -1.0
            for label, centroid in self._centroids.items():
                score = float(np.dot(embedding, centroid))
                if score > best_score:
                    best_label, best_score = label, score
            if best_label is not None and best_score >= self.similarity_threshold:
                count = self._counts[best_label]
                updated = self._centroids[best_label] * count + embedding
                updated /= np.linalg.norm(updated) + 1e-12
                self._centroids[best_label] = updated.astype(np.float32)
                self._counts[best_label] = count + 1
                return best_label, best_score

            label = f"Speaker {len(self._centroids) + 1}"
            self._centroids[label] = embedding.copy()
            self._counts[label] = 1
            return label, None
