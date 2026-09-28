import asyncio
import os
import stat

import numpy as np
import torch

from src.stt.speaker import SpeakerVerifier


class FakeEncoder:
    def encode_batch(self, signal):
        # Positive-valued samples represent the enrolled speaker; negative-valued
        # samples represent a different speaker for this deterministic unit test.
        if signal.mean().item() >= 0:
            embedding = [1.0, 0.0]
        else:
            embedding = [0.0, 1.0]
        return torch.tensor(embedding, dtype=torch.float32).reshape(1, 1, 2)


def test_speaker_profile_enrollment_is_local_and_matches_only_enrolled_speaker(tmp_path):
    profile_path = tmp_path / "speaker-profile.npz"
    verifier = SpeakerVerifier(profile_path=profile_path, classifier=FakeEncoder())
    verifier.enroll([
        np.ones(16000, dtype=np.float32),
        np.full(16000, 0.5, dtype=np.float32),
    ])

    enrolled = SpeakerVerifier(profile_path=profile_path, classifier=FakeEncoder())
    assert enrolled.enrolled
    assert enrolled.verify(np.ones(16000, dtype=np.float32)) == (True, 1.0)
    matched, score = enrolled.verify(np.full(16000, -1.0, dtype=np.float32))
    assert matched is False
    assert score == 0.0
    assert stat.S_IMODE(os.stat(profile_path).st_mode) == 0o600


def test_short_audio_fails_closed(tmp_path):
    verifier = SpeakerVerifier(profile_path=tmp_path / "speaker-profile.npz", classifier=FakeEncoder())
    verifier.enroll([np.ones(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)])
    assert verifier.verify(np.ones(1000, dtype=np.float32)) == (False, -1.0)


def test_daemon_speaker_gate_blocks_nonmatching_audio_without_profile_setup():
    from src.main import AdamDaemon

    class RejectingVerifier:
        def verify(self, audio):
            return False, 0.1

    daemon = AdamDaemon.__new__(AdamDaemon)
    daemon.speaker_verifier = RejectingVerifier()
    assert asyncio.run(daemon._speaker_allowed(np.ones(16000, dtype=np.float32))) is False
