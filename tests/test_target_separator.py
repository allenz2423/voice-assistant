import numpy as np
import torch

from src.stt.target_separator import TargetSpeakerSeparator


class FakeSeparator:
    def separate_batch(self, mixture):
        length = mixture.shape[1]
        return torch.stack(
            [torch.full((1, length), 0.1), torch.full((1, length), 4.0)], dim=-1
        )


class FakeVerifier:
    enrolled = True
    threshold = 0.25

    def verify(self, audio):
        score = 0.1 if float(np.mean(audio)) < 0.5 else 0.4
        return score >= self.threshold, score


def test_target_separator_selects_profile_match_and_scales_without_clipping():
    separator = TargetSpeakerSeparator(FakeVerifier())
    separator._separator = FakeSeparator()

    result = separator.isolate(np.zeros(16000, dtype=np.float32))

    assert result is not None
    assert result.source_index == 1
    assert result.score == 0.4
    assert np.max(np.abs(result.audio)) <= 0.951


def test_target_separator_rejects_sources_below_profile_threshold():
    class RejectVerifier:
        enrolled = True
        threshold = 0.75

        def verify(self, _audio):
            return False, 0.4

    separator = TargetSpeakerSeparator(RejectVerifier())
    separator._separator = FakeSeparator()
    assert separator.isolate(np.zeros(16000, dtype=np.float32)) is None
