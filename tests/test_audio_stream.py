import threading
from collections import deque

import numpy as np

from src.audio.stream import AudioStreamManager


def test_recent_audio_keeps_five_second_tail_and_can_exclude_current_chunk():
    stream = AudioStreamManager.__new__(AudioStreamManager)
    stream.sample_rate = 4
    stream._recent_audio_seconds = 2.0
    stream._recent_audio_chunks = deque()
    stream._recent_audio_samples = 0
    stream._recent_audio_lock = threading.Lock()

    stream._remember_recent_audio(np.array([0, 1, 2, 3], dtype=np.float32))
    stream._remember_recent_audio(np.array([4, 5, 6, 7], dtype=np.float32))
    stream._remember_recent_audio(np.array([8, 9], dtype=np.float32))

    np.testing.assert_array_equal(
        stream.get_recent_audio(duration_s=1.0, exclude_latest_samples=2),
        np.array([4, 5, 6, 7], dtype=np.float32),
    )
