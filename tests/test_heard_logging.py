import json
import stat
import types
import wave

import numpy as np

from src.main import AdamDaemon


def test_heard_capture_saves_private_audio_and_transcript_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    daemon = AdamDaemon.__new__(AdamDaemon)
    daemon.stream = types.SimpleNamespace(sample_rate=16000)

    capture_id = daemon._save_heard_capture(np.zeros(16000, dtype=np.float32))
    assert capture_id
    audio_path = tmp_path / "adam" / "heard-captures" / f"heard-{capture_id}.wav"
    metadata_path = audio_path.with_suffix(".json")

    with wave.open(str(audio_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 16000
    assert stat.S_IMODE(audio_path.stat().st_mode) == 0o600

    daemon._log_heard_transcript(capture_id, "Hey Adam, what time is it?", "mixed-audio")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["duration_seconds"] == 1.0
    assert metadata["transcripts"][0]["text"] == "Hey Adam, what time is it?"
    assert stat.S_IMODE(metadata_path.stat().st_mode) == 0o600
