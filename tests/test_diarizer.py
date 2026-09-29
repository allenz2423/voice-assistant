import os

import numpy as np

from src.stt.diarizer import NemotronDiarizer, SpeakerSpan


def test_parse_rttm_speaker_spans():
    spans = NemotronDiarizer.parse_rttm(
        "SPEAKER utt 1 0.125 0.500 <NA> <NA> speaker_0 <NA> <NA>\n"
        "SPEAKER utt 1 0.750 0.250 <NA> <NA> speaker_1 <NA> <NA>\n"
    )
    assert spans == [SpeakerSpan("speaker_0", 0.125, 0.625), SpeakerSpan("speaker_1", 0.75, 1.0)]


def test_exclusive_speaker_audio_drops_overlapping_frames():
    audio = np.arange(20000, dtype=np.float32)
    spans = [SpeakerSpan("a", 0.0, 0.75), SpeakerSpan("b", 0.5, 1.25)]

    exclusive = NemotronDiarizer.exclusive_speaker_audio(audio, spans, sample_rate=16000)

    assert set(exclusive) == {"a", "b"}
    np.testing.assert_array_equal(exclusive["a"], audio[:8000])
    np.testing.assert_array_equal(exclusive["b"], audio[12000:20000])


def test_diarize_invokes_cli_and_reads_rttm(tmp_path, monkeypatch):
    executable = tmp_path / "nemo-speech"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.write_text('SPEAKER utt 1 0.0 1.0 <NA> <NA> speaker_0 <NA> <NA>\\n')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    diarizer = NemotronDiarizer()
    spans = diarizer.diarize(np.zeros(16000, dtype=np.float32))

    assert spans == [SpeakerSpan("speaker_0", 0.0, 1.0)]
