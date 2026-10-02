import json
import stat
import time
import wave

import numpy as np

from src.audio.meeting import MeetingSession
from src.main import AdamDaemon, meeting_command_kind
from src.stt.diarizer import NemotronDiarizer, SpeakerSpan
from src.stt.meeting_speakers import MeetingSpeakerRegistry


class EnergyVAD:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def is_speech(self, chunk, **kwargs):
        return float(np.sqrt(np.mean(np.asarray(chunk) ** 2))) > 0.02, 0.9


def test_meeting_session_records_audio_and_writes_labeled_turns(tmp_path):
    processed = []
    meeting = MeetingSession(
        root=tmp_path,
        sample_rate=16000,
        silence_duration=0.2,
        process_segment=lambda *args: processed.append(args),
        vad_factory=EnergyVAD,
    )
    directory = meeting.start()
    silence = np.zeros(512, dtype=np.float32)
    voice = np.full(512, 0.1, dtype=np.float32)
    for chunk in [silence, silence, voice, voice, voice] + [silence] * 10:
        meeting.enqueue_audio(chunk)
    meeting.append_turn("You", "Please note this point.", 0.2, 0.5)
    meeting.stop("test")

    with wave.open(str(directory / "meeting.wav"), "rb") as audio_file:
        assert audio_file.getframerate() == 16000
        assert audio_file.getnchannels() == 1
        assert audio_file.getnframes() == 15 * 512
    assert len(processed) == 1
    assert len(processed[0][0]) >= 3 * 512
    assert "You: Please note this point." in (directory / "transcript.txt").read_text()
    turn = json.loads((directory / "turns.jsonl").read_text().splitlines()[0])
    assert turn["speaker"] == "You"
    assert turn["text"] == "Please note this point."
    assert stat.S_IMODE((directory / "meeting.wav").stat().st_mode) == 0o600
    assert json.loads((directory / "session.json").read_text())["stop_reason"] == "test"


def test_meeting_vad_segments_speech_while_audio_is_still_recorded(tmp_path):
    processed = []
    meeting = MeetingSession(
        root=tmp_path,
        sample_rate=16000,
        silence_duration=0.2,
        process_segment=lambda audio, context, start, end: processed.append(audio),
        vad_factory=EnergyVAD,
    )
    meeting.start()
    for chunk in [np.full(512, 0.1, np.float32)] * 4 + [np.zeros(512, np.float32)] * 10:
        meeting.enqueue_audio(chunk)
    meeting.stop("test")
    assert len(processed) == 1


def test_meeting_stop_releases_transcriber_when_vad_initialization_fails(tmp_path):
    class BrokenVAD:
        def __init__(self, sample_rate=16000):
            raise RuntimeError("VAD model unavailable")

    meeting = MeetingSession(
        root=tmp_path,
        sample_rate=16000,
        process_segment=lambda *_: None,
        vad_factory=BrokenVAD,
    )
    meeting.start()
    meeting.enqueue_audio(np.ones(512, dtype=np.float32))
    meeting.stop("test")

    assert meeting._segment_thread is not None
    assert not meeting._segment_thread.is_alive()


def test_meeting_command_phrases():
    assert meeting_command_kind("meeting mode") == "start"
    assert meeting_command_kind("start meeting mode") == "start"
    assert meeting_command_kind("meeting over") == "stop"
    assert meeting_command_kind("could you stop meeting mode") == "stop"
    assert meeting_command_kind("what happened in the meeting?") is None


def test_meeting_speaker_registry_reuses_anonymous_labels():
    class Encoder:
        enrolled = False

        def embedding(self, audio):
            return np.asarray(audio[:2], dtype=np.float32)

    registry = MeetingSpeakerRegistry(Encoder(), similarity_threshold=0.8)
    first, _ = registry.label(np.array([1.0, 0.0], dtype=np.float32))
    same, score = registry.label(np.array([0.98, 0.02], dtype=np.float32))
    other, _ = registry.label(np.array([0.0, 1.0], dtype=np.float32))
    assert first == same == "Speaker 1"
    assert score > 0.8
    assert other == "Speaker 2"


def test_meeting_speaker_registry_marks_enrolled_voice_as_you():
    class Encoder:
        def embedding(self, _audio):
            raise AssertionError("the enrolled voice should not need a new cluster")

    class Verifier:
        enrolled = True

        def verify(self, _audio):
            return True, 0.42

    assert MeetingSpeakerRegistry(Encoder(), Verifier()).label(np.ones(16000, np.float32)) == ("You", 0.42)


def test_meeting_transcribes_original_mixed_turns_with_speaker_labels():
    audio = np.linspace(-0.8, 0.8, 32000, dtype=np.float32)
    turns = []

    class Session:
        session_dir = object()

        def append_turn(self, speaker, text, start, end):
            turns.append((speaker, text, start, end))

    class Diarizer:
        def diarize(self, _audio):
            return [SpeakerSpan("a", 0.0, 1.2), SpeakerSpan("b", 0.6, 1.8)]

        speaker_turns = staticmethod(NemotronDiarizer.speaker_turns)

    class Registry:
        def label(self, samples):
            return ("Speaker 1" if np.mean(samples) < 0 else "Speaker 2", None)

    daemon = AdamDaemon.__new__(AdamDaemon)
    daemon.meeting_session = Session()
    daemon.meeting_speaker_registry = Registry()
    daemon.speaker_diarizer = Diarizer()
    daemon.stream = type("Stream", (), {"sample_rate": 16000})()
    captured_audio = []

    def transcribe(samples, *, kind="final"):
        assert kind == "meeting"
        captured_audio.append(samples.copy())
        return f"turn {len(captured_audio)}"

    daemon._transcribe_stt = transcribe
    daemon._process_meeting_segment(audio, np.zeros(0, np.float32), 4.0, 6.0)

    assert [turn[0] for turn in turns] == ["Speaker 1", "Speaker 1 + Speaker 2", "Speaker 2"]
    assert [turn[1] for turn in turns] == ["turn 1", "turn 2", "turn 3"]
    assert turns[0][2:] == (4.0, 4.6)
    assert turns[1][2:] == (4.6, 5.2)
    assert turns[2][2:] == (5.2, 5.8)
    np.testing.assert_array_equal(captured_audio[0], audio[:9600])
    np.testing.assert_array_equal(captured_audio[1], audio[9600:19200])
    np.testing.assert_array_equal(captured_audio[2], audio[19200:28800])
