import asyncio
import json
import stat
import time
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from src.audio.meeting import MeetingSession
from src.llm.tools import ADAM_TOOLS
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
    assert meeting_command_kind("meeting mode on") == "start"
    assert meeting_command_kind("MEETING MODE, ON!!!") == "start"
    assert meeting_command_kind("meeting-mode-on") == "start"
    assert meeting_command_kind("meeting mode off") == "stop"
    assert meeting_command_kind("Meeting Mode OFF.") == "stop"
    assert meeting_command_kind("meeting.mode.off?") == "stop"
    for wording in (
        "meeting mode",
        "start meeting",
        "start meeting mode",
        "begin meeting mode",
        "meeting over",
        "stop meeting mode",
        "could you stop meeting mode",
        "please meeting mode on",
        "what happened in the meeting?",
    ):
        assert meeting_command_kind(wording) is None


def test_meeting_mode_tool_has_structured_action_schema():
    tool = next(tool for tool in ADAM_TOOLS if tool.name == "meeting_mode")

    assert tool.parameters["required"] == ["action"]
    assert tool.parameters["properties"]["action"]["enum"] == ["start", "stop"]


def test_natural_meeting_control_requests_keep_tools_available():
    from src.llm.brain import _can_answer_without_tools, _is_meeting_mode_control_request

    for phrase in (
        "meeting mode",
        "MEETING MODE!",
        "Start meeting mode.",
        "Could you please turn on meeting mode?",
        "Would you start the meeting recording?",
        "Meeting's over; stop recording.",
        "Please begin transcription for this meeting.",
    ):
        assert _is_meeting_mode_control_request(phrase)
        assert not _can_answer_without_tools(phrase)

    for phrase in (
        "How do I start meeting mode?",
        "Don't start meeting recording.",
        "What happened in the meeting?",
    ):
        assert not _is_meeting_mode_control_request(phrase)


def test_pretrained_wake_text_dispatches_meeting_and_falls_back_to_brain():
    async def scenario():
        class Arbiter:
            def __init__(self):
                self.current_state = "USER_SPEAKING"

            async def set_state(self, state):
                self.current_state = state

        class Stream:
            def __init__(self):
                self.flush_calls = 0
                self.quench_durations = []

            def flush(self):
                self.flush_calls += 1

            def quench(self, duration):
                self.quench_durations.append(duration)

        daemon = AdamDaemon.__new__(AdamDaemon)
        daemon.meeting_session = SimpleNamespace(active=False)
        daemon._start_meeting = AsyncMock()
        daemon._stop_meeting = AsyncMock()
        daemon.arbiter = Arbiter()
        daemon.stream = Stream()
        daemon.memory_manager = SimpleNamespace(
            retrieve_context=lambda command: f"context for {command}"
        )
        daemon._execute_turn = AsyncMock()

        await daemon._execute_pretrained_wake_command("meeting mode on")

        daemon._start_meeting.assert_awaited_once_with()
        daemon._stop_meeting.assert_not_awaited()
        daemon._execute_turn.assert_not_awaited()
        assert daemon.arbiter.current_state == "IDLE_LISTENING"
        assert daemon.stream.flush_calls == 1
        assert daemon.stream.quench_durations == [0.4]

        daemon.meeting_session.active = True
        daemon.arbiter.current_state = "USER_SPEAKING"
        await daemon._execute_pretrained_wake_command("meeting mode off")

        daemon._stop_meeting.assert_awaited_once_with("voice command")
        daemon._execute_turn.assert_not_awaited()
        assert daemon.arbiter.current_state == "IDLE_LISTENING"
        assert daemon.stream.flush_calls == 2
        assert daemon.stream.quench_durations == [0.4, 0.4]

        daemon._start_meeting = AsyncMock(side_effect=RuntimeError("recording setup failed"))
        daemon.arbiter.current_state = "USER_SPEAKING"
        try:
            await daemon._execute_pretrained_wake_command("meeting mode on")
        except RuntimeError as exc:
            assert str(exc) == "recording setup failed"
        else:
            raise AssertionError("meeting start error should propagate")
        assert daemon.arbiter.current_state == "IDLE_LISTENING"
        assert daemon.stream.flush_calls == 3
        assert daemon.stream.quench_durations == [0.4, 0.4, 0.4]

        daemon.tts = SimpleNamespace(speak_async=AsyncMock())
        daemon.meeting_session.active = False
        daemon.meeting_session.session_dir = "/previous/meeting"
        await daemon._execute_pretrained_wake_command("meeting mode off")
        daemon.tts.speak_async.assert_awaited_once_with("Meeting mode is already off.")
        daemon._execute_turn.assert_not_awaited()

        await daemon._execute_pretrained_wake_command("start meeting mode")

        daemon._execute_turn.assert_awaited_once_with(
            "start meeting mode", memory_context="context for start meeting mode"
        )

    asyncio.run(scenario())


def test_execute_turn_dispatches_meeting_command_before_brain():
    async def scenario():
        class Arbiter:
            def __init__(self):
                self.current_state = "IDLE_LISTENING"
                self.state_history = []

            async def set_state(self, state):
                self.current_state = state
                self.state_history.append(state)

        class Stream:
            def __init__(self):
                self.flush_calls = 0
                self.quench_durations = []

            def flush(self):
                self.flush_calls += 1

            def quench(self, duration):
                self.quench_durations.append(duration)

        daemon = AdamDaemon.__new__(AdamDaemon)
        daemon.meeting_session = SimpleNamespace(active=False)
        daemon._start_meeting = AsyncMock()
        daemon._stop_meeting = AsyncMock()
        daemon.arbiter = Arbiter()
        daemon.stream = Stream()
        daemon.running = True
        daemon.config = SimpleNamespace(wake=SimpleNamespace(followup_window_seconds=7.0))
        speculative_cancellations = []
        daemon.speculative_router = SimpleNamespace(
            cancel_active=lambda: speculative_cancellations.append(True)
        )
        daemon._monitor_execution_interrupt = AsyncMock(return_value=(False, None))
        daemon.brain = SimpleNamespace(process_user_utterance=AsyncMock())

        response = await daemon._execute_turn("meeting mode on")

        assert response == "Meeting mode is on. Recording now."
        daemon._start_meeting.assert_awaited_once_with()
        daemon.brain.process_user_utterance.assert_not_awaited()
        assert daemon.arbiter.current_state == "IDLE_LISTENING"
        assert daemon.arbiter.state_history == ["PROCESSING_REACT", "IDLE_LISTENING"]
        assert daemon.stream.flush_calls == 1
        assert daemon.stream.quench_durations == [0.4]
        assert speculative_cancellations == [True]

        daemon.tts = SimpleNamespace(speak_async=AsyncMock())
        response = await daemon._execute_turn("meeting mode off")
        assert response == "Meeting mode is already off."
        daemon.tts.speak_async.assert_awaited_once_with("Meeting mode is already off.")
        daemon.brain.process_user_utterance.assert_not_awaited()
        assert daemon.stream.flush_calls == 2
        assert speculative_cancellations == [True, True]

    asyncio.run(scenario())


def test_interrupt_requeue_dispatches_meeting_command_before_brain():
    async def scenario():
        class Arbiter:
            current_state = "IDLE_LISTENING"

            async def set_state(self, state):
                self.current_state = state

        class Stream:
            def __init__(self):
                self.flush_calls = 0
                self.quench_durations = []

            def flush(self):
                self.flush_calls += 1

            def quench(self, duration):
                self.quench_durations.append(duration)

        daemon = AdamDaemon.__new__(AdamDaemon)
        daemon.meeting_session = SimpleNamespace(active=False)
        daemon._start_meeting = AsyncMock()
        daemon._stop_meeting = AsyncMock()
        daemon.arbiter = Arbiter()
        daemon.stream = Stream()
        daemon.running = True
        daemon.config = SimpleNamespace(wake=SimpleNamespace(followup_window_seconds=7.0))
        daemon.speculative_router = SimpleNamespace(cancel_active=lambda: None)
        daemon.earcon = SimpleNamespace(play=lambda _name: None)
        daemon.wake = SimpleNamespace(match_custom_wake_word=lambda _text: (False, ""))
        daemon._monitor_execution_interrupt = AsyncMock(return_value=(True, "meeting mode on"))
        daemon.brain = SimpleNamespace(process_user_utterance=AsyncMock())

        await daemon._execute_turn("check the weather")

        daemon._start_meeting.assert_awaited_once_with()
        daemon.brain.process_user_utterance.assert_awaited_once_with(
            "check the weather", memory_context=None
        )
        assert daemon.arbiter.current_state == "IDLE_LISTENING"
        assert daemon.stream.flush_calls == 1
        assert daemon.stream.quench_durations == [0.4]

    asyncio.run(scenario())


def test_nonexact_meeting_wording_reaches_brain():
    async def scenario():
        class Arbiter:
            current_state = "IDLE_LISTENING"

            async def set_state(self, state):
                self.current_state = state

        class Stream:
            def flush(self):
                pass

            def quench(self, duration):
                pass

        daemon = AdamDaemon.__new__(AdamDaemon)
        daemon.meeting_session = SimpleNamespace(active=False)
        daemon._start_meeting = AsyncMock()
        daemon._stop_meeting = AsyncMock()
        daemon.arbiter = Arbiter()
        daemon.stream = Stream()
        daemon.running = True
        daemon.config = SimpleNamespace(wake=SimpleNamespace(followup_window_seconds=7.0))
        daemon.speculative_router = SimpleNamespace(cancel_active=lambda: None)
        daemon._monitor_execution_interrupt = AsyncMock(return_value=(False, None))
        daemon.brain = SimpleNamespace(process_user_utterance=AsyncMock())

        await daemon._execute_turn("start meeting mode")

        daemon._start_meeting.assert_not_awaited()
        daemon._stop_meeting.assert_not_awaited()
        daemon.brain.process_user_utterance.assert_awaited_once_with(
            "start meeting mode", memory_context=None
        )

    asyncio.run(scenario())


def test_meeting_mode_tool_handler_uses_daemon_lifecycle():
    async def scenario():
        daemon = AdamDaemon.__new__(AdamDaemon)
        daemon.meeting_session = SimpleNamespace(active=False, session_dir=None)
        daemon._start_meeting = AsyncMock()
        daemon._stop_meeting = AsyncMock()
        daemon.tts = SimpleNamespace(speak_async=AsyncMock())

        result = await daemon._execute_meeting_mode_tool("start")

        assert result == "Meeting mode is on. Recording now."
        daemon._start_meeting.assert_awaited_once_with()

        daemon.meeting_session.active = True
        daemon.meeting_session.session_dir = "/tmp/meeting-current"
        result = await daemon._execute_meeting_mode_tool("stop")
        assert result == "Meeting mode is off. I saved the recording and transcript in /tmp/meeting-current."
        daemon._stop_meeting.assert_awaited_once_with("voice command")

        daemon.meeting_session.active = False
        result = await daemon._execute_meeting_mode_tool("stop")
        assert result == "Meeting mode is already off."
        daemon.tts.speak_async.assert_awaited_once_with("Meeting mode is already off.")

    asyncio.run(scenario())


def test_natural_meeting_request_dispatches_structured_tool_once():
    async def scenario():
        from src.llm.brain import AdamBrain

        class TTS:
            engine = "silent"
            pending_barge_in_text = None

            def __init__(self):
                self.spoken = []

            async def speak_async(self, text):
                self.spoken.append(text)

        class Model:
            provider = "custom"

            def __init__(self):
                self.requests = []

            async def chat(self, messages, tools=None, **_kwargs):
                self.requests.append((messages, tools))
                return {"content": "", "tool_calls": [{
                    "id": "meeting-start",
                    "function": {"name": "meeting_mode", "arguments": {"action": "start"}},
                }]}

            def format_tool_response(self, tool_call_id, tool_name, result):
                return {
                    "role": "tool", "tool_call_id": tool_call_id,
                    "name": tool_name, "content": result,
                }

        config = SimpleNamespace(llm=SimpleNamespace(
            provider="custom", local_model="test", cloud_model="test",
            ollama_host="http://127.0.0.1:11434", api_base="https://example.invalid/v1",
            api_key="", temperature=0, num_ctx=8192,
        ))
        tts = TTS()
        brain = AdamBrain(
            config, None, None, None, tts,
            memory_mgr=SimpleNamespace(retrieve_context=lambda _text: None),
        )
        model = Model()
        brain.llm_client = model

        async def meeting_handler(action):
            assert action == "start"
            await tts.speak_async("Meeting mode is on. Recording now.")
            return "Meeting mode is on. Recording now."

        brain.meeting_mode_handler = meeting_handler

        from unittest.mock import patch
        with patch("src.llm.brain.get_open_windows_prompt_context", return_value=""):
            await brain.process_user_utterance("meeting mode")

        assert len(model.requests) == 1
        request_messages, advertised_tools = model.requests[0]
        assert "meeting_mode" in [tool.name for tool in advertised_tools]
        assert "A bare command saying “meeting mode” means start meeting mode." in request_messages[0]["content"]
        assert tts.spoken == ["Meeting mode is on. Recording now."]
        result = next(message for message in brain.messages if message.get("role") == "tool")
        assert "Meeting mode is on. Recording now." in result["content"]
        assistant_reply = next(
            message for message in reversed(brain.messages)
            if message.get("role") == "assistant" and message.get("content")
        )
        assert assistant_reply["content"] == "Meeting mode is on. Recording now."

    asyncio.run(scenario())


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
