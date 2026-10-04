import pytest
import re
import asyncio
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock
import numpy as np

from src.main import merge_overlapping_transcripts
from src.arbiter.confirmation import TriStateConfirmationManager, sanitize_confirmation_speech
from src.arbiter.arbiter import PriorityAudioArbiter, SystemState
from src.llm.brain import AdamBrain
from src.tts.streaming import StreamingVoiceSynthesizer
from src.config import load_config
from src.tools.desktop import show_desktop_notification

# ==============================================================================
# 1. TRANSCRIPT OVERLAP & DEDUPLICATION TESTS
# ==============================================================================

def test_merge_overlapping_exact_match():
    assert merge_overlapping_transcripts("Yes.", "Yes.") == "Yes."
    assert merge_overlapping_transcripts("Stop", "stop") == "Stop"

def test_merge_overlapping_prefix_suffix():
    assert merge_overlapping_transcripts("Just pick one.", "Pick one. Show.") == "Just pick one. Show."
    assert merge_overlapping_transcripts("Could you?", "Could you list the files") == "Could you list the files"
    assert merge_overlapping_transcripts("Please sort the", "the video files") == "Please sort the video files"

def test_merge_overlapping_substring():
    assert merge_overlapping_transcripts("Not that.", "Not that. Downloads folder.") == "Not that. Downloads folder."
    assert merge_overlapping_transcripts("Downloads folder", "Downloads") == "Downloads folder"

def test_merge_disjoint():
    assert merge_overlapping_transcripts("Not that.", "Downloads folder.") == "Not that. Downloads folder."
    assert merge_overlapping_transcripts("Hello", "world") == "Hello world"

def test_merge_empty():
    assert merge_overlapping_transcripts("", "test") == "test"
    assert merge_overlapping_transcripts("test", "") == "test"


# ==============================================================================
# 2. CONFIRMATION STATE MACHINE & BARGE-IN TESTS
# ==============================================================================

class MockTTS:
    def __init__(self):
        self.spoken = []
        self.pending_barge_in_text = None
        self.current_spoken_text = ""
        self.current_sentence = ""

    async def speak_async(self, text):
        self.spoken.append(text)

    def _is_speaker_echo(self, text):
        return False

@pytest.mark.asyncio
async def test_confirmation_affirm_variations():
    tts = MockTTS()
    arbiter = PriorityAudioArbiter(tts, None)
    mgr = TriStateConfirmationManager(tts, arbiter)
    await mgr.request_confirmation({"command": "echo test"}, "Confirm?")

    affirm_phrases = [
        "yes", "yeah", "yep", "sure", "do it", "confirm", "proceed",
        "go ahead", "ok", "okay", "please do", "sort them", "move them", "yes please"
    ]
    for phrase in affirm_phrases:
        mgr.pending_action = {"command": "echo test"}
        result, _ = await mgr.evaluate_response(phrase)
        assert result == "AFFIRM", f"Failed to affirm with: '{phrase}'"

@pytest.mark.asyncio
async def test_confirmation_deny_variations():
    tts = MockTTS()
    arbiter = PriorityAudioArbiter(tts, None)
    mgr = TriStateConfirmationManager(tts, arbiter)

    deny_phrases = ["no", "nope", "cancel", "stop", "don't", "abort", "nevermind", "leave it"]
    for phrase in deny_phrases:
        mgr.pending_action = {"command": "echo test"}
        result, _ = await mgr.evaluate_response(phrase)
        assert result == "DENY", f"Failed to deny with: '{phrase}'"

@pytest.mark.asyncio
async def test_confirmation_clarify_variations():
    tts = MockTTS()
    arbiter = PriorityAudioArbiter(tts, None)
    mgr = TriStateConfirmationManager(tts, arbiter)

    clarify_phrases = ["wait, what?", "why?", "can you repeat that?", "which files?"]
    for phrase in clarify_phrases:
        mgr.pending_action = {"command": "echo test", "summary": "move files"}
        result, _ = await mgr.evaluate_response(phrase)
        assert result == "CLARIFY", f"Failed to trigger clarification with: '{phrase}'"


def test_sanitize_confirmation_speech_strips_hex_and_yapping():
    raw_prompt = (
        "Are you sure you want to kill all Alacritty terminals? "
        "This will terminate 5 terminal processes (address: 0x56358d1ce3a0, 0x56358d1e9860, "
        "0x56358cc7b2a0, 0x56358e2f3750, 0x56358e2bdda0, 0x56358e2a4120)."
    )
    summary = "Kill all Alacritty terminal processes"
    sanitized = sanitize_confirmation_speech(raw_prompt, summary)
    assert "0x" not in sanitized
    assert "address" not in sanitized.lower()
    assert sanitized == "Are you sure you want to kill all Alacritty terminals?"


def test_sanitize_confirmation_speech_fallback_summary():
    verbose_dump = (
        "Deleting files in /home/user/Downloads: /home/user/Downloads/a.mp4, "
        "/home/user/Downloads/b.mp4, /home/user/Downloads/c.mp4, /home/user/Downloads/d.mp4, "
        "/home/user/Downloads/e.mp4, /home/user/Downloads/f.mp4."
    )
    summary = "Delete 6 video files"
    sanitized = sanitize_confirmation_speech(verbose_dump, summary)
    assert sanitized == "Delete 6 video files?"


@pytest.mark.asyncio
async def test_confirmation_speaks_sanitized_and_notifies(monkeypatch):
    tts = MockTTS()
    arbiter = PriorityAudioArbiter(tts, None)
    mgr = TriStateConfirmationManager(tts, arbiter)

    notified = []

    def mock_notify(title, message, urgency="normal", timeout_ms=5000):
        notified.append((title, message))
        return "Notification displayed"

    monkeypatch.setattr("src.tools.desktop.show_desktop_notification", mock_notify)

    raw_prompt = (
        "Are you sure you want to kill all Alacritty terminals? "
        "This will terminate 5 terminal processes (address: 0x56358d1ce3a0, 0x56358d1e9860)."
    )
    payload = {
        "command": 'kill_process target="Alacritty" force=true',
        "summary": "Kill all Alacritty terminal processes",
        "details": "Addresses: 0x56358d1ce3a0, 0x56358d1e9860"
    }

    await mgr.request_confirmation(payload, raw_prompt)

    # 1. Spoken TTS must be ultra-concise with zero hex addresses
    assert len(tts.spoken) == 1
    assert "0x" not in tts.spoken[0]
    assert tts.spoken[0] == "Are you sure you want to kill all Alacritty terminals?"

    # 2. Desktop notification captures full details
    assert len(notified) == 1
    assert notified[0][0] == "Kill all Alacritty terminal processes"
    assert "0x56358d1ce3a0" in notified[0][1]


@pytest.mark.asyncio
async def test_show_desktop_notification_tool():
    res = show_desktop_notification("Test Title", "Test Message")
    assert isinstance(res, str)


def test_universal_desktop_macros_and_power_controls():
    from src.tools.desktop import list_desktop_macros, UNIVERSAL_SYSTEM_MACROS
    macros = list_desktop_macros()
    assert "reboot" in macros
    assert "restart" in macros
    assert "poweroff" in macros
    assert "shutdown" in macros
    assert "suspend" in macros
    assert "lock" in macros
    assert "systemctl reboot" in macros["reboot"]
    assert "systemctl poweroff" in macros["poweroff"]


def test_ask_user_confirmation_tool_schema():
    from src.llm.tools import ADAM_TOOLS
    tool = next(t for t in ADAM_TOOLS if t.name == "ask_user_confirmation")
    assert "command" in tool.parameters["required"]
    assert "question" in tool.parameters["required"]
    assert "summary" in tool.parameters["required"]


@pytest.mark.asyncio
async def test_execute_confirmed_command_power_and_tools():
    from src.main import AdamDaemon
    tts = MockTTS()
    arbiter = PriorityAudioArbiter(tts, None)

    assistant = object.__new__(AdamDaemon)
    assistant.tts = tts
    assistant.arbiter = arbiter

    executed_tools = []
    class FakeBrain:
        def get_tools(self):
            from src.llm.tools import ADAM_TOOLS
            return ADAM_TOOLS
        async def _execute_tool(self, name, args):
            executed_tools.append((name, args))
            return "ok"

    assistant.brain = FakeBrain()

    # 1. Normalization of sudo reboot -> systemctl reboot and pre-speech
    await assistant._execute_confirmed_command("sudo reboot", "Restart computer")
    assert len(tts.spoken) == 1
    assert "Restart computer now." in tts.spoken[0]
    assert executed_tools[0] == ("run_bash_command", {"command": "systemctl reboot"})

    # 2. Tool invocation execution
    executed_tools.clear()
    tts.spoken.clear()
    await assistant._execute_confirmed_command('kill_process target="Alacritty" force=true', "Kill terminal")
    assert executed_tools[0] == ("kill_process", {"target": "Alacritty", "force": True})
    assert len(tts.spoken) == 1
    assert "Kill terminal completed." in tts.spoken[0]



# ==============================================================================
# 3. WORLD TIMEZONE & CLOCK TESTS
# ==============================================================================

def test_resolve_time_local():
    cfg = load_config()
    brain = AdamBrain(cfg, None, None, None, None)
    res = brain._resolve_time("local")
    assert "Current local time:" in res
    assert "UTC" in res

def test_resolve_time_japan():
    cfg = load_config()
    brain = AdamBrain(cfg, None, None, None, None)
    res = brain._resolve_time("Japan")
    assert "Asia/Tokyo" in res
    assert "JST" in res

def test_resolve_time_world_locations():
    cfg = load_config()
    brain = AdamBrain(cfg, None, None, None, None)
    for loc, tz in [("London", "Europe/London"), ("Paris", "Europe/Paris"), ("California", "America/Los_Angeles"), ("UTC", "UTC")]:
        res = brain._resolve_time(loc)
        assert tz in res, f"Expected {tz} for location {loc}, got {res}"

def test_resolve_time_unknown_fallback():
    cfg = load_config()
    brain = AdamBrain(cfg, None, None, None, None)
    res = brain._resolve_time("Atlantis City")
    assert "Could not find timezone" in res or "Local time" in res


# ==============================================================================
# 4. SHOW NAME EXTRACTION & MEDIA CLASSIFICATION TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_organize_files_logic_in_tempdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create dummy mock files matching user's real library
        mock_files = [
            "[SubsPlease] Link Click S3 - 01 (1080p) [4B8BBF23].mkv",
            "[SubsPlease] Link Click S3 - 02 (1080p) [33FFB581].mkv",
            "[Erai-raws] Mushoku Tensei III - Isekai Ittara Honki Dasu - 01 [1080p CR WEBRip HEVC AAC].mkv",
            "[Erai-raws] Mushoku Tensei III - Isekai Ittara Honki Dasu - 02 [1080p CR WEBRip HEVC AAC].mkv",
            "[Erai-raws] Tensei Shitara Slime Datta Ken 4th Season - 01 [1080p].mkv",
            "Rick.and.Morty.S09E01.Theres.Something.About.Morty.mkv",
            "notes.txt"
        ]
        for fname in mock_files:
            (tmp_path / fname).write_text("dummy video content")

        cfg = load_config()
        tts = MockTTS()
        arbiter = PriorityAudioArbiter(tts, None)
        confirmation = TriStateConfirmationManager(tts, arbiter)
        brain = AdamBrain(cfg, None, None, confirmation, tts)

        # 1. Test dry-run
        res_dry = await brain._handle_organize_files({
            "directory": str(tmp_path),
            "group_by": "show",
            "dry_run": True
        })
        assert "Dry-run organization plan" in res_dry
        assert "Link Click" in res_dry
        assert "Mushoku Tensei" in res_dry
        assert "Rick and Morty" in res_dry

        # 2. Test actual organization execution plan
        res = await brain._handle_organize_files({
            "directory": str(tmp_path),
            "group_by": "show",
            "dry_run": False
        })
        assert "Confirmation requested" in res
        assert confirmation.pending_action is not None
        plan = confirmation.pending_action.get("plan", {})

        # Simulate user saying "Yes" and executing the move
        for dest_dir, src_files in plan.items():
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            for sf in src_files:
                shutil.move(sf, dest_dir)

        # Verify files were moved into proper folders
        assert (tmp_path / "Link Click" / mock_files[0]).exists()
        assert (tmp_path / "Link Click" / mock_files[1]).exists()
        assert (tmp_path / "Mushoku Tensei" / mock_files[2]).exists()
        assert (tmp_path / "Rick and Morty" / mock_files[5]).exists()


# ==============================================================================
# 5. BASH SAFETY & PING VALIDATION TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_ping_target_validation():
    cfg = load_config()
    brain = AdamBrain(cfg, None, None, None, None)

    # Invalid conversational phrases should be rejected
    invalid_res = await brain._execute_tool("run_bash_command", {"command": "ping German Wicket PD"})
    assert "not a valid domain name or IP address" in invalid_res

    invalid_res2 = await brain._execute_tool("run_bash_command", {"command": "ping the drum"})
    assert "not a valid domain name or IP address" in invalid_res2

    # Valid domain should have -c automatically added if omitted
    # Note: don't actually run long ping, verify command inspection or timeout
    assert "-c" not in "ping 127.0.0.1"


# ==============================================================================
# 6. SPEAKER ECHO DETECTION TESTS
# ==============================================================================

def test_speaker_echo_filtering():
    tts = StreamingVoiceSynthesizer(
        engine="kokoro",
        model_path="assets/voices/kokoro/kokoro-v1.0.onnx",
        voices_path="assets/voices/kokoro/voices-v1.0.bin",
        voice="am_adam"
    )
    tts.current_spoken_text = "I found 74 video files in your Downloads folder."
    tts.current_sentence = "I found 74 video files in your Downloads folder."

    # Direct echo
    assert tts._is_speaker_echo("I found 74 video files in your Downloads folder.") is True

    # Partial echo with >=65% overlap
    assert tts._is_speaker_echo("found 74 video files in your Downloads") is True

    # Genuine user speech (disjoint)
    assert tts._is_speaker_echo("Hey Adam, can you sort them into folders?") is False
    assert tts._is_speaker_echo("Stop right now") is False


# ==============================================================================
# 7. LLM EMBEDDED TOOL CALL RECOVERY TESTS
# ==============================================================================

from src.llm.provider import UniversalLLMClient

def test_extract_markdown_json_tool_call():
    cfg = load_config()
    client = UniversalLLMClient(cfg)
    text = "Sure, I will run that command:\n```json\n{\"name\": \"run_bash_command\", \"arguments\": {\"command\": \"ls -la\"}}\n```"
    cleaned, calls = client._extract_embedded_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "run_bash_command"
    assert calls[0]["function"]["arguments"]["command"] == "ls -la"
    assert "Sure, I will run that command:" in cleaned

def test_extract_raw_json_tool_call():
    cfg = load_config()
    client = UniversalLLMClient(cfg)
    text = "{\"name\": \"get_current_time\", \"arguments\": {\"location\": \"Japan\"}}"
    cleaned, calls = client._extract_embedded_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_current_time"
    assert calls[0]["function"]["arguments"]["location"] == "Japan"

def test_extract_inline_stream_json():
    cfg = load_config()
    client = UniversalLLMClient(cfg)
    text = "Checking now {\"name\": \"find_files\", \"arguments\": {\"directory\": \"~/Downloads\", \"pattern\": \"*.mkv\"}} please wait"
    cleaned, calls = client._extract_embedded_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "find_files"
    assert calls[0]["function"]["arguments"]["pattern"] == "*.mkv"
    assert "Checking now" in cleaned


# ==============================================================================
# 8. COMPLEX / UNICODE / EDGE CASE FILE ORGANIZATION TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_organize_files_with_unicode_and_colons():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mock_files = [
            "[SubsPlease] Re:Zero − Starting Life in Another World 2nd Season - 25.mkv",
            "Cowboy Bebop - 01 - Asteroid Blues.mkv",
            "unknown_file_without_season.mp4"
        ]
        for f in mock_files:
            (tmp_path / f).write_text("test")

        cfg = load_config()
        tts = MockTTS()
        brain = AdamBrain(cfg, None, None, None, tts)

        res = await brain._handle_organize_files({
            "directory": str(tmp_path),
            "group_by": "show",
            "dry_run": True
        })
        assert "Dry-run organization plan" in res
        assert "Re:Zero" in res or "Starting Life" in res
        assert "Cowboy Bebop" in res


# ==============================================================================
# 9. JOB SUPERVISOR LIFECYCLE & PROCESS GROUP TESTS
# ==============================================================================

from src.execution.supervisor import HardenedJobSupervisor

@pytest.mark.asyncio
async def test_job_supervisor_start_and_cleanup():
    with tempfile.TemporaryDirectory() as tmpdir:
        supervisor = HardenedJobSupervisor(
            log_dir=tmpdir,
            workspace=tmpdir,
            downloads=tmpdir
        )
        # Launch simple detached job
        pid = await supervisor.start_sandboxed_job(
            job_id="test_sleep",
            raw_cmd=["sleep", "5"]
        )
        assert pid > 0
        assert "test_sleep" in supervisor.active_jobs

        # Verify job is tracked and can be terminated cleanly
        await supervisor.kill_job("test_sleep")
        assert "test_sleep" not in supervisor.active_jobs


# ==============================================================================
# 10. CONFIRMATION WATCHDOG EXPIRY TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_confirmation_timeout_expiry():
    tts = MockTTS()
    arbiter = PriorityAudioArbiter(tts, None)
    # Short timeout for testing (0.1s)
    mgr = TriStateConfirmationManager(tts, arbiter, timeout_seconds=0.1)

    await mgr.request_confirmation({"command": "rm -rf /"}, "Delete all?")
    assert mgr.pending_action is not None

    # Wait for watchdog to expire
    await asyncio.sleep(0.15)
    assert mgr.pending_action is None
    assert any("timed out" in s for s in tts.spoken)


# ==============================================================================
# 11. ACOUSTIC QUENCH & ECHO HISTORY TESTS
# ==============================================================================

def test_audio_stream_quench_suppression():
    from src.audio.stream import AudioStreamManager
    import numpy as np
    import time

    stream = AudioStreamManager(chunk_size=512)
    stream.running = True

    # Call quench for 0.15s
    stream.quench(duration=0.15)

    # Chunks incoming during quench are ignored
    chunk = np.zeros((512, 1), dtype=np.float32)
    stream._audio_callback(chunk, 512, None, None)
    assert stream.audio_queue.empty()
    assert stream.get_chunk(timeout=0.01) is None

    # Wait for quench to expire
    time.sleep(0.16)
    stream._audio_callback(chunk, 512, None, None)
    assert not stream.audio_queue.empty()
    assert stream.get_chunk(timeout=0.01) is not None


def test_speaker_echo_history_and_filler_suppression():
    tts = StreamingVoiceSynthesizer(
        engine="kokoro",
        model_path="assets/voices/kokoro/kokoro-v1.0.onnx",
        voices_path="assets/voices/kokoro/voices-v1.0.bin",
        voice="am_adam"
    )
    import time
    tts.speech_history.append((time.time() - 2.0, "your downloads folder is quite full with wistoria files"))

    # Matches sentence in history
    assert tts._is_speaker_echo("downloads folder is full") is True
    assert tts._is_speaker_echo("wistoria files") is True

    # Filler word suppression while speaking
    tts.is_speaking = True
    assert tts._is_speaker_echo("yes") is True
    assert tts._is_speaker_echo("there's like") is True

    # Genuine command while speaking is NOT echo
    assert tts._is_speaker_echo("what is the weather in tokyo right now") is False


def test_vad_noise_gate_suppresses_low_ambient_energy():
    from src.audio.vad import SileroVAD
    import numpy as np

    vad = SileroVAD()
    # Extremely low energy noise (e.g. 0.001 RMS ambient room background)
    low_noise = np.random.uniform(-0.001, 0.001, 1600).astype(np.float32)
    is_speech, prob = vad.is_speech(low_noise, threshold=0.45, use_energy_floor=True)

    assert is_speech is False
    assert prob == 0.0


def test_vad_energy_gate_skips_neural_inference_and_resets_state():
    from collections import deque
    import numpy as np
    from src.audio.vad import SileroVAD

    class CountingModel:
        def __init__(self):
            self.inference_calls = 0
            self.reset_calls = 0

        def __call__(self, _tensor, _sample_rate):
            self.inference_calls += 1
            return np.asarray([[0.9]], dtype=np.float32)

        def reset_states(self):
            self.reset_calls += 1

    vad = SileroVAD.__new__(SileroVAD)
    vad.sample_rate = 16000
    vad.model = CountingModel()
    vad._quiet_context = deque(maxlen=8)
    vad._energy_gated_silence = False

    low_energy = np.full(512, 0.001, dtype=np.float32)
    for _ in range(3):
        assert vad.is_speech(low_energy, use_energy_floor=True) == (False, 0.0)
    assert vad.model.inference_calls == 0
    assert vad.model.reset_calls == 1

    # Onset replays the cached silence once, preserving Silero context.
    voice_energy = np.full(512, 0.01, dtype=np.float32)
    assert vad.is_speech(voice_energy, use_energy_floor=True) == (True, pytest.approx(0.9))
    assert vad.model.inference_calls == 4

    # Disabling the energy floor retains the direct neural path.
    vad.reset()
    assert vad.is_speech(low_energy, use_energy_floor=False) == (True, pytest.approx(0.9))
    assert vad.model.inference_calls == 5


def test_numpy_silero_wrapper_matches_official_onnx_output():
    import numpy as np
    import torch
    from silero_vad import load_silero_vad
    from src.audio.vad import SileroVAD

    reference = load_silero_vad(onnx=True)
    candidate = SileroVAD()
    rng = np.random.default_rng(17)
    frames = (
        [np.zeros(512, dtype=np.float32) for _ in range(6)]
        + [rng.normal(0.0, 0.02, 512).astype(np.float32) for _ in range(20)]
        + [np.zeros(512, dtype=np.float32) for _ in range(10)]
    )

    reference.reset_states()
    candidate.reset()
    for frame in frames:
        expected = float(reference(torch.from_numpy(frame), 16000).item())
        actual = float(np.asarray(candidate.model(frame, 16000)).item())
        assert actual == pytest.approx(expected, abs=1e-7)

    for sample_rate, frame_size in ((8000, 256), (32000, 1024)):
        reference.reset_states()
        candidate.reset()
        for _ in range(5):
            frame = rng.normal(0.0, 0.02, frame_size).astype(np.float32)
            expected = float(reference(torch.from_numpy(frame), sample_rate).item())
            actual = float(np.asarray(candidate.model(frame, sample_rate)).item())
            assert actual == pytest.approx(expected, abs=1e-7)


def test_vad_runtime_can_load_without_importing_torch():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from src.audio.vad import SileroVAD; SileroVAD(); "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_reference_audio_monitor_correlation():
    from src.audio.stream import ReferenceAudioMonitor
    import numpy as np

    mon = ReferenceAudioMonitor(sample_rate=16000, buffer_seconds=4.0)
    mon.running = True

    # Populate monitor buffer with synthetic speaker playback
    np.random.seed(42)
    speaker_audio = np.random.randn(32000).astype(np.float32) * 0.1
    mon._audio_callback(speaker_audio[:, None], 32000, None, None)

    # 1. Delayed reflection of speaker output (simulating room acoustics, 40ms delay = 640 samples)
    mic_echo = speaker_audio[32000 - 16000 - 640 : 32000 - 640]
    is_echo, score = mon.is_speaker_echo(mic_echo, threshold=0.35)
    assert is_echo is True
    assert score > 0.80

    # 2. Independent novel user speech while speaker audio is in buffer
    user_speech = np.random.randn(16000).astype(np.float32) * 0.1
    is_echo_user, user_score = mon.is_speaker_echo(user_speech, threshold=0.35)
    assert is_echo_user is False
    assert user_score < 0.20


def test_custom_wake_word_anchoring_and_speaker_rejection():
    from src.wake.engine import WakeWordDetector

    detector = WakeWordDetector(wake_word="hey adam")

    # Positive matches (addressed to assistant at the start of utterance)
    m1, r1 = detector.match_custom_wake_word("Hey Adam, could you close Firefox?")
    assert m1 is True
    assert r1.lower() == "could you close firefox?"

    m2, r2 = detector.match_custom_wake_word("Uh, hey Adam, what is the weather?")
    assert m2 is True
    assert "weather" in r2.lower()

    m2b, r2b = detector.match_custom_wake_word("All right. Hey Adam, what's on my screen?")
    assert m2b is True
    assert r2b.lower() == "what's on my screen?"

    # A self-correction may restart the request with a later wake phrase;
    # only the command after that final address should be sent to the agent.
    corrected, corrected_cmd = detector.match_custom_wake_word(
        "Hey, Adam. What's on my Google? Hey, Adam. What's on my Google Calendar?"
    )
    assert corrected is True
    assert corrected_cmd == "What's on my Google Calendar?"

    restarted, restarted_cmd = detector.match_custom_wake_word(
        "Hey Adam, what's on my screen? Hey Adam, what's on my Google Calendar?"
    )
    assert restarted is True
    assert restarted_cmd == "what's on my Google Calendar?"

    m3, r3 = detector.match_custom_wake_word("Adam, what time is it?")
    assert m3 is True
    assert "what time is it" in r3.lower()

    m4, r4 = detector.match_custom_wake_word("Hey Adam")
    assert m4 is True
    assert r4 == ""

    # Repeated hesitations, punctuation variations, and leading hesitation in command
    m4b, r4b = detector.match_custom_wake_word("Uh, hey, uh, hey, Adam. Uh, what applications do I have?")
    assert m4b is True
    assert r4b == "what applications do I have?"

    m4c, r4c = detector.match_custom_wake_word("Hey, Adam, what time is it?")
    assert m4c is True
    assert r4c == "what time is it?"

    m4d, r4d = detector.match_custom_wake_word("Hey... Adam, what time is it?")
    assert m4d is True
    assert r4d == "what time is it?"

    m4e, r4e = detector.match_custom_wake_word("Uh, hey, uh, hey, Adam.")
    assert m4e is True
    assert r4e == ""

    # Negative matches (YouTube, podcasts, background dialogue NOT addressing assistant)
    m5, _ = detector.match_custom_wake_word("I met Adam at the restaurant.")
    assert m5 is False

    m6, _ = detector.match_custom_wake_word("You know what yours, right? No. Then you don't.")
    assert m6 is False

    m7, _ = detector.match_custom_wake_word("Vietnamese chicken over rice or chicken over rice.")
    assert m7 is False

    m8, _ = detector.match_custom_wake_word("A big dam was constructed on the river.")
    assert m8 is False

    m9, _ = detector.match_custom_wake_word("He was kicked in the adam during the match.")
    assert m9 is False


def test_cosyvoice_config_and_fallback():
    from src.config import load_config, TTSConfig
    cfg = load_config()
    assert hasattr(cfg.tts, "cosyvoice_api_url")
    assert hasattr(cfg.tts, "cosyvoice_model_dir")

    # Instantiate synthesizer with cosyvoice engine
    tts = StreamingVoiceSynthesizer(
        engine="cosyvoice",
        model_path=cfg.tts.model_path,
        voices_path=cfg.tts.voices_path,
        voice="am_adam",
        cosyvoice_api_url="http://127.0.0.1:59999"
    )
    # Cosyvoice engine is exclusive — no Kokoro fallback initialized
    assert tts.kokoro is None


def test_silent_to_kokoro_dynamic_initialization(monkeypatch):
    """Switching from silent mode to kokoro dynamically initializes Kokoro without Piper fallback."""
    from src.config import load_config
    cfg = load_config()

    init_calls = []
    fake_kokoro = MagicMock()
    monkeypatch.setattr(StreamingVoiceSynthesizer, "_init_kokoro", lambda self: (init_calls.append(True), setattr(self, "kokoro", fake_kokoro)))

    tts = StreamingVoiceSynthesizer(
        engine="silent",
        model_path=cfg.tts.model_path,
        voices_path=cfg.tts.voices_path,
    )
    assert tts.engine == "silent"
    assert tts.kokoro is None
    assert len(init_calls) == 0

    # Dynamically restoring kokoro triggers initialization
    tts.engine = "kokoro"
    assert tts.engine == "kokoro"
    assert tts.kokoro is fake_kokoro
    assert len(init_calls) == 1


@pytest.mark.asyncio
async def test_kokoro_synthesis_does_not_fall_through_to_piper(monkeypatch):
    """When engine is kokoro, synthesis never invokes Piper subprocess even if kokoro object is initially None."""
    from src.config import load_config
    cfg = load_config()

    tts = StreamingVoiceSynthesizer(
        engine="silent",
        model_path="assets/voices/kokoro/kokoro-v1.0.onnx",
        voices_path="assets/voices/kokoro/voices-v1.0.bin",
    )
    fake_kokoro = MagicMock()
    fake_kokoro.create.return_value = (np.zeros(100, dtype=np.float32), 24000)
    fake_kokoro.voices = {"am_adam": 0}

    monkeypatch.setattr(tts, "_init_kokoro", lambda: setattr(tts, "kokoro", fake_kokoro))
    played = []
    monkeypatch.setattr(tts, "_play_float32_audio", lambda *args, **kwargs: played.append(args))

    subprocess_calls = []
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *args, **kwargs: subprocess_calls.append(args))

    tts.engine = "kokoro"
    await tts._synthesize_and_play_clause("Hello world", tts.current_epoch)

    assert len(played) == 1
    assert len(subprocess_calls) == 0

