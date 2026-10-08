import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from src.runtime import configure_native_threading

_CPU_THREAD_LIMIT = configure_native_threading()

import sys
import time
import re
import json
import difflib
import signal
import asyncio
import threading
import wave
import math
import uuid
from datetime import datetime
import numpy as np
from pathlib import Path

from src.config import load_config
from src.telemetry.events import configure_telemetry, emit_event, reset_trace_id, set_trace_id
from src.tools.desktop_timing import timed_stage
from src.audio.earcon import RobustEarconEngine
from src.audio.stream import AudioStreamManager
from src.audio.meeting import MeetingSession
from src.tts.streaming import StreamingVoiceSynthesizer
from src.wake.engine import WakeWordDetector
from src.stt.transcriber import CLOUD_STT_PROVIDERS, OpenAITranscriber, WhisperTranscriber, create_transcriber
from src.stt.speaker import SpeakerVerifier, default_profile_path, default_user_profile_path
from src.stt.diarizer import NemotronDiarizer, SpeakerSpan
from src.stt.meeting_speakers import MeetingSpeakerRegistry
from src.stt.target_separator import TargetSpeakerSeparator
from src.arbiter.arbiter import PriorityAudioArbiter, SystemState
from src.arbiter.confirmation import TriStateConfirmationManager
from src.execution.probe import HardwareEncoderProbe
from src.execution.supervisor import HardenedJobSupervisor
from src.llm.brain import AdamBrain
from src.audio.endpoint import SemanticEndpointer
from src.llm.speculative import SpeculativeRouter
from src.memory.manager import MemoryManager, extract_memory_command
from src.tools.weather import get_weather_report
from src.tools.web import web_search
from src.tools.system_telemetry import get_system_status
from src.tools.desktop import (
    ensure_gui_environment,
    configure_desktop_aliases,
    configure_desktop_macros,
    configure_disabled_capabilities
)

# Apply the same cap to libraries loaded while importing the application.
_CPU_THREAD_LIMIT = configure_native_threading()

def merge_overlapping_transcripts(p: str, s: str) -> str:
    """Merges two overlapping transcription snippets without repeating words or phrases."""
    p = p.strip()
    s = s.strip()
    if not p: return s
    if not s: return p
    if p.lower() == s.lower(): return p
    if s.lower().startswith(p.lower()): return s
    if p.lower().endswith(s.lower()): return p

    p_words = p.split()
    s_words = s.split()
    clean_p = [re.sub(r'[^\w]', '', w).lower() for w in p_words]
    clean_s = [re.sub(r'[^\w]', '', w).lower() for w in s_words]

    clean_p_str = " ".join(clean_p)
    clean_s_str = " ".join(clean_s)

    # Substring containment
    if clean_s_str in clean_p_str:
        return p
    if clean_p_str in clean_s_str:
        return s

    max_overlap = 0
    for k in range(1, min(len(clean_p), len(clean_s)) + 1):
        if clean_p[-k:] == clean_s[:k]:
            max_overlap = k

    if max_overlap > 0:
        return p + " " + " ".join(s_words[max_overlap:])

    return f"{p} {s}"


def meeting_command_kind(command: str) -> str | None:
    """Match only the two exact meeting controls reserved for direct dispatch."""
    normalized = re.sub(r"[^a-z0-9]+", " ", (command or "").casefold()).strip()
    if normalized == "meeting mode on":
        return "start"
    if normalized == "meeting mode off":
        return "stop"
    return None


def _should_attempt_speaker_isolation(
    transcript: str,
    wake_detector: WakeWordDetector,
    *,
    pretrained_wake_triggered: bool = False,
    transcript_confidence: float | None = None,
    speaker_similarity: float | None = None,
    speaker_threshold: float = 0.25,
) -> bool:
    """Use SepFormer for wake hints or low-confidence, speaker-like overlap."""
    if pretrained_wake_triggered or wake_detector.may_contain_custom_wake_word(transcript):
        return True
    if transcript_confidence is None or speaker_similarity is None:
        return False
    if not math.isfinite(transcript_confidence) or not math.isfinite(speaker_similarity):
        return False
    # The enrolled-speaker threshold is calibrated for one speaker; allow some
    # score loss when the local transcript suggests overlapping voices.
    similarity_floor = max(0.15, speaker_threshold * 0.6)
    return transcript_confidence <= -0.6 and speaker_similarity >= similarity_floor


class _LockedTranscriber:
    """Share the STT backend safely between meeting and barge-in workers."""

    def __init__(self, transcriber, lock: threading.Lock):
        self._transcriber = transcriber
        self._lock = lock

    def transcribe(self, audio: np.ndarray) -> str:
        with self._lock:
            return self._transcriber.transcribe(audio)

class AdamDaemon:
    """Master orchestrator for the Adam Voice Terminal Agent."""
    def __init__(self, config_path="config.yaml", enable_webui: bool | None = None):
        self.config = load_config(config_path)
        if enable_webui is not None:
            self.config.webui.enabled = enable_webui
        self.sidecar_runner = None
        self.sidecar_bridge = None
        configure_telemetry(self.config.telemetry)
        if hasattr(self.config, "desktop"):
            configure_desktop_aliases(
                self.config.desktop.application_aliases,
                self.config.desktop.window_aliases
            )
            if hasattr(self.config.desktop, "macros") and self.config.desktop.macros:
                configure_desktop_macros(self.config.desktop.macros)
            if hasattr(self.config.desktop, "disabled_capabilities") or hasattr(self.config.desktop, "disabled_tools"):
                configure_disabled_capabilities(
                    getattr(self.config.desktop, "disabled_capabilities", None),
                    getattr(self.config.desktop, "disabled_tools", None)
                )
        print("=" * 60)
        print(" ADAM: VOICE-ACTIVATED AUTONOMOUS TERMINAL AGENT")
        print("=" * 60)

        # 1. Audio and Speech Infrastructure
        print("[Init] Initializing RobustEarconEngine...")
        self.earcon = RobustEarconEngine(target_sink=self.config.audio.target_sink)

        print(f"[Init] Initializing StreamingVoiceSynthesizer ({self.config.tts.engine})...")
        self.tts = StreamingVoiceSynthesizer(
            engine=self.config.tts.engine,
            model_path=self.config.tts.model_path,
            voices_path=self.config.tts.voices_path,
            voice=self.config.tts.voice,
            device_id=self.config.tts.device_id,
            speed=self.config.tts.speed,
            piper_bin=self.config.tts.piper_bin,
            target_sink=self.config.audio.target_sink,
            sample_rate=self.config.tts.sample_rate,
            cloud_model=getattr(self.config.tts, "cloud_model", "gpt-4o-mini-tts"),
            cloud_voice=getattr(self.config.tts, "cloud_voice", "marin"),
            api_key=getattr(self.config.tts, "api_key", ""),
            cosyvoice_api_url=getattr(self.config.tts, "cosyvoice_api_url", "http://localhost:50000"),
            cosyvoice_model_dir=getattr(self.config.tts, "cosyvoice_model_dir", "pretrained_models/CosyVoice2-0.5B"),
            config_path=getattr(self.config.tts, "config_path", "assets/voices/en_US-ryan-high.onnx.json"),
        )

        print("[Init] Initializing PriorityAudioArbiter...")
        self.arbiter = PriorityAudioArbiter(self.tts, self.earcon)
        self.confirmation = TriStateConfirmationManager(self.tts, self.arbiter)

        # 2. Execution and Probing
        print("[Init] Initializing HardwareEncoderProbe & JobSupervisor...")
        self.probe = HardwareEncoderProbe()
        self.supervisor = HardenedJobSupervisor(
            log_dir=self.config.execution.jobs_log_dir,
            workspace=self.config.execution.workspace_dir,
            downloads=self.config.execution.downloads_dir
        )

        # 3. STT and Wake Word Engines
        stt_label = self.config.stt.cloud_model if self.config.stt.provider == "openai" else self.config.stt.model_size
        print(f"[Init] Initializing Speech-to-Text engine ({self.config.stt.provider}: {stt_label})...")
        self.stt = create_transcriber(self.config.stt, shared_api_key=self.config.llm.api_key)
        self._stt_lock = threading.Lock()
        self.wake_spotter = None

        speaker_cfg = self.config.speaker_verification
        profile_path = speaker_cfg.profile_path or None
        self.speaker_verifier = None
        if speaker_cfg.enabled:
            user_profiles = {}
            for user in speaker_cfg.users:
                if user.profiles:
                    user_profiles[user.name] = {
                        profile_name: path or default_user_profile_path(user.name, profile_name)
                        for profile_name, path in user.profiles.items()
                    }
                else:
                    user_profiles[user.name] = {
                        "default": user.profile_path or default_user_profile_path(user.name)
                    }
            candidate_verifier = SpeakerVerifier(
                profile_path=profile_path,
                threshold=speaker_cfg.threshold,
                user_profiles=user_profiles or None,
            )
            if candidate_verifier.enrolled:
                self.speaker_verifier = candidate_verifier
                print(
                    f"[Speaker] {len(candidate_verifier.profiles)} local voice profile(s) found; "
                    "command speaker checks enabled."
                )
            else:
                print("[Speaker] No local voice profile enrolled; speaker checks are opt-in and currently off.")
        else:
            print("[Speaker] Speaker verification disabled in config.")

        self.speaker_diarizer = None
        diarization_cfg = self.config.speaker_diarization
        if diarization_cfg.enabled:
            candidate_diarizer = NemotronDiarizer(
                executable=diarization_cfg.executable,
                model=diarization_cfg.model,
                device=diarization_cfg.device,
                timeout=diarization_cfg.timeout_seconds,
            )
            if not candidate_diarizer.available:
                print("[Diarization] Transformers runtime is missing; run setup with Nemotron diarization enabled.")
            elif not candidate_diarizer.supports_model():
                print(
                    f"[Diarization] No Nemotron model is configured ({diarization_cfg.model!r})."
                )
            else:
                self.speaker_diarizer = candidate_diarizer
                print(f"[Diarization] Nemotron enabled via Transformers ({diarization_cfg.device}); model loads on first use.")

        print("[Init] Initializing openWakeWord Detector...")
        self.wake = WakeWordDetector(
            wake_word=self.config.wake.wake_word,
            threshold=self.config.wake.threshold,
            aliases=getattr(self.config.wake, "aliases", []),
            energy_floor=getattr(self.config.audio, "wake_energy_floor", 0.0005),
        )
        stt_provider = str(getattr(self.config.stt, "provider", "local")).lower()
        if self.wake.is_custom_mode and stt_provider in CLOUD_STT_PROVIDERS:
            wake_cfg = self.config.stt
            print(
                f"[Wake] Preloading local '{wake_cfg.fallback_model}' model for wake spotting.",
                flush=True,
            )
            self.wake_spotter = WhisperTranscriber(
                model_size=wake_cfg.fallback_model,
                device=wake_cfg.fallback_device,
                device_index=wake_cfg.device_index,
                compute_type=wake_cfg.fallback_compute_type,
            )
            if isinstance(self.stt, OpenAITranscriber) and self.stt.attach_local_fallback(self.wake_spotter):
                print("[STT] Sharing the wake spotter as the local fallback to avoid loading a duplicate model.", flush=True)

        print("[Init] Initializing AudioStreamManager...")
        self.stream = AudioStreamManager(
            target_source=self.config.audio.target_source,
            sample_rate=16000,
            chunk_size=self.config.audio.chunk_size
        )
        meeting_cfg = self.config.meeting
        profile_path = self.config.speaker_verification.profile_path or None
        self.meeting_voice_encoder = self.speaker_verifier or SpeakerVerifier(profile_path=profile_path)
        self.meeting_speaker_registry = None
        self.meeting_session = MeetingSession(
            root=meeting_cfg.output_dir,
            sample_rate=self.stream.sample_rate,
            max_duration_seconds=meeting_cfg.max_duration_hours * 60.0 * 60.0,
            silence_duration=meeting_cfg.silence_duration,
            speech_threshold=self.config.audio.vad_threshold_idle,
            max_segment_seconds=meeting_cfg.max_segment_seconds,
            process_segment=self._process_meeting_segment,
        )
        self.stream.set_audio_tap(self.meeting_session.enqueue_audio)
        self.target_speaker_separator = None
        self.tts.set_audio_context(
            self.stream, self.wake, self.earcon,
            stt=_LockedTranscriber(self.stt, self._stt_lock),
            speaker_verifier=self.speaker_verifier,
        )

        # 4. Tiered Speech Architecture (Semantic Endpointing & Speculative Tool Pre-flight)
        print("[Init] Initializing SemanticEndpointer & SpeculativeRouter...")
        self.endpointer = SemanticEndpointer()
        self.speculative_router = SpeculativeRouter({
            "get_weather": get_weather_report,
            "web_search": web_search,
            "get_system_status": get_system_status,
            "battery_status": get_system_status,
        })

        # 5. Dynamic Memory & Agent Brain
        self.memory_manager = MemoryManager()

        print("[Init] Initializing AdamBrain ReAct Agent...")
        self.brain = AdamBrain(
            self.config,
            self.supervisor,
            self.probe,
            self.confirmation,
            self.tts,
            arbiter=self.arbiter,
            speculative_router=self.speculative_router,
            preload_ocr=bool(getattr(self.config.computer_control, "ocr_preload_on_startup", False)),
            preload_vision=bool(getattr(
                getattr(self.config, "computer_vision", None),
                "preload_on_startup",
                False,
            )),
            memory_mgr=self.memory_manager,
        )
        # MeetingSession belongs to AdamDaemon; expose its structured control
        # action to the brain without coupling the brain back to this daemon.
        self.brain.meeting_mode_handler = self._execute_meeting_mode_tool
        self.idea_router = None
        idea_cfg = getattr(self.config, "idea_routing", None)
        if idea_cfg is not None and idea_cfg.enabled:
            if idea_cfg.require_enrolled_speaker and self.speaker_verifier is None:
                print(
                    "[IdeaRouter] Disabled: enroll a speaker profile before enabling wake-free actions or background capture.",
                    flush=True,
                )
            else:
                try:
                    from src.intent.idea_router import IdeaRouter
                    candidate_router = IdeaRouter(
                        ideas_path=idea_cfg.ideas_path,
                        model_id=idea_cfg.model,
                        command_threshold=idea_cfg.command_threshold,
                        background_threshold=idea_cfg.background_threshold,
                        minimum_margin=idea_cfg.minimum_margin,
                    )
                    candidate_router.prepare()
                    self.idea_router = candidate_router
                    # Share the already loaded ONNX encoder with MemoryManager
                    self.memory_manager.embedder._encoder_fn = candidate_router.encode
                    self.memory_manager._rebuild_indexes()
                    print(
                        f"[IdeaRouter] Enabled with {idea_cfg.model}; idle speech is transcribed locally. "
                        "Commands and background calendar candidates use separate confidence thresholds.",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[IdeaRouter] Disabled because setup is incomplete ({exc}).", flush=True)
        self.speculative_router.set_executor("get_current_time", lambda location="local": self.brain._resolve_time(location))

        self.conversation_deadline = 0.0
        self.exit_keywords = re.compile(
            r"\b(goodbye|bye|that's all|thats all|that is all|exit|quit|stop listening|nevermind|nothing|dismissed|see ya|see you later)\b",
            re.IGNORECASE
        )

        self.running = False
        self._is_shutting_down = False
        self._active_heard_capture_id = None

    async def _record_utterance(self, **kwargs) -> np.ndarray:
        """Record and archive each finalized mic utterance, including rejected speech."""
        trace_id = str(uuid.uuid4())
        self._active_capture_trace_id = trace_id
        trace_token = set_trace_id(trace_id)
        emit_event("audio.capture_started", trace_id=trace_id, component="audio")
        try:
            audio_data = await asyncio.to_thread(self.stream.record_utterance, **kwargs)
        except Exception as exc:
            emit_event(
                "audio.capture_completed", trace_id=trace_id, component="audio", status="error",
                attributes={"error_type": type(exc).__name__},
            )
            raise
        finally:
            reset_trace_id(trace_token)
        emit_event(
            "audio.capture_completed", trace_id=trace_id, component="audio", status="ok",
            attributes={"sample_count": int(len(audio_data)), "sample_rate_hz": int(self.stream.sample_rate)},
        )
        self._active_heard_capture_id = self._save_heard_capture(audio_data)
        return audio_data

    def _save_heard_capture(self, audio_data: np.ndarray) -> str | None:
        """Save a private WAV and transcript sidecar for every finalized mic capture."""
        if audio_data is None or len(audio_data) == 0:
            return None
        try:
            state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            output_dir = state_home / "adam" / "heard-captures"
            output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(output_dir, 0o700)
            capture_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
            output_path = output_dir / f"heard-{capture_id}.wav"
            pcm = (
                np.clip(np.asarray(audio_data, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767
            ).astype(np.int16)
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.stream.sample_rate)
                wav_file.writeframes(pcm.tobytes())
            os.chmod(output_path, 0o600)

            metadata_path = output_path.with_suffix(".json")
            metadata_path.write_text(json.dumps({
                "capture_id": capture_id,
                "captured_at": datetime.now().astimezone().isoformat(),
                "sample_rate": self.stream.sample_rate,
                "duration_seconds": len(pcm) / self.stream.sample_rate,
                "transcripts": [],
            }, indent=2) + "\n", encoding="utf-8")
            os.chmod(metadata_path, 0o600)
            print(f"[HeardCapture] Saved mic audio: {output_path}", flush=True)
            return capture_id
        except Exception as exc:
            print(f"[HeardCapture] Could not save mic audio ({exc}).", flush=True)
            return None

    def _log_heard_transcript(self, capture_id: str | None, text: str | None, stage: str) -> None:
        if not capture_id:
            return
        try:
            state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            metadata_path = state_home / "adam" / "heard-captures" / f"heard-{capture_id}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["transcripts"].append({
                "stage": stage,
                "logged_at": datetime.now().astimezone().isoformat(),
                "text": text or "",
            })
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            os.chmod(metadata_path, 0o600)
        except Exception as exc:
            print(f"[HeardCapture] Could not save transcript ({exc}).", flush=True)

    async def _speaker_allowed(
        self, audio_data, *, include_score: bool = False
    ) -> bool | tuple[bool, float | None]:
        """Fail closed on a failed or non-matching speaker check when enrolled."""
        if self.speaker_verifier is None:
            return (True, None) if include_score else True
        try:
            matched, score = await asyncio.to_thread(self.speaker_verifier.verify, audio_data)
        except Exception as exc:
            print(f"[Speaker] Verification unavailable; ignoring utterance ({exc}).", flush=True)
            capture_id = getattr(self, "_active_heard_capture_id", None)
            if capture_id:
                transcript = await asyncio.to_thread(self._transcribe_wake_candidate, audio_data)
                self._log_heard_transcript(capture_id, transcript, "speaker-check-error")
            return (False, None) if include_score else False
        if not matched:
            print(f"[Speaker] Utterance did not match enrolled voice (score={score:.3f}); ignoring.", flush=True)
            capture_id = getattr(self, "_active_heard_capture_id", None)
            if capture_id:
                transcript = await asyncio.to_thread(self._transcribe_wake_candidate, audio_data)
                self._log_heard_transcript(capture_id, transcript, "speaker-rejected")
        else:
            print(f"[Speaker] Enrolled voice matched (score={score:.3f}).", flush=True)
        return (matched, float(score)) if include_score else matched

    def _save_wake_capture(self, audio_data: np.ndarray) -> None:
        """Save a confirmed wake utterance as private, mono 16-bit PCM WAV."""
        if audio_data is None or len(audio_data) == 0:
            return
        try:
            state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            output_dir = state_home / "adam" / "wake-captures"
            output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(output_dir, 0o700)
            timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
            output_path = output_dir / f"wake-{timestamp}.wav"
            pcm = (np.clip(np.asarray(audio_data, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.stream.sample_rate)
                wav_file.writeframes(pcm.tobytes())
            os.chmod(output_path, 0o600)
            print(f"[WakeCapture] Saved confirmed wake audio: {output_path}", flush=True)
        except Exception as exc:
            print(f"[WakeCapture] Could not save wake audio ({exc}).", flush=True)

    def _transcribe_wake_candidate(
        self, audio_data: np.ndarray, *, with_confidence: bool = False
    ) -> str | tuple[str, float | None]:
        """Use local ASR for custom wake spotting; reserve cloud ASR for wake candidates."""
        provider = str(getattr(self.config.stt, "provider", "local")).lower()
        if provider not in CLOUD_STT_PROVIDERS:
            text = self._transcribe_stt(audio_data, kind="wake")
            return (text, None) if with_confidence else text

        with self._stt_lock:
            if self.wake_spotter is None:
                cfg = self.config.stt
                print(
                    f"[Wake] Loading local '{cfg.fallback_model}' model for private wake spotting.",
                    flush=True,
                )
                self.wake_spotter = WhisperTranscriber(
                    model_size=cfg.fallback_model,
                    device=cfg.fallback_device,
                    device_index=cfg.device_index,
                    compute_type=cfg.fallback_compute_type,
                )
                if isinstance(self.stt, OpenAITranscriber):
                    self.stt.attach_local_fallback(self.wake_spotter)
            if with_confidence and hasattr(self.wake_spotter, "transcribe_with_confidence"):
                return self.wake_spotter.transcribe_with_confidence(audio_data)
            text = self.wake_spotter.transcribe(audio_data)
            return (text, None) if with_confidence else text

    def _transcribe_stt(self, audio_data: np.ndarray, *, kind: str = "final") -> str:
        """Serialize local/cloud STT calls shared by commands and meeting worker."""
        trace_token = None
        if kind in {"final", "partial", "wake"} and getattr(self, "_active_capture_trace_id", None):
            trace_token = set_trace_id(self._active_capture_trace_id)
        stt_provider = str(getattr(self.config.stt, "provider", "local"))
        stt_model = (
            getattr(self.config.stt, "cloud_model", "")
            if stt_provider.lower() in CLOUD_STT_PROVIDERS
            else getattr(self.config.stt, "model_size", "")
        )
        span_id = str(uuid.uuid4())
        emit_event(
            "stt.started", span_id=span_id, component="stt", provider=stt_provider,
            model=stt_model, status="started", attributes={"kind": kind},
        )
        try:
            with self._stt_lock:
                with timed_stage("stt.transcribe", kind=kind, provider=stt_provider):
                    text = self.stt.transcribe(audio_data)
        except Exception as exc:
            emit_event(
                "stt.completed", span_id=span_id, component="stt", status="error",
                provider=stt_provider, model=stt_model,
                attributes={"kind": kind, "error_type": type(exc).__name__},
            )
            if trace_token is not None:
                reset_trace_id(trace_token)
            raise
        emit_event(
            "stt.completed", span_id=span_id, component="stt", status="ok",
            provider=stt_provider, model=stt_model,
            attributes={
                "kind": kind,
                **({"usage": self.stt.last_usage} if getattr(self.stt, "last_usage", None) else {}),
            },
        )
        if trace_token is not None:
            reset_trace_id(trace_token)
        return text

    def _separate_enrolled_speaker(self, audio_data: np.ndarray) -> np.ndarray | None:
        if self.speaker_verifier is None or not self.speaker_verifier.enrolled:
            return None
        try:
            if self.target_speaker_separator is None:
                cache = Path.home() / ".cache" / "adam" / "sepformer-libri2mix"
                self.target_speaker_separator = TargetSpeakerSeparator(
                    self.speaker_verifier, model_dir=str(cache)
                )
            result = self.target_speaker_separator.isolate(
                audio_data, sample_rate=self.stream.sample_rate
            )
            if result is None:
                return None
            print(
                f"[Separation] Enrolled voice selected source {result.source_index} "
                f"(score={result.score:.3f}).",
                flush=True,
            )
            return result.audio
        except Exception as exc:
            print(f"[Separation] Target voice isolation failed ({exc}).", flush=True)
            return None

    def _process_meeting_segment(
        self, audio: np.ndarray, context: np.ndarray, start_seconds: float, end_seconds: float
    ) -> None:
        """Transcribe a VAD segment and attribute speech to stable meeting labels."""
        session = self.meeting_session
        if session.session_dir is None:
            return
        registry = self.meeting_speaker_registry
        turns = []
        if self.speaker_diarizer is not None:
            context = np.asarray(context, dtype=np.float32).reshape(-1)
            context_samples = len(context)
            combined = np.concatenate((context, audio)) if context_samples else audio
            try:
                spans = self.speaker_diarizer.diarize(combined)
                if context_samples:
                    offset = context_samples / self.stream.sample_rate
                    spans = [
                        SpeakerSpan(
                            speaker=span.speaker,
                            start=max(0.0, span.start - offset),
                            end=span.end - offset,
                        )
                        for span in spans
                        if span.end > offset
                    ]
                turns = self.speaker_diarizer.speaker_turns(
                    audio, spans, sample_rate=self.stream.sample_rate
                )
            except Exception as exc:
                print(f"[Meeting] Diarization failed for a speech segment ({exc}).", flush=True)

        if turns:
            # Learn stable participant labels from natural diarized turns, then
            # transcribe intact time crops from the original audio mix. No
            # samples are masked or source-separated in meeting mode.
            speaker_labels: dict[str, tuple[str, float | None]] = {}
            for turn in turns:
                if (
                    len(turn.speakers) == 1
                    and len(turn.audio) >= int(0.5 * self.stream.sample_rate)
                    and turn.speakers[0] not in speaker_labels
                ):
                    speaker_labels[turn.speakers[0]] = (
                        registry.label(turn.audio) if registry else ("Unknown", None)
                    )

            for turn in turns:
                participants = [speaker_labels.get(speaker) for speaker in turn.speakers]
                known = [item for item in participants if item is not None]
                if len(turn.speakers) == 1 and known:
                    label, score = known[0]
                elif len(turn.speakers) > 1 and len(known) == len(turn.speakers):
                    label = " + ".join(dict.fromkeys(item[0] for item in known))
                    score = None
                elif len(turn.speakers) > 1:
                    label, score = "Overlapping speakers", None
                else:
                    label, score = "Unknown", None

                text = self._transcribe_stt(turn.audio, kind="meeting").strip()
                if text:
                    turn_start = start_seconds + turn.start
                    turn_end = min(end_seconds, start_seconds + turn.end)
                    session.append_turn(label, text, turn_start, turn_end)
                    score_text = f" (voice score={score:.3f})" if score is not None else ""
                    print(f"[Meeting] {label}{score_text}: {text}", flush=True)
            return

        # When diarization is unavailable, retain the entire mixed segment and
        # mark it with the best voice-profile/cluster match we can make.
        text = self._transcribe_stt(audio, kind="meeting").strip()
        if text:
            label, score = registry.label(audio) if registry else ("Unknown", None)
            session.append_turn(label, text, start_seconds, end_seconds)
            score_text = f" (voice score={score:.3f})" if score is not None else ""
            print(f"[Meeting] {label}{score_text}: {text}", flush=True)

    async def _dispatch_meeting_command(
        self, command: str, *, cleanup: bool = True
    ) -> tuple[bool, bool]:
        """Return (handled, succeeded) for a meeting-mode command."""
        action = meeting_command_kind(command)
        if action is None:
            return False, False

        succeeded = True
        try:
            if action == "start":
                await self._start_meeting()
            elif self.meeting_session.active:
                await self._stop_meeting("voice command")
            else:
                await self._speak_meeting_already_off()
        except Exception as exc:
            succeeded = False
            verb = "start" if action == "start" else "stop"
            print(f"[Meeting] Failed to {verb} recording ({type(exc).__name__}: {exc}).", flush=True)
            try:
                await self.tts.speak_async(f"Meeting mode failed to {verb}.")
            except Exception as speech_exc:
                print(f"[Meeting] Failure acknowledgment failed ({type(speech_exc).__name__}).", flush=True)
        finally:
            if cleanup and self.arbiter.current_state != SystemState.AWAITING_CONFIRMATION:
                try:
                    await self.arbiter.set_state("IDLE_LISTENING")
                except Exception as exc:
                    print(f"[Meeting] State cleanup failed ({type(exc).__name__}: {exc}).", flush=True)
                try:
                    self.stream.flush()
                except Exception as exc:
                    print(f"[Meeting] Audio flush failed ({type(exc).__name__}: {exc}).", flush=True)
                try:
                    self.stream.quench(duration=0.4)
                except Exception as exc:
                    print(f"[Meeting] Audio quench failed ({type(exc).__name__}: {exc}).", flush=True)
        return True, succeeded

    async def _meeting_turn_response(self, command: str, *, cleanup: bool = True) -> str | None:
        """Dispatch meeting controls and provide a local response to non-voice callers."""
        action = meeting_command_kind(command)
        if action is None or not hasattr(self, "meeting_session"):
            return None
        was_active = self.meeting_session.active
        handled, succeeded = await self._dispatch_meeting_command(command, cleanup=cleanup)
        if not handled:
            return None
        if not succeeded:
            verb = "start" if action == "start" else "stop"
            return f"Meeting mode failed to {verb}."

        if action == "start":
            if self.meeting_session.active:
                return "Meeting mode is already on." if was_active else "Meeting mode is on. Recording now."
            return "Meeting mode failed to start."
        if was_active and self.meeting_session.active:
            return "Meeting mode failed to stop."

        if not was_active:
            return "Meeting mode is already off."
        directory = getattr(self.meeting_session, "session_dir", None)
        if directory is None:
            return "Meeting mode is off."
        return f"Meeting mode is off. I saved the recording and transcript in {directory}."

    async def _execute_meeting_mode_tool(self, action: str) -> str:
        """Run the structured meeting-mode tool through the daemon lifecycle."""
        action = str(action or "").strip().casefold()
        if action == "start":
            was_active = self.meeting_session.active
            try:
                await self._start_meeting()
            except Exception as exc:
                print(f"[Meeting] Structured start failed ({type(exc).__name__}: {exc}).", flush=True)
                return "Meeting mode failed to start."
            return "Meeting mode is already on." if was_active else "Meeting mode is on. Recording now."
        if action == "stop":
            was_active = self.meeting_session.active
            directory = self.meeting_session.session_dir
            if was_active:
                try:
                    await self._stop_meeting("voice command")
                except Exception as exc:
                    print(f"[Meeting] Structured stop failed ({type(exc).__name__}: {exc}).", flush=True)
                    return "Meeting mode failed to stop."
                if directory is not None:
                    return f"Meeting mode is off. I saved the recording and transcript in {directory}."
                return "Meeting mode is off. Recording stopped."
            await self._speak_meeting_already_off()
            return "Meeting mode is already off."
        return "Meeting mode action must be 'start' or 'stop'."

    async def _speak_meeting_already_off(self) -> None:
        try:
            await self.tts.speak_async("Meeting mode is already off.")
        except Exception as exc:
            print(f"[Meeting] Already-off acknowledgment failed ({type(exc).__name__}: {exc}).", flush=True)

    async def _execute_pretrained_wake_command(self, command: str) -> None:
        """Route pretrained-wake transcripts through meeting controls or Brain."""
        handled, _ = await self._dispatch_meeting_command(command)
        if handled:
            return
        memory_context = self.memory_manager.retrieve_context(command)
        await self._execute_turn(command, memory_context=memory_context)

    async def _start_meeting(self) -> None:
        if self.meeting_session.active:
            await self._speak_meeting_acknowledgement("Meeting mode is already on.")
            return
        self.meeting_speaker_registry = MeetingSpeakerRegistry(
            self.meeting_voice_encoder,
            enrolled_verifier=self.speaker_verifier,
            similarity_threshold=self.config.meeting.speaker_similarity_threshold,
        )
        try:
            directory = self.meeting_session.start()
        except Exception:
            self.meeting_speaker_registry = None
            raise
        self.conversation_deadline = 0.0
        print(f"[Meeting] Recording and transcription started: {directory}", flush=True)
        if self.speaker_diarizer is None:
            print("[Meeting] Speaker diarization is unavailable; turns will be labeled Unknown or by voice similarity.", flush=True)
        # Keep Adam's acknowledgement out of the meeting tap, then quench room
        # reflections before allowing capture to resume.
        await self._speak_meeting_acknowledgement("Meeting mode is on. Recording now.")

    async def _speak_meeting_acknowledgement(self, message: str) -> None:
        session = self.meeting_session
        suspension_id = session.suspend_capture()
        try:
            await self.tts.speak_async(message)
        except Exception as exc:
            # The recording has already started. Keep it active and let callers
            # report success truthfully even if speech output failed.
            print(f"[Meeting] Acknowledgment failed ({type(exc).__name__}: {exc}).", flush=True)
        finally:
            quench_duration = 0.4
            quenched = False
            try:
                self.stream.quench(duration=quench_duration)
                quenched = True
            except Exception as exc:
                print(f"[Meeting] Acknowledgment audio cleanup failed ({type(exc).__name__}: {exc}).", flush=True)
            finally:
                if quenched:
                    # The stream invokes the meeting tap before applying its
                    # quench, so keep the tap suspended through that interval.
                    try:
                        await asyncio.sleep(quench_duration)
                    finally:
                        session.resume_capture(suspension_id)
                else:
                    session.resume_capture(suspension_id)

    async def _stop_meeting(self, reason: str = "voice command") -> None:
        if not self.meeting_session.active:
            return
        directory = await asyncio.to_thread(self.meeting_session.stop, reason)
        self.conversation_deadline = 0.0
        self.meeting_speaker_registry = None
        print(f"[Meeting] Recording stopped ({reason}). Files saved in {directory}", flush=True)
        try:
            await self.tts.speak_async(f"Meeting mode is off. I saved the recording and transcript in {directory}.")
        except Exception as exc:
            # stop() completed successfully, so speech failure does not change
            # the saved recording's outcome reported to callers.
            print(f"[Meeting] Stop acknowledgment failed ({type(exc).__name__}: {exc}).", flush=True)

    def _contains_registered_voice(self, audio_data: np.ndarray) -> tuple[bool, float]:
        """Check short windows so background-only utterances never reach ASR."""
        if self.speaker_verifier is None:
            return True, 1.0
        audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        sample_rate = self.stream.sample_rate
        window_size = min(len(audio), int(1.2 * sample_rate))
        if window_size < int(0.5 * sample_rate):
            return False, -1.0

        step = int(0.6 * sample_rate)
        last_start = len(audio) - window_size
        starts = list(range(0, last_start + 1, step))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

        best_score = float("-inf")
        for start in starts:
            try:
                matched, score = self.speaker_verifier.verify(audio[start:start + window_size])
            except Exception:
                continue
            best_score = max(best_score, score)
            if matched:
                print(
                    f"[Speaker] Registered voice found in a short window "
                    f"(score={score:.3f}); allowing wake-word ASR.",
                    flush=True,
                )
                return True, float(score)

        print(
            f"[Speaker] No enrolled voice matched the mixed audio (best score={best_score:.3f}); trying wake-word fallback.",
            flush=True,
        )
        return False, float(best_score)

    def _registered_speaker_audio(
        self, audio_data: np.ndarray, context_audio: np.ndarray | None = None
    ) -> np.ndarray | None:
        """Diarize with recent context, then keep the enrolled voice from this utterance."""
        if self.speaker_diarizer is None or self.speaker_verifier is None:
            return None
        utterance = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        context = (
            np.asarray(context_audio, dtype=np.float32).reshape(-1)
            if context_audio is not None
            else np.zeros(0, dtype=np.float32)
        )
        context_samples = len(context)
        try:
            diarization_audio = np.concatenate((context, utterance)) if context_samples else utterance
            spans = self.speaker_diarizer.diarize(diarization_audio)
            if context_samples:
                context_seconds = context_samples / self.stream.sample_rate
                spans = [
                    SpeakerSpan(
                        speaker=span.speaker,
                        start=max(0.0, span.start - context_seconds),
                        end=span.end - context_seconds,
                    )
                    for span in spans
                    if span.end > context_seconds
                ]
            speaker_audio = self.speaker_diarizer.exclusive_speaker_audio(
                utterance, spans, sample_rate=self.stream.sample_rate
            )
        except Exception as exc:
            print(f"[Diarization] Failed; rejecting this utterance ({exc}).", flush=True)
            return None

        best_audio = None
        best_score = float("-inf")
        best_speaker = None
        for speaker, isolated_timeline in speaker_audio.items():
            try:
                matched, score = self.speaker_verifier.verify(isolated_timeline)
            except Exception:
                continue
            if not matched or score <= best_score:
                continue
            best_audio, best_score, best_speaker = isolated_timeline, score, speaker
        if best_audio is not None:
            print(
                f"[Diarization] Best enrolled-speaker match is {best_speaker} "
                f"(voice score={best_score:.3f}).",
                flush=True,
            )
            return best_audio
        print("[Diarization] No non-overlapping segment matched the enrolled voice.", flush=True)
        return None

    def _create_partial_callback(self, loop: asyncio.AbstractEventLoop, wake_word: str = ""):
        """Returns a non-blocking callback invoked on streaming partial audio chunks.
        Performs semantic endpoint analysis to dynamically scale VAD silence duration
        and dispatches speculative pre-flight tool execution while speech is ongoing."""
        if getattr(self.config.stt, "provider", "local").lower() in CLOUD_STT_PROVIDERS:
            # Avoid a cloud transcription request for every partial audio prefix.
            return lambda _audio_chunk: None

        def on_partial(audio_chunk) -> float | None:
            if len(audio_chunk) < int(0.5 * self.stream.sample_rate):
                return None

            # Filter out non-enrolled speakers on CPU (0% GPU) before invoking heavy ASR
            if self.speaker_verifier is not None:
                try:
                    matched, _ = self.speaker_verifier.verify(audio_chunk)
                    if not matched:
                        return None
                except Exception:
                    pass

            try:
                partial_text = self._transcribe_stt(audio_chunk, kind="partial")
                if not partial_text:
                    return None

                # 1. Semantic linguistic endpoint analysis
                endpoint_state, target_silence_s = self.endpointer.analyze(partial_text, wake_word=wake_word)
                if not isinstance(target_silence_s, (int, float)) or isinstance(target_silence_s, bool):
                    raise TypeError("SemanticEndpointer.analyze returned a non-numeric silence duration")
                target_silence_s = float(target_silence_s)
                if not math.isfinite(target_silence_s) or target_silence_s <= 0:
                    raise ValueError("SemanticEndpointer.analyze returned a non-finite or non-positive silence duration")
                emit_event(
                    "audio.endpoint_candidate", component="endpoint", status="ok",
                    attributes={"state": endpoint_state.value, "target_silence_s": target_silence_s},
                )

                # 2. Speculative preflight dispatch (non-blocking threadsafe call on asyncio loop)
                loop.call_soon_threadsafe(
                    lambda text=partial_text: asyncio.create_task(
                        self.speculative_router.preflight(text, wake_word=wake_word)
                    )
                )

                return target_silence_s
            except Exception as exc:
                emit_event(
                    "audio.endpoint_callback_error", component="endpoint", status="error",
                    attributes={"error_type": type(exc).__name__},
                )
                print(f"[Endpoint] Partial endpoint analysis failed ({type(exc).__name__}); keeping current silence timeout.", flush=True)
                return None

        return on_partial

    async def _execute_confirmed_command(
        self, cmd: str, summary: str, allowed_tools: list[str] | None = None,
    ):
        cmd = cmd.strip()
        if allowed_tools is not None:
            scoped_tool_names = {t.name for t in self.brain.get_tools()}
            scoped_first_token = re.split(r"[\s(]", cmd, maxsplit=1)[0].strip()
            confirmed_tool_dispatch = (
                scoped_first_token in scoped_tool_names
                and (
                    "(" in cmd
                    or re.search(
                        r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))',
                        cmd,
                    ) is not None
                )
            )
            required_tool = (
                scoped_first_token
                if confirmed_tool_dispatch
                else "run_bash_command"
            )
            if required_tool not in allowed_tools:
                await self.tts.speak_async(
                    "The confirmed action is outside this request's allowed tool scope, so I did not execute it."
                )
                return

        # Normalization for common power commands
        if cmd in ["sudo reboot", "reboot"]:
            cmd = "systemctl reboot"
        elif cmd in ["sudo poweroff", "poweroff", "shutdown", "shutdown -h now", "sudo shutdown -h now"]:
            cmd = "systemctl poweroff"

        is_shutdown_reboot = any(term in cmd for term in ["reboot", "poweroff", "shutdown", "halt"])
        if is_shutdown_reboot:
            msg = f"{summary} now." if summary else "Restarting now."
            await self.tts.speak_async(msg)

        tool_executed = False
        tool_names = {t.name for t in self.brain.get_tools()}
        first_token = re.split(r"[\s(]", cmd, maxsplit=1)[0].strip()
        if first_token in tool_names:
            import ast
            args = {}
            if "(" in cmd and cmd.endswith(")"):
                call_str = cmd[len(first_token):].strip()
                try:
                    tree = ast.parse(f"dummy{call_str}").body[0].value
                    for kw in getattr(tree, "keywords", []):
                        args[kw.arg] = ast.literal_eval(kw.value)
                except Exception:
                    pass
            if not args:
                for match in re.finditer(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))', cmd):
                    k = match.group(1)
                    v = match.group(2) or match.group(3) or match.group(4)
                    if v.lower() == "true":
                        v = True
                    elif v.lower() == "false":
                        v = False
                    args[k] = v

            if args or "(" in cmd:
                print(f"[Confirmation] Executing confirmed tool: {first_token}({args})")
                out = await self.brain._execute_tool(first_token, args)
                print(f"[Confirmation] Tool output: {out}")
                tool_executed = True

        if not tool_executed:
            print(f"[Confirmation] Executing confirmed shell command: {cmd}")
            out = await self.brain._execute_tool("run_bash_command", {"command": cmd})
            print(f"[Confirmation] Command output: {out}")

        if not is_shutdown_reboot:
            action_summary = summary or "Action"
            await self.tts.speak_async(f"{action_summary} completed.")

    async def _monitor_execution_interrupt(
        self,
        react_task: asyncio.Task,
    ) -> tuple[bool, str | None]:
        """Monitors microphone while react_task runs.
        Returns (True, interrupt_text) if user interrupted, or (False, None) if completed normally.
        """
        speech_buffer = []
        speech_samples = 0
        silence_samples = 0
        min_speech_samples = int(self.stream.sample_rate * 0.35)
        pause_seconds = max(0.6, float(self.config.audio.vad_silence_duration))
        pause_samples_needed = int(self.stream.sample_rate * pause_seconds)
        max_continuous_speech = int(
            self.stream.sample_rate * self.config.audio.max_utterance_seconds
        )

        while not react_task.done():
            # 1. Check if TTS caught barge-in during speech
            if getattr(self.tts, "pending_barge_in_text", None):
                barge_text = self.tts.pending_barge_in_text
                self.tts.pending_barge_in_text = None
                self.tts.pending_barge_in_audio = None
                self.brain.cancel_active_execution()
                react_task.cancel()
                try:
                    await react_task
                except (asyncio.CancelledError, Exception):
                    pass
                return True, barge_text

            # 2. If TTS is actively speaking, its own monitor handles mic stream
            if getattr(self.stream, "is_assistant_speaking", False):
                await asyncio.sleep(0.03)
                continue

            # 3. Read chunk from microphone
            chunk = await asyncio.to_thread(self.stream.get_chunk, timeout=0.04)
            if chunk is None or react_task.done():
                if react_task.done():
                    break
                await asyncio.sleep(0.01)
                continue

            # 4. Check for speech via VAD
            is_speech, prob = self.stream.vad.is_speech(
                chunk,
                threshold=self.config.audio.vad_threshold_speaking,
            )
            if is_speech:
                speech_buffer.append(chunk)
                speech_samples += len(chunk)
                silence_samples = 0
            else:
                if speech_samples > 0:
                    silence_samples += len(chunk)
                    speech_buffer.append(chunk)

            # 5. Evaluate speech when user pauses or reaches max speech length
            has_paused = (silence_samples >= pause_samples_needed)
            reached_cap = (speech_samples >= max_continuous_speech)

            if speech_samples >= min_speech_samples and (has_paused or reached_cap):
                audio_snippet = np.concatenate(speech_buffer)
                speech_buffer = []
                speech_samples = 0
                silence_samples = 0

                try:
                    text = await asyncio.to_thread(self._transcribe_wake_candidate, audio_snippet)
                except Exception:
                    text = ""

                text_clean = text.strip() if text else ""
                if text_clean:
                    matched_wake, rem_cmd = self.wake.match_explicit_wake_word(text_clean)

                    # Tool execution is interrupted only by the configured
                    # wake phrase. Standalone words such as "stop" or "wait"
                    # are ordinary speech while Adam is working.
                    if matched_wake:
                        # Verify speaker if configured
                        if self.speaker_verifier is not None and self.speaker_verifier.enrolled:
                            authorized = False
                            try:
                                authorized, _ = await asyncio.to_thread(
                                    self.speaker_verifier.verify, audio_snippet
                                )
                            except Exception:
                                authorized = False
                            if not authorized:
                                print(f"[Interruption] Ignored unverified speaker: '{text_clean}'", flush=True)
                                continue

                        print(f"\n[Interruption] >>> Interruption detected during tool execution ({text_clean})! Halting task <<<", flush=True)
                        self.earcon.play("interrupt")
                        self.tts.advance_epoch()
                        self.brain.cancel_active_execution()
                        react_task.cancel()
                        try:
                            await react_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        return True, rem_cmd if rem_cmd else text_clean

        try:
            await react_task
        except (asyncio.CancelledError, Exception):
            pass
        return False, None

    async def _execute_turn(
        self,
        command_text: str,
        memory_context: str | None = None,
        allowed_tools: list[str] | None = None,
    ) -> str | None:
        """Executes user commands through the ReAct agent with active real-time interruption monitoring."""
        action = meeting_command_kind(command_text)
        meeting_session = getattr(self, "meeting_session", None)
        if (
            action
            and meeting_session is not None
            and (allowed_tools is None or "meeting_mode" in allowed_tools)
        ):
            try:
                await self.arbiter.set_state("PROCESSING_REACT")
                meeting_response = await self._meeting_turn_response(command_text)
            finally:
                speculative_router = getattr(self, "speculative_router", None)
                if speculative_router is not None:
                    speculative_router.cancel_active()
            if meeting_response is not None:
                return meeting_response

        current_cmd: str | None = command_text
        current_mem = memory_context
        first_command = True

        await self.arbiter.set_state("PROCESSING_REACT")
        try:
            while current_cmd and self.running:
                cmd_to_run = current_cmd
                current_cmd = None
                if (
                    not first_command
                    and meeting_command_kind(cmd_to_run)
                    and (allowed_tools is None or "meeting_mode" in allowed_tools)
                ):
                    meeting_response = await self._meeting_turn_response(cmd_to_run, cleanup=False)
                    if meeting_response is not None:
                        return meeting_response
                first_command = False
                trace_id = getattr(self, "_active_capture_trace_id", None)
                trace_token = set_trace_id(trace_id) if trace_id else None
                try:
                    process_kwargs = {"memory_context": current_mem}
                    if allowed_tools is not None:
                        process_kwargs["allowed_tools"] = allowed_tools
                    react_task = asyncio.create_task(
                        self.brain.process_user_utterance(cmd_to_run, **process_kwargs)
                    )
                finally:
                    if trace_token is not None:
                        reset_trace_id(trace_token)
                current_mem = None

                interrupted, interrupt_cmd = await self._monitor_execution_interrupt(react_task)
                if not react_task.done():
                    try:
                        await react_task
                    except (asyncio.CancelledError, Exception):
                        pass
                if not interrupted:
                    break

                print(f"[Interruption] Execution halted mid-task by user: '{interrupt_cmd}'", flush=True)
                if interrupt_cmd:
                    stripped_cmd = re.sub(
                        r"^\s*(?:(?:hey\s+)?adam\s*[,;:]?\s*)?"
                        r"(?:stop|wait|hold on|cancel|nevermind|never mind|abort|pause|quiet|shut up)\b"
                        r"(?:\s*(?:and|then|please|,|;|\.|\bthat\b|\bit\b)\s*)*",
                        "",
                        interrupt_cmd,
                        flags=re.IGNORECASE,
                    ).strip()

                    matched_new, rem_new = self.wake.match_custom_wake_word(stripped_cmd)
                    if matched_new and rem_new:
                        stripped_cmd = rem_new.strip()

                    if stripped_cmd:
                        print(f"[Interruption] >>> Executing new command: \"{stripped_cmd}\" <<<", flush=True)
                        self.earcon.play("captured")
                        current_cmd = stripped_cmd
                    else:
                        print("[Interruption] Task aborted by user request.", flush=True)
                        break
                else:
                    break
        finally:
            self.speculative_router.cancel_active()
            if self.arbiter.current_state != SystemState.AWAITING_CONFIRMATION:
                await self.arbiter.set_state("IDLE_LISTENING")
                self.stream.flush()
                self.stream.quench(duration=0.4)
                if getattr(getattr(self, "meeting_session", None), "active", False):
                    self.conversation_deadline = 0.0
                    print("[Meeting] Returning to continuous meeting capture.", flush=True)
                else:
                    self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                    print(f"[Follow-up] Window open for {self.config.wake.followup_window_seconds:.1f}s (no wake word needed)...", flush=True)

    async def run(self):
        """Main non-blocking asynchronous event loop."""
        self.running = True
        loop = asyncio.get_running_loop()

        # Preload and warm up the LLM model in VRAM
        await self.brain.warmup()

        ensure_gui_environment()
        self.stream.start()
        self.earcon.play("done")
        print(f"\n[Adam] System is online and listening. (Say '{self.config.wake.wake_word}' or issue commands)\n")

        # Start optional WebUI sidecar if enabled
        if getattr(self.config, "webui", None) and self.config.webui.enabled:
            from sidecar.bridge import DaemonBridge
            from sidecar.server import start_sidecar
            self.sidecar_bridge = DaemonBridge(self)
            self.sidecar_runner = await start_sidecar(self.config.webui, self.sidecar_bridge)
            print(
                f"[WebUI] Sidecar listening on http://{self.config.webui.host}:{self.config.webui.port} "
                "(loopback only, opt-in).",
                flush=True,
            )

        while self.running:
            state = self.arbiter.current_state

            if self.meeting_session.max_duration_reached:
                await self._stop_meeting("four-hour limit")
                state = self.arbiter.current_state

            # ---------------- STATE 1: IDLE_LISTENING ----------------
            if state == SystemState.IDLE_LISTENING:
                is_in_followup = (
                    not self.meeting_session.active
                    and time.time() < self.conversation_deadline
                )
                if not is_in_followup and self.conversation_deadline > 0.0:
                    self.conversation_deadline = 0.0
                    print("[Follow-up] Conversation window closed. Returning to standby.", flush=True)

                # Check for pending natural barge-in utterance captured during speech
                if getattr(self.tts, "pending_barge_in_text", None):
                    barge_cmd = self.tts.pending_barge_in_text
                    barge_audio = getattr(self.tts, "pending_barge_in_audio", None)
                    barge_speaker_verified = getattr(
                        self.tts, "pending_barge_in_speaker_verified", False
                    )
                    self.tts.pending_barge_in_text = None
                    self.tts.pending_barge_in_audio = None
                    self.tts.pending_barge_in_speaker_verified = False
                    barge_capture_id = self._save_heard_capture(barge_audio)
                    self._log_heard_transcript(barge_capture_id, barge_cmd, "barge-in-transcript")
                    self._active_heard_capture_id = barge_capture_id

                    if (
                        barge_audio is not None
                        and not barge_speaker_verified
                        and not await self._speaker_allowed(barge_audio)
                    ):
                        self.stream.flush()
                        self.stream.quench(duration=0.4)
                        continue

                    # If user is still speaking, capture remainder of speech until natural pause
                    trailing_audio = await self._record_utterance(
                        silence_duration=self.config.audio.vad_silence_duration,
                        max_duration=self.config.audio.max_utterance_seconds,
                        idle_threshold=self.config.audio.vad_threshold_idle,
                        speaking_threshold=self.config.audio.vad_threshold_speaking,
                        on_partial_audio=self._create_partial_callback(loop, wake_word="")
                    )
                    if len(trailing_audio) > 0:
                        if not await self._speaker_allowed(trailing_audio):
                            self.stream.flush()
                            self.stream.quench(duration=0.4)
                            continue
                        if barge_audio is not None:
                            import numpy as np
                            barge_audio = np.concatenate((barge_audio, trailing_audio))
                        else:
                            barge_audio = trailing_audio
                        trailing_text = await asyncio.to_thread(self._transcribe_stt, trailing_audio)
                        self._log_heard_transcript(
                            self._active_heard_capture_id, trailing_text, "barge-in-trailing-audio"
                        )
                        if trailing_text and not self.tts._is_speaker_echo(trailing_text):
                            barge_cmd = merge_overlapping_transcripts(barge_cmd, trailing_text)

                    if self.tts._is_speaker_echo(barge_cmd):
                        print(f"[Barge-In] Discarding residual speaker echo: \"{barge_cmd}\"", flush=True)
                        self.stream.flush()
                        self.stream.quench(duration=0.4)
                        continue

                    if barge_audio is not None and len(trailing_audio) > 0 and not await self._speaker_allowed(barge_audio):
                        self.stream.flush()
                        self.stream.quench(duration=0.4)
                        continue

                    # Strip wake word if user included it
                    matched, rem = self.wake.match_custom_wake_word(barge_cmd)
                    if matched:
                        self._save_wake_capture(barge_audio)
                        if rem:
                            barge_cmd = rem
                        else:
                            # User spoke only the wake word during barge-in
                            self.earcon.play("wake")
                            await self.arbiter.set_state("USER_SPEAKING")
                            print("[Mic] Listening for command...", flush=True)
                            prompt_audio = await self._record_utterance(
                                silence_duration=self.config.audio.vad_silence_duration,
                                max_duration=self.config.audio.max_utterance_seconds,
                                idle_threshold=self.config.audio.vad_threshold_idle,
                                speaking_threshold=self.config.audio.vad_threshold_speaking,
                                on_partial_audio=self._create_partial_callback(loop, wake_word="")
                            )
                            self.earcon.play("captured")
                            if len(prompt_audio) > 0:
                                if not await self._speaker_allowed(prompt_audio):
                                    await self.arbiter.set_state("IDLE_LISTENING")
                                    continue
                                prompt_text = await asyncio.to_thread(self._transcribe_stt, prompt_audio)
                                self._log_heard_transcript(
                                    self._active_heard_capture_id, prompt_text, "barge-in-command"
                                )
                                if prompt_text:
                                    barge_cmd = prompt_text
                                else:
                                    await self.arbiter.set_state("IDLE_LISTENING")
                                    self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                                    continue
                            else:
                                await self.arbiter.set_state("IDLE_LISTENING")
                                self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                                continue

                    is_interrupt_phrase = bool(
                        self.exit_keywords.search(barge_cmd) or
                        re.search(r"\b(stop|wait|hold on|quiet|shut up|nevermind|cancel|silence|shh)\b", barge_cmd, re.IGNORECASE)
                    )

                    if is_interrupt_phrase:
                        print(f"[Barge-In] Interruption command heard: \"{barge_cmd}\". Speech halted; conversation window open.", flush=True)
                        self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                        continue

                    # Execute the barged-in command directly
                    print(f"[Barge-In] >>> Executing barged-in command: \"{barge_cmd}\" <<<", flush=True)
                    self.earcon.play("captured")
                    await self._execute_turn(barge_cmd)
                    continue

                chunk = self.stream.get_chunk(timeout=0.08)
                if chunk is None:
                    await asyncio.sleep(0.01)
                    continue

                # Path A: Custom Wake Word via VAD + ASR spotting OR within follow-up window
                idea_listening = self.idea_router is not None and not self.meeting_session.active
                if self.wake.is_custom_mode or is_in_followup or idea_listening:
                    pretrained_wake_triggered = False
                    if idea_listening and not self.wake.is_custom_mode:
                        pretrained_wake_triggered, wake_score = self.wake.predict(chunk)
                        if pretrained_wake_triggered:
                            print(f"\n[Wake] Wake word detected! (Confidence: {wake_score:.2f})", flush=True)
                    dynamic_floor = max(0.0035, min(0.022, self.stream.ref_monitor.speaker_rms * 0.40)) if (self.stream.ref_monitor and self.stream.ref_monitor.is_active) else 0.0025
                    is_speech, prob = self.stream.vad.is_speech(
                        chunk,
                        threshold=self.config.audio.vad_threshold_idle,
                        energy_floor=dynamic_floor
                    )
                    if is_speech or pretrained_wake_triggered:
                        print(f"\n[Mic] Speech detected (prob={prob:.2f}). Recording utterance...", flush=True)
                        diarization_context = self.stream.get_recent_audio(
                            duration_s=5.0,
                            exclude_latest_samples=len(chunk),
                        )
                        partial_cb = self._create_partial_callback(loop, wake_word="") if is_in_followup else None
                        audio_data = await self._record_utterance(
                            silence_duration=self.config.audio.vad_silence_duration,
                            # In standby, long background speech can keep VAD
                            # open indefinitely and postpone wake-word checks.
                            # Commands are short; cap each candidate while the
                            # follow-up path still accepts longer requests.
                            max_duration=self.config.audio.max_utterance_seconds,
                            idle_threshold=self.config.audio.vad_threshold_idle,
                            speaking_threshold=self.config.audio.vad_threshold_speaking,
                            initial_chunk=chunk,
                            on_partial_audio=partial_cb
                        )
                        if len(audio_data) > 0:
                            full_utterance_audio = audio_data
                            already_isolated = False
                            speaker_similarity = None
                            wake_transcript_confidence = None
                            # First use ASR to look for the wake phrase in the mixed
                            # microphone signal. Nemotron is more expensive and should
                            # only run after there is a wake-word candidate.
                            wake_candidate_audio = audio_data
                            duration_s = len(audio_data) / self.stream.sample_rate
                            if self.speaker_diarizer is not None:
                                has_registered_voice, speaker_similarity = await asyncio.to_thread(
                                    self._contains_registered_voice, audio_data
                                )
                                if not has_registered_voice:
                                    transcript, wake_transcript_confidence = await asyncio.to_thread(
                                        self._transcribe_wake_candidate,
                                        audio_data,
                                        with_confidence=True,
                                    )
                                    self._log_heard_transcript(
                                        self._active_heard_capture_id, transcript, "speaker-prefilter-mixed-wake-check"
                                    )
                                    if not _should_attempt_speaker_isolation(
                                        transcript or "",
                                        self.wake,
                                        pretrained_wake_triggered=pretrained_wake_triggered,
                                        transcript_confidence=wake_transcript_confidence,
                                        speaker_similarity=speaker_similarity,
                                        speaker_threshold=float(getattr(self.speaker_verifier, "threshold", 0.25)),
                                    ):
                                        self.speculative_router.cancel_active()
                                        continue
                                    isolated = await asyncio.to_thread(
                                        self._separate_enrolled_speaker, audio_data
                                    )
                                    isolated_text = (
                                        await asyncio.to_thread(self._transcribe_wake_candidate, isolated)
                                        if isolated is not None else ""
                                    )
                                    isolated_match, _ = self.wake.match_custom_wake_word(
                                        isolated_text or ""
                                    )
                                    if isolated_match:
                                        audio_data = full_utterance_audio = isolated
                                        already_isolated = True
                                    else:
                                        self.speculative_router.cancel_active()
                                        continue
                            else:
                                speaker_allowed, speaker_similarity = await self._speaker_allowed(
                                    audio_data, include_score=True
                                )
                                if not speaker_allowed:
                                    transcript, wake_transcript_confidence = await asyncio.to_thread(
                                        self._transcribe_wake_candidate,
                                        audio_data,
                                        with_confidence=True,
                                    )
                                    if not _should_attempt_speaker_isolation(
                                        transcript or "",
                                        self.wake,
                                        pretrained_wake_triggered=pretrained_wake_triggered,
                                        transcript_confidence=wake_transcript_confidence,
                                        speaker_similarity=speaker_similarity,
                                        speaker_threshold=float(getattr(self.speaker_verifier, "threshold", 0.25)),
                                    ):
                                        self.speculative_router.cancel_active()
                                        continue
                                    isolated = await asyncio.to_thread(
                                        self._separate_enrolled_speaker, audio_data
                                    )
                                    isolated_text = (
                                        await asyncio.to_thread(self._transcribe_wake_candidate, isolated)
                                        if isolated is not None else ""
                                    )
                                    isolated_match, _ = self.wake.match_custom_wake_word(
                                        isolated_text or ""
                                    )
                                    if isolated_match:
                                        audio_data = full_utterance_audio = isolated
                                        already_isolated = True
                                    else:
                                        self.speculative_router.cancel_active()
                                        continue

                            if (
                                not is_in_followup
                                and (self.wake.is_custom_mode or idea_listening)
                            ):
                                text, wake_transcript_confidence = await asyncio.to_thread(
                                    self._transcribe_wake_candidate,
                                    audio_data,
                                    with_confidence=True,
                                )
                            else:
                                text = await asyncio.to_thread(self._transcribe_stt, audio_data)
                            self._log_heard_transcript(
                                self._active_heard_capture_id, text, "mixed-audio"
                            )
                            matched, remaining_cmd = self.wake.match_custom_wake_word(text or "")
                            if pretrained_wake_triggered and not self.wake.is_custom_mode:
                                matched, remaining_cmd = True, text or ""
                            mixed_wake_command = remaining_cmd if matched else ""

                            # If the full utterance was not recognized as a wake, retry
                            # the most recent five seconds. This catches a wake phrase
                            # obscured by speech earlier in a long utterance.
                            if (
                                not matched
                                and not is_in_followup
                                and self.speaker_diarizer is not None
                                and duration_s > 5.0
                            ):
                                wake_candidate_audio = audio_data[-int(5.0 * self.stream.sample_rate):]
                                tail_text = await asyncio.to_thread(
                                    self._transcribe_wake_candidate, wake_candidate_audio
                                )
                                self._log_heard_transcript(
                                    self._active_heard_capture_id, tail_text, "wake-candidate-tail"
                                )
                                tail_matched, tail_remaining = self.wake.match_custom_wake_word(tail_text or "")
                                if tail_matched:
                                    text = tail_text
                                    matched, remaining_cmd = tail_matched, tail_remaining
                                    mixed_wake_command = tail_remaining

                            # Keep meeting transcription on the original mixed
                            # stream, but always try the enrolled-voice separator
                            # on an unrecognized wake candidate. Only this
                            # command lane uses the isolated copy. This also lets
                            # normal wake listening work while other people speak.
                            if (
                                not matched
                                and not already_isolated
                                and not is_in_followup
                                and _should_attempt_speaker_isolation(
                                    text or "",
                                    self.wake,
                                    pretrained_wake_triggered=pretrained_wake_triggered,
                                    transcript_confidence=wake_transcript_confidence,
                                    speaker_similarity=speaker_similarity,
                                    speaker_threshold=float(getattr(self.speaker_verifier, "threshold", 0.25)),
                                )
                                and (
                                    self.meeting_session.active
                                    or self.speaker_verifier is not None
                                    or (self.stream.ref_monitor and self.stream.ref_monitor.is_active)
                                )
                            ):
                                isolated = await asyncio.to_thread(
                                    self._separate_enrolled_speaker, full_utterance_audio
                                )
                                if isolated is not None:
                                    isolated_text = await asyncio.to_thread(
                                        self._transcribe_wake_candidate, isolated
                                    )
                                    isolated_match, isolated_command = self.wake.match_custom_wake_word(
                                        isolated_text or ""
                                    )
                                    if isolated_match:
                                        audio_data = isolated
                                        text = isolated_text
                                        matched, remaining_cmd = isolated_match, isolated_command
                                        mixed_wake_command = ""
                                        already_isolated = True

                            # In standby, a local miss ends the attempt. Do not send
                            # ordinary enrolled-speaker chatter to the configured ASR.
                            if (
                                not is_in_followup
                                and not matched
                                and self.wake.is_custom_mode
                                and self.idea_router is None
                            ):
                                self.speculative_router.cancel_active()
                                continue

                            # Ambient idea routing has already required a positive
                            # enrolled-speaker check before transcription. Short,
                            # single-speaker wake commands often yield no exclusive
                            # Nemotron segment, so do not let diarization discard a
                            # command that passed speaker verification.
                            if (
                                self.speaker_diarizer is not None
                                and matched
                                and not already_isolated
                                and not idea_listening
                            ):
                                registered_audio = await asyncio.to_thread(
                                    self._registered_speaker_audio,
                                    full_utterance_audio,
                                    diarization_context,
                                )
                                if registered_audio is None:
                                    isolated = await asyncio.to_thread(
                                        self._separate_enrolled_speaker, full_utterance_audio
                                    )
                                    isolated_text = (
                                        await asyncio.to_thread(self._transcribe_wake_candidate, isolated)
                                        if isolated is not None else ""
                                    )
                                    isolated_match, isolated_command = self.wake.match_custom_wake_word(
                                        isolated_text or ""
                                    )
                                    if not isolated_match:
                                        self.speculative_router.cancel_active()
                                        continue
                                    audio_data = isolated
                                    text = isolated_text
                                    matched, remaining_cmd = isolated_match, isolated_command
                                    already_isolated = True
                                else:
                                    audio_data = registered_audio
                                    duration_s = len(audio_data) / self.stream.sample_rate
                                    # Require the enrolled speaker's isolated audio to
                                    # contain the wake phrase too; the mixed transcript
                                    # alone cannot establish who said it.
                                    text = await asyncio.to_thread(self._transcribe_stt, audio_data)
                                    self._log_heard_transcript(
                                        self._active_heard_capture_id, text, "enrolled-speaker-audio"
                                    )
                                    matched, remaining_cmd = self.wake.match_custom_wake_word(text or "")
                                    if not matched:
                                        local_text = await asyncio.to_thread(
                                            self._transcribe_wake_candidate, audio_data
                                        )
                                        self._log_heard_transcript(
                                            self._active_heard_capture_id,
                                            local_text,
                                            "enrolled-speaker-wake-check",
                                        )
                                        local_matched, local_remaining = self.wake.match_custom_wake_word(local_text or "")
                                        if local_matched:
                                            text = local_text
                                            matched, remaining_cmd = local_matched, local_remaining
                                    if not matched:
                                        isolated = await asyncio.to_thread(
                                            self._separate_enrolled_speaker, full_utterance_audio
                                        )
                                        isolated_text = (
                                            await asyncio.to_thread(self._transcribe_wake_candidate, isolated)
                                            if isolated is not None else ""
                                        )
                                        isolated_match, isolated_command = self.wake.match_custom_wake_word(
                                            isolated_text or ""
                                        )
                                        if isolated_match:
                                            audio_data = isolated
                                            text = isolated_text
                                            matched, remaining_cmd = isolated_match, isolated_command
                                            already_isolated = True
                                if not matched:
                                    print("[Diarization] Enrolled voice matched, but its isolated audio did not contain the wake phrase.", flush=True)
                                    self.speculative_router.cancel_active()
                                    continue
                                if mixed_wake_command and remaining_cmd:
                                    mixed_words = re.findall(r"[a-z0-9]+", mixed_wake_command.lower())
                                    isolated_words = re.findall(r"[a-z0-9]+", remaining_cmd.lower())
                                    agreement = difflib.SequenceMatcher(
                                        None, mixed_words, isolated_words, autojunk=False
                                    ).ratio()
                                    if agreement < 0.65:
                                        print(
                                            f"[Diarization] Mixed and enrolled-speaker commands disagree "
                                            f"(agreement={agreement:.2f}); rejecting ambiguous command.",
                                            flush=True,
                                        )
                                        self.speculative_router.cancel_active()
                                        continue
                            elif (
                                self.speaker_diarizer is None
                                and matched
                                and self.wake.is_custom_mode
                                and not is_in_followup
                                and str(self.config.stt.provider).lower() in CLOUD_STT_PROVIDERS
                            ):
                                # Only send audio to cloud ASR after local wake spotting.
                                # Preserve the local result if cloud transcription misses.
                                local_text = text
                                text = await asyncio.to_thread(self._transcribe_stt, audio_data)
                                self._log_heard_transcript(
                                    self._active_heard_capture_id, text, "cloud-command-transcript"
                                )
                                cloud_matched, cloud_remaining = self.wake.match_custom_wake_word(text or "")
                                if cloud_matched:
                                    matched, remaining_cmd = cloud_matched, cloud_remaining
                                else:
                                    text = local_text
                            self._log_heard_transcript(
                                self._active_heard_capture_id, text, "final-transcript"
                            )
                            if text:
                                print(f"[Speech] >>> Heard ({duration_s:.1f}s): \"{text}\" <<<", flush=True)
                            else:
                                print(f"[Speech] (No words recognized in {duration_s:.1f}s audio)", flush=True)
                                self.speculative_router.cancel_active()
                                continue

                            # Exit phrase handling in follow-up mode
                            if is_in_followup and self.exit_keywords.search(text):
                                print(f"[Follow-up] Exit phrase heard: \"{text}\". Closing conversation.", flush=True)
                                self.speculative_router.cancel_active()
                                self.conversation_deadline = 0.0
                                self.earcon.play("done")
                                await self.arbiter.set_state("IDLE_LISTENING")
                                continue

                            matched, remaining_cmd = self.wake.match_custom_wake_word(text)
                            target_cmd = None
                            matched_memory_context = None

                            if matched:
                                # At this point the wake was confirmed, and when
                                # diarization is active audio_data is the isolated
                                # enrolled-speaker track for the full utterance.
                                self._save_wake_capture(audio_data)
                                print(f"[Wake] >>> MATCHED WAKE WORD '{self.wake.raw_wake_word}'! Command: \"{remaining_cmd}\" <<<", flush=True)
                                if remaining_cmd:
                                    target_cmd = remaining_cmd
                                else:
                                    # User spoke only the wake word
                                    self.earcon.play("wake")
                                    await self.arbiter.set_state("USER_SPEAKING")
                                    print("[Mic] Listening for command...", flush=True)
                                    prompt_cb = self._create_partial_callback(loop, wake_word="")
                                    prompt_audio = await self._record_utterance(
                                        silence_duration=self.config.audio.vad_silence_duration,
                                        max_duration=self.config.audio.max_utterance_seconds,
                                        idle_threshold=self.config.audio.vad_threshold_idle,
                                        speaking_threshold=self.config.audio.vad_threshold_speaking,
                                        on_partial_audio=prompt_cb
                                    )
                                    self.earcon.play("captured")
                                    if len(prompt_audio) > 0:
                                        if not await self._speaker_allowed(prompt_audio):
                                            self.speculative_router.cancel_active()
                                            await self.arbiter.set_state("IDLE_LISTENING")
                                            continue
                                        prompt_text = await asyncio.to_thread(self._transcribe_stt, prompt_audio)
                                        self._log_heard_transcript(
                                            self._active_heard_capture_id, prompt_text, "wake-prompt"
                                        )
                                        if prompt_text:
                                            print(f"[Speech] >>> Prompt: \"{prompt_text}\" <<<", flush=True)
                                            target_cmd = prompt_text
                            elif is_in_followup:
                                # When speakers are actively playing desktop audio (YouTube, Spotify, games),
                                # suppress follow-up mode to prevent speaker voices from triggering!
                                if self.stream.ref_monitor and self.stream.ref_monitor.is_active:
                                    print(f"[Follow-up] Desktop audio is active on speakers; ignoring background speech without wake word.", flush=True)
                                    self.speculative_router.cancel_active()
                                    continue
                                if self.tts._is_speaker_echo(text):
                                    print(f"[Follow-up] Discarding acoustic echo of recent assistant speech: \"{text}\"", flush=True)
                                    self.speculative_router.cancel_active()
                                    self.stream.flush()
                                    self.stream.quench(duration=0.4)
                                    continue
                                print(f"[Follow-up] Continuous conversation turn: \"{text}\"", flush=True)
                                target_cmd = text
                            elif idea_listening and self.idea_router is not None:
                                if self.speaker_verifier is None or not await self._speaker_allowed(audio_data):
                                    self.speculative_router.cancel_active()
                                    continue
                                memory_text = extract_memory_command(text)
                                if memory_text is not None:
                                    rec = self.memory_manager.save(memory_text)
                                    print(f"[Memory] Saved user memory {rec.id}: \"{rec.text}\"", flush=True)
                                    await self.tts.speak_async("Memory saved.")
                                    self.stream.flush()
                                    self.stream.quench(duration=0.4)
                                    continue

                                matched_memory_context = self.memory_manager.retrieve_context(text)
                                if matched_memory_context is not None:
                                    target_cmd = text
                                    print(f"[Memory] Matched saved memory context for utterance.", flush=True)

                                if target_cmd is not None:
                                    idea_match = None
                                else:
                                    idea_match = self.idea_router.match(text)
                                if idea_match is not None and idea_match.accepted:
                                    print(
                                        f"[IdeaRouter] Matched {idea_match.idea_id} "
                                        f"(score={idea_match.score:.3f}, margin={idea_match.margin:.3f}).",
                                        flush=True,
                                    )
                                    if idea_match.route == "command":
                                        target_cmd = text
                                    elif idea_match.route == "background_calendar":
                                        idea = next(
                                            item for item in self.idea_router._load_ideas()
                                            if item["id"] == idea_match.idea_id
                                        )
                                        await self.arbiter.set_state("PROCESSING_REACT")
                                        try:
                                            await self.brain.process_background_observation(text, idea)
                                        finally:
                                            self.speculative_router.cancel_active()
                                            if self.arbiter.current_state != SystemState.AWAITING_CONFIRMATION:
                                                await self.arbiter.set_state("IDLE_LISTENING")
                                                self.stream.flush()
                                                self.stream.quench(duration=0.4)
                                        continue
                                elif idea_match is not None:
                                    print(
                                        f"[IdeaRouter] Ignored {idea_match.idea_id} "
                                        f"(score={idea_match.score:.3f}, margin={idea_match.margin:.3f}).",
                                        flush=True,
                                    )

                            if target_cmd:
                                memory_text = extract_memory_command(target_cmd)
                                if memory_text is not None:
                                    rec = self.memory_manager.save(memory_text)
                                    print(f"[Memory] Saved user memory {rec.id}: \"{rec.text}\"", flush=True)
                                    await self.tts.speak_async("Memory saved.")
                                    if self.meeting_session.active:
                                        self.stream.flush()
                                        self.stream.quench(duration=0.4)
                                    continue

                                if matched_memory_context is None:
                                    matched_memory_context = self.memory_manager.retrieve_context(target_cmd)
                                    if matched_memory_context is not None:
                                        print(f"[Memory] Matched saved memory context for command.", flush=True)

                                meeting_handled, _ = await self._dispatch_meeting_command(target_cmd)
                                if meeting_handled:
                                    continue
                                ensure_gui_environment()
                                self.earcon.play("captured")
                                await self._execute_turn(
                                    target_cmd,
                                    memory_context=matched_memory_context,
                                )
                            else:
                                self.speculative_router.cancel_active()
                                if text and idea_listening and not is_in_followup:
                                    print("[IdeaRouter] No high-confidence idea match; staying idle.", flush=True)
                                elif text and not is_in_followup:
                                    print(f"[Wake] (Phrase heard but not addressed to assistant; resuming listening)", flush=True)
                                if self.arbiter.current_state != SystemState.AWAITING_CONFIRMATION:
                                    await self.arbiter.set_state("IDLE_LISTENING")

                # Path B: Standard Pretrained openWakeWord ONNX (e.g. "hey jarvis", "alexa")
                else:
                    triggered, score = self.wake.predict(chunk)
                    if triggered:
                        print(f"\n[Wake] Wake word detected! (Confidence: {score:.2f})")
                        self.earcon.play("wake")
                        await self.arbiter.set_state("USER_SPEAKING")

                        # Record utterance following wake word
                        print("[Mic] Listening for prompt...")
                        prompt_cb = self._create_partial_callback(loop, wake_word="")
                        audio_data = await self._record_utterance(
                            silence_duration=self.config.audio.vad_silence_duration,
                            max_duration=self.config.audio.max_utterance_seconds,
                            idle_threshold=self.config.audio.vad_threshold_idle,
                            speaking_threshold=self.config.audio.vad_threshold_speaking,
                            on_partial_audio=prompt_cb
                        )

                        self.earcon.play("captured")

                        if len(audio_data) > 0:
                            if not await self._speaker_allowed(audio_data):
                                self.speculative_router.cancel_active()
                                self.wake.reset()
                                continue
                            # The detector fires on the live chunk; prepend it so
                            # the saved sample includes the wake phrase itself.
                            capture_audio = np.concatenate((chunk, audio_data))
                            self._save_wake_capture(capture_audio)
                            text = await asyncio.to_thread(self._transcribe_stt, audio_data)
                            self._log_heard_transcript(
                                self._active_heard_capture_id, text, "pretrained-wake-command"
                            )
                            if text:
                                memory_text = extract_memory_command(text)
                                if memory_text is not None:
                                    rec = self.memory_manager.save(memory_text)
                                    print(f"[Memory] Saved user memory {rec.id}: \"{rec.text}\"", flush=True)
                                    await self.tts.speak_async("Memory saved.")
                                    self.speculative_router.cancel_active()
                                    self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                                    self.wake.reset()
                                    continue

                                await self._execute_pretrained_wake_command(text)
                            else:
                                self.speculative_router.cancel_active()
                        else:
                            self.speculative_router.cancel_active()

                        self.wake.reset()

            # ---------------- STATE 2: AWAITING_CONFIRMATION ----------------
            elif state == SystemState.AWAITING_CONFIRMATION:
                text = None
                # 1. Immediately consume pending barge-in answer if user spoke while prompt was being asked
                if getattr(self.tts, "pending_barge_in_text", None):
                    text = self.tts.pending_barge_in_text
                    self.tts.pending_barge_in_text = None
                    self.tts.pending_barge_in_audio = None
                    print(f"[Confirmation] Consuming barge-in answer: '{text}'", flush=True)
                else:
                    # 2. Otherwise record verbal response from user
                    audio_data = await self._record_utterance(
                        silence_duration=self.config.audio.vad_silence_duration,
                        max_duration=self.config.audio.max_utterance_seconds,
                        idle_threshold=self.config.audio.vad_threshold_idle,
                        speaking_threshold=self.config.audio.vad_threshold_speaking,
                        on_partial_audio=self._create_partial_callback(loop, wake_word="")
                    )
                    if len(audio_data) > 0:
                        # Voice filter disabled for confirmations: short answers like "yes" or "no"
                        # follow an already-verified wake-word turn and produce unreliable scores on short audio.
                        text = await asyncio.to_thread(self._transcribe_stt, audio_data)
                        self._log_heard_transcript(
                            self._active_heard_capture_id, text, "confirmation-response"
                        )
                        print(f"[Confirmation] User said: '{text}'", flush=True)

                if text:
                    res = await self.confirmation.evaluate_response(text)
                    result, subsequent_cmd = res if isinstance(res, tuple) else (res, None)

                    if result == "AFFIRM":
                        self.earcon.play("done")
                        action = self.confirmation.last_confirmed_action or self.confirmation.pending_action
                        action_scope = None
                        if isinstance(action, dict):
                            action_scope = action.pop("_allowed_tools", None)
                        self.confirmation.last_confirmed_action = None
                        required_action_tool = {
                            "organize_files": "organize_files",
                            "batch_transcode": "transcode_video",
                        }.get(action.get("type")) if isinstance(action, dict) else None
                        invalid_action_scope = action_scope is not None and (
                            not isinstance(action_scope, list)
                            or any(not isinstance(name, str) or not name.strip() for name in action_scope)
                            or len(set(action_scope)) != len(action_scope)
                        )
                        action_out_of_scope = (
                            action_scope is not None
                            and required_action_tool is not None
                            and required_action_tool not in action_scope
                        )
                        if action:
                            if invalid_action_scope or action_out_of_scope:
                                await self.tts.speak_async(
                                    "I could not safely resume the confirmed action within its original tool scope, so I did not execute it."
                                )
                            elif action.get("type") == "organize_files":
                                plan = action.get("plan", {})
                                import shutil
                                moved = 0
                                collisions = 0
                                for dest_dir_str, src_files in plan.items():
                                    dest_dir = Path(dest_dir_str)
                                    dest_dir.mkdir(parents=True, exist_ok=True)
                                    for sf_str in src_files:
                                        sf = Path(sf_str)
                                        target_file = dest_dir / sf.name
                                        if target_file.exists():
                                            target_file = dest_dir / f"{sf.stem} (1){sf.suffix}"
                                            collisions += 1
                                        try:
                                            if sf.is_symlink():
                                                real_target = sf.resolve()
                                                sf.unlink()
                                                shutil.copy2(real_target, target_file)
                                            else:
                                                shutil.move(sf, target_file)
                                            moved += 1
                                        except Exception as me:
                                            print(f"[Confirmation] Move error for {sf.name}: {me}", flush=True)

                                summary = action.get("summary", "File organization")
                                msg = f"{summary} completed. Moved {moved} files."
                                if collisions > 0:
                                    msg += f" Renamed {collisions} colliding files."
                                await self.tts.speak_async(msg)
                            elif action.get("type") == "batch_transcode":
                                files = action.get("files", [])
                                encoder_args = action.get("encoder_args", [])
                                output_dir = action.get("output_dir", "")
                                if files:
                                    infile = files[0]
                                    outfile = f"{output_dir}/{Path(infile).stem}_av1.mkv"
                                    cmd = ["ffmpeg", "-y", "-i", infile] + encoder_args + [outfile]
                                    pid = await self.supervisor.start_sandboxed_job(
                                        job_id=f"transcode_{Path(infile).stem[:10]}",
                                        raw_cmd=cmd,
                                        arbiter=self.arbiter
                                    )
                                    await self.tts.speak_async(f"Started background transcoding with PID {pid}. I'll alert you when it's done.")
                            elif action.get("command"):
                                cmd = str(action.get("command", "")).strip()
                                summary = action.get("summary", "Action completed")
                                await self._execute_confirmed_command(
                                    cmd, summary, allowed_tools=action_scope
                                )
                            else:
                                summary = action.get("summary", "the requested action")
                                print(f"[Confirmation] Resuming brain with user affirmation: {summary}")
                                if action_scope is None:
                                    await self._execute_turn(f"User confirmed: proceed with {summary}")
                                else:
                                    await self._execute_turn(
                                        f"User confirmed: proceed with {summary}",
                                        allowed_tools=action_scope,
                                    )

                        if self.arbiter.current_state != SystemState.AWAITING_CONFIRMATION:
                            await self.arbiter.set_state("IDLE_LISTENING")
                            self.stream.flush()
                            self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                    elif result == "DENY":
                        self.confirmation.last_confirmed_action = None
                        await self.arbiter.set_state("IDLE_LISTENING")
                        self.stream.flush()
                        if subsequent_cmd:
                            print(f"[Confirmation] Executing subsequent command: '{subsequent_cmd}'", flush=True)
                            await self._execute_turn(subsequent_cmd)
                        else:
                            self.conversation_deadline = 0.0
                    elif result == "NEW_COMMAND" and subsequent_cmd:
                        self.confirmation.last_confirmed_action = None
                        print(f"[Confirmation] Switching to new command: '{subsequent_cmd}'", flush=True)
                        await self._execute_turn(subsequent_cmd)

                await asyncio.sleep(0.1)

            # ---------------- STATE 3: ASSISTANT_SPEAKING (Barge-in check) ----------------
            elif state == SystemState.ASSISTANT_SPEAKING:
                self.stream.is_assistant_speaking = True
                chunk = self.stream.get_chunk(timeout=0.05)
                if chunk is not None:
                    # Check for wake word or interrupt keywords during speech
                    triggered, _ = self.wake.predict(chunk)
                    if triggered:
                        print("\n[Barge-In] Wake word detected during speech! Aborting...")
                        self.earcon.advance_epoch()
                        self.tts.advance_epoch()
                        self.earcon.play("interrupt")
                        await self.arbiter.set_state("IDLE_LISTENING")
                        self.stream.flush()

                await asyncio.sleep(0.02)

            else:
                await asyncio.sleep(0.05)

        if getattr(self, "sidecar_runner", None) is not None:
            try:
                await self.sidecar_runner.cleanup()
            except Exception as exc:
                print(f"[WebUI] Error cleaning up sidecar: {exc}", flush=True)
            self.sidecar_runner = None
        self._is_shutting_down = False

    def shutdown(self):
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        print("\n[Adam] Shutting down cleanly...")
        self.running = False
        try:
            if getattr(self, "sidecar_runner", None) is not None:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.sidecar_runner.cleanup())
                    else:
                        loop.run_until_complete(self.sidecar_runner.cleanup())
                except Exception:
                    pass
                self.sidecar_runner = None
            brain = getattr(self, "brain", None)
            if brain is not None:
                brain.close()
            self.stream.stop()
            self.earcon.close()
        except Exception:
            pass

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Adam Voice-Activated Autonomous Terminal Agent")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--webui", action="store_true", help="Enable optional WebUI sidecar for this session")
    args = parser.parse_args()

    daemon = AdamDaemon(config_path=args.config, enable_webui=True if args.webui else None)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def sig_handler(sig, frame):
        daemon.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        loop.run_until_complete(daemon.run())
    except (KeyboardInterrupt, SystemExit):
        daemon.shutdown()

if __name__ == "__main__":
    main()
