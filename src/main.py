import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import time
import re
import difflib
import signal
import asyncio
import wave
from datetime import datetime
import numpy as np
from pathlib import Path

from src.config import load_config
from src.audio.earcon import RobustEarconEngine
from src.audio.stream import AudioStreamManager
from src.tts.streaming import StreamingVoiceSynthesizer
from src.wake.engine import WakeWordDetector
from src.stt.transcriber import WhisperTranscriber, create_transcriber
from src.stt.speaker import SpeakerVerifier
from src.stt.diarizer import NemotronDiarizer
from src.arbiter.arbiter import PriorityAudioArbiter, SystemState
from src.arbiter.confirmation import TriStateConfirmationManager
from src.execution.probe import HardwareEncoderProbe
from src.execution.supervisor import HardenedJobSupervisor
from src.llm.brain import AdamBrain
from src.audio.endpoint import SemanticEndpointer
from src.llm.speculative import SpeculativeRouter
from src.tools.weather import get_weather_report
from src.tools.web import web_search
from src.tools.system_telemetry import get_system_status
from src.tools.desktop import (
    ensure_gui_environment,
    configure_desktop_aliases,
    configure_desktop_macros,
    configure_disabled_capabilities
)

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

class AdamDaemon:
    """Master orchestrator for the Adam Voice Terminal Agent."""
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
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
        self.wake_spotter = None

        speaker_cfg = self.config.speaker_verification
        profile_path = speaker_cfg.profile_path or None
        self.speaker_verifier = None
        if speaker_cfg.enabled:
            candidate_verifier = SpeakerVerifier(
                profile_path=profile_path,
                threshold=speaker_cfg.threshold,
            )
            if candidate_verifier.enrolled:
                self.speaker_verifier = candidate_verifier
                print("[Speaker] Local voice profile found; command speaker checks enabled.")
            else:
                print("[Speaker] No local voice profile enrolled; speaker checks are opt-in and currently off.")
        else:
            print("[Speaker] Speaker verification disabled in config.")

        self.speaker_diarizer = None
        diarization_cfg = self.config.speaker_diarization
        if diarization_cfg.enabled:
            if self.speaker_verifier is None:
                print("[Diarization] Configured, but inactive until a speaker profile is enrolled.")
            else:
                candidate_diarizer = NemotronDiarizer(
                    executable=diarization_cfg.executable,
                    model=diarization_cfg.model,
                    device=diarization_cfg.device,
                    timeout=diarization_cfg.timeout_seconds,
                )
                if not candidate_diarizer.available:
                    print(f"[Diarization] {diarization_cfg.executable} not found; using the existing whole-utterance speaker check.")
                elif not candidate_diarizer.supports_model():
                    print(
                        f"[Diarization] Installed nemo-speech does not include {diarization_cfg.model}; "
                        "install a build whose model index includes Nemotron-3-Diarization."
                    )
                else:
                    self.speaker_diarizer = candidate_diarizer
                    print(f"[Diarization] Nemotron enabled via {diarization_cfg.executable} ({diarization_cfg.device}).")

        print("[Init] Initializing openWakeWord Detector...")
        self.wake = WakeWordDetector(
            wake_word=self.config.wake.wake_word,
            threshold=self.config.wake.threshold,
            aliases=getattr(self.config.wake, "aliases", [])
        )
        stt_provider = str(getattr(self.config.stt, "provider", "local")).lower()
        if self.wake.is_custom_mode and stt_provider in ("openai", "openrouter", "custom"):
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

        print("[Init] Initializing AudioStreamManager...")
        self.stream = AudioStreamManager(
            target_source=self.config.audio.target_source,
            sample_rate=16000,
            chunk_size=self.config.audio.chunk_size
        )
        self.tts.set_audio_context(self.stream, self.wake, self.earcon, stt=self.stt)

        # 4. Tiered Speech Architecture (Semantic Endpointing & Speculative Tool Pre-flight)
        print("[Init] Initializing SemanticEndpointer & SpeculativeRouter...")
        self.endpointer = SemanticEndpointer()
        self.speculative_router = SpeculativeRouter({
            "get_weather": get_weather_report,
            "web_search": web_search,
            "get_system_status": get_system_status,
            "battery_status": get_system_status,
        })

        # 5. Agent Brain
        print("[Init] Initializing AdamBrain ReAct Agent...")
        self.brain = AdamBrain(
            self.config,
            self.supervisor,
            self.probe,
            self.confirmation,
            self.tts,
            arbiter=self.arbiter,
            speculative_router=self.speculative_router
        )
        self.speculative_router.set_executor("get_current_time", lambda location="local": self.brain._resolve_time(location))

        self.conversation_deadline = 0.0
        self.exit_keywords = re.compile(
            r"\b(goodbye|bye|that's all|thats all|that is all|exit|quit|stop listening|nevermind|nothing|dismissed|see ya|see you later)\b",
            re.IGNORECASE
        )

        self.running = False
        self._is_shutting_down = False

    async def _speaker_allowed(self, audio_data) -> bool:
        """Fail closed on a failed or non-matching speaker check when enrolled."""
        if self.speaker_verifier is None:
            return True
        try:
            matched, score = await asyncio.to_thread(self.speaker_verifier.verify, audio_data)
        except Exception as exc:
            print(f"[Speaker] Verification unavailable; ignoring utterance ({exc}).", flush=True)
            return False
        if not matched:
            print(f"[Speaker] Utterance did not match enrolled voice (score={score:.3f}); ignoring.", flush=True)
        else:
            print(f"[Speaker] Enrolled voice matched (score={score:.3f}).", flush=True)
        return matched

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

    def _transcribe_wake_candidate(self, audio_data: np.ndarray) -> str:
        """Use local ASR for custom wake spotting; reserve cloud ASR for wake candidates."""
        provider = str(getattr(self.config.stt, "provider", "local")).lower()
        if provider not in ("openai", "openrouter", "custom"):
            return self.stt.transcribe(audio_data)

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
        return self.wake_spotter.transcribe(audio_data)

    def _contains_registered_voice(self, audio_data: np.ndarray) -> bool:
        """Check short windows so background-only utterances never reach ASR."""
        if self.speaker_verifier is None:
            return True
        audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        sample_rate = self.stream.sample_rate
        window_size = min(len(audio), int(1.2 * sample_rate))
        if window_size < int(0.5 * sample_rate):
            return False

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
                return True

        print(
            f"[Speaker] No registered voice found (best score={best_score:.3f}); skipping ASR.",
            flush=True,
        )
        return False

    def _registered_speaker_audio(self, audio_data: np.ndarray) -> np.ndarray | None:
        """Keep only the enrolled speaker's non-overlapping audio timeline."""
        if self.speaker_diarizer is None or self.speaker_verifier is None:
            return None
        try:
            spans = self.speaker_diarizer.diarize(audio_data)
            speaker_audio = self.speaker_diarizer.exclusive_speaker_audio(
                audio_data, spans, sample_rate=self.stream.sample_rate
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
        if getattr(self.config.stt, "provider", "local").lower() in ("openai", "openrouter", "custom"):
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
                partial_text = self.stt.transcribe(audio_chunk)
                if not partial_text:
                    return None

                # 1. Semantic linguistic endpoint analysis
                analysis = self.endpointer.analyze(partial_text, wake_word=wake_word)

                # 2. Speculative preflight dispatch (non-blocking threadsafe call on asyncio loop)
                loop.call_soon_threadsafe(
                    lambda text=partial_text: asyncio.create_task(
                        self.speculative_router.preflight(text, wake_word=wake_word)
                    )
                )

                return analysis.recommended_silence_s
            except Exception:
                return None

        return on_partial

    async def _execute_confirmed_command(self, cmd: str, summary: str):
        cmd = cmd.strip()
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

        while self.running:
            state = self.arbiter.current_state

            # ---------------- STATE 1: IDLE_LISTENING ----------------
            if state == SystemState.IDLE_LISTENING:
                is_in_followup = time.time() < self.conversation_deadline
                if not is_in_followup and self.conversation_deadline > 0.0:
                    self.conversation_deadline = 0.0
                    print("[Follow-up] Conversation window closed. Returning to standby.", flush=True)

                # Check for pending natural barge-in utterance captured during speech
                if getattr(self.tts, "pending_barge_in_text", None):
                    barge_cmd = self.tts.pending_barge_in_text
                    barge_audio = getattr(self.tts, "pending_barge_in_audio", None)
                    self.tts.pending_barge_in_text = None
                    self.tts.pending_barge_in_audio = None

                    if barge_audio is not None and not await self._speaker_allowed(barge_audio):
                        self.stream.flush()
                        self.stream.quench(duration=0.4)
                        continue

                    # If user is still speaking, capture remainder of speech until natural pause
                    trailing_audio = await asyncio.to_thread(
                        self.stream.record_utterance,
                        silence_duration=self.config.audio.vad_silence_duration,
                        max_duration=45.0,
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
                        trailing_text = await asyncio.to_thread(self.stt.transcribe, trailing_audio)
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
                            prompt_audio = await asyncio.to_thread(
                                self.stream.record_utterance,
                                silence_duration=self.config.audio.vad_silence_duration,
                                max_duration=45.0,
                                idle_threshold=self.config.audio.vad_threshold_idle,
                                speaking_threshold=self.config.audio.vad_threshold_speaking,
                                on_partial_audio=self._create_partial_callback(loop, wake_word="")
                            )
                            self.earcon.play("captured")
                            if len(prompt_audio) > 0:
                                if not await self._speaker_allowed(prompt_audio):
                                    await self.arbiter.set_state("IDLE_LISTENING")
                                    continue
                                prompt_text = await asyncio.to_thread(self.stt.transcribe, prompt_audio)
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
                    await self.arbiter.set_state("PROCESSING_REACT")
                    try:
                        await self.brain.process_user_utterance(barge_cmd)
                    finally:
                        if self.arbiter.current_state != SystemState.AWAITING_CONFIRMATION:
                            await self.arbiter.set_state("IDLE_LISTENING")
                            self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                            print(f"[Follow-up] Window open for {self.config.wake.followup_window_seconds:.1f}s (no wake word needed)...", flush=True)
                    continue

                chunk = self.stream.get_chunk(timeout=0.08)
                if chunk is None:
                    await asyncio.sleep(0.01)
                    continue

                # Path A: Custom Wake Word via VAD + ASR spotting OR within follow-up window
                if self.wake.is_custom_mode or is_in_followup:
                    dynamic_floor = max(0.0035, min(0.022, self.stream.ref_monitor.speaker_rms * 0.40)) if (self.stream.ref_monitor and self.stream.ref_monitor.is_active) else 0.0025
                    is_speech, prob = self.stream.vad.is_speech(
                        chunk,
                        threshold=self.config.audio.vad_threshold_idle,
                        energy_floor=dynamic_floor
                    )
                    if is_speech:
                        print(f"\n[Mic] Speech detected (prob={prob:.2f}). Recording utterance...", flush=True)
                        partial_cb = self._create_partial_callback(loop, wake_word="") if is_in_followup else None
                        audio_data = await asyncio.to_thread(
                            self.stream.record_utterance,
                            silence_duration=self.config.audio.vad_silence_duration,
                            max_duration=45.0,
                            idle_threshold=self.config.audio.vad_threshold_idle,
                            speaking_threshold=self.config.audio.vad_threshold_speaking,
                            initial_chunk=chunk,
                            on_partial_audio=partial_cb
                        )
                        if len(audio_data) > 0:
                            full_utterance_audio = audio_data
                            # First use ASR to look for the wake phrase in the mixed
                            # microphone signal. Nemotron is more expensive and should
                            # only run after there is a wake-word candidate.
                            wake_candidate_audio = audio_data
                            duration_s = len(audio_data) / self.stream.sample_rate
                            if self.speaker_diarizer is not None:
                                has_registered_voice = await asyncio.to_thread(
                                    self._contains_registered_voice, audio_data
                                )
                                if not has_registered_voice:
                                    self.speculative_router.cancel_active()
                                    continue
                            else:
                                if not await self._speaker_allowed(audio_data):
                                    self.speculative_router.cancel_active()
                                    continue

                            if self.wake.is_custom_mode and not is_in_followup:
                                text = await asyncio.to_thread(self._transcribe_wake_candidate, audio_data)
                            else:
                                text = await asyncio.to_thread(self.stt.transcribe, audio_data)
                            matched, remaining_cmd = self.wake.match_custom_wake_word(text or "")
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
                                tail_matched, tail_remaining = self.wake.match_custom_wake_word(tail_text or "")
                                if tail_matched:
                                    text = tail_text
                                    matched, remaining_cmd = tail_matched, tail_remaining
                                    mixed_wake_command = tail_remaining

                            # In standby, a local miss ends the attempt. Do not send
                            # ordinary enrolled-speaker chatter to the configured ASR.
                            if self.wake.is_custom_mode and not is_in_followup and not matched:
                                self.speculative_router.cancel_active()
                                continue

                            if self.speaker_diarizer is not None and matched:
                                registered_audio = await asyncio.to_thread(
                                    self._registered_speaker_audio, full_utterance_audio
                                )
                                if registered_audio is None:
                                    self.speculative_router.cancel_active()
                                    continue
                                audio_data = registered_audio
                                duration_s = len(audio_data) / self.stream.sample_rate
                                # Require the enrolled speaker's isolated audio to
                                # contain the wake phrase too; the mixed transcript
                                # alone cannot establish who said it.
                                text = await asyncio.to_thread(self.stt.transcribe, audio_data)
                                matched, remaining_cmd = self.wake.match_custom_wake_word(text or "")
                                if not matched:
                                    local_text = await asyncio.to_thread(
                                        self._transcribe_wake_candidate, audio_data
                                    )
                                    local_matched, local_remaining = self.wake.match_custom_wake_word(local_text or "")
                                    if local_matched:
                                        text = local_text
                                        matched, remaining_cmd = local_matched, local_remaining
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
                                and str(self.config.stt.provider).lower() in ("openai", "openrouter", "custom")
                            ):
                                # Only send audio to cloud ASR after local wake spotting.
                                # Preserve the local result if cloud transcription misses.
                                local_text = text
                                text = await asyncio.to_thread(self.stt.transcribe, audio_data)
                                cloud_matched, cloud_remaining = self.wake.match_custom_wake_word(text or "")
                                if cloud_matched:
                                    matched, remaining_cmd = cloud_matched, cloud_remaining
                                else:
                                    text = local_text
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
                                    prompt_audio = await asyncio.to_thread(
                                        self.stream.record_utterance,
                                        silence_duration=self.config.audio.vad_silence_duration,
                                        max_duration=45.0,
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
                                        prompt_text = await asyncio.to_thread(self.stt.transcribe, prompt_audio)
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

                            if target_cmd:
                                ensure_gui_environment()
                                self.earcon.play("captured")
                                await self.arbiter.set_state("PROCESSING_REACT")
                                try:
                                    await self.brain.process_user_utterance(target_cmd)
                                finally:
                                    self.speculative_router.cancel_active()
                                    if self.arbiter.current_state != SystemState.AWAITING_CONFIRMATION:
                                        await self.arbiter.set_state("IDLE_LISTENING")
                                        self.stream.flush()
                                        self.stream.quench(duration=0.4)
                                        self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                                        print(f"[Follow-up] Window open for {self.config.wake.followup_window_seconds:.1f}s (no wake word needed)...", flush=True)
                            else:
                                self.speculative_router.cancel_active()
                                if text and not is_in_followup:
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
                        audio_data = await asyncio.to_thread(
                            self.stream.record_utterance,
                            silence_duration=self.config.audio.vad_silence_duration,
                            max_duration=45.0,
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
                            text = await asyncio.to_thread(self.stt.transcribe, audio_data)
                            if text:
                                await self.arbiter.set_state("PROCESSING_REACT")
                                try:
                                    await self.brain.process_user_utterance(text)
                                finally:
                                    self.speculative_router.cancel_active()
                                    if self.arbiter.current_state != SystemState.AWAITING_CONFIRMATION:
                                        await self.arbiter.set_state("IDLE_LISTENING")
                                        self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                                        print(f"[Follow-up] Window open for {self.config.wake.followup_window_seconds:.1f}s...", flush=True)
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
                    audio_data = await asyncio.to_thread(
                        self.stream.record_utterance,
                        silence_duration=self.config.audio.vad_silence_duration,
                        max_duration=30.0,
                        idle_threshold=self.config.audio.vad_threshold_idle,
                        speaking_threshold=self.config.audio.vad_threshold_speaking,
                        on_partial_audio=self._create_partial_callback(loop, wake_word="")
                    )
                    if len(audio_data) > 0:
                        # Voice filter disabled for confirmations: short answers like "yes" or "no"
                        # follow an already-verified wake-word turn and produce unreliable scores on short audio.
                        text = await asyncio.to_thread(self.stt.transcribe, audio_data)
                        print(f"[Confirmation] User said: '{text}'", flush=True)

                if text:
                    res = await self.confirmation.evaluate_response(text)
                    result, subsequent_cmd = res if isinstance(res, tuple) else (res, None)

                    if result == "AFFIRM":
                        self.earcon.play("done")
                        action = self.confirmation.last_confirmed_action or self.confirmation.pending_action
                        if action:
                            if action.get("type") == "organize_files":
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
                                await self._execute_confirmed_command(cmd, summary)
                            else:
                                summary = action.get("summary", "the requested action")
                                print(f"[Confirmation] Resuming brain with user affirmation: {summary}")
                                await self.brain.process_user_utterance(f"User confirmed: proceed with {summary}")

                        if self.arbiter.state != SystemState.AWAITING_CONFIRMATION:
                            await self.arbiter.set_state("IDLE_LISTENING")
                            self.stream.flush()
                            self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                    elif result == "DENY":
                        await self.arbiter.set_state("IDLE_LISTENING")
                        self.stream.flush()
                        if subsequent_cmd:
                            print(f"[Confirmation] Executing subsequent command: '{subsequent_cmd}'", flush=True)
                            await self.arbiter.set_state("PROCESSING_REACT")
                            await self.brain.process_user_utterance(subsequent_cmd)
                            await self.arbiter.set_state("IDLE_LISTENING")
                            self.stream.flush()
                            self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds
                        else:
                            self.conversation_deadline = 0.0
                    elif result == "NEW_COMMAND" and subsequent_cmd:
                        print(f"[Confirmation] Switching to new command: '{subsequent_cmd}'", flush=True)
                        await self.arbiter.set_state("PROCESSING_REACT")
                        await self.brain.process_user_utterance(subsequent_cmd)
                        await self.arbiter.set_state("IDLE_LISTENING")
                        self.stream.flush()
                        self.conversation_deadline = time.time() + self.config.wake.followup_window_seconds

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

        self._is_shutting_down = False

    def shutdown(self):
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        print("\n[Adam] Shutting down cleanly...")
        self.running = False
        try:
            self.stream.stop()
            self.earcon.close()
        except Exception:
            pass

def main():
    daemon = AdamDaemon()
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
