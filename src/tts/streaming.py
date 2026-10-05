import os
import re
import time
import asyncio
import difflib
import threading
from collections import deque
from src.audio.earcon import resolve_pulse_device_index, setup_audio_routing
from src.telemetry.events import emit_event, get_speech_role, new_span_id

MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def _format_date_slash(m):
    month, day, year = int(m.group(1)), int(m.group(2)), m.group(3)
    if 1 <= month <= 12 and 1 <= day <= 31:
        return f"{MONTH_NAMES[month]} {day}, {year}"
    return m.group(0)

def _format_date_dash(m):
    year, month, day = m.group(1), int(m.group(2)), int(m.group(3))
    if 1 <= month <= 12 and 1 <= day <= 31:
        return f"{MONTH_NAMES[month]} {day}, {year}"
    return m.group(0)

def _num_to_words(n: int) -> str:
    """Converts an integer (up to 999,999,999) to English words for phonetic echo comparison."""
    if n == 0:
        return "zero"
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def _to_words(val):
        if val >= 1000000000:
            return _to_words(val // 1000000000) + " billion " + _to_words(val % 1000000000)
        if val >= 1000000:
            return _to_words(val // 1000000) + " million " + _to_words(val % 1000000)
        if val >= 1000:
            return _to_words(val // 1000) + " thousand " + _to_words(val % 1000)
        if val >= 100:
            return _to_words(val // 100) + " hundred " + _to_words(val % 100)
        if val >= 20:
            return tens[val // 10] + " " + ones[val % 10]
        return ones[val]

    return " ".join(_to_words(n).split())

def clean_speech_text(text: str) -> str:
    """Strips markup, normalizes IPs, dates, timestamps, decimals, and abbreviations so TTS speaks fluently."""
    if not text:
        return ""
    # 1. Strip tool pseudotags, thoughts, and XML/HTML
    cleaned = re.sub(r"<tool_response>[\s\S]*?</tool_response>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<thought>[\s\S]*?</thought>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)

    # 2. Strip code blocks and markdown symbols
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*[\s\S]*$", "", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"[*#_~]", "", cleaned)

    # 2b. Strip markdown tables (pipes and table separator rows)
    cleaned = re.sub(r"\|[^\n]+\|", " ", cleaned)
    cleaned = re.sub(r"^[ \t]*[-|: ]{3,}[ \t]*$", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\|", " ", cleaned)

    # 2c. Strip unicode emojis and symbols (Surrogates / Dingbats / Emoticons)
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff]|"
        "[\u2600-\u26ff]|[\u2700-\u27bf]",
        flags=re.UNICODE
    )
    cleaned = emoji_pattern.sub("", cleaned)

    # 3. Normalize calendar dates
    cleaned = re.sub(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})\b", _format_date_slash, cleaned)
    cleaned = re.sub(r"\b(\d{4})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b", _format_date_dash, cleaned)

    # 4. Normalize IP addresses: 1.1.1.1 -> 1 dot 1 dot 1 dot 1
    cleaned = re.sub(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b", r"\1 dot \2 dot \3 dot \4", cleaned)

    # 5. Normalize decimals: 13.2 -> 13 point 2
    cleaned = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", cleaned)

    # 6. Normalize units of measurement
    cleaned = re.sub(r"(\d+)\s*ms\b", r"\1 milliseconds", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+)\s*fps\b", r"\1 frames per second", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+)\s*%", r"\1 percent", cleaned)
    cleaned = re.sub(r"(\d+)\s*GB\b", r"\1 gigabytes", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+)\s*MB\b", r"\1 megabytes", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+)\s*KB\b", r"\1 kilobytes", cleaned, flags=re.IGNORECASE)

    # 7. Normalize month abbreviations with period
    cleaned = re.sub(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.", r"\1", cleaned, flags=re.IGNORECASE)

    # 8. Normalize AM / PM
    cleaned = re.sub(r"\b([Aa])\.[Mm]\.", "AM", cleaned)
    cleaned = re.sub(r"\b([Pp])\.[Mm]\.", "PM", cleaned)

    # 9. Normalize common abbreviations
    cleaned = re.sub(r"\be\.g\.", "for example", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi\.e\.", "that is", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bvs\.", "versus", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\betc\.", "etcetera", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bapprox\.", "approximately", cleaned, flags=re.IGNORECASE)

    # 10. Strip any leftover raw JSON objects from spoken text
    cleaned = re.sub(r"\{\s*\"name\"\s*:[\s\S]*?\}\s*\}", "", cleaned)

    return re.sub(r"\s+", " ", cleaned).strip()


def _ensure_nvidia_libs():
    """Preloads bundled NVIDIA CUDA shared libraries from virtualenv into global symbol table so ONNX Runtime can find cublas."""
    try:
        import ctypes
        from pathlib import Path
        venv_dir = Path(__file__).resolve().parent.parent.parent / ".venv"
        for lib in sorted(venv_dir.glob("lib/python*/site-packages/nvidia/*/lib/*.so*")):
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
    except Exception:
        pass


class StreamingVoiceSynthesizer:
    """Natural streaming Voice Synthesizer supporting Kokoro-82M (CUDA/CPU) and Piper with barge-in cancellation and epoch barriers."""
    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?。！？])\s*(?=[A-Z0-9\u3040-\u30ff\u4e00-\u9fff])|\n+")

    @property
    def engine(self) -> str:
        return self._engine

    @engine.setter
    def engine(self, new_engine: str):
        self._engine = str(new_engine or "").lower()
        if self._engine == "kokoro" and self.kokoro is None:
            self._init_kokoro()
        elif self._engine == "cosyvoice" and not getattr(self, "_cosyvoice_checked", False):
            self._init_cosyvoice()

    def __init__(
        self,
        engine="kokoro",
        model_path="assets/voices/kokoro/kokoro-v1.0.onnx",
        voices_path="assets/voices/kokoro/voices-v1.0.bin",
        voice="af_nicole",
        device_id=0,
        speed=1.05,
        piper_bin="piper",
        target_sink="Adam_Playback_Sink",
        sample_rate=24000,
        cloud_model="gpt-4o-mini-tts",
        cloud_voice="marin",
        api_key="",
        cosyvoice_api_url="http://localhost:50000",
        cosyvoice_model_dir="pretrained_models/CosyVoice2-0.5B",
        config_path="assets/voices/en_US-ryan-high.onnx.json"
    ):
        setup_audio_routing(target_sink)
        self._engine = str(engine or "").lower()
        self.model_path = model_path
        self.voices_path = voices_path
        self.config_path = config_path
        self.voice = voice
        self.device_id = device_id
        self.speed = speed
        self.piper_bin = piper_bin
        self.target_sink = target_sink
        self.sample_rate = 24000 if self._engine == "openai" else sample_rate
        self.cloud_model = cloud_model
        self.cloud_voice = cloud_voice
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.cosyvoice_api_url = cosyvoice_api_url
        self.cosyvoice_model_dir = cosyvoice_model_dir
        self.cosyvoice_model = None
        self._cosyvoice_checked = False
        self.pulse_idx = None
        self._pulse_idx_resolved = False
        self.current_epoch = 0
        self._lock = asyncio.Lock()
        self.active_piper_proc = None
        self.stream = None
        self.is_speaking = False
        self.mic_stream = None
        self.wake_detector = None
        self.earcon = None
        self.kokoro = None
        self.stt = None
        self.current_spoken_text = ""
        self.current_sentence = ""
        self.pending_barge_in_text = None
        self.pending_barge_in_audio = None
        self.pending_barge_in_speaker_verified = False
        self.speaker_verifier = None
        self.speech_history: deque[tuple[float, str]] = deque(maxlen=30)

        if self._engine == "cosyvoice":
            self._init_cosyvoice()
        elif self._engine == "kokoro":
            self._init_kokoro()
        elif self._engine == "silent":
            print("[TTS] Silent mode — responses will appear as desktop notifications only.")
        elif self._engine == "openai":
            print(f"[TTS] OpenAI cloud speech enabled ({self.cloud_model}, voice {self.cloud_voice}).")

    def _init_cosyvoice(self):
        """Check CosyVoice once without delaying assistant startup."""
        import urllib.request
        self._cosyvoice_checked = True
        print(f"[TTS] Checking CosyVoice server at {self.cosyvoice_api_url}...")
        try:
            with urllib.request.urlopen(f"{self.cosyvoice_api_url}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    print("[TTS] CosyVoice server ready.")
                    return
        except Exception:
            pass
        print("[TTS] CosyVoice server unavailable; it will be checked again at synthesis time.")


    def _init_kokoro(self):
        """Initializes Kokoro ONNX on configured device with graceful fallback to CPU."""
        if self.kokoro is not None:
            return
        try:
            _ensure_nvidia_libs()
            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            sess_opt = ort.SessionOptions()
            sess_opt.log_severity_level = 3  # Quiet logs

            available = ort.get_available_providers()
            sess = None

            if self.device_id >= 0 and "CUDAExecutionProvider" in available:
                try:
                    providers = [("CUDAExecutionProvider", {"device_id": self.device_id})]
                    sess = ort.InferenceSession(self.model_path, sess_opt, providers=providers)
                    print(f"[TTS] Kokoro-82M loaded on CUDA device {self.device_id}.")
                except Exception as e:
                    print(f"[TTS] CUDA device {self.device_id} unavailable ({type(e).__name__}), falling back to CPU.")

            if sess is None:
                sess = ort.InferenceSession(self.model_path, sess_opt, providers=["CPUExecutionProvider"])
                print("[TTS] Kokoro-82M loaded on CPU (AVX-optimized).")


            self.kokoro = Kokoro.from_session(sess, self.voices_path)

            # Pre-warmup
            try:
                self.kokoro.create("Ready.", voice=self.voice, speed=self.speed, lang="en-us")
            except Exception:
                pass
            print(f"[TTS] Kokoro-82M initialized successfully with voice '{self.voice}'!")
        except Exception as e:
            print(f"[TTS] Failed to initialize Kokoro: {e}. Falling back to silent notification mode.")
            self.engine = "silent"

    def set_audio_context(self, mic_stream, wake_detector, earcon, stt=None, speaker_verifier=None):
        """Wires mic, wake detector, earcon, and STT engine for natural barge-in abort."""
        self.mic_stream = mic_stream
        self.wake_detector = wake_detector
        self.earcon = earcon
        self.stt = stt
        self.speaker_verifier = speaker_verifier

    def advance_epoch(self):
        """Immediately aborts any ongoing synthesis and speech playback."""
        self.current_epoch += 1
        self.is_speaking = False
        if self.active_piper_proc and self.active_piper_proc.returncode is None:
            try:
                self.active_piper_proc.terminate()
            except Exception:
                pass
        stream = self.stream
        if stream is not None:
            # PortAudio abort may block while another thread is inside write().
            # Never let an interrupt detector or the main event loop wait on it.
            def abort_output():
                try:
                    if stream.active:
                        stream.abort()
                except Exception as e:
                    print(f"[TTS] Could not abort output stream: {e}", flush=True)

            threading.Thread(target=abort_output, name="adam-tts-abort", daemon=True).start()

    async def _play_interruptibly(
        self, playback_fn, *args, epoch: int, utterance_id: str | None = None,
    ):
        """Run blocking audio playback without pinning speech completion after barge-in."""
        playback_span = new_span_id()
        playback_attributes = {
            "speech_role": get_speech_role(),
            "utterance_id": utterance_id,
        }
        emit_event(
            "playback.started", span_id=playback_span, component="playback", status="started",
            attributes=playback_attributes,
        )
        playback_state = {"submitted": False, "error": False}
        playback_task = asyncio.create_task(
            asyncio.to_thread(
                playback_fn, *args, playback_span_id=playback_span, playback_state=playback_state
            )
        )
        while not playback_task.done():
            if epoch != self.current_epoch:
                # The worker owns its stream and will observe the epoch change
                # after its current blocking write returns.
                playback_task.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None
                )
                emit_event(
                    "playback.completed", span_id=playback_span, component="playback", status="cancelled",
                    attributes=playback_attributes,
                )
                return
            await asyncio.wait({playback_task}, timeout=0.05)
        try:
            await playback_task
        except Exception as exc:
            emit_event(
                "playback.completed", span_id=playback_span, component="playback", status="error",
                attributes={**playback_attributes, "error_type": type(exc).__name__},
            )
            raise
        # The worker may finish in the same scheduling window as a barge-in.
        # If the epoch changed before this coroutine resumed, its buffer may
        # have been aborted after submission; do not report confirmed playback.
        playback_status = (
            "cancelled" if epoch != self.current_epoch else
            "error" if playback_state["error"] else
            "ok" if playback_state["submitted"] else
            "skipped"
        )
        emit_event(
            "playback.completed", span_id=playback_span, component="playback", status=playback_status,
            attributes=playback_attributes,
        )

    def speak_now(self, text: str):
        """Non-blocking fire-and-forget speech."""
        asyncio.create_task(self.speak_async(text))

    def _is_speaker_echo(self, transcribed_text: str, max_age: float = 25.0) -> bool:
        """Determines if the transcribed text is an acoustic echo of assistant's own speech."""
        now = time.time()
        ref_parts = []
        if getattr(self, "current_sentence", ""):
            ref_parts.append(self.current_sentence)
        if getattr(self, "current_spoken_text", ""):
            ref_parts.append(self.current_spoken_text)
        for ts, phrase in getattr(self, "speech_history", []):
            if now - ts <= max_age:
                ref_parts.append(phrase)

        ref_text = " ".join(ref_parts).strip()
        if not ref_text:
            return False

        # Contraction expansions
        contractions = {
            r"\bive\b": "i have",
            r"\bim\b": "i am",
            r"\byoure\b": "you are",
            r"\bhes\b": "he is",
            r"\bshes\b": "she is",
            r"\btheyre\b": "they are",
            r"\bwere\b": "we are",
            r"\bthats\b": "that is",
            r"\btheres\b": "there is",
            r"\bwhats\b": "what is",
            r"\bdont\b": "do not",
            r"\bdoesnt\b": "does not",
            r"\bdidnt\b": "did not",
            r"\bcant\b": "cannot",
            r"\bwont\b": "will not",
            r"\bisnt\b": "is not",
            r"\barent\b": "are not",
        }
        clean_ref = ref_text.replace("'", "").lower()
        clean_ref = re.sub(r'(\d+),(\d+)', r'\1\2', clean_ref)
        clean_ref = re.sub(r'\b\d+\b', lambda m: f"{m.group(0)} {_num_to_words(int(m.group(0)))}", clean_ref)
        for k, v in contractions.items():
            clean_ref = re.sub(k, v, clean_ref)

        clean_heard = transcribed_text.replace("'", "").lower()
        clean_heard = re.sub(r'(\d+),(\d+)', r'\1\2', clean_heard)
        clean_heard = re.sub(r'\b\d+\b', lambda m: f"{m.group(0)} {_num_to_words(int(m.group(0)))}", clean_heard)
        for k, v in contractions.items():
            clean_heard = re.sub(k, v, clean_heard)

        heard_words = re.findall(r"\b[a-zA-Z0-9]+\b", clean_heard)
        if not heard_words:
            return True

        meaningful_words = [w for w in heard_words if w not in {"s", "t", "d", "m", "re", "ve", "ll"}]
        ref_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", clean_ref))

        # 1. Direct phrase substring match (either direction)
        heard_phrase = " ".join(heard_words)
        ref_phrase = " ".join(re.findall(r"\b[a-zA-Z0-9]+\b", clean_ref))
        if heard_phrase in ref_phrase:
            return True
        for part in ref_parts:
            part_phrase = " ".join(re.findall(r"\b[a-zA-Z0-9]+\b", part.replace("'", "").lower()))
            if heard_phrase in part_phrase or (len(heard_words) >= 4 and part_phrase in heard_phrase):
                return True

        # 2. Short fragment match (<= 3 words): if assistant is speaking or spoke in last 3s,
        # single words or filler tokens are almost always acoustic bleed/reverb
        if len(meaningful_words) <= 3:
            all_in_ref = all(w in ref_words for w in meaningful_words)
            common_bleed = {
                "yes", "yeah", "yep", "no", "nope", "there", "theres", "like", "so",
                "and", "the", "um", "uh", "okay", "ok", "you", "know", "just", "it",
                "its", "is", "are", "am", "was", "were", "have", "not", "that", "thats"
            }
            if all_in_ref or (self.is_speaking and all(w in common_bleed or w in ref_words for w in meaningful_words)):
                return True

        # 3. Fuzzy & prefix match: count matches using exact, prefix stem, or SequenceMatcher
        match_count = 0
        for hw in meaningful_words:
            if any(
                hw == rw or
                (len(hw) >= 4 and rw.startswith(hw)) or
                (len(rw) >= 4 and hw.startswith(rw)) or
                (len(hw) >= 4 and len(rw) >= 4 and difflib.SequenceMatcher(None, hw, rw).ratio() >= 0.72)
                for rw in ref_words
            ):
                match_count += 1

        match_ratio = match_count / len(meaningful_words) if meaningful_words else 0.0
        return match_ratio >= 0.35

    async def speak_async(self, text: str):
        """Synthesizes and speaks text with clause chunking and active barge-in monitoring."""
        # Silent mode: show notification instead of speaking
        if self.engine == "silent":
            clean_text = clean_speech_text(text)
            if clean_text:
                await asyncio.to_thread(
                    lambda: __import__("subprocess").run(
                        ["notify-send", "-a", "Adam", "-t", "5000", "Adam", clean_text],
                        capture_output=True
                    )
                )
            return

        epoch_at_call = self.current_epoch
        async with self._lock:
            # If an interruption occurred while waiting for the speech lock, discard speech immediately!
            if epoch_at_call != self.current_epoch:
                return

            self.is_speaking = True
            self.pending_barge_in_text = None
            self.pending_barge_in_audio = None
            if self.mic_stream:
                self.mic_stream.is_assistant_speaking = True

            clean_text = clean_speech_text(text)
            if not clean_text:
                self.is_speaking = False
                if self.mic_stream:
                    self.mic_stream.is_assistant_speaking = False
                return

            self.current_spoken_text = clean_text
            utterance_id = new_span_id()

            stop_event = asyncio.Event()
            barge_task = asyncio.create_task(self._barge_in_monitor(stop_event))

            try:
                epoch = self.current_epoch
                sentences = [s.strip() for s in self.SENTENCE_SPLIT_REGEX.split(clean_text) if s.strip()]
                if not sentences:
                    sentences = [clean_text]

                for sentence in sentences:
                    if epoch != self.current_epoch:
                        break
                    self.current_sentence = sentence
                    await self._synthesize_and_play_clause(
                        sentence, epoch, utterance_id=utterance_id,
                    )
                    self.speech_history.append((time.time(), sentence.lower()))
            finally:
                stop_event.set()
                barge_task.cancel()
                try:
                    await barge_task
                except (asyncio.CancelledError, Exception):
                    pass
                self.is_speaking = False
                self.speech_history.append((time.time(), clean_text.lower()))
                self.current_spoken_text = ""
                self.current_sentence = ""
                if self.mic_stream:
                    self.mic_stream.is_assistant_speaking = False
                    self.mic_stream.quench(duration=0.30)

    async def _barge_in_monitor(self, stop_event: asyncio.Event):
        """Monitors microphone for intentional interruption while TTS is active."""
        if not self.mic_stream:
            return

        import numpy as np
        speech_buffer: list[np.ndarray] = []
        silence_samples = 0
        speech_samples = 0
        wake_word_detected = False
        # 350ms minimum speech at mic sample rate
        min_speech_samples = int(0.35 * self.mic_stream.sample_rate)
        # Give the complete wake phrase time to reach ASR before considering a barge-in.
        max_continuous_speech_samples = int(1.20 * self.mic_stream.sample_rate)
        # Ignore tiny pauses so a short hesitation does not split the wake phrase.
        pause_samples_needed = int(0.60 * self.mic_stream.sample_rate)

        while not stop_event.is_set():
            chunk = await asyncio.to_thread(self.mic_stream.get_chunk, timeout=0.03)
            if chunk is None:
                continue

            # 1. Wake word check (if openWakeWord active)
            wake_triggered_on_chunk = False
            if self.wake_detector and not self.wake_detector.is_custom_mode:
                triggered, _ = self.wake_detector.predict(chunk)
                if triggered:
                    # A wake detector hit is only a candidate. Keep playback
                    # alive until STT and, when enrolled, speaker verification
                    # confirm that this is an authorized interruption.
                    wake_triggered_on_chunk = True

            # 2. Silero VAD without acoustic energy override (threshold=0.88)
            is_speech, prob = self.mic_stream.vad.is_speech(chunk, threshold=0.88, use_energy_floor=False)

            if is_speech:
                speech_buffer.append(chunk)
                speech_samples += len(chunk)
                silence_samples = 0
                wake_word_detected = wake_word_detected or wake_triggered_on_chunk
            else:
                if speech_samples > 0:
                    silence_samples += len(chunk)
                    speech_buffer.append(chunk)

            # Check if we should evaluate the speech buffer
            has_paused = (silence_samples >= pause_samples_needed)
            reached_continuous_cap = (speech_samples >= max_continuous_speech_samples)

            if speech_samples >= min_speech_samples and (has_paused or reached_continuous_cap):
                audio_snippet = np.concatenate(speech_buffer)
                duration_s = len(audio_snippet) / self.mic_stream.sample_rate

                if self.stt:
                    # Run fast STT on the short audio snippet
                    text = await asyncio.to_thread(self.stt.transcribe, audio_snippet)
                    text_clean = text.strip()
                    words = re.findall(r"\b[a-zA-Z0-9]+\b", text_clean.lower())

                    if words:
                        if self._is_speaker_echo(text_clean):
                            # Residual acoustic echo of Kokoro speaking; ignore and continue
                            speech_buffer = []
                            speech_samples = 0
                            silence_samples = 0
                        else:
                            matched_wake = False
                            if self.wake_detector:
                                matched, _ = self.wake_detector.match_explicit_wake_word(text_clean)
                                if matched:
                                    matched_wake = True
                            matched_wake = matched_wake or wake_word_detected

                            # Only the configured wake phrase can interrupt
                            # speech or an active tool turn; standalone command
                            # words must not cancel execution.
                            if matched_wake:
                                speaker_verified = False
                                if self.speaker_verifier is not None:
                                    try:
                                        authorized, score = await asyncio.to_thread(
                                            self.speaker_verifier.verify, audio_snippet
                                        )
                                    except Exception as exc:
                                        authorized, score = False, float("nan")
                                        print(
                                            f"[Speaker] Barge-in verification unavailable; keeping playback active ({type(exc).__name__}).",
                                            flush=True,
                                        )
                                    if not authorized:
                                        print(
                                            f"[Speaker] Rejected barge-in (score={score:.3f}); playback continues.",
                                            flush=True,
                                        )
                                        speech_buffer = []
                                        speech_samples = 0
                                        silence_samples = 0
                                        wake_word_detected = False
                                        continue
                                    speaker_verified = True
                                    print(f"[Speaker] Authorized barge-in (score={score:.3f}).", flush=True)
                                print(f"\n[Barge-In] >>> Spoken interruption detected ({duration_s:.2f}s): \"{text_clean}\"! Aborting playback <<<", flush=True)
                                self.pending_barge_in_text = text_clean
                                self.pending_barge_in_audio = audio_snippet.copy()
                                self.pending_barge_in_speaker_verified = speaker_verified
                                self.advance_epoch()
                                if self.earcon:
                                    self.earcon.advance_epoch()
                                    self.earcon.play("interrupt")
                                break
                            else:
                                # Utterances without the wake phrase are ignored during playback.
                                speech_buffer = []
                                speech_samples = 0
                                silence_samples = 0
                                wake_word_detected = False
                    else:
                        speech_buffer = []
                        speech_samples = 0
                        silence_samples = 0
                        wake_word_detected = False

            elif silence_samples >= pause_samples_needed and speech_samples < min_speech_samples:
                # Sound was shorter than min_speech_samples (e.g. 50ms transient click) -> reset
                speech_buffer = []
                speech_samples = 0
                silence_samples = 0

    async def stream_tokens(self, token_async_generator):
        """Streams tokens from LLM, splits at clause boundaries, and synthesizes immediately."""
        # Silent mode: collect all tokens then notify
        if self.engine == "silent":
            full = ""
            async for token in token_async_generator:
                full += token
            clean_text = clean_speech_text(full)
            if clean_text:
                await asyncio.to_thread(
                    lambda: __import__("subprocess").run(
                        ["notify-send", "-a", "Adam", "-t", "5000", "Adam", clean_text],
                        capture_output=True
                    )
                )
            return

        async with self._lock:
            epoch = self.current_epoch
            utterance_id = new_span_id()
            buffer = ""
            async for token in token_async_generator:
                if epoch != self.current_epoch:
                    break
                buffer += token
                parts = self.CLAUSE_SPLIT_REGEX.split(buffer)
                if len(parts) > 1:
                    clause = (parts[0] + parts[1]).strip()
                    buffer = "".join(parts[2:])
                    if clause:
                        await self._synthesize_and_play_clause(
                            clause, epoch, utterance_id=utterance_id,
                        )

            if buffer.strip() and epoch == self.current_epoch:
                await self._synthesize_and_play_clause(
                    buffer.strip(), epoch, utterance_id=utterance_id,
                )


    async def _synthesize_cosyvoice(self, clause: str):
        """Synthesizes speech via the CosyVoice2 local streaming server."""
        import numpy as np
        import urllib.request
        import urllib.parse

        if not self.cosyvoice_api_url:
            return None, self.sample_rate

        try:
            # Use the pre-cached cross-lingual speaker endpoint
            form_data = urllib.parse.urlencode({
                "tts_text": clause,
                "spk_id": "default",
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.cosyvoice_api_url}/inference_sft",
                data=form_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST"
            )

            def _fetch():
                chunks = []
                with urllib.request.urlopen(req, timeout=20.0) as resp:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        chunks.append(chunk)
                if not chunks:
                    return None, 24000
                raw = b"".join(chunks)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                return audio, 24000

            return await asyncio.to_thread(_fetch)
        except Exception as e:
            print(f"[TTS] CosyVoice HTTP error: {e}")

        return None, self.sample_rate

    async def _synthesize_openai(self, clause: str) -> bytes | None:
        """Requests raw 24 kHz PCM from OpenAI's speech endpoint."""
        import aiohttp

        if not self.api_key:
            print("[TTS] OpenAI cloud speech needs an API key (tts.api_key or OPENAI_API_KEY).", flush=True)
            return None
        try:
            timeout = aiohttp.ClientTimeout(total=45)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.cloud_model,
                        "voice": self.cloud_voice,
                        "input": clause,
                        "response_format": "pcm",
                    },
                ) as response:
                    if response.status != 200:
                        detail = await response.text()
                        raise RuntimeError(f"OpenAI speech failed ({response.status}): {detail}")
                    return await response.read()
        except Exception as e:
            print(f"[TTS] OpenAI cloud speech error: {e}", flush=True)
            return None


    async def _synthesize_and_play_clause(
        self, clause: str, epoch: int, *, utterance_id: str | None = None,
    ):
        if epoch != self.current_epoch:
            return
        clause_span = new_span_id()
        emit_event("tts.clause_started", span_id=clause_span, component="tts", status="started")

        if self.engine == "openai":
            pcm = await self._synthesize_openai(clause)
            if pcm and epoch == self.current_epoch:
                emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="ok")
                await self._play_interruptibly(
                    self._play_raw_pcm, pcm, epoch, epoch=epoch, utterance_id=utterance_id,
                )
            elif pcm:
                emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="cancelled")
            elif not pcm:
                emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="error")
            return

        if self.engine == "cosyvoice":
            samples, sr = await self._synthesize_cosyvoice(clause)
            if samples is not None and len(samples) > 0:
                if epoch != self.current_epoch:
                    emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="cancelled")
                    return
                emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="ok")
                await self._play_interruptibly(
                    self._play_float32_audio, samples, sr, epoch,
                    epoch=epoch, utterance_id=utterance_id,
                )
                return

        if self.engine == "kokoro" or (self.engine == "cosyvoice" and self.kokoro):
            if self.kokoro is None and self.engine == "kokoro":
                self._init_kokoro()
            if self.kokoro:
                try:
                    # Detect language: if Japanese kana or kanji present, use 'ja'
                    is_ja = any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in clause)
                    lang = "ja" if is_ja else "en-us"
                    samples, sr = await asyncio.to_thread(
                        self.kokoro.create,
                        clause,
                        voice=self.voice if self.voice in getattr(self.kokoro, "voices", {}) or hasattr(self.kokoro, "get_voice") else "am_onyx",
                        speed=self.speed,
                        lang=lang
                    )
                except Exception as e:
                    print(f"[TTS] Kokoro synthesis error: {e}")
                    emit_event(
                        "tts.clause_synthesized", span_id=clause_span, component="tts", status="error",
                        attributes={"error_type": type(e).__name__},
                    )
                    return

                if epoch != self.current_epoch:
                    emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="cancelled")
                    return
                if len(samples) == 0:
                    emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="error")
                    return

                emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="ok")
                await self._play_interruptibly(
                    self._play_float32_audio, samples, sr, epoch,
                    epoch=epoch, utterance_id=utterance_id,
                )
                return
            elif self.engine == "kokoro":
                emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="error")
                return

        if self.engine != "piper":
            return

        piper_model = self.model_path
        if not (os.path.exists(f"{piper_model}.json") or os.path.exists(re.sub(r"\.onnx$", ".onnx.json", piper_model))):
            candidates = [
                getattr(self, "config_path", "").replace(".json", ""),
                "assets/voices/en_US-ryan-high.onnx",
                "assets/voices/en_US-lessac-medium.onnx",
                "assets/voices/en_US-bryce-medium.onnx",
                "assets/voices/en_US-joe-medium.onnx",
            ]
            for cand in candidates:
                if cand and os.path.exists(cand) and (os.path.exists(f"{cand}.json") or os.path.exists(re.sub(r"\.onnx$", ".onnx.json", cand))):
                    piper_model = cand
                    break
            else:
                print(f"[TTS] Cannot use Piper: '{self.model_path}' has no Piper JSON config and no fallback Piper voice model was found.", flush=True)
                emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="error")
                return

        # Fallback to Piper
        env = os.environ.copy()
        env["PULSE_SINK"] = self.target_sink
        env["PIPEWIRE_NODE"] = self.target_sink

        # Launch Piper to stream raw 16-bit mono PCM to stdout
        try:
            proc = await asyncio.create_subprocess_exec(
                self.piper_bin,
                "-m", piper_model,
                "--output-raw",
                "--length-scale", "0.92",
                "--sentence-silence", "0.05",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
        except Exception as e:
            print(f"[TTS] Could not start Piper ({self.piper_bin}): {e}", flush=True)
            emit_event(
                "tts.clause_synthesized", span_id=clause_span, component="tts", status="error",
                attributes={"error_type": type(e).__name__},
            )
            return
        self.active_piper_proc = proc

        # Send clause to Piper stdin
        try:
            stdout_data, stderr_data = await proc.communicate(input=f"{clause}\n".encode("utf-8"))
        except Exception as e:
            print(f"[TTS] Piper synthesis failed: {e}", flush=True)
            emit_event(
                "tts.clause_synthesized", span_id=clause_span, component="tts", status="error",
                attributes={"error_type": type(e).__name__},
            )
            return

        if proc.returncode != 0:
            detail = stderr_data.decode("utf-8", errors="replace").strip()
            print(f"[TTS] Piper exited with status {proc.returncode}: {detail or 'no error details'}", flush=True)
            emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="error")
            return
        if epoch != self.current_epoch or not stdout_data:
            emit_event(
                "tts.clause_synthesized", span_id=clause_span, component="tts",
                status="cancelled" if epoch != self.current_epoch else "error",
            )
            if not stdout_data and epoch == self.current_epoch:
                print("[TTS] Piper produced no audio. Check that tts.model_path is a valid Piper voice model (.onnx).", flush=True)
            return

        # Play audio buffer through sounddevice
        emit_event("tts.clause_synthesized", span_id=clause_span, component="tts", status="ok")
        await self._play_interruptibly(
            self._play_raw_pcm, stdout_data, epoch, epoch=epoch, utterance_id=utterance_id,
        )

    def _play_float32_audio(
        self, audio_data, sample_rate: int, epoch: int, *, playback_span_id: str | None = None,
        playback_state: dict | None = None,
    ):
        if epoch != self.current_epoch:
            return

        import sounddevice as sd
        import numpy as np
        self._resolve_pulse_device()
        # Pad 200ms silence so sounddevice ring buffer fully drains before stream closes
        tail_samples = int(sample_rate * 0.06)
        audio_data = np.concatenate([audio_data, np.zeros(tail_samples, dtype=np.float32)])

        try:
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=self.pulse_idx,
                blocksize=2048,
                latency=0.08,
            ) as stream:
                self.stream = stream
                chunk_size = 2048
                first_buffer = True
                for i in range(0, len(audio_data), chunk_size):
                    if epoch != self.current_epoch:
                        stream.abort()
                        return
                    if first_buffer:
                        first_buffer = False
                        stream.write(audio_data[i:i + chunk_size])
                        if playback_state is not None:
                            playback_state["submitted"] = True
                        emit_event(
                            "playback.buffer_submitted", span_id=playback_span_id,
                            component="playback", status="ok",
                        )
                    else:
                        stream.write(audio_data[i:i + chunk_size])
        except Exception as e:
            if playback_state is not None:
                playback_state["error"] = True
            if epoch == self.current_epoch:
                print(f"[TTS] Playback error: {e}")


    def _play_raw_pcm(
        self, pcm_bytes: bytes, epoch: int, *, playback_span_id: str | None = None,
        playback_state: dict | None = None,
    ):
        if epoch != self.current_epoch:
            return

        import sounddevice as sd
        import numpy as np
        self._resolve_pulse_device()
        audio_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # Pad 200ms silence so ring buffer fully drains before stream closes
        tail_samples = int(self.sample_rate * 0.20)
        audio_data = np.concatenate([audio_data, np.zeros(tail_samples, dtype=np.float32)])

        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.pulse_idx,
                blocksize=2048,
                latency=0.08,
            ) as stream:
                self.stream = stream
                chunk_size = 2048
                first_buffer = True
                for i in range(0, len(audio_data), chunk_size):
                    if epoch != self.current_epoch:
                        stream.abort()
                        return
                    if first_buffer:
                        first_buffer = False
                        stream.write(audio_data[i:i + chunk_size])
                        if playback_state is not None:
                            playback_state["submitted"] = True
                        emit_event(
                            "playback.buffer_submitted", span_id=playback_span_id,
                            component="playback", status="ok",
                        )
                    else:
                        stream.write(audio_data[i:i + chunk_size])
        except Exception as e:
            if playback_state is not None:
                playback_state["error"] = True
            if epoch == self.current_epoch:
                print(f"[TTS] Playback error: {e}")

    def _resolve_pulse_device(self):
        """Defer PortAudio device enumeration until audio is actually played."""
        if not self._pulse_idx_resolved:
            self.pulse_idx = resolve_pulse_device_index()
            self._pulse_idx_resolved = True
