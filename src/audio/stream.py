import os
import time
import queue
import threading
from collections import deque
import numpy as np
from src.audio.vad import SileroVAD
from src.audio.earcon import resolve_pulse_device_index
from src.telemetry.events import emit_event, new_span_id


def _take_audio_tail(chunks: list[np.ndarray], sample_count: int) -> list[np.ndarray]:
    selected = []
    remaining = sample_count
    for chunk in reversed(chunks):
        if remaining <= 0:
            break
        take = min(len(chunk), remaining)
        selected.append(chunk[-take:])
        remaining -= take
    return list(reversed(selected))


def setup_mic_routing(target_source=""):
    if target_source and target_source not in ("Adam_Clean_Mic", "default"):
        os.environ["PULSE_SOURCE"] = target_source

class ReferenceAudioMonitor:
    """Monitors @DEFAULT_SINK@.monitor (speaker output) in real-time.
    Provides speaker energy telemetry and normalized cross-correlation
    to reject acoustic bleed from YouTube, music, movies, or games."""
    def __init__(self, sample_rate: int = 16000, buffer_seconds: float = 6.0):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_idx = 0
        self.lock = threading.Lock()
        self.running = False
        self.stream = None
        self.pulse_idx = None
        self.speaker_rms = 0.0
        self.is_active = False

    def start(self):
        if self.running:
            return
        if self.pulse_idx is None:
            self.pulse_idx = resolve_pulse_device_index()
        if self.pulse_idx is None:
            return
        try:
            import sounddevice as sd
            orig_pulse_source = os.environ.get("PULSE_SOURCE")
            os.environ["PULSE_SOURCE"] = "@DEFAULT_SINK@.monitor"
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=512,
                    device=self.pulse_idx,
                    callback=self._audio_callback
                )
                self.running = True
                self.stream.start()
            finally:
                if orig_pulse_source is not None:
                    os.environ["PULSE_SOURCE"] = orig_pulse_source
                else:
                    os.environ.pop("PULSE_SOURCE", None)
        except Exception as e:
            print(f"[ReferenceMonitor] Notice: Default sink monitor unavailable ({e}). Continuing without reference monitor.")

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0]
        rms = float(np.sqrt(np.mean(chunk**2)))
        self.speaker_rms = rms
        self.is_active = rms >= 0.003

        with self.lock:
            n = len(chunk)
            if self.write_idx + n <= self.buffer_size:
                self.buffer[self.write_idx:self.write_idx + n] = chunk
                self.write_idx = (self.write_idx + n) % self.buffer_size
            else:
                part1 = self.buffer_size - self.write_idx
                self.buffer[self.write_idx:] = chunk[:part1]
                part2 = n - part1
                self.buffer[:part2] = chunk[part1:]
                self.write_idx = part2

    def get_recent_reference(self, duration_s: float) -> np.ndarray:
        """Returns the most recent duration_s of speaker playback in chronological order."""
        with self.lock:
            n_samples = min(self.buffer_size, int(duration_s * self.sample_rate))
            if n_samples <= 0:
                return np.zeros(0, dtype=np.float32)
            idx = self.write_idx
            if idx >= n_samples:
                return self.buffer[idx - n_samples:idx].copy()
            else:
                tail = self.buffer[self.buffer_size - (n_samples - idx):]
                head = self.buffer[:idx]
                return np.concatenate([tail, head])

    def is_speaker_echo(self, mic_audio: np.ndarray, threshold: float = 0.35) -> tuple[bool, float]:
        """Calculates normalized cross-correlation between microphone utterance and recent speaker reference.
        Returns (True, max_corr) if microphone audio is predominantly an acoustic reflection of speaker playback."""
        if not self.running or len(mic_audio) < int(0.25 * self.sample_rate):
            return False, 0.0

        ref_audio = self.get_recent_reference(len(mic_audio) / self.sample_rate + 0.3)
        if len(ref_audio) < len(mic_audio):
            return False, 0.0

        ref_rms = float(np.sqrt(np.mean(ref_audio**2)))
        if ref_rms < 0.002:
            return False, 0.0

        mic_rms = float(np.sqrt(np.mean(mic_audio**2)))
        if mic_rms < 1e-5:
            return False, 0.0

        try:
            import scipy.signal
            mic_8k = mic_audio[::2]
            ref_8k = ref_audio[::2]

            if len(ref_8k) < len(mic_8k):
                return False, 0.0

            mic_norm = (mic_8k - np.mean(mic_8k)) / (np.std(mic_8k) + 1e-6)
            ref_norm = (ref_8k - np.mean(ref_8k)) / (np.std(ref_8k) + 1e-6)

            corr = scipy.signal.correlate(ref_norm, mic_norm, mode='valid', method='fft')
            max_corr = float(np.max(corr)) / len(mic_8k)
            return max_corr >= threshold, max_corr
        except Exception:
            return False, 0.0

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


class AudioStreamManager:
    """Continuous audio capture from default mic at 16kHz mono with dynamic VAD and speaker reference monitor."""
    def __init__(self, target_source="Adam_Clean_Mic", sample_rate=16000, chunk_size=512):
        setup_mic_routing(target_source)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size  # 512 samples = 32ms at 16kHz
        self.pulse_idx = None
        self.vad = SileroVAD(sample_rate=sample_rate)
        self.audio_queue = queue.Queue(maxsize=100)
        self.running = False
        self.stream = None
        self._thread = None
        self.is_assistant_speaking = False
        self.quench_until = 0.0
        self.ref_monitor = ReferenceAudioMonitor(sample_rate=self.sample_rate)
        self._recent_audio_seconds = 5.0
        self._recent_audio_chunks = deque()
        self._recent_audio_samples = 0
        self._recent_audio_lock = threading.Lock()
        self._audio_tap = None

    def set_audio_tap(self, callback=None) -> None:
        """Register a nonblocking observer for every captured mic chunk."""
        self._audio_tap = callback

    def start(self):
        if self.running:
            return
        import sounddevice as sd
        self.pulse_idx = resolve_pulse_device_index()
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            device=self.pulse_idx,
            callback=self._audio_callback
        )
        self.stream.start()
        if self.ref_monitor:
            self.ref_monitor.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.running:
            return
        chunk = indata[:, 0].copy()
        tap = getattr(self, "_audio_tap", None)
        if tap is not None:
            try:
                tap(chunk)
            except Exception:
                pass
        if time.time() < self.quench_until:
            return
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(chunk)
            except Exception:
                pass

    def get_chunk(self, timeout=0.1) -> np.ndarray | None:
        if time.time() < self.quench_until:
            self.flush()
            return None
        try:
            chunk = self.audio_queue.get(timeout=timeout)
            self._remember_recent_audio(chunk)
            return chunk
        except queue.Empty:
            return None

    def _remember_recent_audio(self, chunk: np.ndarray) -> None:
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1).copy()
        if audio.size == 0:
            return
        limit = int(self._recent_audio_seconds * self.sample_rate)
        with self._recent_audio_lock:
            self._recent_audio_chunks.append(audio)
            self._recent_audio_samples += len(audio)
            excess = self._recent_audio_samples - limit
            while excess > 0 and self._recent_audio_chunks:
                oldest = self._recent_audio_chunks[0]
                if len(oldest) <= excess:
                    self._recent_audio_chunks.popleft()
                    self._recent_audio_samples -= len(oldest)
                    excess -= len(oldest)
                else:
                    self._recent_audio_chunks[0] = oldest[excess:].copy()
                    self._recent_audio_samples -= excess
                    excess = 0

    def get_recent_audio(self, duration_s: float = 5.0, exclude_latest_samples: int = 0) -> np.ndarray:
        """Return recently consumed mic audio, optionally excluding the current chunk."""
        requested = min(self._recent_audio_seconds, max(0.0, duration_s))
        requested_samples = int(requested * self.sample_rate)
        with self._recent_audio_lock:
            chunks = list(self._recent_audio_chunks)
        if exclude_latest_samples > 0:
            remaining = max(0, sum(map(len, chunks)) - int(exclude_latest_samples))
            selected = []
            for chunk in chunks:
                if remaining <= 0:
                    break
                take = min(len(chunk), remaining)
                selected.append(chunk[:take])
                remaining -= take
            chunks = selected
        available = sum(map(len, chunks))
        take = min(requested_samples, available)
        if take <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(_take_audio_tail(chunks, take))

    def quench(self, duration: float = 0.45):
        """Suppresses incoming audio for duration seconds to let room reflections and speaker audio decay."""
        self.quench_until = max(self.quench_until, time.time() + duration)
        self.flush()

    def flush(self):
        """Flushes buffered audio chunks to prevent stale audio playback across state transitions."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def record_utterance(
        self,
        silence_duration=1.25,
        max_duration=60.0,
        idle_threshold=0.25,
        speaking_threshold=0.85,
        initial_chunk=None,
        on_partial_audio=None,
        partial_interval_s=0.50,
    ) -> np.ndarray:
        """Records speech chunks until silence is sustained for silence_duration.
        Supports streaming partial audio callbacks to dynamically adapt silence_duration
        via semantic endpointing and speculative tool pre-flight."""
        while time.time() < self.quench_until:
            time.sleep(0.02)
        self.flush()

        threshold = speaking_threshold if self.is_assistant_speaking else idle_threshold
        dynamic_floor = max(0.0035, min(0.022, self.ref_monitor.speaker_rms * 0.40)) if (self.ref_monitor and self.ref_monitor.is_active) else 0.0025

        frames = []
        speech_started = False
        speech_start_emitted = False
        endpoint_reason = "max_duration"
        silence_start = None
        start_time = time.time()
        last_partial_time = start_time

        self.vad.reset()

        if initial_chunk is not None and len(initial_chunk) > 0:
            frames.append(initial_chunk)
            speech_started = True
            speech_start_emitted = True
            emit_event("audio.speech_started", component="audio", status="ok", attributes={"source": "initial_chunk"})

        while time.time() - start_time < max_duration:
            chunk = self.get_chunk(timeout=0.1)
            if chunk is None:
                continue

            frames.append(chunk)
            is_speech, prob = self.vad.is_speech(
                chunk,
                threshold=threshold,
                use_energy_floor=not self.is_assistant_speaking,
                energy_floor=dynamic_floor
            )

            if is_speech:
                speech_started = True
                if not speech_start_emitted:
                    speech_start_emitted = True
                    emit_event("audio.speech_started", component="audio", status="ok", attributes={"source": "vad"})
                silence_start = None
            else:
                if speech_started:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= silence_duration:
                        # User stopped speaking
                        endpoint_reason = "silence"
                        break
                else:
                    # Keep rolling window of pre-speech frames
                    if len(frames) > int(0.5 * self.sample_rate / self.chunk_size):
                        frames.pop(0)

            if speech_started and on_partial_audio is not None:
                now = time.time()
                if now - last_partial_time >= partial_interval_s:
                    last_partial_time = now
                    # Only invoke if at least 0.25s of speech accumulated
                    if len(frames) * self.chunk_size >= int(0.25 * self.sample_rate):
                        callback_span = new_span_id()
                        emit_event("audio.partial_callback_started", span_id=callback_span, component="audio")
                        try:
                            partial_arr = np.concatenate(frames)
                            new_silence = on_partial_audio(partial_arr)
                            if new_silence is not None and isinstance(new_silence, (int, float)):
                                silence_duration = float(new_silence)
                                emit_event(
                                    "audio.endpoint_candidate_applied", span_id=callback_span,
                                    component="endpoint", status="ok",
                                    attributes={"target_silence_s": silence_duration},
                                )
                            emit_event(
                                "audio.partial_callback_completed", span_id=callback_span,
                                component="audio", status="ok",
                            )
                        except Exception as exc:
                            emit_event(
                                "audio.partial_callback_completed", span_id=callback_span,
                                component="audio", status="error",
                                attributes={"error_type": type(exc).__name__},
                            )
                            pass

        emit_event(
            "audio.endpoint_decided", component="endpoint", status="ok",
            attributes={
                "reason": endpoint_reason,
                "effective_silence_s": float(silence_duration),
                "speech_started": bool(speech_started),
            },
        )

        if frames:
            audio_res = np.concatenate(frames)
            if self.ref_monitor:
                is_echo, corr_score = self.ref_monitor.is_speaker_echo(audio_res, threshold=0.35)
                if is_echo:
                    print(f"\n[Mic] Discarding speaker acoustic reflection ({corr_score*100:.1f}% match to desktop audio).", flush=True)
                    return np.array([], dtype=np.float32)
            return audio_res
        return np.array([], dtype=np.float32)

    def stop(self):
        self.running = False
        if self.ref_monitor:
            self.ref_monitor.stop()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
