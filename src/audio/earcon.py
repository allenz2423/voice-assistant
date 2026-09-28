import os
import queue
import threading
import sounddevice as sd
import numpy as np

def setup_audio_routing(target_sink: str = ""):
    """Injects routing variables for both ALSA PipeWire-plugin and PulseAudio layers if explicitly set."""
    if target_sink and target_sink not in ("Shin_Playback_Sink", "default"):
        os.environ["PIPEWIRE_NODE"] = target_sink
        os.environ["PULSE_SINK"] = target_sink

def resolve_pulse_device_index() -> int | None:
    """Finds the integer index for the PortAudio 'pulse' ALSA bridge."""
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev["name"] == "pulse" and dev["max_output_channels"] > 0:
                return idx
    except Exception:
        pass
    return None

class RobustEarconEngine:
    """Hardened non-blocking earcon engine with PortAudio recovery and epoch barriers."""
    def __init__(self, target_sink="Shin_Playback_Sink", sample_rate=48000):
        setup_audio_routing(target_sink)
        self.sample_rate = sample_rate
        self.pulse_idx = resolve_pulse_device_index()
        self.queue = queue.Queue(maxsize=16)
        self.current_epoch = 0
        self.running = True
        self.stream = None
        self._epoch_lock = threading.Lock()
        self._stream_lock = threading.Lock()

        # Pre-synthesize earcon tones in RAM (float32)
        self.chimes = {
            "wake": self._generate_tone(880.0, 0.05),       # 50ms A5 beep
            "captured": self._generate_tone(1760.0, 0.03),  # 30ms A6 blip (got command)
            "done": self._generate_tone(587.33, 0.08),      # 80ms D5 soft chime
            "interrupt": self._generate_tone(330.0, 0.03),  # 30ms E4 low click
            "cancel": self._generate_tone(220.0, 0.06),     # 60ms A3 low soft tone
        }

        self.worker = threading.Thread(target=self._stream_loop, daemon=True)
        self.worker.start()

    def advance_epoch(self):
        """Invalidates pending audio and aborts current stream with mutual exclusion."""
        with self._epoch_lock:
            self.current_epoch += 1
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

        with self._stream_lock:
            if self.stream and self.stream.active:
                try:
                    self.stream.abort()
                except Exception:
                    pass

    def play(self, tone_name: str):
        """Non-blocking, thread-safe audio dispatch with active epoch timestamp."""
        with self._epoch_lock:
            epoch = self.current_epoch
        try:
            self.queue.put_nowait((tone_name, epoch))
        except queue.Full:
            pass

    def _render_tone(self, tone_name: str) -> np.ndarray:
        return self.chimes.get(tone_name, self.chimes["wake"])

    def _generate_tone(self, freq: float, duration: float) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = 0.15 * np.sin(2 * np.pi * freq * t)
        fade = int(self.sample_rate * 0.005)
        tone[-fade:] *= np.linspace(1, 0, fade)
        return tone.astype(np.float32)

    def _stream_loop(self):
        # Bind to pulse ALSA device to guarantee PULSE_SINK routing
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.pulse_idx
            ) as stream:
                self.stream = stream
                while self.running:
                    try:
                        tone_name, epoch = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    with self._epoch_lock:
                        if epoch != self.current_epoch:
                            continue  # Dropped by epoch barrier

                    pcm_data = self._render_tone(tone_name)

                    with self._stream_lock:
                        # Essential fix: PortAudio streams remain stopped after abort()
                        if stream.stopped:
                            try:
                                stream.start()
                            except Exception:
                                continue
                        try:
                            stream.write(pcm_data)
                        except sd.PortAudioError:
                            pass
        except Exception as e:
            print(f"[EarconEngine] Stream error: {e}")

    def close(self):
        self.running = False
        self.advance_epoch()
