# Shin: Voice-Activated Autonomous Terminal Agent
## Hardened Production Architecture & Implementation Plan

Shin is a 100% non-blocking, interruptible, hybrid voice terminal agent engineered for Linux (Arch/PipeWire/Dual NVIDIA GPUs). It combines autonomous terminal execution, background job supervision, acoustic echo cancellation, and low-latency audio feedback.

---

## 1. System Specifications & Core Directives

1. **Non-Blocking & Full Barge-In (Interruptibility)**:
   - Full-duplex audio stream via PipeWire WebRTC AEC (`Shin_Clean_Mic`).
   - Microphone pipeline free-runs across **all** states, including `ASSISTANT_SPEAKING`.
   - Atomic **Generation Epoch Barriers** eliminate race conditions during speech barge-in.
   - PortAudio lifecycle recovery prevents thread crashes after stream aborts.
2. **Background Job Supervision & Proactive Wake-Up**:
   - Long-running commands (transcoding, builds, downloads) run inside isolated **Bubblewrap (`bwrap`)** containers with private PID namespaces (`--unshare-pid`) and merged-/usr relative symlinks.
   - GPU device nodes (`/dev/dri`, `/dev/nvidia*`) and POSIX shared memory (`/dev/shm`) are explicitly mounted.
   - Output logs stream to persistent NVMe disk storage (`~/.local/state/shin/jobs/`), completely protecting `/tmp` `tmpfs` RAM-disk from exhaustion.
   - Background tasks report to a **Priority Audio Arbiter** that queues alerts and speaks **only** during `IDLE_LISTENING`, preventing context poisoning during user speech or confirmations.
3. **Flip-of-a-Switch Local vs. Cloud LLM**:
   - A single configuration switch (`provider: "local" | "cloud"`) toggles between local Ollama (`qwen2.5-coder:7b` / `gemma4:e4b`) and cloud providers (Groq, Gemini, Anthropic).
   - Unified canonical Pydantic tool schemas with full tripartite dialect compilation (OpenAI `tool_calls`, Gemini `functionDeclarations`, and Anthropic `input_schema` / `tool_use` / `tool_result`).
4. **Instant Audible Earcons & Rapid Feedback**:
   - Sub-300ms total perceived turnaround.
   - Dedicated non-blocking audio engine feeding pre-synthesized PCM earcons (<0.05ms thread handoff, zero asyncio event loop stalls).
   - Clause-level streaming Piper TTS synthesizes and plays speech in chunks as tokens arrive.

---

## 2. Host Hardware Topology & Resource Allocation

### Hardware Inventory
* **GPU 0 (GTX 1080 Ti, GP102, CC 6.1, 11 GB VRAM)**:
  - *Compute Reality:* Pascal architecture lacks Tensor Cores. FP16 runs at 1/64 FP32 speed. CUDA 13.x drops CC 6.1 support.
  - *Execution Policy:* `faster-whisper` runs using PyPI CUDA 12 wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12==9.*`) with `compute_type="int8_float32"`.
  - *VRAM Guardrails:* Ollama context capped at 4k tokens (`num_ctx 4096`) to prevent KV-cache expansion from causing CUDA OOM alongside Whisper.
* **GPU 1 (RTX 3070, GA104, CC 8.6, 8 GB VRAM)**:
  - *Workload:* Dedicated to desktop display (Hyprland ~191 MiB, Noctalia ~148 MiB), with **~7.2 GB of VRAM completely free**.
  - *Encoding Reality:* 
    - Ampere lacks hardware AV1 encoding.
    - Arch rolling `ffmpeg 9.0.2` requires NVENC API 13.1 (NVIDIA driver $\ge 610.00$), while installed driver is 580.178.04 (API 13.0). Direct `hevc_nvenc` calls currently fail with error 218 until the driver is updated.
    - *Policy:* Multi-threaded CPU **`libsvtav1`** runs at ~4x realtime speed and serves as the primary transcode engine inside the sandbox, with `hevc_nvenc` auto-enabled once the driver is updated to 610+.
* **Audio Subsystem**:
  - *Playback Sink:* System default or configured audio sink (`@DEFAULT_AUDIO_SINK@`).
  - *Capture Source:* System default or configured microphone (`@DEFAULT_AUDIO_SOURCE@`).
  - *Virtual Routing:* Both `PIPEWIRE_NODE` and `PULSE_SINK` environment variables are set to `Shin_Playback_Sink`, and PortAudio binds to the `pulse` ALSA bridge device index.
  - *Physical Volume Knob Policy:* Set analog potentiometer to a fixed unity gain position (~50% line-out). All listening volume adjustments are handled digitally (via `wpctl set-volume @DEFAULT_AUDIO_SINK@ ...`) so digital reference attenuation matches acoustic playback exactly.

---

## 3. Red-Team Vulnerability Analysis & Hardened Countermeasures

### 3.1. Audio & AEC Failures: The PortAudio Routing Trap & Thread Death
* **The Vulnerability**:
  1. Virtual PipeWire sinks do NOT appear in PortAudio's ALSA device enumeration. Querying `Shin_Playback_Sink` returns `None`, causing playback to default to physical ALSA output, completely bypassing WebRTC AEC and re-triggering the self-interruption loop.
  2. Calling `stream.abort()` during barge-in sets the PortAudio stream to `stopped`. The next `stream.write()` raises `PortAudioError: Stream is stopped [-9983]`, crashing the earcon worker thread permanently.
  3. PipeWire 1.6.9 ignores legacy PulseAudio keys; WebRTC parameters require the `webrtc.` prefix.
* **The Fix**:
  - Target PortAudio's `pulse` device index directly and inject both `PIPEWIRE_NODE="Shin_Playback_Sink"` and `PULSE_SINK="Shin_Playback_Sink"`.
  - In the earcon worker thread, protect stream operations with `_stream_lock` and call `if stream.stopped: stream.start()` before `stream.write()`.
  - Configure PipeWire AEC with verified `webrtc.*` keys.

---

### 3.2. Microsecond Race Conditions & Generation Epoch Barriers
* **The Vulnerability**:
  If the user says *"Stop!"* in the exact microsecond between the LLM finishing its generation and the audio worker thread popping the first TTS PCM chunk:
  - The state machine cancels the LLM task (which already completed).
  - The decoupled audio worker thread plays the old sentence anyway while the user is trying to speak.
* **The Fix**:
  - **Atomic Generation Epochs**: Every interruption or state reset atomically increments `current_epoch`.
  - All audio frames in queues and pipelines are tagged with their creation `epoch_id`.
  - If `frame.epoch_id != current_epoch`, the audio worker instantly drops the frame and aborts playback.

---

### 3.3. Bubblewrap Arch Merged-/usr & Process Isolation
* **The Vulnerability**:
  1. Binding `/bin`, `/lib`, `/lib64` as directories on Arch breaks merged-/usr invariants and omits `/sbin`.
  2. Omitting `--unshare-pid` leaves `/proc` visible to sandboxed tasks and fails to clean up grandchild processes on parent exit.
  3. Missing `--tmpfs /dev/shm` breaks POSIX shared memory required by CUDA and multiprocessing.
* **The Fix**:
  - Use canonical Arch symlink arguments: `--symlink usr/lib /lib`, `--symlink usr/lib /lib64`, `--symlink usr/bin /bin`, `--symlink usr/bin /sbin`.
  - Add `--unshare-pid` so `bwrap` acts as PID 1 and reaps all child processes atomically upon exit.
  - Mount `--tmpfs /dev/shm`, bind `~/Downloads`, and use `--dev-bind-try` for all dynamic `/dev/nvidia*` nodes.

---

### 3.4. Priority Audio Arbiter & State Gating
* **The Vulnerability**:
  Background task completion alerts blurt out over active user speech or during `AWAITING_CONFIRMATION`, leading to context corruption and unintended affirmative execution.
* **The Fix**:
  - Implement a **Gated Notification Queue** inside `PriorityAudioArbiter`.
  - When the system is in `USER_SPEAKING`, `PROCESSING_REACT`, `ASSISTANT_SPEAKING`, or `AWAITING_CONFIRMATION`, notifications are held in a min-heap priority queue.
  - Only when the state returns to `IDLE_LISTENING` are alerts popped, chimed, and spoken.

---

### 3.5. Confirmation Safety: Word Boundaries & Active Expiry Watchdog
* **The Vulnerability**:
  1. Checking `"no" in text` matches `"I don't know"`, `"now hold on"`, or `"annoyed"`, triggering unintended denials or safety aborts.
  2. Checking `time.time() > deadline` only when a transcript arrives fails if the user says nothing at all. The state machine locks in `AWAITING_CONFIRMATION` forever.
* **The Fix**:
  - **Regex Word Boundaries**: Enforce `re.search(r'\b(no|cancel|stop|abort)\b', text, re.I)` and `re.search(r'\b(yes|yeah|confirm|proceed|go ahead)\b', text, re.I)`.
  - **Active Asyncio Watchdog Timer**: `request_confirmation()` spawns an explicit `asyncio.create_task(self._expiry_watchdog(10.0))` that fires after 10 seconds of silence, cancelling the action and returning state to `IDLE_LISTENING`.

---

## 4. Hardened Production Code Components

### A. Non-Blocking Audio Engine with Epoch Barriers (`src/audio/earcon.py`)

```python
import os
import queue
import threading
import sounddevice as sd
import numpy as np

def setup_audio_routing(target_sink="Shin_Playback_Sink"):
    """Injects routing variables for both ALSA PipeWire-plugin and PulseAudio layers."""
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

        self.chimes = {
            "wake": self._generate_tone(880, 0.05),        # 50ms A5 beep
            "captured": self._generate_tone(1760, 0.03),    # 30ms A6 blip (got command)
            "done": self._generate_tone(587.33, 0.08),     # 80ms D5 soft chime
            "interrupt": self._generate_tone(330, 0.03),   # 30ms E4 low click
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
                        continue  # Dropped due to epoch barrier

                pcm_data = self._render_tone(tone_name)

                with self._stream_lock:
                    if stream.stopped:
                        try:
                            stream.start()
                        except Exception:
                            continue
                    try:
                        stream.write(pcm_data)
                    except sd.PortAudioError:
                        pass
```

---

### B. Sandboxed Background Job Supervisor (`src/execution/supervisor.py`)

```python
import asyncio
import os
import signal
import glob
from pathlib import Path

class HardenedJobSupervisor:
    """Spawns jobs in Bubblewrap sandboxes with full GPU access, path bindings, and bwrap parent death-signals."""
    def __init__(self, log_dir="~/.local/state/shin/jobs", workspace="~/workspace"):
        self.log_dir = Path(log_dir).expanduser()
        self.workspace = Path(workspace).expanduser()
        self.downloads = Path("~/Downloads").expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.active_jobs: dict[str, dict] = {}

    def _get_gpu_device_args(self) -> list[str]:
        args = []
        if os.path.exists("/dev/dri"):
            args.extend(["--dev-bind-try", "/dev/dri", "/dev/dri"])
        for dev_path in glob.glob("/dev/nvidia*"):
            args.extend(["--dev-bind-try", dev_path, dev_path])
        return args

    async def start_sandboxed_job(self, job_id: str, raw_cmd: list[str], arbiter) -> int:
        log_file = self.log_dir / f"{job_id}.log"
        out_fp = await asyncio.to_thread(open, log_file, "wb")

        # Canonical Arch Linux merged-/usr layout and isolated PID namespace
        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib", "/lib64",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/bin", "/sbin",
            "--ro-bind", "/etc", "/etc",
            "--ro-bind", "/sys", "/sys",                            # Needed for NVML / GPU topology
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/dev/shm",                                  # CUDA/POSIX shared memory
            "--tmpfs", "/tmp",
            "--unshare-pid",                                        # bwrap acts as PID 1; auto-reaps all descendants
            "--ro-bind", str(self.downloads), str(self.downloads),  # Access downloads
            "--bind", str(self.workspace), str(self.workspace),      # Output directory
            "--chdir", str(self.workspace),
            "--die-with-parent",                                    # Kills sandbox if Shin exits
        ] + self._get_gpu_device_args() + ["--"] + raw_cmd

        proc = await asyncio.create_subprocess_exec(
            *bwrap_cmd,
            stdout=out_fp,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True
        )

        self.active_jobs[job_id] = {
            "proc": proc,
            "pgid": proc.pid,
            "out_fp": out_fp,
            "log_file": log_file
        }

        asyncio.create_task(self._watch_job(job_id, proc, out_fp, arbiter))
        return proc.pid

    async def kill_job(self, job_id: str):
        """Kills the process group cleanly."""
        job = self.active_jobs.get(job_id)
        if not job:
            return
        pgid = job["pgid"]
        try:
            os.killpg(pgid, signal.SIGTERM)
            await asyncio.sleep(1.5)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _watch_job(self, job_id: str, proc: asyncio.subprocess.Process, out_fp, arbiter):
        returncode = await proc.wait()
        await asyncio.to_thread(out_fp.close)
        self.active_jobs.pop(job_id, None)

        status = "succeeded" if returncode == 0 else f"failed with code {returncode}"
        await arbiter.enqueue_notification(
            priority=1,
            message=f"Job {job_id} {status}."
        )
```

---

### C. Priority Audio Arbiter with Gated Notifications (`src/arbiter/arbiter.py`)

```python
import asyncio
import heapq
from enum import Enum

class SystemState(Enum):
    IDLE_LISTENING = "IDLE_LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING_REACT = "PROCESSING_REACT"
    ASSISTANT_SPEAKING = "ASSISTANT_SPEAKING"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"

class PriorityAudioArbiter:
    """Manages audio access and notifications without conversational collision."""
    def __init__(self, tts_engine, earcon_engine):
        self.tts = tts_engine
        self.earcon = earcon_engine
        self.current_state = SystemState.IDLE_LISTENING
        self.notification_queue: list[tuple[int, str]] = []  # Min-heap (priority, message)
        self._state_lock = asyncio.Lock()

    async def set_state(self, new_state: str):
        async with self._state_lock:
            self.current_state = SystemState(new_state)
            if self.current_state == SystemState.IDLE_LISTENING:
                await self._drain_notifications_locked()

    async def enqueue_notification(self, priority: int, message: str):
        async with self._state_lock:
            heapq.heappush(self.notification_queue, (priority, message))
            if self.current_state == SystemState.IDLE_LISTENING:
                await self._drain_notifications_locked()

    async def _drain_notifications_locked(self):
        """Delivers pending notifications only during IDLE_LISTENING."""
        while self.notification_queue and self.current_state == SystemState.IDLE_LISTENING:
            priority, message = heapq.heappop(self.notification_queue)
            self.current_state = SystemState.ASSISTANT_SPEAKING
            self.earcon.play("done")
            await asyncio.sleep(0.1)
            await self.tts.speak_async(message)
            self.current_state = SystemState.IDLE_LISTENING
```

---

### D. Tri-State Confirmation Machine with Active Watchdog (`src/arbiter/confirmation.py`)

```python
import asyncio
import re

class TriStateConfirmationManager:
    """Handles confirmations with active asyncio watchdog timer and word-boundary safety checks."""
    AFFIRM_REGEX = re.compile(r"\b(yes|yeah|confirm|proceed|go ahead|yep|sure|do it)\b", re.IGNORECASE)
    DENY_REGEX = re.compile(r"\b(no|cancel|stop|don't|dont|abort|nevermind)\b", re.IGNORECASE)
    CLARIFY_REGEX = re.compile(r"\b(what|why|repeat|which|wait|explain|how)\b", re.IGNORECASE)

    def __init__(self, tts_engine, arbiter, timeout_seconds=10.0):
        self.tts = tts_engine
        self.arbiter = arbiter
        self.timeout_seconds = timeout_seconds
        self.pending_action = None
        self.watchdog_task = None

    def request_confirmation(self, action_payload: dict, prompt_text: str):
        self.pending_action = action_payload
        asyncio.create_task(self.arbiter.set_state("AWAITING_CONFIRMATION"))
        self.tts.speak_now(prompt_text)

        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
        self.watchdog_task = asyncio.create_task(self._expiry_watchdog(self.timeout_seconds))

    async def _expiry_watchdog(self, duration: float):
        """Active timer that breaks state deadlocks if the user says nothing."""
        try:
            await asyncio.sleep(duration)
            if self.pending_action:
                self._cancel_confirmation("Confirmation timed out after 10 seconds of silence.")
        except asyncio.CancelledError:
            pass

    async def evaluate_response(self, user_transcript: str) -> str:
        """Classifies response with strict word boundaries."""
        if not self.pending_action:
            return "EXPIRED"

        text = user_transcript.strip()

        # 1. Affirmative triggers
        if self.AFFIRM_REGEX.search(text):
            if self.watchdog_task:
                self.watchdog_task.cancel()
            action = self.pending_action
            self.pending_action = None
            asyncio.create_task(self.arbiter.set_state("PROCESSING_REACT"))
            return "AFFIRM"

        # 2. Negative triggers
        elif self.DENY_REGEX.search(text):
            if self.watchdog_task:
                self.watchdog_task.cancel()
            self._cancel_confirmation("Action cancelled.")
            return "DENY"

        # 3. Clarification triggers
        elif self.CLARIFY_REGEX.search(text):
            if self.watchdog_task:
                self.watchdog_task.cancel()
            self.watchdog_task = asyncio.create_task(self._expiry_watchdog(self.timeout_seconds))
            
            explanation = f"I am waiting to execute: {self.pending_action.get('summary', 'this command')}. Should I proceed?"
            self.tts.speak_now(explanation)
            return "CLARIFY"

        # 4. Unrecognized phrase
        else:
            self.tts.speak_now("Please answer yes or no.")
            return "CLARIFY"

    def _cancel_confirmation(self, message: str):
        self.pending_action = None
        self.tts.speak_now(message)
        asyncio.create_task(self.arbiter.set_state("IDLE_LISTENING"))
```

---

## 5. PipeWire 1.6.9 AEC Configuration

Deploy to `~/.config/pipewire/pipewire.conf.d/50-shin-aec.conf`:

```spa
context.modules = [
  {   name = libpipewire-module-echo-cancel
      args = {
          library.name = "aec/libspa-aec-webrtc"
          aec.args = {
              webrtc.high_pass_filter = true
              webrtc.noise_suppression = true
              webrtc.gain_control = false
              webrtc.voice_detection = true
              webrtc.transient_suppression = true
          }
          node.latency = 256/48000
          resample.quality = 4
          capture.props = {
              node.name = "Shin_AEC_Capture"
              node.description = "Shin Physical Mic In"
              target.object = "alsa_input.pci-0000_2b_00.3.analog-stereo"
              stream.dont-remix = true
              node.passive = true
          }
          source.props = {
              node.name = "Shin_Clean_Mic"
              node.description = "Shin AEC Filtered Output"
              media.class = "Audio/Source"
              audio.rate = 48000
              audio.channels = 1
              audio.position = [ MONO ]
          }
          sink.props = {
              node.name = "Shin_Playback_Sink"
              node.description = "Shin AEC Virtual Reference Sink"
              media.class = "Audio/Sink"
              audio.rate = 48000
              audio.channels = 2
              audio.position = [ FL FR ]
          }
          playback.props = {
              node.name = "Shin_AEC_Playback"
              node.description = "Shin Audio Hardware Forwarder"
              target.object = "@DEFAULT_AUDIO_SINK@"
              node.passive = true
          }
      }
  }
]
```

---

## 6. Phased Implementation Roadmap

### Phase 1: Core Foundation & Sandboxed Audio
- [ ] Initialize project with `uv` in repository root directory.
- [ ] Deploy PipeWire `50-shin-aec.conf` with verified `webrtc.*` syntax.
- [ ] Implement `src/audio/earcon.py` binding to the `pulse` device index with `PIPEWIRE_NODE` + `PULSE_SINK` routing and PortAudio restart-on-abort recovery.
- [ ] Set up Piper TTS with `PULSE_SINK=Shin_Playback_Sink` environment target.

### Phase 2: Speech & Wake Word Loop
- [ ] Set up continuous `Shin_Clean_Mic` audio consumer stream.
- [ ] Integrate `openWakeWord` + `silero-vad` with dynamic thresholding (0.5 IDLE $\rightarrow$ 0.88 SPEAKING).
- [ ] Deploy `faster-whisper` on GPU 0 with `compute_type="int8_float32"` using CUDA 12 PyPI wheels.
- [ ] Verify interrupt spotter with instant generation epoch advance.

### Phase 3: Sandboxed Execution & Supervisor
- [ ] Build `src/execution/supervisor.py` with Arch-canonical `bwrap` symlinks, `--unshare-pid`, `--tmpfs /dev/shm`, `--ro-bind /sys /sys`, and `--dev-bind-try` for GPU nodes.
- [ ] Route all job logs to `~/.local/state/shin/jobs/` (protecting `/tmp` `tmpfs`).
- [ ] Implement `PriorityAudioArbiter` and `TriStateConfirmationManager` with active watchdog timers.

### Phase 4: Tripartite LLM Provider Switch
- [ ] Implement `CanonicalTool` with `to_openai()`, `to_gemini()`, and `to_anthropic()`.
- [ ] Build multi-provider adapter supporting Ollama, Groq, Gemini, and Anthropic Claude.
- [ ] Cap local Ollama context to 4k tokens to protect Pascal VRAM.

### Phase 5: End-to-End Integration & Benchmark
- [ ] Test barge-in: Interrupt Shin mid-sentence and verify PortAudio stream recovers cleanly on subsequent tones.
- [ ] Test security & isolation: Verify that attempts to touch `~/.ssh` or `/var/run/docker.sock` in sandboxed jobs fail.
- [ ] Test Slime Tensei batch transcode: Execute CPU `libsvtav1` transcode on `~/Downloads` video files inside the sandbox and verify completion alert pops only after returning to `IDLE_LISTENING`.
