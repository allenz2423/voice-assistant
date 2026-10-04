# Adam on Desky: read-only resource review

**Scope:** Read-only local inspection on 2026-10-04. Adam PID 1990519 (`python -m src.main`) was sampled for 10 seconds. No config/environment files, secrets, or credentials were read; no services or app settings were changed.

## Machine and telemetry

- CachyOS Linux, kernel `7.2.8-1-cachyos`, x86_64. AMD Ryzen 9 5900XT: 16 cores / 32 threads, boost enabled, max reported 4.98 GHz.
- 31 GiB RAM visible (about 15 GiB available during inspection); 31 GiB swap, 2.8 MiB used. This is a 32 GiB host, so a 16 GiB product target still needs validation.
- NVIDIA driver 580.178.04; GTX 1080 Ti (11 GiB) and RTX 3070 (8 GiB). `nvidia-smi` and `pmon` work. During the process sample the 1080 Ti reported 0% GPU, 494 MiB used, 8.2 W, 25 C; RTX 3070 reported 40%, 1,398 MiB, 25 W, 39 C. A later instantaneous RTX sample was 87%, 1,371 MiB, 40 W, showing concurrent host GPU activity. `nvidia-smi --query-compute-apps` attributed 470 MiB on the 1080 Ti and 336 MiB on the RTX 3070 to Adam; `pmon` saw Adam at 0% SM/memory utilization at its instant. These are device snapshots, not assistant energy measurements.
- CPU energy: `/sys/class/powercap/intel-rapl:0` exposes `package-0` and `core` names/ranges, but `energy_uj` reads fail with **Permission denied**; no sudo or permission change attempted. This Intel-named interface is not a validated AMD meter. AMD `k10temp` provides thermal sensors only; no CPU power/energy input was found. No battery devices, and no whole-system AC meter. Thus CPU/host energy is unavailable; GPU board power is available via NVIDIA telemetry only.

## Adam idle sample

- No child processes were present. Across the 10-second sample, PID 1990519 had **RSS 2,698,044 KiB (2.57 GiB), PSS 2,669,213 KiB (2.55 GiB), USS 2,658,480 KiB (2.54 GiB), 75 threads**, and **1.3% of one CPU core** average utilization. PSS/USS came from `/proc/1990519/smaps_rollup`; CPU from `/proc` ticks. These are a single idle-window sample, not startup/active peaks.
- Largest resident mapping groups by PSS: anonymous mappings ~1,007 MiB; `[heap]` ~439 MiB; `libtorch_cpu.so` ~163 MiB; `/dev/nvidiactl` ~154 MiB; CUDA `libcublasLt.so.12` ~102 MiB and `.so.13` ~81 MiB; `libnvrtc.so.13` ~74 MiB and `libnvJitLink.so.13` ~71 MiB. Also resident: Torch CUDA, cuDNN, OpenCV `cv2`, ONNX Runtime including its CUDA provider, and CTranslate2. This shows mapped/resident runtime components, but cannot assign anonymous heap/model allocations to a specific feature. The 75 threads persist despite a source-level native thread cap default of four; their individual owners were not isolated.
- On a 16 GiB machine, this measured idle Adam footprint alone is about 16% of installed RAM; it leaves roughly 13.5 GiB before the OS, desktop, other apps, and active-turn peaks. This is a lower-bound budget illustration, not a 16 GiB test.

## Likely resource causes from source

- `src/audio/stream.py:158` constructs `SileroVAD`; `src/audio/vad.py:1-42` imports Torch and loads ONNX Silero, then runs inference on 512-sample blocks before applying the RMS energy floor. That ordering is a concrete candidate for avoidable always-on CPU work.
- `src/main.py:116-337` eagerly builds the audio/STT/wake/brain graph. It constructs `MemoryManager`, starts audio infrastructure, and passes `preload_ocr=True`; configured OCR is loaded at startup in `src/llm/brain.py:541-650`. Vision preloading is configurable there. Local Whisper loads its model in its constructor (`src/stt/transcriber.py:31-61`). Which optional models are active depends on configuration, which was intentionally not inspected.
- `src/main.py:1-54` calls `configure_native_threading()` before and after application imports; `src/runtime.py:17-61` caps common native pools (default four) and Torch/OpenCV threads. The measured 75 total threads show the cap does not bound total process threads.
- Repository power notes already caution that NVIDIA power is board telemetry rather than wall power, and identify VAD gating and optional OmniParser residency as experiments, not established savings (`docs/power_consumption.md:1-18,42-67`).

## Next experiments, ordered by expected impact

1. **Attribute idle RAM/VRAM by feature (high expected impact; medium risk):** On a disposable/research run, capture PSS/USS, thread count, and per-GPU process memory after base startup and after enabling OCR, local STT, local wake/intent, and vision one at a time. Keep quality/features fixed for the baseline and record cold-start latency. This resolves whether model/runtime initialization explains the 2.54 GiB private footprint and both GPU contexts.
2. **Make optional heavy runtimes on-demand and releaseable (high; medium):** Compare startup OCR/vision preload with first-use loading and idle close/unload. Record warm/cold latency, peak host/GPU memory, and NVIDIA board power. Preserve an explicit fast-path option if cold-start latency is unacceptable.
3. **Measure always-on audio work (medium-high; low-to-medium):** Profile per-thread CPU/wakeups and replay fixed labeled silence/noise/speech through `SileroVAD`. Track false rejects/alarms, endpoint behavior, CPU time, and supported energy channel. The laptop now has an initial quiet-frame energy-gate microbenchmark; broader speech-quality evaluation remains open.
4. **Run a real 16 GiB envelope test (high product relevance; medium):** On a 16 GiB laptop, measure OS/desktop baseline and Adam idle/active peaks with the inexpensive vision configuration, representative screenshots, local/cloud STT choice, swap, latency, and task quality. Set a memory headroom target before choosing model/preload defaults.

## Commands and paths inspected

Hardware/OS: `uname -a`, `/etc/os-release`, `lscpu`, `free -h`, `lspci`, `/dev/dri`. Process: `ps`, `/proc/1990519/{stat,status,smaps,smaps_rollup}`, `pgrep -P`. GPU: `nvidia-smi --query-gpu`, `--query-compute-apps`, `pmon`. Power access: `/sys/class/powercap`, `/sys/class/hwmon`, `/sys/class/power_supply` (read-only; no readable CPU energy counter, battery, or wall telemetry).

Source/docs reviewed: `src/main.py`, `src/runtime.py`, `src/audio/{stream,vad}.py`, `src/stt/transcriber.py`, `src/wake/engine.py`, `src/llm/brain.py`, `src/tools/omniparser.py`, `src/memory/{manager,embedder}.py`, `pyproject.toml`, `docs/power_consumption.md`, `docs/power-cutting-proposal.md`, `docs/latency-cutting-proposal.md`, and `docs/latency-minimization-research.md`.

## Follow-up after the review

The source finding about running VAD before the energy floor was acted on after the Desky sample. Quiet input now returns before neural inference, resets Silero once at the start of a quiet run, and stores at most eight quiet frames to warm its state when speech resumes. On the laptop, five trials of a synthetic 300-frame quiet sequence (9.6 seconds) measured median process CPU time of 23.24 ms for inference-then-discard versus 0.90 ms with the energy gate; both produced the same no-speech result. This follow-up is a local laptop microbenchmark, separate from the Desky measurements above, and is not a microphone-quality or wattage evaluation.

The Silero VAD also no longer imports Torch merely to wrap model input for ONNX. In isolated laptop processes, the previous official wrapper plus ONNX model measured 376.8 MiB USS, while the NumPy/ONNX replacement measured 42.7 MiB USS with Torch absent. The NumPy path matched the official wrapper's output exactly on deterministic 8 kHz, 16 kHz, and 32 kHz frame sequences. A locally generated speech sequence with leading/trailing silence had zero decision mismatches; median process CPU time was 103.01 ms for the official Torch wrapper and 46.88 ms for the NumPy wrapper with energy gating. The memory comparison isolates this component and will be smaller or nonexistent in configurations that load Torch for speaker verification or diarization anyway.
