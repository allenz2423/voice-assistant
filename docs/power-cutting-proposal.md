# Power reduction implementation plan

**Status:** Ordered experiments; no savings are established. The initial measurement hardening in `tools/measure_power.py` is implemented. Hardware validation and offline event-window analysis remain prerequisites for comparative results. Each runtime experiment is opt-in until its acceptance gate is met.

## Prerequisites: make power traces interpretable

`tools/measure_power.py` is a prototype sampler, not a calibrated system energy meter. Its initial schema, source status, and monotonic timing work is implemented. Before treating output as comparative evidence:

1. Correlate sampler output offline with the shared app JSONL by `trace_id` and `clock_ns`. The sampler accepts a trace ID but does not consume event files or integrate over task boundaries. Do not automatically upload traces.
2. Validate source parsing and actual sample cadence on representative hardware. Unavailable fields remain explicit; a missing reading is not zero.
3. For RAPL, the tool reads range metadata. A decreasing counter is ambiguous between wrap and reset: the modulo delta is retained as a candidate, while interval average watts is null. Validate counter behavior on target hardware before using wrap candidates.
4. Battery current/voltage signs and units vary by driver. The sampler keeps raw sysfs values and paths, and does not derive battery watts until an operator validates sign and scaling against known charging/discharging conditions.
5. Store sample timestamps and actual elapsed interval. Do not infer task energy from a single instantaneous sample. Integrate only a validated power source over a defined workload window and report missing-sample coverage. RAPL is package energy, `nvidia-smi` is GPU telemetry, and battery discharge is platform/battery telemetry; do not add them together.

For whole-system wall energy use an external meter; record meter model, calibration/accuracy, sampling interval, and whether display/peripherals are included. Linux powercap documents counter ranges and domains; NVIDIA documents device-specific telemetry availability ([Linux powercap](https://docs.kernel.org/power/powercap/powercap.html), [NVIDIA `nvidia-smi`](https://docs.nvidia.com/deploy/nvidia-smi/index.html)).

## Implementation sequence

### 1. Optional RMS short-circuit before Silero

**Code:** `src/audio/vad.py`; configuration model in `src/config.py` (`AudioConfig`) and the `audio` section of `config.yaml`; evaluation in a new focused VAD evaluation command or documented reproducible invocation.

**Change:** Add `vad_energy_gate_enabled: false` and a separately named RMS threshold to `AudioConfig`. Validate threshold as finite and nonnegative; reject invalid configuration at startup. With the option omitted, preserve the current ungated behavior. In `SileroVAD.is_speech`, compute RMS before inference and bypass the model only when the gate is enabled and RMS is below threshold. Preserve the current return type and return a zero probability for skipped blocks. Do not change the existing model threshold or its default behavior. Confirm exact VAD call sites before implementation; `src/main.py` currently passes dynamic `threshold` and `use_energy_floor` values.

**State/failure behavior:** Skipping inference also skips Silero's recurrent state update. The gate stays disabled by default; if config is absent or invalid, retain current ungated behavior through validated defaults. Do not silently catch model failures as “silence”; retain existing failure behavior.

**Operator dependency:** Provide labeled, licensed audio representative of target microphones and environments, including quiet speech, whisper, noise, and speech onset after silence/noise. Freeze microphone gain, sample rate, model artifact, and thresholds for each run.

**Acceptance:** Compare enabled/disabled on identical held-out clips. Report per-class false accept/reject counts and rates, sample counts, CPU time, validated package/system energy per clip, missing telemetry coverage, and p50/p95. Before measuring, set numeric bounds for allowed false-reject and false-accept change and minimum resource improvement. Keep the feature opt-in unless all bounds pass.

### 2. Add evidence for custom wake-word fallback before redesign

**Code:** Instrument candidate generation and transcription call sites in `src/wake/engine.py` and `src/main.py`; do not alter wake routing in the instrumentation change.

**Change:** Emit aggregate counters by mode/reason (standard model, custom phrase, candidate accepted/rejected, transcription attempted/failed). Never write microphone PCM, transcript, configured wake phrase, or speaker data to routine telemetry. If per-run correlation is needed, use a random run ID and discard it with the trace.

**Operator dependency:** Supply representative continuous background recordings and intended wake examples with consent and a retention plan. openWakeWord documents model training/evaluation and threshold evaluation on representative audio ([project and evaluation guidance](https://github.com/dscripka/openWakeWord)).

**Acceptance:** Evaluate any proposed classifier/model only on held-out recordings. Report false accepts per hour, false rejects, candidate transcription calls, energy, and wake-to-command latency with recording duration and environment. No wake path change until predeclared error bounds pass.

### 3. Profile persistent versus on-demand vision worker

**Code:** Worker lifecycle in `src/tools/omniparser.py`; add a configurable idle timeout only after baseline profiling and confirm the worker can restart without losing model/config state.

**Change:** Keep current persistent behavior as default. Prototype shutdown after an explicit idle timeout and lazy restart. Expose timeout in the existing vision/tool configuration only after locating the canonical config owner; do not add a duplicate setting. A worker crash/restart must be visible in logs, and a failed restart must return the existing tool failure rather than report an observation.

**Operator dependency:** Name the GPU, driver, workload, whether it drives a display, and a supported board-power source. No claim that a persistent process pins a GPU state is established.

**Acceptance:** Paired repeated idle and inference runs on the same device; report idle board power, board energy per task, request success, and cold/warm request latency. Keep the timeout only if its measured energy benefit meets a predeclared bound and restart latency/reliability remain within declared limits. Do not generalize across GPUs.

### 4. Profile screenshot and OCR before changing capture

**Code:** Platform capture in `src/tools/desktop.py`; image/OCR path in `src/tools/ocr.py` and `src/tools/omniparser.py`.

**Change:** Add stage timers around capture, decode/resize, OCR, and matching; use monotonic time and aggregate dimensions/format, not image content. Do not add a crop or shared-memory path until a measured stage is a bottleneck. Any crop prototype must retain a full-screen fallback and translate returned boxes to screen coordinates.

**Acceptance:** On fixed desktop tasks record compositor/backend, resolution/scale, p50/p95 per stage, memory, energy if available, target recall, false matches, and coordinate error. Include targets outside the proposed crop and capture failures. Require no regression beyond predeclared bounds.

### 5. Benchmark an optional NVIDIA power limit manually

**Code:** No application change in this phase. Do not apply a limit automatically from Adam.

**Operator dependency:** On a supported NVIDIA device, query the supported range and current setting, confirm required privileges, choose a reversible setting, and record a restore command before changing it. Consult the device-specific [`nvidia-smi` reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

**Acceptance:** Compare identical workloads at baseline and supported limits; report exact board/driver/limit, throughput, end-to-end latency, task success, and board energy. Restore the original limit after each test. A lower peak watt reading alone does not pass; energy and quality bounds must pass.

### 6. Compare local and remote execution on battery

**Code:** Provider selection is in `src/config.py` (`LLMConfig`) and `src/llm/provider.py`; model/tool loop is `src/llm/brain.py`. The configured provider must be recorded from effective config (`config.yaml` can override defaults). Cloud usage is currently discarded; use the instrumentation plan in [`cloud_cost_and_latency_optimization.md`](cloud_cost_and_latency_optimization.md) before reporting per-turn cloud cost.

**Change:** Benchmark configurations only; no automatic battery-aware routing. Keep task set, prompt/context, tool outcomes, audio playback, provider/model, and network conditions matched. Treat battery, RAPL, GPU, and wall measurements as separate metrics.

**Operator dependency:** Validate battery sign/units on the actual laptop or use a suitable external analyzer; define whether the boundary includes wake listening, network radio tail, TTS, and idle time. Define privacy and cost constraints before sending any task to a remote provider.

**Acceptance:** Report task-level energy and latency distributions for completed tasks plus quality/tool success. Include provider/model/config identity, usage availability, network, sample counts, and missing telemetry. Do not compare local full-response latency with remote first-token latency; current provider calls are non-streaming.

### 7. Compare supported ASR precision modes

**Code:** Faster-whisper settings in `src/config.py`/`src/stt/transcriber.py`; Qwen ASR artifacts and runtime in the same files. Keep results for those two backends separate.

**Change:** Benchmark only supported `compute_type` values for the exact faster-whisper model/runtime and available Qwen GGUF artifacts. Do not assume LLM precision settings or artifact compatibility from ASR results. CTranslate2 lists supported compute types and runtime behavior ([quantization documentation](https://opennmt.net/CTranslate2/quantization.html)).

**Operator dependency:** Freeze model artifact hash, runtime/library version, device, decode options, audio set, and language mix.

**Acceptance:** Report word error rate or task-specific transcription error, missed/changed critical words, completion latency, peak memory, and energy per clip. Keep a mode only within predeclared accuracy bounds and with measured resource benefit.

## Shared run protocol

1. Identify one change and write the workload, baseline, repetitions, telemetry source, quality metric, minimum benefit, and maximum acceptable regression before runs.
2. Save the effective config with secrets removed, code revision, model/artifact identifiers, raw trace, and analysis command. Do not log private audio, transcript, prompt, screenshot, API key, or authorization header.
3. Run baseline and one change at a time under matched conditions. Record warm/cold state, background activity, power source, and interruptions. An interrupted or under-covered trace is invalid, not zero energy.
4. Publish raw traces or a reproducible data reference, analysis method, per-run values, sample count, missing-data count, and p50/p95. Distinguish measured results from hypotheses.

No numeric savings target is claimed here. Numeric acceptance thresholds are experiment-owner inputs and must be fixed before measurement; this repository does not yet contain the needed representative evaluation sets or validated whole-system energy source.
