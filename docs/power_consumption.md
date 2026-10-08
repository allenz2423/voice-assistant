# Power consumption: measurement and implementation plan

**Status:** No reference-machine power or energy results are established here. This document describes the current instrumentation, its gaps, and the work needed before making a power claim. The previously published watts, joules, latency percentiles, VRAM occupancy, and annual savings lacked traceable raw measurements and are not valid baselines.

## What the current code measures

[`tools/measure_power.py`](../tools/measure_power.py) exposes `query_nvidia_smi()`, `query_laptop_battery()`, `query_rapl_energy()`, `get_cpu_stat()`, and `run_benchmark(duration_sec, interval_sec, output_path=None)`. The command line accepts duration, interval, optional JSON output, operator labels, and an optional trace ID. Output is versioned JSON with monotonic and UTC timestamps, host metadata, per-source status/errors, raw battery values, GPU readings, and RAPL counters/deltas. Output files are restricted to the current user.

- **NVIDIA:** CSV fields are parsed independently; unavailable values remain null with per-device unavailable-field lists. The tool records the query, driver version, UUID, model, clocks, temperature, and display-active state where available. This is GPU telemetry, not whole-system power. Sampling is sequential, so channels within a sample are not simultaneous.
- **Battery:** Reads raw values and sysfs paths under `BAT*` entries. It does not infer discharge watts or signs from those values. Battery telemetry is absent on desktops and may be unavailable or platform-dependent.
- **RAPL:** Reads `energy_uj` and `max_energy_range_uj` for direct Intel package zones. It derives interval energy and average watts when counters are valid. A counter decrease produces a modulo wrap candidate, but the interval is marked ambiguous and average watts remains unavailable because sysfs alone cannot distinguish wrap from reset. This is not a general CPU meter and may be missing on unsupported/restricted systems.
- **CPU:** `get_cpu_stat()` reads raw `/proc/stat` counters for potential callers, but `run_benchmark()` does not use it. The trace makes no CPU utilization, context-switch, scheduler-wakeup, or idle-residency claim.
- **Timing and limits:** Samples use `time.monotonic_ns()` on an anchored cadence and retain actual intervals. GPU readings are instantaneous; the tool does not integrate GPU energy or derive energy per interaction. Short bursts may be missed, and host telemetry is not calibrated wall power. The optional trace ID enables offline correlation with app events on the same host boot; this tool does not join events or calculate per-turn energy.

`src/audio/stream.py` defines `AudioStreamManager(target_source="Adam_Clean_Mic", sample_rate=16000, chunk_size=512)`. Production construction in `src/main.py` passes `sample_rate=16000` and configured `chunk_size` (default 1,280 samples, or 80 ms at 16 kHz); the class default is 512 samples (32 ms). The audio callback copies/enqueues chunks, while `get_chunk(timeout=0.1)` consumes them. Chunk duration is not a measured callback frequency, scheduler wakeup rate, or CPU cost. `src/audio/vad.py` defines `SileroVAD.is_speech(audio_chunk_16k, threshold=0.25, use_energy_floor=True, energy_floor=0.0025)`. It runs Silero inference over 512-sample blocks before calculating/applying the RMS floor, so the current floor does not avoid inference. `src/main.py` calls this VAD in its speech and wake/turn paths; the meeting pipeline also constructs/uses VAD in `src/audio/meeting.py`.

The optional persistent vision worker is `src/tools/omniparser.py::OmniParserScreenshotGrounder`; `src/llm/brain.py` preloads it when configured. Persistence is established by code, but its GPU residency or power effect is not. The LLM configuration defaults to local Ollama and uses `keep_alive=-1` in `src/llm/provider.py`; provider selection is configurable. Neither fact establishes an energy advantage for local or remote inference.

## Energy definitions and comparison boundary

Energy is the integral of power over time, `E = ∫P(t)dt`. For sampled readings, retain timestamps and state the integration rule; trapezoidal integration is appropriate for reasonably frequent point samples, but does not recover bursts missed between samples. Keep these quantities separate:

- NVIDIA board power and board energy estimated from its samples;
- RAPL domain energy from energy-counter deltas;
- host battery discharge over a defined interval;
- whole-system AC energy from a wall meter.

Do not add overlapping domains (for example, a GPU board estimate and wall power) or call board power whole-system power. For a whole assistant interaction, specify whether the boundary includes display, microphone/speaker, network, and external cloud compute. Local and cloud comparisons need matched user-visible output and a declared boundary; client-side wall power alone does not include cloud-server energy.

Annualization is only arithmetic after a stable, independently measured additional load is established: `kWh/year = ΔP(W) × 8,760 / 1,000`. State the meter, baseline, duty cycle, and workload behind `ΔP`; the formula does not validate those assumptions.

## Remaining implementation and validation

The sampler hardening, raw metadata, monotonic cadence, source status, and versioned JSON output are implemented. Remaining work before treating traces as comparative energy evidence:

1. Keep application events in the shared JSONL format specified by [`latency-cutting-proposal.md`](latency-cutting-proposal.md); correlate offline by `trace_id` and `clock_ns`. The sampler accepts a trace ID but does not consume or join event files.
2. Validate parsing, permissions, actual interval behavior, and RAPL wrap/reset handling on representative hardware. A modulo delta is not proof of wrap; ambiguous decreases are excluded from average watts.
3. Battery current/voltage signs and units vary by driver. Keep raw sysfs values and paths, and derive battery watts only after validating sign and scaling against a known charge/discharge condition.
4. Add offline interval integration only with a defined event boundary, source coverage checks, and retained raw samples. Do not claim per-turn energy from an unmarked continuous trace.
5. Validate with automated checks and a representative wall-meter comparison before publishing any measured result. Document sensor uncertainty; do not substitute one power domain for another.

## Operator-only experiment procedure

These steps require access to the target machine and cannot be supplied by code changes alone.

1. Choose the metric and boundary first: whole-system AC energy (plug-in wall meter), battery discharge (unplugged host under controlled charge state), GPU board, or readable RAPL domain. Use a wall meter for a whole-system claim. Do not infer machine power by summing component counters.
2. Record machine model, CPU/GPU, RAM, OS/kernel, driver/runtime, power profile, display/brightness, audio devices, model/provider, precision, application settings, and other GPU/CPU workloads. Record whether model/worker state is cold or warm.
3. Define a repeatable workload and event boundaries. For audio/VAD, use the same labeled audio corpus and number of frames; for an interaction, use the same prompt, input, tools, output length/quality, and playback behavior. Run an idle baseline of matched duration and the active workload; alternate or randomize run order where practical.
4. Save the raw trace and manifest for every repetition. Use a sampling interval short enough for the stage duration and state the sensor's own update rate. The current 1-second default is not adequate evidence for short bursts. Record warm-up, run count, excluded runs, and why.
5. Report the per-run energy and latency distribution, baseline-subtracted result, channel/domain, coverage, and measurement uncertainty. A claimed improvement must exceed observed run-to-run variation and instrument resolution, while meeting a predeclared speech/recognition/interaction quality and latency bound. Set those application-specific bounds before comparing variants; this repository has no established universal threshold.

## Initial implementation decisions

- **Output format:** Keep the sampler output versioned JSON for the first implementation; fix its CLI help rather than adding CSV.
- **Event contract:** Use the shared app event writer in `src/telemetry/events.py` and correlate offline. This keeps stage instrumentation in the runtime and avoids a second marker format. Event writes are opt-in and failure must not interrupt audio/LLM work.
- **Initial measurement boundary:** Collect every supported channel, but classify each result by domain. Require a plug-in wall meter for whole-system claims. Without one, report only battery, GPU-board, or named RAPL-domain results; never silently substitute.
- **Sensor coverage:** Store missing/error states per channel. Reject an energy result when the chosen channel has a gap spanning the event interval or when its sampling/update rate cannot resolve the measured stage. Keep raw readings so the analysis can be repeated.
- **Quality bounds:** Keep proposed runtime changes disabled until an experiment owner records the false-end/false-reject, task-quality, latency, and minimum-benefit bounds for that workload. No universal threshold is supported by current evidence.

## Candidate experiments and decision criteria

| Hypothesis | Exact code path | Experiment | Decision evidence / dependency |
|---|---|---|---|
| An RMS pre-check before Silero might save CPU work on quiet frames. | `src/audio/vad.py::SileroVAD.is_speech`; callers in `src/main.py` and `src/audio/meeting.py` | Implement a gated variant and compare it with current inference on labeled silence, ambient noise, quiet speech, whisper, and normal speech using identical audio. Count inferred/skipped frames; measure CPU-domain energy or wall/battery energy and processing time. | Adopt only if the chosen speech miss/false-alarm bounds are met by condition and the measured energy/time change exceeds noise. Threshold and acceptable quality bounds remain product decisions. A pre-gate must preserve the existing return shape `(bool, probability)` and account for adaptive `energy_floor` call arguments.
| Closing the resident OmniParser worker while idle might change GPU residency/power. | `src/tools/omniparser.py::OmniParserScreenshotGrounder.load/close/_run`; preload in `src/llm/brain.py` | Compare resident idle and worker-closed idle on the same GPU, display state, driver, and duration. Then compare identical screenshot requests, including cold-start latency and failures. | Require repeated raw GPU traces and request latency. This worker may run CPU-only depending on `computer_vision.device`; do not treat process existence as GPU activity or infer causation from P-state alone. NVIDIA tooling and GPU availability are dependencies.
| Crop-based OCR might reduce work. | `src/tools/desktop.py`, `src/tools/ocr.py` | Compare existing full-frame and proposed crop processing for the same target/display scale; validate text recall and coordinate remapping as well as time/energy. | Requires named display/compositor, fixed screenshots, and a correct full-frame fallback. No power benefit is assumed before measurement.
| A GPU power limit may change board energy or throughput. | Operator configuration; no automatic cap is implemented in this repository | Query whether the specific board/driver supports a limit; run identical workloads at supported settings and capture board energy, end-to-end latency, and output quality. | Requires privileges/support and NVIDIA tooling; report per-device results. Never apply an unverified limit or generalize one device's result. NVIDIA documents query/control options in its [`nvidia-smi` reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

## Published-source scope

NVIDIA documents `nvidia-smi` telemetry and supported controls; its GPU power reading is a board telemetry quantity and is not an AC wall-meter measurement ([NVIDIA `nvidia-smi` reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html)). Linux documents CPU idle-state selection as dependent on available states, driver/governor, and platform behavior; that documentation does not give this process a fixed wakeup cost ([Linux CPU idle states](https://docs.kernel.org/admin-guide/pm/cpuidle.html)). The Linux powercap interface documents energy counters and maximum counter range, which is the information required to reason about wrap; this harness reads that range for direct package zones but does not enumerate nested zones ([Linux Power Capping Framework](https://docs.kernel.org/power/powercap/powercap.html)). Horowitz's 2014 overview gives historical approximate operation/memory-access energy comparisons; it is background on hardware mechanisms, not a modern GPU or application benchmark ([Horowitz, ISSCC 2014](https://doi.org/10.1109/ISSCC.2014.6757323)). None of these sources provides a power result for this repository.
