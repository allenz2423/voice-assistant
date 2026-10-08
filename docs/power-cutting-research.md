# Power optimization: research and evidence review

**Status:** Research notes for the local measurement plan. This is not a report of measured results. No cited publication or repository trace establishes this assistant's watts, joules per turn, battery runtime, or savings.

## Findings from published sources

### Operation and data-movement energy are technology-specific

Horowitz's ISSCC 2014 overview compares approximate costs for computation and data movement in the technology context discussed in that presentation. It supports the qualitative point that moving data can carry meaningful energy cost. It does not provide a transferable joules-per-token coefficient, a modern GPU estimate, or a measurement of this codebase. Predicting assistant inference energy from model parameter count alone would exceed what this source supports ([Horowitz, 2014](https://doi.org/10.1109/ISSCC.2014.6757323)).

**Engineering consequence:** Measure the target model, precision, kernels, batching, input/output lengths, cache state, host, and complete workload on the intended hardware. State the meter domain and boundary. GPU board power, CPU package energy, battery discharge, and AC wall energy are distinct quantities.

### CPU idle residency depends on the machine

Linux's CPU idle documentation describes idle states and the kernel driver/governor mechanisms that select among states available on a platform. It does not imply that a particular Python callback or thread causes a specified wakeup rate or package-power cost ([Linux CPU idle states](https://docs.kernel.org/admin-guide/pm/cpuidle.html)).

**Engineering consequence:** To attribute CPU energy to audio handling, measure actual process/stage CPU time and, if the claim concerns wakeups or idle residency, collect scheduler or kernel residency counters on the target platform. Audio chunk duration alone is not a wakeup measurement.

### GPU telemetry and control need device-specific interpretation

NVIDIA's `nvidia-smi` reference documents GPU query fields, P-state reporting, and supported power-limit controls. A reported GPU power value is GPU telemetry, not whole-system AC consumption; a P-state reading does not identify why the GPU is in that state. Support and controls vary with board and driver ([NVIDIA `nvidia-smi` reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html)).

**Engineering consequence:** Preserve raw timestamped telemetry and control for display activity, clocks, temperature, driver, other processes, and workload. A persistent CUDA-capable process does not by itself establish a power penalty. A cap experiment requires checking supported limits and measuring both energy and application behavior on that specific card.

### Energy-counter wrap must be handled from counter metadata

The Linux Power Capping Framework describes energy counters such as `energy_uj` and the `max_energy_range_uj` value used to interpret a counter wrap ([Linux powercap documentation](https://docs.kernel.org/power/powercap/powercap.html)). The current sampler reads both values for direct package-level `intel-rapl:*` zones and records interval deltas. A decreasing counter is ambiguous between wrap and reset, so the sampler retains a modulo candidate but marks watts unavailable for that interval. Nested zones are not enumerated.

**Engineering consequence:** Retain raw counters and zone metadata, use wrap-aware deltas only when the range is known, identify domains, and avoid summing parent and child domains. Treat resets, missing permissions, and unsupported interfaces as unavailable intervals. The current RAPL stream is not validated enough for long traces or domain-specific claims.

### Training studies do not establish inference energy here

Strubell et al. study energy/carbon considerations for NLP model training and reporting practices. Their work is relevant background for documenting energy boundaries, but it is not a measurement of this assistant's local or cloud inference workload, nor does it establish a joules-per-turn value ([Strubell et al., ACL 2019](https://aclanthology.org/P19-1355/)).

### Android sensor hubs do not establish desktop wake-word savings

Android's Context Hub Runtime Environment describes an interface for supported low-power context-sensing subsystems on Android devices ([Android CHRE](https://source.android.com/docs/core/interaction/contexthub)). It does not show that this desktop Python audio pipeline has access to such an always-on subsystem or quantify a power saving from moving wake-word detection there.

## Repository facts that bound the proposals

- `src/audio/stream.py::AudioStreamManager.__init__(target_source="Adam_Clean_Mic", sample_rate=16000, chunk_size=512)` configures an input callback that copies and queues chunks. `src/main.py` constructs it with a configured chunk size that defaults to 1,280 samples; the class's direct default is 512. Neither value is a measured callback/scheduler rate. `get_chunk(timeout=0.1)` waits on a queue; the documented source has no `sleep(0.005)` polling loop.
- `src/audio/vad.py::SileroVAD.is_speech(audio_chunk_16k, threshold=0.25, use_energy_floor=True, energy_floor=0.0025)` performs model calls over 512-sample blocks before calculating and applying the RMS floor. The current floor cannot skip that inference. Calls come from main speech/turn paths and the meeting pipeline, so a pre-inference gate must be checked against each caller's thresholds, adaptive floor, and quality behavior.
- `src/tools/omniparser.py::OmniParserScreenshotGrounder` can preload and keep a worker process alive; `src/llm/brain.py` invokes preload when enabled. This says nothing about whether the configured device is CUDA or CPU at runtime, whether the GPU enters a different state, or whether shutting down the worker reduces power.
- `tools/measure_power.py::run_benchmark()` samples NVIDIA GPU telemetry, raw battery sysfs values, and direct Intel RAPL package zones. It does not measure wall power, CPU utilization (although `get_cpu_stat()` exists, it is unused), context switches, scheduler wakeups, application-stage latency, or per-turn energy. Versioned metadata, per-source statuses, monotonic cadence, and field-level NVIDIA parsing are implemented; traces remain unvalidated on representative hardware, and short bursts can still be missed.
- `src/llm/provider.py` selects a configurable provider, defaults local inference to Ollama, and configures local model `keep_alive=-1`. This establishes a local model residency setting in that path; it does not show that local processing saves energy relative to a cloud path. A valid comparison needs matched work and output quality, measurement over a stated system boundary, and separate treatment of client energy and unmeasured remote server energy.

## Research-supported conclusions vs. hypotheses

Published references support careful instrumentation, domain distinctions, and platform-specific treatment of idle/power behavior. They do not establish this repository's:

- GPU idle power or an OmniParser worker's contribution to it;
- audio callback wakeup frequency, CPU idle residency, or cost of processing silence;
- local/cloud energy difference, per-token or per-interaction energy, or battery runtime;
- savings from an RMS gate, worker shutdown, OCR crops, zero-copy capture, thread affinity, quantization, or GPU power caps.

These are falsifiable engineering hypotheses. For each, compare a baseline and a changed implementation on the same machine, with matched inputs, model/configuration, display state, thermal/warm state, and measurement interval. Save raw traces and repeat runs. Report distributions, source coverage, instrument limitations, and application quality/latency alongside energy. Do not annualize an assumed load; annualization may only be derived from a measured stable load and a stated duty cycle.

## Ordered path from research to an actionable measurement

### Implemented sampler work

`tools/measure_power.py` now uses monotonic timestamps and an anchored sampling cadence; validates positive duration/interval inputs; records per-source status and errors; parses NVIDIA fields independently; records query duration and actual sample intervals; and writes versioned metadata and raw samples. RAPL records counter values, zone identity, and `max_energy_range_uj`; counter decreases retain an ambiguous wrap candidate and do not produce interval average watts. These features are implemented in the sampler but remain unvalidated on representative hardware.

### Remaining implementation and validation

1. Add event markers for idle, wake, STT, LLM/tool, TTS, and complete interaction boundaries. Integrate supported channels only across covered event intervals; report sample gaps rather than extrapolating silently. Retain raw data.
2. Add focused checks for source availability, malformed readings, missing permissions, wrap/reset, event coverage, serialization, and summary arithmetic.
3. Compare at least one host-power trace against a wall meter before making a whole-system claim. The current sampler does not enumerate nested RAPL zones or establish representative hardware behavior.

### Operator-only work

1. Supply the target hardware/software inventory and select one primary domain: AC wall meter, battery, GPU board, or RAPL zone. State which components and services fall inside the experiment boundary.
2. Run a matched idle baseline and repeatable workload. Fix input, model/provider, output/quality target, precision, tool actions, audio/display state, thermal state, and warm/cold conditions. Include network and playback if the declared interaction boundary includes them.
3. Repeat each condition, preserve raw traces and configuration, and report uncertainty and run-to-run spread. A result is actionable only when the effect exceeds the observed variation and sensor resolution, and predeclared application quality/latency limits still pass.

## References

- Horowitz, M. (2014). [Computing's Energy Problem (and what we can do about it)](https://doi.org/10.1109/ISSCC.2014.6757323). ISSCC plenary.
- Linux kernel documentation. [CPU idle states](https://docs.kernel.org/admin-guide/pm/cpuidle.html).
- Linux kernel documentation. [Power Capping Framework](https://docs.kernel.org/power/powercap/powercap.html).
- NVIDIA. [`nvidia-smi` documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
- Strubell, E., Ganesh, A., & McCallum, A. (2019). [Energy and Policy Considerations for Deep Learning in NLP](https://aclanthology.org/P19-1355/). ACL.
- Android Open Source Project. [Context Hub Runtime Environment](https://source.android.com/docs/core/interaction/contexthub).
