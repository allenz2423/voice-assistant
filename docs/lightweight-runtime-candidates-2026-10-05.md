# Lightweight runtime candidates for Adam on a 16 GB laptop

October 5, 2026 · Research and code review. No model was installed, service changed, or live benchmark run for this note.

## Decision in brief

Adam does not need a different general-purpose brain on the laptop: its main LLM is hosted. Its local burden comes from *coexisting* speech, speaker, vision, browser, and memory components. The best candidates are (1) making costly optional capabilities truly short-lived, (2) replacing the PyTorch speaker path with a measured ONNX alternative, (3) choosing the smallest ASR/TTS combination that preserves real-voice quality, and (4) preferring app/DOM/accessibility evidence before spawning visual or browser models. Intel NPU execution is a promising experiment, not an assumed saving.

The constraint is **low steady-state use plus bounded peak use while the user's normal apps remain open**. A 16 GB machine can still feel unusable if Adam holds several gigabytes idle or creates a short-lived peak that triggers swapping. Disk model size, per-process RSS, PSS, service cgroup memory, and whole-system available memory answer different questions; none can stand in for all the others.

## What is already measured in this repository

These are the most relevant existing measurements from [the codebase status](codebase-status.md) and [desktop handoff](desktop-handoff-2026-10-05.md), with their original limitations:

| Component/state | Local evidence | Interpretation |
| --- | --- | --- |
| Laptop Adam service, later snapshot | About **1.7 GB current / 1.9 GB peak cgroup memory** | Snapshot from the handoff, not a controlled concurrent-workload peak. |
| Isolated startup after delaying Torch/Transformers imports | Median PSS fell from about **1.10 GiB to 0.62 GiB**; cgroup memory from about **0.83 GiB to 0.55 GiB**, ten samples | Strong evidence for lazy loading at startup; later live service values include other state. |
| ONNX Silero VAD wrapper vs Torch-based wrapper | **42.7 MiB vs 376.8 MiB USS** in isolated processes | Large saving for VAD alone; Torch can still load for enrolled-speaker verification, separation, or diarization. |
| Local `base.en` vs `small.en` Faster-Whisper | **530 MiB vs 917 MiB USS**; median **0.260 s vs 0.650 s**; mean WER **0.140 vs 0.099** on ten generated utterances | Real tradeoff, but synthetic speech is inadequate to choose production ASR. |
| Nemotron diarization, CPU subprocess | About **1.26 GiB PSS** after inference | Optional capability with a meaningful peak; no matched diarization-quality baseline. |
| CPU PP-OCRv6 small | **111–112 MiB USS** after cleanup across repeated reads; **1.48 s** warm median on one synthetic screen | Existing OCR is relatively modest; loading it for every task would still waste time and memory. |
| Separate read-only Edge process for a synthetic report | About **577 MiB additional PSS** while open; returned near baseline after release | Reusing a permitted existing browser surface can be cheaper than launching a browser solely for observation. |
| OmniParser CPU trial | Roughly **0.5–0.75 GB RSS**, about **238 s** for one run | Not viable as the routine CPU path on current evidence; task benefit is unmeasured. |

The [current code](../src/main.py) initializes VAD, wake, transcriber, TTS, and optional speaker capabilities through one service; some models are lazy, but their runtimes may remain mapped after first use. [`speaker.py`](../src/stt/speaker.py) uses SpeechBrain/Torch when an enrolled speaker is checked, [`diarizer.py`](../src/stt/diarizer.py) uses Transformers/Torch for Nemotron, and [`target_separator.py`](../src/stt/target_separator.py) also imports Torch. [`transcriber.py`](../src/stt/transcriber.py) uses CTranslate2 via Faster-Whisper; [`ocr.py`](../src/tools/ocr.py) already bounds OCR work and trims its heap; [`browser_navigation.py`](../src/tools/browser_navigation.py) releases an idle headless profile after a turn. Those details matter more than the mere fact that a package is installed: a dependency on disk costs little RAM until imported and loaded.

## Candidate 1: lifecycle and concurrency control

**Keep the always-on path small.** Wake detection, VAD, audio I/O, and task routing need low startup and steady-state cost. Diarization, source separation, OCR, OmniParser, browser automation, and rich TTS need not all stay resident. Adam already made a successful first move by delaying Torch imports. Next, audit which optional model sessions remain in the daemon after one use and which retain native allocator arenas or thread pools. Put the heaviest seldom-used feature in a worker that can exit after the task or an idle timeout; use a bounded queue so two visual/speech jobs do not create duplicate model loads.

This should be selective. A worker process adds IPC, cold-start latency, and sometimes duplicate runtime memory. Use it where clean process exit actually returns a large mapped model or fragmented heap to the OS; do not split every tiny ONNX component into its own process. A worker for Nemotron/separation is more plausible than one for Silero VAD. An explicit cache policy can keep one frequently used model warm while releasing rare ones. A resource coordinator should know the current model loads and prioritize wake/acknowledgment over background diarization or visual annotation.

[ONNX Runtime documents shared arenas](https://onnxruntime.ai/docs/performance/tune-performance/memory.html) and [thread-pool settings](https://onnxruntime.ai/docs/performance/tune-performance/threading.html). These are candidates for several ONNX sessions *in the same process*, not a guaranteed saving across independent workers. Adam already caps native threads; check per-session pools, spinning, and memory reuse under mixed audio/vision load before tuning them further. A hard cgroup cap is a safety backstop, not an optimization: it can throttle or kill the assistant instead of reducing its required working set. [Linux cgroup v2 exposes `memory.current`, `memory.peak`, and pressure/event counters](https://docs.kernel.org/admin-guide/cgroup-v2.html) for observing this.

**Expected upside:** high for idle and post-task memory; uncertain for cold latency. **Main risk:** first use becomes too slow or overlapping voice tasks queue behind a heavy job. **Priority:** first, because it works with the current models.

## Candidate 2: keep or replace the local ASR engine

Faster-Whisper/CTranslate2 already supports CPU INT8 and bounded CPU threads. Its authors' [published benchmark](https://github.com/SYSTRAN/faster-whisper#benchmark) shows INT8 often reduces memory versus full precision, but the benchmark uses another CPU and long audio; Adam's own short-utterance numbers are more relevant. The first controlled comparison should be **the existing `base.en` and `small.en`** on real microphone utterances, including names, commands, noisy speech, pauses, and varied speakers. A smaller model that misses commands can cause retries and ultimately more work.

[`whisper.cpp`](https://github.com/ggml-org/whisper.cpp) is the strongest alternate runtime to test because it offers a small native dependency footprint and quantized Whisper weights. Its published model memory figures and [quantization support](https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md) are useful for screening, but not evidence of Adam's PSS or transcription quality. Compare the same model, beam/decoding settings, audio inputs, and warmed/cold states against CTranslate2. Quantization may reduce memory at an accuracy cost. Use a bounded native worker or direct library binding only after a fixture comparison; a CLI subprocess per utterance could impose avoidable cold starts.

OpenVINO GenAI also provides a [Whisper pipeline on CPU, GPU, or NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html). This is worth an isolated trial if the laptop's Linux driver exposes the NPU and the exact model converts/runs correctly. [OpenVINO's system requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html) say NPU support needs a compatible driver and OS. Offloading computation may lower CPU occupancy or energy, but can increase startup, conversion effort, device memory, or latency; it does not make model weights disappear from 16 GB of shared system memory. Do not redesign the speech stack around NPU availability until measured on the actual laptop.

Cloud ASR is a resource-saving branch already supported in Adam's provider design. It removes the local ASR model but brings upload latency, availability, cost, and data-handling choices. It may be useful as an optional route for long or difficult audio; it is not inherently the most responsive route for a short wake-to-command cycle. Compare whole voice-turn latency and command accuracy, not model inference time alone.

**Expected upside:** medium to high active memory if a smaller or external ASR meets quality; modest idle upside when local ASR is already lazy. **Priority:** second, with real-microphone evidence.

## Candidate 3: speaker identity and diarization without a permanent Torch footprint

The largest avoidable *runtime family* is PyTorch: Adam's enrolled-speaker checker uses SpeechBrain ECAPA, and meeting/overlap paths can load Nemotron and separation models. [Sherpa-ONNX exposes speaker embedding and verification](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/c-api/docs/speaker-embedding.dox) and [offline speaker diarization](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/sherpa-onnx-offline-speaker-diarization.cc) using ONNX segmentation, embeddings, and clustering. This is a concrete replacement candidate for *some* Torch uses, not a demonstrated drop-in: model outputs, thresholds, enrollment files, and overlap behavior differ. Existing user voice profiles were produced by a particular embedding model; replacing it requires fresh enrollment or a validated compatibility bridge.

Evaluate speaker verification separately from diarization. A small ONNX embedding model may handle “is this the enrolled user?” without loading SpeechBrain/Torch, but must be tested on genuine and impostor voices at multiple microphone distances and background conditions. Offline diarization can be delayed until after a meeting or submitted to a worker; it does not need to block the first response. [Pyannote's open-source pipeline](https://github.com/pyannote/pyannote-audio) is a quality comparator, but it is itself PyTorch based and therefore not automatically a lighter replacement. [`whisper.cpp` tinydiarize](https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md) marks speaker turns with special models; it is a weaker feature than stable speaker identification and should not be treated as equivalent to full diarization.

If Nemotron remains much better for overlapped speech, keep it as a cold, optional worker and preempt/queue it when the user starts a normal command. This preserves capability without forcing its 1.26 GiB observed CPU-process footprint into the usual voice path.

**Expected upside:** potentially high after speaker/meeting use, but unmeasured for Adam. **Risk:** false accepts/rejects or wrong meeting speakers. **Priority:** high-impact research candidate, implementation only after paired quality tests.

## Candidate 4: voice output tiers

Adam already supports Kokoro ONNX and Piper. [Piper's project documents local ONNX voices and streaming raw audio](https://github.com/rhasspy/piper); it is a credible smaller voice path to compare with Kokoro. That original repository is archived and [points to a new upstream](https://github.com/OHF-Voice/piper1-gpl), so a future dependency update needs separate maintenance/license review. The right comparison is not just model-file size: measure first audible sound, synthesis time for short and long utterances, sustained memory, intelligibility, naturalness, and whether the user still wants that voice. Keep a high-quality option for normal speech if a smaller engine sounds unacceptable; use a short prerecorded or low-cost acknowledgment path only if it is genuinely task-relevant and does not mask the full response's delay.

Kokoro's current code creates an ONNX session and prewarms it on first use. Its session may remain in memory afterward. Test on-demand loading versus keeping it warm during a conversation; an idle timeout could reclaim memory while avoiding repeated cold starts during a multi-turn exchange. Cloud TTS can lower local RAM but adds network and recurring cost, so it belongs in the same quality/latency comparison rather than being a default memory fix.

**Expected upside:** medium, depending on measured Kokoro/Piper footprints and voice preference. **Priority:** after ASR and speaker path, because audible quality is a product requirement.

## Candidate 5: visual path and browser reuse

For text-heavy computer use, structured data can save both inference and model turns: the handoff's 56-row report was answered 3/3 with DOM text and 0/3 with screenshots in a small synthetic trial. Adam already has AT-SPI, scoped OCR, and optional CDP. Choose the cheapest source that supplies the needed evidence: app/document API or active-page DOM, then AT-SPI, then scoped OCR, and only then a screenshot plus heavier region detector when geometry or icons matter. The ordering is a **hypothesis to test per task**; a canvas, scanned PDF, or unlabeled UI may require vision immediately. The [Linux control-surface survey](linux-application-control-surfaces-2026-10-05.md) details the available routes.

Avoid launching a new browser for passive observation of an existing user's browser if a permitted extension/debug endpoint or accessibility path is already present. When Adam launches an isolated browser to act, release it after the turn as the current code intends. Do not infer that DOM text is free: a fresh browser measured roughly 577 MiB additional PSS in the local report test. OCR's measured footprint is already far lower than the OmniParser CPU trial; focus on whether a detector changes task success enough to justify its measured peak. Crop and downscale only where text/icon legibility survives.

**Expected upside:** high for browser-heavy tasks and avoiding unnecessary model starts; case-dependent for accuracy. **Priority:** parallel research, with visual comparisons in the testing phase as requested.

## Candidate 6: memory growth when Adam saves many memories

Saving broadly is compatible with a light runtime if retrieval storage is designed around growth. Adam's embedding dimension is [384 float32 values](../src/memory/embedder.py), or **1,536 bytes per stored vector** before metadata and index overhead. That is about **146 MiB at 100,000 memories** and **1.43 GiB at one million** for the dense matrix alone. Current code eliminated a duplicate Python-float copy of loaded vectors and uses adaptive lexical scoring, which helps. The long-term concern is resident matrix/index size and startup/rewrite cost, not a need to discard memories.

Candidate progression: preserve all records durably; keep recent/frequently used metadata and lexical postings readily searchable; mmap or page older vector blocks; search time/date/lexical candidates first; score a bounded vector subset when possible; and only then consider an on-disk ANN index. This is an engineering design hypothesis. It should be driven by recall accuracy and realistic corpus sizes, because an ANN index can add memory and miss facts. SQLite or another store is a persistence choice, not a retrieval-quality solution by itself. The exact crossover depends on access patterns, Python object overhead, and whether the existing embedding model remains resident.

**Expected upside:** low at small stores, critical at hundreds of thousands of memories. **Priority:** establish a corpus-size forecast and recall benchmark before changing storage.

## Candidate comparison and order

| Candidate | Likely resource benefit | Quality/latency risk | Evidence status | First action |
| --- | --- | --- | --- | --- |
| Release heavy optional workers; bound concurrency | High steady-state/post-task | Cold starts and queueing | Lazy-import improvement already measured | Audit loaded models after each feature, then prototype one worker |
| ONNX speaker verification/diarization | Potentially high after use | Voice identity and overlap errors | Official implementations exist; Adam comparison absent | Paired quality + memory study |
| `base.en`/`small.en` and `whisper.cpp` comparison | Medium to high active use | Command misses or startup overhead | Adam synthetic ASR numbers only | Real-microphone suite, same decoding settings |
| Piper vs Kokoro, idle timeout | Medium | Voice quality and first sound | Both are already supported; no matched resource study | Paired short/long utterance sample |
| Active app semantics before new browser/vision process | High for some desktop tasks | Missing visual information | One strong but tiny synthetic DOM result | Matched task study |
| OpenVINO NPU for Whisper/OCR | Unknown | Driver/model compatibility, startup | Official support, no Adam laptop result | Check device in isolated environment, then compare |
| Tiered memory index for large stores | High only at scale | Recall misses and complexity | Vector arithmetic + synthetic index microbenchmarks | Forecast corpus size, benchmark retrieval quality |
| Hard RAM cap or swapping as primary fix | No inherent reduction | Latency spikes or OOM | Kernel controls exist | Use as observability/safety limit only |

## Measurement design for the next phase

Freeze the exact Adam commit and model files. On the laptop, collect a **matched idle baseline**, then wake/listen, one short command, a noisy command, a speaker-verified command, a meeting/diarization task, TTS, scoped OCR, browser DOM, and a full computer-use task. Run each candidate from the same initial service state, both cold and warm. Include overlap cases (TTS while listening, OCR while ASR loads, browser while memory retrieval runs) because their peak can be larger than any isolated feature. Record task correctness and false accepts as well as first acknowledgment, first audible output, p50/p95 completion, CPU time, process-tree PSS/RSS, service `memory.current`/`memory.peak`, system `MemAvailable`, swap activity, and available battery-energy readings. Note which child processes and mapped models are present at every peak. Compare incremental resource use against the same base workload; do not add isolated PSS figures from different experiments as if they were simultaneous.

The laptop's 16 GB is a ceiling, not a target for Adam. A winning candidate must leave the desktop usable, preserve speech and computer-use accuracy, and reduce *measured* idle, peak, or energy use enough to justify any added complexity. Desky experiments remain paused under the handoff, so this document does not imply cross-host validation.
