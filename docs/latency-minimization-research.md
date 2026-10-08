# Latency minimization: research notes

**Status:** Literature and platform-documentation review, separate from application measurements. This repository has no reproducible end-to-end latency benchmark artifact. Findings below motivate experiments; none supplies a product latency target for this assistant.

## Human turn-taking does not set an assistant deadline

Levinson and Torreira review conversational turn transitions often on the order of 200 ms while discussing language-production latencies over 600 ms. The article explains how conversational participants prepare responses before a turn ends. These are observations and processing arguments about human conversation; they do not establish a voice-assistant TTFA target or the preferred silence timeout for this recognizer ([Levinson & Torreira, 2015](https://doi.org/10.3389/fpsyg.2015.00731)).

**Engineering use:** This work supports investigating overlap and preparation where the action is safe. It does not justify translating human turn gaps into endpointing constants, 600 ms/1,200 ms breakdown cutoffs, or a promise that the assistant should answer within 200 ms. Select endpoint thresholds using labeled assistant audio and a measured false-end/late-end tradeoff.

## Conversational fillers are an intervention to evaluate

Boukaram, Ziadee, and Sakr evaluate conversational fillers as a way to mitigate delayed responses from a virtual agent in their study conditions ([HAI 2021 paper](https://doi.org/10.1145/3472307.3484181)). That study does not establish that fillers make any assistant feel instant, that one phrase works across tasks/users, or that fillers reduce objective system latency.

**Engineering use:** A separate controlled evaluation may compare no filler with a short acknowledgment, measuring perceived delay alongside trust, interruption, task completion, and annoyance. The filler must not assert that a tool succeeded before its result is known. Onset, wording, repetition, and whether to use fillers are product questions, not constants supplied by the paper. A newer LLM-agent study exists, but its virtual-reality setting and experimental conditions also cannot be applied directly as a latency target for this desktop assistant ([CUI 2025 study](https://doi.org/10.1145/3719160.3736636)).

## Streaming only helps if the whole path supports it

The repository's TTS class has a clause-consuming async iterator, but the active LLM client returns completed response dictionaries. Therefore, currently there is no LLM-to-TTS token stream to measure. To claim an overlap benefit, instrumentation must separately record provider request start, first content event, clause readiness, synthesis completion, first playback buffer submission, and completion.

Tool-enabled generation requires special treatment: the brain uses ReAct and may receive both text and tool calls in a completed response. Speaking a partial text prefix before finish/tool-call semantics are known could expose tool-oriented content or claim success before execution. A constrained first step is streaming a final tool-free answer. Any broader use needs provider-specific incremental tool-call parsing and a policy for when text is safe to speak.

Cancellation must cover both sides of the pipeline. Cancelling local TTS or advancing its playback epoch does not by itself cancel the network request. Conversely, a provider stream can fail after a clause has played, when restarting a full buffered answer would repeat spoken content. Define before implementation whether to use a pre-audio buffered fallback and a distinct post-audio recovery path; record partial answers as partial, not successful completion. Even a closed client stream does not prove a provider stopped remote computation.

## Endpointing and audio buffer settings

Human conversation findings do not validate a specific VAD silence duration for this application. The caller unpacks the semantic analyzer's `(state, silence_seconds)` tuple, validates that the timeout is finite and positive, emits an endpoint-candidate event, and returns it to the recording loop. The callback's broad exception handler and event coverage at the actual recording-loop endpoint remain areas for hardening. The partial callback still performs synchronous speaker verification/STT on the recording thread, so measure that work as a possible source of capture/endpoint delay.

PipeWire's `PIPEWIRE_LATENCY` is a stream latency request and graph configuration affects the result; it does not guarantee a physical DAC or end-to-end microphone-to-speaker latency ([PipeWire `pipewire(1)` manual](https://docs.pipewire.org/page_man_pipewire_1.html)). Validate local buffer submission separately from acoustic loopback and count underruns when tuning.

## Claims unsupported by these sources

The cited work does not support universal filler-onset targets, an “instant” response promise, fixed endpoint silence thresholds, machine-specific latency waterfalls, guaranteed response continuation after a separately spoken filler, or a cancellation bound inferred from the presence of an async task. These require measurements under stated conditions or a user study whose population/tasks match the product question.

## Evaluation protocol

1. Define and log speech start, labeled speech end (evaluation only), application endpoint, final transcript, provider first content, tool start/end, TTS clause readiness, playback buffer submission, and acoustic first sound. Store monotonic event times and correlate them to one turn ID.
2. Keep buffer submission distinct from audible output. Use loopback or an external microphone for physical onset claims.
3. Compare the buffered baseline against one change at a time: endpoint policy, provider streaming, or a separately designed acknowledgment intervention.
4. Use fixed workloads covering simple replies, tools, visual tasks, errors, and interruptions. For endpoint tuning use labeled audio and report false early endpoints as well as delayed endpoints. Keep a holdout set out of tuning.
5. Report sample count, p50/p95, workload/configuration, failures and partial outcomes. For user-perception results, describe participants, tasks, conditions, and uncertainty; do not substitute literature averages for assistant measurements.
6. Publish schema and anonymized traces without transcripts, credentials, tool arguments/results, or audio/image content. Never present synthetic trace examples as observations.

## References

- Levinson, S. C., & Torreira, F. (2015). [Timing in turn-taking and its implications for processing models of language](https://doi.org/10.3389/fpsyg.2015.00731). *Frontiers in Psychology*, 6, 731.
- Boukaram, H.-A., Ziadee, M., & Sakr, M. F. (2021). [Mitigating the Effects of Delayed Virtual Agent Response Time Using Conversational Fillers](https://doi.org/10.1145/3472307.3484181). Proceedings of the 9th International Conference on Human-Agent Interaction, 130–138.
- [Mitigating Response Delays in Free-Form Conversations with LLM-powered Intelligent Virtual Agents](https://doi.org/10.1145/3719160.3736636). Proceedings of the 7th ACM Conference on Conversational User Interfaces (2025). Study context is embodied agents in virtual reality; findings should not be assumed to transfer to this assistant.
- PipeWire developers. [`pipewire(1)` manual](https://docs.pipewire.org/page_man_pipewire_1.html), stream latency options.
