# Latency improvement proposal

**Status:** Implementation plan. This document reports no measured performance results. The repository has no checked-in reproducible end-to-end latency benchmark.

## Goal and measurement contract

Measure one assistant turn from recorded input through the first audio buffer submitted for playback, and through response completion. A submitted buffer is not proof of sound at the listener; claims about audible onset require loopback or an external microphone measurement. Use `time.monotonic_ns()` for durations and for the power sampler's correlation clock. Never subtract wall-clock timestamps or add durations from overlapping work.

For offline evaluation, define ground-truth speech end as a sample offset in a labeled recording. Map that offset to the capture clock in the playback harness. In live use, the true speech-end instant is unknowable; record the application's endpoint decision and do not label it as ground truth.

### Event schema

Emit one structured JSON event per lifecycle marker, with these fields:

| Field | Meaning |
|---|---|
| `schema_version` | Integer version of this event contract. |
| `trace_id` | Random opaque ID for one user turn; generated before utterance capture. |
| `span_id`, `parent_span_id` | Random IDs linking a stage span to its parent; null parent for the turn span. |
| `event` | Stable name from the required event list below. |
| `clock_ns` | `time.monotonic_ns()` at the event. Comparable across processes on the same host boot; never use it as a wall timestamp. |
| `status` | `started`, `ok`, `error`, `cancelled`, or `skipped`, as applicable. |
| `provider`, `model`, `component` | Configured identifiers, or null where not applicable. Never include credentials or request headers. |
| `attempt` | Zero-based retry/attempt number; omit if not applicable. |
| `attributes` | Small allowlisted, non-content metadata such as audio sample count/rate, tool name, endpoint state, or exception class. No transcript, prompt, tool arguments/results, audio, or image bytes. |

Calculate span duration from matching start/end `clock_ns` values. If an end event is missing, report an incomplete span; do not silently discard it. Keep event names and metadata values bounded to prevent log injection and unbounded cardinality.

Required events (where a stage applies): `turn.started`, `audio.speech_started`, `audio.endpoint_decided`, `stt.started`, `stt.completed`, `llm.request_started`, `llm.first_content` (streaming only), `llm.completed`, `tool.started`, `tool.completed`, `tts.clause_started`, `tts.clause_synthesized`, `playback.buffer_submitted`, `playback.completed`, `turn.completed`, and `turn.cancelled`. Include `tool_name` and outcome status, but not tool payloads. `audio.endpoint_decided` records the selected silence timeout and decision reason/state. For evaluation only, record labeled speech-end sample offset in the dataset sidecar, not as a claimed live measurement.

Report distributions by stage and workload with sample count, p50/p95, failures, missing spans, and configuration (machine, OS/audio stack, device, STT/LLM/TTS providers and models, settings, and code revision). Define each metric as a timestamp difference. In particular: labeled speech end → app endpoint; app endpoint → final STT; app endpoint → LLM first content; labeled speech end → first playback submission; and turn start → playback completion. Keep local buffer-submission latency separate from measured physical acoustic onset.

## Repository behavior and known interface issue

- [`src/audio/stream.py`](../src/audio/stream.py) invokes `on_partial_audio` in the recording thread, at a default 500 ms cadence, once the buffered audio is at least 250 ms. This callback is synchronous: it may run speaker verification and local STT before recording continues. [`src/main.py`](../src/main.py) further skips partial audio shorter than 500 ms. Cloud STT providers (`openai`, `openrouter`, `custom`) return a no-op callback to avoid partial-prefix requests. Instrument callback start/end and errors; its runtime is part of capture/endpoint delay.
- The endpoint tuple-unpack repair is implemented. [`SemanticEndpointer.analyze`](../src/audio/endpoint.py) returns `tuple[EndpointState, float]`; [`src/main.py`](../src/main.py) unpacks the result, checks that the silence duration is numeric, finite, and positive, emits an endpoint-candidate event, and returns the duration. The callback still catches broad `Exception` values and emits an error event; narrowing that catch and adding an explicit tuple/interface check remain proposed hardening. The implementation is:

  ```python
  endpoint_state, target_silence_s = self.endpointer.analyze(
      partial_text, wake_word=wake_word
  )
  ...
  return target_silence_s
  ```

  Keep `EndpointState` for an `endpoint_state` event/diagnostic. Narrow exception handling and log the exception class with the trace ID; do not silently convert programming errors into the default timeout. Consider an explicit tuple/interface check, while preserving the existing finite and positive duration validation. Do not modify the endpoint's heuristic timeout constants as part of this interface repair.
- [`src/llm/provider.py`](../src/llm/provider.py) exposes `chat()` returning a completed dict. Ollama, OpenAI-compatible, Anthropic, and Gemini paths currently request/parse whole completions; the OpenAI-compatible request sends a JSON payload without `stream: true`. It has no common token-stream interface.
- [`src/llm/brain.py`](../src/llm/brain.py) runs a ReAct loop with tool schemas. A provider response can contain both text and tool calls. It speaks a response directly only when content is present and there are no tool calls; after tool execution it commonly makes a final `chat(..., tools=[])` request for a user-facing summary. Do not send text from a tool-enabled response to TTS until the completed response confirms there are no tool calls.
- [`StreamingVoiceSynthesizer.stream_tokens`](../src/tts/streaming.py) accepts an async iterable of strings and splits them on clause punctuation before synthesis. It has no call site. Its existence does not mean LLM token streaming is currently active. Its playback uses the same synthesizer/output methods as ordinary speech; synthesis and playback are not atomic.
- TTS interruption uses an epoch barrier: `advance_epoch()` increments `current_epoch`, terminates the active Piper process when possible, and asynchronously aborts active output; synthesis/playback paths check the epoch. The audio/network generation task must also be cancelled and its response closed. Cancelling TTS alone does not cancel an LLM request.

## Work plan

### 1. Add turn and stage instrumentation

The shared writer and disabled-by-default config are implemented in `src/telemetry/events.py` and `src/config.py`; the default path is `~/.local/state/adam/telemetry/events.jsonl`, overridable by `telemetry.path`. Capture, STT, endpoint candidate, turn, LLM request, tool, TTS, and playback events are wired. The sampler writes a separate versioned JSON document using the same monotonic clock and accepts an optional trace ID. Do not use transcript text as a key. If telemetry is disabled or its writer fails, assistant behavior continues unchanged; issue at most one rate-limited local warning without request content. Remaining coverage and hierarchy gaps should be filled at:

1. Turn creation and VAD speech-start detection in `src/main.py` / `src/audio/stream.py`.
2. Partial callback start/end, including speaker verification, partial STT, endpointer, and speculative preflight dispatch. Distinguish “scheduled” from “completed”; preflight runs as an asyncio task. Current code records endpoint candidate and callback error, not these individual timings.
3. Recording return/endpoint decision; final STT start/completion at the call sites that invoke `_transcribe_stt` after `_record_utterance`.
4. Every `brain.py` LLM `chat` call, including ReAct hops, tool-free summary/retry, and first content/completion for future streaming.
5. Each tool dispatch and completion/failure/cancellation in the existing tool execution loop. Current events include tool name and outcome, with no arguments or results.
6. TTS clause synthesis and playback buffer submission/completion in `src/tts/streaming.py`; `playback.buffer_submitted` must correspond to the first actual `stream.write`/backend submission, not the beginning of synthesis.
7. Turn completion/cancellation on every exit path.

Do not log text, images, raw tool arguments/results, user IDs, or API secrets. Ensure exceptions and cancellations close spans in `finally` blocks. Preserve error reporting without broad catches swallowing instrumented failures.

**Acceptance evidence:** A saved schema example and trace from each workload category (simple reply, tool result, visual/tool result, provider failure, user interruption) has correctly nested spans, no content fields, and no unexplained missing end events. Report sample count and distributions per the measurement contract. These are the first measurements; do not write targets or expected improvements into the proposal until observed.

### 2. Repair and evaluate endpoint callback

The tuple-unpack fix and endpoint-candidate event are implemented. Remaining work is to harden exception handling and record the effective timeout at the actual `record_utterance` break, because a callback's selected timeout is only a request and the recording loop controls when it takes effect. Preserve the configured timeout when there is no valid partial transcript or for providers whose partial callback is intentionally disabled.

Evaluate the current heuristic and adaptive setting on a fixed, licensed/labeled corpus with sample-level speech-end annotations. Include short standalone commands, unfinished connectors, mid-sentence pauses, low-volume/noisy speech, and multiple speakers. Store per-item endpoint decisions and audio/config IDs; report endpoint delay, false early endpoint, late endpoint, and final transcript/task success together. Split development and holdout items before tuning. Do not infer a universal silence timeout from turn-taking research or tune against the holdout set.

**Acceptance evidence:** Confirm the existing callback receives a valid tuple, applies its float timeout, and surfaces a diagnostic event on invalid return/error. The remaining evaluation artifact contains corpus provenance, labels, configuration, per-item decisions, summary counts/distributions, and holdout results. No target value is assumed in advance; a product owner sets an acceptable tradeoff after seeing the measured false-end/latency curve.

### 3. Introduce provider streaming behind a safe interface

Do not connect a provider token iterator directly to `stream_tokens` from the tool-enabled ReAct request. First add a provider capability contract that identifies whether the configured provider/model supports streaming and how it signals final content, tool-call deltas, finish reason, and errors. Keep the current buffered `chat()` path as the compatibility path for providers without verified support.

For the first implementation, stream only the existing tool-free final summary call (`tools=[]`) after tool execution, and optionally tool-free direct-answer requests after the ReAct layer has selected that no action is required. The latter needs an explicit no-tools request/decision in the brain; do not silently remove tools from the current request, because that changes the assistant's ability to act. Provider adapters must aggregate/validate a completed response and preserve the existing response shape for `brain.py`. For OpenAI-compatible SSE, explicitly request streaming and parse incremental `data:` records, `[DONE]`, finish reason, and error bodies; only enable for tested endpoints. Provider-specific streaming support must be documented/configured rather than assumed for every OpenAI-compatible server. Do not claim stream support for Ollama/Anthropic/Gemini until their adapter code and tool/finish semantics are implemented and verified.

TTS clause streaming begins only after a complete text clause is available. Before enqueueing a clause, apply the existing speech cleaning rules to that clause and retain enough parser state not to split inside code/markup constructs. Do not speak model reasoning, tool arguments, partial JSON, or content from a response that finishes with tool calls. Acknowledgment/filler speech is outside this change; it needs separate user-experience evaluation.

**Failure and cancellation contract:**

- If the provider stream fails before any audio buffer is submitted, close the response and use the existing buffered request once as fallback. Record that the answer was regenerated; do not disguise it as continuation.
- If streaming fails after audio was submitted, cancel remaining generation/synthesis, advance the TTS epoch, and do not replay the answer from its beginning. Speak a fixed, truthful recovery line indicating the response was interrupted and ask the user to retry/continue. Mark the turn `partial/error` in telemetry.
- On barge-in, cancel the provider task, close its response/iterator in `finally`, cancel pending clause synthesis, call the TTS epoch barrier, and do not trigger a fallback request or recovery speech over the user's interruption. Propagate `CancelledError` after cleanup.
- Keep all streaming work attached to the active turn task (no detached generator task). Check interruption/epoch again before starting each clause. Provider cancellation may not stop remote computation; record client-side cancellation time without claiming remote cancellation.

**Acceptance evidence:** For each provider marked streaming-capable, captured protocol fixtures or tests cover normal completion, tool-call finish, malformed/incomplete stream, network error before first content, error after a clause, and user barge-in during provider wait, synthesis, and playback. Compare buffered and streaming paths on the same no-tools workload for first token, first playback submission, completion, answer equivalence, task success, and failure recovery. Include provider/model/version and raw timing distributions; do not claim improvement solely because a token arrived earlier.

### 4. Measure visual capture and tool work before optimizing

Use spans to separate screenshot capture, image conversion/upload, model wait, tool execution, OCR/grounding, and follow-up summary. Profile representative supported desktop backends and resolutions before changing capture strategy or subprocess usage. Do not attribute a full stage's duration to a single operation when spans overlap.

**Acceptance evidence:** Workload/configuration, per-stage distributions, correctness/coordinate validation, and failure rates accompany any optimization proposal. Preserve existing full-screen/backend fallback behavior unless a measured and reviewed replacement covers the same cases.

### 5. Tune output buffering on supported devices

[`PIPEWIRE_LATENCY`](https://docs.pipewire.org/page_man_pipewire_1.html) expresses a requested stream latency and does not establish physical output latency. Compare device settings with loopback/acoustic measurements and underrun/error counts. Change defaults only after measurements on supported audio devices and a defined stability criterion.

## Implementation order and exit criteria

1. Land schema/helper and end-to-end buffered-path traces.
2. Fix endpoint tuple handling and establish labeled baseline and tradeoff curve.
3. Add streaming capability/interface and one tested tool-free provider adapter; define failure/cancellation handling before enabling it.
4. Validate traces, answer equivalence, interruption behavior, and measured performance on holdout workloads.
5. Only then propose additional capture, tool, or playback changes from observed bottlenecks.

No latency target or percentage improvement is specified here. Define those only after baseline collection and an explicit product tradeoff review. See [`latency-minimization-research.md`](latency-minimization-research.md) for the limits of published evidence.
