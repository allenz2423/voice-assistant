# Cloud cost and latency: implementation and measurement plan

**Status:** Phase A is implemented for the OpenRouter-compatible route. Telemetry remains disabled by default. No token, price, cache-hit, or latency results are claimed. Offline cost aggregation and runtime optimizations remain future work.

## Current code contract and blockers

- Effective settings are `LLMConfig` in `src/config.py`, overlaid by `config.yaml`: `provider`, `cloud_model`, `api_base`, `provider_only`, and `allow_provider_fallbacks`. `provider: custom` enters `UniversalLLMClient._chat_openai_compatible` in `src/llm/provider.py`.
- `UniversalLLMClient.chat()` returns normalized `role`, `content`, and `tool_calls`; OpenAI-compatible responses can also preserve optional `reasoning_details`. Usage and returned model/request identifiers are omitted from that conversational return value. When telemetry is enabled, the OpenRouter-compatible adapter records available usage and returned model/request identifiers separately in telemetry. `src/llm/brain.py` invokes the client repeatedly for tool rounds and may make final summary/retry calls. Count each provider request; a user turn is not one request.
- OpenAI-compatible, Anthropic, and Gemini code creates a new `aiohttp.ClientSession` per request. All are non-streaming. Current application timing can measure request-to-complete response, not first generated token or connection reuse.
- Anthropic and Gemini request adapters currently do not implement provider-native structured tool submission. Their response parsing is text-based. Do not use tool-heavy Adam tasks to compare those providers until that behavior is implemented and validated. The OpenAI-compatible adapter is the first comparable route.
- Configuration is effective runtime state, not evidence of provider identity. OpenRouter routing can select/fallback among inference providers according to request options; save the redacted effective model and routing options, and use response metadata when available.

## Phase A: per-request usage telemetry (required before cost optimization)

### Files and interfaces

1. `src/llm/provider.py`: The configured OpenRouter-compatible adapter extracts normalized token and cost fields into telemetry; the conversational response shape is unchanged. It does not add usage data to the brain response.
2. `src/telemetry/events.py`: The shared JSONL writer uses `time.monotonic_ns()`, opaque trace/span IDs, a usage-field allowlist, and a rate-limited content-free warning. It is disabled unless configured.
3. `src/config.py`: `TelemetryConfig` defaults to disabled and `~/.local/state/adam/telemetry/events.jsonl`; set `telemetry.path` to override it. The path expands `~` and environment variables. `jobs_log_dir` is unchanged.
4. Each OpenAI-compatible request emits start and completion events when telemetry is enabled. HTTP, provider, parse, and transport failures are recorded without response bodies or exception text. Telemetry write failure does not fail the LLM request. Other compatible endpoints are marked accounting-unsupported.

### Normalized event schema

The intended normalized accounting record for each HTTP/model request (not each user turn) is:

```json
{
  "configured_provider": "custom",
  "configured_model": "...",
  "returned_model": "... or null",
  "provider_request_id": "... or null",
  "accounting_status": "available | missing | error | unsupported",
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cached_input_tokens": null,
    "cache_write_tokens": null,
    "other_units": null,
    "provider_reported_cost": null,
    "currency": null
  },
  "routing": {"provider": null, "region": null},
  "is_streaming": false
}
```

The current event stores available accounting values in `attributes.usage` and `accounting_status` alongside request metadata. It omits unavailable fields instead of writing nulls, and does not yet provide routing, other-unit, or streaming fields. The enclosing shared event provides `schema_version`, `trace_id`, `span_id`, `parent_span_id`, `event`, `clock_ns`, `status`, `provider`, `model`, `component`, and `attributes`. Calculate request duration from its start/completion events.

`null` means unavailable, not zero. Do not infer token counts from character length. The implementation maps OpenRouter `usage.prompt_tokens`, `completion_tokens`, `total_tokens`, optional prompt cache counters, and `usage.cost`. Cost is recorded as `provider_reported_cost` with `currency: "credits"`; it is not a dollar amount. The response `model` and `id` are recorded, but the actual provider route is not inferred. Other compatible endpoints are marked `unsupported`.

For later adapters, use their primary API response contracts: Anthropic `usage.input_tokens`, `output_tokens`, `cache_read_input_tokens`, and `cache_creation_input_tokens`; Gemini `usageMetadata.promptTokenCount`, `candidatesTokenCount`, and `cachedContentTokenCount`; Ollama `prompt_eval_count` and `eval_count` for local runtime diagnostics only. Preserve explicit missing/unsupported states. Citations: [OpenAI API response usage](https://platform.openai.com/docs/api-reference/chat/object), [OpenRouter response format and usage](https://openrouter.ai/docs/api-reference/overview#response-format), [Anthropic Messages response](https://docs.anthropic.com/en/api/messages), [Gemini usage metadata](https://ai.google.dev/api/generate-content#UsageMetadata), and [Ollama chat response](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion).

### Redaction and failure semantics

- Persist allow-listed accounting metadata only. Exclude all prompt/message/tool/image/audio content, tool arguments/results, transcripts, `reasoning_details`, endpoint query parameters, authorization headers, API keys, and exception text that may include request URLs or bodies.
- Normalize provider/model identifiers and request IDs only after validating bounded string length; allow `null`. Do not write API base URLs because they may contain credentials/query parameters. Save only a coarse adapter and operator-supplied deployment label if needed.
- `accounting_status=missing` on a successful OpenRouter response without accounting fields; `unsupported` for an unmapped endpoint; specific error states on transport/provider/parse failure. A successful HTTP response has event `status=ok` even when accounting is unsupported. A model fallback response is not treated as a successful provider response.
- Use local file permissions restricted to the current user. Rotate by size or age with an operator-set retention period. No network exporter in this phase. Telemetry is opt-in, and disabling it stops new writes without affecting requests.
- Avoid duplicate events on retries: each actual outgoing request gets its own event and request ID; aggregate turns later using a random ephemeral turn ID that is not linked to user text. Use a caller-provided correlation ID only if it contains no identity or conversation data.

### Phase A verification still needed

- With telemetry disabled, no trace file is created and provider behavior is unchanged.
- Capture sanitized provider response fixtures and verify normalized fields and failure statuses against them. Do not use live credentials for this check.
- A fixture containing sentinel prompt text, image data URL, fake key, tool arguments, and reasoning content must not appear in the output file. This acceptance check remains to be run.
- Reconcile cost fields with provider billing data before reporting monetary totals. Credit cost must not be converted to currency without a dated rate source.
- Ensure events cover the primary loop at `src/llm/brain.py` and the final summary/retry calls; one event corresponds to one request, including tool loops.

## Phase B: cost calculation (offline and dated)

Do not bake prices into runtime code. Add an analysis script or notebook under `tools/` that reads the schema-versioned JSONL plus an operator-supplied rate card snapshot. Store rates with source URL, retrieval date, currency, model/provider, input/output rate, cached-input rate, cache-write rate, and any unit rates relevant to the API. Keep rates as explicit configuration/data, not claims in docs.

Per request calculate only from reported usage supported by that rate card:

`request cost = input + cached input + cache writes + output + other billable units`

Avoid double-counting cached tokens: verify each vendor's definition for whether cached tokens are a subset of input tokens, then apply the corresponding rate split. Sum request costs by ephemeral turn ID only when all requests have complete compatible usage/rates; otherwise report partial/unavailable cost and coverage. Never treat missing usage as zero. Include a manual reconciliation against provider billing exports for a fixed window before publishing cost totals.

Vendor rates and cache rules change; cite and timestamp the actual selected model's first-party pricing page for each analysis. Examples of primary docs, not universal configuration: [DeepSeek pricing and cache fields](https://api-docs.deepseek.com/quick_start/pricing/), [Anthropic pricing](https://docs.anthropic.com/en/docs/about-claude/pricing), and [OpenRouter pricing/usage](https://openrouter.ai/docs/faq#how-does-pricing-work). Prompt caching is provider-specific and may not be guaranteed: [DeepSeek context caching](https://api-docs.deepseek.com/guides/kv_cache), [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), and [OpenRouter caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching).

## Phase C: latency instrumentation

Add monotonic stage timestamps at the call boundaries, with no content logging:

| Field | Location/definition |
|---|---|
| `speech_end` | Caller in `src/main.py`, timestamp when its selected utterance-finalization path completes. Record the chosen path/config; this is not necessarily the true physical end of speech. |
| `request_start` and `response_complete` | `UniversalLLMClient.chat()` around each provider adapter call. Duration includes current session setup and complete body parse. |
| `response_headers` and `first_token` | Not available from current non-streaming adapter. Add only with an intentional streaming interface and provider event parsing; first body byte is not necessarily a usable token. |
| `tool_start` / `tool_end` | Existing tool dispatch loop in `src/llm/brain.py`; record each tool name from an allow-list and outcome enum, not arguments or result text. |
| `first_audio` / `audio_complete` | At the playback submission boundaries in `src/tts/streaming.py`; distinguish buffer queued/submitted from acoustically audible without loopback/hardware measurement. |

Emit request duration even if latency tracing is enabled without usage accounting. Use separate per-request and per-turn summaries; tool/model stages can overlap or nest, so do not add nested durations. Initial benchmark should report request-to-full-response and speech-end-to-audio-submission only. Implement streaming and shared HTTP session reuse as separate candidates after baseline; session reuse must have explicit close/lifecycle ownership and timeout behavior.

## Optimization experiments after instrumentation

1. **Visual context:** The existing tools are selected from the tool registry; there is no `get_quick_ocr_summary` helper. Measure screenshot/OCR use, input usage, provider request count, and visual-task success. Change tool policy only after a fixed visual/nonvisual set proves the image is unnecessary for a specified class of request.
2. **Context/caching:** Measure actual message/token usage first. Any cache behavior or cache metadata is provider-specific. Prototype only for a provider/model whose API documents the mechanism; confirm cache-read/write usage fields and total cost before retaining it. Do not reorder conversation messages in a way that changes semantics.
3. **Deterministic spoken acknowledgements:** Only replace a model completion after the tool outcome contract is explicit. `src/llm/brain.py` receives structured `ComputerControlResult` statuses and other tools may return strings/partial results; define per-tool success, failure, uncertain, and partial wording. Never announce completion before verified success. Compare user task success and recovery rate.
4. **Parallel tool calls:** `src/llm/brain.py` currently dispatches calls in sequence. Declare independence for a specific tool pair, ensure neither reads state changed by the other, and preserve order for dependent calls. Compare correctness and partial-failure recovery; do not infer concurrency from a model returning multiple calls.
5. **Streaming/session reuse:** Current adapters are non-streaming and create a session per request. Implement only as separate changes with provider-specific response parsing, cancellation/timeout tests, guaranteed session close, and exact TTFT semantics. Do not claim first-token improvement from non-streaming duration.

## Benchmark runbook and operator inputs

1. Operator chooses a task corpus covering simple answer, local tool, visual task, multi-tool task, failure, and partial completion. Remove personal data or obtain permission; use the same corpus for each candidate.
2. Record code revision and effective config with secrets removed; pin provider/model/routing, temperature, context, tool definitions, region/network, and warm/cold status. Keep provider routing fixed where the vendor allows it; record actual returned route when available.
3. Before runs, define the minimum cost/latency improvement and maximum task-success/partial-completion regression. Run baseline then one change at a time with enough repetitions to report distribution and sample count; no universal sample count is asserted here.
4. Collect per-request records and aggregate turns. Report p50/p95 latency, cost coverage, request count, tool success/partial/failure, and task quality. Include missing-usage count and errors. Compare billing export for a fixed period before using results for budgets.
5. Retain raw telemetry locally only for the agreed retention window; publish redacted aggregates and reproducibility details. Never publish prompt content, private screenshots, credentials, or transcripts.

## Known prerequisite/configuration issue

Keep local credentials private and out of documentation, telemetry, and shared benchmark artifacts. Supply credentials through an environment variable or secrets store. Also confirm that the `provider_only` value in the effective config is valid for the selected OpenRouter model and intended route before a benchmark; the actual route and cost cannot be assumed from the requested model alone.
