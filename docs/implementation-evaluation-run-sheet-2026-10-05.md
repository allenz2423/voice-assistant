# Adam implementation evaluation run sheet

This sheet defines repeatable Loop 0/1 tasks for the October 5 implementation plan. Fixtures contain synthetic data only. A fixture folder is private to its owner and disposable; preserve raw results only after checking they contain no prompts, transcripts, screenshots, action arguments, credentials, or unrelated personal content.

## Setup and run identity

Create a new fixture folder for each independent run series:

```bash
python tools/create_implementation_fixtures.py \
  --output-root /tmp/adam-implementation-fixtures \
  --run-id <unique-run-id>
```

Treat `expected.json` as the task specification and `invoices.json` as the independent row-level oracle. The report dates are generated for the run date in the host's IANA timezone. Never re-use a fixture folder: the generator fails if its target already exists.

For each sample, record a fresh opaque run ID, commit, host label, OS, app/build, provider/model route, non-secret config identifier, scenario, fixture ID, cold/warm/recovery phase, starting state, exact prompt, and objective oracle. Do not record config contents or secret values. Preserve each sample's raw outcome and classify failures as observation, target/focus, action, app delay, lost goal, blocked permission, provider transport, resource pressure, fixture ambiguity, or false success.

## Tasks and objective outcomes

Use the exact prompts in `expected.json`; replace `<LOCAL_REPORT_URL>` with a URL served from `127.0.0.1` for the generated report. For each local browser/GUI trial, serve only the fixture directory on loopback. Do not expose a fixture server to the network.

| ID | Task and exact prompt | Pass oracle |
| --- | --- | --- |
| F1 | Factual: `prompts.factual` (“Compare rigatoni and penne in two short sentences.”) | Apply the explicit review rubric in `expected.json.oracles.factual`. Record the first task-relevant response separately from any earcon/spinner. |
| F2 | File: `prompts.file_status` | Exact first line and UTF-8 byte length match `oracles.file_status`; source hash remains unchanged. |
| F3 | System: `prompts.system_status` | Logical CPU count matches `os.cpu_count()` (or `multiprocessing.cpu_count()` if unavailable); rounded memory percentage matches a contemporaneous `/proc/meminfo` calculation `(MemTotal - MemAvailable) / MemTotal * 100`, within 1 percentage point. Adam's current status tool prints an integer percentage, so retain `MemTotal` and `MemAvailable` as the raw inputs. |
| M1 | Seed only `oracles.memory.records` into a new memory store via `MemoryManager.save`; use `prompts.memory.day`, `.week`, `.year`, and `.timezone` verbatim. | Day returns the exact date and 9:30–10:15 a.m. local interval; last week returns the two current fixture records and excludes the prior-year distractor; last year returns only the prior-year record; timezone names the fixture IANA zone. Compare against all three expected texts and `expected_record_dates`. |
| G1 | Open `settings.html`; prompt: `prompts.settings` | Fixture output reads `Compact=on; Sync=off; Notifications=off`. Re-read the page after interaction; a click alone does not pass. |
| G2 | Open `scratchpad.html`; prompt: `prompts.scratchpad` | The fixture displays `Saved: Call the dentist Tuesday at 2 pm`; the exact text must be present in the saved-state output, not only in the input field. |
| B1 | Browser: with the synthetic report already open in Adam's isolated browser, use `prompts.browser_template` with its loopback report URL inserted | Supplier, both invoice IDs, and total match `oracles.invoices`; answer must use all 56 report rows. |
| T1 | Terminal: `prompts.terminal` | `answer.txt` equals `3\n`; `source.txt` still has the recorded SHA-256 and byte length. |
| D1 | Document: `prompts.document` | Reopened file equals `oracles.document.final_text` byte for byte; the owner line and final newline remain unchanged. |
| W1 | Multi-step desktop work order: open the 56-row report in an isolated browser and `work-order.txt` in a disposable document editor; use `prompts.work_order_template` with the loopback report URL inserted. | Read all 56 rows, calculate the largest overdue supplier and its invoice IDs, edit the four requested work-order fields, save, then reopen. The reopened file equals `oracles.work_order.final_text` byte for byte, and the invoice report retains its recorded SHA-256. A spoken claim or changed editor buffer alone does not pass. |
| X1 | Mixed: `prompts.mixed` | `task-output.txt` equals `oracles.mixed_output_line`; CPU/memory status matches a contemporaneous system snapshot. Record which browser, filesystem, and system-status tools were available to the model. |

`expected.json` contains the concrete prompt strings, fixture paths, starting states, and computed outputs for the selected run. Keep each task in its own trial directory when the UI or agent can mutate state. Reset the source/document/work-order/settings files from the generator before each repetition; never rely on the agent's final prose as the only success signal. For W1, start with the report and work order visible in separate fixture-only windows, record both window identities, and require fresh scoped observation after switching windows. Compare the independently recomputed winner from the rendered report with the reopened work-order bytes.

### F1 factual scoring rubric

Keep the exact prompt unchanged: “Compare rigatoni and penne in two short sentences.” Score the answer as a pass when it uses exactly two short complete sentences, accurately distinguishes the shapes, includes a correct texture or sauce comparison, and triggers none of `expected.json.oracles.factual.reject_if`. Review for factual correctness, not exact wording; the examples in `accepted_examples` are illustrative, not exhaustive.

Penne variants are a source-backed caveat, not a required part of the answer: Barilla describes smooth Penne Lisce and ridged Penne Rigate. A generic penne description can pass if it avoids universal surface claims. Reject an unqualified claim that penne in general are ridged or are smooth, a categorical claim that penne are lighter or smoother in sauce pairing than rigatoni, or an exclusive sauce rule. Sauce recommendations describe suitability; they do not exclude other pairings. For future reviews, an unqualified statement that penne is "better for lighter sauces" triggers the existing `categorical_sauce_comparison` rejection, even when the preceding sentence correctly describes rigatoni with a hearty sauce. A qualified suggestion that penne can pair with a lighter sauce is not automatically rejected; the answer still needs an accurate comparison and every other rubric criterion. Keep earlier run classifications and oracle versions attached to their original records rather than silently rescoring them.

Use these official Barilla references for independent review: [Rigatoni](https://www.barilla.com/en-us/products/pasta/classic-blue-box/rigatoni), [Penne Rigate](https://www.barilla.com/en-ca/products/pasta/classic-blue-box/penne-rigate), [Penne Lisce](https://www.barilla.com/en-ca/products/pasta/classic-blue-box/penne-lisce), and the [pasta and sauce pairing guide](https://www.barilla.com/en-us/help-with/pasta-kitchen-tips/pasta-sauce-pairing-guide). Under this clarified rubric, the three outcomes audited in Checkpoint 53 remain failures: each was recorded as making an unqualified ridged-penne claim and a categorical sauce/texture comparison. No raw provider response is copied here.

### Headless F1 text capture

For a provider-backed F1 baseline that does not need Adam's daemon, microphone, desktop, or audio playback, use the isolated text-run CLI after the route metadata gate is open. This is a headless Brain-call proxy and remains unscored until a reviewer applies the F1 oracle; it cannot measure request-end-to-verified-state or audible acknowledgment. No live inference is run by the CLI's tests.

From the repository root, create a new owner-only directory and choose a fresh opaque run ID for each invocation:

```bash
umask 077
mkdir -m 700 /tmp/adam-f1-<series-id>
uv run --no-sync python -m tools.run_implementation_factual_trial \
  --config /path/to/existing/protected/config.yaml \
  --prompt "Compare rigatoni and penne in two short sentences." \
  --fixture-run-id <series-id> \
  --run-id <unique-run-id> \
  --config-id <non-secret-stable-route-label> \
  --output /tmp/adam-f1-<series-id>/runs.jsonl \
  --expected-model deepseek/deepseek-v4.1-flash \
  --metadata-http-status 200 \
  --metadata-model-id deepseek/deepseek-v4.1-flash \
  --metadata-endpoint-count 31 \
  --metadata-checked-at 2026-10-06T04:27:13Z \
  --route-ready-confirmed \
  --confirm-provider-inference
```

The config path must already exist and name a regular, non-symlink file owned by the invoking user with no group/other permissions (mode 0600 or stricter). The latest metadata-only check returned HTTP 200 for model ID `deepseek/deepseek-v4.1-flash` with 31 serving endpoints at `2026-10-06T04:27:13Z`. Refresh that check before a later run: the CLI accepts metadata no more than one hour old. Supply the intended model in `--expected-model` and copy the metadata response's exact model ID, endpoint count, HTTP status, and UTC check time into the corresponding options. The runner requires `custom`, exact equality between the expected model, configured `cloud_model`, and metadata model ID, the exact API base `https://openrouter.ai/api/v1`, a successful HTTP status, at least one endpoint, and `allow_provider_fallbacks: false`; it records the attestation and configured `provider_only`. The model remains an explicit exact ID; the API base cannot be redirected by these options. `--route-ready-confirmed` confirms the supplied metadata attestation, while `--confirm-provider-inference` separately authorizes one live factual turn. The runner loads configured credentials for the provider client and error redaction, and never serializes or prints them. It pins the exact F1 prompt, disables memory, skills, custom/model tools, desktop, and browser for the in-memory Brain copy, and captures TTS text without playback. Any unexpected model tool call is recorded and stopped before dispatch.

The run JSONL is append-only under that private directory; each companion `<run-id>.events.jsonl` is created exclusively. Both files are mode 0600, and the parent directory must be owned by the current user with mode 0700 or stricter. Keep them private: they include the exact public fixture prompt, normalized model output, provider usage/cost when reported, route/run identifiers, and operational telemetry. HTTP error response bodies are not available from `UniversalLLMClient`; errors retain safe status/accounting metadata and a reason for unavailable raw detail. Missing provider accounting remains null with reasons. The record stays `outcome: not_scored`, leaves audible acknowledgment and verified-completion timing null, and reports only `headless_brain_call`, the measured `AdamBrain.process_user_utterance` interval; `brain_final_text_submission` is a text-capture timing point. These fields do not measure CLI setup/cleanup, voice latency, or task completion latency. Apply the F1 shape/sauce rubric manually and attach any independently measured completion evidence separately. The CLI does not repeat automatically; retain each raw record, inspect reported cost/retries, then decide whether another sample fits the route's authorized budget.

### F1 CLI startup checkpoint (2026-10-06)

Two attempts stopped before a provider-backed F1 result. Metadata-only checks at `2026-10-06T04:18:53Z` and `2026-10-06T04:27:13Z` each returned HTTP 200 for `deepseek/deepseek-v4.1-flash` with 31 serving endpoints; these checks did not make inference requests.

The first attempt invoked the absolute script path once through `uv run python` and exited 1. Its wrapper captured but discarded stdout and stderr, and its mode-0700 output directory contained neither the run JSONL nor the event log. The exact exception is unconfirmed. The runner creates those files before constructing Brain or calling the provider, so the local artifacts support that execution stopped before a provider request; no server-side receipt was collected.

The second attempt invoked the direct script through `uv run --no-sync python` and exited 1. Private stderr showed `ModuleNotFoundError: No module named 'src'` while importing `src.config`, before argument parsing and runner setup. That failure could not issue a provider request. Its diagnostics remain in owner-only local files; raw diagnostics and private paths are not part of this report. Neither attempt produced an inference sample.

## Headless F2 file-status capture

For a provider-backed F2 baseline without Adam's daemon, microphone, desktop capture, or audio playback, generate a fresh fixture and use its `expected.json` as the sole prompt and source oracle. The runner requires the exact generated prompt/path, checks the source bytes against the fixture's recorded SHA-256, byte-length, and first-line oracles, and requires a private fixture directory. It offers only `read_file` and permits one dispatch to that exact `source.txt` path with at most 512 characters. Brain is limited to two logical calls to its configured primary `UniversalLLMClient.chat`; the current provider retry layer can make up to three retries (four HTTP attempts) for each logical call, so there can be up to eight client-issued HTTP attempts. The private record separates logical chat calls from observed HTTP-attempt and retry-event telemetry; provider-side routing or retries beyond Adam's client are not observable. The record includes the normalized provider answer, tool trace, and before/after source hashes. Objective readback and immutability evidence are reported separately, while the natural-language answer is always marked `manual_review_required`; substring presence is never scored as success. This measures a headless Brain call and file-state evidence, not user request-end timing or audible acknowledgment.

The operator must perform a fresh external metadata-only model check immediately before the trial and report its exact configured model ID, HTTP 200 status, positive endpoint count, and UTC check time. The CLI validates those supplied values and rejects a timestamp older than one hour, but it does not fetch or cryptographically verify the metadata; `--route-ready-confirmed` records the operator's confirmation. Keep the existing config file owner-only (0600 or stricter), use `custom`, the pinned OpenRouter API base, and `allow_provider_fallbacks: false`. Route confirmation and live inference are separate required flags:

```bash
umask 077
python tools/create_implementation_fixtures.py \
  --output-root /tmp/adam-implementation-fixtures \
  --run-id <fresh-fixture-id>
mkdir -m 700 /tmp/adam-f2-<series-id>
uv run --no-sync python -m tools.run_implementation_file_status_trial \
  --fixture-dir /tmp/adam-implementation-fixtures/<fresh-fixture-id> \
  --config /path/to/existing/protected/config.yaml \
  --run-id <unique-opaque-run-id> \
  --config-id <non-secret-stable-route-label> \
  --output /tmp/adam-f2-<series-id>/runs.jsonl \
  --expected-model <exact-provider-model-id> \
  --metadata-http-status 200 \
  --metadata-model-id <exact-provider-model-id> \
  --metadata-endpoint-count <positive-count> \
  --metadata-checked-at <fresh-ISO-8601-UTC-time-ending-in-Z> \
  --route-ready-confirmed \
  --confirm-provider-inference
```

The fixture path is the runner's only task input; do not substitute a prompt or another file. Preserve the generated fixture unchanged for the trial. Do not enable service startup, microphone capture, desktop control, or TTS playback for this headless run.

### Headless F3 system-status capture

Generate a fresh owner-private fixture and use its exact `prompts.system_status` text: “According to a fresh system status check, how many logical CPU cores does this system have and what percentage of memory is in use?” The F3 runner requires the generated fixture directory and prompt, disables memory, skills, custom tools, desktop, browser, secondary tool-free routing, and playback, and offers only `get_system_status`. It allows one empty-object dispatch and at most two logical calls to the primary provider. Immediately around the dispatch it records timestamped logical CPU count plus raw `/proc/meminfo` `MemTotal` and `MemAvailable` KiB values, and derives contemporaneous memory use; the tool's full result, normalized provider answer, and allowlisted request/retry/usage telemetry stay in the private run record and companion event file. The snapshot comparison is evidence for review only: every run remains `manual_review_required`, even when the readback aligns with both snapshots. This is a headless Brain call and does not measure user request-end timing, independent task completion, or audible acknowledgment.

Before each provider-backed run, perform a fresh external metadata-only check for the exact configured model and supply its HTTP 200 status, exact model ID, positive endpoint count, and UTC timestamp. The runner validates that operator-reported attestation and requires the protected config, `custom`, the pinned OpenRouter API base, and fallbacks disabled; it does not fetch or cryptographically verify metadata itself. Route confirmation and live inference are separate required flags. Use a new fixture ID and run ID, and keep fixture and output directories owner-only (0700 or stricter); the config and output/event files must be owner-only (0600 or stricter). The runner records provider-reported token and cost totals when complete and records reasons when accounting is unavailable. Retain all attempts, errors, and retry evidence; independently review the answer and do not promote the F3 result to an automatic pass.

```bash
umask 077
python tools/create_implementation_fixtures.py \
  --output-root /path/to/private/implementation-fixtures \
  --run-id <fresh-fixture-id>
mkdir -m 700 /path/to/private/f3-runs
uv run --no-sync python -m tools.run_implementation_system_status_trial \
  --fixture-dir /path/to/private/implementation-fixtures/<fresh-fixture-id> \
  --config /path/to/existing/protected/config.yaml \
  --run-id <unique-opaque-run-id> \
  --config-id <non-secret-stable-route-label> \
  --output /path/to/private/f3-runs/runs.jsonl \
  --expected-model <exact-provider-model-id> \
  --metadata-http-status 200 \
  --metadata-model-id <exact-provider-model-id> \
  --metadata-endpoint-count <positive-count> \
  --metadata-checked-at <fresh-ISO-8601-UTC-time-ending-in-Z> \
  --route-ready-confirmed \
  --confirm-provider-inference
```

### Headless T1 terminal capture

For a provider-backed T1 sample, generate a fresh owner-private fixture and use its exact `prompts.terminal` text and `oracles.terminal` values. The runner requires `answer.txt` to be absent, checks the source SHA-256 and independently recomputes the `ITEM:` line count before inference, and offers only one `run_bash_command` schema. The tool description names `source.txt`, `answer.txt`, and the requested operation generically; it does not disclose the ready-made command or absolute paths, so the model must construct the command from the task and fixture path. The runner accepts only either exact generated write spelling: `grep -c '^ITEM:' <source.txt> > <answer.txt>`, or `cd <fixture-dir> && grep -c '^ITEM:' source.txt > answer.txt`. After a successful write, it accepts only `cat <absolute-answer.txt>` or `cd <fixture-dir> && cat answer.txt` as read-only verification. Bare `cat answer.txt` remains refused. It allows at most three terminal tool-call attempts, one successful write, and four logical provider calls. Refused calls have no side effects and may be corrected within the remaining attempt limit. The runner interprets accepted operations in process without starting a shell, reads the verification file with a no-follow fixture-scoped open, and refuses all other commands and paths. The objective oracle checks the one successful fixture write, exact `answer.txt` bytes, and unchanged `source.txt`; refused or extra calls and attempt-limit status are reported separately under `dispatch_policy`. A safe readback does not count as an extra write or fail the file-state oracle. The natural-language response is retained for review. This measures model tool selection/command construction and the resulting fixture state, not general terminal or shell execution.

Before each provider-backed run, perform a fresh external metadata-only check for the exact configured model and supply its HTTP 200 status, exact model ID, positive endpoint count, and UTC timestamp. The runner validates that operator-reported attestation and requires an existing owner-only config, `custom`, the pinned OpenRouter API base, and provider fallbacks disabled; it does not fetch or cryptographically verify metadata itself. Route confirmation and live inference are separate required flags. Use a new fixture ID and run ID, keep the fixture and output directories owner-only (0700 or stricter), and keep the config and output/event files owner-only (0600 or stricter). The runner captures TTS text without playback, disables memory, skills, custom tools, desktop, browser, and secondary tool-free routing, and records the exact answer bytes, source hash, provider responses, tool trace, and allowlisted request/retry telemetry in private files.

```bash
umask 077
python tools/create_implementation_fixtures.py \
  --output-root /path/to/private/implementation-fixtures \
  --run-id <fresh-fixture-id>
mkdir -m 700 /path/to/private/t1-runs
uv run --no-sync python -m tools.run_implementation_terminal_trial \
  --fixture-dir /path/to/private/implementation-fixtures/<fresh-fixture-id> \
  --config /path/to/existing/protected/config.yaml \
  --run-id <unique-opaque-run-id> \
  --config-id <non-secret-stable-route-label> \
  --output /path/to/private/t1-runs/runs.jsonl \
  --expected-model <exact-provider-model-id> \
  --metadata-http-status 200 \
  --metadata-model-id <exact-provider-model-id> \
  --metadata-endpoint-count <positive-count> \
  --metadata-checked-at <fresh-ISO-8601-UTC-time-ending-in-Z> \
  --route-ready-confirmed \
  --confirm-provider-inference
```

### Headless D1 document capture

For one provider-backed D1 sample, generate a fresh owner-private fixture and let the runner take the exact `prompts.document`, `paths.document_file`, and `oracles.document` from its `expected.json`. The exact prompt is `In "<absolute document.txt path>", change DRAFT to FINAL and save it. Leave every other character unchanged.` The independent pass oracle is that the saved file's raw bytes equal `oracles.document.final_text` exactly (`STATUS: FINAL\nOWNER: Adam\n`), preserving the owner line and final newline. The runner checks the starting bytes and oracle before inference and checks final bytes and `expected.json` integrity after the Brain call; it does not score the final answer as proof of the file state.

The runner exposes only `read_file` and `write_file` schemas for the exact generated `document.txt` path. Model reads are optional and bounded to two dispatches; the runner permits at most one write, and accepts only `overwrite=true` with content exactly equal to the generated final text. Fixture reads and the atomic file replacement use guarded in-process directory-descriptor handlers: the runner opens the fixture path component by component without following symlinks, uses `O_NOFOLLOW` for fixture entries, and replaces only `document.txt` relative to the pinned directory descriptor. It does not delegate these operations to Adam's normal filesystem handlers. Other paths, symlinks, and disallowed write content are refused without following or modifying them; an unexpected starting state is rejected before inference.

The private record retains the normalized answer, provider/tool trace, and allowlisted request/retry/usage/cost telemetry, including missing-accounting reasons, for manual review. Keep the record and event log private. Prose remains `manual_review_required`, even when the objective byte oracle passes. This headless capture measures the D1 Brain and guarded file-state path. It does not measure broad tool selection, a visible editor or GUI save/reopen flow, request-end-to-verified-state time, voice latency, audible acknowledgment, or laptop resource use; treat those as separate evidence.

Before each provider-backed sample, perform a fresh external metadata-only check for the intended model. The runner requires an existing regular, non-symlink config owned by the current user and private to that user (0600 or stricter), provider `custom`, API base `https://openrouter.ai/api/v1`, the exact configured model ID, provider fallbacks disabled, and an operator-reported HTTP 200 response with a positive endpoint count and UTC check time no more than one hour old. It validates the supplied attestation but does not fetch or cryptographically verify metadata. Route readiness and live inference require separate flags. Use a new fixture and run ID for every sample, keep fixture and output directories owner-only (0700 or stricter), and keep the config and output/event files owner-only (0600 or stricter). The fixture directory is the runner's only task input; do not substitute a prompt or path.

```bash
umask 077
python tools/create_implementation_fixtures.py \
  --output-root /path/to/private/implementation-fixtures \
  --run-id <fresh-fixture-id>
mkdir -m 700 /path/to/private/d1-runs
uv run --no-sync python -m tools.run_implementation_document_trial \
  --fixture-dir /path/to/private/implementation-fixtures/<fresh-fixture-id> \
  --config /path/to/existing/protected/config.yaml \
  --run-id <unique-opaque-run-id> \
  --config-id <non-secret-stable-route-label> \
  --output /path/to/private/d1-runs/runs.jsonl \
  --expected-model <exact-provider-model-id> \
  --metadata-http-status 200 \
  --metadata-model-id <exact-provider-model-id> \
  --metadata-endpoint-count <positive-count> \
  --metadata-checked-at <fresh-ISO-8601-UTC-time-ending-in-Z> \
  --route-ready-confirmed \
  --confirm-provider-inference
```

### Headless M1 memory capture

For one provider-backed M1 sample, generate a fresh fixture today and select exactly one of `day`, `week`, `year`, or `timezone`; the runner loads that prompt and its oracle directly from `expected.json`. Use a new opaque run ID for each invocation and an existing owner-only output directory. The runner validates the generated date, IANA timezone, three record texts, and four exact prompts; seeds only those records into a new explicit-path `MemoryManager` with embeddings disabled; runs one fresh, tool-less `AdamBrain`; captures TTS text without playback; and removes the temporary store after the turn. Keep the operator-reported metadata check and live inference as separate confirmations, using the same protected-config, exact `custom` route, pinned OpenRouter API base, and fallbacks-off requirements described above. The record retains the retrieved context and normalized answer but always remains `outcome: not_scored` / `manual_review_required`; review the answer against the selected M1 oracle independently. The runner does not score success from substring matches, start Adam's service, access a microphone, or open desktop/browser tools. Its tests mock the provider and make no live inference.

```bash
umask 077
python tools/create_implementation_fixtures.py \
  --output-root /tmp/adam-implementation-fixtures \
  --run-id <fresh-fixture-id>
mkdir -m 700 /tmp/adam-m1-<series-id>
uv run --no-sync python -m tools.run_implementation_memory_trial \
  --fixture-dir /tmp/adam-implementation-fixtures/<fresh-fixture-id> \
  --prompt-id day \
  --config /path/to/existing/protected/config.yaml \
  --run-id <unique-opaque-run-id> \
  --config-id <non-secret-stable-route-label> \
  --output /tmp/adam-m1-<series-id>/runs.jsonl \
  --expected-model <exact-provider-model-id> \
  --metadata-http-status 200 \
  --metadata-model-id <exact-provider-model-id> \
  --metadata-endpoint-count <positive-count> \
  --metadata-checked-at <fresh-ISO-8601-UTC-time-ending-in-Z> \
  --route-ready-confirmed \
  --confirm-provider-inference
```

## GUI capture proof

Use a disposable Xvfb session for settings/browser trials when available. Before any provider-backed screenshot call, capture and locally inspect the intended window only. Record the X server/display, window ID, exact title, owning PID, window bounds, crop bounds, and image width/height. Confirm with the local image viewer that no desktop, other window, notification, account name, or unrelated content is visible. Also confirm capture identity is stable immediately before and after the screenshot. If any identity or crop field is missing or ambiguous, mark the trial `scope_blocked` and do not upload an image. A loopback HTML fixture contains no user data, but does not waive the window-scope proof.

## Run phases and timing

For each selected task, distinguish:

1. **Cold start:** service/app process start through ready state, recording startup time and starting/peak memory.
2. **Warm turn:** prompt end through first meaningful acknowledgment and verified final state.
3. **Long wait:** a task that requires more than one model/tool turn; record each meaningful progress update. Target acknowledgment is within 5 seconds, and long work should show useful progress about every 10 seconds.
4. **Recovery:** immediately after completion, then at 30 and 120 seconds; note what remains loaded, background work, memory, CPU, and whether the fixture/app returned to its known state.

Capture request-end, first acknowledgment, first tool start, first meaningful progress, final response, final-state verification, and any audible-response timestamps separately. Earcons, spinner animation, and generic “working” text are not meaningful acknowledgment. Keep screen/audio evidence local and record only sanitized evidence metadata in the report.

For every run record elapsed milliseconds; tool/model round count; retry count and raw HTTP 429/outage status; provider-reported input/output tokens and cost when available; process-tree RSS/PSS; Adam service cgroup `memory.current`, `memory.peak`, and CPU usage where available; system CPU, available RAM, swap, and power-source/instrument status. Label host-only GPU/battery/RAPL readings as their actual sensor readings; do not call them whole-system watts unless measured with a wall meter. Unknown/unavailable values remain null with a reason.

When telemetry is enabled, record the exact `trace_id` attached to the task's JSONL events. Use `tools/summarize_implementation_run_events.py` to join one private run record to the private event log by that ID and the request-end-to-verified-completion monotonic clock window. It reports `request_to_verified_completion_ms` from those validated bounds; copy that value to the run record's `timing_ms.request_end_to_verified_state`. This is request-end-to-verified-state elapsed time, not voice-to-voice latency. Use a trace only when it covers the task's provider events; if an ingress path splits events across IDs, mark the summary unavailable until task correlation is fixed. The summarizer validates matching-event clock order and summarizes provider calls, retry signals, durations, usage, accounting status, and first tool-start time. Provider request starts and completions must pair one-to-one by `span_id`; equal event counts alone do not establish complete accounting. It counts HTTP 429 responses separately from retries triggered by a provider-reported 429 code. If a provider start has no matching completion, the total 429 response count remains null with a missing reason because the attempt history is incomplete. Cost is totaled only when every provider request in the window has a completed event with an available cost in the same currency; otherwise the total remains null with reasons and known per-currency subtotals. Keep the original run record and event log as private evidence. The event log contains operational metadata, not the objective task oracle; retain the run's raw outcome separately.

Where a caller scopes speech, TTS playback events carry only its declared `acknowledgment`, `progress`, or `final` role and an opaque utterance ID; they do not record speech text. Other direct TTS callers remain unclassified and do not satisfy role-tagged summary fields. The summary counts a playback only when one `playback.started` span has exactly one matching `playback.completed` event with status `ok`, and it groups clauses sharing one utterance ID as a single utterance. The role identifies intended purpose, not whether the words were useful. Its acknowledgment/progress clocks use the `playback.started` time of a playback that later completed successfully; this is an output-path timing proxy, not a measured audible-onset timestamp or proof that a listener heard it. Capture actual audible-response timestamps separately, and leave timing unavailable with a reason when no tagged, confirmed event exists. If no event has the recorded trace ID within the task window, treat telemetry as unavailable rather than manufacturing a join.

## Repetition and reporting

Begin with three exploratory repetitions per task to find fixture and routing defects. Reset state between repetitions and retain every raw outcome, including provider failures. Select important conditions for at least ten matched repetitions when route capacity allows. Report raw samples plus sample count, median, and p95 only when the sample size supports them; do not hide retries or classify a tool return as task success. Do not compare current results with historical numbers as if they were matched. For repeated run records, keep one private JSON object per line and use:

```bash
uv run --no-sync python tools/summarize_implementation_run_series.py \
  --records /private/implementation-runs.jsonl
```

The series summary groups by exact scenario, provider/model/config ID, phase, host, OS, app, build, commit, and SHA-256 fingerprint of `starting_state`; the raw starting state is not repeated in the summary. Percentiles are withheld unless every matched-context field is present and at least ten runs have a valid `timing_ms.request_end_to_verified_state`. Median is the ordinary median; p95 uses nearest rank `ceil(0.95 * n)`. All sample outcome and retry rows stay in the output, including failures and provider errors. Missing normalized counts and usage values remain null with field-specific reasons; usage accounting status and bounded source missing reasons are retained. The JSONL input and output can contain run IDs, route metadata, missing reasons, and costs; keep both private. The elapsed statistic is request-end to verified-state time, never voice-to-voice latency.

Do not run the actual-microphone voice suite with synthetic audio, and do not claim voice-to-voice latency or recognition quality without a real authorized microphone trial. Record that work as deferred if the physical mic or private test conditions are unavailable.

## Per-run record template

For Loop 6 resource evidence, preserve the sampler's raw JSON trace as a private local artifact. Bind it to the evaluation record with the sampler schema version, sampler trace ID, SHA-256 of the exact trace bytes, and the sampled process PID plus /proc start-time ticks. Record request end and verified task completion in the same host CLOCK_MONOTONIC nanosecond domain used by the sampler. Do not put the private trace path or raw trace in a committed report. Summarize only after checking that the hash and process identity match. The summarizer reports sampled maxima of cgroup memory.current and complete process-tree RSS/PSS during the request interval, plus samples at completion and approximately 30 and 120 seconds later. It labels cgroup values as whole-cgroup measurements that may include unrelated processes. Its maximum sampled memory.current can miss brief peaks between samples; the sampler's memory.peak counter may predate the request and is not a task-attributed peak. A cgroup CPU delta is unavailable if any consecutive valid in-task counter drops. Recovery tolerance must be smaller than the smallest gap between requested offsets so one sample cannot stand in for multiple recovery points.

```json
{
  "run_id": "opaque-id",
  "trace_id": "telemetry-trace-id",
  "scenario": "B1",
  "fixture_id": "fixture-run-id",
  "commit": "git-sha",
  "host": "host-label",
  "os": "distribution and kernel label",
  "app": "Adam plus tested application names",
  "build": "Adam and tested application versions",
  "route": {"provider": "", "model": "", "config_id": ""},
  "phase": "cold|warm|long_wait|recovery",
  "starting_state": "",
  "prompt_id": "prompts.browser_template",
  "oracle_id": "oracles.invoices",
  "outcome": "pass|fail|blocked|provider_error",
  "observed_state": "",
  "timing_ms": {
    "request_end_to_ack": null,
    "request_end_to_progress": null,
    "request_end_to_final": null,
    "request_end_to_verified_state": null
  },
  "event_clock_ns": {
    "request_end": null,
    "verified_completion": null
  },
  "resource_trace": {
    "sampler_schema_version": 1,
    "sampler_run_id": "",
    "sha256": "",
    "target_pid": null,
    "target_starttime_ticks": null
  },
  "tool_model_rounds": null,
  "provider_call_count": null,
  "retries": null,
  "http_429_response_count": null,
  "http_429_retry_trigger_count": null,
  "usage": {
    "input_tokens": null,
    "output_tokens": null,
    "reported_cost": null,
    "currency": null,
    "accounting_status": null,
    "missing_reasons": []
  },
  "resources": {
    "process_tree_rss_bytes": null,
    "process_tree_pss_bytes": null,
    "cgroup_memory_current_bytes": null,
    "cgroup_memory_peak_bytes": null,
    "cgroup_cpu_usage_usec": null,
    "system_cpu_percent": null,
    "available_ram_bytes": null,
    "swap_used_bytes": null,
    "power_instrument": ""
  },
  "capture": {"window_id": null, "title": null, "pid": null, "bounds": null, "crop": null, "image_dimensions": null, "local_scope_inspected": null},
  "failure_class": null,
  "notes_redacted": ""
}
```
