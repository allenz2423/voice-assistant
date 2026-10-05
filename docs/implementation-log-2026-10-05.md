# Implementation log — October 5, 2026

## Checkpoint 1 — Loop 0 audit

- Checkout: `/home/incoming/voice-assistant`, branch `main`, commit `23725a1c8a5bb8cbae998c05a9a9859209d5ee51`.
- Runtime host: `desky`. `git worktree list` reports only this checkout; no laptop worktree is mounted here.
- Existing tree state at audit: branch 12 commits ahead of `origin/main`; seven unrelated research drafts are untracked and were left untouched.
- Reconciliation: `docs/codebase-status.md` is dated October 4 and names an older commit. The October 5 handoff records later evidence and explicitly keeps Desky read-only after the crash. Historical results are not a new baseline.
- No code changes, tests, benchmarks, app launches, service actions, or provider-backed screen runs were performed. The computer-use run sheet is in `docs/computer-use-plan.md`; the Xvfb settings harness cited by the handoff is only reported in the laptop's local overnight-analysis directory and is not a repository fixture.
- Measurement note: current-host inventory identifies Desky hardware, not the 16 GB laptop. No Desky telemetry from this session is treated as an Adam benchmark or laptop result.

## Next checkpoint — Loop 0 on the laptop checkout

Create a repository-owned run sheet and deterministic fixtures before any baseline call. Cover factual answer, file/system status, dated synthetic-memory recall, a settings toggle, the generated 56-row browser report, terminal text/action, a scratch document task, and a mixed multi-step desktop request. Record exact prompts, initial/expected states, route/config metadata without secrets, cold/warm/recovery phase, capture window identity and crop proof, first meaningful acknowledgment, task/tool outcomes, retries, latency, process-tree PSS/RSS, cgroup memory, CPU, available RAM/swap, and cost when reported. Inspect screenshots locally before any image upload. Begin with three exploratory repeats per task; take selected conditions to ten matched runs when provider capacity permits.

First GUI replay candidate: the disposable notification/settings dialog task, with Compact enabled and Sync/Notifications left disabled. Verify the final state from the fixture itself. Pair it with the generated 56-row invoice report and a mixed browser-plus-memory request to ensure tool access remains broad.

## Deferred gate

At the time of the first audit, the current host was Desky and the timeline's post-crash pause had not yet been lifted. The user's later direct instruction to carry out the whole implementation plan authorized the continuing code and synthetic-fixture work on this checkout. The live workstation remains separate from disposable Xvfb work; no physical monitor, mic, service configuration, or live Adam service was changed.

## Checkpoint 2 — static instrumentation and fixture audit

Reviewed source and existing tests only; nothing was executed.

- `ComputerController._capture` brackets the capture with focused-window identity reads, logs image dimensions, and withholds an action snapshot token when identity is absent or changes. The desktop result reports scope, target, backend, dimensions, and origin. `tests/test_screenshot_scope.py` verifies crops, partial-crop rejection, HiDPI scaling, and fail-closed monitor scope with synthetic backends. This is source/test evidence only; before any provider-backed run, the laptop fixture still needs live window ID/title/PID, crop bounds/dimensions, and a local-only inspection.
- Desktop timing already records stage durations without screen content. The event writer allowlists scalar operational attributes and documented usage fields and writes JSONL with owner-only file permissions. Use an opaque trace/run ID and keep prompts, transcripts, images, OCR, and action arguments out of the log.
- `tools/measure_power.py` samples NVIDIA board power, battery sysfs, and Intel RAPL where readable. It explicitly is not a wall meter and does not record process-tree PSS/RSS or the service cgroup. Those memory measurements must be collected alongside this sampler; GPU board watts must remain separately labeled.
- No repository fixture bundle exists for the eight timeline scenarios. The handoff's local invoice/settings artifacts cannot be assumed present on this host.

### Run sheet draft to materialize on the laptop

For every run store scenario ID, run ID, commit, host, model/provider route, non-secret config ID, cold/warm/recovery phase, starting fixture state, exact prompt, objective oracle, final observed state, tool/model trace IDs, first meaningful acknowledgment time, total time, retries/429s, reported tokens/cost, process-tree PSS/RSS, Adam cgroup current/peak memory and CPU, available RAM/swap, and energy-source status. Keep raw per-run outcomes. For screenshot cases, attach only local evidence metadata and inspection result; stop before upload if identity/crop cannot be proven.

| Scenario | Fixed task | Pass oracle |
| --- | --- | --- |
| Factual | “Compare rigatoni and penne in two short sentences.” | Both distinctions in the rubric are correct; first useful response is timed separately from any processing indicator. |
| File/system | Ask for a fixture-file fact and CPU/memory status. | File answer matches the disposable fixture; CPU core count and memory use match a same-run `/proc` snapshot. |
| Dated memory | Seed only synthetic events with explicit dates, local times, and IANA zone; ask day, week, year, and timezone questions with similar distractors. | Exact expected event set and dates/times are recalled from the isolated store; no personal store is opened. |
| Settings | Start with Compact off and Sync/Notifications on; request Compact on while leaving the other two off. | Fixture state reads Compact=on, Sync=off, Notifications=off after the task. |
| Browser table | Generate a local 56-row invoice report from a fixed seed; ask for the overdue supplier with the largest total and its invoice IDs. | An independent fixture oracle agrees on supplier, IDs, totals, and row count. |
| Terminal | In a disposable directory, inspect a fixed text file and write a derived one-line summary to a second fixture file. | Source remains byte-identical and output exactly matches the precomputed expected line. |
| Document app | Edit and save a scratch document with a fixed requested sentence. | Reopening the scratch file returns the exact requested text and no unrelated content. |
| Mixed multi-step | Read the generated report, write its oracle-checked answer into an isolated output file, and report a same-run system-status fact. | Both file output and status fact match their independent oracles; available browser, filesystem, and system tools are all retained. |

The fixtures, seeds, exact outputs, and any UI automation must be committed only from the laptop checkout after its capture scope is proven. Exploratory baseline starts at three repeats per task; select important conditions for ten matched runs when provider capacity allows. The 5-second acknowledgment and 10-second progress targets are measured from user-request end and meaningful user-visible content, not earcons or spinners.

## Current continuation point

Loop 0 has a documented source audit and a run sheet draft, but its exit gate is still open: the deterministic artifacts and live capture proof have not been created, and no current laptop baseline exists in this environment. The next exact operation is to materialize the table above as laptop-owned fixtures/run sheet, prove the settings-window capture locally, then run three exploratory baseline repetitions on the laptop. The Desky restriction remains in force for code, service, app, and benchmark work.

## Checkpoint 3 — temporal-memory fixture specification

Static review of `src/memory/temporal.py` and `tests/test_memory.py` shows the baseline can use exact ISO dates for deterministic event filtering; relative query expressions supported in code include `last week` and `last year`. The host-local IANA timezone is retained with event metadata. Keep all records in an isolated synthetic memory store and record the fixture timezone.

Seed these three synthetic work records through the normal memory-save path at a fixed capture timestamp in `America/New_York`:

1. `I worked on the Juniper migration on 2026-09-29 from 9:30 to 10:15am.`
2. `I worked on the Juniper invoice export on 2026-10-02 from 3:00 to 4:00pm.`
3. `I worked on the Juniper migration on 2025-10-02 from 11:00am to noon.`

Prompts and expected retrieval sets:

- `What did I work on 2026-09-29, and what local time?` → record 1 only; local time 9:30–10:15am.
- `What Juniper work did I log last week?` with the run date anchored to 2026-10-05 → records 1 and 2, in date order; exclude the semantically similar 2025 record.
- `What Juniper work did I log last year?` → record 3 only.
- `What timezone did I use for the September 29 work entry?` → `America/New_York`, with the event's resolved offset consistent with the 2026-09-29 DST rules.

The evaluation must score retrieved context separately from the model's final wording so a wrong answer can be classified as date parsing, candidate retrieval, timezone interpretation, or answer grounding. The live replay remains laptop-only and awaits the fixture/run sheet gate; these are synthetic prompts, not measurements.

## Checkpoint 4 — GUI fixture guards

- Existing browser unit fixtures in `tests/test_browser_navigation.py` cover local navigation, text entry, stale references, and unsafe URL rejection. They do not contain the 56-row invoice workload.
- `BrowserNavigator._snapshot` caps visible body text at 6,000 characters. Keep the generated invoice fixture under that bound, place the decisive result late in the table as well as in the independently computed oracle, and assert the captured DOM output includes all 56 rows before asking the model to report. This prevents a nominal 56-row fixture from silently becoming a partial-table task.
- The real screenshot path crops to focused-window bounds and checks image geometry; the controller also requires stable identity before issuing an action token. That alone does not prove that a live focused window is the fixture the run intended to expose. The fixture harness must compare exact window ID/title/PID and bounds before invoking the model, verify the resulting crop dimensions, inspect it locally, and fail closed before the provider call on any mismatch. The source path must not be treated as a substitute for this run-level proof.

## Next loop selection

The next eligible loop remains Loop 0 on the laptop because no reproducible task corpus or current laptop baseline is present in this checkout. Highest-impact risk to settle first is screenshot scope: the capture gate must reject an unrelated active window before any provider-backed call. The first replay after that proof is the exact settings task already listed above; the 56-row browser report and mixed browser-plus-filesystem/system request are its unrelated-domain regression cases. Do not start OmniParser comparisons or runtime tuning until this baseline exists.

## Checkpoint 5 — fixed prompt and oracle definitions

These prompt strings complete the run sheet draft. Replace fixture IDs/paths only with deterministic run-specific names; never point at personal files or memory.

1. **Factual:** `Compare rigatoni and penne in two short sentences.` Oracle rubric: correctly distinguish shape (tube with ridges vs. short tube with angled ends) and note both are pasta; record first meaningful answer separately from transport/status UI.
2. **File/system:** run both `Read "<absolute fixture path>/source.txt" and tell me its byte size and exact first line.` and `According to a fresh system status check, how many logical CPU cores does this system have and what percentage of memory is in use?` Record the expanded absolute path as part of the exact prompt. Oracle: file answer matches the disposable fixture; CPU core count and rounded memory use/total match a same-run `/proc` snapshot and the `get_system_status` trace. The existing tool does not report Linux distribution or kernel version. Ignore unrelated paths, process names, and user data.
3. **Dated memory:** seed only the three Juniper records specified above. Run the exact day, `last week`, `last year`, and timezone prompts from Checkpoint 3. Oracle: retrieval-set equality first, then factual agreement of the answer with those records.
4. **Settings:** start Compact off, Sync on, Notifications on. Prompt: `Turn Compact mode on. Leave Sync and Notifications off.` Oracle: inspect fixture values after the task; exact final state is on/off/off.
5. **Browser report:** fixed local HTML with exactly 56 invoice rows and independently stored aggregate oracle. Prompt: `Among overdue invoices, which supplier has the largest total? Give the supplier, invoice IDs, and total.` Oracle: supplier, all contributing invoice IDs, currency total, and all 56 rows present in the browser evidence. Keep visible DOM under the 6,000-character snapshot cap.
6. **Terminal action:** initialize a disposable `source.txt` with three `ITEM:` rows and two unrelated rows. Prompt: `In this fixture directory, count the lines in source.txt that begin with ITEM: and write only that count plus a newline to answer.txt. Do not change source.txt.` Oracle: `answer.txt` is exactly `3\n`; source hash is unchanged.
7. **Document app:** open a disposable text document containing exactly `STATUS: DRAFT\nOWNER: Adam\n`. Prompt: `In the scratch document, change DRAFT to FINAL and save it. Leave every other character unchanged.` Oracle: saved file is exactly `STATUS: FINAL\nOWNER: Adam\n` after reopening.
8. **Mixed multi-step:** use the 56-row local browser report and the same disposable output directory. Prompt: `Find the supplier with the largest overdue invoice total in the local report. Write exactly one line to task-output.txt in the fixture directory: Largest overdue supplier: <supplier>; total: <$amount>. Then tell me the current available RAM.` Oracle: exact file line matches the independent browser aggregate, system-status answer matches a same-run snapshot, and the trace retains browser, filesystem, and system tools.

For each fixture record the seed/version and expected output before a run. Keep the initial screen/file state identical between repeats; restore fixtures from their seed, not from Adam's previous final state. Include one cold service start, a warm conversational turn, one intentionally longer wait for meaningful progress, and the mixed request in the baseline batch. Do not replay personal or recorded speech; the actual-microphone loop requires a later explicitly approved private fixture.

## Checkpoint 6 — file/system routing audit

- `get_system_status` reports logical CPU count/load, RAM used versus total, root storage, available GPU telemetry, and a short CPU utilization sample. It does not report OS distribution or kernel version, so the fixed baseline prompt now asks only for supported CPU and RAM fields.
- Pure system-status turns have a direct read-only dispatch path. Static regression cases confirm ordinary explanatory or mixed status/action prompts keep the broader tool set; no tool-filter change is justified by this audit. The live mixed browser/filesystem/system task remains necessary to confirm model behavior.
- `read_file` accepts an absolute path and returns bytes plus UTF-8 text. Expand and record the run-specific disposable path in the exact prompt; do not rely on an unspecified “fixture directory.” The terminal and document tasks must use the same temporary workspace and retain independent expected hashes/content.

## Checkpoint 7 — dated-memory issue for the baseline

The source path now yields a specific, testable hypothesis. `MemorySearchResult` carries `event_timezone`, but `MemoryManager.retrieve_context` formats the event date, resolved start/end offsets, precision, and original phrase without formatting the IANA timezone name. The existing temporal test checks the stored timezone and UTC offset, while the retrieval-context assertion checks only the event text and start timestamp. Therefore the current model prompt may not contain enough evidence to answer “which timezone” when the saved record is the only source.

Keep the timezone question in the baseline and score raw retrieved context as well as the final answer. If the context lacks the zone, the focused candidate for the later memory loop is to include the stored IANA timezone in the temporal context note, then compare exact-context and dated-recall results for the seeded records. Do not make that source change before the reproducible laptop baseline, and do not infer model failure from this source review alone.

## Checkpoint 8 — make relative-date memory trials repeatable over time

The explicit 2026 dates above describe the October 5 first run; keeping those dates fixed would make `last week` and `last year` stale after the calendar moves. For each later baseline batch, derive the records from the captured host-local date `R` in the recorded IANA zone:

- `previous_week_monday = R - timedelta(days=R.weekday() + 7)`.
- Record 1 uses `previous_week_monday + 1 day`; record 2 uses `previous_week_monday + 4 days`, preserving the Tuesday/Friday schedule inside the prior Monday–Sunday week.
- Record 3 uses January 15 of `R.year - 1`.
- Generate each natural-language memory string with its concrete ISO date, as in Checkpoint 3. Expand the exact ISO date into the day and timezone prompts. Record `R`, each generated date, and the host IANA zone beside the run.

This keeps the same semantic task and oracle while ensuring the relative week/year retrieval windows contain the synthetic events on every later run. On the current laptop evidence, the expected zone is `America/New_York`; assert and record the actual zone before using that expected answer.

## Historical checkpoint — 2026-10-05 16:31 UTC (superseded by later work)

- At that point the continuation still had only static source/document review and the live Desky pause was in force. The user subsequently directed uninterrupted end-to-end implementation, superseding this next-step recommendation.
- The timezone context omission named here was tested and fixed in Loop 5; see the measured result below and the subsequent dated-memory integration replay.

## Loop 5 baseline — synthetic dated-memory context (2026-10-05)

- Commit: `23725a1c8a5bb8cbae998c05a9a9859209d5ee51`.
- Host: Desky. Route: direct `MemoryManager.retrieve_context`; no LLM/provider call. Config difference: temporary synthetic store, dense embedding disabled. Initial screen/audio: not applicable.
- Prompt: `What timezone did I use for the 2026-09-29 work entry?`
- Expected task state: retrieved context contains `America/New_York`, the record's stored IANA zone.
- Fixture: one synthetic Juniper work record saved through `MemoryManager.save`; `event_start_at=2026-09-29T09:30:00-04:00`, `event_timezone=America/New_York`.
- Raw outcomes: run 1, 2, and 3 each returned `- I worked on the Juniper migration on 2026-09-29 from 9:30 to 10:15am. [resolved event time: 2026-09-29T09:30:00-04:00 to 2026-09-29T10:15:00-04:00] [original time phrase: 2026-09-29; from 9:30 to 10:15am]`; `America/New_York` absent in all three contexts. Exact result: 0/3 context passes.
- First acknowledgment, model/tool turns, provider usage/cost, PSS/RSS, service cgroup, CPU, available RAM/swap, and power: not measured by this component-only replay. No full Adam user-visible result is claimed.
- Classification: retrieval formatting drops stored timezone metadata after successful date-filtered retrieval. Hypothesis: adding the stored IANA zone to temporal context will make this fact available to answer grounding without changing candidate selection.

## Loop 5 focused change — include the saved timezone in retrieved context

- Started: 2026-10-05 16:36 UTC. Commit before change: `23725a1c8a5bb8cbae998c05a9a9859209d5ee51`; host: Desky; route: direct in-process `MemoryManager`; provider: none. Synthetic store, one event, same prompt as baseline; no config, service, screen, or audio changes.
- Change: `retrieve_context` now adds `[event timezone: <IANA zone>]` when a memory has timezone metadata. Added a regression assertion for `America/New_York` and the known DST offset.
- Matched raw outcomes: all three post-change contexts include the unchanged event text and resolved times plus `[event timezone: America/New_York]`; task result 3/3 versus baseline 0/3.
- Post-change context retrieval durations: 0.174, 0.066, 0.052 ms (median 0.066 ms, n=3; no baseline duration and no latency claim).
- Focused regression result: `uv run --no-sync pytest tests/test_memory.py -q` → 24 passed in 0.79 s. This includes the existing dated/timezone retrieval checks. Provider cost, first audible response, whole-brain answer quality, process-tree PSS/RSS, cgroup peak, CPU, and power were not measured.
- Decision: retain this focused source change for the demonstrated context-grounding gain. Next verify the actual AdamBrain memory-only path on synthetic data, then run the full laptop recall fixture when that checkout is available.

## Checkpoint 9 — materialized Loop 0 artifacts and GUI scope proof

- Started the implementation run on the user's direct instruction to continue the full plan. Host remains Desky; no laptop checkout is mounted, so all host-labelled numbers below are Desky-only. No model/API key, transcript, personal memory, or real screen content was written to this report.
- Added `docs/implementation-evaluation-run-sheet-2026-10-05.md` with eight task oracles, cold/warm/long-wait/recovery phases, timing/resource fields, repeat guidance, and screenshot-scope proof requirements.
- Added `tools/create_implementation_fixtures.py` and regression tests. It generates private disposable folders with a seed-fixed 56-row invoice table, independently computed overdue totals, a self-resetting settings page, source/document files, exact prompts, and date-rolling synthetic memory records. Fixture dates use the local IANA timezone and recompute the prior-week and prior-year samples from the run date.
- Fixture run `gui-series-20261005-02`: 56 invoices; overdue maximum Juniper Labs, INV-0055 and INV-0056, $2,175.00. Source is 75 UTF-8 bytes and starts `ITEM: olive`; terminal oracle is `3\n`; document final oracle is `STATUS: FINAL\nOWNER: Adam\n`.
- Initial browser capture proof used disposable Xvfb display `:111` and only the loopback settings fixture. Exact window identity was X11 ID `4194326`, title `Adam Fixture Settings — Mozilla Firefox`, PID `150185`; desktop/window bounds and crop were `(0, 0, 1280, 900)`, image `1280x900`. Adam's screenshot capture reported the same identity immediately before and after capture and issued an action-capable token. The captured crop was viewed locally; it showed only the synthetic preferences page inside the Firefox window. The earlier Firefox privacy tab was closed before this proof. No screenshot was sent to any provider.
- Local GUI sanity replay manually set the three synthetic controls to `Compact=on; Sync=off; Notifications=off`; the page's visible `STATE` readback agreed. A reload exposed a real fixture defect: Firefox restored changed checkbox values, so the documented initial state was not repeatable. The page now assigns the initial values on each `pageshow`; a second local replay showed the initial state, final state, and reset-to-initial state correctly in locally inspected window crops. A regression test guards the reset handler.
- `uv run --no-sync pytest tests/test_implementation_fixtures.py tests/test_memory.py -q` → **27 passed**. `python -m py_compile tools/create_implementation_fixtures.py` and `git diff --check` passed.
- This meets the reproducible-state and screenshot-scope portions of Loop 0. The fixture/run sheet can be regenerated from the repository; the screenshots and fixture instances remain under `/tmp` and are not committed.

## Checkpoint 10 — repeated synthetic recall and first live-route attempt

- Memory fixture run `memory-series-20261005-01`, run date `2026-10-05`, zone `America/New_York`; three synthetic records were saved through `MemoryManager.save` into a new temporary store on each repetition. The previous-week records were September 29 and October 2, 2026; the prior-year distractor was January 15, 2025.
- Raw retrieval outcomes (each returned context also carried `[event timezone: America/New_York]`):
  - `day`: 3/3 exact single-record matches with `2026-09-29T09:30:00-04:00` through `2026-09-29T10:15:00-04:00`; retrieval ms `0.320, 0.061, 0.060`.
  - `week`: 3/3 returned exactly the September 29 and October 2 Juniper records in date order and excluded the 2025 distractor; retrieval ms `0.185, 0.067, 0.068`.
  - `year`: 3/3 returned only the January 15, 2025 record; retrieval ms `0.139, 0.053, 0.052`.
  - `timezone`: 3/3 returned the September 29 event and IANA zone; retrieval ms `0.058, 0.048, 0.049`.
- Aggregate fixture result: **12/12** expected context sets, dates/times, and zones. Per-query median retrieval time was 0.061 ms (day), 0.068 ms (week), 0.053 ms (year), and 0.049 ms (timezone). These are in-process context-retrieval timings, not voice, model, or end-to-end answer latency.
- Added `test_brain_recall_request_receives_synthetic_event_timezone`, which exercises `AdamBrain.process_user_utterance` with a fake model client and isolated synthetic memory. It checks that the date, offset, and IANA zone reach the model's user message. It passed in the 27-test focused run; this verifies prompt wiring without claiming live model answer quality.
- Current configured route (non-secret fields only): provider `custom`, model `stealth/space-bunny-alpha`, OpenRouter `/api/v1`, provider fallbacks disabled, TTS `silent`. One isolated direct-Brain settings request returned HTTP 404 in 330 ms; no screenshot was attached (`has_image=False`) and no UI action occurred. The response was `I couldn't reach the language model. Please try again shortly.` Usage/cost were unavailable and no HTTP 429 was observed.
- Follow-up authenticated metadata GET returned HTTP 200 for model ID `stealth/space-bunny-alpha` but zero provider endpoints at that time. This is a provider route outage, not a fixture or application task result. As required by the timeline, preserve this failure and defer provider-dependent trials; do not silently switch models or treat it as a passing baseline. The local and synthetic work continues meanwhile.

## Checkpoint 11 — instrumentation check and continuation

- Read-only Desky Adam process snapshot, separate from the synthetic request: PID 1787 RSS 2,030,644 KiB and PSS 2,006,470 KiB. Its service cgroup sample was 3,350,310,912 bytes current / 3,459,547,136 bytes peak, with 591,096,455 microseconds cumulative CPU at the time of reading. System `MemAvailable` was 24,238,256 KiB and `SwapFree` 32,770,076 KiB; `MemTotal` was 32,773,192 KiB. These are not an Adam request-attributed profile and do not describe the 16 GB laptop.
- Power sampler probe: 2.0 seconds, four samples. NVIDIA board readings ranged 7.43–7.72 W (GTX 1080 Ti) and 7.95–8.32 W (RTX 3070). No battery was present; Intel RAPL was only partial because `energy_uj` permission was denied. There is no wall-power measurement, so these idle board readings do not establish whole-system power or Adam energy use. Trace is `/tmp/adam-implementation-fixtures/power-probe.json` and contains host hardware metadata; it is not committed.
- Current gate: Loop 0 task definitions, deterministic state, and local scope proof are materialized. Loop 1 is waiting only on a serving OpenRouter endpoint; the initial current-route request and zero-endpoint metadata are recorded above. Cold live-service restart, provider-backed task repetitions, voice acknowledgement, and actual microphone trials remain unmeasured. No laptop baseline is claimed.
- Next eligible work: keep using synthetic isolated stores and fixture oracles; continue focused regression/instrumentation work while checking provider availability in later checkpoints. Resume three exploratory Adam runs per task only when the configured route reports at least one serving endpoint. Keep actual microphone, 16 GB concurrent-workload, and Shenzhen I/O work deferred until their physical/authorization gates are met.

## Checkpoint 12 — local tool-path smoke and optional browser dependency

- A single provider-free fixture smoke ran Adam's actual `read_file`, `run_bash_command`, `write_file`, and `get_system_status` handlers against the disposable fixture. File answer was 75 bytes / `ITEM: olive`; terminal output was exactly `3\n` and the source SHA-256 stayed unchanged; the reopened scratch document matched the exact final text; mixed output matched the browser-computed invoice oracle; CPU count was 32 and reported memory use was 26%, matching the same-host snapshot within 1 percentage point.
- In an isolated Microsoft Edge headless profile, the rendered local invoice page exposed all 56 table rows and 2,458 body-text characters. Independently summing the overdue rows yielded Juniper Labs / INV-0055 and INV-0056 / $2,175.00, matching the fixture oracle. This exercises the browser-rendered fixture and tool handlers, not Adam's model planning.
- The optional Playwright package is not installed, so the BrowserNavigator-backed replay returned `ModuleNotFoundError`; no package was installed. The existing Edge route completed the local DOM check without provider access. A first check incorrectly expected the aggregate total to appear as a literal page string; the invoice page correctly contains the two contributing amounts only, so the validator now sums those rendered rows.
- The fixture's first version used a vague “leave off” phrasing while those controls started on. That prompt was made explicit (“Set Compact mode to on, Sync to off, and Notifications to off.”). Manual local interaction and visual readback passed. Firefox's form-state restoration on reload also led to the explicit `pageshow` reset and a manual initial → final → initial replay; the last inspected state is back at the initial values.

## Checkpoint 13 — tighten fixture and status oracles

- Extended `tests/test_implementation_fixtures.py` to parse the generated HTML table and independently recompute its overdue winner from the rendered rows. It now also seeds the generated rolling-date records through `MemoryManager.save` and verifies exact day/week/year/timezone retrieval against fixture expectations.
- The run sheet's memory-percentage tolerance is now 1 percentage point because `get_system_status` formats this field as an integer. Each live trial still records raw `MemTotal` and `MemAvailable` values for recalculation.
- Latest focused verification: `uv run --no-sync pytest tests/test_implementation_fixtures.py tests/test_memory.py -q` → **28 passed in 0.79 s**. This remains focused synthetic coverage, not the full-suite integration gate.

## Checkpoint 14 — restore visible scratchpad routing without narrowing note workflows

- Started: 2026-10-05 17:02 UTC; completed focused checks: 2026-10-05 17:18 UTC. Base commit: `3a395b9` (fixture and run-sheet checkpoint). Host: Desky. Config/provider route: configured `custom` / `stealth/space-bunny-alpha`, provider unavailable as recorded above; no provider request, service restart, or live desktop action in this loop. Initial screen/audio: not applicable; this was a deterministic intent-classifier and tool-selection regression test.
- Exact request: `In the local scratchpad, enter "Call the dentist Tuesday at 2 pm" in the Note text field, click Save note, and tell me the saved text.` Expected state: classify this as one visible UI interaction, offer the compact desktop tools, and summarize only the exact saved text when the UI evidence confirms it. Negative comparison: `Click the Notes app and write a note for tomorrow's appointment.` must stay on the broad tool route.
- Failure before the change: the generic saved-text readback regression returned no summary because `note` in the visible `Note text field` / `Save note` controls tripped the broad “other tool” guard. The earlier focused result was 83 passed / 1 failed in `tests/test_brain_system_status_routing.py`.
- Change: classifier masks only visible control phrases matching `note(s) [text] field/box/input/area` and `click/press/tap [the] save/create/new/edit note` before applying the broad note-tool guard. Added positive and negative route assertions, including the exact compact tool set for the scratchpad and preservation of all supplied tools for the Notes-app workflow. The existing readback assertions still require evidence beginning with a saved confirmation and matching the requested text; a generic “Done” or unconfirmed text does not pass.
- Matched local replay: `uv run --no-sync pytest tests/test_brain_system_status_routing.py tests/test_brain_tool_round_trip.py tests/test_screenshot_scope.py -q` → **121 passed in 2.99 s**; `git diff --check` passed. This confirms classifier, tool filtering, readback behavior, adjacent round trips, and screenshot-scope tests. It does not claim an actual application save or model-planned end-to-end task.
- First acknowledgment, model/tool turns, provider usage/cost, end-to-end latency, process-tree PSS/RSS, service cgroup, available RAM/swap, and power were not measured because there was no provider-backed task or service workload in this regression loop. GUI fixture screenshots were not used.
- Decision: retain the narrow routing repair. Next eligible Loop 3 task: exercise remaining visible-form and app-launch classifier cases against the same broad-domain negatives, then run an isolated local UI readback fixture if available. Keep provider-dependent Loop 1 repeats deferred while the configured endpoint remains absent; do not infer end-to-end success from these local checks.

## Checkpoint 15 — first full local integration suite

- Started after `82a6d23` on 2026-10-05 17:18 UTC; completed approximately 17:20 UTC. Host: Desky. Provider route remained unavailable; tests used their local fixtures/mocks and made no live model request. Initial screen/audio: not applicable to the repository suite.
- Command: `uv run --no-sync pytest -q` → **558 passed, 4 skipped in 48.77 s**. Follow-up skip report: all four skips are in `tests/test_browser_navigation.py`, which requires optional browser-control dependencies. No failure was hidden in the skip count.
- This was the first full-suite pass after retaining the timezone-context change, deterministic fixtures, and visible scratchpad routing fix. It is a code integration result, not the Loop 6 laptop voice/desktop acceptance gate: no microphone, sustained live service session, provider cost, end-to-end user latency, or request-attributed resource profile was measured.
- Continuing from this checkpoint exposed an untested false-success case in the saved-text path; see Checkpoint 16. The full suite is rerun after that correction below.

## Checkpoint 16 — prevent unverified scratchpad saves from being reported as complete

- Started: 2026-10-05 17:20 UTC; final focused and full-suite verification completed: 2026-10-05 17:30 UTC. Starting commit: `82a6d23`. Host: Desky. Provider route: configured `custom` / `stealth/space-bunny-alpha`; no provider call. Initial screen/audio: a synthetic controller inspection returned no image, and the tool executor returned only synthetic OCR lines. No live window or user data was used.
- Exact task: `In the local scratchpad, enter "Call the dentist Tuesday at 2 pm" in the Note text field, click Save note, and tell me the saved text.` Expected states: return the exact saved text only when OCR shows `Saved: ...`; report a clear failure when OCR says `Not saved`; report that the save cannot be verified when only the typed text is visible.
- Failure reproduced through `AdamBrain.process_user_utterance`: the mocked model issued a structured `computer_control` sequence (click the field, type the requested text, click Save note), then answered `Saved.`. When tool OCR said `Not saved`, or showed the typed text without a saved indicator, Adam spoke the unsupported `Saved.` unchanged. These two parameter cases failed before the correction; the positive saved-text case passed.
- Change: `_summarize_desktop_readback_if_generic` now gives explicit negative save evidence precedence and, for saved-text requests, returns an uncertainty response unless a visible `Saved: <text>` confirmation is present. Explicit failure wins even if both “Saved” and “Not saved” OCR lines occur. Added a four-case full `AdamBrain` tool-loop regression with a synthetic controller and tool executor, plus direct helper assertions for negative and missing evidence.
- Post-change results: focused routing/tool/screenshot-scope tests → **125 passed in 3.04 s**; final complete repository suite `uv run --no-sync pytest -q` → **562 passed, 4 skipped in 46.65 s**; skip reasons → four optional browser-control cases; focused skip-module check → **34 passed, 4 skipped in 4.59 s**; `git diff --check` passed. The only skip is optional browser-navigation coverage because its dependencies are absent.
- The tool path and response logic are exercised with mocks; this proves truthful response handling for the specified OCR states, not that a real application accepted a save. First-audible acknowledgment latency, provider turns/cost, whole-task wall time, PSS/RSS, cgroup memory, CPU, RAM/swap, and energy were not measured for this synthetic turn.
- Decision: retain the fix with its regression coverage and this log. Next eligible loop: Loop 4, use the same private invoice/settings fixtures to compare the local evidence available from rendered browser DOM, OCR text, and scoped screenshot capture, recording which fixture oracle each representation can establish. This offline comparison will not stand in for same-route model trials; those remain deferred until the configured provider serves the model.

## Checkpoint 17 — offline Loop 4 evidence comparison

- Started: 2026-10-05 17:30 UTC; completed: 2026-10-05 17:37 UTC. Starting commit: `6683b01`. Host: Desky. Model/provider: none; only an isolated Microsoft Edge process loaded fixtures from an ephemeral HTTP server bound to `127.0.0.1`, with an isolated browser profile and background networking disabled. No live desktop, user data, audio, or provider screenshot was involved.
- Fixture: `evidence-series-20261005-01`, synthetic invoice table and settings page. Browser build: Microsoft Edge 154.0.4258.53. Exact browser task oracle: Juniper Labs; INV-0055 and INV-0056; `$2,175.00` total.
- Browser DOM result: Edge `--dump-dom` returned all **56** invoice rows; parsing rendered table cells and independently summing only `overdue` rows produced Juniper Labs, INV-0055/INV-0056, 217,500 cents, exactly matching `expected.json`. Rendered body text length was 2,458 characters. Raw single-run DOM dump/render time: **382.74 ms** (does not include a model turn or Adam tool loop).
- Scoped screenshot representation: a local-only 1280×900 PNG, 129,238 bytes, was inspected locally. It showed invoice rows INV-0001 through INV-0026; the decisive INV-0055/0056 rows were below the viewport. Headless screenshot render time: **464.15 ms**. This viewport alone cannot establish the invoice winner or total. The settings page at the same viewport showed all three checkboxes and the `STATE Compact=off; Sync=on; Notifications=on` status; DOM output matched that initial state. Settings DOM dump/render was 375.34 ms and screenshot render was 435.74 ms (one sample each).
- OCR, OCR region boxes, AT-SPI, and Playwright branches were unavailable in this environment: `rapidocr`, `tesseract`, `pyatspi`, and Playwright were not installed. No package was installed. The repository's four browser-navigation skips are the corresponding optional browser-control tests; Edge's local CLI fallback supplied DOM and screenshot artifacts without claiming BrowserNavigator coverage.
- Decision/inference from this fixture: full rendered DOM is the suitable evidence source for a text-heavy table whose decisive rows fall below the fold; a single viewport screenshot is insufficient. A sparse settings screen is readable from a local screenshot and also exposes a direct DOM state oracle. These are evidence-availability findings only; there was no model to compare completion accuracy, turns, or answer latency, so no global evidence-selection policy was changed.
- No first meaningful acknowledgment, model/tool rounds, provider usage/cost, Adam end-to-end timing, PSS/RSS, cgroup memory, CPU, available RAM/swap, or power were measured. Raw DOM/screenshots and browser profiles remain under private `/tmp` folders and are not committed.
- Next eligible work: recheck the configured route and, if it serves, start matched provider-backed Loop 4 trials on these same fixtures. If unavailable, continue Loop 3 local UI/tool-path checks and Loop 5 synthetic memory regressions; keep OCR/AT-SPI and same-route comparisons explicitly deferred until their dependencies/route are available.

## Checkpoint 18 — configured route remains unavailable

- Rechecked at 2026-10-05 17:38:26 UTC from Desky using one authenticated `GET /models/stealth/space-bunny-alpha/endpoints` against the configured OpenRouter API base. HTTP status: **200**; returned model ID matched; serving endpoint count: **0**. This metadata request made no model inference and incurred no inference usage/cost.
- Route/config difference: none; provider `custom`, model `stealth/space-bunny-alpha`, fallbacks remain disabled. No service restart, model change, screenshot, or user content was involved.
- Loop 1 and same-route Loop 4 trials remain deferred. Continue with local Loop 3 and synthetic Loop 5 work; recheck only at a later checkpoint rather than repeatedly polling.

## Checkpoint 19 — current memory recall honors corrections and excludes a similar distractor

- Started: 2026-10-05 17:38 UTC; completed: 2026-10-05 17:49 UTC. Starting commit: `c9e573a`. Host: Desky. Model/provider route: none; a temporary private JSON store used `MemoryEmbedder(disabled=True)`, and the AdamBrain integration used a fake client. No personal memory, inference provider, or live service was involved.
- Synthetic store: `I prefer dark mode in my code editor.`; later `Correction: I now prefer light mode in my code editor.`; and the semantically similar but unrelated distractor `I prefer sepia mode in my photo editor.` Categories were `preferences`.
- Exact current-state query: `What theme do I currently prefer in my code editor?`. Before the correction-priority change, retrieval returned the older dark-mode record first (score **0.6578**) and the explicit light-mode correction second (score **0.6204**); the query was also excluded from the memory-only route by the generic `current/currently` guard. After correction ordering alone, the current context still included the photo-editor distractor: its lexical score was **0.530** (code-editor records: 0.767 and 0.732).
- Exact history query: `Did I change my code editor from dark to light mode?`. Expected outcome: retain both code-editor records in chronological order and do not promote the correction as the answer to a history question.
- Focused change: current/currently/latest preference questions with retrieved personal memory may use the memory-only route while explicit live domains (CPU, appointment, calendar, etc.) remain tool-routed. For current/latest contexts, explicit correction/update records sort first; date-filtered event history remains chronological, and ordinary history queries retain score order. The memory recall prompt now tells the model to prefer a later explicit correction for current state and preserve earlier versions for history.
- A matched distractor test showed the BM25-only fallback also accepted the photo-editor record at 0.4 token coverage. The lexical acceptance floor for BM25-only retrieval is now 0.5 on the hybrid and raw-BM25 branches; the existing dense-score acceptance path is unchanged. Post-change context begins with the light-mode correction, retains the older code-editor fact, and excludes the photo-editor distractor. The history query still returns dark then light and excludes the photo-editor fact.
- The AdamBrain synthetic round trip confirms that the current preference query receives the corrected context first, uses `MEMORY_RECALL_SYSTEM_PROMPT`, and offers no external tools. `What is my current CPU usage?` and `When is my current appointment?` remain outside the memory-only route.
- Verification: `uv run --no-sync pytest tests/test_memory.py -q` → **27 passed in 0.82 s**; final `uv run --no-sync pytest -q` → **564 passed, 4 skipped in 46.63 s**. All four skips are optional browser-navigation tests because browser-control dependencies are absent. A prior combined memory/routing/tool run before the final lexical threshold adjustment was 146 passed; the full suite includes the final adjustment.
- No provider-backed answer accuracy, first acknowledgment, API usage/cost, end-to-end latency, cgroup/PSS/RSS, CPU, available RAM/swap, or power was measured. The model answer in the round-trip test is fixed synthetic output; retrieved-context accuracy is the measured result.
- Decision: retain the change because it corrects the measured stale-first ordering and removes the tested similar distractor without breaking the full suite. Next eligible Loop 3 task: replay `evidence-series-20261005-01/settings.html` in a fresh isolated Xvfb window, verify the three-control final state from the resulting screenshot, and reset/re-read it. Continue to defer provider-backed trials while the endpoint count is zero.
