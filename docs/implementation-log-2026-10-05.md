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

The current host is Desky and the implementation timeline says it remains read-only until the user explicitly lifts the post-crash pause. Code edits, benchmarks, and GUI/provider runs therefore wait for the laptop checkout or an explicit change to that host gate. Continue permitted repository/document review meanwhile; do not infer results from historical reports.

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

## Latest status — 2026-10-05 16:31 UTC

- Last measured result: none in this continuation; evidence is static source/document review only. No performance, success-rate, or host-state benchmark was produced.
- Current blocker: this checkout is on Desky, which remains read-only under the timeline's post-crash gate; no laptop checkout is mounted.
- Next loop: Loop 0, on the laptop checkout.
- Exact deciding task: initialize the disposable settings fixture with Compact off, Sync on, Notifications on; prove the focused window ID/title/PID/bounds and crop locally; then ask Adam, `Turn Compact mode on. Leave Sync and Notifications off.` Pass only if fixture readback is Compact=on, Sync=off, Notifications=off. Do not upload any screenshot unless the window/crop proof passes.
- First code candidate after baseline: evaluate whether synthetic timezone recall fails because `retrieve_context` omits the stored IANA timezone; if confirmed, make one focused context-format change and compare the same dated prompts plus prior passing recall cases.

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
