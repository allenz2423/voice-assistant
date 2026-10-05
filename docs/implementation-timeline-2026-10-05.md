# Adam implementation loops

October 5, 2026 · Agent execution plan. Progress is determined by evidence and exit gates, with no time windows or target dates.

## Destination and current boundary

Build Adam into a responsive, general-purpose voice and desktop assistant that remembers broadly, completes ordinary computer tasks with truthful outcome reports, and remains light on the 16 GB laptop. The main LLM remains hosted. Improvements must preserve broad tool access on mixed requests; settings pages, browsers, OBS, and Shenzhen I/O are evaluation cases, not special product modes.

The current [handoff](desktop-handoff-2026-10-05.md) already records opt-in browser DOM/navigation, scoped screenshot/OCR, AT-SPI, a model/tool loop, memory-date handling, lazy imports, retry/progress plumbing, and narrow synthetic wins. It also records unresolved whole-task GUI success, real-microphone voice latency and quality, concurrent memory peak, reliable power attribution, and a matched OmniParser comparison. Do not reimplement completed foundations merely because an older plan describes them as proposed. [Computer-use handoff](computer-use-plan.md) calls for observed failure-driven changes, with the model retaining task strategy.

**Access boundary:** Use the current checkout for code and synthetic fixtures. Desky remains read-only and unmodified until the user explicitly lifts the pause after the crash. The laptop path must stand on its own; the agent can continue any work that does not depend on unavailable hardware.

## Loop map and routing

The agent begins with **Loop 0** and **Loop 1** so every later change has a reproducible comparison. It then selects the next loop from the largest observed failure or resource burden. Loops 2–5 can interleave; none has a reserved slot or deadline. After a retained change, rerun its relevant baseline and consider **Loop 6** for integration. **Loop 7** is gated by the existing Desky pause. A failed gate returns the agent to the relevant loop rather than advancing by sequence number.

| Loop | Entry signal | Work until | Exit evidence |
| --- | --- | --- | --- |
| 0. Reproducible fixtures | No stable baseline or a fixture is ambiguous | Task state and expected result are independently observable | Fixture/run sheet and scoped capture evidence. |
| 1. Current behavior | Fixture exists or a retained change needs comparison | Current Adam path has repeated raw outcomes | Baseline report, resource trace, and failure classification. |
| 2. Lightweight runtime | High idle/peak use, retained model, or avoidable CPU load | A focused resource candidate is kept or rejected | Matched memory/latency/quality comparison and checkpoint. |
| 3. Computer use | Ordinary-app task fails or falsely reports success | A generic fix is kept or failure is explicitly unresolved | Task replay plus unrelated-app regression. |
| 4. Visual evidence | Observation quality or cost drives a desktop failure | Evidence route is kept, narrowed, or rejected | Matched task and resource results. |
| 5. Voice and memory | Real speech or recall failure appears | Focused fix is kept or rejected | Real-voice or dated-recall comparison. |
| 6. Integration | One or more changes have survived their local loops | Cross-feature regression is resolved | Laptop acceptance report and integrated checkpoint. |
| 7. Paired host/final stress | Desky pause is lifted and ordinary apps pass | Host comparison and final stress result are recorded | Paired report and final checkpoint. |

**Selection rule:** Favor a reproducible user-visible failure over speculative optimization. Among similar failures, choose the one with the largest effect on task success, first response, or laptop resource headroom. If a candidate is inconclusive, either gather the missing evidence or close it; do not keep cycling indefinitely without a sharper hypothesis.

**Agent continuation rule:** At every checkpoint, record the last measured result, the current blocker if any, the next loop to run, and the exact task/fixture that will decide it. Continue into that loop without requiring a new planning pass. A missing optional model, rate-limited provider, or unavailable app should lead to another useful eligible loop; only work dependent on that missing condition waits. Preserve the user's Desky pause and any direct stop instruction.

## The repeated implementation loop

Every proposed improvement uses the same loop, repeated until a measurable benefit is retained or the idea is closed.

```text
Observed baseline / reproducible failure
    → state one hypothesis and affected tasks
    → choose one focused change
    → implement in a reversible branch or focused commit scope
    → focused code checks and matched live/synthetic task replay
    → inspect raw outcomes, latency, memory, cost, and failure cases
    → keep + checkpoint commit | revise and loop | revert/close
    → update dated evidence before the next candidate
```

At the start of each loop record the commit, host, model/provider route, non-secret config differences, initial screen/audio, exact prompt, and expected task state. At the end record the result, action/tool trace, first meaningful acknowledgment, p50/p95 time when sample size supports it, CPU, process-tree PSS/RSS, service cgroup memory/peak, available RAM and swap, and provider-reported cost when present. An assertion or tool return is not proof that the user's desired state changed. If the effect is small relative to noise, call it inconclusive and either repeat or close the candidate.

**Checkpoint rule:** Commit after each retained large checkpoint and update the dated progress report with the measured result and limitations. Stage only files belonging to that checkpoint; the current tree contains unrelated untracked research drafts. Before committing code, review the diff and run the relevant focused checks plus the full suite at the integrated checkpoint. Do not bundle unrelated changes to create a larger-looking win. If a candidate fails, revert its experiment or leave it isolated; do not silently keep it in the main path.

## Loop 0 — make tasks reproducible

1. Reconcile the current Git commit and implementation with [PLAN](PLAN.md), [codebase status](codebase-status.md), and the dated handoff. Mark historical measurements as historical, and capture only settings needed to reproduce an experiment. Do not copy secrets into reports.
2. Define a compact task suite with objective pass/fail states: factual answer; file/system status; dated memory recall; settings toggle; browser report/table; terminal text/action; one document/app task; one longer multi-step desktop task. Keep both single-domain and mixed-domain prompts so tool filtering cannot accidentally shrink Adam's general ability.
3. Put GUI fixtures on a disposable workspace or isolated Xvfb session where possible. Before any provider-backed screenshot run, prove the capture contains only the intended window. Record the window ID/title/PID, crop bounds, image dimensions, and a local-only visual inspection. If the scope cannot be proven, the run stops before image upload.
4. Make a run sheet that distinguishes cold start, warm turn, and post-task recovery. Include end-to-end voice timestamps and the 5-second meaningful acknowledgment / roughly 10-second progress targets. Define the exact evidence that proves each task's result.
5. Check whether a local power measurement is actually usable. The prior AC/full-battery readings do not establish watts saved by Adam; do not hold all implementation work hostage to missing energy counters.

**Exit gate:** Another engineer or agent can repeat the same fixture and tell success from failure without reading Adam's final sentence as ground truth. Commit the fixture/run-sheet checkpoint if new repository artifacts are added. Return here whenever a later task has an ambiguous starting or ending state.

## Loop 1 — measure current behavior

Run the existing Adam path before tuning. Start with three exploratory repeats per task to expose fixture errors and large variance; select the important conditions for **at least 10 matched runs** when provider capacity permits. Keep exact raw outcomes, not only aggregate percentages. Include a cold service start and a warm turn, a longer model wait that should surface progress, and a mixed desktop request that needs more than one tool category.

Measure separate idle, wake/listening, ASR, TTS, OCR, browser, diarization/meeting, and combined-workload intervals. For memory, measure the service cgroup and child processes alongside PSS/RSS; record what remained loaded after a task. For energy, only report a delta if measurement conditions and resolution make it defensible. Record 429/retry outliers rather than hiding them in medians. Save the baseline as a dated report and commit it.

**Exit gate:** A task suite and resource envelope exist that subsequent changes can be compared against on the same laptop. If a fixture or instrument is unreliable, return to Loop 0. Do not fabricate a baseline from old snapshots.

## Loop 2 — reduce measured runtime burden

**Subloop A: find retained loads.** Exercise each optional component once in isolation, then observe idle recovery. Rank candidates by *measured incremental post-task memory and duration*, not disk size or library reputation. Examine Torch/Transformers-backed speaker verification, separation, and Nemotron alongside browser and visual workers; select the largest measured avoidable burden. The [runtime candidate review](lightweight-runtime-candidates-2026-10-05.md) explains the alternatives.

**Subloop B: one lifecycle change.** Choose the highest measured offender. Prototype a lazy or short-lived worker, explicit unload, or bounded job queue. Compare warm and cold response time, whole-task correctness, cgroup peak, post-task recovery, and CPU. Avoid replacing a large model and changing scheduling in the same experiment. If the worker saves RAM but exceeds the first-response target, try keeping the model warm only during an active conversation, then rerun the same fixture.

**Subloop C: alternate runtime only if needed.** Paired real-audio trials can compare Faster-Whisper `base.en`/`small.en` and `whisper.cpp`; speaker embedding may compare current SpeechBrain with a sherpa-onnx model; TTS may compare Kokoro/Piper. Preserve enrolled-speaker acceptance and do not treat a different embedding model as compatible with existing enrollment. Investigate Intel NPU only in an isolated trial after confirming the Linux driver, exact model support, and whole-system benefit. Stop the candidate if conversion/driver overhead or quality is worse.

**Keep gate:** meaningful reduction in idle/peak RAM, CPU, or battery use without a meaningful loss in voice quality, responsiveness, or task success. Report the tradeoff if the result is mixed. Commit each retained mechanism separately, then update the baseline before the next resource candidate.

## Loop 3 — repair general computer-use failures

Start with ordinary apps on the laptop: settings, browser, file manager, terminal, office/document editor, and a media or creator app if installed. For each, specify the initial state and visible/typed state that proves completion. Run the current agent first. Classify failures into observation missing, wrong target, stale target/focus, action failed, delayed app response, lost multi-step goal, blocked permission, or false success claim. This classification determines the *next* code change; it does not impose a universal controller.

For a repeated failure, select one repair: better scoped observation, a documented app control surface, stronger target identity/freshness, bounded related-action batching, changed tool feedback, recovery after uncertain results, or clearer final-state evidence. For example, use DOM for a page task and an app API for a task whose state that API exposes, while AT-SPI/screenshot remain available for native dialogs. Do not hard-code a path through one app. Rerun the failing task and at least one unrelated task to detect a generality regression. Repeat until the selected ordinary-app set is reliable enough to make the final game stress test informative.

**Keep gate:** higher exact task success or truthful recovery, with no new screenshot-scope leak, no blind repetition of consequential actions, and acceptable latency. Do not treat a click, focus change, or app launch as the requested outcome. Commit each retained generic fix; describe tasks that still fail.

## Loop 4 — choose visual evidence by task

On the *same* private screens and model route, compare screenshot-only, scoped OCR, OCR plus region detections, AT-SPI, browser DOM where applicable, and useful combinations. Include one text-heavy screen, one icon/geometry-heavy screen, one native dialog, and one browser page. Ensure annotations and region coordinates are actually present in the model's input and usable by the tool. Measure full-task success, model/tool turns, first useful response, total time, OCR/detector startup and warm cost, and process-tree memory. The prior three-trial report/fixture results guide hypotheses but are not acceptance evidence.

Test OmniParser on CPU first in an isolated laptop fixture. Its prior single CPU run was slow, so stop early if exploratory runs cannot fit a normal task budget. A GPU or Desky route waits for host authorization. Pick a default/fallback policy from measured task classes rather than a global claim that one representation is best. If a new evidence source improves one fixture but harms an unrelated app, narrow its trigger or revert it.

**Keep gate:** better correctness or fewer turns for the target class at an acceptable incremental memory/time cost. Commit the evidence-selection change and matched report together.

## Loop 5 — repair real voice and recall failures

**Voice loop:** On the actual microphone, record repeated enrolled and other-speaker commands, near/far/noisy speech, interruptions, pauses, and overlapping audio. Compare wake misses/false triggers, speaker false accepts/rejects, ASR command errors, endpointing, first task-relevant audible response, and total voice-to-voice time. If a weakness is established, change one stage—wake gate, VAD, ASR model/decoding, speaker fallback, or TTS lifecycle—and repeat the same audio/task set. Preserve recorded fixtures only in approved private locations; reports should contain scores and redacted examples.

**Memory loop:** Continue saving broadly. Evaluate which statements become memories, corrections, and retrieval/answer accuracy over day/week/year/timezone questions. Include semantically similar distractors and old versus recent facts. The goal is correct recall, not shaving milliseconds from a small in-RAM vector index. Change capture, temporal interpretation, candidate selection, or answer grounding only when a concrete recall failure identifies it. Rerun both the failure and prior passing cases after each change.

**Keep gate:** real-voice quality and memory answer accuracy improve or stay stable while local memory/latency remains practical. Distinguish transcription error from retrieval error before changing memory code. Commit independent speech and memory wins separately.

## Loop 6 — integrate and regress

Run the full ordinary-app, voice, memory, browser, tool, and mixed-request suite from a known state, plus a sustained session that overlaps listening, a desktop task, TTS, and optional model work. Check full test-suite results, actual live service state, first meaningful acknowledgment, truthful progress/finish, tool failures/retries, p50/p95 task time, cgroup peak, swap, post-task recovery, and actual per-task API cost. Review the final diff for host-specific assumptions and accidental disclosures. Repeat a failed class after fixing it; do not declare acceptance from a single lucky run.

**Exit gate:** selected tasks pass repeatedly, no severe false-success or capture-scope regression remains, and the laptop stays usable under concurrent work. Where the 5-second acknowledgment or 10-second progress target remains unmet, document the precise outlier path (for example provider 429) and whether it is controllable. Update [PLAN](PLAN.md), [codebase status](codebase-status.md), and the dated progress report, then commit the integrated checkpoint.

## Loop 7 — compare hosts and run the final stress case (authorization-dependent)

After the user lifts Desky's pause, begin with a fresh status/read-only audit and a recoverable configuration. Repeat the selected *same* suite with host-specific resources reported separately. Do not average away the 16 GB laptop result. Keep shared changes in code; keep installation, GPU, driver, and app availability in host config.

Only after general computer-use success, attempt one or two Shenzhen I/O levels as an unguided stress test on a visible authorized session. Use one fixed prompt and record navigation, failures, recovery, latency, and exact outcome. Do not add game-specific behavior or use the game result as the sole release gate. Finish with a final cross-host report, live-service status, remaining risks, and a checkpoint commit. Until authorization arrives, the laptop-only work can complete and be reported without this stage.

## What triggers a loop, pause, or direction change

| Trigger | Immediate response | Next loop |
| --- | --- | --- |
| Screenshot includes unrelated windows or unknown scope | Stop provider-backed GUI runs; restore fixture and inspect locally. | Fix capture/identity, replay scope checks, then resume the same task. |
| Crash, OOM, or swap storm | Stop that candidate, preserve non-secret diagnostics, restore known service state. | Reduce concurrency/model residency or revert; rerun from baseline. |
| Provider 429 or outage dominates timing | Keep the raw failed samples and route metadata. | Repeat when stable; do not attribute transport delay to a local optimization. |
| Change helps one fixture but breaks a mixed request | Keep broad capability as the default. | Narrow the trigger or revert; rerun both fixtures. |
| New model is lighter but fails real speech or task state | Do not ship it as default. | Tune only a focused cause, compare again, or close the candidate. |
| No material difference after matched repeats | Mark inconclusive or no benefit. | Stop the candidate and move to the next measured bottleneck. |
| Desky remains paused | Continue laptop-only work and read-only document/code review. | Revisit Loop 7 only after explicit authorization. |

The agent continues selecting and repeating loops until the integrated laptop exit gate is met. The commits and dated evidence should show which changes survived each loop and why. Desky's gate remains separate from laptop completion.
