# Adam implementation timeline and iteration loops

October 5, 2026 · Planning artifact. Durations are **focused engineering days**, not calendar promises. The laptop's availability, provider rate limits, and the still-paused Desky experiments can extend elapsed time.

## Destination and current boundary

Build Adam into a responsive, general-purpose voice and desktop assistant that remembers broadly, completes ordinary computer tasks with truthful outcome reports, and remains light on the 16 GB laptop. The main LLM remains hosted. Improvements must preserve broad tool access on mixed requests; settings pages, browsers, OBS, and Shenzhen I/O are evaluation cases, not special product modes.

The current [handoff](desktop-handoff-2026-10-05.md) already records opt-in browser DOM/navigation, scoped screenshot/OCR, AT-SPI, a model/tool loop, memory-date handling, lazy imports, retry/progress plumbing, and narrow synthetic wins. It also records unresolved whole-task GUI success, real-microphone voice latency and quality, concurrent memory peak, reliable power attribution, and a matched OmniParser comparison. Do not reimplement completed foundations merely because an older plan describes them as proposed. [Computer-use handoff](computer-use-plan.md) calls for observed failure-driven changes, with the model retaining task strategy.

**Access boundary:** Prepare code, synthetic fixtures, documents, and read-only analysis in the current checkout. Run disruptive laptop trials when the laptop is available. Desky remains read-only and unmodified until the user explicitly lifts the pause after the crash. The release path must work on the laptop without Desky data; paired-host acceptance can occur later.

## Overall schedule

The ranges include investigation, a focused change, and a matched retest. Multiple loops may be required; the schedule should expand in response to evidence rather than force a weak candidate through a deadline.

| Stage | Earliest position | Focused days for first pass | Output / exit gate |
| --- | --- | ---: | --- |
| 0. Freeze work and fixtures | Now | 1–2 | Exact baseline commit/config summary; private repeatable fixtures; measurement sheet; no unrelated worktree changes. |
| 1. Laptop baseline | Once available | 2–3 | Repeated voice, memory, desktop, browser, resource, and latency baseline with raw outcomes. |
| 2. Lightweight runtime loop | After baseline; can overlap Stage 3 analysis | 3–7 | One or two retained lifecycle or model-loading changes with reduced measured cost and no meaningful capability loss. |
| 3. Ordinary-app computer-use loop | After baseline | 5–10 | Representative tasks completed, failures classified, and only evidence-supported generic fixes retained. |
| 4. Visual-evidence comparison | Alongside Stage 3 | 3–6 | Matched screenshot/OCR/AT-SPI/DOM/OmniParser results; an explicit default/fallback policy. |
| 5. Voice and memory loop | After stable resource baseline; voice needs microphone access | 4–8 | Real-microphone and date-recall quality results; selected speech/memory fixes verified end to end. |
| 6. Integrated laptop acceptance | After retained changes | 2–4 | Full regression and mixed-workload run; usable first response, truthful outcomes, bounded resources, no major regressions. |
| 7. Desky comparison and final stress case | Only when separately authorized | 2–5 | Paired results by host; Shenzhen I/O attempted only after general navigation passes; final status documented. |

First laptop-only pass: roughly **20–40 focused days** if candidates are tractable, with loops and availability dominating the actual calendar. Stages 2–5 are workstreams, not a rigid waterfall. A failed comparison can end a candidate in one day; a promising but flaky route can require more cycles. The external Desky gate is intentionally outside that estimate.

An illustrative sequence, counting from the first available laptop workday rather than from today's date:

| Relative window | Main work | What can overlap | Checkpoint |
| --- | --- | --- | --- |
| Preparation before Day 1 | Stage 0 fixture and run-sheet work | Code reading and source review | Reproducible baseline package. |
| Days 1–3 | Stage 1 laptop baseline | No feature changes during measurement | Baseline report commit. |
| Days 4–10 | Stage 2 first resource cycles and Stage 3 first ordinary-app cycles | Stage 4 fixture setup and exploratory evidence trials | One focused commit per retained fix; refresh baseline after each. |
| Days 11–20 | More Stage 3/4 failure-driven cycles | Stage 5 real voice and memory trials when microphone access permits | Ordinary-app and evidence-selection checkpoints. |
| Days 21–30 | Stage 5 fixes, remaining resource candidates, Stage 6 integrated run | Documentation and host-independent cleanup | Integrated laptop checkpoint. |
| Beyond Day 30 if needed | Repeat any loop whose gate failed | Desky Stage 7 only after pause is lifted | Final paired report and stress-test checkpoint. |

The day numbers are a planning example, not a deadline or permission to skip a failed gate. If a stage needs more cycles, later rows shift. If the laptop is unavailable, fixture/document work can advance while live measurements wait.

## The repeated implementation loop

Every proposed improvement uses the same short loop. This is the unit of work, usually **half a day to three days** per candidate, repeated until a measurable benefit is retained or the idea is closed.

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

## Stage 0 — baseline preparation (1–2 days; can start now)

1. Reconcile the current Git commit and implementation with [PLAN](PLAN.md), [codebase status](codebase-status.md), and the dated handoff. Mark historical measurements as historical, and capture only settings needed to reproduce an experiment. Do not copy secrets into reports.
2. Define a compact task suite with objective pass/fail states: factual answer; file/system status; dated memory recall; settings toggle; browser report/table; terminal text/action; one document/app task; one longer multi-step desktop task. Keep both single-domain and mixed-domain prompts so tool filtering cannot accidentally shrink Adam's general ability.
3. Put GUI fixtures on a disposable workspace or isolated Xvfb session where possible. Before any provider-backed screenshot run, prove the capture contains only the intended window. Record the window ID/title/PID, crop bounds, image dimensions, and a local-only visual inspection. If the scope cannot be proven, the run stops before image upload.
4. Make a run sheet that distinguishes cold start, warm turn, and post-task recovery. Include end-to-end voice timestamps and the 5-second meaningful acknowledgment / roughly 10-second progress targets. Define the exact evidence that proves each task's result.
5. Check whether a local power measurement is actually usable. The prior AC/full-battery readings do not establish watts saved by Adam; do not hold all implementation work hostage to missing energy counters.

**Gate:** Another engineer can repeat the same fixture and tell success from failure without reading Adam's final sentence as ground truth. Commit the fixture/run-sheet checkpoint if new repository artifacts are added.

## Stage 1 — laptop baseline (2–3 available-laptop days)

Run the existing Adam path before tuning. Start with three exploratory repeats per task to expose fixture errors and large variance; select the important conditions for **at least 10 matched runs** when provider capacity permits. Keep exact raw outcomes, not only aggregate percentages. Include a cold service start and a warm turn, a longer model wait that should surface progress, and a mixed desktop request that needs more than one tool category.

Measure separate idle, wake/listening, ASR, TTS, OCR, browser, diarization/meeting, and combined-workload intervals. For memory, measure the service cgroup and child processes alongside PSS/RSS; record what remained loaded after a task. For energy, only report a delta if measurement conditions and resolution make it defensible. Record 429/retry outliers rather than hiding them in medians. Save the baseline as a dated report and commit it.

**Gate:** A task suite and resource envelope exist that subsequent changes can be compared against on the same laptop. If the laptop is unavailable, continue Stage 0 and read-only code analysis; do not fabricate a baseline from old snapshots.

## Stage 2 — lightweight runtime loop (3–7 days initially)

**Loop A: find retained loads.** Exercise each optional component once in isolation, then observe idle recovery. Rank candidates by *measured incremental post-task memory and duration*, not disk size or library reputation. Start with the largest avoidable path: Torch/Transformers-backed speaker verification, separation, and Nemotron; compare with browser and visual workers. The [runtime candidate review](lightweight-runtime-candidates-2026-10-05.md) explains the alternatives.

**Loop B: one lifecycle change.** Choose the highest measured offender. Prototype a lazy or short-lived worker, explicit unload, or bounded job queue. Compare warm and cold response time, whole-task correctness, cgroup peak, post-task recovery, and CPU. Avoid replacing a large model and changing scheduling in the same experiment. If the worker saves RAM but exceeds the first-response target, try a prewarm window or keep the model warm only during an active conversation, then rerun the same fixture.

**Loop C: alternate runtime only if needed.** Paired real-audio trials can compare Faster-Whisper `base.en`/`small.en` and `whisper.cpp`; speaker embedding may compare current SpeechBrain with a sherpa-onnx model; TTS may compare Kokoro/Piper. Preserve enrolled-speaker acceptance and do not treat a different embedding model as compatible with existing enrollment. Investigate Intel NPU only in an isolated trial after confirming the Linux driver, exact model support, and whole-system benefit. Stop early if conversion/driver overhead or quality is worse.

**Keep gate:** meaningful reduction in idle/peak RAM, CPU, or battery use without a meaningful loss in voice quality, responsiveness, or task success. Report the tradeoff if the result is mixed. Commit each retained mechanism separately, then update the baseline before the next resource candidate.

## Stage 3 — general computer-use loop (5–10 days initially)

Start with ordinary apps on the laptop: settings, browser, file manager, terminal, office/document editor, and a media or creator app if installed. For each, specify the initial state and visible/typed state that proves completion. Run the current agent first. Classify failures into observation missing, wrong target, stale target/focus, action failed, delayed app response, lost multi-step goal, blocked permission, or false success claim. This classification determines the *next* code change; it does not impose a universal controller.

For a repeated failure, select one repair: better scoped observation, a documented app control surface, stronger target identity/freshness, bounded related-action batching, changed tool feedback, recovery after uncertain results, or clearer final-state evidence. For example, use DOM for a page task and an app API for a task whose state that API exposes, while AT-SPI/screenshot remain available for native dialogs. Do not hard-code a path through one app. Rerun the failing task and at least one unrelated task to detect a generality regression. Repeat until the selected ordinary-app set is reliable enough to make the final game stress test informative.

**Keep gate:** higher exact task success or truthful recovery, with no new screenshot-scope leak, no blind repetition of consequential actions, and acceptable latency. Do not treat a click, focus change, or app launch as the requested outcome. Commit each retained generic fix; describe tasks that still fail.

## Stage 4 — visual evidence loop (3–6 days, paired with Stage 3)

On the *same* private screens and model route, compare screenshot-only, scoped OCR, OCR plus region detections, AT-SPI, browser DOM where applicable, and useful combinations. Include one text-heavy screen, one icon/geometry-heavy screen, one native dialog, and one browser page. Ensure annotations and region coordinates are actually present in the model's input and usable by the tool. Measure full-task success, model/tool turns, first useful response, total time, OCR/detector startup and warm cost, and process-tree memory. The prior three-trial report/fixture results guide hypotheses but are not acceptance evidence.

Test OmniParser on CPU first in an isolated laptop fixture. Its prior single CPU run was slow, so stop early if exploratory runs cannot fit a normal task budget. A GPU or Desky route waits for host authorization. Pick a default/fallback policy from measured task classes rather than a global claim that one representation is best. If a new evidence source improves one fixture but harms an unrelated app, narrow its trigger or revert it.

**Keep gate:** better correctness or fewer turns for the target class at an acceptable incremental memory/time cost. Commit the evidence-selection change and matched report together.

## Stage 5 — real voice and broad memory loop (4–8 days)

**Voice loop:** On the actual microphone, record repeated enrolled and other-speaker commands, near/far/noisy speech, interruptions, pauses, and overlapping audio. Compare wake misses/false triggers, speaker false accepts/rejects, ASR command errors, endpointing, first task-relevant audible response, and total voice-to-voice time. If a weakness is established, change one stage—wake gate, VAD, ASR model/decoding, speaker fallback, or TTS lifecycle—and repeat the same audio/task set. Preserve recorded fixtures only in approved private locations; reports should contain scores and redacted examples.

**Memory loop:** Continue saving broadly. Evaluate which statements become memories, corrections, and retrieval/answer accuracy over day/week/year/timezone questions. Include semantically similar distractors and old versus recent facts. The goal is correct recall, not shaving milliseconds from a small in-RAM vector index. Change capture, temporal interpretation, candidate selection, or answer grounding only when a concrete recall failure identifies it. Rerun both the failure and prior passing cases after each change.

**Keep gate:** real-voice quality and memory answer accuracy improve or stay stable while local memory/latency remains practical. Distinguish transcription error from retrieval error before changing memory code. Commit independent speech and memory wins separately.

## Stage 6 — integrated laptop acceptance (2–4 days)

Run the full ordinary-app, voice, memory, browser, tool, and mixed-request suite from a known state, plus a sustained session that overlaps listening, a desktop task, TTS, and optional model work. Check full test-suite results, actual live service state, first meaningful acknowledgment, truthful progress/finish, tool failures/retries, p50/p95 task time, cgroup peak, swap, post-task recovery, and actual per-task API cost. Review the final diff for host-specific assumptions and accidental disclosures. Repeat a failed class after fixing it; do not declare acceptance from a single lucky run.

**Exit gate:** selected tasks pass repeatedly, no severe false-success or capture-scope regression remains, and the laptop stays usable under concurrent work. Where the 5-second acknowledgment or 10-second progress target remains unmet, document the precise outlier path (for example provider 429) and whether it is controllable. Update [PLAN](PLAN.md), [codebase status](codebase-status.md), and the dated progress report, then commit the integrated checkpoint.

## Stage 7 — later Desky comparison and final stress test (authorization-dependent)

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
| Desky remains paused | Continue laptop-only work and read-only document/code review. | Revisit paired-host work only after explicit authorization. |

This is an iterative timeline, not a promise that every candidate will be implemented. The commits and dated evidence should show which changes survived each loop and why.
