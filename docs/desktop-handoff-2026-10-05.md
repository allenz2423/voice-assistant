# Desktop handoff — October 5, 2026

## Purpose

Continue the evidence-led work on Adam as a broad personal voice and desktop agent. The goal is to improve task completion, tool reliability, response latency, memory use, and power use across the 16 GB laptop and Desky. Keep the assistant general-purpose; settings dialogs, browser reports, terminal reading, and Shenzhen I/O are evaluation examples, not separate product modes.

This handoff describes the state pushed at commit `4685cbe` (`Add isolated browser DOM aid and lifecycle controls`) and the next work. It does not authorize experiments on Desky: the user paused them after the desktop crashed. The desktop Codex agent may review existing code, logs, screenshots, and measurements and return analysis, but must not edit the codebase, change machine configuration, run benchmarks, or interact with Desky until the user explicitly lifts that pause. The laptop is needed during the day, so defer disruptive laptop work until it is available.

## Read these first

1. `docs/PLAN.md` — overall objective, measurements, current workstreams, and open acceptance tasks.
2. `docs/codebase-status.md` — architecture, request flow, tool catalog, visual and speech paths, current limitations.
3. `docs/progress-report-10052026.md` — timestamped experiments and outcomes.
4. `docs/computer-use-plan.md` — general computer-use responsibilities, scenario set, safety and evaluation criteria.
5. This file — current handoff boundaries and suggested next steps.

Treat the dated progress report and current Git state as newer than older status paragraphs in the plan/status documents. In particular, the current laptop route is DeepSeek; some historical sections still describe earlier models or service states.

## Current pushed state

- Branch `main` is pushed to `origin/main` at `4685cbe`.
- The commit adds opt-in isolated browser navigation, DOM/text prefetch for explicitly requested Adam-browser tasks, memory release after read-only headless browser turns, URL restoration, mixed-domain tool preservation, and regression tests.
- The laptop Adam service was active and silent at the last check, using `deepseek/deepseek-v4.1-flash` through Relace with the WebUI connected and 66 tools available. Recheck the live status before relying on it; do not read or print API keys from `config.yaml`.
- The ignored local `:memory:.ses` file is machine/session state and is intentionally not in Git.
- The isolated browser feature is opt-in and was tested against generated local pages. It is not enabled in the laptop's active configuration by this commit.

## Evidence gathered overnight

### Visual and computer use

- Window-scoped OCR crops to verified focused-window bounds and maps text coordinates back into the full screenshot. In three matched Free Space Bunny trials per condition, both cropped and full-frame modes succeeded 3/3. Crop mode reduced input image area from 1.60 MP to 0.41 MP and median OCR time from 3.70 s to 2.91 s (21%). The sample does not establish a reliable whole-task latency win.
- In a separate synthetic settings task, an OCR text-state check removed false “screen unchanged” progress reports in all three candidate runs. Task success and model-call counts were unchanged; the small sample does not establish a resource improvement.
- A local generated report with 56 dense invoice rows was answered correctly in 3/3 Free Space Bunny runs with browser DOM text and 0/3 screenshot-only runs. DOM extraction took 7–11 ms. Read-only headless Edge added about 577 MiB PSS while open and returned near baseline after the turn released the browser.
- Desktop screenshot-prefetch and tool filtering have promising but small A/B results documented in `docs/PLAN.md`. Broad/mixed requests are intended to keep all tools needed by every requested domain.
- **Not yet done:** compare OmniParser against the same image/OCR task baseline. Do not assume it helps; measure success, turns, latency, CPU, and process-tree memory. The laptop's available OmniParser weights/runtime exist, and an isolated Xvfb settings fixture is in its local overnight-analysis directory, but that harness is not currently a repository artifact.
- General app navigation is not yet validated broadly. Shenzhen I/O remains a final stress test only after general computer use is capable; do not add game-specific navigation logic.

### Speech and memory

- Dated memory now stores the capture timezone and handles DST-specific offsets, overnight shifts, and ambiguous/nonexistent local clock times without inventing timestamps. Broader natural-language date phrasing remains open.
- An idle-startup availability check no longer imports PyTorch/Transformers unnecessarily. Across 10 isolated samples, median startup PSS fell about 44% (1,153,762 to 649,297 KiB) and cgroup memory fell about 34% (870,202 to 573,928 KiB). This is an isolated startup comparison, not a concurrent-workload peak.
- Loading the actual cached diarization model in a synthetic CPU subprocess raised PSS to about 1.26 GiB after inference. The silence fixture did not measure recognition quality; full concurrent speech/diarization/TTS peak memory remains open.
- Wake-word fallback now uses Whisper confidence and speaker similarity to decide when to run separation. Initial eSpeak mixtures routed 14/16 overlap cases to separation, skipped 8/8 clear background clips in the first set, and triggered on 3/24 varied background clips. These are synthetic results; real microphone and noisy-room accuracy is unmeasured.
- Cloud Whisper context hints did not change the three tested transcripts. Local VAD filtering did not help clean or 20 dB samples and worsened the 10 dB sample. Lowering the endpoint silence threshold truncated longer pauses, so production ASR settings were not changed. Human microphone quality and full voice-to-voice latency remain open.

### Latency, reliability, and resources

- The user target is a task-relevant first acknowledgment within 5 seconds. Longer work should show meaningful progress about every 10 seconds. The latest typed DeepSeek rigatoni/penne run series had a 3.92 s median, but HTTP 429 retries pushed some turns past 11 seconds; the target is not yet reliable. Do not count a processing spinner or capture earcon as semantic acknowledgment.
- Across three synthetic live file reads, DeepSeek via Relace completed all three (7.77 s median, one 30.51 s retry outlier); free Space Bunny completed all three (3.33 s median). These are narrow tool-path samples, not general model rankings or voice latency.
- Failed/uncertain tool results and false capability refusals receive up to three corrective prompts. Provider transport failures receive up to three retries. Live WebSocket checks showed tool starts/results and recovery/retry progress arriving before final responses.
- Laptop battery readings were full on AC and `power_now=0`; Intel RAPL energy counters were unreadable and NVIDIA telemetry was unavailable. A 30-second idle-listening sample showed about 3.5% of one logical CPU and 1.325 GB cgroup memory. This is not a power/energy result. Do not claim an Adam power reduction without usable energy readings.
- The active laptop service's later `systemctl` snapshot showed approximately 1.7 GB current memory / 1.9 GB peak, which is not directly comparable to isolated startup PSS results. Collect matched RSS/PSS/cgroup snapshots under the same workload when the machine is available.

## Operating instructions for the desktop agent

Until the user lifts the Desky pause:

1. Do read-only analysis of existing Desky artifacts only. Report the specific evidence, file paths, and uncertainty; do not infer new benchmark results from stale plans.
2. Do not start or stop Adam, change models or configuration, install packages, launch GUI applications, interact with monitors, run tests/benchmarks, or edit/commit code on Desky.
3. Do not wake physical monitors, enter credentials, interact with Bitwarden, or attempt the game. Defer CAPTCHAs if they arise in a later explicitly authorized run.
4. Do not print secrets, tokens, private file listings, or unrelated screenshot contents. Use synthetic fixtures and redact sensitive values in logs.

For repository sync, first check `git status --short --branch` and the current commit. If the tree is clean, fetch and fast-forward `main`; never discard local work with a hard reset. Confirm `4685cbe` or a later commit is present. The laptop checkout that created this handoff already has the pushed commit.

## Next evaluation sequence after machine access is authorized

1. **Freeze a comparable baseline.** Record the commit, host, live model/provider, non-secret settings relevant to the experiment, fixture state, exact prompt, and starting screen. Measure exact task state, first meaningful acknowledgment, total latency, model/tool turns, retries, reported token use/cost, process-tree PSS/RSS, cgroup memory, CPU/GPU utilization, and power readings where available.
2. **Finish the visual evidence comparison.** On the same private synthetic fixture and free vision model, compare screenshot-only, OCR text, OCR plus OmniParser, and DOM/accessibility text where relevant. Use OmniParser CPU first on the laptop; separately measure any GPU route only on an authorized host. Make sure the model sees the region annotations and can use them. Include cold startup and warm inference; report worker memory separately from the Adam process.
3. **Use matched repetitions.** Run the same task and initial state for each condition. Treat small samples as exploratory; use at least 10 runs per selected scenario/condition for acceptance decisions where provider availability permits. Report p50/p95 and raw run outcomes. A change that helps one task but breaks another broad capability is not a win.
4. **Change one focused factor at a time.** State the target metric and likely regressions before implementation. Run the same baseline afterward. Keep a change only when task success is not meaningfully worse and the targeted improvement is measured; otherwise revert or label it inconclusive.
5. **Preserve broad agent behavior.** Shared behavior belongs in the codebase; host-specific installs/routes belong in config. Only narrow schemas for clearly single-purpose requests. Mixed requests must retain tools for every domain. Do not build a game-specific navigator.
6. **Complete the unfinished workstreams.** Add representative real-microphone ASR trials, 16 GB concurrent memory tests, voice acknowledgment/heartbeat measurements, and usable laptop power readings. Recheck memory recall with date/week/year questions against the current temporal memory schema.
7. **Run Shenzhen I/O last.** First show general navigation success across ordinary apps. Then use one or two levels as a stress test on the user's visible computer, with one fixed prompt and no strategic hints. Stop before consequential actions; defer CAPTCHA or authentication blockers instead of working around them.
8. **Finish with paired host evidence.** After the user authorizes Desky experiments again, rerun the winning shared changes on both systems. Keep host-specific results separate and record the final live-service status. Update the plan and timestamped progress report before committing and pushing.

## Acceptance reminders

- Acknowledgment target: within 5 seconds of the end of the request; ordinary factual answers should meet the same target.
- Long tasks should provide a meaningful status update when progress has been absent for about 10 seconds.
- Task correctness and truthful outcome reporting matter alongside speed. A tool returning is not proof that the user's goal succeeded.
- Track low-cost API usage as a simple per-task cost, not only token totals. Some current logs do not contain provider-reported cost.
- Separate measured improvements from hypotheses. Synthetic audio/GUI and silent-TTS tests do not prove real microphone quality, audible feedback, or user-desktop success.
