# Adam: General-Purpose Voice and Desktop Agent
## Current Focus and Technical Notes

## Overall Goal: General-Purpose Agent

Adam should act as a capable, agentic assistant that can understand and carry out a broad range of user requests across voice, desktop, and other available tools. The plan improves general capabilities; individual apps and tasks are examples for evaluation, not limits on what Adam is meant to handle.

## End Goal and Live Evaluation

The end goal is an agent that feels responsive on ordinary requests, makes reliable progress on multi-step tasks, and can use lower-cost vision models effectively with visual or text aids, without imposing a large local compute or memory burden. Adam should remain useful on a 16 GB laptop and modest desktops when the main LLM is hosted.

The project is complete when Adam has become:

1. An intelligent personal assistant that remembers useful context and handles broad everyday requests.
2. A capable visual assistant that can navigate desktop tasks with lower-cost models.
3. Much lower-latency in ordinary interactions and useful progress on longer tasks.
4. More reliable at selecting, calling, and verifying tools.
5. Practical on lower-spec hardware, including a 16 GB laptop and older desktops.
6. Lower-power during idle listening and active use.

### Improvement Loop

Repeat this loop throughout the work; run each baseline and meaningful retest on both the 16 GB laptop and Desky, using the same task, model route, settings under comparison, and starting state on both:

1. Benchmark first: record task success, CPU use, GPU use, RAM, power readings where available, end-to-end latency and time to first response, tool-call reliability, model/provider route, token usage, and actual request cost.
2. Analyze the failures and propose a few focused changes, stating which measurements each change is expected to improve and what might regress.
3. Implement one focused change (or a tightly related small set) so its effect is identifiable.
4. Run the same live tests again and compare the measurements and task outcomes with baseline.
5. Keep changes that produce a measurable improvement without unacceptable regressions; revise or revert changes that do not.
6. Repeat across the general assistant, visual navigation, tool use, and each machine's resource targets. Finish with a complete end-to-end regression benchmark on both machines; label machine-specific results rather than averaging away hardware differences.

Keep Adam's default tool set broad so it remains a general-purpose agent. Put tools that depend on a particular host or installation in `config.yaml`; put behavior that should work on every system in shared code. Narrow tool schemas only for clear, single-purpose routing where the request is fully served by a dedicated tool. Mixed requests must retain the tools needed for every requested capability.

Use the live Adam bot for behavioral evaluation, not only mocked unit tests or direct model prompts. The systemd `adam` service is currently active on `desky`; use it as the first integration target. Keep a recoverable test configuration and change only the route or visual evidence under comparison. Keep each task and starting state consistent across variants. Record task success, end-to-end latency, time to first acknowledgment, actual model/provider route, tool calls and duration, token usage and actual request cost, plus CPU, RAM, GPU, and available power readings. Restore the normal service configuration after each live test batch and repeat enough to distinguish a stable result from a one-off.

Initial experience targets, to confirm against a live baseline:

- A user-facing acknowledgment that shows Adam recognized the task within 5 seconds of the end of the user's request. For voice, this means an audible, task-relevant response; the wake/capture earcon alone does not count. For text, it means visible assistant content. A quick task's answer can serve as its acknowledgment; a longer task should receive a concise confirmation of the understood goal.
- A straightforward conversational answer, such as comparing rigatoni and penne, should also reach its first useful response within 5 seconds; taking close to a minute is a clear failure.
- For longer tasks, acknowledge promptly and give a meaningful update whenever there has been no visible progress for about 10 seconds.
- Improve visual-task success and completion time with lower-cost models while keeping added local vision compute and memory as low as practical. Model quality alone is not the metric; account for the aid's inference time and the tool/model turns it saves.

Live model candidates provided for the first comparison (check route availability and current cost when each test runs):

- [`stealth/space-bunny-alpha`](https://openrouter.ai/stealth/space-bunny-alpha) — free candidate; test promptly because OpenRouter currently lists it as scheduled to end on 2026-10-05.
- [`deepseek/deepseek-v4.1-flash`](https://openrouter.ai/deepseek/deepseek-v4.1-flash) — route through Relace by setting the OpenRouter provider restriction to `relace`.
- [`qwen/qwen3.7-flash`](https://openrouter.ai/qwen/qwen3.7-flash).

Treat the stated approximate per-request costs as estimates. Capture actual provider usage/cost for each run and express results in simple per-task amounts as well as token counts.

Evaluation sequence (run shared conversational and tool-call tasks on both laptop and Desky; include host-specific resource measurements for each):

1. [ ] Establish a live baseline for a short conversational request, a simple tool request, a multi-step task, and the desktop stress tasks. Include a short answer such as rigatoni versus penne; close to a minute is a failure.
2. [ ] Run the same tasks through all available candidate models on the live Adam bot using its actual tool-calling path; record latency, task success, tool behavior, selected provider, and actual cost. For DeepSeek through OpenRouter, pin the provider with `provider_only: ["relace"]` rather than changing the LLM protocol provider to `relace`.
3. [ ] For visual tasks, compare the image alone with OCR text, OmniParser region detections, accessibility/browser text, and useful combinations. Measure task success and latency against local CPU/RAM/GPU use and the number of model/tool turns.
4. [ ] Tune prompts, evidence selection, image handling, and safe tool batching from the observed failures; rerun the same live tasks after each meaningful change.
5. [ ] Carry winning settings into laptop speech, memory, power, and memory-footprint evaluation; retain a repeatable set of end-to-end acceptance tasks.

### Initial Live Model Baseline (2026-10-03)

Exploratory live-provider comparisons ran the production `AdamBrain` and tool loop on `desky`, with the three listed model routes, a fresh in-memory configuration, isolated empty memory storage, and silent TTS. This exercised Adam's real prompt construction, provider calls, and (for one case) tool loop, but it did not submit a request through the already-running systemd daemon or measure microphone, STT, audible acknowledgment, or speech playback. The table shows two repeated runs of the same no-tool prompt; provider-reported cost was captured on the latest run only. Treat timing as an early range, not a stable benchmark.

| Model route | Same prompt: brain time (2 runs) | Input tokens | Latest provider-reported cost* | Notes |
| --- | ---: | ---: | ---: | --- |
| `stealth/space-bunny-alpha` | 2.82–6.54 s | 15,566 | 0.00000 credits | One run missed the provisional 5 s first-response target before real TTS. |
| `deepseek/deepseek-v4.1-flash` via Relace | 5.46–11.87 s | 15,305 | 0.000251 credits | High latency variance; route was pinned to Relace with provider fallbacks disabled. |
| `qwen/qwen3.7-flash` | 2.97–3.37 s | 15,740 | 0.000296 credits | Fastest and most consistent in these two samples. |

\*OpenRouter's response reported cost in credits; confirm the billing conversion and collect multiple samples before treating this as a per-task price. All three models received about 15k input tokens for a short conversational turn. A previous similar prompt measured 5.81 s for Space Bunny, 4.40 s for DeepSeek, and 3.71 s for Qwen.

Space Bunny also answered a read-only CPU/RAM question through `get_system_status`: the tool took 28 ms and the complete two-call brain loop took 4.11 s, with about 15.8k input tokens on the first call and 16.1k on the second. The tool path worked, but its answer included a minor unsupported flourish (“barely working”).

The large prompt cost on an ordinary no-tool question is an immediate optimization lead: inspect the system prompt, memory context, and tool-schema payload, then test whether simple conversational turns can omit irrelevant tools/context without hurting tool selection or general capability. Keep the full live-tool path for requests that need it. The active `adam` systemd service was confirmed running, but exposes no text/request socket and listens through a continuous microphone loop. Therefore the next comparison must add a controlled request through the daemon's speech path before claiming end-to-end live-bot results; keep the existing brain/provider measurements labeled as live production-code integration tests, not full daemon tests.

### Paired Laptop and Desky Baseline (2026-10-03)

The same short conversational prompt and read-only system-status prompt were each sent once through each of the three candidate model routes on both machines (six cases per host). The harness used production `AdamBrain`, isolated memory, silent TTS, and real provider calls; it did not use the active microphone/speaker daemon. This is an initial comparison, not a statistically stable benchmark.

| Host | Simple-answer brain latency | System-status brain latency | Reported cost across the three routes (credits) | Host/resource notes |
| --- | ---: | ---: | ---: | --- |
| Desky | 2.4–7.6 s | 5.0–10.2 s | 0.000893 / 0.001634 (simple / status) | Adam service `MemoryCurrent` ~1.65 GB in one snapshot; harness peak RSS ~100 MB. NVIDIA snapshots showed no request-time utilization increase. |
| Laptop | 4.4–7.9 s | 5.5–17.2 s | 0.000826 / 0.001962 (simple / status) | Adam service `MemoryCurrent` ~2.82 GB in one snapshot; harness peak RSS ~80 MB. `nvidia-smi` unavailable; 20-second idle battery sample reported full battery, but RAPL readings were inaccessible. |

The provider reports costs in `credits`; confirm their billing conversion before treating them as USD. A prompt-only instruction reduced extra tool calls inconsistently, so the implemented change also narrows the tool schema for ordinary system-status questions to `get_system_status` (and `list_processes` only when process details are requested). Explicit shell requests, compound actions, and requests for another capability keep the full tool set. Host-specific tools belong in `config.yaml`; shared behavior belongs in the codebase. On the final paired retest, telemetry input tokens fell from 112,132 to 32,991 on Desky (71%) and from 168,672 to 27,984 on the laptop (83%) across the three model routes. Extra tool calls fell from four to three on Desky and eight to three on the laptop; every final status task used exactly one status tool. Provider-reported telemetry cost fell from 0.001634 to 0.000851 credits on Desky and 0.001962 to 0.000582 on the laptop. Simple conversational turns still use roughly 14–16k input tokens.

The latency effect is not established: the laptop's three status timings improved in this one-sample retest, while Desky DeepSeek rose from 10.20 to 19.49 seconds. Treat lower schema/token/tool counts as the clear result and latency/cost as early, cache- and provider-sensitive observations. The focused routing tests pass (`20 passed` across `test_brain_system_status_routing.py` and `test_brain_tool_round_trip.py`). Next compare repeated p50/p95 runs, reduce the conversational prompt/context burden without losing general tool capability, and repair power/resource measurement gaps.

An additional Qwen A/B experiment found that the always-loaded desktop setup skills account for avoidable context on ordinary requests. A conditional prompt now includes those skills for desktop-related requests and their immediate follow-ups, while general requests retain the shared system prompt and the complete tool catalog. In matched conversation samples, input fell from 14,647 to 12,069 tokens on the laptop and from 15,803 to 13,225 on Desky; time-tool requests showed a similar 16–18% reduction across both LLM calls. The Qwen time request still selected and completed exactly one `get_current_time` call on each host with all 68 laptop / 71 Desky tools exposed. The production conditional prompt measured 2.95 s for the conversational request and 4.08 s for the time-tool request on the laptop, and 2.78 s / 3.43 s on Desky. Latency samples are too sparse and provider-sensitive to establish a consistent speedup; retain the change for reduced input/context and preserve full desktop guidance on GUI tasks. The focused suite now passes `20` tests.

Desky's local Codex agent produced the comparison and next-step recommendations in `/home/incoming/.local/state/adam/overnight-analysis/2026-10-03-desky/results-comparison.md`. Its host snapshots put Adam's service `MemoryCurrent` near 1.65 GB on Desky and 2.82 GB on the laptop, but those one-time readings do not isolate resident models; track total service/cgroup memory separately from per-harness RSS.

### Iteration: direct system-status dispatch (2026-10-03)

The live `AdamBrain` path now dispatches a narrowly defined, factual hardware-status request straight to the read-only `get_system_status` tool without a model request. Requests for process lists, shell commands, explanation, comparison, or advice stay on the model path with the relevant tools available. The result is spoken directly, avoiding model latency or invented telemetry. The formatter now includes NVIDIA utilization and does not claim an AMD GPU is active when ROCm reports a driver error; CPU utilization is sampled from `/proc/stat` during the explicit status call. Before direct dispatch, three free `stealth/space-bunny-alpha` runs per host had two-call medians of 5.86 s on the laptop and 5.57 s on Desky. After skipping the final answer call, one-model-call medians were 1.45 s and 1.32 s. Direct dispatch then measured 104 ms median on the laptop and 103 ms on Desky over three production-code runs, with zero model requests. Per-request input tokens fell from roughly 4.3k to 2.05k on the laptop and from 5.8k to 2.76k on Desky in the one-model-call variant. The full suite passes on both hosts: 306 passed, 11 skipped on the laptop; 313 passed, 4 skipped on Desky. The different skip counts reflect host-specific OCR/GPU support. Desky returned both NVIDIA GPUs and utilization; the laptop has no supported GPU telemetry in this environment. The 100 ms CPU sample varied noticeably across laptop smoke runs, so treat it as an instantaneous sample, not a stable load average. All these are silent-TTS brain-path timings, not end-to-end voice latency. Paid cross-checks before direct dispatch also completed with one status call (DeepSeek via Relace on Desky, Qwen and DeepSeek on the laptop). Keep this iteration; gather larger paired samples before treating model-backed latency improvements as stable. Artifacts are under `/home/incoming/.local/state/adam/overnight-analysis/2026-10-03-{laptop,desky}/` and `/tmp/adam-free-status-after-cpu-sample-*.log`.

### Iteration: compact tools for focused desktop navigation (2026-10-03)

For a single-domain screen interaction, Adam now exposes only `computer_control`, `capture_screenshot`, `observe_desktop`, `focus_window`, and `list_windows`; compound tasks and requests that require another domain keep the full tool set. On Desky, a matched free-model A/B on a private Xvfb/Openbox dialog used 70 tools in the baseline and five after filtering. Baseline → filtered: six → four model calls; 96,422 → 29,584 aggregate input tokens (69% fewer); 22.15 → 18.82 seconds end-to-end (15% faster in this single pair). On the laptop, a separate repeat used 67 tools in the baseline and five after filtering; input fell from 77,665 to 29,287 tokens (62% fewer), while model calls fell from five to four. Elapsed time was 20.12 seconds baseline and 40.04 seconds filtered because one free-model request took 21.63 seconds; earlier laptop filtered samples ranged from 15.15 to 31.72 seconds. This confirms a large prompt reduction, not a reliable latency win. The dialogs disappeared and window enumeration reported none remaining, but the click results were labeled `uncertain` because focus was not stable for a post-action snapshot; keep outcome verification and action-status wording under investigation. Laptop traces are in `/home/incoming/.local/state/adam/overnight-analysis/2026-10-03-laptop/shenzhen-sandbox/nav-{baseline,filtered}.log`; Desky traces and harness are under `/home/incoming/.local/state/adam/overnight-analysis/2026-10-03-desky/shenzhen-sandbox/`. Desky Codex reviewed the Desky traces.

The actual Shenzhen I/O level test remains blocked: Desky's physical monitors are asleep, so testing uses a private Xvfb/Openbox display. The game binary was found and launched there, but exited during `SteamAPI_Init` with `connect to global user failed`; the isolated run did not enter a level. A prior Gamescope headless capture was black. Do not wake the physical display; resolve isolated Steam/game startup or use another private rendering path, then attempt one or two levels with the free vision model. The Xvfb dialog tests exercise basic screenshot and window-state handling, but the observed click was uncertain and this task does not substitute for the game benchmark.

### Iteration: lazy OmniParser startup (2026-10-03)

Adam previously loaded the OmniParser subprocess during every service startup. It now starts on the first visual-grounding request by default; set `computer_vision.preload_on_startup: true` to opt back into warm startup. After restarting the service, Desky's cgroup memory went from about 1.5 GB to 0.89 GB with the worker absent at idle. On the laptop, the worker process (about 749 MB RSS before the change) was also absent, but the service cgroup stayed around 1.8 GB, so a total-memory reduction was not established on that host. A cold blank-image probe, including model load and inference, took 2.64 seconds on laptop CPU and 2.04 seconds on Desky GPU; warm inference itself took about 0.50 seconds and 0.05 seconds respectively. That first-use delay is the tradeoff for avoiding startup and idle worker cost. Power was not directly measurable. Later, the laptop's service memory fell to 0.88 GB after correcting its ASR route; keep these causes separate when comparing the earlier measurement. Latest full suites: laptop 312 passed, 11 skipped; Desky 319 passed, 4 skipped. Recheck RAM during normal navigation and long-idle intervals, measure laptop power, and assess whether the first-use delay is acceptable before treating this as a final policy.

### Iteration: correct cloud ASR routing (2026-10-03)

The laptop configuration specified `stt.provider: cloud`, but the factory did not recognize that value and silently fell through to local Faster-Whisper `small.en`. The factory and wake-candidate path now treat `cloud` as a cloud provider, use the configured OpenRouter Whisper endpoint for command transcription, and retain the local `base.en` model for private custom-wake spotting. OpenRouter's current transcription API accepts OpenAI-style multipart uploads and reports request usage/cost; Adam now records only reported cost metadata in optional telemetry, without retaining transcript text. A 3.62-second synthetic command was correctly transcribed on all three repeats per route. Laptop: local `small.en` warm median 649 ms; cloud Whisper Turbo median 520 ms, with reported per-request costs from $0.000012 to $0.000111. Desky: local `small.en` warm median 125 ms; the same cloud model median 681 ms at $0.000012 per request in three samples. This is a synthetic-speech smoke test, not a voice-quality benchmark, and cloud latency varies. On laptop restart, service cgroup memory fell from about 1.79 GB to 0.80 GB while keeping local wake spotting; Desky remains configured for local ASR. The correction and cost reporting tests pass on both hosts. Repeat on representative speech before changing the preferred route. [OpenRouter transcription API](https://openrouter.ai/blog/tutorials/transcription-on-openrouter/).

### Iteration: safe parallel reads and ordinary conversation routing (2026-10-03)

An instruction-only prompt experiment asking the model to batch lookups was not reliable: a laptop research run grew from 76.9 seconds / 7 model calls / 12 tools to 139.8 seconds / 12 calls / 21 tools and unexpectedly touched browser observation tools. The added prompt text was reverted. Batching is now enforced in the executor for a narrow set of independent native reads (`web_search`, `fetch_webpage`, `get_weather`, `get_system_status`, and `list_processes`); blocking web, status, and process implementations run in worker threads so `asyncio.gather` does not stall on synchronous I/O. A batch runs only when every call is native, available, schema-valid, and allowlisted. Results are still added to conversation history in model call order. GUI observation, desktop mutations, file operations, custom tools, and mixed batches stay sequential.

The free Space Bunny research task emitted paired web searches in all five Desky tool-call hops. That run completed in 44.8 seconds with 6 model calls and 10 tools; the paired tool durations sum to 7.9 seconds, while overlapping each pair took about 4.6 seconds in total. A laptop run completed in 55.1 seconds with 7 model calls and 12 tools. Earlier same-topic samples ranged from 76.9 to 97.7 seconds with 7–10 calls and 12–20 tools, so the live result is encouraging but not a controlled speedup estimate; provider latency and the model's chosen research depth varied. The gateway response did not expose usage/cost metadata in these runs. Repeat the same model prompt enough times and collect per-hop timings and usage before quantifying an end-to-end improvement. The free model initially returned HTTP 404 because the test inherited `provider_only: ["dekallm"]`; clearing the provider restriction for this route produced HTTP 200. The repo setting was left unchanged.

A second experiment omits tool schemas for short ordinary questions when wording contains no live-information, memory, system, file, chat, desktop, or action intent. The guard is conservative and tested against both allowed and disallowed examples; images always retain the normal tool path. On the free-model pasta comparison, both full-catalog laptop runs answered without using tools (4.4 and 6.5 seconds); tool-free runs took 1.5 and 3.4 seconds. On Desky, full-catalog runs took 6.9 and 11.2 seconds, while tool-free runs took 2.4 and 3.7 seconds. The currently gated laptop production path repeated at 2.7 seconds twice and sent zero tools. Each variant recommended rigatoni, but some answers inaccurately described penne as smooth; measure factual quality across a wider conversational set before expanding this shortcut. These are silent-TTS brain-path measurements, not audible end-to-end latency. CPU/RAM impact and billed token counts were not available from these requests.

Two more free-model requests also used zero tools on both hosts: explaining entropy took 2.8 seconds on the laptop and 4.5 seconds on Desky; drafting a short thank-you note took 1.6 and 2.1 seconds. Both answers were useful and followed the requested format. Long-running model or tool waits now get a short spoken heartbeat after 10 seconds; fast waits skip it. A mocked timing test verifies both cases. On a live multi-step free-model research run, the heartbeat fired during a pending model call at 31.8 seconds on the laptop and 33.2 seconds on Desky; existing hop progress had already spoken at 12.4 and 12.7 seconds. The corresponding silent-TTS brain runs completed in 43.1 and 44.6 seconds. This confirms the update reaches the TTS interface while a real task is still running, but audible playback and overlap remain unverified because both machines were kept silent.

Pure factual status-plus-process requests also skip the model and now run the two independent local reads concurrently. Three production-code runs had a 133 ms median on laptop and 114 ms on Desky, zero model calls. This is a small saving over the earlier serial medians of 149 ms and 134 ms; process sampling and host load are noisy at this scale.

Latest complete suites after the heartbeat and final gate revision: laptop `328 passed, 11 skipped`; Desky `335 passed, 4 skipped`. The intermediate gate revision had two desktop-routing failures; the classifier was tightened and the previously failing screen-chat and workspace tests now pass. Keep the tool gate narrow, retain batching only for independent allowlisted reads, and repeat the paired live task set before calling the latency changes stable.

After the live tests, both `adam` services remained active with `tts.engine: silent`. The laptop service cgroup used about 0.73 GB; Desky used about 1.40 GB. Desky's main process showed about 2.5 GB RSS / 2.48 GB PSS, including roughly 1.1 GB file-backed mappings, so use cgroup memory for the service limit and record process PSS separately. This is a single post-run snapshot; compare again at a matched idle interval before attributing the higher Desky figure to any code change.

### Active-service microphone injection check (2026-10-03)

The laptop service has no text-request socket, so a controlled `eSpeak` utterance (“Hey Adam, what is one plus one?”) was sent to its existing PulseAudio capture stream through a temporary null sink. The original physical source was restored and the null sink unloaded in a `finally` block. The daemon captured 3.76 seconds and its local wake-candidate ASR correctly recognized the phrase, but speaker verification rejected eSpeak's voice and the request did not reach the model. This verifies audio capture and the local wake transcript only, not end-to-end command latency.

A follow-up used a previously stored wake recording after a separate local classifier labeled it a general question. That classification was wrong: the clip asked about a desktop chat and the active laptop service performed a focus plus click/type/press/inspect sequence. Results included `uncertain`; the final GUI outcome could not be confirmed. No dedicated message-send, file, purchase, close, or application-launch tool was called, but the UI sequence did include Enter, so a submitted chat message cannot be ruled out. The temporary audio route was restored and removed, the service stayed active and silent, and window enumeration afterward showed only the existing Steam window. Do not replay stored wake audio for live-bot benchmarking. Use fresh synthetic speech with speaker verification appropriately accounted for and a dedicated isolated GUI before repeating an end-to-end service test.

The Shenzhen I/O level test also remains blocked: a fresh launch in Xvfb using the installed Linux binary plus explicit `SteamAppId`/`SteamGameId` still exited at `SteamAPI_Init(): connect to global user failed`. The game could not connect to Steam's global user. No physical display was woken and no level was entered. Resolve headless Steam authentication or provide another isolated launch path before retrying the one-or-two-level stress test.

### Iteration: grounded personal drafting and action claims (2026-10-03)

A 15-request repeated baseline on the free `stealth/space-bunny-alpha` route used five short, no-tool prompts three times each on both hosts. The model-path median was 2.72 s on the laptop (range 1.23–3.72 s) and 2.25 s on Desky (range 1.40–4.25 s). The provider response logs reported an average 3,472 input / 93 output tokens per call on the laptop and 4,388 / 105 on Desky. Answers to factual/conversational prompts were generally useful, though some pasta answers incorrectly called penne smooth. In the six baseline thank-you drafts, four supplied a specific neighbor name that the user had not provided.

The shared system prompt now says to use only personal facts from the request or trusted memory, omit or mark unknown details, and never claim an action completed without a successful tool result. The repeated 15-call set after the drafting rule averaged 3,512 input / 84 output tokens on the laptop and 4,428 / 98 on Desky: about 40 extra input tokens per call, with no consistent latency direction. Three repeated drafts per host after adding the drafting rule supplied no unsupported proper names (two used a placeholder); some drafts still invented that the user had been away. A stricter follow-up still made one direct, untrue clipboard-copy claim in six drafts. After adding the explicit tool-result rule, the next six drafts made no direct claim that copying had happened, but continued to offer copy/save actions and assume the user had been away. Keep the small prompt rules as a cautious reduction in names and completed-action claims; personal-context invention remains open. All these requests used no tools and silent TTS through the production `AdamBrain` path, not the active microphone daemon. The test logs include input/output tokens but do not include provider-reported billed cost. Artifacts are in each host's `free-chat-repeat*.jsonl` and `free-draft-guard-repeat*.jsonl` under `/home/incoming/.local/state/adam/overnight-analysis/2026-10-03-{laptop,desky}/`.

The latest full suites after this prompt update passed: laptop `336 passed, 11 skipped`; Desky `343 passed, 4 skipped`. Next test a broader set of user-voice drafts and action-claim scenarios, check whether supported models obey the rules consistently, and include actual provider usage/cost telemetry before drawing conclusions.

At 07:42 EDT both systemd services were restarted to load the tested code and then verified `active` with silent TTS. Their immediate cgroup readings were about 1.00 GB on the laptop and 0.89 GB on Desky. These readings are lower than the pre-restart post-test snapshots above; take comparable idle samples before drawing a memory trend.

At 09:31 EDT, both services were still active with silent TTS after the isolated Steam attempt. A new cgroup snapshot read about 1.06 GB on the laptop and 1.21 GB on Desky. This is one sample after additional idle time, not evidence of a cause or stable growth rate; compare repeated samples at matched workload and service age.

At 09:55 EDT both live `adam` services were restarted to load the shared prompt and verified active with `tts.engine: silent`. Startup cgroup readings settled to about 0.99 GB on the laptop and 0.89 GB on Desky; these are immediate post-restart snapshots, not a peak or long-idle measurement.

### Iteration: compact ordinary-chat context and precise skill matching (2026-10-03)

The earlier tool-free gate omitted tool schemas but still sent the full desktop/tool system prompt, open-window state, local time, and an automatic memory lookup. A matched free Space Bunny run now uses a short conversational prompt and sends only the current request when the conservative gate says no external information, action, memory, image, desktop history, or specialized skill is needed. Memory lookup, desktop snapshots, local time, skills, and the full shared prompt remain on the normal path when any such context is needed. A drafting request without a save/send destination is now treated as composition; an explicit destination such as “write this into my notes” keeps tools enabled.

On Desky, unrelated queries were also matching user-created skills and injecting 1.6–4.1k characters of irrelevant procedures. Automatic skill matching now indexes skill identity, title, description, and the declared `When to Use` section rather than procedure bodies, with its minimum score raised from 0.5 to 4.0. In host-specific probes, ordinary pasta, entropy, drafting, food, and ice questions stopped loading those skills, while explicit CPU/GPU, workspace, and browser requests still matched their corresponding skills. Unit fixtures cover positive triggers and rejection of a term found only in a procedure body.

The same five short requests were run three times each per host with the free `stealth/space-bunny-alpha` model. Against the prior tool-filtered/full-context version, mean input tokens fell from 3,512 to 260 per request on the laptop (92.6%) and from 4,428 to 260 on Desky (94.1%). Mean output tokens were 84→79 on laptop and 98→116 on Desky. Brain-path medians were 2.16→2.28 s on laptop and 2.37→2.57 s on Desky; ranges overlapped widely, so there is no measured latency win in these samples. All 30 requests completed with no tool calls. The six compact-prompt thank-you drafts used no unsupported proper names and made no clipboard/save offers; two still assumed the user had been away. Pasta descriptions continued to vary, including occasional overgeneralization about smooth penne. The measured gain is a roughly 93–94% input-token reduction with mostly acceptable answer quality, not a proven speed increase.

The time/date intent gate was tightened during validation: “What time is it?” and “What date is it?” retain tools, while “What is time complexity?” remains ordinary chat. Personal questions that mention the user's own history or preferences also retain automatic memory retrieval; a regression test verifies that “What's my favorite food?” reaches memory and includes the retrieved fact in model context. One free-model Tokyo-time request per host called `get_current_time` and answered from its result (2.88 s laptop; 2.73 s Desky). At that iteration, the focused skill/routing suite passed (`45 passed` on each host) and the full suites passed with laptop `351 passed, 11 skipped` and Desky `358 passed, 4 skipped`. These remain silent-TTS production-`AdamBrain` integration runs, not active microphone-to-speaker service trials. Per-request provider cost was not available. Traces are `free-chat-repeat-compact-final.jsonl` and `free-time-tool-smoke-compact.jsonl` in the per-host overnight-analysis directories.

## Current Focus: Speech Recognition

Speech recognition is slow and unreliable on the laptop, while Desky provides a faster local-GPU reference. The code supports local Qwen3 ASR, Faster-Whisper, and cloud transcription with a lazy local Whisper fallback; setup also has a CPU-only Faster-Whisper path. Measure the route each host actually selects, including backend fallbacks, rather than assuming the configured model is the one doing the work.

1. [x] Record the current paths: laptop cloud STT with CPU `base.en` wake spotting and local fallback; Desky local `small.en` on CUDA 0 (`int8_float32`).
2. [ ] Expand the matched quality/latency comparison to human speech and difficult speech conditions. On three fresh eSpeak phrases, laptop `base.en` CPU int8 loaded in 0.46 s and ran at 0.24–0.27 s warm; Desky CPU `base.en` ran at 0.25–0.36 s. `small.en` CPU took 0.62 s on laptop and 0.80–0.84 s on Desky. Desky's configured `small.en` CUDA path loaded in 1.13 s and ran at 0.11–0.12 s warm. Both models got two easy phrases verbatim; on “3:45,” `base.en` heard “free for the 5” while `small.en` produced “3.45 to 8 PM.” The laptop's configured OpenRouter Whisper endpoint repeated the shift transcript identically across three calls at 0.52–2.60 s, reporting $0.0000124 per request; it rendered 3:45 as `3.45`. The local temporal parser now accepts this explicit dotted-time form. A new matched CPU/GPU probe using identical local clips measured laptop `base.en` at 0.25–0.27 s and Desky CUDA `small.en` at 0.12–0.20 s. Beam size 5 was 5–25% slower in these short runs and did not fix the laptop's spoken-time error; it partially corrected “rigatoni” on Desky. A fixed vocabulary prompt recovered “penne” on both hosts, but changed the laptop's “three forty-five” into the wrong `3.4 to 5.00 pm` and left Desky's time transcript unchanged. Do not enable larger beams or a static vocabulary prompt globally from this sample. eSpeak results do not establish human live-mic quality; numeric formatting also inflates word error rate.
3. [x] Share the laptop's already loaded wake `base.en` model with the cloud transcriber's local fallback when model/device/compute settings match, and guard both uses with the same lock. A fallback probe reduced incremental PSS from 455 MB to 57 MB on laptop and from 183 MB to 71 MB on Desky's CPU comparison, while fallback-call time fell from 0.46 s to 0.27–0.29 s. The live Desky config uses local CUDA `small.en`, so it does not load this separate CPU fallback. The laptop service now logs that its wake model is shared as the fallback.
4. [ ] Recheck with fresh human speech through the live service, including speaker verification and end-to-end acknowledgement latency; synthetic audio has not passed enrolled-speaker verification.

After the shared-fallback code change, full suites passed on laptop (`336 passed, 11 skipped`) and Desky (`343 passed, 4 skipped`). Both services are active with silent TTS; latest cgroup snapshots were about 0.93 GB on laptop and 0.89 GB on Desky. The model comparison ran offline on synthetic speech and did not send audio to a cloud provider.

## Current Focus: Desktop Navigation

Frontier vision models already navigate difficult desktop tasks reliably; that behavior does not need to be retested. The goal is to improve general desktop navigation with lower-cost models. The code has OCR text and clickable regions, OmniParser's detected screen regions, and separate OCR-oriented and screenshot-grounded paths; these are not yet one universally fused pipeline. Evaluate how the available evidence helps lower-cost models understand screens and choose actions across applications, rather than making Shenzhen I/O the scope of desktop control.

1. [ ] Reproduce difficult desktop navigation tasks with lower-cost models; as a concrete stress test, have Adam attempt to beat one or two Shenzhen I/O levels.
2. [ ] Identify whether failures come from OCR, visual-region grounding, model interpretation, action choice, or checking whether the task succeeded.
3. [ ] Improve how lower-cost models use the available OCR text, OmniParser regions, screenshots, and other supported screen evidence across applications.
4. [ ] Repeat representative tasks, including the one or two Shenzhen I/O levels, and check for broader improvement.

New isolated-launch evidence: Gamescope's headless backend starts successfully on Desky and creates an Xwayland display (`:2`) without using the physical display. Launching the game binary directly in that isolated display still fails `SteamAPI_Init()` because it cannot connect to Steam's global user. A temporary per-game launch option was tried through a separate Xvfb Steam session, but Steam displayed a sign-in prompt instead of reusing the stored session. No credentials were entered. Steam's queued app update (`359550`) had zero bytes downloaded and no download files when checked, so the client was stopped and restarted safely. The temporary launch option was removed, the Steam configuration was restored byte-for-byte from its pre-test backup, and the ordinary Steam client was restarted with `-silent`; the sign-in window is minimized. No game level was launched. Next, when an authenticated Steam session is available, set the per-game Gamescope launch option through Steam, verify isolated screenshot/input access from Adam, and attempt the requested one or two levels. Do not wake the physical display or enter credentials on the user's behalf.

Synthetic visual-evidence A/B (2026-10-03): on three dense 12-option settings screens, the free `stealth/space-bunny-alpha` route selected the exact requested label and a point inside its card in all three screenshot-only trials on both hosts. Adding OCR text and detected text-region centers stayed at 3/3 on both hosts; it did not improve accuracy. On Desky, CPU OCR took 4.7–5.2 seconds per screen. The model-call median was 3.63 s from image only versus 2.57 s with OCR on Desky, and 1.31 s versus 1.52 s on the laptop; treat the small timing sample as noisy. OCR increased prompt input from about 2,297 to 3,968 tokens. Including Desky's OCR time makes the aided path slower overall; laptop OCR is not installed in the test environment, so the laptop's comparison reused the exact same OCR output produced on Desky and does not include laptop extraction cost. For these clear, text-heavy screens, keep OCR optional and on demand. This does not establish the best aid for visually ambiguous controls, tiny text, complex layouts, or game navigation. A small natural-language variant had an ambiguous target (“microphone and speaker preferences” could plausibly mean “Audio Devices”), so it is excluded from the score. Direct Qwen3.7 Flash and DeepSeek V4.1 Flash probes on the dense screenshot did not return usable answers within the tested output budgets; the DeepSeek request spent 600 output tokens (550 reasoning tokens) and took 22.1 s. Treat those route probes as inconclusive, not as an accuracy comparison. Artifacts are under each host's `~/.local/state/adam/overnight-analysis/` directory.

An X11 dialog A/B found that a single-purpose desktop request was spending an extra model turn to ask for an initial screenshot. Adam now captures a fresh, action-capable screenshot before the first turn for these requests (not for OCR-only or mixed-domain requests) and includes its Snapshot ID with the image. On a private, marker-verified 1280×720 Xvfb/Openbox dialog, the free Space Bunny model closed the dialog and confirmed it absent in both runs on each host. Laptop baseline → initial-screenshot latency was 39.74 → 6.26 s and 8.71 → 5.56 s, with model calls falling from 6 → 3 and 5 → 3. Desky latency was 10.58 → 8.02 s and 9.76 → 5.20 s, with calls falling from 5 → 4 and 4 → 3. These paired samples are promising but small; the 39.74-second laptop baseline is a provider/task-loop outlier, so use the medians (24.22 → 5.91 s laptop; 10.17 → 6.61 s Desky) as provisional, not stable estimates. The initial local capture took roughly 20–170 ms and sent a 6.8 KB PNG; the path used no OCR, OmniParser, or local model. Focus changes after closing the dialog are reported as uncertain by the controller, but subsequent window enumeration confirmed the requested dialog was gone. Test logs are `desktop-prefetch-live-bench*.log` in each host's overnight-analysis directory. The benchmark explicitly verified its capture source against the isolated Xvfb size and root pixel before any model call.

A harder synthetic preferences task required enabling one checkbox, changing a radio choice, preserving a second checked option, applying, and verifying the written state. All four baseline/prefetch trials produced the exact expected final state on both hosts. Desky improved from 19.36 s / 5 model calls to 9.71 s / 4 calls; laptop used 5 calls in both variants and measured 12.19 s baseline versus 18.21 s with prefetch, so its timing did not improve in this single pair. Provider latency varied substantially; keep the initial screenshot for now because it consistently removes turns on the simple interaction and reduced calls on Desky's multi-step interaction, while continuing to check complex-task latency. A UI label containing “Start minimized” initially caused the intent gate to expose 67 tools; the gate now treats that phrase as a control label while “start Firefox” still retains the broad tool catalog. The corrected run used five desktop tools. Logs are `desktop-settings-bench-after-intent-fix.log` in each host's overnight-analysis directory. These synthetic settings tasks do not replace the Shenzhen I/O stress test.

The isolated Xvfb setup also exposed backend-detection false positives: an installed `i3-msg` binary could make any X11 display look like i3, installed Wayland tools could override an unknown session, and `get_active_backend(refresh_env=False)` still refreshed the user's live desktop environment. Backend and skill environment resolution now honor explicit session/socket evidence instead of treating an installed command as proof that its window manager is active. Regression tests cover generic X11, generic Wayland, and actual i3 session selection.

## Current Focus: Desktop Interfacing

Desktop interfacing is the general ability to observe and operate the user's desktop across supported environments. The code already has screenshot and window observation, browser and accessibility information where available, and Wayland/X11 input backends. A dispatched input is not proof that the requested outcome happened; Adam must verify the resulting state.

1. [ ] Map which observation and input paths are available on each supported desktop environment.
2. [ ] Identify where focus, input dispatch, window changes, or result verification fail during ordinary desktop tasks.
3. [ ] Improve reliable, state-grounded interaction while retaining fresh-snapshot checks and appropriate confirmation for consequential actions.
4. [ ] Verify outcomes across representative applications and supported desktop environments.

## Current Focus: Laptop Power Use

Power use that is acceptable on a desktop can significantly reduce laptop battery life. The repository already has `tools/measure_power.py`, which records host-exposed telemetry such as battery, GPU, and Intel RAPL readings when available; it does not measure total system AC power. Record what each machine can actually measure and do not attribute system-wide drain to Adam without a baseline.

With both silent `adam` services active and otherwise idle, a 12-second sampler run on the laptop reported battery status `Full` and `power_now_raw=0`, no NVIDIA utility, and Intel RAPL counters blocked by permissions. This is not a usable battery-discharge or whole-system power baseline. On Desky, the 12 samples averaged 8.76 W for the GTX 1080 Ti and 17.68 W for the RTX 3070; battery telemetry was unavailable and RAPL was partial. GPU board readings include the host's other activity and do not measure total system power. Traces are in `/tmp/adam-power-laptop.json` and `/tmp/adam-power-desky.json` on their respective machines. For a meaningful laptop comparison, repeat while discharging or use an external meter; compare the same idle/listen/transcribe/task workloads.

An isolated native-thread A/B found that importing Adam on Desky created 95 OS threads by default, with 32-thread OpenBLAS/OpenCV pools. Setting native pool limits to four, Torch intra-op/inter-op to 4/1, and OpenCV to four reduced the isolated import process to 11 threads; import time and peak RSS stayed effectively unchanged (1.39 s / 1,190 MB default versus 1.36 s / 1,191 MB capped). The configured local ASR routes were checked with the same synthetic audio: capped laptop CPU `base.en` still transcribed the ordinary command exactly at about 0.22 s, while Desky CUDA `small.en` stayed about 0.12 s. The known laptop time-phrase miss remained; the cap did not introduce it. A separate CTranslate2 `cpu_threads=0` versus `4` probe did not change ASR transcripts, warm latency, memory, or thread count on either host, so the model's built-in setting stays unchanged. A shared default cap of four now runs before and after application imports, with `ADAM_CPU_THREADS` as an override and lower explicit library limits preserved. After restarting both silent services, the live thread snapshot fell from 42 to 34 on the laptop and 149 to 63 on Desky. Current cgroup snapshots are about 0.59 GB and 1.00 GB respectively, but memory readings vary and no battery/RAPL power meter is currently usable, so neither the memory nor power difference is a controlled result. Full suites pass on the final source: laptop `370 passed, 11 skipped`; Desky `377 passed, 4 skipped`. Keep this cap provisionally and compare long-running voice, visual, and CPU-heavy tasks under an actual power measurement when available.

1. [ ] Compare a baseline with Adam running under repeatable conditions such as idle, listening, transcription, and active tasks.
2. [ ] Identify which components and background activity increase measured power on the laptop.
3. [ ] Reduce unnecessary laptop power use while keeping Adam responsive.
4. [ ] Measure again using the same workload and source, then check battery impact and responsiveness.

## Current Focus: Memory Use

Memory use is a compatibility requirement, not just a desktop optimization. The laptop has a hard 16 GB RAM limit, and Adam should also be practical on older, modest desktops such as an OptiPlex. The core assistant should remain useful without a local LLM and without keeping every optional model resident at once. ASR, wakeword detection, speaker recognition, diarization, TTS, and desktop vision can otherwise add up to an excessive footprint. Configured local ASR models load when the transcriber is created; cloud STT defers its local Whisper fallback until needed. OCR and OmniParser are preloaded when enabled, while speaker and diarization models load on first use.

1. [ ] Measure total and per-component memory use at startup, first model use, and peak concurrent use, leaving room for the OS and other applications within the 16 GB laptop limit.
2. [ ] Identify the largest resident models, duplicated runtimes, worker pools, queues, and audio buffers; distinguish eagerly loaded components from optional first-use models.
3. [ ] Keep the core voice path lean: load optional models only when needed, release them when practical, bound concurrency and buffers, and provide a low-memory configuration with graceful feature tradeoffs.
4. [ ] Verify the core assistant on the 16 GB laptop and modest desktop hardware, including operation without a local LLM; check that memory pressure does not cause swapping, out-of-memory failures, or unusable responsiveness.

## Current Focus: Fast Memory Recall

Adam should quickly recall things the user told it earlier, including dated events. The JSON store now keeps optional event type, normalized date/range, start/end time, timezone, precision, and original time phrase alongside each memory. Creation time is not event time: existing records without event metadata stay undated, and ordinary utterances are not silently persisted as memories. Explicit memory commands and the LLM memory tool remain the save paths.

1. [x] Keep memory creation tied to explicit memory saves and the existing memory tool; ordinary utterances are not silently persisted as events.
2. [x] Store distinct event metadata alongside the original statement; never use record creation time as a substitute for an unknown event date.
3. [x] Resolve supported relative dates against the record's capture time. Year-level phrases such as “last year” and “21 years ago” remain year ranges instead of fabricated exact days.
4. [x] Save event type, date range, local start/end time when understood, timezone, precision, and original time phrase. Date-bearing events from different dates are kept separate during deduplication.
5. [x] Add an in-process date index to the existing JSON store. A 10,001-record synthetic check returned this week's matching work events in 0.023 ms median (0.029 ms p95), versus 26.458 ms median (26.886 ms p95) on unbounded BM25. This index rebuilds at load; measure persistence and concurrent updates before deciding whether a SQLite migration is warranted.
6. [x] Filter date-bounded work queries by local calendar window and event type before scoring text; retain lexical/semantic retrieval for unstructured queries.
7. [ ] Extend date and time parsing for DST transitions, overnight ranges, varied phrasing, and timezone changes. The current parser deliberately supports only explicit patterns and leaves old undated records unknown.
8. [x] Test “Last year, I stole a car” and “I worked today from 3:45 to 8:00pm,” including reload, week-bounded recall, similar dated events, malformed date metadata, and the ASR-produced “3.45 to 8 p.m.” form. Latest focused memory tests passed on both hosts (`15 passed` each); the full suite after the dotted-time parser change passed on laptop (`336 passed, 11 skipped`) and Desky (`343 passed, 4 skipped`).

Successful reads no longer rewrite the entire JSON store to persist access counters; counters stay in memory until the next explicit mutation. A matched synthetic fact-retrieval run on both hosts used eight individually labeled work facts and unrelated distractors at 100, 1,000, and 10,001 records. Before the change, specific-query precision@3 fell to 33% at 1,000 and 10,001 records, although each expected fact remained rank one. Requiring at least 0.4 query-token coverage for the BM25-threshold acceptance path raised precision@3 to 100% at both larger sizes, with no misses and rank one unchanged. Week-bounded retrieval still returned all 8/8 dated work facts; its context was 1,025 characters (128 whitespace-delimited words, a token-count proxy). At 10,001 records, date-bounded lookup measured 0.050 ms median / 0.054 ms p95 on the laptop and 0.075 ms / 0.078 ms on Desky; unbounded BM25 measured 37.56 ms and 58.56 ms median respectively. Loading the synthetic 10,001-record store took about 116 ms on the laptop and 145 ms on Desky. Keep the coverage gate. Focused memory tests passed (`16 passed`); the full suites passed with laptop `358 passed, 11 skipped` and Desky `365 passed, 4 skipped`. Both silent Adam services were restarted with the change. These are synthetic retrieval metrics, not model-based personal-memory recall; they do not measure disk reload under a real store, concurrent writes, or live-bot recall quality. Next, evaluate which user statements should become saved events and verify week/month questions through the live bot without storing private test data.

## Current Focus: Latency

Latency is central to how capable Adam feels. Users should not wait around 10 seconds just to learn that Adam heard them. Long tasks can also frustrate users when there is no sign of progress and they cannot tell whether Adam is still working. Adam is expected to feel highly agentic and capable, close to nigh omnipotent, so it should make steady progress on multi-step requests instead of adding avoidable pauses between steps.

1. [ ] Measure end-of-utterance to first acknowledgment and first response/action. Existing telemetry records capture, STT, brain-turn, tool, TTS, and playback stages; use the correlated timing data and add only missing measurements.
2. [ ] Measure end-to-end completion time for typical and multi-step tasks and identify delays across capture, recognition, reasoning, tool use, and response playback.
3. [ ] Keep users informed during long tasks. A 10-second wait heartbeat now complements progress updates after every third tool/model loop; verify audible timing, interruption behavior, and non-overlap on a long live task.
4. [x] Batch independent safe reads into one model turn and execute eligible native reads concurrently, preserving result order. Measure real search groups and keep dependent reads, mixed batches, custom tools, and desktop actions sequential.
5. [ ] Skip irrelevant tool schemas for clearly ordinary conversation only when a conservative intent gate confirms no action, recall, system, file, desktop, or live-information need; broaden only after quality tests across general topics.
6. [ ] Use existing bounded desktop action sequences where they safely reduce round trips, and verify task outcomes instead of assuming a dispatched action succeeded.
7. [ ] Recheck acknowledgment time, completion time, progress visibility, and task quality after latency improvements to ensure Adam feels responsive and capable without sacrificing reliability.

Next measurements: a human-spoken end-to-end sample on each host when an enrolled speaker is available; repeat the complex lower-cost-model desktop task across more applications and screens; and attempt Shenzhen I/O once an authenticated Steam session is available for the isolated launch.

### Iteration note (2026-10-03)

The conservative ordinary-chat route reduced Qwen3.7 Flash input for a thank-you draft from 12,195 tokens to 131 after fixing a false memory-context match. A five-prompt production-`AdamBrain` comparison on each host tested reasoning suppression with the same compact prompt and silent TTS. The prior median LLM-call times were 6.25 s on the laptop and 7.24 s on Desky with its existing `think: true` setting. With `llm.disable_reasoning_for_tool_free`, they were 1.29 s and 1.19 s. Reported reasoning tokens fell from 2,024 to zero on laptop and from 3,424 to zero on Desky. Provider-reported cost fell from 0.000679 to 0.000048 credits on laptop and from 0.000493 to 0.000049 credits on Desky. These are provider credits, not converted currency. Answers remained concise and on topic in these samples.

I also tested `llm.tool_free_reasoning_effort: low` with the free [Space Bunny Alpha](https://openrouter.ai/stealth/space-bunny-alpha) route and the laptop's configured [Qwen3.8 27B](https://openrouter.ai/qwen/qwen3.8-27b) / Dekallm route. On five Space Bunny prompts per host, silent-TTS `AdamBrain` call medians were 1.83 s on laptop and 1.40 s on Desky, with zero reasoning tokens and zero provider credits. On five Qwen3.8 prompts, low effort reduced output tokens from a mean 125 to 90, reported reasoning tokens from 460 to 219, and provider credits from 0.001905 to 0.001393; the median LLM-call time was 1.15 s versus 1.11 s at default, so this sample shows a cost reduction but no latency gain. Both hosts now set effort to low for tool-free turns; tool-enabled calls still use their normal reasoning setting. Space Bunny requires reasoning and supports low effort, so the disable-reasoning option remains off in both host configs.

`llm.disable_reasoning_for_tool_free` defaults off and `llm.tool_free_reasoning_effort` defaults to null; configuring both is rejected. Either option applies only to OpenRouter requests with no tools; tool-enabled agent turns retain the configured reasoning. [OpenRouter's provider-routing guide](https://openrouter.ai/docs/guides/routing/provider-selection) documents that parameter support varies by endpoint and that routing can ignore unsupported parameters unless `provider.require_parameters` is enabled. These are model-call results, not end-to-end voice latency: the probes used production `AdamBrain` with silent TTS and did not measure speech capture, audible acknowledgment, or playback. After the final code and config changes, full suites passed on laptop (`357 passed, 11 skipped`) and Desky (`364 passed, 4 skipped`). Both Adam services were restarted and verified active, online, and silent; immediate cgroup usage was about 1.05 GB on the laptop and 0.89 GB on Desky (startup snapshots, not peaks). Desky's read-only Codex review agreed with the need for true speech E2E, scored memory retrieval, and a deterministic OCR/region-evidence UI comparison; these are now the next measurements above. Test artifacts are in each host's `~/.local/state/adam/overnight-analysis/2026-10-03-*/` directory.

A further synthetic-memory live-bot test exposed that Desky automatically matched the `open_up_work_email` skill to “What times did I work this week?” because both shared only the generic word “work.” Automatic skill selection now requires at least two distinct query terms to overlap the skill's title, description, or declared trigger for multiword requests; one-token skill discovery remains available. The unrelated skill no longer blocks the memory-only route, while “open my work email” still selects it. The free Space Bunny live `AdamBrain` path then answered the same week-recall query three times on each host using eight temporary synthetic shift records, with no tool calls, 829 input tokens per request, and all eight dates/times represented in every answer. Laptop completion times were 2.39, 3.44, and 1.92 seconds; Desky times were 1.96, 1.82, and 2.43 seconds. Silent TTS was used, so these are model-path timings, not speech end-to-end results. The tests used temporary stores and did not read or change personal memories. The fixed skill-match and routing tests passed on both hosts (`50 passed` each); that iteration's full suites passed on laptop (`360 passed, 11 skipped`) and Desky (`367 passed, 4 skipped`). Logs are `memory-live-route-after-skill-gate.log` in each host's dated overnight-analysis directory.

### Current verification checkpoint (2026-10-03)

After the subsequent desktop screenshot-prefetch, navigation-intent, backend-detection, and native-thread-cap changes, the latest complete suites pass on the laptop (`370 passed, 11 skipped`) and Desky (`377 passed, 4 skipped`). `git diff --check` passes. Both running `adam` services use silent TTS and the two hosts have matching hashes for the latest shared agent, memory, desktop, test, and plan files. Current service thread counts are 34 on the laptop and 63 on Desky; current cgroup snapshots are about 0.59 GB and 1.00 GB respectively. These snapshots do not establish peak memory under concurrent speech/vision load. Current Desky GPU board readings at idle were 8.31 W and 17.57 W with 0% utilization; these include other host activity and are not Adam-attributable. The laptop is on AC/full battery (`power_now=0`); its exposed RAPL sysfs tree has no readable `energy_uj` counters, so this check cannot provide a laptop power comparison. Keep the speech end-to-end, peak-concurrency memory, repeatable laptop power, and Shenzhen I/O evaluations open. The Shenzhen attempt still requires an authenticated isolated Steam session; do not wake the physical display or enter credentials.

Adam is a 100% non-blocking, interruptible, hybrid voice terminal agent engineered for Linux (Arch/PipeWire/Dual NVIDIA GPUs). It combines autonomous terminal execution, background job supervision, acoustic echo cancellation, and low-latency audio feedback.

## Hardware Baseline: Desktop (`desky`)

- OS: CachyOS rolling release.
- CPU: AMD Ryzen 9 5900XT, 16 cores / 32 threads.
- RAM: 31 GiB, with 18 GiB available at the time checked; 31 GiB swap.
- GPUs: NVIDIA GeForce GTX 1080 Ti (11 GiB) and RTX 3070 (8 GiB), driver 580.178.04.
- Storage: 512 GB Intel NVMe SSD and 2 TB WD_BLACK SN770 NVMe SSD.

## Codex CLI Notes: Desktop (`desky`)

- These are host-specific operational notes, not an Adam codebase integration. Codex CLI was installed at `/usr/bin/codex` on `desky`; version 0.158.0 and a managed app-server daemon were observed on 2026-10-02.
- `codex exec` runs a non-interactive agent prompt. `--ephemeral` avoids persisting the session.
- `--sandbox read-only` runs with read-only access; `--sandbox workspace-write` allows workspace writes.
- `--dangerously-bypass-approvals-and-sandbox` is accepted and runs with `danger-full-access`, bypassing approvals and sandbox restrictions. This was verified with an ephemeral prompt.
- `--approve-for-me` routes approval requests through automatic review with the workspace-write sandbox.
- A managed Codex app-server daemon was also running on `desky` during inspection.

---

## Historical Architecture Proposal (Reference Only)

The detailed machine-specific architecture, code examples, and phased roadmap below are retained as historical context. They describe an earlier Arch/PipeWire/dual-NVIDIA-oriented proposal and may not match the current repository or the general-purpose goal above. Do not treat them as current implementation facts or active requirements without checking the code and configuration first.

## 1. System Specifications & Core Directives

1. **Non-Blocking & Full Barge-In (Interruptibility)**:
   - Full-duplex audio stream via PipeWire WebRTC AEC (`Adam_Clean_Mic`).
   - Microphone pipeline free-runs across **all** states, including `ASSISTANT_SPEAKING`.
   - Atomic **Generation Epoch Barriers** eliminate race conditions during speech barge-in.
   - PortAudio lifecycle recovery prevents thread crashes after stream aborts.
2. **Background Job Supervision & Proactive Wake-Up**:
   - Long-running commands (transcoding, builds, downloads) run inside isolated **Bubblewrap (`bwrap`)** containers with private PID namespaces (`--unshare-pid`) and merged-/usr relative symlinks.
   - GPU device nodes (`/dev/dri`, `/dev/nvidia*`) and POSIX shared memory (`/dev/shm`) are explicitly mounted.
   - Output logs stream to persistent NVMe disk storage (`~/.local/state/adam/jobs/`), completely protecting `/tmp` `tmpfs` RAM-disk from exhaustion.
   - Background tasks report to a **Priority Audio Arbiter** that queues alerts and speaks **only** during `IDLE_LISTENING`, preventing context poisoning during user speech or confirmations.
3. **Flip-of-a-Switch Local vs. Cloud LLM**:
   - A single configuration switch (`provider: "local" | "cloud"`) toggles between local Ollama (`qwen2.5-coder:7b` / `gemma4:e4b`) and cloud providers (Groq, Gemini, Anthropic).
   - Unified canonical Pydantic tool schemas with full tripartite dialect compilation (OpenAI `tool_calls`, Gemini `functionDeclarations`, and Anthropic `input_schema` / `tool_use` / `tool_result`).
4. **Instant Audible Earcons & Rapid Feedback**:
   - Sub-300ms total perceived turnaround.
   - Dedicated non-blocking audio engine feeding pre-synthesized PCM earcons (<0.05ms thread handoff, zero asyncio event loop stalls).
   - Clause-level streaming Piper TTS synthesizes and plays speech in chunks as tokens arrive.

---

## 2. Host Hardware Topology & Resource Allocation

### Hardware Inventory
* **GPU 0 (GTX 1080 Ti, GP102, CC 6.1, 11 GB VRAM)**:
  - *Compute Reality:* Pascal architecture lacks Tensor Cores. FP16 runs at 1/64 FP32 speed. CUDA 13.x drops CC 6.1 support.
  - *Execution Policy:* `faster-whisper` runs using PyPI CUDA 12 wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12==9.*`) with `compute_type="int8_float32"`.
  - *VRAM Guardrails:* Ollama context capped at 4k tokens (`num_ctx 4096`) to prevent KV-cache expansion from causing CUDA OOM alongside Whisper.
* **GPU 1 (RTX 3070, GA104, CC 8.6, 8 GB VRAM)**:
  - *Workload:* Dedicated to desktop display (Hyprland ~191 MiB, Noctalia ~148 MiB), with **~7.2 GB of VRAM completely free**.
  - *Encoding Reality:*
    - Ampere lacks hardware AV1 encoding.
    - Arch rolling `ffmpeg 9.0.2` requires NVENC API 13.1 (NVIDIA driver $\ge 610.00$), while installed driver is 580.178.04 (API 13.0). Direct `hevc_nvenc` calls currently fail with error 218 until the driver is updated.
    - *Policy:* Multi-threaded CPU **`libsvtav1`** runs at ~4x realtime speed and serves as the primary transcode engine inside the sandbox, with `hevc_nvenc` auto-enabled once the driver is updated to 610+.
* **Audio Subsystem**:
  - *Playback Sink:* System default or configured audio sink (`@DEFAULT_AUDIO_SINK@`).
  - *Capture Source:* System default or configured microphone (`@DEFAULT_AUDIO_SOURCE@`).
  - *Virtual Routing:* Both `PIPEWIRE_NODE` and `PULSE_SINK` environment variables are set to `Adam_Playback_Sink`, and PortAudio binds to the `pulse` ALSA bridge device index.
  - *Physical Volume Knob Policy:* Set analog potentiometer to a fixed unity gain position (~50% line-out). All listening volume adjustments are handled digitally (via `wpctl set-volume @DEFAULT_AUDIO_SINK@ ...`) so digital reference attenuation matches acoustic playback exactly.

---

## 3. Red-Team Vulnerability Analysis & Hardened Countermeasures

### 3.1. Audio & AEC Failures: The PortAudio Routing Trap & Thread Death
* **The Vulnerability**:
  1. Virtual PipeWire sinks do NOT appear in PortAudio's ALSA device enumeration. Querying `Adam_Playback_Sink` returns `None`, causing playback to default to physical ALSA output, completely bypassing WebRTC AEC and re-triggering the self-interruption loop.
  2. Calling `stream.abort()` during barge-in sets the PortAudio stream to `stopped`. The next `stream.write()` raises `PortAudioError: Stream is stopped [-9983]`, craadamg the earcon worker thread permanently.
  3. PipeWire 1.6.9 ignores legacy PulseAudio keys; WebRTC parameters require the `webrtc.` prefix.
* **The Fix**:
  - Target PortAudio's `pulse` device index directly and inject both `PIPEWIRE_NODE="Adam_Playback_Sink"` and `PULSE_SINK="Adam_Playback_Sink"`.
  - In the earcon worker thread, protect stream operations with `_stream_lock` and call `if stream.stopped: stream.start()` before `stream.write()`.
  - Configure PipeWire AEC with verified `webrtc.*` keys.

---

### 3.2. Microsecond Race Conditions & Generation Epoch Barriers
* **The Vulnerability**:
  If the user says *"Stop!"* in the exact microsecond between the LLM finiadamg its generation and the audio worker thread popping the first TTS PCM chunk:
  - The state machine cancels the LLM task (which already completed).
  - The decoupled audio worker thread plays the old sentence anyway while the user is trying to speak.
* **The Fix**:
  - **Atomic Generation Epochs**: Every interruption or state reset atomically increments `current_epoch`.
  - All audio frames in queues and pipelines are tagged with their creation `epoch_id`.
  - If `frame.epoch_id != current_epoch`, the audio worker instantly drops the frame and aborts playback.

---

### 3.3. Bubblewrap Arch Merged-/usr & Process Isolation
* **The Vulnerability**:
  1. Binding `/bin`, `/lib`, `/lib64` as directories on Arch breaks merged-/usr invariants and omits `/sbin`.
  2. Omitting `--unshare-pid` leaves `/proc` visible to sandboxed tasks and fails to clean up grandchild processes on parent exit.
  3. Missing `--tmpfs /dev/shm` breaks POSIX shared memory required by CUDA and multiprocessing.
* **The Fix**:
  - Use canonical Arch symlink arguments: `--symlink usr/lib /lib`, `--symlink usr/lib /lib64`, `--symlink usr/bin /bin`, `--symlink usr/bin /sbin`.
  - Add `--unshare-pid` so `bwrap` acts as PID 1 and reaps all child processes atomically upon exit.
  - Mount `--tmpfs /dev/shm`, bind `~/Downloads`, and use `--dev-bind-try` for all dynamic `/dev/nvidia*` nodes.

---

### 3.4. Priority Audio Arbiter & State Gating
* **The Vulnerability**:
  Background task completion alerts blurt out over active user speech or during `AWAITING_CONFIRMATION`, leading to context corruption and unintended affirmative execution.
* **The Fix**:
  - Implement a **Gated Notification Queue** inside `PriorityAudioArbiter`.
  - When the system is in `USER_SPEAKING`, `PROCESSING_REACT`, `ASSISTANT_SPEAKING`, or `AWAITING_CONFIRMATION`, notifications are held in a min-heap priority queue.
  - Only when the state returns to `IDLE_LISTENING` are alerts popped, chimed, and spoken.

---

### 3.5. Confirmation Safety: Word Boundaries & Active Expiry Watchdog
* **The Vulnerability**:
  1. Checking `"no" in text` matches `"I don't know"`, `"now hold on"`, or `"annoyed"`, triggering unintended denials or safety aborts.
  2. Checking `time.time() > deadline` only when a transcript arrives fails if the user says nothing at all. The state machine locks in `AWAITING_CONFIRMATION` forever.
* **The Fix**:
  - **Regex Word Boundaries**: Enforce `re.search(r'\b(no|cancel|stop|abort)\b', text, re.I)` and `re.search(r'\b(yes|yeah|confirm|proceed|go ahead)\b', text, re.I)`.
  - **Active Asyncio Watchdog Timer**: `request_confirmation()` spawns an explicit `asyncio.create_task(self._expiry_watchdog(10.0))` that fires after 10 seconds of silence, cancelling the action and returning state to `IDLE_LISTENING`.

---

## 4. Hardened Production Code Components

### A. Non-Blocking Audio Engine with Epoch Barriers (`src/audio/earcon.py`)

```python
import os
import queue
import threading
import sounddevice as sd
import numpy as np

def setup_audio_routing(target_sink="Adam_Playback_Sink"):
    """Injects routing variables for both ALSA PipeWire-plugin and PulseAudio layers."""
    os.environ["PIPEWIRE_NODE"] = target_sink
    os.environ["PULSE_SINK"] = target_sink

def resolve_pulse_device_index() -> int | None:
    """Finds the integer index for the PortAudio 'pulse' ALSA bridge."""
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev["name"] == "pulse" and dev["max_output_channels"] > 0:
                return idx
    except Exception:
        pass
    return None

class RobustEarconEngine:
    """Hardened non-blocking earcon engine with PortAudio recovery and epoch barriers."""
    def __init__(self, target_sink="Adam_Playback_Sink", sample_rate=48000):
        setup_audio_routing(target_sink)
        self.sample_rate = sample_rate
        self.pulse_idx = resolve_pulse_device_index()
        self.queue = queue.Queue(maxsize=16)
        self.current_epoch = 0
        self.running = True
        self.stream = None
        self._epoch_lock = threading.Lock()
        self._stream_lock = threading.Lock()

        self.chimes = {
            "wake": self._generate_tone(880, 0.05),        # 50ms A5 beep
            "captured": self._generate_tone(1760, 0.03),    # 30ms A6 blip (got command)
            "done": self._generate_tone(587.33, 0.08),     # 80ms D5 soft chime
            "interrupt": self._generate_tone(330, 0.03),   # 30ms E4 low click
        }

        self.worker = threading.Thread(target=self._stream_loop, daemon=True)
        self.worker.start()

    def advance_epoch(self):
        """Invalidates pending audio and aborts current stream with mutual exclusion."""
        with self._epoch_lock:
            self.current_epoch += 1
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

        with self._stream_lock:
            if self.stream and self.stream.active:
                try:
                    self.stream.abort()
                except Exception:
                    pass

    def play(self, tone_name: str):
        with self._epoch_lock:
            epoch = self.current_epoch
        try:
            self.queue.put_nowait((tone_name, epoch))
        except queue.Full:
            pass

    def _render_tone(self, tone_name: str) -> np.ndarray:
        return self.chimes.get(tone_name, self.chimes["wake"])

    def _generate_tone(self, freq: float, duration: float) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = 0.15 * np.sin(2 * np.pi * freq * t)
        fade = int(self.sample_rate * 0.005)
        tone[-fade:] *= np.linspace(1, 0, fade)
        return tone.astype(np.float32)

    def _stream_loop(self):
        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.pulse_idx
        ) as stream:
            self.stream = stream
            while self.running:
                try:
                    tone_name, epoch = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                with self._epoch_lock:
                    if epoch != self.current_epoch:
                        continue  # Dropped due to epoch barrier

                pcm_data = self._render_tone(tone_name)

                with self._stream_lock:
                    if stream.stopped:
                        try:
                            stream.start()
                        except Exception:
                            continue
                    try:
                        stream.write(pcm_data)
                    except sd.PortAudioError:
                        pass
```

---

### B. Sandboxed Background Job Supervisor (`src/execution/supervisor.py`)

```python
import asyncio
import os
import signal
import glob
from pathlib import Path

class HardenedJobSupervisor:
    """Spawns jobs in Bubblewrap sandboxes with full GPU access, path bindings, and bwrap parent death-signals."""
    def __init__(self, log_dir="~/.local/state/adam/jobs", workspace="~/workspace"):
        self.log_dir = Path(log_dir).expanduser()
        self.workspace = Path(workspace).expanduser()
        self.downloads = Path("~/Downloads").expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.active_jobs: dict[str, dict] = {}

    def _get_gpu_device_args(self) -> list[str]:
        args = []
        if os.path.exists("/dev/dri"):
            args.extend(["--dev-bind-try", "/dev/dri", "/dev/dri"])
        for dev_path in glob.glob("/dev/nvidia*"):
            args.extend(["--dev-bind-try", dev_path, dev_path])
        return args

    async def start_sandboxed_job(self, job_id: str, raw_cmd: list[str], arbiter) -> int:
        log_file = self.log_dir / f"{job_id}.log"
        out_fp = await asyncio.to_thread(open, log_file, "wb")

        # Canonical Arch Linux merged-/usr layout and isolated PID namespace
        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib", "/lib64",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/bin", "/sbin",
            "--ro-bind", "/etc", "/etc",
            "--ro-bind", "/sys", "/sys",                            # Needed for NVML / GPU topology
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/dev/shm",                                  # CUDA/POSIX shared memory
            "--tmpfs", "/tmp",
            "--unshare-pid",                                        # bwrap acts as PID 1; auto-reaps all descendants
            "--ro-bind", str(self.downloads), str(self.downloads),  # Access downloads
            "--bind", str(self.workspace), str(self.workspace),      # Output directory
            "--chdir", str(self.workspace),
            "--die-with-parent",                                    # Kills sandbox if Adam exits
        ] + self._get_gpu_device_args() + ["--"] + raw_cmd

        proc = await asyncio.create_subprocess_exec(
            *bwrap_cmd,
            stdout=out_fp,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True
        )

        self.active_jobs[job_id] = {
            "proc": proc,
            "pgid": proc.pid,
            "out_fp": out_fp,
            "log_file": log_file
        }

        asyncio.create_task(self._watch_job(job_id, proc, out_fp, arbiter))
        return proc.pid

    async def kill_job(self, job_id: str):
        """Kills the process group cleanly."""
        job = self.active_jobs.get(job_id)
        if not job:
            return
        pgid = job["pgid"]
        try:
            os.killpg(pgid, signal.SIGTERM)
            await asyncio.sleep(1.5)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _watch_job(self, job_id: str, proc: asyncio.subprocess.Process, out_fp, arbiter):
        returncode = await proc.wait()
        await asyncio.to_thread(out_fp.close)
        self.active_jobs.pop(job_id, None)

        status = "succeeded" if returncode == 0 else f"failed with code {returncode}"
        await arbiter.enqueue_notification(
            priority=1,
            message=f"Job {job_id} {status}."
        )
```

---

### C. Priority Audio Arbiter with Gated Notifications (`src/arbiter/arbiter.py`)

```python
import asyncio
import heapq
from enum import Enum

class SystemState(Enum):
    IDLE_LISTENING = "IDLE_LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING_REACT = "PROCESSING_REACT"
    ASSISTANT_SPEAKING = "ASSISTANT_SPEAKING"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"

class PriorityAudioArbiter:
    """Manages audio access and notifications without conversational collision."""
    def __init__(self, tts_engine, earcon_engine):
        self.tts = tts_engine
        self.earcon = earcon_engine
        self.current_state = SystemState.IDLE_LISTENING
        self.notification_queue: list[tuple[int, str]] = []  # Min-heap (priority, message)
        self._state_lock = asyncio.Lock()

    async def set_state(self, new_state: str):
        async with self._state_lock:
            self.current_state = SystemState(new_state)
            if self.current_state == SystemState.IDLE_LISTENING:
                await self._drain_notifications_locked()

    async def enqueue_notification(self, priority: int, message: str):
        async with self._state_lock:
            heapq.heappush(self.notification_queue, (priority, message))
            if self.current_state == SystemState.IDLE_LISTENING:
                await self._drain_notifications_locked()

    async def _drain_notifications_locked(self):
        """Delivers pending notifications only during IDLE_LISTENING."""
        while self.notification_queue and self.current_state == SystemState.IDLE_LISTENING:
            priority, message = heapq.heappop(self.notification_queue)
            self.current_state = SystemState.ASSISTANT_SPEAKING
            self.earcon.play("done")
            await asyncio.sleep(0.1)
            await self.tts.speak_async(message)
            self.current_state = SystemState.IDLE_LISTENING
```

---

### D. Tri-State Confirmation Machine with Active Watchdog (`src/arbiter/confirmation.py`)

```python
import asyncio
import re

class TriStateConfirmationManager:
    """Handles confirmations with active asyncio watchdog timer and word-boundary safety checks."""
    AFFIRM_REGEX = re.compile(r"\b(yes|yeah|confirm|proceed|go ahead|yep|sure|do it)\b", re.IGNORECASE)
    DENY_REGEX = re.compile(r"\b(no|cancel|stop|don't|dont|abort|nevermind)\b", re.IGNORECASE)
    CLARIFY_REGEX = re.compile(r"\b(what|why|repeat|which|wait|explain|how)\b", re.IGNORECASE)

    def __init__(self, tts_engine, arbiter, timeout_seconds=10.0):
        self.tts = tts_engine
        self.arbiter = arbiter
        self.timeout_seconds = timeout_seconds
        self.pending_action = None
        self.watchdog_task = None

    def request_confirmation(self, action_payload: dict, prompt_text: str):
        self.pending_action = action_payload
        asyncio.create_task(self.arbiter.set_state("AWAITING_CONFIRMATION"))
        self.tts.speak_now(prompt_text)

        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
        self.watchdog_task = asyncio.create_task(self._expiry_watchdog(self.timeout_seconds))

    async def _expiry_watchdog(self, duration: float):
        """Active timer that breaks state deadlocks if the user says nothing."""
        try:
            await asyncio.sleep(duration)
            if self.pending_action:
                self._cancel_confirmation("Confirmation timed out after 10 seconds of silence.")
        except asyncio.CancelledError:
            pass

    async def evaluate_response(self, user_transcript: str) -> str:
        """Classifies response with strict word boundaries."""
        if not self.pending_action:
            return "EXPIRED"

        text = user_transcript.strip()

        # 1. Affirmative triggers
        if self.AFFIRM_REGEX.search(text):
            if self.watchdog_task:
                self.watchdog_task.cancel()
            action = self.pending_action
            self.pending_action = None
            asyncio.create_task(self.arbiter.set_state("PROCESSING_REACT"))
            return "AFFIRM"

        # 2. Negative triggers
        elif self.DENY_REGEX.search(text):
            if self.watchdog_task:
                self.watchdog_task.cancel()
            self._cancel_confirmation("Action cancelled.")
            return "DENY"

        # 3. Clarification triggers
        elif self.CLARIFY_REGEX.search(text):
            if self.watchdog_task:
                self.watchdog_task.cancel()
            self.watchdog_task = asyncio.create_task(self._expiry_watchdog(self.timeout_seconds))

            explanation = f"I am waiting to execute: {self.pending_action.get('summary', 'this command')}. Should I proceed?"
            self.tts.speak_now(explanation)
            return "CLARIFY"

        # 4. Unrecognized phrase
        else:
            self.tts.speak_now("Please answer yes or no.")
            return "CLARIFY"

    def _cancel_confirmation(self, message: str):
        self.pending_action = None
        self.tts.speak_now(message)
        asyncio.create_task(self.arbiter.set_state("IDLE_LISTENING"))
```

---

## 5. PipeWire 1.6.9 AEC Configuration

Deploy to `~/.config/pipewire/pipewire.conf.d/50-adam-aec.conf`:

```spa
context.modules = [
  {   name = libpipewire-module-echo-cancel
      args = {
          library.name = "aec/libspa-aec-webrtc"
          aec.args = {
              webrtc.high_pass_filter = true
              webrtc.noise_suppression = true
              webrtc.gain_control = false
              webrtc.voice_detection = true
              webrtc.transient_suppression = true
          }
          node.latency = 256/48000
          resample.quality = 4
          capture.props = {
              node.name = "Adam_AEC_Capture"
              node.description = "Adam Physical Mic In"
              target.object = "alsa_input.pci-0000_2b_00.3.analog-stereo"
              stream.dont-remix = true
              node.passive = true
          }
          source.props = {
              node.name = "Adam_Clean_Mic"
              node.description = "Adam AEC Filtered Output"
              media.class = "Audio/Source"
              audio.rate = 48000
              audio.channels = 1
              audio.position = [ MONO ]
          }
          sink.props = {
              node.name = "Adam_Playback_Sink"
              node.description = "Adam AEC Virtual Reference Sink"
              media.class = "Audio/Sink"
              audio.rate = 48000
              audio.channels = 2
              audio.position = [ FL FR ]
          }
          playback.props = {
              node.name = "Adam_AEC_Playback"
              node.description = "Adam Audio Hardware Forwarder"
              target.object = "@DEFAULT_AUDIO_SINK@"
              node.passive = true
          }
      }
  }
]
```

---

## 6. Phased Implementation Roadmap

### Phase 1: Core Foundation & Sandboxed Audio
- [ ] Initialize project with `uv` in repository root directory.
- [ ] Deploy PipeWire `50-adam-aec.conf` with verified `webrtc.*` syntax.
- [ ] Implement `src/audio/earcon.py` binding to the `pulse` device index with `PIPEWIRE_NODE` + `PULSE_SINK` routing and PortAudio restart-on-abort recovery.
- [ ] Set up Piper TTS with `PULSE_SINK=Adam_Playback_Sink` environment target.

### Phase 2: Speech & Wake Word Loop
- [ ] Set up continuous `Adam_Clean_Mic` audio consumer stream.
- [ ] Integrate `openWakeWord` + `silero-vad` with dynamic thresholding (0.5 IDLE $\rightarrow$ 0.88 SPEAKING).
- [ ] Deploy `faster-whisper` on GPU 0 with `compute_type="int8_float32"` using CUDA 12 PyPI wheels.
- [ ] Verify interrupt spotter with instant generation epoch advance.

### Phase 3: Sandboxed Execution & Supervisor
- [ ] Build `src/execution/supervisor.py` with Arch-canonical `bwrap` symlinks, `--unshare-pid`, `--tmpfs /dev/shm`, `--ro-bind /sys /sys`, and `--dev-bind-try` for GPU nodes.
- [ ] Route all job logs to `~/.local/state/adam/jobs/` (protecting `/tmp` `tmpfs`).
- [ ] Implement `PriorityAudioArbiter` and `TriStateConfirmationManager` with active watchdog timers.

### Phase 4: Tripartite LLM Provider Switch
- [ ] Implement `CanonicalTool` with `to_openai()`, `to_gemini()`, and `to_anthropic()`.
- [ ] Build multi-provider adapter supporting Ollama, Groq, Gemini, and Anthropic Claude.
- [ ] Cap local Ollama context to 4k tokens to protect Pascal VRAM.

### Phase 5: End-to-End Integration & Benchmark
- [ ] Test barge-in: Interrupt Adam mid-sentence and verify PortAudio stream recovers cleanly on subsequent tones.
- [ ] Test security & isolation: Verify that attempts to touch `~/.ssh` or `/var/run/docker.sock` in sandboxed jobs fail.
- [ ] Test Slime Tensei batch transcode: Execute CPU `libsvtav1` transcode on `~/Downloads` video files inside the sandbox and verify completion alert pops only after returning to `IDLE_LISTENING`.
