# Proposed codebase fixes

**Purpose:** Turn the current status and evaluation findings into actionable engineering proposals. These are proposals, not claims that the fixes are implemented.

## Priority and dependencies

Treat the proposals as a sequence of gates, not a requirement to build every feature before measuring Adam again:

| Priority | Work | Dependency / gate |
| --- | --- | --- |
| P0 | Baseline non-GUI latency/resource behavior; prevent unintended screen capture before another visual run | Non-GUI baselines can proceed immediately. GUI/model vision evaluation waits for capture scope verification. |
| P1 | Tray lifecycle indicator and capture-source metadata | State must be wired to the daemon's real lifecycle. Capture metadata must travel with the image and be checked before OCR, visual grounding, or provider submission. |
| P2 | Dedicated GUI sessions for background tasks | Build after identifying which target applications need the real user session. No implicit switch between interactive and background modes. |
| P3 | Durable long-run checkpoints and controls | Needed before unattended, multi-hour tasks; not a prerequisite for ordinary short requests or the first matched benchmark loop. |
| P4 | Optional remote web/phone channels | Separate opt-in features. Keep disabled until the local run state and approval semantics are durable and tested. |
| P1, parallel track | Speech and memory evaluation sets | Can proceed alongside non-GUI baselines; changes are retained only after paired evaluation. |

For performance comparisons, keep one short, versioned workload set and its exact model/provider/configuration. Target a task-relevant first-turn acknowledgment within 5 seconds of the end of the user's request; a capture earcon alone does not count. Report task success alongside time-to-first-acknowledgment, time-to-first-useful-result, completion time, tool-call count, request cost, CPU/GPU/RAM, and power when measurable. Include cold-start and warm runs where model loading affects the user. Mark unavailable readings as unavailable; do not infer power or GPU use from CPU time or device names.

## Implementation delegation

Offload simpler, bounded implementation tasks to the Antigravity CLI (`agy`), including straightforward web UI work, desktop notifications, and similar features with clear acceptance criteria. Use the `gemini-3.8-flash-medium` model and invoke it with `--dangerously-skip-permissions` so it can complete the assigned task without stopping for per-action permission prompts. Keep each delegation scoped to a specific deliverable; when it finishes, inspect the diff and run the relevant checks before retaining its changes. Reserve the harder architecture, visual grounding, benchmarking, and cross-machine integration work for direct iteration and verification.

## 1. Add a tray icon for request status

**Current status:** Adam has wake/capture earcons, occasional spoken progress updates, and desktop notifications, but no persistent visual indicator of what it is doing. The spoken update is delayed and configuration-dependent; terminal logs may not be visible to the user.

**Fix:** Add a system tray icon with distinct states for idle, listening, working (including tool use), responding, and error. Include a tooltip or menu with a short status, and return to idle after completion. Use the tray/status-notifier mechanism supported by the target Linux environments, and define a graceful fallback for environments without a working tray host.

**Why:** The user can glance at the desktop to tell whether Adam heard the request and whether it is still working, without relying on audio or a terminal. A tray icon is a familiar desktop-wide cue; support varies across environments, so the implementation should detect or document its fallback.

**Acceptance criteria:**

- State changes are driven by the actual audio/request/tool lifecycle, not inferred from elapsed time.
- Listening, active work, response playback/notification, and failure are visually distinguishable.
- State returns to idle after completion, cancellation, or failure; stale busy state is cleared on restart.
- Tray absence or failure does not affect voice operation, and the user still receives the existing audio/notification feedback.
- The behavior is checked on the project's supported desktop environments.

## 2. Scope desktop captures to the intended display

**Current status:** A live screenshot included another monitor with unrelated conversation content, and that image was sent to a model provider. Workspace isolation alone did not prevent capture of other monitors.

**Fix:** Make the capture target explicit (window, monitor, or full desktop), default computer-use flows to the intended target, and verify the resulting capture bounds before sending image content to a provider. Report when the requested scope cannot be honored.

Keep the target and the mechanism separate in the result: identify which window/output/desktop was requested, which backend produced the pixels, the desktop-space bounds, and the final image dimensions. Crop to the approved target before OCR, OmniParser, image attachment, or any other processing that can transmit data. A backend-wide screenshot followed by an unchecked label is not proof of scope. A full-desktop view is an explicit opt-in for tasks that truly need cross-window context; it must never be substituted after a window or monitor capture fails.

**Why:** This prevents unrelated screens from entering model requests and makes visual tests safer and more reproducible.

**Acceptance criteria:**

- Capture results identify their target and dimensions.
- Capture verification checks decoded image dimensions against the requested bounds (including output origin/offset), and rejects empty, malformed, truncated, or mismatched images before they reach OCR or a provider.
- A scoped capture never silently falls back to a full multi-monitor image.
- Unsupported or failed scoping is visible to the user/caller and prevents unintended image submission.
- Tests cover single- and multi-monitor layouts, negative output origins, mixed resolutions/scaling, focused windows at output edges, windows spanning outputs, malformed images, and fallback behavior using synthetic fixtures or an isolated test display.

## 3. Give background GUI tasks an isolated virtual display

**Current status:** Background computer tasks can interact with the same desktop session the user is using. A spare workspace does not isolate screenshots from other monitors, and a private Xvfb attempt could not use the host's active Steam session. The current setup therefore provides neither reliable GUI isolation nor a general solution for applications that require a logged-in desktop session.

**Fix:** Add an explicit execution mode for GUI work:

- **Interactive mode:** Adam uses the user's desktop when the task requires the user's open apps or visible session, with clear status feedback and correctly scoped captures.
- **Background mode:** Adam starts or connects to a dedicated virtual desktop session, runs the task there, and captures only that session. Keep the session available for inspection or handoff when practical.

Make the session lifecycle explicit: create/start, identify, capture, stop, and clean up. Detect when an application needs a real logged-in desktop, GPU, audio device, or other host resource that the virtual session does not provide; report that limitation rather than silently switching to the user's desktop. The virtual display isolates GUI state, but it is not a security sandbox for shell, files, network access, or credentials.

Do not assume Xvfb is a drop-in background session for applications tied to the host's interactive login (for example, a running Steam client). First check the app/session requirements and report a clear unsupported-session result. If a test requires the real desktop, use interactive mode only with an explicit user-visible run state and scoped screenshots.

**Why:** Background tasks should not move the user's cursor, change their windows, or include unrelated desktop content in screenshots. A dedicated session also makes GUI evaluation repeatable and allows the user to inspect what Adam sees.

**Acceptance criteria:**

- Background tasks cannot capture or control the user's active desktop or other monitors.
- Each task's screenshot source is identifiable and scoped to its virtual session.
- Applications that require an unavailable session resource fail with a clear explanation; there is no implicit fallback to the user's desktop.
- The session can be inspected or handed off when supported, and is cleaned up after completion, cancellation, or failure.
- Tests cover session startup, capture scoping, failure handling, cleanup, and separation from the active desktop.
- Permissions for shell, filesystem, network, and credentials are enforced independently of display isolation.

## 4. Make long-running tasks resumable and observable

**Current status:** The status report describes bounded request/tool loops and short live GUI attempts, but does not establish durable task state or recovery after a process, provider, or virtual-display failure. An eight-hour task cannot rely on one uninterrupted in-memory run.

**Fix:** Add a durable run record for long-running tasks. Persist the user's goal, run/session identifiers, completed steps, current checkpoint, next intended action, timestamps, and errors. After interruption or restart, re-observe the task environment and compare it with the checkpoint before resuming; do not blindly replay the last action. Provide progress heartbeat, bounded retries, configurable runtime/model-call/cost/resource limits, and pause/stop controls. Notify the user on completion, failure, a request for input, or a detected stall. Preserve logs and the virtual session for inspection or handoff where practical.

**Why:** Long runs will encounter transient failures, stalled applications, service restarts, and user interruptions. Durable state and observable progress make recovery safer and help the user distinguish slow work from stuck work.

**Acceptance criteria:**

- A run can resume after Adam or a provider call is interrupted, using a fresh observation before taking another action.
- Checkpoints record completed work and the next intended action; repeated side effects are not replayed without verification.
- A heartbeat reports the last meaningful progress time, and a stalled run pauses or asks for input after a configured threshold.
- Runtime, retries, model calls/cost, and applicable resource use have configurable bounds; hitting a bound stops safely and explains why.
- The user can pause or stop a run and inspect its current state/session without losing the run record.
- Completion, failure, stall, and user-input-required states generate a notification and remain visible in the run history.
- Verbal confirmation remains the default; the web UI and its network listener are disabled unless the user opts in.
- An unanswered, declined, expired, or unreachable confirmation never authorizes the pending action.
- Logs and checkpoints survive process restart and are retained or cleaned up according to an explicit policy.
- Tests cover interruption/resume, stale checkpoints, duplicate-action prevention, limit handling, pause/stop, and failure notification.

### Interaction UX: verbal by default, optional web UI

Keep voice as the default interaction and confirmation path. Adam asks through TTS and accepts a spoken response through its existing confirmation flow. For a long-running task, persist the pending question and pause the task while waiting; the short audio-response window must not cause Adam to execute the action or discard the run. A configurable timeout may leave the task paused or stop it safely, but must never count as approval.

Provide two optional remote interaction channels, both disabled until the user opts in:

| Channel | Purpose | Default |
| --- | --- | --- |
| Voice | Talk to Adam and answer confirmations while using the desktop | Enabled as today |
| Adam web UI | Full browser chat, run status/history, controls, and approvals | Off |
| ntfy.sh | Lightweight phone notification with approve/deny actions only | Off |

#### Adam web UI

The web UI is a browser-based client for Adam, not just an approval page. It should include:

- **Chat:** Send and receive messages through Adam's normal request and tool path. Preserve conversation history and show whether Adam is responding, waiting on a tool, paused, or finished.
- **Run view:** Show active and recent long runs, their goal, state, elapsed time, last meaningful progress, completed milestones, current checkpoint, and any error or pending question.
- **Run controls:** Pause, resume, stop, and inspect a run. Require a fresh observation before resuming GUI work after disconnection or restart.
- **Approvals:** Show the exact pending operation and consequence, with explicit Approve and Deny controls. Display who/which channel answered and when.

Keep ordinary chat separate from instructions to an active run. The user must explicitly choose to send a message to the run (or the UI must make the selected run unmistakable); never silently reinterpret a new chat message as authorization or as a change to the run's goal. A run instruction can be queued while an action is in progress, then acknowledged as applied, rejected, or awaiting clarification.

The web server is off by default. When enabled for use on the desktop, bind to loopback unless the user chooses another interface. Phone/browser access requires a deliberate network setup, authentication, and HTTPS or a trusted VPN/tunnel; do not expose the service to the network automatically. Pairing/setup should make the enabled address and access scope clear. Keep the UI usable after a browser refresh or daemon restart by loading persisted run state, and show a disconnected/stale status when live updates are unavailable.

#### ntfy.sh approval actions

ntfy.sh is an optional, lightweight approval vector for users who want a phone notification without using Adam's web UI. It is approval-only: do not send chat history, screenshots, routine progress, or task transcripts through it. A notification should state what Adam is waiting to do in a concise, privacy-conscious way and provide Approve and Deny buttons. Button presses return a response to Adam; they do not directly execute the operation.

Treat ntfy.sh as an external service. Explain that notification metadata and content pass through it, require explicit setup of the topic/access and response path, and keep secrets and sensitive task details out of notification content. The callback path must be authenticated and reachable from the phone; never assume a local-only Adam address is remotely reachable. A response must be bound to the run and exact pending operation, expire, and be accepted once. Missing, delayed, duplicated, denied, or invalid responses leave the action unauthorized.

#### Channel behavior and safety

Use the enabled channels as response surfaces for the same pending confirmation. A valid response from one channel resolves that confirmation across all channels; withdraw or mark other prompts stale where possible. Serialize consequential approvals per run so concurrent prompts cannot be confused. Record the decision, channel, timestamp, and pending operation in the durable run log.

If the web UI and ntfy.sh are disabled, verbal confirmation remains the only confirmation path. If Adam cannot hear a response, the user declines, or a remote channel is unreachable, pause or stop safely and explain how to resume. Never infer consent from silence, notification delivery, timeout, or a general chat message that was not an explicit response to the displayed approval.

**Acceptance criteria:**

- Voice remains the default; the web UI listener and ntfy.sh integration are both off until separately enabled.
- A spoken confirmation can pause a long-running task durably without losing its checkpoint or requiring the user to answer within a short audio capture window.
- The web UI supports two-way chat through Adam's normal request handling and clearly separates general conversation from instructions to a selected run.
- The UI shows current and recent run state, last progress, pending questions/approvals, and pause/resume/stop controls; state survives page refresh and daemon restart.
- The UI is authenticated. It binds to loopback by default; remote access requires explicit network configuration and a protected transport.
- ntfy.sh carries only opt-in, approval-specific notifications; it never becomes a chat or routine telemetry channel.
- Approval responses through voice, web UI, or ntfy.sh are tied to the exact run and operation, accepted once, and recorded. A response through one channel invalidates the others.
- Decline, timeout, missing response, stale/duplicate response, restart, or transport failure never authorizes the action.
- The user is told when ntfy.sh is enabled that it is an external service and that notification content passes through it.

## 5. Establish repeatable end-to-end performance baselines

**Current status:** Unit tests pass, but the recorded latency, resource, power, and cost observations are sparse and not consistently paired across the 16 GB laptop and Desky. A passing suite does not establish user-perceived response time or power use.

**Fix:** Create a small repeatable benchmark set for short conversation, a simple read-only tool request, a multi-step request, and a bounded GUI task. Record the model/provider route, success, time to first acknowledgment, completion time, tool calls, tokens/cost, and available CPU/GPU/RAM/power measurements on both hosts.

Keep baseline and candidate runs paired by machine, model/provider, prompt, settings, and starting state. Run enough repetitions to show spread (at minimum report every sample and median/range); do not present one fast run as a stable improvement. A GUI run is comparable only when Adam sees the same target state and the capture source is verified. Include no-op or fixture-based checks for tool correctness so lower latency cannot hide a worse answer or unsafe action.

**Why:** Comparable measurements show whether a code change improves responsiveness or resource use and expose regressions that unit tests cannot catch.

**Acceptance criteria:**

- Each run records machine, configuration, model/provider, task, starting conditions, and timestamp.
- Reports separate observed values from unavailable measurements and one-off samples.
- Before/after comparisons use the same task and settings; results are not presented as stable from a single run.

## 6. Evaluate long-term and temporal memory with a durable set

**Current status:** Memory has semantic and lexical retrieval, with temporal-event work in progress, but long-range recall quality lacks a representative evaluation set. Relative dates, timezone boundaries, and corrections are open questions.

**Fix:** Define a versioned set of dated memories and questions spanning exact phrases, relative dates, day/week/year windows, timezones, and corrected facts. Measure whether retrieval returns the right event and preserves both statement time and event time.

**Why:** Unit coverage of individual cases does not show whether retrieval remains accurate across varied histories or whether a change introduces regressions.

**Acceptance criteria:**

- Expected relevant memories and date interpretations are explicit for each case.
- Evaluation includes ambiguous dates and corrected or superseded facts.
- Retrieval changes are compared against the same set, with failures inspectable by case.

## 7. Measure the complete speech path

**Current status:** The speech path includes wake detection, VAD, optional speaker checks/diarization, STT, and TTS. Lazy STT loading is intended to reduce startup and resident memory, but the status report has no matched end-to-end latency, quality, or power series.

**Fix:** Add repeatable speech evaluations that measure request recognition quality, time from end of utterance to acknowledgment and final response, and memory/resource use with the configured pipeline on the laptop and Desky.

**Why:** Component tests and model-call timing do not tell whether the complete voice interaction feels responsive or whether lazy loading improves the deployed experience.

**Acceptance criteria:**

- Measurements distinguish wake/capture, STT, model/tool work, and TTS time where instrumentation allows.
- Test audio and configurations are repeatable, and recognition errors are recorded alongside latency.
- Results identify optional speaker/diarization stages and their added cost.

## Suggested order

1. Freeze the current revision/configuration and record non-GUI baselines on both machines (simple chat, read-only tool call, speech path, latency, cost, CPU/GPU/RAM, and power where available). This can proceed while screenshot capture is being fixed.
2. Implement and test capture scope plus pre-processing/provider-send verification. Do not run another live visual-model evaluation until scope checks pass on the target desktop setup.
3. Add the tray status indicator and test lifecycle transitions for normal completion, interruption, errors, restart, and unavailable tray hosts.
4. Improve desktop grounding against a fixed set of general computer-use tasks. Compare the same inexpensive/free model routes, tool sets, and initial screens. Establish the general-computer-use pass gate before attempting Shenzhen I/O; the game is the final stress test, not a prerequisite for earlier work or a proxy for ordinary navigation quality.
5. Establish dedicated background GUI sessions only for applications that can run there, and verify the separation from the user's active session. Unsupported apps should return a clear limitation.
6. Add durable run records and pause/resume controls before starting unattended long tasks; keep the web UI and ntfy.sh optional and disabled unless individually enabled.
7. Evaluate speech, temporal memory, and resource optimizations against their fixed datasets and paired baselines. Retain a change only when quality/success stays acceptable and the targeted measure improves.

The order is intentionally staged: screenshot privacy is a gate for GUI tests, while general latency/speech baselines do not need to wait for tray, virtual-session, or remote-UI work. Durable background-run infrastructure is a prerequisite for unattended multi-hour work, not for the initial optimize-and-measure loop. Shenzhen I/O is held until general computer navigation has passed its task set; only then use one or two levels as a final stress test.

## Related documents

- [Codebase status](codebase-status.md)
- [Development plan](PLAN.md)
- [Progress report for 2026-10-03](progress-report-10032026.md)
