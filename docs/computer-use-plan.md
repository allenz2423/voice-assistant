# Adam Computer Use: Detailed Implementation Handoff

**Document status:** Engineering handoff and follow-on plan
**Last reviewed against code:** 2026-09-30
**Audience:** Engineers working on Adam's model loop, desktop tools, voice interaction, and computer-use evaluation
**Priority:** Agenticness and voice responsiveness, while preserving user intent, honest evidence, cancellation, and mechanical safety.

## How to use this document

This document describes the product problem, the intended behavior, the current implementation, and the recommended follow-on work. It is deliberately explicit about what already exists so an implementation handoff does not accidentally rebuild completed work or assume that a design proposal is already implemented.

The plan treats Adam as an adaptive model-led agent. The model owns task interpretation and strategy selection. The surrounding software makes the available state and actions useful, validates and executes the chosen actions, and reports what happened. It may enforce mechanical facts it can test; it must not pretend to make uncertain task reasoning deterministic.

The computer-use research overview is in [computer-use-agent-research.md](computer-use-agent-research.md). Tool protocol and cross-provider harness work is in [tool-calling-research.md](tool-calling-research.md). The OBS case study in [computer-use-obs-findings.md](computer-use-obs-findings.md) supplies one motivating failure scenario; its proposed architecture is not automatically the architecture to build.

## Executive summary

Adam needs to complete open-ended tasks in a live desktop environment. The next screen, control, permission prompt, and successful route may not be known in advance. In voice use, unnecessary model round trips between ordinary inputs are especially costly. Adam should be able to continue through clear, low-risk work, learn from fresh observations, change strategy when a reasonable attempt fails, ask only when a meaningful decision belongs to the user, and report exactly what the evidence supports.

The present code already includes several important pieces: a model/tool loop, desktop observation and visual input, short-lived screenshot snapshot IDs, active-window identity checks, per-action refreshes, model-selected coordinate-free CUA sequences, step outcomes, cancellation boundaries, and configured sequence ceilings. The plan should therefore focus on validating and improving this implementation with representative tasks, not replacing it with a universal controller.

The highest-priority engineering question is now: **Does the shipped tool loop let Adam complete representative, varied tasks quickly and recoverably, or do concrete failures reveal additional bottlenecks in task continuity, observation quality, action semantics, permissions, or evidence?** Gather that evidence first. Add abstractions only when multiple observed tasks justify them.

```text
User's requested outcome and scope
                 ↓
Adam interprets the request and chooses a next step
                 ↓
Tools expose useful current state and supported actions
                 ↓
Executor validates mechanics and performs the selected action
                 ↓
Adam receives result, error, and fresh observation when available
                 ↓
Adam continues, changes strategy, asks, reports a blocker, or finishes
```

This is a conceptual relationship, not a fixed sequence of mandatory stages. Adam may observe, act, or ask in the order that makes sense for the current task.

## 1. Problem

### 1.1 The product problem

Computer use is uncertain. Applications load at different speeds; dialogs may appear; a click may have no effect; the available controls vary; a second route may become better after observing a failure. The user's goal can remain stable while Adam's plan changes several times.

An agent that stops after launching or focusing an application is not completing the requested task. An agent that treats a dispatched click as proof of success is overclaiming. An agent that requests a model decision between every low-level input can be too slow for voice. An agent that executes a fixed script through changed UI state can be wrong in a less visible way. The system needs initiative and adaptation, with the host enforcing only the real boundaries it can verify.

### 1.2 The agenticness and speed problem

Voice adds a user-visible cost to every round trip: audio capture, model latency, tool execution, and response delivery. Requiring Adam to deliberate after every click or keypress can make it seem hesitant even when the user request is clear. But removing all checkpoints is not a sound speed strategy: new geometry, changed focus, or an unexpected dialog may alter what action is appropriate.

The goal is **adaptive checkpointing**:

- Adam can choose a brief sequence of related low-level inputs when later inputs do not depend on a new visual decision.
- The executor refreshes state and keeps its mechanical checks in force.
- The executor pauses and returns control to Adam when the next action requires a new target selection, changed task interpretation, permission decision, or other model judgment.
- Speed is measured together with completion, recovery, and claim accuracy.

### 1.3 The reliability problem

The system needs to distinguish at least four different things:

1. **Intent:** what result the user asked for and what scope they authorized.
2. **Action choice:** what Adam decided to try based on current evidence.
3. **Execution facts:** what the tool dispatched, rejected, cancelled, or observed.
4. **Task outcome:** what the available evidence supports about the user's intended result.

Collapsing these into a single “success” flag causes errors. A click handler returning normally says something about dispatch; it may say nothing about the desired application state. A screenshot can show visible state while missing hidden or delayed state. A model reviewing a screenshot is another interpretation, not ground truth.

### 1.4 Motivating general failure modes

The OBS investigation documented a useful case: Adam could launch and focus OBS and a recording timer could be active while the intended display remained unselected in a screen-sharing chooser. The task's meaningful outcome was not simply “OBS is recording”; it involved the requested display being captured. The case illustrates general risks:

- mistaking intermediate progress for completion;
- sending too much irrelevant window or accessibility state to a small model;
- using a stale screenshot after focus changes;
- failing to expose the needed target through the current observation source;
- conflicting short instructions that encourage an early “done” response;
- confusing tool execution facts with task evidence.

Treat this case as one diagnostic fixture. Do not design a one-off OBS workflow or require every future task to have an app-specific verifier.

## 2. Goals and non-goals

### 2.1 Goals

The computer-use path should:

1. Let Adam choose actions and adapt its strategy throughout an active request.
2. Avoid a model round trip after every primitive input when Adam has selected a safe related sequence.
3. Return useful current observations and clear capability gaps at a reasonable context and latency cost.
4. Preserve existing freshness, focus, bounds, serialization, and cancellation mechanics.
5. Return per-action execution facts and partial progress clearly.
6. Distinguish a requested result from intermediate app state and action dispatch.
7. Support recovery from failure without blind retries of possibly completed side effects.
8. Make permission handling follow the user's actual scope and the existing confirmation policy.
9. Let Adam finish when evidence supports the requested outcomes and express partial or uncertain results naturally.
10. Measure voice responsiveness and task quality together.

### 2.2 Non-goals

This handoff does not require:

- a deterministic controller that selects the task strategy;
- a mandatory observe → target → act → verify stage sequence for every task;
- a universal semantic target namespace or normalized UI tree;
- one universal verifier or task predicate service;
- a fixed action path for a particular app such as OBS;
- inspection or model deliberation after every click, keypress, or scroll;
- a replacement of the current desktop driver before evidence demonstrates its limits;
- automatic background CUA after Adam has returned to the user;
- adoption of DOM, AT-SPI, a new driver, a vision model, or an application adapter just because it is available elsewhere;
- a large benchmark platform before a small repeatable evaluation has shown a need.

## 3. Current implementation: verified starting point

This section records the implementation visible in the repository at the review date. Treat function names and behavior as a starting point for an implementation pass; check the current branch before editing.

### 3.1 Model loop and tool surface

`src/llm/brain.py` owns the conversation/tool loop. Adam can use multiple tool/model rounds during an active request, retains tool-call history with matching result messages, normalizes call IDs, validates calls through the shared tool validation code, and emits structured execution status metadata to the model. The brain maintains recent task context for desktop operations and can stop a batch after a handoff or policy boundary.

The model-facing computer-use options include:

- `observe_desktop` for available window, accessibility, and browser information;
- `capture_screenshot` for a fresh visual observation;
- `computer_control` for visual inspection and input;
- `focus_window` and `launch_application` for locating/focusing an application;
- optional `desktop_task` using the configured OCR/Jev path.

The installed tools are filtered by runtime availability. The `desktop_task` path is distinct from the general model-led `computer_control` path; its status should be treated according to its contract and not mistaken for a mandatory architecture.

### 3.2 Desktop observation and action mechanics

`src/tools/computer_control.py` implements `ComputerController`. The controller supports desktop backends according to availability, captures screenshots, maintains the active window identity and bounds where supported, and associates an actionable observation with a short-lived `snapshot_id`.

Before a mutation, it checks that the submitted snapshot is the latest token and that the active window identity still matches. Pointer coordinates are checked against screenshot bounds and active-window bounds where available. Actions run through a serialized path. The controller refreshes a screenshot after actions according to configured, app-aware delays and returns a new token only when focus stability can be confirmed. These are useful mechanical protections; they do not prove the app accepted the input or reached the user's intended state.

The current snapshot design should be understood narrowly:

- it ties an action to a recent captured state and window identity;
- it catches stale tokens and detected focus/window changes;
- it does not prove that no visual/layout change occurred inside the same window;
- it does not authorize an action;
- it is not durable across requests and must not be treated as a future-session credential.

### 3.3 Current sequence support

The `computer_control` schema supports a model-selected `action="sequence"` with an `actions` list. `ComputerController.run_sequence` executes supported steps sequentially, calls the same single-action path for each step, and carries forward the newly returned snapshot token. It reports which steps ran and returns the most recent observation where available.

As currently implemented:

- the configured default is at most eight actions per sequence, clamped in code to a maximum of eight;
- the configured default text budget is 20,000 characters across all typing steps in one sequence; every individual typing step also has the existing `max_text_length` bound;
- the configured default elapsed-time budget is 45 seconds, clamped in code to a maximum of 120 seconds and checked between steps; an in-flight action is allowed to finish under its own timeout;
- cancellation is checked at action boundaries;
- the executor allows coordinate-free continuations such as click then type, type then type, and a final key after typing;
- it pauses before later spatial actions that would need Adam to select a new target from fresh state;
- each dispatched step uses the regular action path, with its snapshot validation and fresh observation behavior;
- a sequence result describes execution and observations, not task completion.

These sequence rules intentionally limit reuse of old target geometry. They are not a semantic workflow engine. The next work should evaluate whether this balance is useful across ordinary tasks and whether the returned partial result gives Adam enough context to continue efficiently.

### 3.4 Prompt and skill guidance

`SYSTEM_PROMPT` in `src/llm/brain.py` tells Adam to keep track of requested outcomes, continue beyond intermediate launch/focus/click steps, use fresh observations, avoid stale coordinates, handle failures deliberately, and avoid claims beyond evidence. It permits short coordinate-free sequences and contextual selection of narrow temporary permission options when user intent is clear and policy allows it.

`skills/computer_use.md` provides a longer operational guide covering tool choice, observation, current snapshots, relevant app actions, permission dialogs, prompt injection in screen content, consequential actions, and evidence-based completion. Keep the system prompt concise enough for the model to use reliably; put detailed procedures in the skill or tool descriptions where that improves clarity without creating contradictory instructions.

### 3.5 Tool results and request bounds

The brain wraps tool output in correlated metadata such as call ID, origin, status, result completeness, effect status, and duration. This gives Adam meaningful distinctions such as invalid input, denied, partial, cancelled, timed out, uncertain, unavailable, and ordinary execution output.

The active model loop has a generous `max_tool_rounds` ceiling. Hitting it is a resource limit, not evidence that the task completed. The implementation should preserve partial-progress reporting and cancellation rather than lower the ceiling to create artificial responsiveness.

### 3.6 Existing tests and configuration

Relevant current code and test locations include:

| Area | Starting point |
| --- | --- |
| Brain/tool orchestration | `src/llm/brain.py` |
| Desktop observation and input | `src/tools/computer_control.py` |
| Tool definitions and argument validation | `src/llm/tools.py` |
| Behavioral guide | `skills/computer_use.md` |
| Sequence and snapshot configuration | `config.yaml.example`, under `computer_control` |
| Controller mechanics tests | `tests/test_computer_control.py` |
| Brain tool round-trip tests | `tests/test_brain_tool_round_trip.py` |
| Completion assessment tests | `tests/test_brain_completion_verifier.py` |
| Tool argument validation tests | `tests/test_tool_argument_validation.py` |
| OBS investigation | `docs/computer-use-obs-findings.md` |

The implementation handoff should update these tests when behavior changes. This document itself does not ask an engineer to run a broad test suite before deciding which behaviors matter; it lists the checks that should protect changes once implementation work begins.

## 4. Target behavior and responsibility boundaries

### 4.1 Adam owns task strategy

For a given user request and current evidence, Adam decides:

- what outcome the user asked for;
- which app or tool is relevant;
- whether it needs more observation before acting;
- which action or short sequence is promising;
- whether the result changed its understanding;
- whether it should continue, change approach, ask, report a blocker, or stop;
- whether the evidence is enough to make a user-facing completion claim.

These are uncertain judgments. Adam may choose poorly. The system should make errors visible and enable recovery rather than mask model judgment behind a predetermined controller.

### 4.2 The execution harness owns enforceable mechanics

The host can enforce facts within its control:

- tool is installed and currently available;
- arguments parse and satisfy that tool's supported contract;
- call/result IDs match;
- action is dispatched through the supported backend;
- snapshot token and detected focus/window state are current enough according to the implementation;
- coordinates and text/input sizes are within supported bounds;
- actions are serialized where concurrent input would conflict;
- per-call time/action limits and cancellation are respected;
- current authorization and confirmation policy is applied;
- returned status accurately distinguishes dispatch, observed result, failure, timeout, and partial completion.

The host should not claim it can establish that the application semantically understood input, that a broad UI state never diverged, or that the user's full task succeeded unless an observation source supports that claim.

### 4.3 Page and application content remain untrusted data

Text visible in a webpage, file, dialog, title, accessibility tree, OCR result, or screenshot can help Adam understand the current task state. It is not a new instruction from the user. It cannot expand the user's authorized scope, request unrelated actions, or override system/developer instructions. Treat this consistently in prompts, tool descriptions, and tests.

## 5. Functional requirements

### 5.1 Task interpretation and continuity

- Keep the user's desired outcomes available across tool/model rounds within the active request.
- Preserve enough recent action and observation context for Adam to avoid repeating a failed action blindly.
- Do not force a tool call on every assistant turn; answering, asking, or stopping can be correct.
- Do not stop merely because an app launched, a window focused, a dialog closed, or one input dispatched when the request asks for further work.
- If the user interrupts, the assistant returns, or a configured resource ceiling is reached, stop further CUA and describe partial progress accurately.
- If a future explicit resume feature is added, obtain a fresh observation and reassess scope before acting. Do not reuse an old `snapshot_id` as authority.

### 5.2 Observation quality and context cost

- Provide a fresh observation when Adam needs it to make a target or state decision.
- Prefer a focused app/window observation when it contains enough relevant context; use broader monitor/window inventory when discovery needs it.
- Avoid repeatedly sending large unrelated accessibility trees or window inventories when a current focused screenshot and concise metadata suffice.
- Retain provenance: distinguish screenshot/OCR/model interpretation, accessible state, browser data, and direct application results.
- Expose missing or conflicting observation capabilities as limitations rather than inventing targets.
- Do not add DOM, accessibility, OCR, app APIs, or semantic target normalization without a repeatable task demonstrating what current observations cannot establish.

### 5.3 Action selection and execution

- Preserve single-action control so Adam can stop and deliberate whenever it wants.
- Permit short model-selected sequences where Adam judges the later action can proceed without a new target selection.
- Keep each action within the existing tool contract; do not convert a sequence into a list of hidden app-specific workflows.
- Revalidate snapshot/focus/window/bounds mechanics for every action as supported by the backend.
- Do not reuse a coordinate from an old screenshot after an observation that changes target geometry is needed.
- Stop at a sequence boundary when the next action needs a new visual target, action dispatch fails, cancellation occurs, a permission/scope decision is needed, or existing confirmation policy applies.
- Return successful earlier steps and latest usable state when later steps are stopped or fail.
- Never represent reaching a sequence ceiling as task completion.

### 5.4 User intent, permissions, and consequential actions

- Carry out ordinary actions clearly implied by the user's request without asking for per-click approval.
- A narrow, temporary selection or grant may proceed when it is unambiguous, clearly implied, and permitted by existing policy.
- Ask a short focused question when the target is ambiguous, several materially different choices exist, or the access would be broader/persistent than the request implies.
- Retain the existing confirmation policy for purchases, deletions, external submissions, publishing, or other consequential actions.
- Never let page or dialog content expand authorization.
- Do not batch past a new permission boundary or consequential action that needs confirmation.

### 5.5 Recovery and failure handling

- On invalid arguments, return an actionable validation message so Adam can correct the call.
- On stale state, return the reason and ask for a fresh observation before target selection.
- On an unavailable capability, disclose the limitation and let Adam choose another tool or report the blocker.
- On timeout or cancellation, report whether a side effect may have occurred.
- Do not blindly retry an action that may have completed despite a timeout.
- Permit a repeated action when fresh state or a revised plan makes it a reasonable next step; identical arguments alone do not prove a loop.
- Use no-progress and resource safeguards to return control, not to decide success.

### 5.6 Outcome assessment and evidence

- Treat tool dispatch, app state, and task completion as separate.
- Use authoritative app/artifact readback when it directly supports the claim and is available.
- Use screenshot/OCR/vision or model assessment when appropriate, but communicate evidence limits.
- Avoid an extra model assessment after every action or when direct readback already supports the user-facing statement.
- Keep any model-based assessment as an interpretation, not an independent ground truth service.
- Report supported partial progress and remaining uncertainty naturally.

### 5.7 Voice responsiveness

- Acknowledge promptly when a task is likely to take time; do not narrate every input.
- Execute clear routine work without unnecessary waits or approval turns.
- Give concise progress when the request is long, a meaningful decision is needed, or silence would make the user unsure that Adam is working.
- Minimize model round trips where the next input is already clear and mechanically safe.
- Measure time to first acknowledgement, total task latency, and interruption/clarification burden alongside task success.

## 6. Workstreams and proposed implementation methods

The workstreams are ordered by expected user impact and dependency. Workstream 1 is substantially implemented in the current branch; its immediate task is validation and focused hardening. Other items are follow-on only when evaluation or source review confirms the gap.

### Workstream 1 — Validate and harden adaptive CUA sequences

**Status:** Initial implementation exists in `ComputerController.run_sequence`, the `computer_control` schema, brain dispatch, config, and tests.
**Priority:** Highest. Direct impact on model round trips and voice latency.

#### Problem

One model decision for every click or keypress creates avoidable latency. A sequence that freely reuses old coordinates, however, can act on an unexpected screen. The implementation must remove needless round trips without silently turning into a script executor.

#### Solution

Keep the model in charge of sequence choice. Execute a short ordered group of related inputs with the same single-action mechanics, then return step results and fresh state. Pause when the next step needs Adam to choose a new visual target or when an actual boundary prevents safe continuation.

#### Justification

Voice users feel model latency directly. A clear click-then-type action may not need another model call in between. A second click usually does need a new target decision from a fresh observation. Distinguishing those cases improves speed while preserving adaptive reasoning.

#### What will change

First, no large redesign is required. Review real traces and existing tests for sequence behavior, then fix concrete correctness or usability gaps. Maintain a backward-compatible single-action API. If policy changes are needed, update the controller, schema descriptions, skill, prompt, and tests together so they agree.

#### Proposed methods of implementation

1. Exercise the existing supported cases: one action; click then type; type then type; type then a final press; later spatial input returning `partial` with the latest screenshot/token; stale start token; focus identity change; malformed step; timeout; cancellation.
2. Confirm that each actual mutation enters `run`, therefore using the normal snapshot validation and capture path. Confirm the tool result returns the latest screenshot and token to Adam.
3. Check cancellation responsiveness at action boundaries. If an individual action can block beyond the sequence ceiling, identify the subprocess/driver timeout and make the limit honest; do not imply the sequence timeout preempts a currently blocking call if it only checks between steps.
4. Review size bounds as well as action-count bounds. Text input has a per-action maximum; confirm aggregate sequence input is also bounded enough for the model context and execution service.
5. Verify expected focus changes within a sequence are not misclassified as failure, while genuine focus/window uncertainty prevents spatial input.
6. Keep a sequence as a list of ordinary actions, not a macro name, an app-specific procedure, or a hidden plan. Adam can always request an observation and choose a new route.
7. Confirm ceiling results include completed step outcomes and do not claim the user's task completed.
8. Update tool descriptions and `skills/computer_use.md` if runtime behavior changes. Avoid conflicting instructions about whether another screenshot is needed.

#### Acceptance criteria

- Adam can choose a short related sequence without a model round trip between every primitive input.
- Single-action calls and explicit observation remain available.
- Later coordinate-bearing actions pause for a new target choice instead of reusing old geometry.
- Each step preserves snapshot/focus/bounds checks available in the controller.
- A partial result identifies completed steps and includes current usable state where possible.
- Cancellation and time limits stop future actions; limits are described accurately at their actual enforcement boundary.
- No sequence result is treated as proof of task-level completion.
- Evaluation shows lower latency or fewer turns on suitable tasks without reduced success, recovery, or claim accuracy.

### Workstream 2 — Build a representative CUA baseline and evaluation set

**Status:** The repository has an OBS case study and controller/brain tests; a compact end-to-end task baseline remains the key measurement task.
**Priority:** Start alongside sequence hardening; use results to decide later work.

#### Problem

Individual unit tests can prove snapshot rejection or result serialization but do not establish that Adam can complete an open-ended task through the live voice/model/tool loop. One OBS scenario alone cannot show that an architectural change generalizes.

#### Solution

Create three to five disposable, repeatable scenarios across at least two task types. Give each scenario an allowed outcome and reliable test-time observations, while allowing more than one valid action path.

#### Justification

Agenticness and speed can trade off. Measuring only action count rewards short but incomplete behavior; measuring only success can conceal unacceptable voice latency. A small fixture set can expose actual bottlenecks without creating a benchmark project.

#### What will change

Use existing test conventions and temporary apps/files. Add a small evaluation helper only if needed to capture traces and timing consistently. Keep private user screens, personal browser profiles, and real consequential operations out of the fixtures.

#### Proposed methods of implementation

Use scenarios such as:

1. **Exact text entry:** focus a scratch editor field, enter a harmless specified string, and read it back. Check saved bytes only when saving is part of the request.
2. **Desktop choice:** select a named item among visually similar choices. Include an ambiguity variant where Adam should ask rather than guess.
3. **Permission/dialog:** complete a task requiring a narrow temporary selection, then a separate case where the requested scope is ambiguous or broad.
4. **Browser task:** navigate to a local/test page, use its ordinary UI, and report a requested title or visible state from a fresh observation.
5. **Recovery/blocker:** make an action fail or a capability unavailable and verify Adam inspects, changes approach, asks, or reports the blocker honestly.
6. **OBS-style end state:** test “record the requested display,” where recording active and correct display selected are distinct facts. Use a disposable configuration and inspect captured output only as a test oracle where feasible.

Record the exact request, intended scope, tool/model turns, timestamps, sequence steps, action/evidence status, final explanation, interruptions, and failure category. Measure at minimum:

- task completion and partial completion;
- recovery following failed or uncertain action;
- false completion/unsupported claims;
- unnecessary approval and clarification turns;
- time to first acknowledgement;
- end-to-end latency, including model, screenshot, wait, and driver time where distinguishable;
- cancellation behavior and post-cancellation actions.

For meaningful comparison, compare inspect-after-every-input with adaptive checkpoints on the same fixture. Change one factor at a time when comparing prompt, model, observation source, or action policy. Report p50/p95 only after enough trials to make them useful.

#### Acceptance criteria

- Scenarios permit multiple valid paths and judge requested outcomes rather than exact click sequences.
- At least one scenario exercises sequence behavior and one exercises a real user decision boundary.
- Latency and task quality are reported together.
- Evaluation traces identify which evidence supported the final claim without retaining unnecessary screen content.

### Workstream 3 — Improve observations only where tasks show a gap

**Status:** Current general path is screenshot-guided; `observe_desktop` exposes available desktop/browser state. Optional OCR/visual grounding and a Jev path exist behind configuration.
**Priority:** Evidence-driven; do not start with adapter consolidation.

#### Problem

A screenshot provides broad visual coverage but requires interpretation. Accessibility, browser DOM, OCR, and app APIs can expose useful structure but vary in coverage and may be stale or incomplete. Sending all available state can overwhelm a smaller model; hiding gaps can force it to guess.

#### Solution

Use the narrowest observation that answers Adam's current question, with provenance and limitations. Add a read-only source when a repeatable scenario demonstrates that current tools cannot expose a needed fact or target reliably.

#### Justification

The best source depends on the task and application. No single adapter or vision model is universally strongest. A hybrid approach may help, but each additional integration has maintenance, privacy, permission, and ambiguity costs.

#### What will change

Initially, improve only prompt/tool result details or focused observation selection demonstrated by the baseline. If repeated failures remain, add one source or capability at a time and retain the baseline for comparison.

#### Proposed methods of implementation

1. For every failed task, classify the cause: discovery, observation relevance, visual interpretation, target ambiguity, action mechanics, timing, authorization, or outcome evidence.
2. Check actual target coverage rather than assuming that a running process or recognized application exposes a DOM/accessibility node.
3. Scope broad window discovery separately from focused work. Use relevant window identity, dialog state, and a screenshot when that is what the decision needs.
4. Return concise accessible/browser facts rather than full unrelated trees where possible. Preserve details needed for disambiguating repeated labels.
5. If adding a source (for example, AT-SPI, browser DOM/CDP, app API, OCR, or a vision model), evaluate it on the failure fixture and record freshness, coverage, ambiguity, permission requirements, and latency.
6. When sources disagree, present disagreement as uncertainty. Do not merge them into a falsely authoritative “unified state.”
7. Do not expose raw backend IDs to Adam unless the tool contract requires them; retain local provenance to explain what produced a result.

#### Acceptance criteria

- The added observation source resolves a repeated failure that the old source could not.
- Adam receives enough relevant context to choose correctly without routine unrelated-state flooding.
- Staleness, missing capability, and source disagreement remain visible.
- Added latency and privacy/logging costs are measured and justified.

### Workstream 4 — Calibrate task completion and evidence

**Status:** Tool results explicitly leave `goal_status` as `not_assessed`; Adam receives fresh screenshots and is instructed to assess the requested outcome. There is no separate per-action computer outcome model call in the current path.
**Priority:** High where false completion occurs; do not add a universal verifier.

#### Problem

Adam can stop after intermediate progress or claim more than an observation supports. A separate model call may help interpret complex evidence but can add latency and share the same model weaknesses. An action handler cannot verify all task semantics.

#### Solution

Keep task-level outcome judgment with Adam, grounded in relevant evidence. Use direct application/artifact readback when available. Do not add a separate model-based assessment call unless the evaluation shows a meaningful accuracy benefit that justifies its latency.

#### Justification

Evidence differs by claim. An OBS recording timer does not prove the intended display was captured; visible buffer content does not prove a requested file was saved. Conversely, demanding an extra model check for every routine action adds cost without guaranteeing truth.

#### What will change

Use the evaluation set to classify false completion and the cost of any proposed extra assessment. Refine `SYSTEM_PROMPT`, skill guidance, or tool outputs when a concrete scenario shows the need. If a separate model assessment is proposed, compare its accuracy benefit and latency with direct readback and Adam's normal evidence-based reasoning before adding it.

#### Proposed methods of implementation

1. For every scenario, state the user's requested outcome and what evidence could reasonably support that outcome. This is an evaluation oracle, not a required runtime universal predicate.
2. Distinguish dispatch evidence from post-action observation and authoritative app/artifact readback.
3. Measure false completion under Adam's normal evidence-based reasoning. If a separate outcome assessment is proposed, measure whether it catches errors, adds false alarms, and justifies its latency.
4. Skip redundant assessment when direct readback already answers the relevant question.
5. Keep an assessment result scoped: “visible timer is active” is not “requested content was recorded.”
6. Ensure partial/uncertain outcomes return control to Adam or result in a calibrated user-facing response.
7. Never convert a timeout, action ceiling, model-call ceiling, or no-progress detector into a success result.

#### Acceptance criteria

- Intermediate focus, navigation, or dispatch is not reported as task completion when more was requested.
- Claims cite or describe evidence within its scope.
- Additional model assessment is retained only where measured benefit justifies its cost.
- Partial progress and uncertainty are expressible without a fabricated failure or success.

### Workstream 5 — Keep authorization contextual and enforce policy in the right layer

**Status:** Current prompt and skill guidance describe narrow implied temporary selections, ambiguity, scope, and consequential operations.
**Priority:** Preserve and regression-test as CUA evolves.

#### Problem

Overly broad “always ask” instructions create unnecessary approval loops. Overly permissive instructions let a page or dialog widen the requested task or make a consequential choice the user did not authorize.

#### Solution

Interpret access in the context of the user request. Proceed with narrow temporary actions clearly implied by the request when existing policy permits. Ask when target or scope is materially unclear, broader/persistent access is requested, or existing policy requires confirmation.

#### Justification

The model should have room to act on clear intent without asking for every routine input, but authorization remains bounded by the user's request and system policy. Dialog text is untrusted context, not authorization.

#### What will change

Do not broaden access policy merely to improve benchmark speed. Include clear/ambiguous permission scenarios in evaluation. Change prompt, policy, or confirmation plumbing only if those scenarios reveal unnecessary friction or a scope violation.

#### Proposed methods of implementation

1. Separate ordinary reversible app input from consequential external actions in prompts and policy.
2. For chooser tasks, provide enough observation to distinguish options (for example, display name and resolution), then let Adam compare them with the request.
3. Treat a narrow session-only choice differently from persistent system access or a broader grant.
4. Confirm that UI text cannot grant authority or override user instructions.
5. Keep host-side confirmations authoritative for operations that require them; model instructions alone do not enforce permission.
6. Measure both redundant approvals and unsafe scope expansion.

#### Acceptance criteria

- Adam does not ask for approval between routine steps clearly inside the request.
- Ambiguous or broad grants lead to a focused question or a denial as policy requires.
- Consequential actions retain existing confirmation behavior.
- Untrusted on-screen text cannot expand scope.

### Workstream 6 — Voice experience and progress behavior

**Status:** Runtime cue/progress behavior exists; sequence support reduces some input-to-model round trips.
**Priority:** Evaluate alongside every CUA change.

#### Problem

Silence can make a long request feel stuck; constant narration makes an agent noisy; unnecessary model turns make it slow. The same behavior that is appropriate for chat may feel poor in audio.

#### Solution

Acknowledge promptly, carry out clear ordinary work without commentary between every input, and provide concise progress when a meaningful wait or user decision occurs. Keep user-facing language natural while preserving detailed machine-readable results for Adam.

#### Justification

The user needs to know that Adam is working and needs to make real decisions. They do not need to hear every click. Round-trip reductions should improve time to completion, not simply lower a count.

#### What will change

Use fixture recordings/timestamps to check acknowledgement and progress cadence. Adjust cue/progress behavior only when it improves user understanding without delaying the work or talking over interaction.

#### Proposed methods of implementation

1. Record time from request acceptance to first acknowledgement and to completion.
2. Mark meaningful wait states (application launch, long load, or external operation) separately from ordinary input.
3. Ensure progress messages do not trigger an extra tool/model cycle by themselves unless needed.
4. Keep failure/blocker reports concise but specific: what completed, what remains, and what choice or capability is missing.

#### Acceptance criteria

- User receives prompt acknowledgement on longer tasks.
- Routine low-level actions do not produce spoken narration one by one.
- Progress is not phrased as completion unless the requested outcome is supported.
- Changes are evaluated with interruption and latency observations.

### Workstream 7 — Add abstractions only after repeatable evidence

**Status:** Possible future direction. No universal controller or target framework is a prerequisite.
**Priority:** Deferred until earlier evaluation identifies a repeated cross-task need.

#### Problem

Multiple sources and app-specific operations may eventually improve coverage. Premature abstractions can hide uncertainty, multiply integration paths, and create a large amount of code before proving that the current path is insufficient.

#### Solution

Add the smallest shared capability that resolves a repeated task failure across more than one scenario. Keep the model-facing contract understandable and keep provenance visible.

#### Justification

Architecture should follow demonstrated reliability, latency, or coverage needs. A capability adapter may be worthwhile when several tasks benefit; an adapter framework is not useful merely because several integrations are imaginable.

#### What will change

Potential future additions include a focused browser observation, richer accessibility support, task-specific app APIs, event-driven wait/state capture, or a different desktop driver. Each requires its own evidence and rollout plan.

#### Proposed methods of implementation

1. Require at least one repeatable failure and identify whether it is app-specific or general.
2. Prototype the smallest interface that can address it.
3. Compare with the baseline for outcome success, recovery, maintenance burden, access scope, and voice latency.
4. Preserve fallback to current screenshot/input paths where possible.
5. Keep model strategy outside the adapter: adapters expose concrete facts and operations, not a task plan.
6. For Cua Driver or another replacement driver, evaluate in a disposable Linux/Wayland environment before considering migration. Check focus, coordinates, screenshot freshness, cancellation, dependencies, and session ownership.

#### Acceptance criteria

- A new abstraction addresses repeated evidence, not a hypothetical future need.
- The model still chooses task strategy and can change route.
- The new source or driver reports capability limits and evidence provenance.
- Improvement is visible in task outcome or latency without unacceptable new access or maintenance costs.

## 7. Delivery sequence

### Phase 0 — Code and behavior audit

Confirm the current branch still matches Section 3. Trace a single action, a coordinate-free sequence, a sequence pause before a later target, a timeout/cancellation, and a completed task. Check prompt/skill consistency and configured sequence values.

**Deliverable:** a short baseline report with current behavior, any mismatch from this document, and a small disposable scenario list.

**Exit condition:** engineers know which planned pieces are already shipped and which user-visible gaps remain.

### Phase 1 — CUA sequence hardening and baseline runs

Run the selected scenarios through the active brain/tool path. Check step-by-step state freshness, returned screenshots/tokens, cancellation, sequence ceilings, coordinate reuse, and single-action fallback. Fix only demonstrated correctness gaps.

**Deliverable:** test coverage for mechanical behavior and trace measurements for turn count, latency, completion, and recovery.

**Exit condition:** adaptive sequences reduce unnecessary round trips on suitable work and stop before a new spatial decision is guessed.

### Phase 2 — Prompt, permission, and completion calibration

Review failures for over-asking, premature completion, unsupported evidence, unnecessary assessment calls, or failure to ask at genuine ambiguity. Update system prompt, skill, tool descriptions, or assessment call conditions together and avoid contradictions.

**Deliverable:** scenario-backed prompt/policy adjustments and before/after behavior notes.

**Exit condition:** Adam proceeds on clear in-scope work, pauses at real boundaries, and makes evidence-calibrated claims.

### Phase 3 — Targeted observation improvements

Only after repeatable observation failures remain, add or refine one source at a time. Keep task context focused and preserve source provenance. Compare against Phase 0/1 traces.

**Deliverable:** a small capability addition with a specific failure it resolves and measurements of the tradeoff.

**Exit condition:** added observation materially improves coverage, target resolution, or claim quality for more than one relevant task or is clearly justified for a high-value specific task.

### Phase 4 — Generalize and maintain

Review task corpus and failures periodically. Promote patterns into shared tools only when several tasks benefit. Recheck behavior when providers, model, desktop backend, permission system, or input driver changes.

**Deliverable:** a maintained set of representative tasks and an evidence-backed backlog.

**Exit condition:** no generalization is adopted without an owner, a user-visible objective, and a way to detect regressions.

## 8. Observability and evaluation details

### 8.1 What to record

For each test run, retain enough data to explain the outcome:

- anonymized scenario/request identifier;
- model/provider and relevant configuration version;
- tool name, call ID, parser origin if available, and duration;
- action type and whether it was dispatched;
- sequence step index and stop reason;
- observation type, timestamp/freshness metadata, and source provenance;
- whether the result was complete, partial, uncertain, blocked, or unavailable;
- final task assessment and evidence summary;
- acknowledgement, progress, and end-to-end timestamps.

Avoid retaining full screenshots, OCR text, page content, credentials, or user-provided text unless the test explicitly requires it and the fixture is disposable. Redact typed content from logs. Keep enough information to reproduce a failure without creating a new sensitive-data store.

### 8.2 Failure taxonomy

Classify a failed task before choosing a fix:

| Category | Example | Likely response |
| --- | --- | --- |
| Intent ambiguity | “Open the right display” when multiple options match | Ask a focused question or gather disambiguating metadata. |
| Discovery | Correct app/window not identified | Improve window/app discovery or task context. |
| Observation coverage | Needed control/state absent from screenshot/accessibility | Evaluate another read-only source or report capability gap. |
| Interpretation | State was visible but Adam chose the wrong target | Improve context, prompt, grounding, or model; preserve alternative routes. |
| Freshness/focus | Target screenshot no longer matches active window | Reobserve; fix synchronization only if mechanics failed to catch a real mismatch. |
| Dispatch | Input backend rejected or failed to send event | Fix driver/availability/error reporting. |
| Timing | App had not settled before observation | Tune bounded app-aware waits or use an explicit wait observation. |
| Authorization | Unnecessary question or scope expansion | Refine context/policy boundary; do not solve both with blanket permission. |
| Completion evidence | Intermediate progress mistaken for the requested result | Improve evidence availability or completion instructions. |
| Voice interaction | Slow acknowledgement, too many spoken updates, or silent long wait | Tune cue/progress cadence and remove unnecessary model round trips. |

Do not label every failure “vision.” A better vision model cannot fix missing task state, stale focus, a policy conflict, or an incomplete tool protocol.

## 9. Security, privacy, and safe execution constraints

These requirements remain in force while improving autonomy:

- Scope actions to the user's request and actual permissions.
- Treat all observed interface content as untrusted data.
- Do not type credentials or private content inferred from the screen unless the user explicitly supplied and authorized it for the task.
- Keep desktop input serialized; concurrent screen mutations can invalidate one another's state.
- Bound sequence duration, number of inputs, text size, and waits with configurable limits that are high enough for ordinary work.
- Make cancellation stop subsequent sequence steps.
- Do not automatically replay possibly completed side effects after timeouts.
- Stop CUA when the user-facing request ends; explicit resume requires fresh state.
- Keep logs task-scoped, bounded, and redacted.
- Require existing confirmation for consequential external actions and material scope changes.

These are boundaries around execution, not a deterministic strategy for the task.

## 10. Open implementation questions

Resolve these from current code and measured scenarios. Do not block the initial hardening phase on speculative framework decisions.

1. Does the sequence timeout cover an individual in-flight driver call, or only the interval between action boundaries? Which lower-level timeout governs a blocked screenshot/input process?
2. Is aggregate text across a sequence bounded, or only the input size of each individual action?
3. Does each returned per-step message retain enough evidence for Adam to understand which action ran, what was observed, and why execution paused?
4. Which configured model/provider receives screenshots, OCR, or optional visual-grounding output, and how does that affect latency and target accuracy?
5. Do future task traces show that a separate completion-assessment call would improve user-facing accuracy enough to justify its extra model call?
6. Which app/backend/task failures repeat often enough to justify a new observation adapter or app API?
7. What retention/redaction settings apply to local action and screenshot diagnostics?
8. What are reasonable latency targets for acknowledgements and task completion on the current hardware and provider?

## 11. Handoff checklist

Before implementation begins:

- [ ] Confirm current branch and identify which changes are already present.
- [ ] Run a trace of single-action and sequence calls; do not infer runtime behavior from schema text alone.
- [ ] Identify the deployed model/provider, desktop backend, screenshot path, and enabled optional vision/OCR components.
- [ ] Select disposable tasks with observable outcomes and more than one valid action path.
- [ ] Capture a baseline for latency, tool/model rounds, recovery, and false completion.
- [ ] Decide which concrete failure this workstream addresses.

Before a change is considered ready:

- [ ] Adam still chooses strategy and can request a new observation at any time.
- [ ] The host only claims mechanical guarantees that the current code can enforce.
- [ ] Single-action behavior and adaptive sequence behavior both work.
- [ ] Freshness, focus, bounds, cancellation, and authorization behavior are covered.
- [ ] Partial results tell Adam what ran and what remains uncertain.
- [ ] Voice latency improves or the reliability benefit justifies its cost.
- [ ] The final user-facing claim matches the evidence.
- [ ] Prompts, skill docs, tool descriptions, implementation, and tests agree.
- [ ] Sensitive content is not added to routine logs.

## Recommended initial scope

Start by auditing and evaluating the existing adaptive `computer_control` sequence path through the active voice/model/tool loop. Fix concrete gaps in freshness, cancellation, partial results, prompt consistency, or completion claims. Use a compact disposable task set to measure speed and task outcome together. Add new observation sources, adapters, or controller abstractions only after those runs show a repeatable need.

The intended result is not a deterministic computer. It is an agent that can make useful choices, act promptly, notice when its evidence or assumptions are insufficient, recover from failure, and explain its progress honestly.
