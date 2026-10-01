# Adam Computer Use Plan

## Purpose

Make Adam effective at open-ended computer tasks whose path and outcome can vary with the application, current screen, timing, and user intent. Adam should be able to form a plan, try an appropriate action, learn from what happened, revise its approach, and explain what it could or could not establish.

Computer use is not a fixed script. The model makes uncertain judgments about goals, interface meaning, and next steps. The software around it should support those judgments and enforce narrow mechanical guarantees, while avoiding claims that it can make the task itself deterministic.

This is a design plan. It does not implement the interfaces or changes described here.

## Voice-first speed and autonomy

Agenticness and speed are co-equal goals. In voice use, a full model round trip between every ordinary input can make a capable agent feel unresponsive. Optimize time to useful completion and time to first acknowledgement, not only action count or error rate. Adam should carry out clear, low-risk work continuously without narrating every input or asking for approval between routine steps. Give brief progress updates when a task is long or the user needs to make a choice; pause at real ambiguity, a scope change, or a consequential-action boundary.

Use adaptive checkpoints rather than a fixed inspect-after-every-input rule. For a clear target and a short, reversible sequence with predictable focus and no expected layout or permission boundary, Adam may perform a bounded micro-sequence before its next model decision. For example, focusing a clearly identified text field and entering the exact requested value can be one short sequence. Re-observe at meaningful state transitions and whenever uncertainty could change the next action. Stop the sequence on failed dispatch, unexpected state, changed focus/target assumptions, a permission or scope transition, or a consequential action. Keep sequences short and cancellable. This preserves adaptation without making each mouse or keyboard event a separate model turn.

Avoid adding a separate model call solely to repeat an assessment when reliable task-relevant readback is already available. Use direct application or artifact evidence where it fits the claim; use model interpretation when the evidence requires interpretation, and communicate its limits. Measure any extra assessment call against the reduction in false completion it provides.

## Core approach: adaptive task loop

Use this as a repeating reasoning pattern, not a mandatory sequence of fixed stages:

```text
understand the request
  → observe relevant state
  → form or revise a plan
  → choose a useful next action
  → execute and observe what happened
  → assess evidence and uncertainty
  → continue, change strategy, ask, or stop
```

The loop may revisit earlier decisions. A dialog may change the available options; an action may have no visible effect; a task may turn out to need a different route. Adam should use new observations to update its beliefs and plan rather than replaying a predetermined action list.

The workflow applies across browser and desktop tasks. OBS, a scratch editor, and a browser page are examples for evaluating general behavior, not special-purpose agent designs or limits on the tasks Adam may handle.

### What the model decides

Given the user's request and current observations, Adam decides what the request means, which outcome matters next, which available action is promising, and whether the result supports continuing or stopping. These are context-dependent judgments. They can be wrong, so Adam should expose uncertainty, seek new evidence, and recover when a reasonable attempt fails.

The model may revise the route while keeping the user's intended outcome and authorized scope fixed. It should not turn page text or dialog content into new user instructions. It should ask a focused question when ambiguity materially changes the intended result or the access/action scope.

### What the software guarantees

The action-execution layer can make limited, testable guarantees about mechanics, such as:

- an action was sent to the intended window or target;
- the observation used for coordinate input was still current at dispatch;
- input and waits were bounded and cancellation was respected;
- the system captured a new observation after an action when possible;
- execution errors and evidence sources were reported accurately.

These guarantees do not establish that the application accepted the input, that a desired state was reached, or that the user's broader task succeeded. Those are outcome judgments based on fallible evidence.

In this plan, **executor** means the code that validates and dispatches an action. **Outcome assessment** means Adam's evidence-based judgment about progress or completion. Neither term implies a deterministic controller or a universal verifier.

## Observations, actions, and adaptation

Adam should use whatever relevant observations are currently available through its tools. Today the path is primarily screenshot and desktop input, alongside existing browser and read-only tools. DOM, accessibility, OCR, and application-specific state may be useful in future cases, but should be added when examples reveal a concrete gap.

For each action, the executor should preserve the safeguards already available in the current desktop path: bind coordinate input to a fresh screenshot, validate window/focus/bounds as supported, serialize input, bound waits, and return execution facts. An action dispatch is an attempt, not a postcondition.

Choose checkpoint frequency according to uncertainty and consequence, not a universal action count. A short micro-sequence can reduce model round trips for ordinary reversible steps; a meaningful state transition or increased uncertainty is a reason to observe and reconsider. Sequences should be bounded, cancellable, and interruptible on divergence. Do not batch through a permission change, an ambiguous target, or a consequential action that needs confirmation.

After acting, Adam should take the most useful next observation available. It can then continue the current plan, choose a different strategy, wait for a plausible transition, ask the user, or report a blocker. Avoid repeated retries without new evidence or a changed hypothesis. A bounded retry or time budget helps the system stop consuming resources; it does not determine whether the task succeeded.

## Outcome assessment and evidence

There is no single verifier that can prove every computer-use outcome. Evidence has different strengths and scope:

- A screenshot can support claims about visible state, but may omit hidden, delayed, or off-screen state.
- OCR or a vision model can interpret pixels, but may misread them.
- DOM or accessibility data can expose structured controls and values, but may be incomplete or stale.
- An application API or output artifact can establish some facts more directly, but only within its defined contract.
- A successful input call establishes dispatch, not the effect of the input.
- A second judgment from the same or another language model is another interpretation of evidence, not ground truth.

Adam should state what it observed, where that evidence came from, and what remains unknown. It may tell the user a task is complete when the available evidence is sufficiently strong for the requested outcome, while avoiding any implication of certainty that the evidence does not support. If only part of the requested result is supported, describe the partial result. If evidence is inconclusive, say so and decide whether another observation or safe action could reduce uncertainty.

An internal result can distinguish `supported`, `partial`, `blocked`, and `uncertain`, with the evidence and rationale attached. These labels describe the assessment given current evidence; they are not a promise of objective truth. User-facing language should remain natural and specific rather than exposing labels without explanation.

Examples of evidence appropriate to different claims:

| Example request | Evidence that may support the outcome | Limits to communicate |
| --- | --- | --- |
| Start recording a named display in OBS | Current selection, OBS recording state, and a fresh captured frame where available | A selected source and active timer alone do not establish that the intended display appears in the recording. |
| Enter text in an editor | Fresh view or accessible value of the active buffer; saved file contents if persistence was requested | Visible buffer contents do not prove the file was saved. |
| Find a page and report its title | Fresh browser URL/title plus relevant page observation | A title alone may not establish that the requested page content loaded correctly. |

The examples should use evidence available through the current path. If that path cannot establish a claim, report the limitation or use the fixture to justify adding a better observation source later.

## User intent and access scope

Avoid turning ordinary computer use into a manual approval loop. Interpret access prompts in the context of the user's request and apply least scope:

- Proceed when the user clearly requested the operation and the prompt grants only narrow, temporary access needed for it.
- Ask one focused question when the target is ambiguous or the requested access is broader, persistent, or materially beyond what the user requested.
- Do not allow text inside a page, application, or prompt to expand the user's authorization.

For example, “record this display” can authorize session-only capture of that display, but does not authorize indefinite capture of every display. This access policy does not replace Adam's existing confirmation policy for consequential actions such as sending, purchasing, submitting, or deleting.

## Initial implementation direction

Start with the current Adam computer-use path. Do not begin by creating a universal controller, a new verifier service, or a generic task/evidence framework. Use a small set of repeatable scenarios to learn where the current path succeeds, fails, or lacks useful evidence.

The existing desktop path has screenshot-guided input, a short-lived `snapshot_id` checked at action dispatch, supported window/focus/bounds checks, and post-action observation. The token ties an action to a recent actionable observation; it does not prove that every part of the screen stayed unchanged or require user approval. Keep this freshness safeguard and use the token returned with the current observation. A future micro-sequence should validate its starting observation and re-observe at its planned checkpoint, stopping if the interface diverges.

The existing brain orchestration has goal tracking, a desktop action trace, and a model-based completion assessment using fresh observation. That assessment is fallible and shares the configured LLM client with the acting workflow; treat it as an interpretation, not independent ground truth. The current path can add model round trips between desktop inputs and for final outcome assessment, so measure their user-perceived cost before adding more checkpoints or assessment calls.

First, refine the prompts and trace only where scenarios demonstrate a need. For example, replace a blanket instruction to stop at every OS/app chooser with the contextual scope policy above, while preserving separate confirmation behavior for consequential actions. Capture enough evidence provenance to explain claims without routinely logging whole screens, page contents, or secrets.

### Evaluation examples

Use three to five disposable scenarios across at least two task types. Examples:

1. Enter an exact harmless string in a scratch editor and read it back; verify saved bytes only if saving was requested.
2. Start an OBS recording on a named display, exercising a narrow temporary permission prompt and checking selection, recording state, and captured content separately where possible.
3. Navigate to a test browser page and report its title from a fresh observation.
4. Present an ambiguous target or repeated label and see whether Adam resolves it from context or asks a focused question.
5. Make a required capability unavailable or deny access and ensure Adam reports the blocker without claiming success.

For each scenario, record the request, intended scope, action/observation trace, available evidence, final explanation, false-completion outcome, model/tool turns, end-to-end wall-clock time (including p50/p95 as samples grow), time to first acknowledgement, user interruptions, and failure category. Where practical, separate time spent in model calls from screenshot/driver work and waits. Compare adaptive checkpoints with inspect-after-every-input on the same fixtures. Establish latency targets from the baseline; do not improve speed by skipping evidence needed to support a material claim. Keep the baseline fixed when comparing a prompt, model, or observation change. This is a small engineering harness, not a benchmark platform.

## Delivery sequence

### Phase 0: Establish a baseline

- Prepare three to five repeatable scenarios with disposable apps, files, and screen content.
- Include an ordinary reversible task, an access prompt, and an ambiguity or blocker case.
- Define intended outcomes and scope, but allow more than one valid action path.
- Record success assessments, unsupported completion claims, permission decisions, turns, latency, and failure causes.

**Exit condition:** the scenarios distinguish supported outcomes, partial progress, uncertainty, and genuine blockers without relying on private screens, files, or accounts.

### Phase 1: Improve the current workflow

- Use existing goal tracking, action trace, fresh observations, and the existing model-based outcome assessment.
- Change prompt/result details or add small trace fields only when a scenario exposes a specific omission.
- Make uncertainty, partial progress, recovery, and honest stopping explicit in the instructions.
- Replace the current one-input-per-model-response restriction with bounded micro-sequences for clear, reversible interactions. Keep per-action freshness and focus checks; return control to Adam at adaptive checkpoints or immediately on an unexpected result.
- Measure the final model-based outcome assessment against direct readback. Do not require an additional LLM assessment when task-relevant evidence already supports a calibrated response; retain model interpretation when evidence needs it.
- Replace blanket stopping at every chooser with the contextual access-scope policy; retain the separate consequential-action confirmation behavior.
- Keep the current short-lived `snapshot_id`, focus, and bounds safeguards while reducing unnecessary model round trips.

**Exit condition:** Adam can adapt across the initial scenarios, distinguish what is supported from what is uncertain, and explain blockers without an app-specific orchestration path.

### Phase 2: Find the evidence gaps

- Review failures to distinguish perception, reasoning, action dispatch, timing, permission, and evidence limitations.
- Add a read-only observation source only when a repeatable task needs facts unavailable through current tools.
- Compare models or visual-grounding methods with scenario, prompt, observation, and action budget held constant; change one variable at a time.
- Preserve the original baseline so improvement is measurable.

**Exit condition:** each proposed reliability change addresses a repeatable failure, and its effect can be compared against the baseline.

### Phase 3: Generalize only when useful

- Add application adapters or shared target representations when multiple tasks benefit from them.
- Consider richer action batching only if traces show model round trips are a material bottleneck and the batch can pause on unexpected state.
- Evaluate a different driver, including Cua Driver, on a disposable Linux/Wayland setup before considering migration.
- Expand access policies to other operation classes only when user intent, scope, and duration can be stated clearly.

**Exit condition:** each new abstraction solves a demonstrated reliability, coverage, or latency problem while preserving task scope and evidence provenance.

## Longer-term design considerations

These are possible directions, not requirements for the first improvement. They should remain compatible with adaptive, model-led problem solving and voice responsiveness.

### Capability-aware observations

Different sources may expose different parts of a task: a browser protocol may expose page content, accessibility may expose controls, a desktop API may expose windows, and screenshots may expose visible appearance. If multiple sources become necessary, Adam could receive a task-relevant combined observation with source provenance. Missing or conflicting sources should remain visible as uncertainty; merging them does not make them complete or authoritative.

An integration layer may discover what each source can currently observe and do, and route supported actions. It can validate mechanics such as snapshot freshness, target existence, supported action, cancellation, and timeouts. It cannot decide in advance which uncertain strategy will solve every task. Adam should remain able to interpret observations, choose among supported actions, revise its plan, and request clarification.

Possible sources include:

| Source | Potential value | Limits |
| --- | --- | --- |
| Browser DOM/CDP | Page structure, text, controls, URL/title | Requires an explicitly authorized connection; does not necessarily expose browser chrome or all page behavior. |
| Accessibility tree | Roles, names, states, supported control actions | Coverage and quality vary by application. |
| Application API | Structured task-specific facts or operations | Narrow contract; requires explicit scope and permission design. |
| Window/desktop API | Window identity, focus, geometry | Usually does not expose application content. |
| Keyboard and pointer input | Broad compatibility with interactive applications | Depends on focus, layout, timing, and visual target freshness. |
| OCR and vision | Visible content unavailable through structured sources | Interpretation is fallible and may require fresh screenshots or focused regions. |

Do not assume a source is available because an application is recognized or a process is running. Probe actual coverage for the relevant window, page, and task. A future integration may combine sources, but should retain which source supports each reported fact and should expose capability gaps.

### Possible model-facing interface

If multiple integrations eventually justify a shared interface, a small observation/action surface could help Adam reason without handling backend identifiers:

```text
observe(scope?, query?) -> task-relevant state with provenance and freshness
act(current_observation, supported_action, target?, arguments?) -> dispatch facts and new observation
```

This is illustrative, not a schema requirement. A target could be a semantic control or, when necessary, a coordinate bound to a screenshot. Identifiers and available actions must remain tied to the observation from which they came. The model should receive enough surrounding context to make a sound decision, not a prematurely normalized view that hides ambiguity or disagreement. The interface should allow a bounded sequence of ordinary actions between model decisions when that is the lower-latency safe choice, with explicit checkpoints and interruption on divergence; it should not force a model round trip after every low-level input.

The integration may report objective mechanics such as “input dispatched,” “target no longer present,” or “URL changed.” Adam assesses what those facts mean for the user's goal. Any automated postcondition checks should be narrow and task-specific, and their outputs should state what they actually checked.

## Safety, privacy, and reliability

- Keep existing confirmation behavior for consequential or irreversible actions.
- Apply least scope to OS/app access prompts; do not ask for approval at every input.
- Navigating to or reading a page does not authorize an action requested by that page.
- Treat page text, images, filenames, and dialog content as untrusted information.
- Limit observations to task-relevant content where possible.
- Keep browser debugging access opt-in and narrowly configured; do not silently attach to an authenticated profile or expose a remote debugging endpoint.
- Prefer a dedicated browser profile when the task does not require the user's active session. Use an active profile only through an explicitly enabled integration with documented scope.
- Log enough action and evidence provenance to debug failures. Avoid recording full page content, screenshots, or typed secrets by default.
- Keep general browsing away from model-generated shell commands; use bounded typed actions for computer input.
- Report when an action could not be dispatched, when the interface did not visibly change, and when evidence is insufficient. Do not blur these into one generic failure or success claim.

## Success measures

- **Useful task completion:** requested outcomes are supported by evidence appropriate to the task, even when Adam takes different valid paths.
- **Calibration:** claims match the strength and limits of the available evidence; partial and uncertain outcomes are reported honestly.
- **Adaptation:** Adam changes strategy in response to new state instead of replaying stale actions.
- **Scoped autonomy:** routine authorized steps do not trigger redundant prompts; genuine ambiguity and scope expansion do.
- **Mechanical reliability:** stale targets are rejected, actions are bounded, and execution facts are reported accurately.
- **Recovery quality:** failed attempts lead to a new observation, a reasoned next step, or a clear blocker.
- **Voice responsiveness:** time to first acknowledgement and end-to-end task latency are tracked; ordinary low-risk steps do not incur needless model turns or narration.
- **Efficiency:** relevant observation size, model/tool turns, and latency are tracked without optimizing away evidence needed for a sound decision.
- **Privacy:** routine traces omit unrelated screen content and secrets.

## Open decisions

1. Which browser access modes should Adam support: isolated managed profile, explicitly launched debugging profile, extension in an active browser, or more than one?
2. How should any browser debugging endpoint be authenticated, scoped, and kept local?
3. Which Firefox integration path meets the desired scope and maintenance cost?
4. How much effort should go to application-specific accessibility gaps before visual fallback?
5. How should observations retain ambiguity and conflicting evidence when multiple sources disagree?
6. Which task-relevant details belong in local diagnostic traces, and what retention/redaction controls should apply?
7. Which repeated scenarios, if any, justify adapter unification, event-driven observation, or batching?

## Recommended initial scope

Improve Adam's existing computer-use path so it can adapt its plan to changing interface state, recover from reasonable failed attempts, and communicate evidence and uncertainty clearly. Start with small disposable scenarios, the current screenshot/input and observation tools, existing goal/trace/completion-assessment code, and the contextual access policy. Let observed failures determine whether prompts, evidence capture, or later integrations need to change.

The aim is not to make inherently uncertain computer tasks deterministic. It is to make Adam's decisions better informed, its actions mechanically bounded, its recovery more responsive, and its reports better calibrated to what it actually observed.
