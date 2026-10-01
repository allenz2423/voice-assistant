# Adam Tool Calling: Agentic Execution Implementation Plan

**Status:** Handoff proposal
**Research date:** 2026-09-30
**Audience:** Engineer implementing Adam's model/tool execution loop
**Priority:** Preserve agent initiative and voice responsiveness while keeping real authorization and execution boundaries enforceable.

## Executive summary

Adam should operate as a model-led agent inside a dependable execution harness. For an active request, Adam chooses the next useful action from the request, the available tools, and the results so far. It can keep going, revise its approach, use another tool, ask a focused question, report a blocker, or finish. The host parses and validates calls, enforces actual authorization and mechanical safety rules, executes them, and returns useful results. It should not dictate a fixed workflow for open-ended work.

The first implementation priority is removing unnecessary model round trips from desktop work. Adam should be able to choose a short sequence of related inputs and have the host execute that sequence with per-action freshness and focus checks. At the same time, malformed calls must become recoverable errors, duplicate-action handling must account for changing state, and approval prompts must reflect actual scope and policy.

This is an incremental change plan, not a request to replace the tool harness wholesale. Start with the CUA agency bottleneck, measure it, then address protocol gaps that affect configured providers or observed failures.

```text
User request and active task context
                  ↓
          Adam chooses next step
                  ↓
 Host parses, validates, authorizes, executes
                  ↓
 Host returns result, failure, and useful evidence
                  ↓
 Adam continues, changes approach, asks, or finishes
```

## 1. Problem

### 1.1 Product problem

Computer-use and tool-using tasks are inherently variable. A desktop may load slowly, a control may move, an action may fail, or a different route may become simpler. The agent needs to respond to what actually happened. A rigid sequence or a mandatory reasoning turn after each mouse or keyboard event makes that adaptation slow and makes voice interaction feel hesitant.

At the opposite extreme, a host that blindly executes model output, hides malformed calls, or replays uncertain side effects is not a dependable agent harness. The design needs to let Adam make strategic decisions while the host enforces the limited guarantees it can actually establish.

### 1.2 Current implementation bottlenecks

The following findings come from the inspected implementation, particularly `src/llm/brain.py`, `src/llm/provider.py`, and the desktop control path. Confirm each against the current branch before implementation because function names and behavior may change.

| Finding | Effect on agency or reliability | Likely code location |
| --- | --- | --- |
| The brain blocks a second desktop mutation in one assistant response. | Adam must spend another model round trip between inputs that could safely be executed together. This adds voice latency and interrupts a coherent plan. | `src/llm/brain.py`, `desktop_input_attempted` gate near the tool dispatch loop. |
| System guidance says to stop at some chooser or permission prompts. | Adam may ask the user even when a narrow, temporary grant is clearly implied by the requested task. | `SYSTEM_PROMPT` in `src/llm/brain.py`. |
| JSON argument parsing can replace malformed input with `{}`. | The actual parse failure is hidden; the tool may receive misleading empty arguments instead of Adam receiving a useful correction. | Tool argument handling in `src/llm/brain.py`. |
| Tool schemas are stored, but incoming arguments are not uniformly validated against them. | A schema shown to a model is not itself a runtime guarantee. Different tools can fail later and inconsistently. | `CanonicalTool` in `src/llm/tools.py` and the brain dispatch boundary. |
| Text-parsed call trust is inconsistent. | `text_fallback` and well-formed structured formats such as DSML do not follow one clearly stated policy. A blanket write ban would also discard useful structured calls. | Parser origin in `src/llm/provider.py`; policy in `src/llm/brain.py`. |
| Some provider adapters do not implement a complete native tool-call round trip. | Sending a schema alone would not be enough: assistant tool-call history, IDs, and matching tool results must also be represented. | Anthropic and Gemini paths in `src/llm/provider.py`. |
| Duplicate-action checks rely heavily on recent signatures. | The same action can be valid after the screen or task state changes, while repeated calls in an unchanged state can indicate a loop. Identical arguments alone do not settle the question. | Recent desktop action tracking and continuation filters in `src/llm/brain.py`. |
| Completion assessment can add an extra model call. | An additional interpretation can increase latency without establishing ground truth. | Desktop outcome assessment in `src/llm/brain.py`. |
| Active task context is primarily request-local/in-memory. | A failed provider call can interrupt work. Adam should recover during the active request where possible, but must not silently resume desktop actions after returning to the user. | Brain loop and request/session state. |

These issues do not justify building a generic agent framework first. The central problem is that the current execution path constrains Adam's useful decisions more than necessary and sometimes fails to return precise enough feedback for recovery.

## 2. Solution

### 2.1 Target behavior

For every active request, maintain a compact representation of the user objective, relevant authorization scope, actions taken, and useful results. The model receives tool outputs and remains responsible for choosing the next strategy. The harness provides a safe and interpretable way to carry out that strategy.

Adam may choose a single tool call or a short CUA micro-sequence. A micro-sequence is a model-selected series of related low-level actions executed without a model round trip between every input. It is an execution convenience, not a workflow template: the host must not restrict Adam to a fixed list of sequences or decide the task's semantic plan.

After execution, return per-step outcomes, errors, and fresh observations where available. If the next decision depends on newly observed screen geometry, return to Adam for that decision. If a planned step can safely use the geometry already selected by Adam, the host can proceed while still applying its ordinary mechanics checks.

### 2.2 Responsibility boundary

| Harness responsibility | Adam responsibility |
| --- | --- |
| Resolve the requested tool and parse its arguments. | Choose which tool or strategy advances the user's request. |
| Validate calls against the active tool contract. | Decide whether to continue, change approach, ask, report a blocker, or stop. |
| Enforce actual authorization, scope, cancellation, and mechanical CUA safeguards. | Interpret tool results in relation to the user's goal. |
| Correlate each result with the exact call and report failure clearly. | Decide whether another observation or action is useful. |
| Distinguish what was dispatched from what was observed afterward. | Make completion claims supported by available evidence. |

The harness does not infer that an action succeeded merely because dispatch returned. It also does not claim to know that a screen diverged unless it has an observation source that can establish that fact.

### 2.3 Operating rules

1. **Agent initiative:** Do not force a tool call on every model turn or a model turn after every primitive action.
2. **Contextual authorization:** Apply policy to the requested operation and actual scope. A narrow temporary grant clearly implied by the request can proceed where existing policy permits. Ask when the target or scope is materially ambiguous, access is broader or persistent, or an existing consequential-action policy requires confirmation.
3. **Action freshness:** Keep desktop coordinates bound to the observation from which they were chosen. Enforce snapshot, window, focus, and bounds checks already supported by the desktop path.
4. **Useful failures:** Return parse errors, validation errors, timeouts, stale state, denials, and uncertain side effects as distinct outcomes that Adam can use to recover.
5. **No blind side-effect retry:** If a write may have happened before a timeout, report uncertainty and let Adam inspect state before deciding whether to retry.
6. **Honest completion:** Budgets, stop conditions, and lack of progress never prove task success.
7. **No implicit background CUA:** When the response ends or the user interrupts, do not keep acting on the desktop. Explicit resumption requires a fresh observation and renewed assessment of scope.

## 3. Justification

### 3.1 Why model-led execution

Open-ended tasks do not have one reliable action path. Making the harness own a fixed observe/target/act/verify workflow would encode assumptions that may fail on an unfamiliar application or an unexpected state. Adam should choose actions and revise its plan from actual observations; deterministic code should enforce only mechanical properties it can establish.

This separation also avoids treating a second model judgment as ground truth. A model-based completion check may be useful evidence at the point of answering, but it is another interpretation of the same available evidence, not a universal verifier.

### 3.2 Why micro-sequences

Making Adam call the model after every click or keypress adds latency and can make voice interaction feel unresponsive. Letting Adam request a bounded sequence keeps strategic choices with the model while avoiding needless round trips for closely related inputs. The host still validates each mutation and can return control when the sequence cannot safely continue from its existing information.

### 3.3 Why incremental tool validation

Structured schemas can improve both model selection and host validation, but existing tools use different schema shapes and providers have different calling protocols. A broad strict-schema migration before checking compatibility risks removing working capabilities without fixing the main agency problem. Start with one reliable dispatch boundary, preserve parser provenance, and tighten contracts in response to observed errors.

### 3.4 Why bounded active requests

Longer tool loops help Adam repair failures and complete multi-step work. A low fixed hop count works against that. Generous configurable resource ceilings, cancellation, and state-aware no-progress handling bound runaway execution without dictating a normal task's length. A ceiling produces partial progress or a blocker, never a success result.

## 4. What will change

### In scope

- Allow model-selected sequential CUA micro-sequences while retaining single-action calls.
- Preserve existing per-action snapshot, focus, window, and bounds protections.
- Make chooser and permission handling reflect actual scope and existing policy.
- Normalize call parsing, validation, authorization, execution, and result reporting at a shared boundary.
- Preserve call IDs and parser/provider provenance so failures can be correlated and diagnosed.
- Return actionable tool errors and distinguish execution status from result completeness.
- Make repetition/no-progress handling sensitive to action history and available state evidence.
- Avoid unnecessary completion-model calls and measure task outcome and voice latency.
- Complete native provider tool-call round trips for providers that Adam actually uses, when a measured need justifies the work.

### Out of scope for the first implementation

- Replacing the current brain with a universal workflow engine or planner.
- A universal semantic target model, task predicate, verifier service, or CUA driver rewrite.
- A required Pydantic model for every tool or strict-schema migration across all providers.
- Background desktop work or automatic resumption after Adam has returned.
- A generic persistent-goals subsystem before explicit resume behavior and lifecycle are defined.
- A large benchmark platform. Begin with a few repeatable disposable tasks and useful logs.

### Expected files and components

| Component | Expected work |
| --- | --- |
| `src/llm/brain.py` | Remove the one-desktop-input-per-response constraint; dispatch and return micro-sequences; improve parse/validation feedback; tune repeat/no-progress handling; adjust system guidance; avoid unnecessary completion checks. |
| Desktop control implementation (locate the current `computer_control` tool/executor) | Reuse its current freshness and focus checks for each step; expose sequential execution with bounded step outcomes and cancellation. Keep semantic strategy out of the executor. |
| `src/llm/tools.py` | Establish a shared runtime validation path using existing schema dictionaries; add typed/Pydantic contracts incrementally where they simplify contracts. |
| `src/llm/provider.py` | Preserve parser origin and provider call IDs; make textual formats distinguishable by grammar and ambiguity; implement complete native tool-call round trips only for configured providers that need them. |
| Existing logs/metrics or a small evaluation script | Capture task success/partial success, latency, unnecessary approval turns, recovery, and unsupported completion claims. |

## 5. Proposed methods of implementation

Implement in priority order. Each workstream below states the problem, solution, justification, expected changes, proposed implementation methods, and acceptance criteria so it can be assigned and reviewed independently.

### Workstream 1 — Adaptive CUA micro-sequences

**Problem**

The current brain blocks a second desktop mutation in the same assistant response, making Adam spend an extra model round trip between inputs. This is the clearest agency and voice-latency bottleneck.

**Solution**

Let Adam choose either one desktop action or a short sequential action list in one tool call. Execute the list in order, report the outcome of each step, and return control to Adam when a later step needs information not available from the current observation.

**Justification**

The model should control task strategy, but it does not need to reconsider every low-level input. A short sequence can reduce latency while leaving strategic adaptation with Adam. Mechanical safeguards remain active for every mutation.

**What will change**

Remove the brain-level gate that rejects a second desktop mutation in one response. Extend the existing desktop action interface/executor to accept a sequence without removing the current single-action form. Return per-step dispatch and observation details, plus fresh state when available.

**Proposed methods of implementation**

1. Inspect the current `computer_control` tool and executor before changing its schema. Reuse their existing snapshot, active-window, focus, and bounds checks instead of duplicating them in a new generic layer.
2. Add an optional ordered `actions` form (or an equivalent backward-compatible representation). Each action must use an already supported operation and its existing argument format. Do not create a fixed menu of “focus then type” workflows.
3. Treat sequence execution as sequential. Validate freshness and mechanical constraints before every mutation. Keep every coordinate tied to the screenshot that supplied it.
4. At entry, validate the supplied snapshot. After a mutation, use the desktop path's existing refreshed state/token. Continue only with planned steps that do not depend on newly inferred geometry. If Adam needs to select a new target from a new screenshot, return that screenshot/state to Adam rather than guessing.
5. Stop the sequence when dispatch fails, cancellation occurs, an observable unexpected state invalidates later steps, authorization scope changes and requires a user-intent decision, or existing policy requires confirmation. Expected focus/target changes are not by themselves divergence. Do not claim generic divergence detection without evidence.
6. Apply generous configurable per-call ceilings for elapsed time, action count, and input size. These are runaway controls for one tool call, not a fixed task-step budget. On reaching a ceiling, return completed step results and fresh state when possible; Adam decides the next move.
7. Make the sequence interruptible. Do not start background continuation when the user-facing request ends.

**Acceptance criteria**

- Adam can issue two or more related desktop mutations in one assistant response when it chooses to.
- Existing single-action calls continue to work.
- Every mutation still passes through the existing freshness/focus/window/bounds checks.
- A later step needing a new visual target returns control to Adam with a fresh observation.
- Partial sequence completion reports exactly which steps were dispatched and which were not; it does not claim task success.
- No extra user approval is requested for routine actions or narrow temporary access clearly implied by the request and permitted by policy.
- Voice task latency improves on suitable tasks without lowering evidence quality for material completion claims.

### Workstream 2 — Contextual permission and chooser behavior

**Problem**

Broad prompt wording can make Adam stop at a chooser or permission dialog even when the requested task clearly implies a narrow, temporary grant. A blanket rule weakens initiative; blanket permission would exceed the user's intended scope.

**Solution**

Replace generic stop instructions with a contextual decision: proceed only when the target and narrow temporary access are clearly implied and existing policy permits it; otherwise ask a concise clarification or confirmation.

**Justification**

The user authorizes an outcome and scope, not every click individually. Conversely, page/dialog content cannot expand that scope. Permission decisions should follow the actual requested operation and existing security rules.

**What will change**

Revise the relevant system guidance in `src/llm/brain.py`. Keep any actual host-enforced authorization checks unchanged unless code inspection shows they are incorrectly blocking already authorized actions.

**Proposed methods of implementation**

1. Enumerate current chooser/permission cases and determine which are model guidance versus host-enforced policy.
2. Define practical categories in the prompt: clearly implied narrow temporary access; materially ambiguous target or scope; broader/persistent access; and actions requiring confirmation under existing policy.
3. Ask one focused question only for the latter categories. Do not teach Adam to treat every permission-looking UI as a hard stop.
4. Keep runtime authorization authoritative; prompt wording must not grant access that the host policy denies.

**Acceptance criteria**

- A disposable task requiring a clearly implied narrow temporary grant does not trigger a redundant approval turn where policy permits it.
- Materially ambiguous, broad, persistent, or policy-gated access still asks or is denied as required.
- No permission decision can be expanded by untrusted page or dialog text.

### Workstream 3 — Reliable parsing, validation, and call correlation

**Problem**

Malformed JSON can be hidden by substituting empty arguments, schemas are not uniformly enforced at runtime, and parser-origin policies are inconsistent. That leaves Adam with poor error feedback and makes calls harder to diagnose.

**Solution**

Create a clear shared dispatch boundary that retains call ID and origin, resolves the active tool, parses arguments, validates its contract, applies semantic authorization, executes, and returns a correlated result. Make malformed calls recoverable errors rather than silently changing their meaning.

**Justification**

Tool schemas only help if they are enforced. Preserving well-defined structured compatibility formats keeps provider flexibility, while distinguishing them from ambiguous free-text guesses allows risk-specific policy instead of a blanket write ban.

**What will change**

Update argument parsing and error reporting in `src/llm/brain.py`, tool contract validation in or near `src/llm/tools.py`, and parser metadata/normalization in `src/llm/provider.py`.

**Proposed methods of implementation**

1. Define one normalized call representation with tool name, provider call ID, parser origin, raw argument payload, parsed arguments, and validation outcome.
2. Do not turn invalid JSON into `{}`. Return a bounded `invalid_input` result with a concise explanation and the exact call ID; let Adam correct it in the normal loop.
3. Resolve against the active tool registry and validate against the tool's supported JSON Schema before dispatch. Reuse the schemas already supplied to models.
4. Keep Pydantic as an option for Python-authored contracts or typed results, not a requirement to migrate every tool immediately. Avoid assuming that a Pydantic wrapper around a schema dictionary provides validation by itself.
5. Preserve origin labels for native calls and textual formats. Treat a correctly parsed structured grammar (such as DSML) separately from an ambiguous free-text guess. Apply the same schema validation and semantic authorization after parsing.
6. Restrict genuinely ambiguous parser classes where needed; do not deny every mutation merely because the call arrived through a textual compatibility format.
7. Keep error messages safe and bounded. Include the argument path or expected type where useful, not secrets or unbounded raw model output.

**Acceptance criteria**

- Malformed JSON never silently becomes an empty argument object.
- Invalid arguments produce a correlated, actionable result Adam can use to repair the call.
- Native and well-defined structured text calls use the same runtime schema and authorization checks.
- Parser origin is available in logs and tool history for debugging.
- Existing supported tools remain available during incremental validation rollout; compatibility failures are visible before constraints are tightened broadly.

### Workstream 4 — Actionable results, retries, and loop recovery

**Problem**

Generic failures and signature-only duplicate checks can either stop useful recovery or suppress a valid action after the state changes. Retrying an uncertain side effect blindly can duplicate work.

**Solution**

Return structured execution status, completeness, bounded data, and a useful message. Use available state/effect evidence to decide whether a repeat is suspicious, and leave strategic retry decisions to Adam when safe.

**Justification**

Adam can recover only if it learns what happened. The host can detect some repeated mechanical behavior, but it cannot infer that two identical arguments always mean the same task-level intent or effect.

**What will change**

Standardize the model-facing result envelope at the shared dispatch boundary. Refine recent-action checks and error handling in the brain. Avoid adding a generic recovery planner.

**Proposed methods of implementation**

Use an envelope equivalent to:

```text
ToolResult:
  status: ok | invalid_input | denied | failed | timed_out | cancelled
  completeness: complete | partial | unavailable
  data: bounded structured result
  message: actionable model-facing explanation
  display: optional concise user-facing summary
  duration_ms: elapsed time
  call_id: exact matching call ID
```

1. Include dispatch outcome separately from observed effect. For desktop actions, include post-action observation/provenance only when the tool actually obtained it.
2. Distinguish stale snapshot, focus failure, parse/validation error, denial, timeout, cancellation, tool defect, and uncertain side effect.
3. On a timeout after a possible write, report uncertainty. Adam should inspect current state before choosing whether to retry.
4. Treat retry categories separately: safe transient transport retry; model correction of invalid input; and a strategy change after an action failure. Do not apply one automatic retry count to all cases.
5. Refine duplicate handling using recent call/result context and state evidence where available. Do not assume identical tool name and arguments are globally idempotent or always a loop.
6. If a repeated action appears to make no progress, return that observation to Adam and let it choose a different strategy, ask, or stop. Keep a generous runaway safeguard, not an early-exit rule.

**Acceptance criteria**

- Adam can distinguish invalid input, denied access, stale desktop state, timeout, cancellation, and uncertain side effects.
- Safe transport failures can recover without replaying non-idempotent actions.
- A legitimate repeated action after a changed state is not categorically blocked.
- A no-progress safeguard cannot be reported as task completion.

### Workstream 5 — Completion claims and voice responsiveness

**Problem**

An extra model-based outcome check can add latency while giving only another interpretation of existing evidence. At the same time, speed must not encourage unsupported claims that a task completed.

**Solution**

Let Adam decide when it has enough evidence to answer. Use direct application readback or other authoritative results where available. Use model-based assessment only at a meaningful answer boundary when it adds value, not after every action.

**Justification**

Dispatch success does not prove application success; a screenshot or OCR output also has limits. A second model judgment is not ground truth. The correct goal is proportionate evidence and honest reporting, not verification after every input.

**What will change**

Review `_verify_computer_outcome` and its call sites in `src/llm/brain.py`. Avoid invoking it for every desktop mutation or when a tool already provides direct authoritative readback.

**Proposed methods of implementation**

1. Map current completion candidates to the evidence they already contain.
2. Skip the extra model call when the user asked a question, the action is intermediate, or the task tool provides sufficient direct readback.
3. Retain a completion assessment only where it can materially change a user-facing claim and no stronger evidence is available.
4. Make partial/inconclusive completion easy to express: state what succeeded, what remains, and what is uncertain.
5. Measure time to first acknowledgement and end-to-end latency, alongside task success and false completion. Do not optimize latency by dropping evidence needed for a material claim.

**Acceptance criteria**

- Completion assessment is not run after every desktop action.
- Adam does not claim success solely because a tool call returned `ok` or a resource ceiling was reached.
- Latency changes are evaluated with task completion and unsupported-claim rates.

### Workstream 6 — Provider capability and native tool-call round trips

**Problem**

Provider adapters vary. A provider may lack native tool schemas or may not preserve assistant tool-call messages and matching results. Adding definitions without round-trip history support can still break multi-turn tool use.

**Solution**

Represent provider capabilities explicitly and complete native tool-call round trips for providers Adam actually uses. Keep a well-defined fallback path where appropriate.

**Justification**

Provider parity is valuable only when tied to configured use or an observed need. Fixing the CUA round-trip bottleneck should not wait on an unused provider or a broad abstraction project.

**What will change**

Inspect adapter behavior in `src/llm/provider.py`, especially the Anthropic and Gemini paths. Add or correct tool definitions, assistant call history, stable call IDs, and matching tool result history where needed.

**Proposed methods of implementation**

1. Identify the providers enabled in the current deployment and their exact tool-call formats.
2. For each configured provider, test the full sequence: model call with tool definitions; assistant tool-call message; execution; matching tool result; next model call.
3. Preserve provider call IDs and structured arguments through normalization.
4. If an adapter cannot support the full protocol, advertise that capability limitation clearly and preserve the current compatible behavior where useful.
5. Do not make strict schema parity or a universal adapter framework a prerequisite for Workstream 1.

**Acceptance criteria**

- Every configured native-tool provider can complete a multi-turn tool round trip without losing call/result correlation.
- Unsupported capabilities are explicit rather than silently approximated.
- Providers not used by the deployment do not delay the highest-priority CUA improvement.

### Workstream 7 — Active request limits, interruption, and explicit resume

**Problem**

An agent needs enough room to recover and finish normal multi-step tasks, but unbounded execution can hang or continue after the user thinks it is done. Request-local context also needs clear interruption behavior.

**Solution**

Use generous configurable request/time/token ceilings, cancellation, and state-aware no-progress protection. Keep work active only while the request is active. Return partial progress at a ceiling and require explicit resumption after the assistant has returned or the process restarts.

**Justification**

Low fixed hop/action counts create premature exits. Background CUA would act on a desktop that may have changed and whose user-visible request has ended. An explicit resume point preserves control and allows fresh observation.

**What will change**

Add or tune active-request resource bounds in the brain loop if needed. Keep continuation state request-local initially; do not add durable goals until a concrete lifecycle and explicit-resume behavior are designed.

**Proposed methods of implementation**

1. Inspect existing HTTP, model, and tool timeout layers before adding another timeout.
2. Add a configurable high-level request ceiling only where no adequate ceiling exists. Separate per-tool timeout from whole-request budget.
3. On a tool timeout, return the structured failure and uncertainty to Adam. On a whole-request ceiling, return partial progress or a blocker, never success.
4. Propagate cancellation through an active CUA sequence at a safe boundary.
5. If the user interrupts or the assistant returns, stop tool execution. On an explicit later resume, obtain a fresh screenshot/state and reassess the user's scope before acting.
6. Do not persist or reuse a stale `snapshot_id` as authority for a future request or process session.

**Acceptance criteria**

- Normal multi-step tasks are not cut off by an arbitrary low hop count.
- Per-tool timeout and whole-request exhaustion produce distinct partial/error outcomes.
- Cancellation stops further desktop mutations.
- No CUA action occurs after the assistant has returned unless the user explicitly resumes and fresh state is collected.

### Workstream 8 — Small evaluation and observability

**Problem**

An agenticness-first change could appear faster while actually reducing success, evidence quality, or user control. Without simple repeatable tasks, regressions in voice responsiveness or recovery will be hard to see.

**Solution**

Use a small set of disposable tasks with multiple valid paths and collect measures tied to the intended behavior. Compare the old and new CUA interaction pattern on the same tasks where feasible.

**Justification**

The goal is capable initiative at voice speed. A useful evaluation must measure both, rather than rewarding only more tool calls or fewer turns.

**What will change**

Add lightweight fixtures, logs, or a small evaluation script using existing project conventions. Avoid creating a large benchmark service before there is evidence it is needed.

**Proposed methods of implementation**

Use three to five repeatable disposable scenarios:

1. A routine desktop task where a short sequence is appropriate.
2. A task requiring a narrow temporary permission/chooser decision.
3. A task where the first action fails but another route exists.
4. An ambiguous target or scope requiring a focused question.
5. A task with uncertain or partial completion evidence.

Record task completion/partial completion, recovery after failure, false completion, unnecessary approval/questions, model and tool turns, time to first acknowledgement, end-to-end latency, and cancellation behavior. Report p50/p95 when sample size makes these useful. Do not prescribe a single correct click path; assess the user outcome and whether Adam's claims match its evidence.

**Acceptance criteria**

- Scenarios allow more than one valid action path.
- Latency is measured alongside completion, recovery, and evidence quality.
- Logs correlate model call, tool call, execution result, and user-facing outcome without storing unnecessary sensitive screen content.

## 6. Recommended delivery order

1. **Inspect and baseline:** Confirm the current code locations, configured providers, desktop safeguards, and a few repeatable tasks. Capture current latency/turn counts.
2. **Implement micro-sequences:** Remove the one-mutation gate and add sequential execution with per-step safeguards/results. This is the highest-priority agency and voice improvement.
3. **Correct permission guidance:** Make narrow implied access contextual while retaining actual host policy.
4. **Fix parse/validation feedback:** Stop silently substituting `{}`; preserve call IDs/origin; validate active tool arguments; return actionable errors.
5. **Improve result/repetition handling:** Distinguish outcomes and completeness; prevent blind retries; account for changed state when identifying loops.
6. **Review completion checks:** Remove unnecessary inference latency while preserving honest evidence-based answers.
7. **Close configured provider gaps:** Complete native tool round trips where deployment configuration or observed failures justify it.
8. **Add request bounds and lightweight evaluation:** Ensure cancellation, partial-at-ceiling behavior, explicit resume, and useful measurements.

Do not block steps 2–6 on implementing a universal tool registry or a framework-wide Pydantic conversion.

## 7. Review checklist for the implementation

- Does Adam choose the next strategy, or did the harness quietly encode a fixed workflow?
- Can Adam act through a reasonable sequence without an unnecessary model round trip?
- Can Adam still request a single action or stop for a new observation whenever it chooses?
- Are all actions validated against actual capabilities and authorization scope?
- Are coordinates and desktop state bound to fresh observations?
- Does every failed, partial, timed out, or cancelled call return enough information for Adam to recover or explain the blocker?
- Can a timeout lead to a duplicate side effect? If so, is uncertainty explicit and state inspected before retry?
- Does the implementation avoid interpreting an action limit, no-progress stop, or second model opinion as proof of success?
- Are provider-specific changes limited to complete round trips for providers actually used?
- Are voice latency, completion, recovery, unnecessary approvals, and claim accuracy considered together?

## 8. Patterns deliberately not prescribed

OpenAI tool calling, PydanticAI, OpenCode, Qwen Code, OpenHands, Claude Code, and Codex illustrate useful patterns around structured calls, validation, execution lifecycle, retries, context, or continuation. Their interfaces and product behaviors are not interchangeable, and a coding harness's long-running behavior does not establish reliability for voice CUA.

Use those projects as design evidence, not as a mandate to copy their architecture. In particular, this plan does not prescribe a universal controller, mandatory deterministic workflow, generic verifier, persistence service, provider abstraction, or automatic continue loop.

## Research references

- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling) — structured tool contracts, strict schemas, and parallel calls.
- [PydanticAI tools](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/) and [retry layers](https://pydantic.dev/docs/ai/core-concepts/retries/) — validation, typed failures, and distinct retry budgets.
- [OpenCode V2 tool design](https://github.com/anomalyco/opencode/blob/dev/specs/v2/tools.md) — canonical execution contract, call context, registry, cancellation.
- [Qwen Code tool loop](https://github.com/QwenLM/qwen-code/blob/main/docs/developers/tools/introduction.md) and [hooks](https://github.com/QwenLM/qwen-code/blob/main/docs/users/features/hooks.md) — execution lifecycle and recoverable stop/tool hooks.
- [OpenHands custom tools](https://github.com/OpenHands/docs/blob/main/sdk/guides/custom-tools.mdx) — typed action/observation contracts.
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code/cli-usage) and [Anthropic long-horizon guidance](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-templates-and-variables) — session continuation and context-window practices.
- [Codex Goals](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex) — durable objectives and evidence-based continuation; product behavior, not a model API guarantee.
- [A reported long-running Codex coding task](https://developers.openai.com/blog/run-long-horizon-tasks-with-codex) — one coding experiment, not CUA/voice reliability evidence.
