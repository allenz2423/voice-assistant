# Adam Tool Calling and Agentic Execution Plan

**Research date:** 2026-09-30
**Purpose:** Guide changes to Adam's tool harness so it can pursue open-ended tasks with initiative, recover from setbacks, and stay responsive in voice use.

## Decision

Adam should be a **model-led agent running inside a reliable execution harness**. The model should keep choosing what to do from the user's request, available tools, and fresh results. It must be able to revise its approach, continue through multiple tool/model turns, and decide when to act, ask, report a blocker, or stop.

“More agentic” means more initiative **within the user's requested outcome and scope**. It does not mean removing authorization, freshness, cancellation, or resource safeguards. The host enforces those real boundaries; it must not prescribe a fixed plan for every task.

```text
user request + current task state
          ↓
Adam chooses a tool/action or a useful question
          ↓
host validates mechanics and actual authorization, then executes
          ↓
Adam receives result, error, and fresh state where useful
          ↓
Adam continues, recovers, changes strategy, asks, or finishes
```

Do not force a model turn after every primitive input. Do not force a tool call on every model turn. Do not make a verifier or workflow engine choose the strategy for Adam.

## Essential boundaries

The model chooses **what to do next**. The host guarantees only the parts it can actually enforce:

| Host enforces | Adam decides |
| --- | --- |
| The tool exists; arguments parse and meet the active tool contract. | Which available tool or strategy best advances the request. |
| Calls are correlated with their results; execution can be cancelled and bounded. | Whether to retry differently, use another tool, ask, or stop. |
| Desktop input uses current-enough state and respects window, focus, and bounds checks. | When another observation is useful, based on uncertainty and consequence. |
| Consequential actions and material scope expansion follow the real authorization policy. | How the observed result bears on the user's goal. |
| The system distinguishes dispatch from observed effect. | Whether the available evidence is enough to claim completion. |

Page or dialog content can inform the task; it cannot enlarge the user's authorization. Routine actions clearly implied by a request should not trigger per-click approval. A narrow temporary access grant may proceed when the task clearly implies it. Ask when the target is materially ambiguous, access is broader or persistent, or an existing consequential-action policy requires confirmation.

## What Adam has today

This describes the inspected code as of the research date; it is not a future architecture requirement.

- `brain.py` runs a model/tool loop and returns tool results to the model. It records some desktop actions, prevents some exact repeats, speaks periodic progress, and runs a model-based outcome assessment after certain desktop completion candidates.
- The loop hard-blocks a second desktop input in one assistant response. Adam must make another model call before another input, even for a short, predictable action sequence.
- Desktop input has useful mechanical safeguards: a short-lived `snapshot_id`, active-window/focus/bounds checks where applicable, and fresh post-action observations.
- A system instruction tells Adam to stop and ask around certain chooser/permission prompts, even when a narrow temporary grant is clearly implied by the task. This should become contextual scope handling.
- Provider text-call parsers label different formats separately. The brain restricts `text_fallback` calls to a read-only list, but DSML uses another origin label and does not pass through that same restriction. Fix the inconsistent policy without banning all well-formed structured calls parsed from text.
- OpenAI-compatible and Ollama paths send native tool schemas and read native tool calls. Anthropic and Gemini currently do not send native tool schemas or parse native structured calls; adding those schemas alone would not fix history round-tripping.
- `CanonicalTool` is a Pydantic wrapper around arbitrary parameter-schema dictionaries; it does not validate incoming arguments against those schemas. Malformed argument JSON can be converted to `{}` in the brain.
- The main tool loop has no obvious shared request/time ceiling at this entry point. Task/action summaries are in memory; Adam does not autonomously resume CUA after going idle or restarting.
- Repetition signatures help stop loops, but an identical action can be legitimate after the screen changes. The current brain can block an exact previously completed action on a continuation utterance and can stop when every call in a same-turn batch has already executed. Do not turn this into a universal same-arguments-means-no-retry rule. Same-model completion assessment is useful evidence interpretation, not ground truth, and can add latency.

These are concrete findings, not a mandate to rebuild every subsystem before improving behavior.

## Desired active-task behavior

### Model-led action loop

Keep the objective and compact progress visible through the active request. After every tool result, make the result available to Adam and let it choose the next move. It may continue the current strategy, revise it, use a different tool, ask a focused question, report a blocker, or finish. Tool errors should usually return control to the model rather than end the task.

The task loop is adaptive, not a prescribed observe/act script. Some actions need a new observation first; others can safely proceed from the current state. The harness should expose useful observations and supported actions without requiring a universal semantic target system or a fixed number of stages.

### Voice-first CUA

Voice users experience each model round trip as delay. Adam should acknowledge promptly and carry out clear, ordinary work without narrating every input. It should speak progress when a task is long, a decision is needed, or the user benefits from an update.

Adam should be free to issue either a single action or a short **sequential micro-sequence** of its own choosing. The executor can perform a predictable series without asking the model to re-decide between each low-level event. Keep single-step control available whenever Adam wants it. Do not hard-code a small list of permitted sequences such as only “focus then type.”

For a micro-sequence, preserve per-action mechanics and return step outcomes. A coordinate stays bound to the screenshot that supplied it. Validate the supplied snapshot at sequence entry; after a mutation, continue only with planned steps that do not need newly inferred screen geometry (for example, typing into the field that the preceding click was intended to focus). If another step needs a new target or screenshot, return fresh state to Adam to choose it. Stop on an actual dispatch failure, an observable unexpected state that invalidates a later step, a scope/permission change that requires a new user-intent decision, cancellation, or an action for which existing policy requires confirmation. A focus/target change expected by the chosen sequence is not itself a stop condition. Proceed through narrow access clearly implied by the request without redundant approval. The executor should not claim general UI-divergence detection without an observation source that can establish it.

Keep each sequence interruptible and subject to protective per-call time, action, and input-size ceilings. These ceilings bound one execution call; on reaching one, return completed step results and fresh state where possible so Adam can decide what to do next. Do not make the call ceiling a task completion rule or impose a low fixed click count. Snapshot freshness is a real guard; it does not require user approval or prove that the whole screen is unchanged.

## Tool contracts without a rigid planner

Tool schemas help Adam understand available actions and give the host a basis for validating mechanics. They must not encode a required task route, a preferred strategy, or arbitrary narrow values that block a legitimate request.

1. Normalize provider calls while preserving provider call ID, parser origin, and raw arguments for diagnosis.
2. Resolve the tool against the active set, parse arguments without hiding errors, and validate against that tool's supported contract before dispatch.
3. Apply authorization to the normalized semantic operation, not to superficial syntax alone.
4. Execute with call correlation, cancellation, appropriate timeouts, and side-effect awareness.
5. Return a bounded result or actionable failure tied to the original call so Adam can recover.

Prefer native structured calls when a provider supports them. For textual compatibility formats, distinguish a well-defined grammar (for example, a correctly parsed DSML call) from an ambiguous free-text guess. Well-formed calls still require schema validation and the same semantic authorization as native calls. Deny or restrict only parser classes whose ambiguity creates a real risk; do not apply one blanket write ban to every non-native origin. Report parse/validation errors so Adam can choose another valid path.

Use JSON Schema as the shared validation boundary initially because Adam's built-in and YAML tools already have schema dictionaries. Add Pydantic-authored contracts where Python typing makes them clearer. Migrate incrementally, validate the active tool set, and record compatibility failures before tightening constraints broadly. Do not require a Pydantic model for every existing tool or convert arbitrary schemas into a supposedly universal strict dialect.

Provider behavior varies. Each adapter must either complete the native round trip—including tool definitions, assistant call history, call IDs, and matching result history—or advertise a clear capability limitation. Preserve safe, useful compatibility paths while doing so. Do not strand a configured provider or make provider parity a prerequisite for fixing adaptive CUA.

## Results, failures, and recovery

Keep execution outcome distinct from result completeness:

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

The result can include what was dispatched, what was observed afterward, relevant fresh state, and evidence provenance. It should tell Adam what failed and what options remain when known. Do not turn a click failure, stale observation, malformed argument, permission denial, timeout, cancellation, and internal defect into one generic “failed” message.

Avoid blind automatic retries of side effects. A timed-out write may have succeeded; return that uncertainty and let Adam inspect state before choosing whether to retry. A repeated action can be legitimate after state changes, so duplicate detection should consider state and effect, not only identical tool name/arguments. Prevent no-progress spinning, but return control to Adam with the evidence instead of terminating useful recovery at the first repetition.

Keep retry layers separate: transport retries for safe transient failures, model correction for invalid arguments when useful, and strategy changes after an action fails. “One retry” may be a scenario-specific budget, not a universal policy. Do not add a special model inference call just to revalidate locally checkable arguments; return the validation result in the normal tool loop.

## Completion and evidence

Adam should decide when to stop acting and answer. A completion check can help at the point Adam is about to claim success, but it must not run after every action or become a universal success oracle. Use direct app/artifact readback when available. A screenshot, OCR result, or model interpretation supports only claims within its scope; a second LLM call is another fallible interpretation.

If evidence supports part of the request, say what is complete and what remains. If evidence is inconclusive, Adam can gather more evidence, try a reasonable alternative, ask, or stop with an honest uncertainty report. Reaching a time/token/tool ceiling is never evidence of success.

## Continuity, interruption, and resume

Coding harnesses show the value of long-running work: an active objective can remain visible while the model iterates through tools, observes results, and repairs failures. Codex Goals add a product-level objective and evidence-based continuation at safe boundaries; Anthropic's long-horizon guidance discusses context-window state management. These examples inspire persistence; they do not make their mechanisms requirements for Adam. A long Codex coding run is not evidence of CUA/voice reliability.

For Adam, first preserve goal, authorized scope, compact progress, and useful evidence across tool/model calls in the active request. If a provider request fails while the user task is still active, use a bounded transport retry or configured fallback when it is safe. Do not automatically replay a tool side effect when it may already have executed. If recovery fails or the user interrupts, report partial progress and retain only enough summary for explicit resumption. Do not resume desktop actions in the background after the assistant has returned. On explicit resume, obtain a fresh desktop observation and re-evaluate the current target and permission scope. Never persist or reuse an old `snapshot_id` as authority to act.

Use generous configurable elapsed-time/request/token ceilings, cancellation, and state-aware no-progress protection as runaway controls. Avoid low fixed action/hop limits that end normal tasks early. When a ceiling is reached, return partial progress or a blocker, never success. After process restart, explicit resume is required; do not silently continue against a desktop that may have changed.

## Implementation sequence

### 1. Remove the largest agency bottleneck

Replace the blanket one-input-per-model-decision gate with model-selected, bounded sequential CUA micro-sequences. Preserve snapshot/window/focus/bounds checks, consequential-action policy, cancellation, and stop-on-divergence behavior. Make chooser handling contextual so clearly implied narrow temporary access does not cause a redundant approval turn.

### 2. Fix call recovery at the shared boundary

Preserve call IDs and parser provenance, reject malformed JSON explicitly instead of substituting `{}`, and return concise validation errors to Adam. Apply consistent trust handling by parser type: support well-formed structured formats under schema validation and semantic authorization; restrict genuinely ambiguous free-text guesses. Do this without adding extra model calls for local checks.

### 3. Measure agency and voice responsiveness

Use three to five repeatable, disposable tasks with multiple valid paths. Include an ordinary desktop task, one permission/scope case, one failed action with a recovery route, and an ambiguity/blocker case. Measure:

- task completion and partial completion;
- recovery after tool/action failure;
- false completion and unsupported claims;
- unnecessary approval/questions;
- model/tool turns and user interruptions;
- time to first acknowledgement and end-to-end latency (including p50/p95 as samples grow).

Compare adaptive checkpoints with inspect-after-every-input on the same tasks. Do not improve speed by skipping evidence needed for a material claim.

### 4. Close measured protocol gaps

Prioritize native provider round trips for providers actually configured in Adam. Add schema constraints, typed results, registry structure, or adapter work where traces reveal actual failures. Preserve useful capabilities while migrating; use compatibility reporting rather than silently dropping tools. Keep the existing loop until a measured problem justifies a broader harness change.

### 5. Optimize concurrency and long-running work

Run independent read-only calls concurrently when their dependencies permit. Keep desktop state mutations sequential, including inside a micro-sequence. Add resumable task state only when its lifecycle and explicit-resume behavior are clear. Do not build background Goals, generic verification services, universal adapter frameworks, or a large benchmark platform as the starting point.

## What to avoid

- A mandatory model round trip after every click, keypress, or other primitive input.
- Tool choice forced on every turn, preventing Adam from answering or choosing another path.
- Per-click approval or a blanket stop at every chooser.
- A fixed UI script, generic task predicate, or universal verifier that dictates how work proceeds.
- Blanket mutation denial for all text-parsed calls, independent of parser structure and semantic authorization.
- Strict-schema migration across every tool/provider before checking compatibility and measured need.
- Automatic retries of possibly completed side effects.
- A low global action/turn cap or treating budget exhaustion as completion.
- Mandatory LLM verification after each action or extra model calls for locally checkable conditions.
- Automatic background CUA continuation after the user-facing request ends.

## Research references

These sources illustrate patterns, not requirements to copy wholesale:

- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling) — structured tool contracts, strict schemas, and parallel calls.
- [PydanticAI tools](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/) and [retry layers](https://pydantic.dev/docs/ai/core-concepts/retries/) — validation, typed failures, and distinct retry budgets.
- [OpenCode V2 tool design](https://github.com/anomalyco/opencode/blob/dev/specs/v2/tools.md) — canonical execution contract, call context, registry, cancellation.
- [Qwen Code tool loop](https://github.com/QwenLM/qwen-code/blob/main/docs/developers/tools/introduction.md) and [hooks](https://github.com/QwenLM/qwen-code/blob/main/docs/users/features/hooks.md) — execution lifecycle and recoverable stop/tool hooks.
- [OpenHands custom tools](https://github.com/OpenHands/docs/blob/main/sdk/guides/custom-tools.mdx) — typed action/observation contracts.
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code/cli-usage) and [Anthropic long-horizon guidance](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-templates-and-variables) — session continuation and context-window practices.
- [Codex Goals](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex) — durable objectives and evidence-based continuation; product behavior, not a model API guarantee.
- [A reported long-running Codex coding task](https://developers.openai.com/blog/run-long-horizon-tasks-with-codex) — one coding experiment, not CUA/voice reliability evidence.
