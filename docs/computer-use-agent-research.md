# Computer-Use Agents: Industry Research and Implications for Adam

**Research snapshot:** 2026-09-30
**Scope:** Public product documentation, open-source repositories, benchmark papers, and the current Adam implementation in this repository. Product capabilities and model names change quickly; links below are the source of truth for current versions. Vendor performance claims are identified as such. This document does not claim access to private internal systems or training details.

## Executive summary

Computer-use agents (CUAs) connect a model to an interactive computer environment. Their capability is not just the vision model. In practice, performance comes from the combination of:

1. **Perception:** screenshots, OCR, accessibility trees, DOM, app APIs, and relevant task state.
2. **Decision-making:** interpreting the request, selecting a target, deciding what to do next, and recovering when the interface differs from expectation.
3. **Execution:** a driver that translates actions into reliable input and preserves the live computer session.
4. **Verification:** checking that the requested application state actually changed, rather than treating an issued click or the model's “done” response as success.
5. **Operations:** isolation, permissions, cancellation, bounded execution, logs, and an evaluation suite that catches regressions.

The strongest public approaches do not rely on pixels alone in every case. They make screenshots a general fallback, while using structured browser or application state and code/API tools when those provide better evidence or control. They also treat the computer as a persistent runtime separate from the model conversation.

Adam already has a screenshot-guided controller with snapshot freshness checks, focus/window validation, serialized input, and an outcome-verification pass. Its OBS case shows the remaining central weakness: a task can make visible progress while its actual requested end state is still false. The highest-value next step is a task benchmark with authoritative success checks, followed by stronger task-scoped observations and model comparisons. Replacing the model alone is not a sufficient fix.

## What counts as a CUA?

“Computer use” covers several related but different capabilities:

- **Visual desktop control:** The model receives screenshots and emits clicks, key presses, typing, scrolling, or drags. It can work across unfamiliar native applications, but must infer layout and state from pixels.
- **Browser automation:** The agent uses a browser through screenshots, DOM/accessibility data, a browser automation API, or a mixture. Browser-only agents can often use richer state than whole-desktop agents.
- **Semantic GUI automation:** The agent acts on named controls and accessibility elements instead of coordinates. It is easier to validate when the application exposes good semantics, but coverage varies by toolkit and app.
- **App/API automation:** The agent calls a direct application API or executes code. This is usually the most deterministic path, but may be unavailable, over-privileged, or insufficient when the user specifically needs the visible UI path tested.
- **Hybrid computer-use agents:** A planner chooses among APIs, browser tools, shell/code, accessibility, and visual actions during one task. This increasingly describes production agent systems better than a pure screenshot loop.

These are not mutually exclusive. A robust system chooses the narrowest useful interface for each subtask and retains visual interaction for controls that have no reliable semantic or API surface.

## Common system architecture

```text
User goal
   ↓
Task state: requested outcomes + evidence needed
   ↓
Orchestrator/model ←→ observations (screen, DOM, accessibility, app state)
   ↓                                      ↑
Policy and action validation              │ fresh result / state diff
   ↓                                      │
Computer driver → browser / desktop / VM ─┘
   ↓
Independent task verifier → complete / incomplete / blocked / uncertain
```

The model conversation and the computer runtime are separate state machines. Conversation history does not itself preserve a logged-in browser, an open document, a running VM, or code variables; the host integration must keep the same session alive and return observations from it. OpenAI documents this explicitly and describes both a structured computer tool and code-execution integration; its current guidance recommends code execution for GPT-6 Astra while retaining structured computer actions as an alternative. [OpenAI computer-use guide](https://developers.openai.com/api/docs/guides/tools-computer-use)

### Perception modes

| Mode | What the agent sees | Strengths | Typical failure modes |
|---|---|---|---|
| Screenshot only | Pixels from the full display or a window | General across applications; resembles human use | Small text, coordinate errors, hidden state, visual ambiguity, prompt injection in page content |
| OCR / visual parser | Text boxes, icons, detected regions, sometimes captions | Helps a smaller model localize controls and text | Missed/false detections; parser output can be stale; regions are not necessarily actionable controls |
| Accessibility tree | Roles, labels, values, enabled/focused state and actions | Named targeting and direct actions; can be more deterministic | Incomplete or misleading application support; large trees; inaccessible custom-drawn controls |
| Browser DOM / accessibility snapshot | Page structure, labels, links, fields, URL and selected state | Strong for web content and semantic operations | Does not automatically expose browser chrome; can differ from rendered behavior; remote debugging grants sensitive access |
| App-specific API / direct state | Authoritative domain state and operations | Best verification and often reliable execution | Limited to integrated apps; can bypass the UI behavior the task intended to test |
| Hybrid | Task-relevant combination of the above | Uses the strongest evidence available at each step | More integration work; state/provenance need careful reconciliation |

Microsoft's OmniParser is an example of visual parsing aimed at pure-vision GUI agents: it detects/captions visible interface elements so a model can target regions more easily. It is a perception aid, not a replacement for a task planner, action policy, or outcome verifier. Its repository also documents different licensing for some model artifacts, which matters before packaging or redistribution. [Microsoft OmniParser](https://github.com/microsoft/OmniParser)

## How major public systems approach computer use

### OpenAI

OpenAI's public API documentation defines a harness boundary: the model requests computer actions, the integrator executes them in a browser/desktop runtime, and the integrator returns a new screenshot. The structured tool can request ordered action arrays; documentation shows batching small sequences and returning a screenshot so the model can inspect the result. A second integration style lets the model generate code for a controlled execution environment, allowing loops and conditional logic within one call. The host must retain runtime state across turns. [OpenAI computer-use guide](https://developers.openai.com/api/docs/guides/tools-computer-use)

The current model guidance describes GPT-6 Astra as having strong computer-use capability and recommends code execution for this model, while the computer-use guide also supports structured actions. The model guidance recommends an isolated browser or VM and keeping a human in the loop for high-impact actions. The guide's safety section calls for restricting the environment, treating screen content as untrusted, confirming consequential actions, bounding runs, enabling cancellation, and checking the actual application outcome. [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model) · [OpenAI computer-use safety guidance](https://developers.openai.com/api/docs/guides/tools-computer-use#run-safely)

**What is and is not public:** The API docs establish tool behavior and current integration patterns. They do not disclose all model training, internal product orchestration, or private runtime implementation. It is therefore fair to say that a capable vision/reasoning model is a key part of OpenAI's public CUA interface; it is not supported by public API docs to reduce the entire system to “just a very good vision model.”

### Anthropic Claude computer use

Anthropic's current platform docs expose a versioned computer toolset with screenshot, zoom, mouse, keyboard, typing, and wait operations. The integrator runs these actions in an environment it controls. Claude may return multiple actions in one response; the host executes them sequentially, stops the remainder of the batch on the first failure, reports which actions did and did not run, and can end the batch with a screenshot. The docs recommend an isolated container/VM, minimizing access to sensitive data, domain allowlists, and human confirmation for consequential steps. Anthropic also documents screenshot prompt-injection classifiers and warns that they are not a substitute for isolation. [Anthropic computer-use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)

This is a useful reference for reducing unnecessary model round trips without running a long, unchecked sequence: keep batches short, ordered, failure-aware, and followed by fresh visual evidence. Anthropic's original October 2024 announcement reported 14.9% on OSWorld in screenshot-only conditions and 22.0% when given more steps for Claude 3.5 Sonnet. Those are historical, vendor-reported results for that model and setup, not a current comparison with today's systems. [Anthropic 2024 announcement](https://www.anthropic.com/news/3-5-models-and-computer-use)

### Google Project Mariner and Gemini

Google described Project Mariner as a web-focused computer-use research prototype and later said its capabilities would be brought to developers through the Gemini API. Google's 2025 I/O keynote also described multitasking and “teach and repeat,” where users demonstrate a task and the system learns plans for similar tasks. Google frames agents more broadly as advanced models plus tools, with computer use only one part of that ecosystem. The keynote is a dated product announcement; check current Gemini API documentation for present availability and interfaces. [Google I/O 2025 keynote](https://blog.google/innovation-and-ai/technology/ai/io-2025-keynote/)

The transferable design idea is reusable task knowledge and coordination, not simply more screenshot resolution. For Adam, any learned routine should still be grounded against current state rather than replaying stale coordinates or assuming the application has not changed.

### Amazon Nova Act

Nova Act is a browser-workflow service combining a custom computer-use model with a Python SDK and workflow lifecycle. AWS describes agents that mix natural language with Python, API/tool calls, and browser interaction, with deployment and monitoring integrations such as AgentCore, CloudWatch, and IAM. AWS's service card describes screenshot-based context, iterative reasoning and action, guardrail validation, and execution translated through Playwright. It explicitly warns that the same prompt can lead to different actions when external UI conditions vary, recommends use-case-specific testing, and advises domain restrictions and least-needed tools. [Nova Act product page](https://aws.amazon.com/nova/act/) · [Nova Act AI Service Card](https://docs.aws.amazon.com/ai/responsible-ai/nova-act/overview.html)

AWS also describes a 100-step maximum and 30-minute browser session in its service card. Those are Nova Act product constraints, not general CUA limits. Its service card is valuable because it makes operational caveats and non-determinism explicit instead of presenting only a successful demo.

### Microsoft UFO / UFO² / UFO³

Microsoft's open-source UFO family focuses on Windows desktop automation and multi-device orchestration. The current repository distinguishes UFO², a Windows desktop agent with GUI plus API actions and deep OS integration, from UFO³, a multi-device orchestration framework that decomposes tasks into DAGs and coordinates heterogeneous agents. [Microsoft UFO repository](https://github.com/microsoft/UFO)

This illustrates a common enterprise split: a task-level orchestrator delegates to a device/app agent, while the device agent uses platform-native semantics and visual fallback. It also illustrates the maintenance cost of deep platform integration: coverage and reliability may depend on application accessibility support, Windows APIs, platform versions, and app-specific behavior.

### ByteDance UI-TARS

UI-TARS is a useful counterpoint to API-first systems because it is explicitly a native GUI agent model. Its paper describes screenshot input and mouse/keyboard output, large-scale GUI data, a unified action space, grounding, multi-step reasoning, and reinforcement learning. The UI-TARS-2 report describes a data-generation flywheel, multi-turn RL, a hybrid GUI environment with terminal/filesystem capabilities, and a unified sandbox for rollouts. These are the authors' published research descriptions; they do not imply that Adam needs to reproduce the training stack to benefit from the architecture. [UI-TARS paper](https://arxiv.org/abs/2501.12326) · [UI-TARS-2 technical report](https://arxiv.org/abs/2509.02544) · [UI-TARS repository](https://github.com/bytedance/UI-TARS)

This supports part of the “strong vision model” hypothesis: a purpose-trained GUI model can learn target grounding and action conventions that a generic vision model may not handle as well. It also supports the broader conclusion: their public account includes action traces, training data, reasoning, repeated environment interaction, and sandboxed rollouts, not only a better image encoder. Reported benchmark rows in the repository are model/team-reported and must be interpreted with their task versions, action budgets, and harness details.

### Browser Use and browser-focused agents

Browser Use offers an open-source Python agent, a CLI for existing agents, and hosted browser-agent infrastructure. This is a browser-specific design point: the system can specialize in browser sessions and account profiles rather than supporting arbitrary native desktop apps. Its repository makes a further distinction between the agent/model and the browser infrastructure. [Browser Use repository](https://github.com/browser-use/browser-use) · [Browser Use developer offerings](https://browser-use.com/developers)

For Adam, browser-focused agents are useful for comparing session persistence, DOM-backed target selection, browser profiles, recording, and cloud isolation. Their capabilities should not be conflated with native desktop control; a browser driver will not necessarily solve file pickers, screen-capture dialogs, or controls outside the browser content area.

### trycua/cua

Cua is not one model-led agent. It separates the agent/model from the computer and automation layer. The repository currently groups:

- **Cua Driver:** Cross-platform operation and inspection for native apps/browser windows, with CLI, MCP, and typed SDK interfaces. Background input is platform-dependent.
- **Cua Sandbox / Fleets:** Fresh isolated computers, either local or provisioned in cloud fleets; agent code can combine shell/API work and visible UI use inside the same computer.
- **Lume:** Local macOS/Linux VM lifecycle on Apple Silicon.
- **Cua Bench:** Repeatable tasks, environment setup, agent adapters, evaluators, and trajectory capture.
- **CUA-S1:** Specialist model research for bounded form decisions over structured elements; the README positions it as a complement to a general agent, not a general planner.

The README makes the key boundary explicit: bring your own agent/model, while Cua supplies the computer, driver, and evaluation machinery. It also distinguishes the MIT driver from optional perception packages with separate licensing conditions. [Cua repository](https://github.com/trycua/cua) · [Cua Driver README](https://github.com/trycua/cua/blob/main/libs/cua-driver/README.md) · [Cua computer-use concepts](https://github.com/trycua/cua/blob/main/docs/content/docs/concepts/what-is-computer-use.mdx)

For Adam, Cua is most relevant as an architectural and evaluation reference. Adopting the driver itself would require checking Linux/Wayland behavior, focus policy, permission semantics, compatibility with Adam's controller, and optional perception licensing. Adam already has desktop-specific code; replacing it without a measured win could add complexity without improving task success. OpenAI's code-execution recommendation is also not automatically the best fit for Adam: letting a model generate runnable code can be powerful inside a tightly isolated runtime, while Adam's local desktop integration benefits from keeping input behind its typed, snapshot-validated controller. This is an architecture inference from the documented integration options and Adam's current trust boundary, not an OpenAI claim.

## What the research says about capability and evaluation

### Useful benchmark families

| Benchmark family | What it probes | Why it matters | Caveat |
|---|---|---|---|
| [OSWorld](https://github.com/xlang-ai/OSWorld) / OSWorld-Verified | Real desktop applications, cross-app tasks, and executable task-specific evaluation | A strong baseline for general desktop control and end-state correctness | Results depend on release, model, prompt, tools, VM, step budget, and scoring version; use the current verified release |
| [OSWorld 2.0 / 2.1](https://github.com/xlang-ai/OSWorld-V2) | Long-horizon professional workflows and harder interaction conditions | Surfaces constraint tracking, changing environments, hidden state, recovery, and verification | Newer benchmark release and task distribution; do not compare raw scores to older OSWorld without careful normalization |
| [WebArena](https://github.com/web-arena-x/webarena) | Multi-step tasks in realistic self-hosted web applications | Reproducible browser workflow measurement | Website/API-specific and increasingly vulnerable to benchmark saturation or task shortcuts |
| [VisualWebArena](https://github.com/web-arena-x/visualwebarena) | Visually grounded web tasks | Tests screenshot grounding beyond DOM-only navigation | Not equivalent to a native desktop benchmark |
| [WorkArena / BrowserGym](https://github.com/ServiceNow/WorkArena) | Enterprise software tasks and extensible browser-agent environments | Tests knowledge-work workflows and supports varied observation/action designs | Tasks may depend on benchmark-specific apps and setup |
| [Cua-Bench](https://github.com/trycua/cua) | User-defined repeatable computer-use tasks with evaluators and environment providers | Useful for building a local Adam-specific regression suite | Results are only as representative and robust as its tasks and graders |

OSWorld's maintainers announced OSWorld-Verified with corrections to task/evaluation issues and recommend comparing against the updated version. The current OSWorld 2 repository also versions task files and assets together. This is a practical reminder to pin benchmark releases and preserve evaluation artifacts. [OSWorld repository](https://github.com/xlang-ai/OSWorld) · [OSWorld 2 repository](https://github.com/xlang-ai/OSWorld-V2)

### How to interpret reported scores

- Verify that two figures use the same task set, software images, task initialization, action interface, screenshot resolution, step limits, retry policy, and evaluator version.
- Separate model-only comparisons from end-to-end agent comparisons. A model's result can shift substantially with tool wrappers, access to DOM or shell, and verification logic.
- Report success and failure reasons, not only an aggregate percentage. For Adam, record target-selection mistakes, stale-state rejection, missed dialogs, premature completion, model/API errors, and verifier failures.
- Repeat nondeterministic workflows. AWS, for example, reports averages over multiple runs for Nova Act evaluations and explicitly notes run-to-run UI variation in its service card.
- Keep benchmark tasks disjoint from prompt tuning where possible. A tuned task is useful for regression but should not be presented as an unbiased general capability score.
- Prefer graders that inspect final application state or saved artifacts. A screenshot judge alone can miss hidden state; a model self-report is weaker still.

Recent long-horizon benchmark work emphasizes that agents fail not only on basic clicking, but also on maintaining constraints, integrating information that arrives mid-task, inferring hidden state, deciding when to ask, and verifying completion. These are directly relevant to Adam's OBS failure. [OSWorld 2.0 paper](https://arxiv.org/abs/2606.29537)

## Adam today

This section describes the repository implementation and is separate from the design intentions in [computer-use-plan.md](computer-use-plan.md).

### Implemented strengths

- `src/tools/computer_control.py` captures screenshots and issues short-lived snapshot IDs. It binds action validity to the current screen/window identity and rejects stale snapshot use.
- The controller stabilizes focus before observations, checks bounds/window context, serializes input, and takes a fresh screenshot after action calls.
- `src/llm/brain.py` instructs the model to keep an outcome checklist, observe after actions, and not equate a tool dispatch with completion. It limits desktop input to one per model decision and includes a computer outcome-verification path.
- `observe_desktop` and browser-related tools provide additional observation paths; optional OCR/visual-grounding components can help find visible targets.
- The controller supports Linux desktop paths including X11/Wayland where available, giving Adam access to the user's actual native desktop rather than only a hosted browser.

### Known gaps and risks

- **The one-input-per-model-decision gate is cautious but creates extra turns and latency.** OpenAI and Anthropic both document short ordered action batches followed by fresh evidence. Adam could evaluate similarly bounded batches for low-risk, reversible sequences while retaining checks around ambiguous or consequential operations.
- **Visual targeting remains sensitive to model quality and screenshot scale.** Grounding tools can suggest regions, but detection is not proof that a region is the correct target. Small dialogs, scaling, and multi-monitor coordinate spaces remain failure points.
- **Observations can be too broad.** The OBS findings record an inventory with many unrelated windows and large accessibility output. A smaller model benefits from one task-scoped active window/dialog plus a concise delta.
- **Observation surfaces are not yet one unified, provenance-aware state model.** The plan proposes selecting among DOM, accessibility, app APIs, keyboard, OCR, and pixels per target; treat that as a target architecture, not a claim that every adapter is already integrated.
- **Task-level verification is the most important gap.** In the OBS case, “recording active” did not prove the intended monitor was selected or that the recording captured the desired display. A successful input or plausible screenshot is not enough when authoritative application state can be checked.
- **The local desktop expands the blast radius.** Adam acts in the user's session, where unrelated files, accounts, and windows may be available. Isolation and task-scoped permissions should be considered when the use case allows them.

See [computer-use-obs-findings.md](computer-use-obs-findings.md) for the observed case and [computer-use-plan.md](computer-use-plan.md) for the proposed adapter/controller design.

## Recommended direction for Adam

These are research recommendations, not changes made by this document.

### Priority 1: Build an Adam CUA evaluation set

Start with 20–40 representative tasks and exact initial states. Include browser navigation, dialogs, file selection, text entry, window management, media/recording, and multi-step workflows. For every task define:

- the request and starting state;
- explicit success predicates;
- allowed tools and whether a destructive/consequential step is involved;
- a machine-checkable verifier where possible;
- an acceptable failure or “ask user” state;
- a trajectory capture policy that avoids logging secrets by default.

Run each model/harness change on the same pinned task set, repeat nondeterministic tasks, and track success, false completion, recovery, action count, latency, and cost. Add the OBS screen-share selection task as a regression case. Cua-Bench is a practical reference for packaging task, environment, adapter, and evaluator separately.

### Priority 2: Make success predicates first-class

For each user task, represent requested end states and their evidence. Example for OBS:

```text
requested state: recording captures the requested monitor
evidence: OBS recording active AND source/display selection equals requested monitor
incomplete: chooser still open OR selected display does not match
uncertain: state cannot be read from supported app/API/UI evidence
```

Use app APIs or structured state for the verifier when available, and visual evidence as a fallback. Return `complete`, `incomplete`, `blocked`, or `uncertain` with supporting facts. The verifier should be independent of the agent's natural-language declaration.

### Priority 3: Make observations task-scoped and hybrid

Keep screenshot support, but pair it with relevant window metadata, focused dialog information, accessibility/DOM nodes when available, and compact state deltas. Do not send every open window or a full unfiltered accessibility tree if the task is clearly about one app. Preserve provenance and freshness for each actionable target. Keep browser chrome, page content, and native dialogs distinct because no single adapter necessarily covers all three.

### Priority 4: Compare models and visual grounders experimentally

Use the exact same Adam runtime and tasks to compare:

- current model;
- one or more higher-capability vision/reasoning models;
- a vision-grounding model/tool such as the existing OmniParser path;
- optionally, an OCR/accessibility-enhanced observation.

Change one variable at a time. If success improves only with a stronger model, model quality is a major bottleneck. If models see the correct target but the action is lost, focus shifts, or “done” is premature, fix the harness/verifier. Avoid relying on model branding or published benchmark scores as a substitute for Adam's own results.

### Priority 5: Prototype safe, short action batches

The current gate ensures a fresh model decision after each input. Test a bounded sequence interface for only obvious, reversible actions, such as focus a clearly identified field then type the requested text. The executor should:

- validate the initial snapshot and action scope;
- cap action count and elapsed time;
- execute sequentially and stop on the first failure;
- prevent a batch from crossing a consequential-action boundary;
- take and return a fresh screenshot/state after the batch;
- report per-action outcomes so the agent can recover.

Keep a one-action mode for ambiguous UI state and critical transitions. Measure task completion and latency before changing the default.

### Priority 6: Clarify the driver boundary

Keep model reasoning separate from input injection, snapshots, focus, permissions, and target freshness. Cua Driver offers a useful cross-platform reference and may be worth a disposable Linux compatibility prototype; evaluate it against Adam's Wayland session before considering adoption. A local backend replacement should preserve Adam's existing snapshot and policy guarantees.

### Priority 7: Improve autonomy through scoped capability, not blanket friction

High autonomy does not require asking the user to approve every click. Use permissions aligned to impact: reading and reversible navigation can usually proceed; sending data, purchases, destructive edits, credential entry, permission grants, and other hard-to-reverse actions should remain gated by user intent/policy. Prefer a dedicated profile or isolated VM for tasks that do not require the user's live desktop. Add cancellation, action/time limits, domain/app allowlists where suitable, and safe handling of prompt injection. [OpenAI safety guidance](https://developers.openai.com/api/docs/guides/tools-computer-use#run-safely) · [Anthropic security guidance](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool#security-considerations) · [Cua safety and isolation](https://github.com/trycua/cua/blob/main/docs/content/docs/concepts/what-is-computer-use.mdx)

### Priority 8: Record evidence useful for debugging

Store structured run traces with task ID, requested outcome, adapter/model versions, snapshot IDs, action type, target provenance, timestamps, tool result, and verifier result. Redact typed secrets and avoid retaining full screenshots/page text by default. Keep enough evidence to explain why a task was called complete, incomplete, or uncertain. Correlate the user-facing answer with the verifier result.

## Suggested decision matrix

| Task situation | Preferred method | Why |
|---|---|---|
| A stable app API can read/write the needed state | App-specific API, with UI verification if the request concerns visible UX | More deterministic and easier to verify |
| A web page has accessible DOM and the task is website-specific | Browser DOM/accessibility + browser actions | Semantic targets and page state are more precise |
| Native app exposes good accessibility actions | Accessibility tree + fresh screenshot | Named controls reduce coordinate guessing |
| Custom/canvas UI or accessibility is poor | Screenshot/vision with bounded actions | Visual interaction is the general fallback |
| Task outcome has high impact or is hard to reverse | Least-privilege environment plus policy gate/confirmation | Model judgment alone is not an authorization boundary |
| Task does not need the user's personal desktop state | Dedicated browser profile, container, or VM | Reduces accidental access and blast radius |
| Task success is ambiguous | Ask a targeted question or report uncertainty | Avoid silently choosing or claiming success |

## Bottom line

The evidence does not support treating CUA as “vision model plus mouse.” Vision gives generality, especially for unfamiliar UI, but reliable systems are built around persistent computer runtimes, typed action interfaces, fresh observations, short recovery loops, semantic state where available, independent verification, and repeatable evaluations. OpenAI, Anthropic, Google, AWS, Microsoft, and Cua expose different pieces of this stack; no public evidence justifies assuming that one model alone accounts for their end-to-end performance.

For Adam, make success measurable first. Then use the task traces to decide whether the biggest remaining limiter is model capability, observation quality, action policy/latency, driver reliability, or verification. This sequence is more likely to yield a durable improvement than changing all of them at once.

## Primary sources and further reading

### Product and implementation documentation

- [OpenAI Computer use guide](https://developers.openai.com/api/docs/guides/tools-computer-use)
- [OpenAI model guidance: GPT-6 family](https://developers.openai.com/api/docs/guides/latest-model)
- [Anthropic Computer use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)
- [Anthropic computer-use announcement (2024)](https://www.anthropic.com/news/3-5-models-and-computer-use)
- [Google I/O 2025 keynote: Project Mariner](https://blog.google/innovation-and-ai/technology/ai/io-2025-keynote/)
- [Amazon Nova Act product page](https://aws.amazon.com/nova/act/)
- [Amazon Nova Act AI Service Card](https://docs.aws.amazon.com/ai/responsible-ai/nova-act/overview.html)
- [Microsoft UFO repository](https://github.com/microsoft/UFO)
- [Microsoft OmniParser repository](https://github.com/microsoft/OmniParser)
- [UI-TARS paper](https://arxiv.org/abs/2501.12326)
- [UI-TARS-2 technical report](https://arxiv.org/abs/2509.02544)
- [UI-TARS repository](https://github.com/bytedance/UI-TARS)
- [Browser Use repository](https://github.com/browser-use/browser-use)
- [Browser Use developer offerings](https://browser-use.com/developers)
- [trycua/cua repository](https://github.com/trycua/cua)
- [Cua Driver README](https://github.com/trycua/cua/blob/main/libs/cua-driver/README.md)
- [Cua computer-use concepts](https://github.com/trycua/cua/blob/main/docs/content/docs/concepts/what-is-computer-use.mdx)

### Benchmarks and research

- [OSWorld repository and OSWorld-Verified updates](https://github.com/xlang-ai/OSWorld)
- [OSWorld paper](https://arxiv.org/abs/2404.07972)
- [OSWorld 2 repository](https://github.com/xlang-ai/OSWorld-V2)
- [OSWorld 2.0 paper](https://arxiv.org/abs/2606.29537)
- [WebArena repository](https://github.com/web-arena-x/webarena)
- [VisualWebArena paper](https://arxiv.org/abs/2401.13649)
- [BrowserGym and WorkArena paper](https://openreview.net/forum?id=BRfqYrikdo)
- [Cua-Bench](https://github.com/trycua/cua)
- [OmniParser technical report](https://arxiv.org/abs/2408.00203)

### Adam repository references

- [Adam computer-use plan](computer-use-plan.md)
- [OBS computer-use findings](computer-use-obs-findings.md)
- [Jev OCR desktop-control notes](jev-ocr-desktop-control.md)
- [`computer_control.py`](../src/tools/computer_control.py)
- [`brain.py`](../src/llm/brain.py)
