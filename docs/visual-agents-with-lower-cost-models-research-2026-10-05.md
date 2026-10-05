# Visual computer use with lower-cost models — research review

**Date:** October 5, 2026

**Question:** How do companies and research groups help models that struggle with raw screenshots operate real interfaces? What should Adam test next?

**Scope:** Public papers, official product documentation, and Adam's existing measurements. Published benchmark results are evidence about the tested setup, not expected Adam performance.

## Executive assessment

The strongest recurring tactic is to **make a UI element easier to identify and bind to an action**. Systems expose accessible controls or DOM references when available, add OCR and detected regions for visual-only controls, focus or zoom the relevant area, constrain the action format, and observe the resulting state. They do not rely on a smaller general model to infer every pixel coordinate from a full desktop image. This is a synthesis of the sources below, not a claim that any company has solved general computer use.

For Adam, the highest-value near-term experiment is a **matched comparison of evidence modes using the same model and task**: screenshot alone, current window OCR, accessibility/DOM where available, and OCR plus OmniParser regions. Keep the visual detector lazy. Measure grounding correctness and complete task success separately, together with extra latency and process-tree memory. Do not assume OmniParser wins: Microsoft's own UFO configuration recommends native UI automation as its default and calls its OmniParser backend slower; the published web-agent SeeAct study found that region marks alone were weaker than grounding that used HTML and visual information. [UFO configuration](https://github.com/microsoft/UFO/blob/main/documents/docs/configuration/system/system_config.md), [SeeAct](https://arxiv.org/abs/2401.01614).

## The problem is several problems

| Failure | What a visual aid can change | What it cannot establish |
| --- | --- | --- |
| **Reading:** text or a state label is illegible in a whole-screen image. | OCR, DOM text, accessibility values, or a crop/zoom can expose it. | That the correct application or date is being read. |
| **Grounding:** the model knows the target in words but clicks the wrong place. | Element references, bounding boxes, numbered regions, and coordinate transforms narrow the choice. | That a detected region is actually actionable. |
| **Planning:** the next step depends on app state or task intent. | A compact observation and action history can improve the choice. | That a low-cost model can plan a long task without error. |
| **Execution:** focus, scale, layout, or loading changes after observation. | Fresh snapshots, focus checks, typed action schemas, and stop-on-failure semantics can catch some errors. | That the app accepted a dispatched click. |
| **Verification:** a control was used but the requested outcome is uncertain. | Re-observe the relevant state, compare text/state, or check an application artifact. | That a screenshot by itself proves hidden state or task completion. |

The original [OSWorld paper](https://arxiv.org/abs/2404.07972) introduced 369 executable desktop tasks and found its then-best baseline completed 12.24% versus over 72% for humans. This is historical baseline evidence, not a current leaderboard. Its analysis identified grounding and operational knowledge as major failure sources. [SeeClick](https://arxiv.org/abs/2401.10935) similarly argues that GUI grounding deserves direct evaluation, separate from full task execution.

## Documented company and research approaches

### 1. Microsoft: structured controls first; visual detection for gaps

Microsoft's [UFO² control detection documentation](https://github.com/microsoft/UFO/blob/main/documents/docs/ufo2/core_features/control_detection/overview.md) offers three backends: Windows UI Automation (UIA), OmniParser, and a hybrid that merges overlapping detections. Its [configuration guide](https://github.com/microsoft/UFO/blob/main/documents/docs/configuration/system/system_config.md) calls UIA the recommended default and describes the visual backend as slower and GPU dependent. The [AppAgent documentation](https://github.com/microsoft/UFO/blob/main/documents/docs/ufo2/app_agent/overview.md) shows screenshots plus control data, annotated control IDs, and actions on identified controls or application APIs. This is a practical answer to weak coordinate prediction: let a model choose among named controls and let the executor bind the choice.

[OmniParser's paper](https://arxiv.org/abs/2408.00203) targets cases where normal UI structure is missing: it detects interactable regions and generates icon descriptions, then supplies that representation to a vision-language agent. Microsoft's [repository](https://github.com/microsoft/OmniParser) describes later detector revisions that improve small-icon detection and interactability prediction. Paper improvements on ScreenSpot, Mind2Web, and AITW were measured with the paper's models, prompts, and tasks. They show that screen parsing can improve grounding, not that every OCR-plus-OmniParser stack is faster or more accurate than accessible controls on Adam.

**Adam implication:** use Linux AT-SPI or browser structure when the target is well represented; test a visual detector for missing custom controls. Windows UIA results do not transfer directly to Linux AT-SPI. Treat disagreement between the accessibility tree and visual detector as an ambiguity to inspect, not a reason to merge everything into a large prompt.

### 2. Browserbase Stagehand: separate discovery, action, and extraction

[Stagehand's official documentation](https://github.com/browserbase/stagehand/blob/main/packages/docs/v3/first-steps/introduction.mdx) exposes `observe`, `act`, and `extract`, alongside a full agent mode. Its [`act` reference](https://github.com/browserbase/stagehand/blob/main/packages/docs/v3/references/act.mdx) allows an observed action object to be executed as a deterministic selector/method/argument tuple. Its [prompting guidance](https://github.com/browserbase/stagehand/blob/main/packages/docs/v2/best-practices/prompting-best-practices.mdx) recommends discovering candidate elements and using specific, atomic instructions. The point is to make a weaker model choose a well-scoped action against page structure, then reuse an explicit action object when appropriate. The framework also offers richer agent modes; a vendor guide does not prove comparative success rates.

**Adam implication:** for an Adam-controlled browser, a compact list of relevant elements and exact page text may be more useful than a full screenshot. This is consistent with Adam's own 56-row generated report: its [handoff](desktop-handoff-2026-10-05.md) records 3/3 correct with DOM text and 0/3 with screenshot alone, while the headless browser added about 577 MiB PSS when open. That small fixture is evidence for dense-text tasks, not a general browser win. Preserve browser lifecycle cleanup and measure cold start, live RAM, and task latency.

### 3. Anthropic: screenshot loop for desktops; page references for browsers

Anthropic's [computer use documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) describes a client-executed screenshot/action loop. It supports short batches such as click, type, screenshot; each action returns a result, and a batch stops after a failure. Its `zoom` operation provides a detailed crop while coordinates remain in the full screenshot's space. The docs recommend explicit screenshot resizing and coordinate scaling to avoid missed clicks on large displays, and recommend pruning old screenshot history so long runs do not accumulate image context.

Its separate [browser use documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/browser-use-tool) reads page structure, accessible elements, forms, and tabs, returning element references that can be clicked without first finding their pixel position. It can still use screenshots and viewport coordinates where needed. The lesson for Adam is architectural: the observation format should match the environment. A browser can expose element references; a game or custom canvas may only expose pixels. Anthropic's own documentation notes that computer use is slower because it often needs a fresh screenshot after an action. [Tool combination guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-combinations).

### 4. Google: normalized actions plus repeated observation

Google's [Gemini Computer Use guide](https://ai.google.dev/gemini-api/docs/computer-use) documents a loop in which the client sends a screenshot, receives an action, scales normalized 0–999 coordinates to the viewport, executes the action, captures the new state, and continues. The available actions include click, type, scroll, key presses, wait, and screenshot; browser, mobile, and desktop environments have different action sets. The model can attach an action intent. This documents the execution contract, not a claim that a specific inexpensive Gemini route will navigate Adam's desktop well.

**Adam implication:** normalized coordinates can simplify multi-resolution tool schemas, but the executor still needs exact transforms and fresh window bounds. The useful transferable practice is a clearly defined observation/action/result loop with post-action evidence, not the coordinate scale itself.

### 5. Open and academic GUI models: training for grounding, cropping for difficult screens

- [Set-of-Mark (Microsoft Research)](https://arxiv.org/abs/2310.11441) overlays numbered masks/boxes so a model can name a region. This helps some fine-grained visual grounding tasks. [SeeAct](https://arxiv.org/abs/2401.01614) is an important counterexample: on web tasks, its authors found marks alone ineffective relative to their grounding approach that used HTML plus visuals. Region overlays are a candidate treatment, not a universal upgrade.
- [ShowUI](https://arxiv.org/abs/2411.17465) trained a 2B vision-language-action model with UI-grounding data and UI-guided visual token selection. The authors report 75.1% zero-shot screenshot-grounding accuracy and 33% fewer redundant visual tokens during training, with 1.4× speed in their setup. This suggests that *which image detail reaches the model* matters; it does not imply a generic Adam crop achieves the same gain or that 2B model weights fit Adam's always-on memory budget.
- [UI-TARS](https://arxiv.org/abs/2501.12326) trains a screenshot-only GUI agent using large-scale GUI data, a unified action space, reasoning traces, and iterative environment interaction. The paper reports strong scores against its baselines on OSWorld and AndroidWorld. Its gains depend on specialized training and extensive data/compute, so they are a possible future model route, not a near-term local preprocessing patch.
- [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) tests high-resolution professional interfaces across 23 applications. Its authors report the strongest tested grounding model at 18.9%, while their cascaded search method reached 48.1% without extra training by narrowing the search area. This is strong evidence that targeted region search can help dense screens, but it uses a strong planner and its percentage is **single-target grounding**, not task-completion rate.

## What these results mean for Adam

Adam already has useful pieces: focused-window screenshots, lazy OCR, AT-SPI where available, an opt-in isolated browser DOM aid, short-lived screenshot IDs, focus/bounds checks, a bounded action sequence, and a lazy OmniParser worker. The [handoff](desktop-handoff-2026-10-05.md) reports a 21% median OCR-time reduction in a three-run crop comparison, with both conditions succeeding 3/3; it does not establish a whole-task latency win. A separate three-run OCR text-state check removed false “screen unchanged” feedback but did not reduce model turns. OmniParser has **not** yet been compared against the same screenshot/OCR task baseline. These local observations fit the literature's distinction between reading, grounding, and outcome verification.

The principal design choice is **which evidence to send for a given task**, not which single visual model to install:

| Task surface | First evidence to test | Escalate when | Main cost or caveat |
| --- | --- | --- | --- |
| Dense browser text/table | DOM text and stable page references, with screenshot when layout matters. | Text extraction misses a visual distinction or the page is canvas-heavy. | Browser process memory, dynamic DOM, hidden/offscreen nodes. |
| Accessible desktop form | Focused screenshot plus short, relevant AT-SPI control list/state. | A visible target is absent or mislabeled. | Linux accessibility quality varies; large trees can bury the target. |
| Custom icon/canvas | Focused screenshot and window OCR; optionally OmniParser regions. | Target remains ambiguous, small, or visually dense. | Detector latency/RAM and false interactable regions. |
| Small text or crowded screen | A specific native-resolution crop/zoom with a coordinate transform. | Full-frame downscaling loses text or tiny targets. | Can lose global context or click in the wrong coordinate frame. |
| Post-action state | Fresh screenshot plus targeted OCR/control state or a task artifact. | Visual state is ambiguous or loading continues. | Extra turn/inference; a click alone is not success. |

This table is an **experiment policy**, not a confirmed best routing policy. Some tasks need both global layout and a detail crop; some controls are only visible in pixels. Keep the model able to request more evidence, and keep mixed-domain tasks' relevant nonvisual tools available.

## Recommended evaluation, in order

1. **Freeze the same task and model route.** Use representative synthetic settings, browser-table, terminal-reading, and custom-control fixtures. Record screen/window state, prompt, image size, OCR/DOM/AT-SPI payload, exact model/provider, actions, outcomes, retry count, and final claim. Make a fixed grounding set with gold clickable regions or element IDs, plus a separate end-to-end task set. Use a private isolated display; do not resume Desky experiments while paused.
2. **Run evidence ablations on the laptop.** Compare screenshot only; screenshot plus current OCR; screenshot plus relevant AT-SPI/DOM; screenshot plus OCR and numbered OmniParser regions; and, on dense screens, a targeted crop/zoom. Keep the answer/action schema otherwise fixed. Report target-in-candidate rate, selected-target accuracy, exact task success, wrong/uncertain actions, turns, p50/p95 latency, input tokens, process-tree PSS/RSS, and CPU. Cold and warm OmniParser numbers must be separate.
3. **Inspect mechanism, then change one factor.** If the correct element never appears, improve extraction/coverage. If it appears but the model selects wrong, test shorter candidate lists, labels, and crops. If selection is correct but action misses, fix coordinate transforms/freshness. If actions succeed but Adam claims completion early, improve outcome evidence. A whole-task score alone cannot identify which layer failed.
4. **Promote only on paired quality and cost.** Require broader task success and honest outcome reporting to stay at least as good as baseline. An added detector should pay for its cold delay, CPU, and RAM with fewer wrong actions or fewer model turns. Avoid keeping a large worker resident merely because it helps an occasional icon. Repeat selected conditions enough for p50/p95 and failure analysis; the handoff suggests at least ten runs per selected scenario/condition where provider availability permits.
5. **Then test across hosts.** When the Desky pause is lifted, repeat the winning shared policy there and keep host measurements separate. Do not assume GPU speedups justify greater idle memory. Shenzhen I/O remains a late stress test after general navigation works; the laptop attempt was still in progress when handoff step 7 was written.

## Source and inference limits

The papers use different models, screen sizes, action budgets, datasets, and success definitions. ScreenSpot and ScreenSpot-Pro measure target grounding; OSWorld measures executable tasks; vendor docs describe supported workflows rather than controlled comparative results. Results from one cannot be read as a single leaderboard for Adam. Microsoft UFO is Windows-centric; Adam is Linux-centric. Anthropic and Google docs describe their model/tool ecosystems and are evidence of design patterns, not evidence that their models or exact APIs are suitable for Adam. The recommendations above are inferences to test on Adam's live path, with laptop resource constraints respected.
