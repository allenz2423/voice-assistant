# General Computer-Use Reliability: OBS Case Study

## Purpose

This is a general design note for Adam's computer-use workflow across desktop applications, browsers, dialogs, and other interactive workloads. OBS is one concrete failure case used to expose the broader problems. Fixes should improve the common observe–act–verify loop rather than add a one-off OBS script.

The target model is around 4B parameters. Keep its choices local and clear; move state synchronization, action validation, and factual checks into deterministic code or capable adapters.

## Case Study: OBS Recording

Adam was asked to open OBS and start recording. OBS launched and received focus. It eventually started recording, but it did not select a display in Hyprland's screen-sharing chooser. A recording can therefore be active while containing no useful desktop capture.

The chooser exposed two displays:

- DP-4: 1920×1080, positioned at `2560,-300`
- DP-5: 2560×1440, positioned at `0,0`

The requested target was the 1440p display, DP-5. The chooser remained open, so selecting a display was a required step before considering the task complete.

### Observed behavior

- Adam could find and launch OBS, and `focus_window` could focus its window.
- `observe_desktop` returned metadata for as many as 29 windows, including unrelated apps and substantial AT-SPI output.
- The focused-window screenshot path worked: it returned a 638×718 OBS image with its controls visible.
- Adam sometimes stopped after observing and said “Done.” In another attempt, it tried a click without a valid `computer_control` snapshot; the controller rejected the click.
- A later click succeeded only after I supplied a specific target and coordinates. That proves pointer input can work, but it is not evidence that Adam can independently identify and select the display.
- A running recording timer did not establish that DP-5 was selected or captured.
- The successful prompting attempts used a separate one-shot `AdamBrain` process, not audio submitted through the running daemon. They exercised the tool loop, but not the complete live voice path.

## General Failure Model

The OBS case points to issues that can affect any application task:

1. **Intermediate progress was mistaken for completion.** Opening or focusing an app, dismissing a dialog, or dispatching a click can leave the user's actual goal unfinished.
2. **Too much unrelated state reached the model.** A long inventory and full trees are costly for a 4B model and can bury the relevant window or dialog.
3. **Focus and observation were not synchronized.** A focus change can make the previous screenshot and target coordinates stale. Adam then has to remember to request another observation and copy its snapshot ID correctly.
4. **The best adapter did not expose the needed target.** Accessibility may omit controls; a screenshot may show them without identifying their meaning. Adapter availability must be checked for the actual target/action, not inferred from the app name.
5. **Instructions conflicted.** The system prompt said, “After an action, say ‘Done,’” while the computer-use skill required verification. A small model may follow the shorter completion shortcut.
6. **A tool response described execution, not outcome.** “Focused,” “clicked,” and “recording started” are facts about individual operations, not proof that the full requested state was reached.

## General Architecture

Treat the controller as the source of truth for observable application state and action mechanics. Adam translates the request into intent and selects the next semantic action. Adapters provide state and actions for their domains. A verifier checks an explicit completion predicate using authoritative state.

```text
User goal
   ↓
Adam identifies intent and next action
   ↓
Controller validates it against the current observation
   ↓
Adapter executes one semantic action
   ↓
Controller returns fresh, task-scoped state
   ↓
Verifier checks the requested outcome
   ├─ incomplete or unknown → Adam chooses a next step or reports the blocker
   └─ complete → report success
```

The controller should own focus/observation synchronization, snapshot validity, action sequencing, stale-target rejection, and bounded retries. Adapters should expose concrete state and supported actions. The verifier may use general facts or a domain-specific adapter; it should only assert facts the adapter can establish. Adam should not have to infer that a click worked or remember low-level synchronization rules.

For OBS, an adapter could expose:

```json
{
  "focused_app": "OBS",
  "blocking_dialog": {
    "type": "screen_share_chooser",
    "displays": [
      {"id": "DP-4", "resolution": "1920x1080"},
      {"id": "DP-5", "resolution": "2560x1440"}
    ]
  },
  "recording": false
}
```

The corresponding completion predicate is `capture_source == requested_display AND recording == true`. This is an example of a reusable pattern: represent the task's relevant entities and state explicitly, then verify the requested predicate. A browser, file manager, settings panel, or other app will have its own adapter facts and predicate.

## Fixes That Generalize

### 1. Return a fresh observation after focus and every action

After focusing a window, return the focused-window observation, screenshot, and snapshot ID together. A focus change invalidates the previous snapshot. After each semantic action, return execution status plus the next observation automatically. Adam should not need to remember a separate inspect call just to recover state.

Reject actions that reference stale snapshots or targets. Bind pointer coordinates to the exact screenshot and geometry that produced them. Prefer semantic element IDs when available. A successful input dispatch must never be reported as task completion by itself.

### 2. Scope observations to the current subtask

Separate broad discovery from focused operation. Use window inventory to find the target app, then send Adam only the relevant window, blocking dialog, actionable controls, and necessary surrounding context. Avoid repeating unrelated window trees on each step.

Use structured data such as accessibility trees, browser DOM, and app APIs when they expose the requested target and action. Use a focused screenshot when they do not. Combine sources when different parts of the same workflow require different adapters. Include only concise capabilities and state needed for the next decision.

### 3. Discover capabilities per target and action

Probe whether an adapter can observe and act on the specific requested control. “OBS supports AT-SPI” or “Firefox is open” is not enough. If the app API or accessibility layer cannot expose a control, report that capability gap and use another available source. Do not silently invent semantic targets or choose the first visual match.

When choices refer to external entities—monitors, files, browser tabs, accounts—return stable identifiers and useful attributes such as name, resolution, path, or title. Resolve “my 1440p screen” against live monitor metadata instead of screen position or list order.

### 4. Make goal completion explicit and independently verifiable

Translate the user request into a predicate before acting. Keep the predicate specific to the workload, but use the same verification contract everywhere:

- Browser research: the requested page or information is present.
- File task: the intended file or directory is open or changed as requested.
- Application workflow: the requested setting, selection, or state is active.
- OBS: the requested display is configured as the capture source and recording is active.

The verifier reports `complete`, `incomplete`, or `unknown` with evidence. If an authoritative state source exists, use it. If not, return uncertainty instead of inferring success from a dispatched click, closed dialog, or timer. Inspecting a resulting artifact—such as an OBS frame or saved file—is useful in a test harness or targeted diagnosis, but should not be required for every routine task when authoritative application state is available.

### 5. Simplify instructions for a 4B model

Remove unconditional instructions such as “After an action, say ‘Done.’” Say: “Report completion only when the verifier confirms the requested final state.” Give the model one next action at a time and a short observation. Keep tool schemas small, use clear action names, and present actionable targets with their available operations. Let the controller enforce freshness, sequencing, and retry limits.

If the next action is ambiguous or the required capability is unavailable, Adam should ask a focused question or report the blocker. It should not guess or repeat a failed action.

## Evaluation Plan

Measure action autonomy and verification separately across workloads. Do not use exact coordinates or a scripted sequence in autonomy tests.

| Check | What it demonstrates |
| --- | --- |
| Select a named item among similar visible choices | Target resolution |
| Continue after opening or focusing an app | Multi-step task continuation |
| Use a browser control and report page state | Browser observation/action integration |
| Open a requested file in the intended application | Cross-app workflow completion |
| Controller reports an app's resulting state | Structured state integration |
| Test harness validates the resulting artifact or captured frame | End-to-end adapter correctness |
| All required predicates pass from a natural-language request | End-to-end task autonomy |

Use **“Record my 1440p screen in OBS”** as one test among several, without supplying DP-5, coordinates, or a scripted sequence. Adam should resolve the display from monitor metadata; the controller should verify the capture source and recording state. A separate test harness may inspect a captured frame to confirm that the controller's reported state matches reality. A test that supplies exact button coordinates measures pointer control, not autonomy.

## Scope

This note analyzes one incident and proposes a general computer-use architecture. The OBS details are examples, not special cases that should define the whole tool API. The same observe–act–verify contract should support desktop applications, browser tasks, dialogs, and future adapters without making Adam learn adapter-specific mechanics.
