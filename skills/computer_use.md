# General Computer Use

Use this workflow for desktop applications, windows, and browsers. Adam operates the desktop to achieve the user's requested outcome.

## Execution Loop

1. Convert the request into a concise checklist of requested end states.
2. Focus or launch the relevant application.
3. Inspect current UI state. Identify target controls from visible labels, context, or visual boxes.
4. Execute one action or a short sequence of immediate coordinate-free inputs (click editable field -> type -> press enter).
5. Verify returned state against requested outcomes. Re-inspect whenever the next action requires a fresh target decision.
6. Deliver the substantive result once all requested deliverables are established.

## Input & Control Mechanics

- Use `observe_desktop` for window metadata or browser DOM. Use `computer_control` for visual interaction.
- Dragging items: inspect first, then use `drag` with the source and ordered path waypoints. It holds the mouse button; always follow it with `drop` using the latest Snapshot ID, optionally giving a final release coordinate. For maze-like paths, provide a waypoint for each turn and use a monitor screenshot when the route crosses windows. Use `computer_control` with `modifier=window` only to move a window itself.
- Always use the current `Snapshot ID`. Do not reuse stale coordinates or targets across turn boundaries.
- Keep clicks within the target application window.
- Window repositioning: Use `drag` with `modifier=window` (uses compositor move modifier). Use `scope=monitor` if destination is outside window bounds.
- Browser tabs: Use `close_browser_tab` with a visible title phrase to close tabs cleanly.
- Screenshot delays: Use `screenshot_delay_seconds` on `launch_application` / `focus_window` (default 3s for browsers, 0.25s for other apps) so the UI has settled before observation.
- Sequences: Max sequence length and text budgets are enforced automatically. Inspect state before retrying after any sequence timeout.

## Window & Workspace Policy

- Freely move windows between any workspaces and resize, tile, maximize, or fullscreen them as needed to complete the task. These reversible layout changes do not require asking the user.
- Use whichever workspace makes the task easiest. You may move either the task window or other windows to arrange the desktop.
- Re-observe after layout changes and restore the previous layout when practical. Never close a window that may contain user data or unsaved work just to rearrange the desktop.
