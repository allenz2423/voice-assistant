# Skill: Sway (Wayland i3-Compatible Tiling Compositor)

## Technical Architecture
- **Compositor**: Sway (i3-compatible Wayland compositor).
- **IPC Mechanism**: UNIX domain socket referenced by `$SWAYSOCK` (`/run/user/<uid>/sway-ipc.<uid>.<pid>.sock`).
- **CLI Utility**: `swaymsg`
- **Environment Signatures**:
  - `SWAYSOCK`
  - `XDG_CURRENT_DESKTOP=sway`
  - `WAYLAND_DISPLAY`

## JSON Inspection Queries (`swaymsg -t <type>`)
Always parse structured JSON output via `swaymsg -t`:
- **Window Tree**: `swaymsg -t get_tree` (hierarchical tree of outputs, workspaces, tiling nodes, and `floating_nodes`).
- **Workspaces List**: `swaymsg -t get_workspaces` (array containing `name`, `focused`, `output`, `visible`).
- **Outputs / Displays**: `swaymsg -t get_outputs` (active monitors, modes, refresh rates, and scales).
- **Inputs**: `swaymsg -t get_inputs` (keyboards, mice, touchpads).

## Verified Commands & Criteria (`swaymsg [criteria] <command>`)
Criteria selectors can match `app_id` (Wayland native) or `class` (XWayland), supporting case-insensitive regex `(?i)`:

### 1. Workspaces & Navigation
- Switch workspace: `swaymsg workspace <number_or_name>`
- Move container to workspace: `swaymsg move container to workspace <number_or_name>`
- Move workspace to different monitor: `swaymsg move workspace to output <output_name>`
- Cycle next/prev workspace: `swaymsg workspace next` / `prev`

### 2. Window Manipulation & Tiling
- Focus window by app_id: `swaymsg '[app_id="(?i)<pattern>"] focus'`
- Focus window by XWayland class: `swaymsg '[class="(?i)<pattern>"] focus'`
- Focus window by title: `swaymsg '[title="(?i)<pattern>"] focus'`
- Kill matching window: `swaymsg '[app_id="(?i)<pattern>"] kill'`
- Toggle floating mode: `swaymsg floating toggle`
- Toggle fullscreen: `swaymsg fullscreen toggle`
- Change layout: `swaymsg layout toggle split` (or `stacking`, `tabbed`, `splitv`, `splith`)
- Split direction for next window: `swaymsg split horizontal` or `split vertical`
- Move container: `swaymsg move <left|right|up|down> 50px`
- Move focus directionally: `swaymsg focus <left|right|up|down>`
- Make window sticky: `swaymsg sticky toggle`

### 3. Scratchpad
- Move container to scratchpad: `swaymsg move scratchpad`
- Show / cycle scratchpad: `swaymsg scratchpad show`

### 4. Compositor & Session
- Hot-reload configuration: `swaymsg reload`
- Lock session: `swaylock` or `loginctl lock-session`

## STRICT EXECUTION RULES & SYNTAX
1. **Targeted Window Moving**:
   - To move a specific background window to a workspace without affecting the active window, dispatch:
     `swaymsg '[con_id=<id>] move container to workspace <workspace>'` or `swaymsg '[app_id="(?i)<pattern>"] move container to workspace <workspace>'`
2. **Current / Active Workspace**:
   - To target the active workspace, query `swaymsg -t get_workspaces` and select the entry with `"focused": true`.
3. **Previous Workspace / Undo**:
   - To toggle back to the previous workspace, run `swaymsg workspace back_and_forth`.
   - To return a container to the previous workspace, run `swaymsg '[con_id=<id>] move container to workspace back_and_forth'`.
4. **Pronouns & Anaphora**:
   - When the user says "move it back", "close it", or "focus on it", "it" refers to the application acted on in the previous turn.
