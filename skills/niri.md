# Skill: Niri (Scrollable-Tiling Wayland Compositor)

## Technical Architecture
- **Compositor**: Niri (infinite horizontal ribbon scrollable tiling compositor).
- **Display Server**: Wayland native.
- **IPC Mechanism**: UNIX domain socket referenced by `$NIRI_SOCKET`.
- **Primary Tool**: `niri msg` (supports structured JSON with `-j`).
- **Environment Signatures**:
  - `NIRI_SOCKET`
  - `XDG_CURRENT_DESKTOP=niri`
  - `WAYLAND_DISPLAY`

## JSON & Inspection Queries
- Query windows: `niri msg -j windows`
- Query workspaces: `niri msg -j workspaces`
- Query outputs/monitors: `niri msg -j outputs`
- Query keyboard layouts: `niri msg -j keyboard-layouts`

## Action Commands (`niri msg action <action>`)
### 1. Window Management
- Focus window by ID: `niri msg action focus-window --id <id>`
- Close window: `niri msg action close-window`
- Toggle floating: `niri msg action toggle-window-floating`
- Toggle fullscreen: `niri msg action fullscreen-window`
- Center column: `niri msg action center-column`
- Move column left / right: `niri msg action move-column-left` / `move-column-right`
- Consume window into column: `niri msg action consume-or-expel-window-left`

### 2. Workspace Management
- Switch workspace: `niri msg action focus-workspace <index_or_name>`
- Move column to workspace: `niri msg action move-column-to-workspace <index_or_name>`
- Move window to workspace: `niri msg action move-window-to-workspace <index_or_name>`

### 3. Session & Utilities
- Screenshot: `grim -` or `niri msg action screenshot`
- Power/Session lock: `loginctl lock-session`
