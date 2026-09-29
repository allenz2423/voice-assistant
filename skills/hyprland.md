# Skill: Hyprland (Wayland Dynamic Tiling Compositor)

## Technical Architecture
- **Compositor**: Hyprland (tested on v0.56.2+).
- **Display Server**: Wayland native. Traditional X11 tools (`wmctrl`, `xdotool`) only observe XWayland windows and cannot control native Wayland surfaces.
- **IPC Mechanism**: Dual UNIX domain sockets in `$XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/`:
  - `.socket.sock`: Synchronous command socket (used by `hyprctl`).
  - `.socket2.sock`: Asynchronous event stream (window focus, workspace changes, monitor events).
- **Environment Signatures**:
  - `HYPRLAND_INSTANCE_SIGNATURE` (format: `<hex>_<timestamp>_<hash>`)
  - `XDG_CURRENT_DESKTOP=Hyprland`
  - `WAYLAND_DISPLAY` (e.g. `wayland-1`)

## JSON Inspection Queries (`hyprctl -j`)
Always parse structured JSON output via `hyprctl -j <subcommand>`:
- `hyprctl -j activewindow`: Returns focused window object (`address`, `class`, `title`, `workspace: {id, name}`, `monitor`, `floating`, `fullscreen`, `pid`, `xwayland`).
- `hyprctl -j activeworkspace`: Returns currently viewed workspace details.
- `hyprctl -j clients`: Array of all open window objects across all workspaces.
- `hyprctl -j monitors`: Array of connected displays with resolutions, refresh rates, active workspace IDs, and scale factors.
- `hyprctl -j workspaces`: Array of all allocated workspaces and client counts.

## Verified Keybind Dispatchers (`hyprctl dispatch <dispatcher> <arg>`)
Execute via `hyprctl dispatch`:

### 1. Workspaces & Navigation
- Switch workspace: `hyprctl dispatch workspace <id>` (e.g., `1`, `2`, `name:code`, `m+1`, `m-1`)
- Move active window to workspace: `hyprctl dispatch movetoworkspace <id>`
- Move active window silently (no focus jump): `hyprctl dispatch movetoworkspacesilent <id>`
- Move a specific targeted window silently: `hyprctl dispatch movetoworkspacesilent <id>,address:0x<hex>` (e.g., `1,address:0x56358ccee9c0`)
- Toggle scratchpad overlay: `hyprctl dispatch togglespecialworkspace`
- Move window to scratchpad: `hyprctl dispatch movetoworkspacesilent special`

### 2. Window Manipulation & Layout
- Target patterns for windows: `address:0x<hex>` (recommended and deterministic), `class:<regex>`, `title:<regex>`, `pid:<int>`
- Focus window by address: `hyprctl dispatch focuswindow address:0x<hex>`
- Focus window by class/title: `hyprctl dispatch focuswindow class:<pattern>` or `title:<pattern>`
- Close window gracefully: `hyprctl dispatch closewindow address:<hex>`
- Force kill active window: `hyprctl dispatch killactive`
- Toggle floating / tiled: `hyprctl dispatch togglefloating active`
- Toggle fullscreen: `hyprctl dispatch fullscreen 0` (exclusive), `hyprctl dispatch fullscreen 1` (maximize keeping bars)
- Pin window (sticky on all workspaces): `hyprctl dispatch pin active`
- Move focus directionally: `hyprctl dispatch movefocus <l|r|u|d>`
- Swap window position in tiling layout: `hyprctl dispatch swapwindow <l|r|u|d>`
- Move window position in tiling layout: `hyprctl dispatch movewindow <l|r|u|d>`
- Center floating window: `hyprctl dispatch centerwindow`
- Toggle split direction (dwindle layout): `hyprctl dispatch layoutmsg togglesplit`
- Window tab grouping: `hyprctl dispatch togglegroup` (merge into tabbed group), `hyprctl dispatch changegroupactive f` (cycle next tab)

### 3. Multi-Monitor Management
- Focus monitor: `hyprctl dispatch focusmonitor <name_or_id>` (e.g. `DP-4`, `DP-5`, `0`, `+1`)
- Move active window to monitor: `hyprctl dispatch movewindowmon <name_or_id>`
- Turn monitors on/off: `hyprctl dispatch dpms on` / `hyprctl dispatch dpms off`

### 4. Compositor & Notification Controls
- Post compositor notification: `hyprctl notify <icon_0_to_5> <time_ms> <color_hex> <message>`
- Reload configuration: `hyprctl reload`

### 5. Session & Power Controls
- Lock session: `hyprlock` (or `loginctl lock-session`)
- Suspend system: `systemctl suspend`
- Reboot / Restart system: `systemctl reboot`
- Power off / Shutdown system: `systemctl poweroff`

## STRICT EXECUTION RULES & SYNTAX
1. **Targeted Window Moving (`movetoworkspacesilent <workspace>,<window>`)**:
   - To move a specific background window to a workspace WITHOUT hijacking user focus or moving the active window, dispatch:
     `hyprctl dispatch movetoworkspacesilent <workspace_id>,address:0x<hex>`
   - The parameters MUST be comma-separated with NO space between `<workspace_id>` and the window identifier (e.g. `1,address:0x56358ccee9c0`).
   - Using `address:0x<hex>` queried from `hyprctl -j clients` is mandatory for deterministic targeting without title/class collision.

2. **Resolving Active / Current Workspace**:
   - Hyprland DOES NOT recognize literal `"current"`, `"active"`, or `"here"` as a workspace identifier. Dispatching `movetoworkspace current` fails with `Invalid workspace`.
   - To target the current workspace, ALWAYS query `hyprctl -j activeworkspace` and extract its numeric `.id` (e.g., `2`).

3. **Silent vs Active Window Movement**:
   - Moving background applications (e.g., Spotify, Discord, browser) MUST ALWAYS use `movetoworkspacesilent` so the user's ongoing work is uninterrupted.
   - Moving the active window can use `movetoworkspace` (follows the window) or `movetoworkspacesilent` (leaves user in current workspace).

4. **Return Value Verification**:
   - `hyprctl dispatch` returns exit status 0 even on syntax and runtime errors (e.g., `Invalid workspace`, `Window not found`).
   - A command is ONLY successful if stdout contains `"ok"` and DOES NOT contain `"invalid"` or error text.

5. **Window Class & Semantic Aliases**:
   - Map colloquial app names to actual process classes:
     - `browser` / `edge` -> `microsoft-edge`
     - `discord` -> `vesktop` or `discord`
     - `terminal` -> `Alacritty` or `kitty`
     - `music` / `spotify` -> `Spotify`
     - `code` / `editor` -> `code` or `codium`

6. **Moving Windows Back / Undo / Previous Workspace**:
   - When the user asks to "move it back", "put it back", or "undo", use `workspace_control(action='move', workspace='back', target='...')`.
   - Hyprland natively supports returning a window to its prior workspace via `hyprctl dispatch movetoworkspacesilent previous,address:0x<hex>`.
   - To switch back to the previous workspace, dispatch `hyprctl dispatch workspace previous`.

7. **Pronoun & Anaphora Resolution**:
   - When the user says "move it back", "close it", or "focus on it", "it" refers to the application or window acted on in the previous turn.
