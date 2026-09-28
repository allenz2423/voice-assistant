# Skill: COSMIC Desktop (System76 Rust Wayland Environment)

## Technical Architecture
- **Compositor**: `cosmic-comp` (built in Rust using Smithay).
- **Current CLI Automation Status**:
  - `cosmic-comp` does **not** currently provide a native public CLI IPC tool like `hyprctl` or `swaymsg`.
  - Window and workspace manipulation is primarily event-driven via Wayland protocols (`ext-foreign-toplevel-list-v1`) and keyboard shortcut handlers.
- **Environment Signatures**:
  - `XDG_CURRENT_DESKTOP=COSMIC`
  - `cosmic-session` / `cosmic-comp` in process tree
  - `WAYLAND_DISPLAY`

## Automation & Control Strategies

### 1. Application Launching & Management
- Applications launch via standard Freedesktop `.desktop` files using `gtk-launch <desktop_id>` or `gio launch <desktop_file_path>`.
- App library can be invoked via `cosmic-app-library`.

### 2. Window & Workspace Interaction
Because there is no official CLI IPC tool:
- **Focusing Existing Windows**: Launching an active single-instance application (e.g. browser, file manager) will trigger the COSMIC compositor to raise that window.
- **Virtual Input (Key Simulation)**:
  When scripting window navigation, simulate COSMIC's default shortcut bindings via `ydotool`:
  - Workspace Switch: `Super + Ctrl + Up/Down`
  - Move Window to Workspace: `Super + Shift + Up/Down`
  - Toggle Window Floating/Tiling: `Super + G`
  - Workspace Overview: `Super + D`
  - Window Launcher: `Super`

### 3. System Settings & Configuration
- Launch Settings UI: `cosmic-settings`
- Audio and Media: Managed via MPRIS (`playerctl`) and PipeWire (`wpctl`).

### 4. Session & Power Control
- **Lock Screen**: `loginctl lock-session`
- **Suspend System**: `systemctl suspend`
- **Power Off**: `systemctl poweroff`
- **Reboot**: `systemctl reboot`
