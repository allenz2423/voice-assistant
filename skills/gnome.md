# Skill: GNOME Shell (Mutter Desktop Environment)

## Technical Architecture
- **Compositor**: Mutter with GNOME Shell.
- **Wayland Security Constraints**:
  - `wmctrl` and `xdotool` do **not** work on native Wayland surfaces in GNOME; they only interact with legacy XWayland applications.
  - Arbitrary JavaScript evaluation via `org.gnome.Shell.Eval` has been disabled by default since GNOME 41 for security hardening.
- **Environment Signatures**:
  - `XDG_CURRENT_DESKTOP` contains `GNOME`
  - `GDMSESSION=gnome`
  - `WAYLAND_DISPLAY` or `DISPLAY`

## Window & Workspace Management
Because upstream GNOME does not expose a public CLI for window manipulation on Wayland, automation relies on standard extensions or D-Bus:

### 1. Window Calls Extension (Recommended for CLI Automation)
If the user has installed the "Window Calls" extension (`org.gnome.Shell.Extensions.Windows`):
- **List Windows**:
  ```bash
  gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell/Extensions/Windows --method org.gnome.Shell.Extensions.Windows.List
  ```
- **Activate Window by ID**:
  ```bash
  gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell/Extensions/Windows --method org.gnome.Shell.Extensions.Windows.Activate <window_id>
  ```
- **Move to Workspace**:
  ```bash
  gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell/Extensions/Windows --method org.gnome.Shell.Extensions.Windows.MoveToWorkspace <window_id> <workspace_num>
  ```

### 2. Application Launching & Window Raising
- When targeting an app by name, use `gtk-launch <app_desktop_id>` or `gio launch <desktop_file_path>`.
- In GNOME, launching an already-running single-instance application (like Firefox, Edge, or Nautilus) raises its focused window.

## System Configuration via GSettings
Use `gsettings` to inspect or toggle GNOME environment features:
- **Dark Mode**: `gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'`
- **Light Mode**: `gsettings set org.gnome.desktop.interface color-scheme 'default'`
- **Night Light On**: `gsettings set org.gnome.settings-daemon.plugins.color night-light-enabled true`
- **Night Light Off**: `gsettings set org.gnome.settings-daemon.plugins.color night-light-enabled false`

## Extensions Management
- **List Extensions**: `gnome-extensions list`
- **Enable Extension**: `gnome-extensions enable <uuid>`
- **Disable Extension**: `gnome-extensions disable <uuid>`
- **Extension Info**: `gnome-extensions info <uuid>`

## Session & Power Control
- **Lock Screen**: `dbus-send --type=method_call --dest=org.gnome.ScreenSaver /org/gnome/ScreenSaver org.gnome.ScreenSaver.Lock` or `loginctl lock-session`
- **Logout (Immediate)**: `gnome-session-quit --logout --no-prompt`
- **Power Off Dialog**: `gnome-session-quit --power-off`
- **System Suspend**: `systemctl suspend`
