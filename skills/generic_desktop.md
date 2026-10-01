# Skill: Generic Linux Desktop (Universal Freedesktop & IPC Standards)

## Technical Architecture
- **Scope**: Universal baseline across Linux desktop environments (XFCE, Cinnamon, MATE, LXQt, Openbox, and unlisted compositors) adhering to Freedesktop.org, EWMH, NetWM, MPRIS, and systemd specifications.
- **Display Servers**: Wayland (`WAYLAND_DISPLAY`) or X11 (`DISPLAY`).

## Universal Automation Standards

### Scope Restriction
The commands below are technical references, not default actions for general app tasks. Use read-only commands to inspect state when relevant. Use window-manager or compositor commands for requested window/workspace/layout management and when a minimal layout change is needed to reveal or operate the task's target. Never use them to imitate an application's controls: a request to start recording means use the recorder's UI or direct app API, not fullscreen, workspace, or input-injection commands. For app tasks, focus the intended app, inspect its current UI, perform the requested action, and inspect the result. If the task view is too small or obstructed, resolve that visibility problem safely and re-observe before continuing.

### 1. Application Discovery & Launching (Freedesktop XDG)
- Standard paths: `~/.local/share/applications`, `/usr/local/share/applications`, `/usr/share/applications`, `/var/lib/flatpak/exports/share/applications`
- **Native Freedesktop Launcher**: `gtk-launch <desktop_id>` (e.g. `gtk-launch firefox.desktop` or `gtk-launch com.discordapp.Discord`)
- **GIO Launcher**: `gio launch <desktop_file_path>`

### 2. Window Management (EWMH / NetWM on X11 & XWayland)
On X11 or with XWayland-compatible windows, `wmctrl` is the universal tool:
- **List Windows**: `wmctrl -l -x` (lists window IDs, desktop numbers, client classes, and window titles)
- **Focus Window**: `wmctrl -a "<target>"` (searches title and class case-insensitively)
- **Close Window Gracefully**: `wmctrl -c "<target>"` (sends `WM_DELETE_WINDOW` client message)
- **Switch Desktop**: `wmctrl -s <zero_indexed_desktop_num>`
- **Move Active Window to Desktop**: `wmctrl -r :ACTIVE: -t <desktop_num>`
- **Toggle Fullscreen**: `wmctrl -r :ACTIVE: -b toggle,fullscreen`
- **Toggle Maximize**: `wmctrl -r :ACTIVE: -b toggle,maximized_vert,maximized_horz`

### 3. Media Playback
- For transport commands (`play`, `pause`, `next`, `previous`, `stop`), use `control_media_app` with the explicitly requested application. It resolves and targets only that app's MPRIS player.
- Never use unscoped `playerctl` transport commands or click the app UI for a simple transport command; global controls may affect another app, such as a browser video.
- Use the requested app's UI to browse playlists, search, or select a specific track.
- `get_now_playing` is read-only and may be used to identify which player currently owns playback.

### 4. Audio Control (PipeWire WirePlumber / PulseAudio)
- **WirePlumber (PipeWire Default)**:
  - Increase / Decrease Volume: `wpctl set-volume -l 1.5 @DEFAULT_AUDIO_SINK@ 5%+` / `5%-`
  - Toggle Mute: `wpctl set-mute @DEFAULT_AUDIO_SINK@ toggle`
  - Query Status: `wpctl status`
- **PulseAudio / pactl**:
  - Volume: `pactl set-sink-volume @DEFAULT_SINK@ +5%` / `-5%`
  - Mute: `pactl set-sink-mute @DEFAULT_SINK@ toggle`

### 5. Clipboard Automation
- **Wayland**: `wl-copy "<text>"` and `wl-paste --no-newline`
- **X11**: `xclip -selection clipboard` or `xsel --clipboard`

### 6. Session & Power Control (systemd-logind)
Universal logind D-Bus commands that work regardless of the active desktop:
- **Lock Screen**: `loginctl lock-session`
- **Suspend System**: `systemctl suspend`
- **Reboot System**: `systemctl reboot`
- **Power Off System**: `systemctl poweroff`
