# Skill: KDE Plasma 6 (KWin Desktop Environment)

## Technical Architecture
- **Environment**: KDE Plasma 6 (powered by KWin compositor).
- **Wayland vs X11 Reality**: On Wayland, traditional X11 automation tools (`wmctrl`, `xdotool`) are blind to native Wayland client surfaces. Use `kdotool`, KWin D-Bus interfaces, or KWin scripting.
- **D-Bus Tooling**: In Plasma 6, use `/usr/bin/qdbus6` (or `/usr/bin/qdbus` if present).
- **Environment Signatures**:
  - `XDG_CURRENT_DESKTOP=KDE`
  - `KDE_SESSION_VERSION=6`
  - `WAYLAND_DISPLAY` or `DISPLAY`

## Virtual Desktops Control via D-Bus
KWin exposes virtual desktop switching directly on the `/KWin` object:
- **Switch Desktop**: `qdbus6 org.kde.KWin /KWin setCurrentDesktop <int>` (1-indexed desktop number).
- **Query Active Desktop**: `qdbus6 org.kde.KWin /KWin currentDesktop`
- **Query Total Desktops**: `qdbus6 org.kde.KWin /KWin numberOfDesktops`
- **Cycle Next Desktop**: `qdbus6 org.kde.KWin /KWin nextDesktop`
- **Cycle Previous Desktop**: `qdbus6 org.kde.KWin /KWin previousDesktop`

## Window Focus & Management
- **Primary Tool**: `kdotool` (Wayland-native `xdotool` clone communicating via KWin Scripting).
  - Search and focus by class: `kdotool search --class "<app_class>" | head -n 1 | xargs -r kdotool windowactivate`
  - Search and focus by title: `kdotool search --name "<window_title>" | head -n 1 | xargs -r kdotool windowactivate`
  - Close window: `kdotool search --class "<app_class>" | head -n 1 | xargs -r kdotool windowclose`
- **Native KWin Scripting via D-Bus** (fallback when `kdotool` is absent):
  KWin scripts can be loaded dynamically via D-Bus:
  ```bash
  # 1. Load script:
  SCRIPT_PATH=$(dbus-send --print-reply --dest=org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript string:"/tmp/focus.js" | awk -F'"' '{print $2}')
  # 2. Run script:
  dbus-send --dest=org.kde.KWin "$SCRIPT_PATH" org.kde.kwin.Script.run
  # 3. Stop script:
  dbus-send --dest=org.kde.KWin "$SCRIPT_PATH" org.kde.kwin.Script.stop
  ```

## Desktop Shell & KWin Shortcuts
Trigger desktop overviews and utilities via `org.kde.kglobalaccel`:
- **Toggle Overview**: `qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut "Overview"`
- **Toggle Desktop Grid**: `qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut "Grid"`
- **Show Desktop**: `qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut "Show Desktop"`
- **Open KRunner**: `qdbus6 org.kde.krunner /App display`
- **Toggle Night Light (Night Color)**: `qdbus6 org.kde.KWin /ColorCorrect org.kde.kwin.ColorCorrect.toggle`

## Session & Power Control
- **Lock Screen**: `qdbus6 org.freedesktop.ScreenSaver /ScreenSaver Lock` or `loginctl lock-session`
- **Suspend Session**: `qdbus6 org.kde.Solid.PowerManagement /org/kde/Solid/PowerManagement/Actions/SuspendSession suspendToRam`
- **Logout / Power Dialog**: `qdbus6 org.kde.Shutdown /Shutdown logoutAndShutdown`
