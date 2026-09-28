# Shin Desktop & Window Manager Skills

This directory contains modular environment skills that provide Shin with deep domain knowledge for interacting with different Linux desktop environments (DEs) and window managers (WMs).

## Available Skills

| Skill File | Desktop Environment / Window Manager | Primary Tooling |
| :--- | :--- | :--- |
| `hyprland.md` | Hyprland (Wayland Tiling) | `hyprctl dispatch`, UNIX sockets |
| `sway.md` | Sway (Wayland i3-compatible) | `swaymsg`, tree container IPC |
| `i3.md` | i3 (X11 Tiling Window Manager) | `i3-msg`, UNIX sockets |
| `kde_plasma.md` | KDE Plasma (Desktop Environment & KWin) | `qdbus`, `kdotool`, `kstart` |
| `gnome.md` | GNOME Shell (Desktop Environment & Mutter) | `gdbus`, `gsettings`, `gnome-extensions` |
| `cosmic.md` | COSMIC (System76 Wayland Environment) | `cosmic-comp`, `cosmic-settings` |
| `generic_desktop.md` | Universal Linux Desktop (X11 / Wayland) | `wmctrl`, `gtk-launch`, `playerctl`, `loginctl` |

## How Skills Work

1. **Automatic Detection:**
   At startup and before handling commands, `SkillManager` evaluates environment variables (`XDG_CURRENT_DESKTOP`, `HYPRLAND_INSTANCE_SIGNATURE`, `SWAYSOCK`, `I3SOCK`), compositor sockets, and available binaries to determine the active DE/WM.

2. **Context Injection:**
   The active environment's skill file is loaded and injected into ShinBrain's system prompt. This gives the local model immediate, zero-latency awareness of:
   - Built-in assistant tools mapping directly to that DE/WM.
   - Exact CLI dispatcher commands, arguments, and syntax.
   - Session and power management hooks.

3. **User-Defined Custom Skills:**
   Users can add or override skills without modifying the core voice assistant repository by placing Markdown files in:
   ```bash
   ~/.config/shin/skills/
   ```
   Any `.md` or `.skills.md` file in that directory will automatically take precedence over built-in skills.
