# Adam Desktop & Window Manager Skills

This directory contains modular environment skills that provide Adam with deep domain knowledge for interacting with different Linux desktop environments (DEs) and window managers (WMs).

## Available Skills

| Skill File | Desktop Environment / Window Manager | Primary Tooling |
| :--- | :--- | :--- |
| `computer_use.md` | General computer-use workflow | Adam desktop and browser tools |
| `hyprland.md` | Hyprland (Wayland Tiling) | `hyprctl dispatch`, UNIX sockets |
| `sway.md` | Sway (Wayland i3-compatible) | `swaymsg`, tree container IPC |
| `i3.md` | i3 (X11 Tiling Window Manager) | `i3-msg`, UNIX sockets |
| `kde_plasma.md` | KDE Plasma (Desktop Environment & KWin) | `qdbus`, `kdotool`, `kstart` |
| `gnome.md` | GNOME Shell (Desktop Environment & Mutter) | `gdbus`, `gsettings`, `gnome-extensions` |
| `cosmic.md` | COSMIC (System76 Wayland Environment) | `cosmic-comp`, `cosmic-settings` |
| `generic_desktop.md` | Universal Linux Desktop (X11 / Wayland) | `wmctrl`, `gtk-launch`, `playerctl`, `loginctl` |

## How Skills Work

1. **Skill Discovery:**
   `SkillManager` discovers top-level Markdown files in the built-in `skills/` directory and `~/.config/adam/skills/`. User files override built-in skills with the same ID. Skill IDs are filenames without `.md` or `.skills.md`.

2. **Startup Loading:**
   When AdamBrain starts, it loads `computer_use.md` and detects the active desktop from environment variables, compositor sockets, and available binaries. It appends the matching desktop skill (or `generic_desktop.md`) to the system prompt.

3. **On-Demand Loading:**
   Adam can call `list_skills` to discover skills and `get_skill_context` to load one into the current conversation. Skill files are read-only guidance; tool availability and runtime policy remain authoritative.

4. **User-Defined Custom Skills:**
   Users can add or override skills without modifying the core voice assistant repository by placing Markdown files in:
   ```bash
   ~/.config/adam/skills/
   ```
   Any `.md` or `.skills.md` file in that directory will automatically take precedence over built-in skills.
