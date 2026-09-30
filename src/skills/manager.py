import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List

class SkillManager:
    """Manages modular capability skills (Markdown context files) for desktop environments,
    window managers, and system automation tools."""

    def __init__(self, custom_skills_dir: Optional[Path] = None):
        self.builtin_skills_dir = Path(__file__).resolve().parent.parent.parent / "skills"
        self.user_skills_dir = custom_skills_dir or (Path.home() / ".config" / "adam" / "skills")

    def get_skill_paths(self) -> List[Path]:
        """Returns all search directories for skills in order of priority (user custom first)."""
        paths = []
        if self.user_skills_dir.exists():
            paths.append(self.user_skills_dir)
        if self.builtin_skills_dir.exists():
            paths.append(self.builtin_skills_dir)
        return paths

    def detect_desktop_environment(self, refresh_env: bool = True) -> str:
        """Detects the currently running Desktop Environment or Window Manager."""
        if refresh_env:
            # Ensure we have refreshed display variables from systemd/sockets
            try:
                from src.tools.desktop import ensure_gui_environment
                ensure_gui_environment()
            except Exception:
                pass

        xdg_current = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        desktop_session = os.environ.get("DESKTOP_SESSION", "").lower()
        gdm_session = os.environ.get("GDMSESSION", "").lower()

        # 1. Primary check: XDG_CURRENT_DESKTOP
        if "hyprland" in xdg_current:
            return "hyprland"
        elif "sway" in xdg_current:
            return "sway"
        elif "i3" in xdg_current:
            return "i3"
        elif "kde" in xdg_current or "plasma" in xdg_current:
            return "kde_plasma"
        elif "gnome" in xdg_current:
            return "gnome"
        elif "cosmic" in xdg_current:
            return "cosmic"
        elif "niri" in xdg_current:
            return "niri"

        # 2. Compositor / WM specific sockets and signatures
        if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
            return "hyprland"
        if os.environ.get("SWAYSOCK"):
            return "sway"
        if os.environ.get("I3SOCK"):
            return "i3"
        if os.environ.get("NIRI_SOCKET"):
            return "niri"
        if os.environ.get("KDE_SESSION_VERSION"):
            return "kde_plasma"

        # 3. Session variables
        if "hyprland" in desktop_session:
            return "hyprland"
        if "sway" in desktop_session:
            return "sway"
        if "i3" in desktop_session:
            return "i3"
        if "niri" in desktop_session:
            return "niri"
        if "kde" in desktop_session or "plasma" in desktop_session:
            return "kde_plasma"
        if "gnome" in desktop_session or "gnome" in gdm_session:
            return "gnome"
        if "cosmic" in desktop_session:
            return "cosmic"

        # 4. Fallback: Binary presence check when display server is active
        if os.environ.get("WAYLAND_DISPLAY"):
            if shutil.which("hyprctl"):
                return "hyprland"
            if shutil.which("swaymsg"):
                return "sway"
            if shutil.which("niri") and os.environ.get("NIRI_SOCKET"):
                return "niri"
            if shutil.which("cosmic-comp"):
                return "cosmic"

        if os.environ.get("DISPLAY"):
            if shutil.which("i3-msg"):
                return "i3"

        return "generic_desktop"

    def list_skills(self) -> List[Dict[str, str]]:
        """Lists all discovered skill files and identifies the active one."""
        active_de = self.detect_desktop_environment()
        discovered = {}

        # Search user directory first, then built-in
        for search_dir in reversed(self.get_skill_paths()):
            for file_path in search_dir.glob("*.md"):
                if file_path.name.lower() in ["readme.md", "skills.md"]:
                    continue
                skill_id = file_path.stem.replace(".skills", "").lower()
                discovered[skill_id] = {
                    "id": skill_id,
                    "filename": file_path.name,
                    "path": str(file_path),
                    "is_active": (skill_id == active_de),
                    "loaded_by_default": (skill_id == "computer_use" or skill_id == active_de),
                }

        return list(discovered.values())

    def load_skill(self, skill_name: str) -> Optional[str]:
        """Loads the content of a skill by name or ID."""
        clean_name = skill_name.strip().lower().replace(".md", "").replace(".skills", "")
        # Skill names are IDs, never paths. This keeps lookup inside the
        # configured skill directories even when a model supplies a bad name.
        if not clean_name or "/" in clean_name or "\\" in clean_name:
            return None
        # Aliases
        alias_map = {
            "plasma": "kde_plasma",
            "kde": "kde_plasma",
            "kwin": "kde_plasma",
            "swaywm": "sway",
            "i3wm": "i3",
            "gnome-shell": "gnome",
            "niri-wm": "niri",
            "generic": "generic_desktop",
            "default": "generic_desktop"
        }
        target_id = alias_map.get(clean_name, clean_name)

        possible_filenames = [
            f"{target_id}.md",
            f"{target_id}.skills.md",
            f"{clean_name}.md",
            f"{clean_name}.skills.md"
        ]

        for search_dir in self.get_skill_paths():
            for fname in possible_filenames:
                p = search_dir / fname
                if p.is_file():
                    try:
                        if not p.resolve().is_relative_to(search_dir.resolve()) or p.stat().st_size > 40_000:
                            continue
                        return p.read_text(encoding="utf-8")
                    except Exception:
                        pass
        return None

    def get_startup_context(self) -> str:
        """Load the core computer-use workflow and detected desktop skill."""
        general = self.load_skill("computer_use") or ""
        desktop = self.get_active_de_context()
        return (
            "=== CORE COMPUTER-USE SKILL (loaded at startup) ===\n"
            f"{general.strip()}\n"
            "=== END CORE COMPUTER-USE SKILL ===\n\n"
            f"{desktop}"
        )

    def get_active_de_context(self) -> str:
        """Retrieves formatted markdown context for the currently active desktop environment."""
        active_de = self.detect_desktop_environment()
        content = self.load_skill(active_de)
        if not content:
            content = self.load_skill("generic_desktop") or ""

        header = f"=== ACTIVE DESKTOP & WINDOW MANAGER SKILL ({active_de.upper()}) ==="
        footer = "=== END DESKTOP SKILL ==="
        return f"{header}\n{content.strip()}\n{footer}"
