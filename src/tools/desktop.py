import os
import json
import re
import select
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional, Any, Dict, List, Set

from src.tools.desktop_timing import log_duration, timed_stage

_HYPRLAND_LUA_DISPATCH: dict[str, bool] = {}


def _which(command: str, environ: dict[str, str] | None = None) -> str | None:
    """Resolve executables using the PATH belonging to the selected session."""
    if environ is None:
        return shutil.which(command)
    return shutil.which(command, path=environ.get("PATH", ""))


def _is_native_wayland_session(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return (
        env.get("XDG_SESSION_TYPE", "").strip().lower() == "wayland"
        or bool(env.get("WAYLAND_DISPLAY"))
    )


def _lua_string(value: str) -> str:
    """Quote a Python string as a Lua string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return f'"{escaped}"'


def _hyprland_dispatch(
    legacy_dispatch: str, legacy_args: str = "", *, lua_expression: str,
    environ: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a dispatcher on both legacy hyprlang and Lua-config Hyprland versions."""
    env = os.environ if environ is None else environ
    hyprctl = _which("hyprctl", environ)
    if not hyprctl:
        raise RuntimeError("hyprctl is unavailable on the selected desktop session PATH.")
    if hyprctl not in _HYPRLAND_LUA_DISPATCH:
        try:
            version = subprocess.run([hyprctl, "version"], capture_output=True, text=True, timeout=2, env=env).stdout
            match = re.search(r"Hyprland\s+(\d+)\.(\d+)", version)
            _HYPRLAND_LUA_DISPATCH[hyprctl] = bool(match and tuple(map(int, match.groups())) >= (0, 55))
        except Exception:
            _HYPRLAND_LUA_DISPATCH[hyprctl] = False

    if _HYPRLAND_LUA_DISPATCH[hyprctl]:
        command = [hyprctl, "dispatch", lua_expression]
    else:
        command = [hyprctl, "dispatch", legacy_dispatch]
        if legacy_args:
            command.append(legacy_args)
    return subprocess.run(command, capture_output=True, text=True, timeout=3, env=env)

def ensure_gui_environment():
    """Dynamically acquires GUI display variables (WAYLAND_DISPLAY, DISPLAY, HYPRLAND_INSTANCE_SIGNATURE,
    SWAYSOCK, I3SOCK, NIRI_SOCKET, KDE_SESSION_VERSION) from systemd user environment or active compositor
    sockets if running inside a headless/systemd daemon."""
    if not os.environ.get("WAYLAND_DISPLAY") or not os.environ.get("DISPLAY"):
        try:
            res = subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True, text=True, timeout=1)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k in [
                            "WAYLAND_DISPLAY", "DISPLAY", "HYPRLAND_INSTANCE_SIGNATURE",
                            "XDG_CURRENT_DESKTOP", "XDG_BACKEND", "XDG_RUNTIME_DIR",
                            "SWAYSOCK", "I3SOCK", "NIRI_SOCKET", "KDE_SESSION_VERSION",
                            "KDE_FULL_SESSION", "DESKTOP_SESSION", "XDG_SESSION_DESKTOP",
                            "DBUS_SESSION_BUS_ADDRESS"
                        ] and v:
                            os.environ[k] = v
        except Exception:
            pass

    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        bus_path = Path(f"/run/user/{os.getuid()}/bus")
        if bus_path.exists():
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"

    # If still not found, check /run/user/<uid>/wayland-* socket
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    if not os.environ.get("WAYLAND_DISPLAY") and runtime_dir.exists():
        for sock in runtime_dir.glob("wayland-*"):
            if sock.is_socket():
                os.environ["WAYLAND_DISPLAY"] = sock.name
                break

    # If Hyprland is active, locate its instance signature socket only if another compositor is not specified
    if (
        not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
        and not os.environ.get("SWAYSOCK")
        and not os.environ.get("I3SOCK")
        and not os.environ.get("NIRI_SOCKET")
        and "kde" not in os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        and "plasma" not in os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        and runtime_dir.exists()
    ):
        hypr_dir = runtime_dir / "hypr"
        if hypr_dir.exists():
            for inst in hypr_dir.iterdir():
                if inst.is_dir() and (inst / ".socket.sock").exists():
                    os.environ["HYPRLAND_INSTANCE_SIGNATURE"] = inst.name
                    break

    # If Sway is active, locate its SWAYSOCK
    if not os.environ.get("SWAYSOCK") and runtime_dir.exists():
        for sock in runtime_dir.glob("sway-ipc.*.sock"):
            if sock.is_socket():
                os.environ["SWAYSOCK"] = str(sock)
                break

    # If i3 is active, locate its I3SOCK
    if not os.environ.get("I3SOCK") and runtime_dir.exists():
        i3_dir = runtime_dir / "i3"
        if i3_dir.exists():
            for sock in i3_dir.glob("ipc-socket.*"):
                if sock.is_socket():
                    os.environ["I3SOCK"] = str(sock)
                    break

    # If Niri is active, locate NIRI_SOCKET
    if not os.environ.get("NIRI_SOCKET") and runtime_dir.exists():
        for sock in runtime_dir.glob("niri*.sock"):
            if sock.is_socket():
                os.environ["NIRI_SOCKET"] = str(sock)
                break


def detect_desktop_environment(
    refresh_env: bool = True, *, environ: dict[str, str] | None = None
) -> str:
    """Detects the currently running Desktop Environment or Window Manager."""
    if refresh_env and environ is None:
        ensure_gui_environment()

    env = os.environ if environ is None else environ
    xdg_current = env.get("XDG_CURRENT_DESKTOP", "").lower()
    desktop_session = env.get("DESKTOP_SESSION", "").lower()
    gdm_session = env.get("GDMSESSION", "").lower()

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
    # If swaymsg is mocked or present and SWAYSOCK is set
    if env.get("SWAYSOCK") and (_which("swaymsg", env) or not _which("hyprctl", env)):
        return "sway"
    if env.get("I3SOCK") and (_which("i3-msg", env) or not _which("hyprctl", env)):
        return "i3"
    if env.get("NIRI_SOCKET"):
        return "niri"
    if env.get("HYPRLAND_INSTANCE_SIGNATURE") and (_which("hyprctl", env) or not _which("swaymsg", env)):
        return "hyprland"
    if env.get("KDE_SESSION_VERSION"):
        return "kde_plasma"

    # 3. Session variables
    if "sway" in desktop_session and (_which("swaymsg", env) or not _which("hyprctl", env)):
        return "sway"
    if "i3" in desktop_session and (_which("i3-msg", env) or not _which("hyprctl", env)):
        return "i3"
    if "niri" in desktop_session:
        return "niri"
    if "hyprland" in desktop_session:
        return "hyprland"
    if "kde" in desktop_session or "plasma" in desktop_session:
        return "kde_plasma"
    if "gnome" in desktop_session or "gnome" in gdm_session:
        return "gnome"
    if "cosmic" in desktop_session:
        return "cosmic"

    # A compositor CLI may be installed for occasional remote control while a
    # different desktop is active. Session identity or its socket is required.
    if env.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return "hyprland"
    if env.get("SWAYSOCK"):
        return "sway"
    if env.get("I3SOCK"):
        return "i3"

    return "generic_desktop"


def _clean_str(s: str) -> str:
    """Removes invisible unicode characters (zero-width spaces) and whitespace."""
    if not s:
        return ""
    return s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "").strip()


def _get_app_directories(environ: dict[str, str] | None = None) -> list[Path]:
    """Returns standard XDG application directories compliant with Freedesktop spec."""
    dirs = []
    env = os.environ if environ is None else environ
    data_home = env.get("XDG_DATA_HOME")
    if data_home:
        dirs.append(Path(data_home) / "applications")
    elif environ is None:
        dirs.append(Path.home() / ".local" / "share" / "applications")

    # Scoped environments must opt in to each registry; silently falling back
    # to the host user's or system's app registry would identify the wrong app.
    data_dirs = env.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share" if environ is None else "")
    for d in data_dirs.split(":"):
        if d.strip():
            dirs.append(Path(d.strip()) / "applications")

    return [d for d in dirs if d.exists()]


def _steamapps_directories(home: Optional[Path] = None) -> list[Path]:
    """Find installed Steam library manifests without starting or contacting Steam."""
    home = home or Path.home()
    candidates = [
        home / ".steam" / "steam" / "steamapps",
        home / ".local" / "share" / "Steam" / "steamapps",
        home / ".var" / "app" / "com.valvesoftware.Steam" / ".local" / "share" / "Steam" / "steamapps",
    ]
    roots = list(candidates)
    for steamapps in candidates:
        library_file = steamapps / "libraryfolders.vdf"
        try:
            content = library_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw_path in re.findall(r'"path"\s+"((?:\\.|[^"\\])*)"', content):
            path = raw_path.replace("\\\\", "\\").replace('\\"', '"')
            roots.append(Path(path) / "steamapps")

    found: list[Path] = []
    seen: set[str] = set()
    for directory in roots:
        try:
            canonical = str(directory.resolve())
        except OSError:
            canonical = str(directory)
        if canonical not in seen and directory.is_dir():
            seen.add(canonical)
            found.append(directory)
    return found


def _scan_steam_entries(home: Optional[Path] = None) -> dict[str, dict]:
    """Expose installed Steam titles as ordinary launchable desktop applications."""
    apps: dict[str, dict] = {}
    for steamapps in _steamapps_directories(home):
        for manifest in steamapps.glob("appmanifest_*.acf"):
            try:
                content = manifest.read_text(encoding="utf-8", errors="replace")
                app_id = re.search(r'"appid"\s+"(\d+)"', content)
                name = re.search(r'"name"\s+"([^"]+)"', content)
                if not app_id or not name:
                    continue
                title = name.group(1).strip()
                if not title:
                    continue
                key = title.casefold()
                apps.setdefault(key, {
                    "name": title,
                    "exec": "steam",
                    "comment": "Installed Steam application",
                    "desktop_path": str(manifest),
                    "desktop_id": manifest.name,
                    "steam_app_id": app_id.group(1),
                })
            except OSError:
                continue
    return apps


def _scan_desktop_entries(environ: dict[str, str] | None = None) -> dict[str, dict]:
    """Scans all standard desktop entries across the system."""
    apps = {}
    for app_dir in _get_app_directories(environ):
        for p in app_dir.glob("*.desktop"):
            try:
                name, exec_cmd, comment, nodisplay = None, None, "", False
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("Name=") and name is None:
                            name = line[5:].strip()
                        elif line.startswith("Exec=") and exec_cmd is None:
                            exec_cmd = line[5:].strip()
                        elif line.startswith("Comment=") and not comment:
                            comment = line[8:].strip()
                        elif line.startswith("NoDisplay=true"):
                            nodisplay = True
                if name and exec_cmd and not nodisplay:
                    clean_exec = re.sub(r"@@[a-zA-Z0-9]*\s*@@", "", exec_cmd)
                    clean_exec = re.sub(r"--[a-zA-Z0-9_-]+=%[a-zA-Z0-9%]", "", clean_exec)
                    clean_exec = re.sub(r"-[a-zA-Z0-9]=%[a-zA-Z0-9%]", "", clean_exec)
                    clean_exec = re.sub(r"%[a-zA-Z0-9%]", "", clean_exec)
                    clean_exec = re.sub(r"--[a-zA-Z0-9_-]+=(?:\s|$)", " ", clean_exec)
                    clean_exec = re.sub(r"\s+", " ", clean_exec).strip()
                    key = name.lower()
                    if key not in apps:
                        apps[key] = {
                            "name": name,
                            "exec": clean_exec,
                            "comment": comment,
                            "desktop_path": str(p),
                            "desktop_id": p.name
                        }
            except Exception:
                continue
    if environ is None:
        for key, entry in _scan_steam_entries().items():
            apps.setdefault(key, entry)
    return apps


APPLICATION_ALIASES: dict[str, list[str]] = {}
WINDOW_ALIASES: dict[str, list[str]] = {}
DESKTOP_MACROS: dict[str, Any] = {}
UNIVERSAL_SYSTEM_MACROS: dict[str, str] = {
    "lock": "loginctl lock-session",
    "suspend": "systemctl suspend",
    "reboot": "systemctl reboot",
    "restart": "systemctl reboot",
    "poweroff": "systemctl poweroff",
    "shutdown": "systemctl poweroff",
}
DEFAULT_BROWSER: str = "microsoft-edge-stable"
_DISABLED_CAPABILITIES: set[str] = set()
_DISABLED_TOOLS: set[str] = set()


def configure_default_browser(browser_name: Optional[str] = None):
    """Configures the default browser command or application."""
    global DEFAULT_BROWSER, APPLICATION_ALIASES, WINDOW_ALIASES
    if browser_name:
        DEFAULT_BROWSER = browser_name.strip()
        clean = DEFAULT_BROWSER.lower()
        clean_stem = Path(clean).stem

        # Prepend to APPLICATION_ALIASES["browser"]
        existing_apps = APPLICATION_ALIASES.get("browser", [])
        to_add = [DEFAULT_BROWSER, clean, clean_stem]
        if "edge" in clean:
            to_add.extend(["microsoft edge", "microsoft-edge", "microsoft-edge-stable"])
        elif "chrome" in clean:
            to_add.extend(["google chrome", "google-chrome", "chrome"])
        elif "firefox" in clean:
            to_add.extend(["firefox", "firefox-esr"])

        merged_apps = []
        for a in to_add + existing_apps:
            if a and a not in merged_apps:
                merged_apps.append(a)
        APPLICATION_ALIASES["browser"] = merged_apps

        # Prepend to WINDOW_ALIASES["browser"]
        existing_wins = WINDOW_ALIASES.get("browser", [])
        to_add_wins = [clean, clean_stem]
        if "edge" in clean:
            to_add_wins.extend(["microsoft-edge", "microsoftedge", "edge"])
        elif "chrome" in clean:
            to_add_wins.extend(["google-chrome", "chrome"])
        elif "firefox" in clean:
            to_add_wins.extend(["firefox"])

        merged_wins = []
        for w in to_add_wins + existing_wins:
            if w and w not in merged_wins:
                merged_wins.append(w)
        WINDOW_ALIASES["browser"] = merged_wins


def get_default_browser() -> str:
    """Returns the configured default browser."""
    return DEFAULT_BROWSER


def configure_desktop_aliases(app_aliases: Optional[dict[str, list[str]]] = None, win_aliases: Optional[dict[str, list[str]]] = None):
    """Configures application and window aliases dynamically from user settings."""
    global APPLICATION_ALIASES, WINDOW_ALIASES
    if app_aliases:
        APPLICATION_ALIASES.update({k.lower().strip(): [x.lower().strip() for x in v] for k, v in app_aliases.items()})
    if win_aliases:
        WINDOW_ALIASES.update({k.lower().strip(): [x.lower().strip() for x in v] for k, v in win_aliases.items()})


def configure_desktop_macros(macros: Optional[dict[str, Any]] = None):
    """Configures user-defined desktop macros."""
    global DESKTOP_MACROS
    if macros:
        DESKTOP_MACROS.update(macros)


def configure_disabled_capabilities(caps: Optional[list[str]] = None, tools: Optional[list[str]] = None):
    """Configures explicitly disabled capabilities or tools."""
    global _DISABLED_CAPABILITIES, _DISABLED_TOOLS
    if caps:
        _DISABLED_CAPABILITIES.update([c.lower().strip() for c in caps])
    if tools:
        _DISABLED_TOOLS.update([t.lower().strip() for t in tools])


def load_desktop_aliases_from_config(config_path: str = "config.yaml"):
    """Loads user-configured application and window aliases from config.yaml if present."""
    load_desktop_config(config_path)


def load_desktop_config(config_path: str = "config.yaml"):
    """Loads all desktop settings (aliases, macros, disabled capabilities) from config.yaml."""
    try:
        p = Path(config_path).expanduser()
        if p.exists():
            import yaml
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            desktop_cfg = data.get("desktop", {})
            configure_default_browser(desktop_cfg.get("default_browser"))
            configure_desktop_aliases(
                desktop_cfg.get("application_aliases"),
                desktop_cfg.get("window_aliases")
            )
            configure_desktop_macros(desktop_cfg.get("macros"))
            configure_disabled_capabilities(
                desktop_cfg.get("disabled_capabilities"),
                desktop_cfg.get("disabled_tools")
            )
    except Exception:
        pass


# Initialize from config.yaml on import
load_desktop_config()


def _resolve_browser_entry(
    apps: dict[str, dict], environ: dict[str, str] | None = None
) -> Optional[dict]:
    """Resolves the default browser application entry, searching desktop entries and system PATH."""
    target_browser = DEFAULT_BROWSER.strip()
    if not target_browser:
        return None

    target_lower = target_browser.lower()
    target_stem = Path(target_lower).stem

    # 1. Check exact key in apps
    if target_lower in apps:
        return apps[target_lower]

    # 2. Check if any scanned app matches executable binary or desktop_id
    for k, v in apps.items():
        v_exec = v.get("exec", "")
        v_id = v.get("desktop_id", "").lower()
        exec_binary = Path(v_exec.split()[0]).name.lower() if v_exec else ""
        if (
            target_lower in v_id
            or target_stem in v_id
            or target_lower == exec_binary
            or target_stem == exec_binary
            or target_lower == v.get("name", "").lower()
        ):
            return v

    # 3. Direct scan of .desktop files for matching Exec or desktop_id (handles shadowed entries)
    for app_dir in _get_app_directories(environ):
        for p in app_dir.glob("*.desktop"):
            if not p.is_file():
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if (
                    target_lower in p.name.lower()
                    or target_stem in p.name.lower()
                    or f"/{target_lower}" in content
                    or f"/{target_stem}" in content
                ):
                    name, exec_cmd = None, None
                    for line in content.splitlines():
                        if line.startswith("Name=") and name is None:
                            name = line[5:].strip()
                        elif line.startswith("Exec=") and exec_cmd is None:
                            exec_cmd = line[5:].strip()
                    if name and exec_cmd:
                        clean_exec = re.sub(r"@@[a-zA-Z0-9]*\s*@@", "", exec_cmd)
                        clean_exec = re.sub(r"%[a-zA-Z0-9%]", "", clean_exec).strip()
                        return {
                            "name": name,
                            "exec": clean_exec,
                            "comment": "Default Web Browser",
                            "desktop_path": str(p),
                            "desktop_id": p.name,
                        }
            except Exception:
                continue

    # 4. Fallback to shutil.which in system PATH
    bin_path = _which(target_browser, environ) or _which(target_stem, environ)
    if bin_path:
        return {
            "name": target_browser,
            "exec": bin_path,
            "comment": "Default Web Browser",
            "desktop_path": "",
            "desktop_id": f"{target_stem}.desktop",
        }

    return None


def _is_default_browser_target(target: str) -> bool:
    """Recognize common aliases for the configured stable browser entry."""
    def canonical(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
        return normalized[:-6] if normalized.endswith("stable") else normalized

    target_names = {str(target or "").strip().lower()}
    target_names.update(APPLICATION_ALIASES.get(next(iter(target_names)), []))
    target_tokens = {canonical(name) for name in target_names}
    for alias, values in APPLICATION_ALIASES.items():
        if any(canonical(value) in target_tokens for value in values):
            target_names.add(alias)

    default_name = canonical(DEFAULT_BROWSER)
    return bool(default_name) and any(canonical(name) == default_name for name in target_names)


def _resolve_application_entry(
    target: str, apps: dict[str, dict], environ: dict[str, str] | None = None
) -> Optional[dict]:
    """Resolves an app name to a desktop entry using exact matches, aliases, length-ranked substrings, and PATH fallback."""
    clean_target = (target or "").strip().lower()
    if not clean_target:
        return None

    # Check if target is a generic browser request or matches the default browser directly
    if clean_target in ["browser", "web browser", "web-browser", "default browser", "internet"]:
        browser_entry = _resolve_browser_entry(apps, environ)
        if browser_entry:
            return browser_entry
    elif (
        clean_target == DEFAULT_BROWSER.lower()
        or clean_target == Path(DEFAULT_BROWSER.lower()).stem
        or _is_default_browser_target(clean_target)
    ):
        browser_entry = _resolve_browser_entry(apps, environ)
        if browser_entry:
            return browser_entry

    # 1. Expand aliases
    targets_to_try = [clean_target]
    if clean_target in APPLICATION_ALIASES:
        targets_to_try.extend(APPLICATION_ALIASES[clean_target])
    for alias_k, alias_targets in APPLICATION_ALIASES.items():
        if alias_k in clean_target:
            targets_to_try.extend(alias_targets)

    seen = set()
    cleaned_targets = []
    for t in targets_to_try:
        if t not in seen:
            seen.add(t)
            cleaned_targets.append(t)

    # 2. Exact match check
    for t in cleaned_targets:
        if t in apps:
            return apps[t]

    # 3. Substring scoring
    candidates = []
    for t in cleaned_targets:
        for k, v in apps.items():
            if t == k:
                candidates.append((0, len(k), v))
            elif k.startswith(t):
                penalty = 100 if "uninstall" in k or "installer" in k else 0
                candidates.append((1 + penalty, len(k), v))
            elif t in k:
                penalty = 100 if "uninstall" in k or "installer" in k else 0
                candidates.append((2 + penalty, len(k), v))
            elif k in t:
                candidates.append((3, len(k), v))

    if candidates:
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0][2]

    # 4. Check system PATH binary
    bin_path = _which(clean_target, environ)
    if bin_path:
        return {
            "name": clean_target,
            "exec": bin_path,
            "comment": "",
            "desktop_path": "",
            "desktop_id": clean_target
        }

    return None


def _find_best_window(target: str, windows: list[dict]) -> Optional[dict]:
    """Finds the best matching window candidate using exact, normalized, alias, and substring ranking."""
    clean_target = _clean_str(target).lower()
    norm_target = re.sub(r"[^a-z0-9]", "", clean_target)
    if not norm_target:
        return None

    # 1. Direct address match
    if clean_target.startswith("0x"):
        for w in windows:
            if clean_target == w.get("address", "").lower():
                return w

    # 2. Exact normalized class match
    for w in windows:
        c_cls = re.sub(r"[^a-z0-9]", "", _clean_str(w.get("class", "")).lower())
        if norm_target == c_cls:
            return w

    # 3. Check semantic aliases
    for key, alias_list in WINDOW_ALIASES.items():
        if norm_target == key or key in norm_target:
            for w in windows:
                c_cls = re.sub(r"[^a-z0-9]", "", _clean_str(w.get("class", "")).lower())
                if any(a in c_cls for a in alias_list):
                    return w

    # 4. Substring match in class, initialClass, or title
    for w in windows:
        cls_clean = _clean_str(w.get("class", "")).lower()
        init_cls = _clean_str(w.get("initialClass", "")).lower()
        title_clean = _clean_str(w.get("title", "")).lower()
        if clean_target in cls_clean or clean_target in init_cls or clean_target in title_clean:
            return w
        norm_title = re.sub(r"[^a-z0-9]", "", title_clean)
        if norm_target in norm_title or norm_target in re.sub(r"[^a-z0-9]", "", cls_clean):
            return w

    # 5. Token match: all words in target exist in combined class and title
    words = clean_target.split()
    if len(words) > 1:
        for w in windows:
            combined = f"{_clean_str(w.get('class', '')).lower()} {_clean_str(w.get('title', '')).lower()}"
            if all(w_token in combined for w_token in words):
                return w

    return None


def _find_qdbus() -> Optional[str]:
    """Locates qdbus6 or qdbus executable."""
    for b in ["qdbus6", "qdbus-qt6", "qdbus", "qdbus-qt5"]:
        p = shutil.which(b)
        if p:
            return p
    return None


# ---------------------------------------------------------------------------
# Compositor / Desktop Environment Backends
# ---------------------------------------------------------------------------

class BaseDesktopBackend:
    name: str = "generic_desktop"

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        # Keep capture subprocesses on the same display session used to choose
        # this backend. Controllers and isolated GUI evaluations may provide a
        # private Xvfb environment that differs from the process environment.
        self.environ = os.environ if environ is None else environ
        self.last_screenshot_origin = (0, 0)
        self.last_screenshot_bounds: tuple[int, int, int, int] | None = None
        self.last_screenshot_target = ""
        self.last_screenshot_scale = (1.0, 1.0)
        self.last_screenshot_monitor_id: str | None = None

    def get_capabilities(self) -> set[str]:
        return {
            "window_focus", "window_close", "window_list",
            "workspace_switch", "window_move", "screenshot",
            "desktop_macro", "app_launch", "app_list"
        }

    def supports(self, capability: str) -> bool:
        if capability == "workspace_control":
            caps = self.get_capabilities()
            return "workspace_switch" in caps or "window_move" in caps
        return capability in self.get_capabilities()

    def focus_window(self, target: str) -> str:
        return f"Window focusing is not supported on {self.name}."

    def workspace_control(self, action: str, workspace: str, target: str, is_current: bool = False, is_back: bool = False) -> str:
        return f"Workspace management is not supported on {self.name}."

    def close_application(self, app_name: str) -> str:
        subprocess.run(["pkill", "-f", app_name], check=False)
        return f"Closed processes matching '{app_name}'."

    def list_windows(self) -> str:
        return "No open desktop windows found or window management is not available in the current session."

    def get_open_windows_prompt_context(self) -> str:
        win_list = self.list_windows()
        return f"Desktop State:\n{win_list}"

    def swap_windows(self, window_one: str, window_two: str) -> str:
        return f"Window swapping is not supported on {self.name}."

    def capture_screenshot(self) -> bytes:
        raise RuntimeError(f"Screenshot capture is not supported on {self.name}.")

    def capture_desktop_screenshot(self) -> bytes:
        """Capture the full desktop only when the caller explicitly requests it."""
        self.last_screenshot_origin = (0, 0)
        self.last_screenshot_bounds = None
        self.last_screenshot_target = "full desktop"
        return self.capture_screenshot()

    def get_default_macros(self) -> dict[str, str]:
        return dict(UNIVERSAL_SYSTEM_MACROS)


class HyprlandBackend(BaseDesktopBackend):
    name = "hyprland"

    def get_capabilities(self) -> set[str]:
        return {
            "window_focus", "window_move", "workspace_switch",
            "window_close", "window_list", "window_swap",
            "screenshot", "desktop_macro", "app_launch", "app_list"
        }

    @staticmethod
    def _run_grim(command: list[str], *, full_desktop: bool = False, environ: dict[str, str] | None = None) -> bytes:
        label = "Full-desktop" if full_desktop else "Focused-monitor"
        try:
            result = subprocess.run(
        command, capture_output=True, timeout=3, env=os.environ if environ is None else environ,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{label} screenshot capture timed out. Check that Hyprland allows "
                "grim screen capture and that the display is awake."
            ) from exc
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or f"{label} screenshot capture failed.")
        if not result.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"{label} screenshot capture returned invalid PNG data.")
        return result.stdout

    def _record_monitor_scale(self, monitor: dict[str, Any]) -> None:
        scale = float(monitor.get("scale", 1.0) or 1.0)
        if scale <= 0:
            raise RuntimeError("Hyprland reported an invalid monitor scale.")
        self.last_screenshot_scale = (scale, scale)

    def focus_window(self, target: str) -> str:
        import json
        t = (target or "").strip()
        try:
            clients_raw = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True).stdout
            clients = json.loads(clients_raw) if clients_raw else []
            match = _find_best_window(t, clients)
            if match:
                addr = match.get("address", "")
                ws = match.get("workspace", {}).get("id", "?")
                title = match.get("title") or match.get("class", "Window")
                selector = f"address:{addr}"
                res = _hyprland_dispatch("focuswindow", selector, lua_expression=f"hl.dsp.focus({{ window = {_lua_string(selector)} }})")
                if "ok" in res.stdout.lower() or res.returncode == 0:
                    return f"Focused '{title[:40]}' on workspace {ws}."
                selector = f"class:{match.get('class')}"
                res = _hyprland_dispatch("focuswindow", selector, lua_expression=f"hl.dsp.focus({{ window = {_lua_string(selector)} }})")
                if "ok" in res.stdout.lower() or res.returncode == 0:
                    return f"Focused '{title[:40]}' on workspace {ws}."
                return f"Failed to focus window: {res.stdout.strip() or res.stderr.strip()}"
            else:
                selector = f"class:{t}"
                res = _hyprland_dispatch("focuswindow", selector, lua_expression=f"hl.dsp.focus({{ window = {_lua_string(selector)} }})")
                if "ok" in res.stdout.lower():
                    return f"Focused window matching '{t}'."
                open_apps = list({c.get("class", "") for c in clients if c.get("class")})
                return f"No open window found matching '{t}'. Open applications are: {', '.join(open_apps)}."
        except Exception as e:
            return f"Error focusing window in Hyprland: {e}"

    def workspace_control(self, action: str, workspace: str, target: str, is_current: bool = False, is_back: bool = False) -> str:
        import json
        ws = workspace
        if is_current:
            try:
                res_ws = subprocess.run(["hyprctl", "activeworkspace", "-j"], capture_output=True, text=True)
                if res_ws.returncode == 0 and res_ws.stdout:
                    ws_data = json.loads(res_ws.stdout)
                    ws = str(ws_data.get("id", "1"))
            except Exception:
                ws = "1"

        if action in ["move", "movetoworkspace"]:
            if target:
                try:
                    clients_raw = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True).stdout
                    clients = json.loads(clients_raw) if clients_raw else []
                    match = _find_best_window(target, clients)
                    if match:
                        addr = match.get("address", "")
                        title = match.get("title") or match.get("class", "Window")
                        curr_ws = str(match.get("workspace", {}).get("id") or match.get("workspace", {}).get("name", ""))
                        if curr_ws and curr_ws != ws:
                            _WINDOW_PREVIOUS_WORKSPACES[addr.lower()] = curr_ws
                            _WINDOW_PREVIOUS_WORKSPACES[match.get("class", "").lower()] = curr_ws
                            _WINDOW_PREVIOUS_WORKSPACES[target.lower()] = curr_ws
                            global _LAST_MOVED_WINDOW
                            _LAST_MOVED_WINDOW = {
                                "address": addr,
                                "class": match.get("class"),
                                "title": title,
                                "previous_workspace": curr_ws
                            }

                        selector = f"address:{addr}"
                        res = _hyprland_dispatch("movetoworkspacesilent", f"{ws},{selector}", lua_expression=f"hl.dsp.window.move({{ workspace = {_lua_string(ws)}, window = {_lua_string(selector)}, follow = false }})")
                        stdout_clean = res.stdout.strip().lower()
                        if "ok" in stdout_clean and "invalid" not in stdout_clean:
                            return f"Moved '{title[:40]}' to workspace {ws}."
                        return f"Failed to move window: {res.stdout.strip() or res.stderr.strip()}"
                    else:
                        return f"Could not find window matching '{target}' to move."
                except Exception as e:
                    return f"Error moving window in Hyprland: {e}"
            else:
                res = _hyprland_dispatch("movetoworkspace", ws, lua_expression=f"hl.dsp.window.move({{ workspace = {_lua_string(ws)} }})")
                stdout_clean = res.stdout.strip().lower()
                if "ok" in stdout_clean and "invalid" not in stdout_clean:
                    return f"Moved active window to workspace {ws}."
                return f"Failed to move active window: {res.stdout.strip() or res.stderr.strip()}"
        else:
            res = _hyprland_dispatch("workspace", ws, lua_expression=f"hl.dsp.focus({{ workspace = {_lua_string(ws)} }})")
            stdout_clean = res.stdout.strip().lower()
            if "ok" in stdout_clean and "invalid" not in stdout_clean:
                return f"Switched to workspace {ws}."
            return f"Failed to switch workspace: {res.stdout.strip() or res.stderr.strip()}"

    def close_application(self, app_name: str) -> str:
        import json
        target = (app_name or "").strip()
        is_pronoun_or_active = not target or target.lower() in [
            "it", "this", "active", "current", "this window", "active window", "focused window", "the app", "the window"
        ]
        try:
            if is_pronoun_or_active:
                act_raw = subprocess.run(["hyprctl", "activewindow", "-j"], capture_output=True, text=True, timeout=1).stdout
                act = json.loads(act_raw) if act_raw else {}
                addr = act.get("address", "")
                title = act.get("title") or act.get("class", "Active Window")
                if addr:
                    selector = f"address:{addr}"
                    res = _hyprland_dispatch("closewindow", selector, lua_expression=f"hl.dsp.window.close({{ window = {_lua_string(selector)} }})")
                    if "ok" in res.stdout.lower() or res.returncode == 0:
                        return f"Closed active window for {title[:40]}."
                res = _hyprland_dispatch("killactive", lua_expression="hl.dsp.window.close()")
                if "ok" in res.stdout.lower() or res.returncode == 0:
                    return "Closed active window."

            clients_raw = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, timeout=1).stdout
            clients = json.loads(clients_raw) if clients_raw else []
            match = _find_best_window(target, clients)
            if match:
                addr = match.get("address", "")
                title = match.get("title") or match.get("class", "Window")
                selector = f"address:{addr}"
                res = _hyprland_dispatch("closewindow", selector, lua_expression=f"hl.dsp.window.close({{ window = {_lua_string(selector)} }})")
                if "ok" in res.stdout.lower() or res.returncode == 0:
                    return f"Closed window for {title[:40]}."
        except Exception:
            pass
        subprocess.run(["pkill", "-f", target], check=False)
        return f"Closed processes matching '{app_name}'."

    def list_windows(self) -> str:
        import json
        try:
            res = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True)
            clients = json.loads(res.stdout) if res.stdout else []
            if clients:
                lines = []
                for c in clients:
                    w_id = c.get("workspace", {}).get("id", "?")
                    title = c.get("title", "Untitled")
                    cls = c.get("class", "Unknown")
                    lines.append(f"Workspace {w_id}: {cls} - {title[:50]}")
                return f"Open windows ({len(clients)} total):\n" + "\n".join(lines[:15])
        except Exception:
            pass
        return "No open desktop windows found in Hyprland."

    def get_open_windows_prompt_context(self) -> str:
        import json
        try:
            clients_res = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, timeout=1)
            active_res = subprocess.run(["hyprctl", "activewindow", "-j"], capture_output=True, text=True, timeout=1)
            ws_res = subprocess.run(["hyprctl", "activeworkspace", "-j"], capture_output=True, text=True, timeout=1)

            clients = json.loads(clients_res.stdout) if clients_res.stdout else []
            active = json.loads(active_res.stdout) if active_res.stdout else {}
            active_ws = json.loads(ws_res.stdout) if ws_res.stdout else {}

            ws_id = active_ws.get("id", "?")
            active_addr = active.get("address", "")
            active_title = active.get("title", "None")
            active_class = active.get("class", "None")

            lines = [f"Active Workspace: {ws_id} | Focused Window: {active_class} (title: \"{active_title[:45]}\")"]
            if not clients:
                lines.append("Open Windows: None (desktop is empty)")
            else:
                lines.append(f"Open Windows ({len(clients)} total):")
                clients_sorted = sorted(clients, key=lambda c: (c.get("workspace", {}).get("id", 99), c.get("class", "").lower()))
                for c in clients_sorted:
                    c_ws = c.get("workspace", {}).get("id", "?")
                    c_cls = c.get("class", "Unknown")
                    c_title = c.get("title", "Untitled")
                    c_addr = c.get("address", "")
                    is_focused = " [FOCUSED]" if c_addr == active_addr else ""
                    lines.append(f"- Workspace {c_ws}: {c_cls}{is_focused} (address: {c_addr}) - \"{c_title[:50]}\"")
            return "\n".join(lines)
        except Exception:
            pass
        return self.list_windows()

    def swap_windows(self, window_one: str, window_two: str) -> str:
        import json
        first_target = (window_one or "").strip()
        second_target = (window_two or "").strip()
        if not first_target or not second_target:
            return "Please specify both windows to reorder."

        try:
            clients_result = subprocess.run(
                ["hyprctl", "clients", "-j"], capture_output=True, text=True, timeout=2
            )
            if clients_result.returncode != 0:
                return f"Could not list Hyprland windows: {clients_result.stderr.strip()}"
            clients = json.loads(clients_result.stdout or "[]")
            first = _find_best_window(first_target, clients)
            second = _find_best_window(second_target, clients)
            if not first:
                return f"Could not find a window matching '{first_target}'."
            if not second:
                return f"Could not find a window matching '{second_target}'."
            first_address = first.get("address", "")
            second_address = second.get("address", "")
            if not first_address or not second_address:
                return "Could not identify both windows safely."
            if first_address == second_address:
                return "Those names match the same window, so there is nothing to swap."
            if first.get("workspace", {}).get("id") != second.get("workspace", {}).get("id"):
                return "Both windows need to be on the same workspace before I can reorder them."

            first_selector = f"address:{first_address}"
            focus = _hyprland_dispatch(
                "focuswindow", first_selector,
                lua_expression=f"hl.dsp.focus({{ window = {_lua_string(first_selector)} }})",
            )
            if focus.returncode != 0 or "ok" not in focus.stdout.lower():
                return f"Could not focus the first window: {focus.stdout.strip() or focus.stderr.strip()}"

            second_selector = f"address:{second_address}"
            swap = _hyprland_dispatch(
                "swapwindow", second_selector,
                lua_expression=f"hl.dsp.window.swap({{ target = {_lua_string(second_selector)} }})",
            )
            result = f"{swap.stdout} {swap.stderr}".strip()
            if swap.returncode != 0 or "ok" not in swap.stdout.lower() or any(
                marker in result.lower() for marker in ("invalid", "can't", "cannot", "error")
            ):
                return f"Could not swap the windows: {result or 'Hyprland rejected the request.'}"

            first_name = first.get("title") or first.get("class") or first_target
            second_name = second.get("title") or second.get("class") or second_target
            return f"Swapped the positions of '{first_name[:35]}' and '{second_name[:35]}'."
        except Exception as e:
            return f"Could not reorder windows in Hyprland: {e}"

    def capture_screenshot(self) -> bytes:
        geometry = None
        output_name = None
        try:
            active_result = subprocess.run(
                ["hyprctl", "activewindow", "-j"], capture_output=True, text=True, timeout=2,
                env=self.environ,
            )
            if active_result.returncode != 0:
                raise RuntimeError("Hyprland did not report the focused window.")
            monitor_id = json.loads(active_result.stdout or "{}").get("monitor")
            monitors_result = subprocess.run(
                ["hyprctl", "monitors", "-j"], capture_output=True, text=True, timeout=2,
                env=self.environ,
            )
            if monitors_result.returncode != 0:
                raise RuntimeError("Hyprland did not report its monitor layout.")
            monitors = json.loads(monitors_result.stdout or "[]")
            monitor = next((item for item in monitors if item.get("id") == monitor_id), None)
            if monitor:
                if monitor.get("dpmsStatus") is False:
                    raise RuntimeError(
                        "The focused monitor is powered off (DPMS); wake the display before requesting a screenshot."
                    )
                x, y = int(monitor.get("x", 0)), int(monitor.get("y", 0))
                width, height = int(monitor.get("width", 0)), int(monitor.get("height", 0))
                output_name = str(monitor.get("name") or "").strip() or None
                if width > 0 and height > 0:
                    geometry = (x, y, width, height)
        except Exception as exc:
            raise RuntimeError(f"Could not verify the focused Hyprland monitor: {exc}") from exc
        if geometry is None:
            raise RuntimeError("Could not verify the focused Hyprland monitor; refusing an unscoped screenshot.")
        x, y, width, height = geometry
        self.last_screenshot_monitor_id = str(monitor.get("id"))
        self._record_monitor_scale(monitor)
        self.last_screenshot_origin = (x, y)
        self.last_screenshot_bounds = (x, y, x + width, y + height)
        self.last_screenshot_target = output_name or f"monitor at {x},{y}"
        # Selecting the compositor output is reliable on mixed-resolution
        # setups; use its geometry only when no output name is available.
        command = (
            ["grim", "-l", "1", "-o", output_name, "-"]
            if output_name
            else ["grim", "-l", "1", "-g", f"{x},{y} {width}x{height}", "-"]
        )
        return self._run_grim(command, environ=self.environ)

    def capture_desktop_screenshot(self) -> bytes:
        self.last_screenshot_origin = (0, 0)
        self.last_screenshot_bounds = None
        self.last_screenshot_target = "full desktop"
        self.last_screenshot_scale = (1.0, 1.0)
        self.last_screenshot_monitor_id = None
        return self._run_grim(["grim", "-l", "1", "-"], full_desktop=True, environ=self.environ)

    def get_default_macros(self) -> dict[str, str]:
        return {
            "toggle_floating": "hyprctl dispatch togglefloating active",
            "fullscreen": "hyprctl dispatch fullscreen 1",
            "pin": "hyprctl dispatch pin active",
            "split": "hyprctl dispatch layoutmsg togglesplit",
            "lock": "hyprlock || loginctl lock-session"
        }


class SwayBackend(BaseDesktopBackend):
    name = "sway"

    def get_capabilities(self) -> set[str]:
        return {
            "window_focus", "window_move", "workspace_switch",
            "window_close", "window_list", "window_swap",
            "screenshot", "desktop_macro", "app_launch", "app_list"
        }

    def _extract_nodes(self, tree: dict) -> list[dict]:
        nodes = []
        def extract(n, current_ws="?"):
            if n.get("type") == "workspace":
                current_ws = n.get("name", "?")
            if n.get("app_id") or n.get("window_properties", {}).get("class"):
                nodes.append({
                    "id": n.get("id"),
                    "class": n.get("app_id") or n.get("window_properties", {}).get("class", ""),
                    "title": n.get("name", ""),
                    "workspace": current_ws,
                    "focused": n.get("focused", False)
                })
            for child in n.get("nodes", []): extract(child, current_ws)
            for child in n.get("floating_nodes", []): extract(child, current_ws)
        extract(tree)
        return nodes

    def focus_window(self, target: str) -> str:
        import json
        t = (target or "").strip()
        try:
            tree_raw = subprocess.run(["swaymsg", "-t", "get_tree"], capture_output=True, text=True).stdout
            if tree_raw:
                tree = json.loads(tree_raw)
                nodes = self._extract_nodes(tree)
                match = _find_best_window(t, nodes)
                if match:
                    res = subprocess.run(["swaymsg", f"[con_id={match['id']}] focus"], capture_output=True, text=True)
                    if res.returncode == 0 and "false" not in res.stdout.lower():
                        return f"Focused '{match['title'][:40]}' in Sway."
        except Exception:
            pass

        res = subprocess.run(["swaymsg", f"[app_id=\"(?i).*{t}.*\"] focus"], capture_output=True, text=True)
        if res.returncode == 0 and "false" not in res.stdout.lower():
            return f"Focused window matching '{t}' in Sway."
        res = subprocess.run(["swaymsg", f"[class=\"(?i).*{t}.*\"] focus"], capture_output=True, text=True)
        if res.returncode == 0 and "false" not in res.stdout.lower():
            return f"Focused window matching '{t}' in Sway."
        return f"Could not find window matching '{t}' in Sway."

    def workspace_control(self, action: str, workspace: str, target: str, is_current: bool = False, is_back: bool = False) -> str:
        import json
        ws = workspace
        if is_current:
            try:
                res_ws = subprocess.run(["swaymsg", "-t", "get_workspaces"], capture_output=True, text=True)
                if res_ws.returncode == 0 and res_ws.stdout:
                    workspaces = json.loads(res_ws.stdout)
                    focused_ws = next((w for w in workspaces if w.get("focused")), None)
                    if focused_ws:
                        ws = str(focused_ws.get("name", "1"))
            except Exception:
                ws = "1"

        if action in ["move", "movetoworkspace"]:
            if target:
                try:
                    tree_raw = subprocess.run(["swaymsg", "-t", "get_tree"], capture_output=True, text=True).stdout
                    if tree_raw:
                        tree = json.loads(tree_raw)
                        nodes = self._extract_nodes(tree)
                        match = _find_best_window(target, nodes)
                        if match:
                            res = subprocess.run(["swaymsg", f"[con_id={match['id']}] move container to workspace {ws}"], capture_output=True, text=True)
                            if res.returncode == 0 and "false" not in res.stdout.lower():
                                return f"Moved '{match['title'][:40]}' to workspace {ws} in Sway."
                            return f"Failed to move window in Sway: {res.stdout.strip()}"
                        return f"Could not find window matching '{target}' to move."
                except Exception as e:
                    return f"Error moving window in Sway: {e}"
            else:
                subprocess.run(["swaymsg", "move", "container", "to", "workspace", ws], check=False)
                return f"Moved active window to workspace {ws} in Sway."
        else:
            subprocess.run(["swaymsg", "workspace", ws], check=False)
            return f"Switched to workspace {ws} in Sway."

    def close_application(self, app_name: str) -> str:
        import json
        target = (app_name or "").strip()
        is_pronoun_or_active = not target or target.lower() in [
            "it", "this", "active", "current", "this window", "active window", "focused window", "the app", "the window"
        ]
        try:
            if is_pronoun_or_active:
                res = subprocess.run(["swaymsg", "kill"], capture_output=True, text=True, timeout=1)
                if res.returncode == 0:
                    return "Closed active window in Sway."
            tree_raw = subprocess.run(["swaymsg", "-t", "get_tree"], capture_output=True, text=True, timeout=1).stdout
            if tree_raw:
                tree = json.loads(tree_raw)
                nodes = self._extract_nodes(tree)
                match = _find_best_window(target, nodes)
                if match:
                    res = subprocess.run(["swaymsg", f"[con_id={match['id']}] kill"], capture_output=True, text=True, timeout=1)
                    if res.returncode == 0 and "false" not in res.stdout.lower():
                        return f"Closed window for '{match['title'][:40]}' in Sway."
        except Exception:
            pass
        subprocess.run(["pkill", "-f", target], check=False)
        return f"Closed processes matching '{app_name}'."

    def list_windows(self) -> str:
        import json
        try:
            res = subprocess.run(["swaymsg", "-t", "get_tree"], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout:
                tree = json.loads(res.stdout)
                nodes = self._extract_nodes(tree)
                if nodes:
                    lines = [f"Workspace {w['workspace']}: {w['class']} - {w['title'][:50]}" for w in nodes]
                    return f"Open windows ({len(lines)} total):\n" + "\n".join(lines[:15])
        except Exception:
            pass
        return "No open desktop windows found in Sway."

    def get_open_windows_prompt_context(self) -> str:
        import json
        try:
            tree_res = subprocess.run(["swaymsg", "-t", "get_tree"], capture_output=True, text=True, timeout=1)
            ws_res = subprocess.run(["swaymsg", "-t", "get_workspaces"], capture_output=True, text=True, timeout=1)
            if tree_res.returncode == 0 and tree_res.stdout:
                tree = json.loads(tree_res.stdout)
                workspaces = json.loads(ws_res.stdout) if ws_res.stdout else []
                active_ws = next((w.get("name") for w in workspaces if w.get("focused")), "?")
                nodes = self._extract_nodes(tree)
                focused_info = next((f"{w['class']} (title: \"{w['title'][:45]}\")" for w in nodes if w["focused"]), "None")

                lines = [f"Active Workspace: {active_ws} | Focused Window: {focused_info}"]
                if not nodes:
                    lines.append("Open Windows: None (desktop is empty)")
                else:
                    lines.append(f"Open Windows ({len(nodes)} total):")
                    for w in nodes:
                        foc_tag = " [FOCUSED]" if w["focused"] else ""
                        lines.append(f"- Workspace {w['workspace']}: {w['class']}{foc_tag} - \"{w['title'][:50]}\"")
                return "\n".join(lines)
        except Exception:
            pass
        return self.list_windows()

    def swap_windows(self, window_one: str, window_two: str) -> str:
        import json
        first_target = (window_one or "").strip()
        second_target = (window_two or "").strip()
        if not first_target or not second_target:
            return "Please specify both windows to reorder."
        try:
            tree_raw = subprocess.run(["swaymsg", "-t", "get_tree"], capture_output=True, text=True, timeout=2).stdout
            if not tree_raw:
                return "Could not query Sway layout."
            tree = json.loads(tree_raw)
            nodes = self._extract_nodes(tree)
            first = _find_best_window(first_target, nodes)
            second = _find_best_window(second_target, nodes)
            if not first or not second:
                return f"Could not find matching windows in Sway."
            res = subprocess.run(["swaymsg", f"[con_id={first['id']}] swap container with con_id {second['id']}"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and "false" not in res.stdout.lower():
                return f"Swapped positions of '{first['title'][:35]}' and '{second['title'][:35]}' in Sway."
            return f"Failed to swap windows in Sway: {res.stdout.strip()}"
        except Exception as e:
            return f"Error swapping windows in Sway: {e}"

    def capture_screenshot(self) -> bytes:
        result = subprocess.run(
            ["grim", "-"],
            capture_output=True,
            timeout=15,
            env=self.environ,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or "Screenshot capture failed.")
        if not result.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Screenshot capture returned invalid PNG data.")
        return result.stdout

    def get_default_macros(self) -> dict[str, str]:
        return {
            "toggle_floating": "swaymsg floating toggle",
            "fullscreen": "swaymsg fullscreen toggle",
            "lock": "swaylock || loginctl lock-session"
        }


class I3Backend(BaseDesktopBackend):
    name = "i3"

    def get_capabilities(self) -> set[str]:
        return {
            "window_focus", "window_move", "workspace_switch",
            "window_close", "window_list", "window_swap",
            "screenshot", "desktop_macro", "app_launch", "app_list"
        }

    def focus_window(self, target: str) -> str:
        t = (target or "").strip()
        res = subprocess.run(["i3-msg", f"[class=\"(?i).*{t}.*\"] focus"], capture_output=True, text=True)
        if res.returncode == 0:
            return f"Focused window matching '{t}' in i3."
        res = subprocess.run(["i3-msg", f"[title=\"(?i).*{t}.*\"] focus"], capture_output=True, text=True)
        if res.returncode == 0:
            return f"Focused window matching '{t}' in i3."
        return f"Could not find window matching '{t}' in i3."

    def workspace_control(self, action: str, workspace: str, target: str, is_current: bool = False, is_back: bool = False) -> str:
        import json
        ws = workspace
        if is_current:
            try:
                res_ws = subprocess.run(["i3-msg", "-t", "get_workspaces"], capture_output=True, text=True)
                if res_ws.returncode == 0 and res_ws.stdout:
                    workspaces = json.loads(res_ws.stdout)
                    focused_ws = next((w for w in workspaces if w.get("focused")), None)
                    if focused_ws:
                        ws = str(focused_ws.get("name", "1"))
            except Exception:
                ws = "1"

        if action in ["move", "movetoworkspace"]:
            if target:
                res = subprocess.run(["i3-msg", f"[class=\"(?i).*{target}.*\"] move container to workspace {ws}"], capture_output=True, text=True)
                if res.returncode == 0:
                    return f"Moved window matching '{target}' to workspace {ws} in i3."
                return f"Failed to move window matching '{target}' in i3."
            else:
                subprocess.run(["i3-msg", "move", "container", "to", "workspace", ws], check=False)
                return f"Moved active window to workspace {ws} in i3."
        else:
            subprocess.run(["i3-msg", "workspace", ws], check=False)
            return f"Switched to workspace {ws} in i3."

    def close_application(self, app_name: str) -> str:
        target = (app_name or "").strip()
        is_pronoun_or_active = not target or target.lower() in [
            "it", "this", "active", "current", "this window", "active window", "focused window", "the app", "the window"
        ]
        if is_pronoun_or_active:
            res = subprocess.run(["i3-msg", "kill"], capture_output=True, text=True, timeout=1)
            if res.returncode == 0:
                return "Closed active window in i3."
        res = subprocess.run(["i3-msg", f"[class=\"(?i).*{target}.*\"] kill"], capture_output=True, text=True, timeout=1)
        if res.returncode == 0:
            return f"Closed window matching '{app_name}' in i3."
        subprocess.run(["pkill", "-f", target], check=False)
        return f"Closed processes matching '{app_name}'."

    def list_windows(self) -> str:
        if shutil.which("wmctrl"):
            try:
                res = subprocess.run(["wmctrl", "-l", "-x"], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    lines = []
                    for l in res.stdout.splitlines():
                        parts = l.split(None, 4)
                        if len(parts) >= 5:
                            lines.append(f"Workspace {parts[1]}: {parts[2]} - {parts[4][:50]}")
                    if lines:
                        return f"Open windows ({len(lines)} total):\n" + "\n".join(lines[:15])
            except Exception:
                pass
        return "No open desktop windows found in i3."

    def swap_windows(self, window_one: str, window_two: str) -> str:
        first = (window_one or "").strip()
        second = (window_two or "").strip()
        if not first or not second:
            return "Please specify both windows to swap in i3."
        res = subprocess.run(["i3-msg", f"[class=\"(?i).*{first}.*\"] swap container with class \"(?i).*{second}.*\""], capture_output=True, text=True)
        if res.returncode == 0:
            return f"Swapped '{first}' and '{second}' in i3."
        return f"Could not swap windows in i3: {res.stdout.strip()}"

    def capture_screenshot(self) -> bytes:
        for tool, args in [("maim", ["maim"]), ("scrot", ["scrot", "-"]), ("import", ["import", "-window", "root", "png:-"]), ("grim", ["grim", "-"])]:
            if _which(tool, self.environ):
                res = subprocess.run(args, capture_output=True, timeout=15, env=self.environ)
                if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                    return res.stdout
        raise RuntimeError("Screenshot capture failed in i3. Please install maim or scrot.")

    def get_default_macros(self) -> dict[str, str]:
        return {
            "toggle_floating": "i3-msg floating toggle",
            "fullscreen": "i3-msg fullscreen toggle",
            "lock": "i3lock || loginctl lock-session"
        }


class NiriBackend(BaseDesktopBackend):
    name = "niri"

    def get_capabilities(self) -> set[str]:
        return {
            "window_focus", "window_move", "workspace_switch",
            "window_close", "window_list", "window_swap",
            "screenshot", "desktop_macro", "app_launch", "app_list"
        }

    def focus_window(self, target: str) -> str:
        import json
        t = (target or "").strip()
        if not t:
            return "Please specify a window title or application to focus."
        try:
            res = subprocess.run(["niri", "msg", "-j", "windows"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and res.stdout:
                windows = json.loads(res.stdout)
                norm_windows = []
                for w in windows:
                    norm_windows.append({
                        "id": str(w.get("id")),
                        "class": w.get("app_id", ""),
                        "title": w.get("title", ""),
                        "address": str(w.get("id"))
                    })
                match = _find_best_window(t, norm_windows)
                if match:
                    act_res = subprocess.run(["niri", "msg", "action", "focus-window", "--id", match["id"]], capture_output=True, text=True, timeout=2)
                    if act_res.returncode == 0:
                        return f"Focused '{match['title'][:40]}' in Niri."
        except Exception:
            pass
        return f"Could not find window matching '{t}' in Niri."

    def workspace_control(self, action: str, workspace: str, target: str, is_current: bool = False, is_back: bool = False) -> str:
        ws_clean = workspace.strip()
        if action not in ["move", "movetoworkspace"]:
            res = subprocess.run(["niri", "msg", "action", "focus-workspace", ws_clean], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                return f"Switched to workspace {ws_clean} in Niri."
            return f"Failed to switch to workspace {ws_clean} in Niri."

        tgt = (target or "").strip()
        if tgt:
            self.focus_window(tgt)
        res = subprocess.run(["niri", "msg", "action", "move-column-to-workspace", ws_clean], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            return f"Moved window to workspace {ws_clean} in Niri."
        return f"Failed to move window to workspace {ws_clean} in Niri."

    def close_application(self, app_name: str) -> str:
        target = (app_name or "").strip()
        is_active = not target or target.lower() in [
            "it", "this", "active", "current", "this window", "active window", "focused window", "the app", "the window"
        ]
        if is_active:
            res = subprocess.run(["niri", "msg", "action", "close-window"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                return "Closed active window in Niri."
            return "Failed to close active window in Niri."

        self.focus_window(target)
        res = subprocess.run(["niri", "msg", "action", "close-window"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            return f"Closed window matching '{target}' in Niri."
        subprocess.run(["pkill", "-f", target], check=False)
        return f"Closed processes matching '{app_name}'."

    def swap_windows(self, window_one: str, window_two: str) -> str:
        self.focus_window(window_one)
        res = subprocess.run(["niri", "msg", "action", "move-column-right"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            return f"Reordered window '{window_one}' in Niri."
        return f"Could not reorder windows in Niri: {res.stderr.strip() or res.stdout.strip()}"

    def list_windows(self) -> str:
        import json
        try:
            res = subprocess.run(["niri", "msg", "-j", "windows"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and res.stdout:
                windows = json.loads(res.stdout)
                lines = []
                for w in windows:
                    ws = w.get("workspace_id", "?")
                    app = w.get("app_id", "App")
                    title = w.get("title", "Untitled")
                    lines.append(f"Workspace {ws}: {app} - {title[:50]}")
                if lines:
                    return f"Open windows ({len(lines)} total):\n" + "\n".join(lines[:15])
        except Exception:
            pass
        return "No open windows found in Niri."

    def get_open_windows_prompt_context(self) -> str:
        import json
        try:
            win_res = subprocess.run(["niri", "msg", "-j", "windows"], capture_output=True, text=True, timeout=2)
            ws_res = subprocess.run(["niri", "msg", "-j", "workspaces"], capture_output=True, text=True, timeout=2)
            windows = json.loads(win_res.stdout) if win_res.returncode == 0 and win_res.stdout else []
            workspaces = json.loads(ws_res.stdout) if ws_res.returncode == 0 and ws_res.stdout else []
            active_ws = next((w.get("name") or str(w.get("id")) for w in workspaces if w.get("is_active")), "?")
            focused_win = next((f"{w.get('app_id', 'App')} (title: \"{w.get('title', '')[:45]}\")" for w in windows if w.get("is_focused")), "None")

            lines = [f"Active Workspace: {active_ws} | Focused Window: {focused_win}"]
            if not windows:
                lines.append("Open Windows: None (desktop is empty)")
            else:
                lines.append(f"Open Windows ({len(windows)} total):")
                for w in windows:
                    foc = " [FOCUSED]" if w.get("is_focused") else ""
                    lines.append(f"- Workspace {w.get('workspace_id', '?')}: {w.get('app_id', 'Unknown')}{foc} - \"{w.get('title', '')[:50]}\"")
            return "\n".join(lines)
        except Exception:
            pass
        return self.list_windows()

    def capture_screenshot(self) -> bytes:
        if _which("grim", self.environ):
            res = subprocess.run(["grim", "-"], capture_output=True, timeout=15, env=self.environ)
            if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                return res.stdout
        raise RuntimeError("Screenshot capture failed in Niri. 'grim' is required.")

    def get_default_macros(self) -> dict[str, str]:
        return {
            "toggle_floating": "niri msg action toggle-window-floating",
            "fullscreen": "niri msg action fullscreen-window",
            "center": "niri msg action center-column",
            "consume_left": "niri msg action consume-or-expel-window-left",
            "lock": "loginctl lock-session"
        }


class KdePlasmaBackend(BaseDesktopBackend):
    name = "kde_plasma"

    def get_capabilities(self) -> set[str]:
        # KDE Plasma does NOT support window swapping natively; window_swap is disabled.
        return {
            "window_focus", "window_move", "workspace_switch",
            "window_close", "window_list", "screenshot",
            "desktop_macro", "app_launch", "app_list"
        }

    def _qdbus_bin(self) -> Optional[str]:
        return _find_qdbus()

    def focus_window(self, target: str) -> str:
        t = (target or "").strip()
        if not t:
            return "Please specify a window title or class to focus."

        # 1. kdotool (Wayland-native for KDE)
        if shutil.which("kdotool"):
            res = subprocess.run(["kdotool", "search", "--name", t, "windowactivate"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                return f"Focused window matching '{t}' via kdotool."
            res = subprocess.run(["kdotool", "search", "--class", t, "windowactivate"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                return f"Focused window matching '{t}' via kdotool."

        # 2. wmctrl (X11 / XWayland)
        if shutil.which("wmctrl"):
            try:
                res = subprocess.run(["wmctrl", "-l", "-x"], capture_output=True, text=True, timeout=1)
                if res.returncode == 0 and res.stdout.strip():
                    windows = []
                    for line in res.stdout.splitlines():
                        parts = line.split(None, 4)
                        if len(parts) >= 5:
                            windows.append({"id": parts[0], "class": parts[2], "title": parts[4]})
                    match = _find_best_window(t, windows)
                    if match:
                        res_act = subprocess.run(["wmctrl", "-i", "-a", match["id"]], capture_output=True, text=True, timeout=1)
                        if res_act.returncode == 0:
                            return f"Focused '{match['title'][:40]}' in KDE Plasma."
            except Exception:
                pass

        # 3. xdotool (X11)
        if shutil.which("xdotool") and os.environ.get("DISPLAY"):
            res = subprocess.run(["xdotool", "search", "--onlyvisible", "--name", t, "windowactivate"], capture_output=True, text=True, timeout=1)
            if res.returncode == 0:
                return f"Focused window '{t}' via xdotool."

        return f"Could not find an open window matching '{t}' in KDE Plasma."

    def workspace_control(self, action: str, workspace: str, target: str, is_current: bool = False, is_back: bool = False) -> str:
        qdbus = self._qdbus_bin()
        ws_clean = workspace.strip()

        # Handle switching
        if action not in ["move", "movetoworkspace"]:
            if ws_clean in ["next", "m+1"]:
                if qdbus:
                    res = subprocess.run([qdbus, "org.kde.KWin", "/KWin", "nextDesktop"], capture_output=True, text=True, timeout=2)
                    if res.returncode == 0:
                        return "Switched to next desktop in KDE Plasma."
                if shutil.which("kdotool"):
                    subprocess.run(["kdotool", "set_desktop", "next"], check=False)
                    return "Switched to next desktop in KDE Plasma."
            elif ws_clean in ["previous", "prev", "back", "m-1"]:
                if qdbus:
                    res = subprocess.run([qdbus, "org.kde.KWin", "/KWin", "previousDesktop"], capture_output=True, text=True, timeout=2)
                    if res.returncode == 0:
                        return "Switched to previous desktop in KDE Plasma."
                if shutil.which("kdotool"):
                    subprocess.run(["kdotool", "set_desktop", "prev"], check=False)
                    return "Switched to previous desktop in KDE Plasma."

            try:
                ws_num = int(ws_clean)
                if qdbus:
                    res = subprocess.run([qdbus, "org.kde.KWin", "/KWin", "setCurrentDesktop", str(ws_num)], capture_output=True, text=True, timeout=2)
                    if res.returncode == 0:
                        return f"Switched to virtual desktop {ws_num} in KDE Plasma."
                if shutil.which("kdotool"):
                    res = subprocess.run(["kdotool", "set_desktop", str(ws_num)], capture_output=True, text=True, timeout=2)
                    if res.returncode == 0:
                        return f"Switched to virtual desktop {ws_num} in KDE Plasma."
                if shutil.which("wmctrl"):
                    subprocess.run(["wmctrl", "-s", str(max(0, ws_num - 1))], check=False)
                    return f"Switched to virtual desktop {ws_num} in KDE Plasma."
            except ValueError:
                pass
            return f"Failed to switch to desktop '{workspace}' in KDE Plasma."

        # Handle moving
        try:
            ws_num = int(ws_clean)
        except ValueError:
            return f"Invalid target desktop number '{workspace}'."

        tgt = (target or "").strip()
        if tgt:
            # Move targeted window
            if shutil.which("kdotool"):
                res = subprocess.run(["kdotool", "search", "--name", tgt, "set_desktop_for_window", str(ws_num)], capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    return f"Moved '{tgt}' to desktop {ws_num} in KDE Plasma."
                res = subprocess.run(["kdotool", "search", "--class", tgt, "set_desktop_for_window", str(ws_num)], capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    return f"Moved '{tgt}' to desktop {ws_num} in KDE Plasma."

            if shutil.which("wmctrl"):
                res = subprocess.run(["wmctrl", "-l", "-x"], capture_output=True, text=True, timeout=1)
                if res.returncode == 0 and res.stdout.strip():
                    windows = []
                    for line in res.stdout.splitlines():
                        parts = line.split(None, 4)
                        if len(parts) >= 5:
                            windows.append({"id": parts[0], "class": parts[2], "title": parts[4]})
                    match = _find_best_window(tgt, windows)
                    if match:
                        subprocess.run(["wmctrl", "-i", "-r", match["id"], "-t", str(max(0, ws_num - 1))], check=False)
                        return f"Moved '{match['title'][:40]}' to desktop {ws_num} in KDE Plasma."

            return f"Could not find window matching '{tgt}' to move in KDE Plasma."
        else:
            # Move active window
            if shutil.which("kdotool"):
                res = subprocess.run(["kdotool", "getactivewindow", "set_desktop_for_window", str(ws_num)], capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    return f"Moved active window to desktop {ws_num} in KDE Plasma."

            if qdbus:
                res = subprocess.run([qdbus, "org.kde.kglobalaccel", "/component/kwin", "invokeShortcut", f"Window to Desktop {ws_num}"], capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    return f"Moved active window to desktop {ws_num} in KDE Plasma."

            if shutil.which("wmctrl"):
                subprocess.run(["wmctrl", "-r", ":ACTIVE:", "-t", str(max(0, ws_num - 1))], check=False)
                return f"Moved active window to desktop {ws_num} in KDE Plasma."

            return "Could not move active window in KDE Plasma."

    def close_application(self, app_name: str) -> str:
        target = (app_name or "").strip()
        is_active = not target or target.lower() in [
            "it", "this", "active", "current", "this window", "active window", "focused window", "the app", "the window"
        ]
        qdbus = self._qdbus_bin()

        if is_active:
            if qdbus:
                res = subprocess.run([qdbus, "org.kde.kglobalaccel", "/component/kwin", "invokeShortcut", "Window Close"], capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    return "Closed active window in KDE Plasma."
            if shutil.which("kdotool"):
                res = subprocess.run(["kdotool", "getactivewindow", "windowclose"], capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    return "Closed active window in KDE Plasma."
            if shutil.which("wmctrl"):
                res = subprocess.run(["wmctrl", "-c", ":ACTIVE:"], capture_output=True, text=True, timeout=1)
                if res.returncode == 0:
                    return "Closed active window in KDE Plasma."
            return "Could not close active window in KDE Plasma."

        # Targeted close
        if shutil.which("kdotool"):
            res = subprocess.run(["kdotool", "search", "--name", target, "windowclose"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                return f"Closed window matching '{target}' in KDE Plasma."
            res = subprocess.run(["kdotool", "search", "--class", target, "windowclose"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                return f"Closed window matching '{target}' in KDE Plasma."

        if shutil.which("wmctrl"):
            res = subprocess.run(["wmctrl", "-l", "-x"], capture_output=True, text=True, timeout=1)
            if res.returncode == 0 and res.stdout.strip():
                windows = []
                for line in res.stdout.splitlines():
                    parts = line.split(None, 4)
                    if len(parts) >= 5:
                        windows.append({"id": parts[0], "class": parts[2], "title": parts[4]})
                match = _find_best_window(target, windows)
                if match:
                    subprocess.run(["wmctrl", "-i", "-c", match["id"]], check=False)
                    return f"Closed window for '{match['title'][:40]}' in KDE Plasma."

        subprocess.run(["pkill", "-f", target], check=False)
        return f"Closed processes matching '{app_name}'."

    def list_windows(self) -> str:
        if shutil.which("kdotool"):
            try:
                res = subprocess.run(["kdotool", "search", "--onlyvisible", ""], capture_output=True, text=True, timeout=2)
                if res.returncode == 0 and res.stdout.strip():
                    win_ids = res.stdout.strip().splitlines()
                    windows = []
                    for wid in win_ids[:15]:
                        wid_clean = wid.strip()
                        if wid_clean:
                            res_title = subprocess.run(["kdotool", "getwindowname", wid_clean], capture_output=True, text=True, timeout=1)
                            t = res_title.stdout.strip() if res_title.returncode == 0 else ""
                            if t:
                                windows.append(f"Window (ID {wid_clean}): {t[:50]}")
                    if windows:
                        return f"Open windows ({len(windows)} total):\n" + "\n".join(windows)
            except Exception:
                pass

        if shutil.which("wmctrl"):
            try:
                res = subprocess.run(["wmctrl", "-l", "-x"], capture_output=True, text=True, timeout=1)
                if res.returncode == 0 and res.stdout.strip():
                    lines = []
                    for l in res.stdout.splitlines():
                        parts = l.split(None, 4)
                        if len(parts) >= 5:
                            ws_idx = str(int(parts[1]) + 1) if parts[1].isdigit() else parts[1]
                            lines.append(f"Desktop {ws_idx}: {parts[2]} - {parts[4][:50]}")
                    if lines:
                        return f"Open windows ({len(lines)} total):\n" + "\n".join(lines[:15])
            except Exception:
                pass

        return "No open desktop windows found or window management is not available in the current session."

    def get_open_windows_prompt_context(self) -> str:
        qdbus = self._qdbus_bin()
        active_ws = "?"
        if qdbus:
            try:
                res = subprocess.run([qdbus, "org.kde.KWin", "/KWin", "currentDesktop"], capture_output=True, text=True, timeout=1)
                if res.returncode == 0 and res.stdout.strip():
                    active_ws = res.stdout.strip()
            except Exception:
                pass

        focused_title = "None"
        if shutil.which("kdotool"):
            try:
                res_act = subprocess.run(["kdotool", "getactivewindow"], capture_output=True, text=True, timeout=1)
                if res_act.returncode == 0 and res_act.stdout.strip():
                    win_id = res_act.stdout.strip()
                    res_name = subprocess.run(["kdotool", "getwindowname", win_id], capture_output=True, text=True, timeout=1)
                    if res_name.returncode == 0 and res_name.stdout.strip():
                        focused_title = res_name.stdout.strip()
            except Exception:
                pass

        lines = [f"Active Desktop: {active_ws} | Focused Window: {focused_title[:45]}"]
        win_list = self.list_windows()
        lines.append(win_list)
        return "\n".join(lines)

    def swap_windows(self, window_one: str, window_two: str) -> str:
        return "Reordering or swapping windows is not supported on KDE Plasma (kwin)."

    def capture_screenshot(self) -> bytes:
        if _which("spectacle", self.environ):
            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                res = subprocess.run(
                    ["spectacle", "-b", "-n", "-o", tmp_path],
                    capture_output=True,
                    timeout=15,
                    env=self.environ,
                )
                if res.returncode == 0 and os.path.exists(tmp_path):
                    with open(tmp_path, "rb") as f:
                        data = f.read()
                    if data.startswith(b"\x89PNG\r\n\x1a\n"):
                        return data
            except Exception:
                pass
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        if _which("grim", self.environ):
            res = subprocess.run(["grim", "-"], capture_output=True, timeout=15, env=self.environ)
            if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                return res.stdout

        for tool, args in [("maim", ["maim"]), ("scrot", ["scrot", "-"]), ("import", ["import", "-window", "root", "png:-"])]:
            if _which(tool, self.environ):
                res = subprocess.run(args, capture_output=True, timeout=15, env=self.environ)
                if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                    return res.stdout

        raise RuntimeError("Screenshot capture failed on KDE Plasma. Neither spectacle nor grim produced valid PNG.")

    def get_default_macros(self) -> dict[str, str]:
        qdbus = self._qdbus_bin() or "qdbus"
        return {
            "overview": f"{qdbus} org.kde.kglobalaccel /component/kwin invokeShortcut 'Overview'",
            "show_desktop": f"{qdbus} org.kde.kglobalaccel /component/kwin invokeShortcut 'Show Desktop'",
            "grid": f"{qdbus} org.kde.kglobalaccel /component/kwin invokeShortcut 'Grid'",
            "krunner": f"{qdbus} org.kde.krunner /App display",
            "night_mode": f"{qdbus} org.kde.KWin /ColorCorrect org.kde.kwin.ColorCorrect.toggle",
            "lock": f"{qdbus} org.freedesktop.ScreenSaver /ScreenSaver Lock || loginctl lock-session",
            "suspend": "systemctl suspend"
        }


class GnomeBackend(BaseDesktopBackend):
    name = "gnome"

    def get_capabilities(self) -> set[str]:
        has_wmctrl = bool(shutil.which("wmctrl"))
        return {
            "window_focus", "window_close", "screenshot",
            "desktop_macro", "app_launch", "app_list"
        } | ({"window_list", "window_move", "workspace_switch"} if has_wmctrl else set())

    def swap_windows(self, window_one: str, window_two: str) -> str:
        return "Window swapping is not supported on GNOME Shell."

    def focus_window(self, target: str) -> str:
        t = (target or "").strip()
        if shutil.which("wmctrl"):
            res = subprocess.run(["wmctrl", "-a", t], capture_output=True, text=True)
            if res.returncode == 0:
                return f"Focused '{t}' in GNOME."

        apps = _scan_desktop_entries()
        selected = _resolve_application_entry(t, apps)
        if selected:
            if shutil.which("gtk-launch"):
                subprocess.run(["gtk-launch", selected["desktop_id"]], capture_output=True, timeout=2, env=os.environ)
                return f"Activated window for '{selected['name']}' in GNOME."
            if shutil.which("gio"):
                subprocess.run(["gio", "launch", selected["desktop_path"]], capture_output=True, timeout=2, env=os.environ)
                return f"Activated window for '{selected['name']}' in GNOME."

        return f"Could not focus window for '{t}' in GNOME."

    def workspace_control(self, action: str, workspace: str, target: str, is_current: bool = False, is_back: bool = False) -> str:
        if shutil.which("wmctrl"):
            try:
                ws_idx = int(workspace) - 1 if workspace.isdigit() and int(workspace) > 0 else int(workspace)
                if action in ["move", "movetoworkspace"]:
                    if target:
                        subprocess.run(["wmctrl", "-r", target, "-t", str(ws_idx)], check=False)
                        return f"Moved '{target}' to desktop {workspace} in GNOME."
                    else:
                        subprocess.run(["wmctrl", "-r", ":ACTIVE:", "-t", str(ws_idx)], check=False)
                        return f"Moved active window to desktop {workspace} in GNOME."
                else:
                    subprocess.run(["wmctrl", "-s", str(ws_idx)], check=False)
                    return f"Switched to desktop {workspace} in GNOME."
            except Exception:
                pass
        return "Workspace control on GNOME requires wmctrl or the Window Calls extension."

    def close_application(self, app_name: str) -> str:
        target = (app_name or "").strip()
        if shutil.which("wmctrl"):
            if not target or target.lower() in ["it", "this", "active", "the window", "the app"]:
                subprocess.run(["wmctrl", "-c", ":ACTIVE:"], check=False)
                return "Closed active window in GNOME."
            else:
                subprocess.run(["wmctrl", "-c", target], check=False)
                return f"Closed window matching '{target}' in GNOME."
        subprocess.run(["pkill", "-f", target], check=False)
        return f"Closed processes matching '{app_name}'."

    def capture_screenshot(self) -> bytes:
        if _which("gnome-screenshot", self.environ):
            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                res = subprocess.run(["gnome-screenshot", "-f", tmp_path], capture_output=True, timeout=15, env=self.environ)
                if res.returncode == 0 and os.path.exists(tmp_path):
                    with open(tmp_path, "rb") as f:
                        data = f.read()
                    if data.startswith(b"\x89PNG\r\n\x1a\n"):
                        return data
            except Exception:
                pass
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        if _which("grim", self.environ):
            res = subprocess.run(["grim", "-"], capture_output=True, timeout=15, env=self.environ)
            if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                return res.stdout

        raise RuntimeError("Screenshot capture failed on GNOME. Neither gnome-screenshot nor grim produced valid PNG.")

    def get_default_macros(self) -> dict[str, str]:
        return {
            "lock": "dbus-send --type=method_call --dest=org.gnome.ScreenSaver /org/gnome/ScreenSaver org.gnome.ScreenSaver.Lock || loginctl lock-session",
            "suspend": "systemctl suspend"
        }


class CosmicBackend(BaseDesktopBackend):
    name = "cosmic"

    def get_capabilities(self) -> set[str]:
        return {
            "window_focus", "window_close", "screenshot",
            "desktop_macro", "app_launch", "app_list"
        }

    def swap_windows(self, window_one: str, window_two: str) -> str:
        return "Window swapping is not supported on COSMIC."

    def focus_window(self, target: str) -> str:
        t = (target or "").strip()
        apps = _scan_desktop_entries()
        selected = _resolve_application_entry(t, apps)
        if selected:
            if shutil.which("gtk-launch"):
                subprocess.run(["gtk-launch", selected["desktop_id"]], capture_output=True, timeout=2, env=os.environ)
                return f"Activated window for '{selected['name']}' in COSMIC."
            if shutil.which("gio"):
                subprocess.run(["gio", "launch", selected["desktop_path"]], capture_output=True, timeout=2, env=os.environ)
                return f"Activated window for '{selected['name']}' in COSMIC."
        return f"Could not focus window for '{t}' in COSMIC."

    def capture_screenshot(self) -> bytes:
        if _which("grim", self.environ):
            res = subprocess.run(["grim", "-"], capture_output=True, timeout=15, env=self.environ)
            if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                return res.stdout
        if _which("cosmic-screenshot", self.environ):
            res = subprocess.run(["cosmic-screenshot"], capture_output=True, timeout=15, env=self.environ)
            if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                return res.stdout
        raise RuntimeError("Screenshot capture failed on COSMIC.")

    def get_default_macros(self) -> dict[str, str]:
        return {
            "lock": "loginctl lock-session",
            "suspend": "systemctl suspend"
        }


class GenericDesktopBackend(BaseDesktopBackend):
    name = "generic_desktop"

    def get_capabilities(self) -> set[str]:
        has_wmctrl = bool(shutil.which("wmctrl"))
        has_xdotool = bool(shutil.which("xdotool"))
        caps = {"desktop_macro", "app_launch", "app_list", "window_close"}
        if has_wmctrl or has_xdotool:
            caps.add("window_focus")
        if has_wmctrl:
            caps.add("window_list")
            caps.add("window_move")
            caps.add("workspace_switch")
        if any(shutil.which(x) for x in ["grim", "maim", "scrot", "import"]):
            caps.add("screenshot")
        return caps

    def focus_window(self, target: str) -> str:
        t = (target or "").strip()
        if shutil.which("wmctrl"):
            try:
                res = subprocess.run(["wmctrl", "-l", "-x"], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    windows = []
                    for line in res.stdout.splitlines():
                        parts = line.split(None, 4)
                        if len(parts) >= 5:
                            windows.append({"id": parts[0], "class": parts[2], "title": parts[4]})
                    match = _find_best_window(t, windows)
                    if match:
                        res = subprocess.run(["wmctrl", "-i", "-a", match["id"]], capture_output=True, text=True)
                        if res.returncode == 0:
                            return f"Focused '{match['title'][:40]}'."
            except Exception:
                pass

        if shutil.which("xdotool") and os.environ.get("DISPLAY"):
            res = subprocess.run(["xdotool", "search", "--onlyvisible", "--name", t, "windowactivate"], capture_output=True, text=True)
            if res.returncode == 0:
                return f"Focused window '{t}' via xdotool."

        return f"Could not find an open window matching '{t}'."

    def workspace_control(self, action: str, workspace: str, target: str, is_current: bool = False, is_back: bool = False) -> str:
        if shutil.which("wmctrl"):
            try:
                ws_idx = int(workspace) - 1 if workspace.isdigit() and int(workspace) > 0 else int(workspace)
                if action in ["move", "movetoworkspace"]:
                    if target:
                        res = subprocess.run(["wmctrl", "-l", "-x"], capture_output=True, text=True)
                        if res.returncode == 0 and res.stdout.strip():
                            windows = []
                            for line in res.stdout.splitlines():
                                parts = line.split(None, 4)
                                if len(parts) >= 5:
                                    windows.append({"id": parts[0], "class": parts[2], "title": parts[4]})
                            match = _find_best_window(target, windows)
                            if match:
                                subprocess.run(["wmctrl", "-i", "-r", match["id"], "-t", str(ws_idx)], check=False)
                                return f"Moved '{match['title'][:40]}' to desktop {workspace}."
                        return f"Could not find window matching '{target}' to move."
                    else:
                        subprocess.run(["wmctrl", "-r", ":ACTIVE:", "-t", str(ws_idx)], check=False)
                        return f"Moved active window to desktop {workspace}."
                else:
                    subprocess.run(["wmctrl", "-s", str(ws_idx)], check=False)
                    return f"Switched to desktop {workspace}."
            except Exception:
                pass
        return "Workspace management is not supported in the current session."

    def close_application(self, app_name: str) -> str:
        target = (app_name or "").strip()
        if shutil.which("wmctrl"):
            try:
                res = subprocess.run(["wmctrl", "-l", "-x"], capture_output=True, text=True, timeout=1)
                if res.returncode == 0 and res.stdout.strip():
                    windows = []
                    for line in res.stdout.splitlines():
                        parts = line.split(None, 4)
                        if len(parts) >= 5:
                            windows.append({"id": parts[0], "class": parts[2], "title": parts[4]})
                    match = _find_best_window(target, windows)
                    if match:
                        subprocess.run(["wmctrl", "-i", "-c", match["id"]], check=False, timeout=1)
                        return f"Closed window for '{match['title'][:40]}'."
            except Exception:
                pass
        subprocess.run(["pkill", "-f", target], check=False)
        return f"Closed processes matching '{app_name}'."

    def list_windows(self) -> str:
        if shutil.which("wmctrl"):
            try:
                res = subprocess.run(["wmctrl", "-l", "-x"], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    lines = []
                    for l in res.stdout.splitlines():
                        parts = l.split(None, 4)
                        if len(parts) >= 5:
                            lines.append(f"Desktop {parts[1]}: {parts[2]} - {parts[4][:50]}")
                    if lines:
                        return f"Open windows ({len(lines)} total):\n" + "\n".join(lines[:15])
            except Exception:
                pass
        return "No open desktop windows found or window management is not available in the current session."

    def swap_windows(self, window_one: str, window_two: str) -> str:
        return "Window swapping is not supported in the current desktop session."

    def capture_screenshot(self) -> bytes:
        for tool, args in [("grim", ["grim", "-"]), ("maim", ["maim"]), ("scrot", ["scrot", "-"]), ("import", ["import", "-window", "root", "png:-"])]:
            if _which(tool, self.environ):
                res = subprocess.run(args, capture_output=True, timeout=15, env=self.environ)
                if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                    return res.stdout
        raise RuntimeError("Screenshot capture failed. Please install grim, maim, or scrot.")


# ---------------------------------------------------------------------------
# Backend Resolver & Capability Registry
# ---------------------------------------------------------------------------

_BACKEND_MAP = {
    "hyprland": HyprlandBackend,
    "sway": SwayBackend,
    "i3": I3Backend,
    "niri": NiriBackend,
    "kde_plasma": KdePlasmaBackend,
    "gnome": GnomeBackend,
    "cosmic": CosmicBackend,
    "generic_desktop": GenericDesktopBackend
}

TOOL_CAPABILITY_MAP = {
    "swap_windows": "window_swap",
    "focus_window": "window_focus",
    "workspace_control": "workspace_control",
    "list_windows": "window_list",
    "close_application": "window_close",
    "capture_screenshot": "screenshot",
    "desktop_macro": "desktop_macro",
    "launch_application": "app_launch",
    "list_applications": "app_list",
}


def get_active_backend(
    refresh_env: bool = True, *, environ: dict[str, str] | None = None
) -> BaseDesktopBackend:
    """Returns the backend instance for the active desktop environment."""
    if refresh_env and environ is None:
        ensure_gui_environment()
    env = os.environ if environ is None else environ
    xdg_current = env.get("XDG_CURRENT_DESKTOP", "").lower()
    desktop_session = env.get("DESKTOP_SESSION", "").lower()

    def backend(cls: type[BaseDesktopBackend]) -> BaseDesktopBackend:
        return cls(environ=env)

    # 1. Direct tool and socket matches
    # Hyprland
    if _which("hyprctl", environ) and (env.get("HYPRLAND_INSTANCE_SIGNATURE") or (env.get("WAYLAND_DISPLAY") and "hyprland" in xdg_current)):
        return backend(HyprlandBackend)

    # Sway
    is_sway_session = (
        bool(env.get("SWAYSOCK"))
        or "sway" in xdg_current
        or "sway" in desktop_session
    )
    if _which("swaymsg", environ) and is_sway_session:
        return backend(SwayBackend)

    # i3
    # The i3 CLI can be installed on any X11 session. DISPLAY alone is not
    # evidence that the active window manager is i3 (for example, private Xvfb
    # sessions use DISPLAY with Openbox or no window manager).
    is_i3_session = (
        bool(env.get("I3SOCK"))
        or "i3" in xdg_current
        or "i3" in desktop_session
    )
    if _which("i3-msg", environ) and is_i3_session:
        return backend(I3Backend)

    # Niri
    if _which("niri", environ) and (env.get("NIRI_SOCKET") or "niri" in xdg_current):
        return backend(NiriBackend)

    # KDE Plasma
    if "kde" in xdg_current or "plasma" in xdg_current or env.get("KDE_SESSION_VERSION") or "kde" in desktop_session or "plasma" in desktop_session:
        return backend(KdePlasmaBackend)

    # GNOME
    if "gnome" in xdg_current or "gnome" in desktop_session:
        return backend(GnomeBackend)

    # COSMIC
    if "cosmic" in xdg_current or "cosmic" in desktop_session:
        return backend(CosmicBackend)

    de = detect_desktop_environment(refresh_env=refresh_env, environ=env)
    backend_cls = _BACKEND_MAP.get(de, GenericDesktopBackend)
    return backend(backend_cls)


def is_capability_supported(capability: str) -> bool:
    """Returns True if the active desktop environment supports the specified capability."""
    if capability in _DISABLED_CAPABILITIES:
        return False
    backend = get_active_backend()
    return backend.supports(capability)


def is_tool_enabled(tool_name: str) -> bool:
    """Returns True if the specified tool should be enabled based on active desktop capabilities and config."""
    if tool_name in _DISABLED_TOOLS:
        return False
    required_cap = TOOL_CAPABILITY_MAP.get(tool_name)
    if not required_cap:
        return True
    return is_capability_supported(required_cap)


def get_active_capabilities() -> dict[str, bool]:
    """Returns a dictionary of all standard desktop capabilities and their active support status."""
    backend = get_active_backend()
    caps = [
        "window_focus", "window_move", "workspace_switch", "window_close",
        "window_list", "window_swap", "screenshot", "desktop_macro",
        "app_launch", "app_list"
    ]
    return {c: is_capability_supported(c) for c in caps}


# ---------------------------------------------------------------------------
# Public Desktop Tool API
# ---------------------------------------------------------------------------

def _active_window_geometry(
    *, environ: dict[str, str] | None = None, backend: BaseDesktopBackend | None = None
) -> tuple[int, int, int, int]:
    """Return the focused window's desktop-space x, y, width, and height."""
    env = os.environ if environ is None else environ
    selected_backend = backend or get_active_backend(environ=environ)
    native_wayland = _is_native_wayland_session(environ)
    if selected_backend.name == "hyprland" and _which("hyprctl", environ):
        result = subprocess.run(["hyprctl", "activewindow", "-j"], capture_output=True, text=True, timeout=2, env=env)
        data = json.loads(result.stdout or "{}")
        x, y = map(int, data["at"])
        width, height = map(int, data["size"])
    elif selected_backend.name == "sway" or (selected_backend.name == "i3" and not native_wayland):
        bounds = _active_window_metadata(environ=environ, backend=selected_backend).get("bounds")
        if not bounds or len(bounds) != 4:
            raise RuntimeError("The focused window has no capturable bounds.")
        x, y, width, height = map(int, bounds)
    elif selected_backend.name == "niri":
        bounds = _active_window_metadata(environ=environ, backend=selected_backend).get("bounds")
        if not bounds or len(bounds) != 4:
            raise RuntimeError("Niri did not report verified bounds for the focused window.")
        x, y, width, height = map(int, bounds)
    elif not native_wayland and _which("xdotool", environ):
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowgeometry", "--shell"],
            capture_output=True, text=True, timeout=2, check=True,
            env=env,
        )
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        x, y, width, height = (int(values[key]) for key in ("X", "Y", "WIDTH", "HEIGHT"))
    else:
        if native_wayland:
            raise RuntimeError(
                f"Focused-window geometry is not verified for the {selected_backend.name} Wayland backend."
            )
        raise RuntimeError("Focused-window screenshots need a verified window-geometry source.")
    if width <= 0 or height <= 0:
        raise RuntimeError("The focused application has no capturable window size.")
    return x, y, width, height


def _active_window_metadata(
    *, environ: dict[str, str] | None = None, backend: BaseDesktopBackend | None = None
) -> dict[str, Any]:
    """Return identity for the focused window where the compositor exposes it."""
    env = os.environ if environ is None else environ
    backend = backend or get_active_backend(environ=environ)
    native_wayland = _is_native_wayland_session(environ)
    try:
        if backend.name == "hyprland":
            result = subprocess.run(["hyprctl", "activewindow", "-j"], capture_output=True, text=True, timeout=2, env=env)
            data = json.loads(result.stdout or "{}")
            at = data.get("at") or ()
            size = data.get("size") or ()
            bounds = tuple(map(int, (*at, *size))) if len(at) == 2 and len(size) == 2 else (0, 0, 0, 0)
            return {
                "id": str(data.get("address") or ""),
                "class": str(data.get("class") or data.get("initialClass") or ""),
                "title": str(data.get("title") or data.get("initialTitle") or ""),
                "bounds": bounds,
                "monitor_id": str(data.get("monitor")) if data.get("monitor") is not None else "",
            }
        if backend.name == "sway" or (backend.name == "i3" and not native_wayland):
            command = ["swaymsg", "-t", "get_tree"] if backend.name == "sway" else ["i3-msg", "-t", "get_tree"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=2, env=env)
            tree = json.loads(result.stdout or "{}")
            def focused(node: dict) -> Optional[dict]:
                if node.get("focused") and (node.get("window") is not None or node.get("app_id") or node.get("window_properties")):
                    return node
                for child in [*node.get("nodes", []), *node.get("floating_nodes", [])]:
                    match = focused(child)
                    if match:
                        return match
                return None
            node = focused(tree) or {}
            props = node.get("window_properties") or {}
            rect = node.get("rect") or {}
            return {
                "id": str(node.get("id") or ""),
                "class": str(node.get("app_id") or props.get("class") or props.get("instance") or ""),
                "title": str(node.get("name") or ""),
                "bounds": (int(rect.get("x", 0)), int(rect.get("y", 0)), int(rect.get("width", 0)), int(rect.get("height", 0))),
            }
        if backend.name == "niri":
            result = subprocess.run(["niri", "msg", "-j", "windows"], capture_output=True, text=True, timeout=2, env=env)
            windows = json.loads(result.stdout or "[]")
            node = next((w for w in windows if w.get("is_focused") or w.get("focused")), {})
            rect = node.get("rect")
            if isinstance(rect, dict):
                bounds = tuple(int(rect.get(key, 0)) for key in ("x", "y", "width", "height"))
            elif isinstance(rect, (list, tuple)) and len(rect) == 4:
                bounds = tuple(map(int, rect))
            else:
                bounds = (0, 0, 0, 0)
            return {
                "id": str(node.get("id") or ""),
                "class": str(node.get("app_id") or ""),
                "title": str(node.get("title") or ""),
                "bounds": bounds,
            }
        if (
            (backend.name == "kde_plasma" or not native_wayland)
            and _which("kdotool", environ)
            and (native_wayland or not _which("xdotool", environ))
        ):
            result = subprocess.run(["kdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True, timeout=2, env=env)
            if result.returncode == 0 and result.stdout.strip():
                return {"id": "", "class": "", "title": result.stdout.strip().splitlines()[-1]}
        if not _is_native_wayland_session(environ) and _which("xdotool", environ):
            result = subprocess.run(["xdotool", "getactivewindow"], capture_output=True, text=True, timeout=2, env=env)
            window_id = result.stdout.strip()
            if result.returncode == 0 and window_id:
                name = subprocess.run(["xdotool", "getwindowname", window_id], capture_output=True, text=True, timeout=2, env=env)
                cls = subprocess.run(["xdotool", "getwindowclassname", window_id], capture_output=True, text=True, timeout=2, env=env)
                return {"id": window_id, "class": cls.stdout.strip(), "title": name.stdout.strip()}
    except Exception:
        pass
    return {"id": "", "class": "", "title": ""}


def _hyprland_monitor_snapshot(
    monitor_id: str, *, environ: dict[str, str] | None = None
) -> tuple[str, str, int, int, int, int, float]:
    """Return identity, logical origin, pixel dimensions, and scale for one output."""
    monitor_id = str(monitor_id or "").strip()
    if not monitor_id:
        raise RuntimeError("Hyprland did not report the focused monitor identity.")
    env = os.environ if environ is None else environ
    result = subprocess.run(
        ["hyprctl", "monitors", "-j"], capture_output=True, text=True,
        timeout=2, check=True, env=env,
    )
    monitors = json.loads(result.stdout or "[]")
    monitor = next((item for item in monitors if str(item.get("id")) == monitor_id), None)
    if not monitor:
        raise RuntimeError("Hyprland did not report metadata for the focused monitor.")
    scale = float(monitor.get("scale", 1.0) or 1.0)
    x, y = int(monitor.get("x", 0)), int(monitor.get("y", 0))
    width, height = int(monitor.get("width", 0)), int(monitor.get("height", 0))
    if scale <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Hyprland reported invalid focused-monitor bounds or scale.")
    return (monitor_id, str(monitor.get("name") or ""), x, y, width, height, scale)


def _active_hyprland_monitor_snapshot(
    *, environ: dict[str, str] | None = None
) -> tuple[str, str, int, int, int, int, float]:
    env = os.environ if environ is None else environ
    result = subprocess.run(
        ["hyprctl", "activewindow", "-j"], capture_output=True, text=True,
        timeout=2, check=True, env=env,
    )
    active = json.loads(result.stdout or "{}")
    return _hyprland_monitor_snapshot(active.get("monitor"), environ=environ)


def _connect_hyprland_focus_event_socket(
    *, environ: dict[str, str] | None = None
) -> socket.socket:
    """Subscribe to this Hyprland instance's focus events for a capture interval."""
    env = os.environ if environ is None else environ
    signature = str(env.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip()
    runtime_dir = str(env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
    if not signature:
        raise RuntimeError("Hyprland instance identity is unavailable for focus-event monitoring.")
    path = Path(runtime_dir) / "hypr" / signature / ".socket2.sock"
    event_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        event_socket.settimeout(2)
        event_socket.connect(str(path))
        event_socket.setblocking(False)
        return event_socket
    except OSError as exc:
        event_socket.close()
        raise RuntimeError("Could not monitor Hyprland focus events during screenshot capture.") from exc


def _hyprland_focus_changed(event_socket: socket.socket) -> bool:
    """Drain Hyprland events and report any focus or focused-monitor transition."""
    data = bytearray()
    partial_frame_deadline = time.monotonic() + 0.1
    try:
        while True:
            try:
                chunk = event_socket.recv(4096)
            except BlockingIOError:
                if not data or data[-1] == ord("\n"):
                    break
                remaining = partial_frame_deadline - time.monotonic()
                if remaining <= 0 or not select.select([event_socket], [], [], remaining)[0]:
                    raise RuntimeError(
                        "Hyprland focus-event monitoring returned an incomplete event."
                    )
                continue
            if not chunk:
                raise RuntimeError("Hyprland focus-event monitoring ended during screenshot capture.")
            data.extend(chunk)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Hyprland focus-event monitoring failed during screenshot capture.") from exc

    lines = data.split(b"\n")
    for line in lines[:-1]:
        event_name, separator, _payload = line.partition(b">>")
        if not separator:
            raise RuntimeError("Hyprland focus-event monitoring returned an invalid event.")
        event_name = bytes(event_name)
        if event_name in {b"activewindow", b"activewindowv2", b"focusedmon", b"focusedmonv2"}:
            return True
    return False


def _verify_hyprland_capture_monitor(
    backend: BaseDesktopBackend,
    expected: tuple[str, str, int, int, int, int, float],
) -> None:
    monitor_id, name, x, y, width, height, scale = expected
    expected_bounds = (x, y, x + width, y + height)
    expected_target = name or f"monitor at {x},{y}"
    if (
        getattr(backend, "last_screenshot_monitor_id", None) != monitor_id
        or getattr(backend, "last_screenshot_target", None) != expected_target
        or getattr(backend, "last_screenshot_origin", None) != (x, y)
        or getattr(backend, "last_screenshot_bounds", None) != expected_bounds
        or getattr(backend, "last_screenshot_scale", None) != (scale, scale)
    ):
        raise RuntimeError(
            "Captured Hyprland output identity, origin, bounds, or scale did not match "
            "the verified focused monitor."
        )


def _active_window_snapshot(
    *, environ: dict[str, str] | None = None, backend: BaseDesktopBackend | None = None
) -> tuple[str, tuple[int, int, int, int], tuple[str, str, int, int, int, int, float] | None]:
    """Return focused-window identity, bounds, and verified output metadata."""
    selected_backend = backend or get_active_backend(environ=environ)
    window = _active_window_metadata(environ=environ, backend=selected_backend)
    bounds = window.get("bounds")
    if not bounds or len(bounds) != 4:
        # On X11, query geometry for the identified X window itself where the
        # backend exposed its ID. This avoids pairing another active window's
        # geometry with the metadata result if focus changes between queries.
        window_id = str(window.get("id") or "").strip()
        if not _is_native_wayland_session(environ) and window_id and _which("xdotool", environ):
            env = os.environ if environ is None else environ
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", window_id],
                capture_output=True, text=True, timeout=2, check=True, env=env,
            )
            values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            bounds = tuple(int(values[key]) for key in ("X", "Y", "WIDTH", "HEIGHT"))
        else:
            bounds = _active_window_geometry(environ=environ, backend=selected_backend)
    bounds = tuple(map(int, bounds))
    if len(bounds) != 4 or bounds[2] <= 0 or bounds[3] <= 0:
        raise RuntimeError("The focused window identity or bounds could not be verified.")
    identity = str(window.get("id") or "").strip()
    if not identity:
        identity = f"{window.get('class', '')}:{window.get('title', '')}".strip(":")
    if not identity:
        raise RuntimeError("The focused window identity could not be verified.")

    monitor_snapshot = None
    if selected_backend.name == "hyprland":
        monitor_snapshot = _hyprland_monitor_snapshot(window.get("monitor_id"), environ=environ)
    return identity, bounds, monitor_snapshot


def screenshot_delay_for_focused_window(
    default_seconds: float = 0.25,
    browser_seconds: float = 1.5,
    expected_application: str | None = None,
    *, environ: dict[str, str] | None = None,
) -> float:
    """Choose a short deterministic screenshot settle delay from the focused app."""
    try:
        if expected_application:
            app_identity = str(expected_application)
        else:
            window = _active_window_metadata(environ=environ)
            app_identity = f"{window.get('class', '')} {window.get('title', '')}"
        browser = re.search(
            r"\b(?:browser|firefox|mozilla|chrome|chromium|edge|msedge|brave|vivaldi|opera|zen)\b",
            app_identity,
            re.IGNORECASE,
        )
        value = browser_seconds if browser else default_seconds
        value = float(value)
        return min(max(value, 0.0), 10.0)
    except (TypeError, ValueError):
        return 0.25


def _expected_application_names(
    target: str, environ: dict[str, str] | None = None
) -> set[str]:
    names = {str(target or "").strip().lower()}
    try:
        apps = _scan_desktop_entries(environ)
        selected = _resolve_application_entry(target, apps, environ)
        if selected:
            names.add(str(selected.get("name") or "").lower())
            names.add(Path(shlex.split(str(selected.get("exec") or ""))[0]).name.lower())
            names.add(str(selected.get("startup_wm_class") or "").lower())
    except Exception:
        pass
    return {name for name in names if name}


def _window_matches_application(window: dict[str, str], expected_names: set[str]) -> bool:
    haystack = " ".join((window.get("class", ""), window.get("title", ""))).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", haystack).strip()
    words = set(normalized.split())
    for name in expected_names:
        candidate = re.sub(r"[^a-z0-9]+", " ", name).strip()
        if not candidate:
            continue
        if candidate in normalized or normalized in candidate:
            return True
        candidate_words = set(candidate.split())
        if candidate_words and any(len(word) >= 3 and word in words for word in candidate_words):
            return True
    return False


def wait_for_application_ready(
    target: Optional[str] = None, timeout: float | None = 15.0, *,
    environ: dict[str, str] | None = None,
) -> None:
    """Wait until a focused app window exists and its identity/bounds settle.

    A finite timeout is appropriate while starting an application. A focus
    transition can take as long as the compositor needs, so callers may pass
    ``None`` to keep waiting until the requested window is actually focused.
    """
    started = time.perf_counter()
    if environ is None:
        with timed_stage("desktop.readiness.gui_environment"):
            ensure_gui_environment()
    deadline = time.monotonic() + max(0.5, timeout) if timeout is not None else None
    with timed_stage("desktop.readiness.target_resolution", target_requested=bool(target)):
        expected_names = _expected_application_names(target, environ) if target else set()
    previous: Optional[tuple[str, str, tuple[int, int, int, int]]] = None
    stable_polls = 0
    poll_count = 0
    metadata_ms = 0.0
    geometry_ms = 0.0
    while deadline is None or time.monotonic() < deadline:
        try:
            poll_count += 1
            probe_started = time.perf_counter()
            window = _active_window_metadata(environ=environ)
            metadata_ms += (time.perf_counter() - probe_started) * 1000
            if expected_names and not _window_matches_application(window, expected_names):
                stable_polls = 0
                previous = None
            else:
                try:
                    geometry_started = time.perf_counter()
                    bounds = _active_window_geometry(environ=environ)
                    geometry_ms += (time.perf_counter() - geometry_started) * 1000
                except Exception:
                    bounds = tuple(window.get("bounds") or (0, 0, 0, 0))
                identity = window.get("id") or f"{window.get('class', '')}:{window.get('title', '')}"
                current = (identity, window.get("class", ""), bounds)
                if identity.strip(":") and current == previous:
                    stable_polls += 1
                else:
                    stable_polls = 1 if identity.strip(":") else 0
                previous = current
                if stable_polls >= 3:
                    with timed_stage("desktop.readiness.presentation_settle"):
                        time.sleep(0.2)  # Allow the compositor to present the mapped window.
                    log_duration(
                        "desktop.readiness.polls", started,
                        status="ready", polls=poll_count,
                        metadata_ms=round(metadata_ms, 1), geometry_ms=round(geometry_ms, 1),
                    )
                    return
        except Exception:
            stable_polls = 0
            previous = None
        time.sleep(0.2)
    log_duration(
        "desktop.readiness.polls", started,
        status="timeout", polls=poll_count,
        metadata_ms=round(metadata_ms, 1), geometry_ms=round(geometry_ms, 1),
    )
    target_text = f" matching '{target}'" if target else ""
    raise TimeoutError(f"The focused application{target_text} did not become ready for a screenshot within {timeout:g} seconds.")


class ScreenshotCapture(tuple):
    """Two-item compatible screenshot result carrying pixel-to-desktop scale."""

    def __new__(
        cls,
        image: bytes,
        origin: tuple[int, int],
        scale: tuple[float, float] = (1.0, 1.0),
        backend: str = "unknown",
        target: str = "",
    ):
        value = super().__new__(cls, (image, origin))
        value.scale = scale
        value.backend = backend
        value.target = target
        return value


def capture_screenshot_with_origin(
    scope: str = "window", *, expected_application: Optional[str] = None, wait_until_ready: bool = True,
    environ: dict[str, str] | None = None,
) -> ScreenshotCapture:
    """Capture the focused window, a verified single monitor, or explicit full desktop."""
    scope = (scope or "window").strip().lower()
    if scope not in {"window", "monitor", "desktop"}:
        raise ValueError("Screenshot scope must be 'window', 'monitor', or 'desktop'.")
    if environ is None:
        ensure_gui_environment()
    if wait_until_ready:
        wait_for_application_ready(expected_application, environ=environ)
    with timed_stage("desktop.capture.backend_selection"):
        backend = get_active_backend(environ=environ)
    if scope == "monitor" and backend.name != "hyprland":
        raise RuntimeError(
            f"Single-monitor capture is not verified for {backend.name}; "
            "request the focused window or explicitly request the full desktop."
        )
    window_snapshot_before = None
    monitor_snapshot_before = None
    if scope == "window":
        if backend.name in {"sway", "niri"}:
            raise RuntimeError(
                f"Focused-window screenshots are disabled for {backend.name} until its "
                "screenshot pixel scale and desktop origin can be verified."
            )
        window_snapshot_before = _active_window_snapshot(environ=environ, backend=backend)
        monitor_snapshot_before = window_snapshot_before[2]
    elif scope == "monitor":
        monitor_snapshot_before = _active_hyprland_monitor_snapshot(environ=environ)
    focus_event_socket = None
    if scope in {"window", "monitor"} and backend.name == "hyprland":
        focus_event_socket = _connect_hyprland_focus_event_socket(environ=environ)
    try:
        with timed_stage("desktop.capture.backend"):
            image = (
                backend.capture_desktop_screenshot()
                if scope == "desktop"
                else backend.capture_screenshot()
            )
        if focus_event_socket and _hyprland_focus_changed(focus_event_socket):
            raise RuntimeError(
                "Hyprland focus or focused monitor changed during screenshot capture; "
                "refusing to attach a potentially mismatched image."
            )
    finally:
        if focus_event_socket:
            focus_event_socket.close()
    if scope == "window":
        window_snapshot_after = _active_window_snapshot(environ=environ, backend=backend)
        if window_snapshot_after != window_snapshot_before:
            raise RuntimeError(
                "The focused window changed identity or bounds during screenshot capture; "
                "refusing to attach a mismatched image."
            )
    if scope in {"window", "monitor"} and backend.name == "hyprland":
        monitor_snapshot_after = _active_hyprland_monitor_snapshot(environ=environ)
        if monitor_snapshot_after != monitor_snapshot_before:
            raise RuntimeError(
                "The focused Hyprland monitor or its layout changed during screenshot capture; "
                "refusing to attach an image with an unverified origin or scale."
            )
        _verify_hyprland_capture_monitor(backend, monitor_snapshot_before)
    origin = getattr(backend, "last_screenshot_origin", (0, 0))
    scale = getattr(backend, "last_screenshot_scale", (1.0, 1.0))
    target = getattr(backend, "last_screenshot_target", "") or {
        "window": "focused window",
        "monitor": "focused monitor",
        "desktop": "full desktop (explicitly requested)",
    }[scope]
    try:
        from io import BytesIO
        from PIL import Image

        with Image.open(BytesIO(image)) as captured_image:
            if captured_image.format != "PNG":
                raise RuntimeError("Screenshot data was not PNG.")
            captured_image.load()
            image_width, image_height = captured_image.size
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Screenshot could not be decoded and verified: {exc}") from exc
    if image_width <= 0 or image_height <= 0:
        raise RuntimeError("Screenshot has empty image dimensions.")
    if scope == "monitor":
        expected_bounds = getattr(backend, "last_screenshot_bounds", None)
        if not expected_bounds:
            raise RuntimeError("The backend did not report bounds for the requested monitor capture.")
        if (image_width, image_height) != (expected_bounds[2] - expected_bounds[0], expected_bounds[3] - expected_bounds[1]):
            raise RuntimeError(
                "Captured image dimensions do not match the reported focused-monitor bounds; "
                "refusing to attach an unverified image."
            )
        return ScreenshotCapture(image, origin, scale, backend.name, target)

    if scope == "window" and backend.name == "hyprland":
        expected_monitor = monitor_snapshot_before
        if not expected_monitor or (image_width, image_height) != (expected_monitor[4], expected_monitor[5]):
            raise RuntimeError(
                "Captured image dimensions do not match the verified focused Hyprland monitor; "
                "refusing to attach an unverified window crop."
            )

    if scope == "desktop":
        return ScreenshotCapture(image, origin, scale, backend.name, target)

    with timed_stage("desktop.capture.window_geometry"):
        _, (x, y, width, height), _ = window_snapshot_before
        left = round((x - origin[0]) * scale[0])
        top = round((y - origin[1]) * scale[1])
        crop_width = round(width * scale[0])
        crop_height = round(height * scale[1])
    try:
        from PIL import Image
        from io import BytesIO

        with timed_stage("desktop.capture.window_crop"):
            with Image.open(BytesIO(image)) as screenshot:
                bounds = (left, top, left + crop_width, top + crop_height)
                if (
                    bounds[0] < 0 or bounds[1] < 0
                    or bounds[2] > screenshot.width or bounds[3] > screenshot.height
                ):
                    raise RuntimeError(
                        "The focused window is not fully contained in the captured image; "
                        "refusing a clipped or mismatched window screenshot."
                    )
                output = BytesIO()
                screenshot.crop(bounds).save(output, format="PNG")
                if (bounds[2] - bounds[0], bounds[3] - bounds[1]) != (crop_width, crop_height):
                    raise RuntimeError("Focused-window crop dimensions do not match the reported window bounds.")
            return ScreenshotCapture(output.getvalue(), (x, y), scale, backend.name, f"focused window at {x},{y}")
    except ImportError as exc:
        raise RuntimeError("Focused-window screenshots require Pillow; install the 'pillow' package.") from exc


def capture_screenshot(scope: str = "window", *, expected_application: Optional[str] = None) -> bytes:
    """Capture the focused monitor or application window as PNG bytes."""
    return capture_screenshot_with_origin(scope, expected_application=expected_application)[0]


def list_applications(query: Optional[str] = "") -> str:
    """Lists installed desktop applications, optionally filtered by a query string."""
    apps = _scan_desktop_entries()
    q = (query or "").strip().lower()

    if q:
        matches = [v["name"] for k, v in apps.items() if q in k or q in v["exec"].lower() or q in v["comment"].lower()]
        if not matches:
            return f"No installed applications found matching '{query}'."
        return f"Installed applications matching '{query}': {', '.join(sorted(matches)[:25])}"
    else:
        app_names = sorted(list({v["name"] for v in apps.values()}))
        count = len(app_names)
        sample = ", ".join(app_names[:20])
        return f"Found {count} installed applications across the system. Sample apps: {sample}"


def launch_application(app_name: str, args: Optional[str] = "") -> str:
    """Launches an installed application or game by name using the desktop environment dispatcher."""
    ensure_gui_environment()
    target = (app_name or "").strip()
    if not target:
        return "Please specify an application name to launch."

    apps = _scan_desktop_entries()
    selected = _resolve_application_entry(target, apps)

    if selected and selected.get("steam_app_id"):
        app_id = str(selected["steam_app_id"])
        if args:
            return "Custom launch arguments are not supported for Steam applications."
        if not shutil.which("steam"):
            return f"Steam is not available to launch {selected['name']}."
        try:
            subprocess.Popen(
                ["steam", f"steam://rungameid/{app_id}"],
                start_new_session=True,
                env=os.environ,
            )
            return f"Launched {selected['name']} through Steam."
        except Exception as exc:
            return f"Failed to launch {selected['name']} through Steam: {exc}"

    clean_target = (target or "").strip().lower()
    exec_cmd = selected["exec"] if selected else (clean_target if shutil.which(clean_target) else target)
    display_name = selected["name"] if selected else app_name

    if args:
        exec_cmd = f"{exec_cmd} {args.strip()}"

    # Dispatch using the detected desktop environment
    try:
        # 1. Hyprland
        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("hyprctl"):
            res = _hyprland_dispatch(
                "exec", exec_cmd,
                lua_expression=f"hl.dsp.exec_cmd({_lua_string(exec_cmd)})",
            )
            if res.returncode == 0:
                return f"Launched {display_name} via Hyprland."
        # 2. Sway
        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("swaymsg"):
            res = subprocess.run(["swaymsg", "exec", exec_cmd], capture_output=True, text=True)
            if res.returncode == 0:
                return f"Launched {display_name} via Sway."
        # 3. i3
        if os.environ.get("DISPLAY") and shutil.which("i3-msg"):
            res = subprocess.run(["i3-msg", "exec", exec_cmd], capture_output=True, text=True)
            if res.returncode == 0:
                return f"Launched {display_name} via i3."
        # 4. Freedesktop gtk-launch (only if no custom args provided)
        if selected and not args and shutil.which("gtk-launch") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            res = subprocess.run(["gtk-launch", selected["desktop_id"]], capture_output=True, text=True, env=os.environ)
            if res.returncode == 0:
                return f"Launched {display_name} via gtk-launch."
        # 5. Freedesktop gio launch (only if no custom args provided)
        if selected and not args and shutil.which("gio") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            res = subprocess.run(["gio", "launch", selected["desktop_path"]], capture_output=True, text=True, env=os.environ)
            if res.returncode == 0:
                return f"Launched {display_name} via gio."

        subprocess.Popen(shlex.split(exec_cmd), start_new_session=True, env=os.environ)
        return f"Launched {display_name}."
    except Exception as e:
        return f"Failed to launch {display_name}: {e}"


def open_in_browser(query_or_url: str) -> str:
    """Opens a website URL or performs a search query in the configured default browser."""
    target = (query_or_url or "").strip()
    if not target:
        return launch_application("browser")

    # Detect if it's already a full URL or domain
    if re.match(r"^https?://", target, re.IGNORECASE):
        url = target
    elif re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$", target):
        url = f"https://{target}"
    else:
        import urllib.parse
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(target)}"

    return launch_application("browser", args=url)



def focus_window(target: str) -> str:
    """Brings a window matching the target title, class, or application alias to the front across compositors."""
    ensure_gui_environment()
    backend = get_active_backend()
    return backend.focus_window(target)


def _is_browser_window(window: dict[str, Any]) -> bool:
    identity = f"{window.get('class', '')} {window.get('title', '')}"
    return bool(re.search(r"browser|firefox|mozilla|chrom(e|ium)|edge|msedge|brave|vivaldi|opera|zen", identity, re.I))


def _browser_title_state(title: str) -> tuple[str, int | None]:
    """Return the visible page title and, where Chromium exposes it, tab count."""
    count_match = re.search(r"\band\s+(\d+)\s+more\s+pages\b", title, re.I)
    count = int(count_match.group(1)) + 1 if count_match else None
    clean = re.sub(r"\s+and\s+\d+\s+more\s+pages\b", "", title, flags=re.I)
    clean = re.sub(r"\s+-\s+(?:Personal|Work|Profile\s*\d*)\s+-\s+.*$", "", clean, flags=re.I)
    clean = re.sub(r"\s+-\s+(?:Microsoft\s*)?Edge\s*$", "", clean, flags=re.I)
    clean = re.sub(r"\s+-\s+Firefox\s*$", "", clean, flags=re.I)
    return clean.strip(), count


def _send_browser_shortcut(key: str) -> tuple[bool, str]:
    """Send a browser shortcut with the session's native input backend."""
    if (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")):
        if shutil.which("wtype"):
            key_arg = "-k" if key == "Tab" else None
            args = ["wtype", "-M", "ctrl"]
            args.extend([key_arg, key] if key_arg else [key.lower()])
            args.extend(["-m", "ctrl"])
            try:
                result = subprocess.run(args, capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    return True, ""
            except Exception as exc:
                error = str(exc)
            else:
                error = result.stderr.strip()
        else:
            error = "wtype is unavailable"

        if shutil.which("ydotool"):
            # Ctrl=29, W=17, Tab=15 (Linux evdev key codes).
            keycode = "17" if key == "w" else "15"
            try:
                runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
                socket_path = Path(runtime) / ".ydotool_socket"
                if not socket_path.exists() and shutil.which("systemctl"):
                    subprocess.run(["systemctl", "--user", "start", "ydotool.service"], capture_output=True, text=True, timeout=5)
                    for _ in range(20):
                        if socket_path.exists():
                            break
                        time.sleep(0.1)
                if socket_path.exists():
                    result = subprocess.run(
                        ["ydotool", "key", "29:1", f"{keycode}:1", f"{keycode}:0", "29:0"],
                        capture_output=True, text=True, timeout=3,
                    )
                    if result.returncode == 0:
                        return True, ""
                    error = result.stderr.strip()
            except Exception as exc:
                error = str(exc)
        return False, error or "Wayland keyboard input failed"

    if shutil.which("xdotool") and os.environ.get("DISPLAY"):
        try:
            result = subprocess.run(["xdotool", "key", f"ctrl+{key.lower()}"], capture_output=True, text=True, timeout=3)
            return result.returncode == 0, result.stderr.strip()
        except Exception as exc:
            return False, str(exc)
    return False, "No keyboard input backend is available (install wtype/ydotool on Wayland or xdotool on X11)."


def close_browser_tab(
    target: str = "browser",
    title_contains: str = "",
    all_matches: bool = False,
) -> str:
    """Close the active browser tab, optionally finding and closing tabs by visible title."""
    ensure_gui_environment()
    browser_target = (target or "browser").strip()
    query = (title_contains or "").strip()

    focus_res = focus_window(browser_target)
    if "not found" in focus_res.lower() or "error" in focus_res.lower():
        if browser_target.casefold() != "browser":
            focus_res = focus_window("browser")
        if "not found" in focus_res.lower() or "error" in focus_res.lower():
            return f"Could not find open browser window: {focus_res}"

    time.sleep(0.15)
    current = _active_window_metadata()
    if not _is_browser_window(current):
        return "Could not safely close a tab: the focused window is not identifiable as a browser."
    if not current.get("title"):
        return "The browser is focused, but this desktop does not expose its active tab title, so the close cannot be verified."

    start_title = str(current["title"])
    page_title, _ = _browser_title_state(start_title)
    visited: set[tuple[str, int | None]] = set()
    closed = 0
    last_error = ""
    max_steps = 100

    for _ in range(max_steps):
        title = str(current.get("title") or "")
        page_title, tab_count = _browser_title_state(title)
        state = (page_title.casefold(), tab_count)
        if state in visited:
            break
        visited.add(state)

        matches = not query or query.casefold() in title.casefold()
        if matches:
            after = current
            for attempt in range(2):
                sent, last_error = _send_browser_shortcut("w")
                if not sent:
                    return f"Failed to send Ctrl+W: {last_error}"
                # Some browser pages briefly swallow the first shortcut while
                # loading or focused on an embedded control. Recheck, then retry
                # once only while the exact same tab is still active.
                time.sleep(0.35 if attempt == 0 else 0.5)
                after = _active_window_metadata()
                after_title = str(after.get("title") or "")
                if after_title != title:
                    break
            else:
                return f"Ctrl+W was sent twice, but the browser still shows the same tab ('{page_title[:80]}'); it was not verified closed."
            after_title = str(after.get("title") or "")
            if after_title and not _is_browser_window(after):
                closed += 1
                if query and tab_count == 1:
                    return f"Closed the matching last browser tab ('{page_title[:80]}'); the browser window closed too."
                return "The browser window closed while attempting to close a tab; stopping without further input."
            closed += 1
            if not query or not all_matches:
                return f"Closed browser tab '{page_title[:100]}' and verified the active tab changed."
            current = after
            continue

        if not query:
            break
        sent, last_error = _send_browser_shortcut("Tab")
        if not sent:
            return f"Could not search browser tabs: {last_error}"
        time.sleep(0.2)
        current = _active_window_metadata()
        next_title = str(current.get("title") or "")
        if not _is_browser_window(current):
            return f"Tab search left the browser unexpectedly; closed {closed} matching tab(s)."
        if next_title == start_title:
            break

    if closed:
        return f"Closed and verified {closed} tab(s) matching '{query}'."
    if query:
        return f"No open browser tab title matched '{query}'. Checked {len(visited)} distinct tab state(s)."
    return "No browser tab was closed."



def close_application(app_name: str) -> str:
    """Closes an application or its window cleanly across compositors."""
    ensure_gui_environment()
    backend = get_active_backend()
    return backend.close_application(app_name)


def list_windows() -> str:
    """Lists currently open desktop windows and their workspace IDs across compositors."""
    ensure_gui_environment()
    backend = get_active_backend()
    return backend.list_windows()


def get_open_windows_prompt_context() -> str:
    """Returns a compact structured summary of currently open applications and active desktop state
    suitable for prompt injection into the LLM context."""
    ensure_gui_environment()
    backend = get_active_backend()
    return backend.get_open_windows_prompt_context()


def swap_windows(window_one: str, window_two: str) -> str:
    """Swap two matching windows' layout positions on the current workspace."""
    ensure_gui_environment()
    if not is_capability_supported("window_swap"):
        backend = get_active_backend()
        return f"Reordering or swapping windows is not supported on {backend.name}."
    backend = get_active_backend()
    return backend.swap_windows(window_one, window_two)


_WINDOW_PREVIOUS_WORKSPACES: dict[str, str] = {}
_LAST_MOVED_WINDOW: dict[str, Any] = {}


def workspace_control(action: str, workspace: Optional[str] = "", target: Optional[str] = "") -> str:
    """Switches active workspace or moves a specific window or active window to a workspace across compositors."""
    global _WINDOW_PREVIOUS_WORKSPACES, _LAST_MOVED_WINDOW
    ensure_gui_environment()
    ws = (workspace or "").strip()
    if ws.lower().startswith("workspace"):
        ws = ws[len("workspace"):].strip()
    word_to_num = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
    }
    if ws.lower() in word_to_num:
        ws = word_to_num[ws.lower()]

    tgt = (target or "").strip()
    # Resolve pronouns (it, that, the app, the window)
    if tgt.lower() in ["it", "that", "this", "the window", "the app"] and _LAST_MOVED_WINDOW:
        tgt = _LAST_MOVED_WINDOW.get("class") or _LAST_MOVED_WINDOW.get("address") or tgt

    is_current = ws.lower() in ["current", "active", "this", "here", "this workspace", "current workspace"]
    is_back = ws.lower() in ["back", "previous", "prev", "undo", "orig", "original", "last"]

    # Resolve 'back' / 'previous' / 'undo' destination workspace
    if is_back:
        if action in ["move", "movetoworkspace"]:
            if not tgt and _LAST_MOVED_WINDOW:
                tgt = _LAST_MOVED_WINDOW.get("class") or _LAST_MOVED_WINDOW.get("address") or ""
            prev_ws = None
            if tgt:
                prev_ws = _WINDOW_PREVIOUS_WORKSPACES.get(tgt.lower()) or _WINDOW_PREVIOUS_WORKSPACES.get(_clean_str(tgt).lower())
            if not prev_ws and _LAST_MOVED_WINDOW:
                prev_ws = _LAST_MOVED_WINDOW.get("previous_workspace")
            if prev_ws:
                ws = prev_ws
            elif not ws:
                ws = "previous"
        else:
            if shutil.which("swaymsg") or shutil.which("i3-msg"):
                ws = "back_and_forth"
            else:
                ws = "previous"

    if not ws:
        return "Please specify a target workspace number or name."

    backend = get_active_backend()
    return backend.workspace_control(action, ws, tgt, is_current=is_current, is_back=is_back)


def manage_clipboard(action: str, text: Optional[str] = "") -> str:
    """Reads or sets system clipboard content using standard Wayland or X11 tools."""
    ensure_gui_environment()
    act = (action or "read").strip().lower()

    if act == "read":
        if shutil.which("wl-paste") and os.environ.get("WAYLAND_DISPLAY"):
            res = subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True)
            clip = res.stdout
            if not clip:
                return "The clipboard is currently empty."
            return f"Clipboard contents: {clip[:500]}"
        elif shutil.which("xclip"):
            res = subprocess.run(["xclip", "-o", "-selection", "clipboard"], capture_output=True, text=True)
            clip = res.stdout
            if not clip:
                return "The clipboard is currently empty."
            return f"Clipboard contents: {clip[:500]}"
        return "Clipboard tool (wl-clipboard or xclip) is not installed."

    elif act == "write":
        val = text or ""
        if shutil.which("wl-copy") and os.environ.get("WAYLAND_DISPLAY"):
            proc = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE, text=True)
            proc.communicate(input=val)
            return "Copied text to clipboard."
        elif shutil.which("xclip"):
            proc = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE, text=True)
            proc.communicate(input=val)
            return "Copied text to clipboard."
        return "Clipboard tool is not available."

    return f"Unknown clipboard action: {action}"


def execute_desktop_macro(macro_name: str) -> str:
    """Executes a user-defined macro from config or a default macro for the active desktop environment."""
    ensure_gui_environment()
    name = (macro_name or "").strip().lower()
    if not name:
        return "Please specify a macro name to run."

    backend = get_active_backend()
    default_macros = backend.get_default_macros()
    all_macros = {**UNIVERSAL_SYSTEM_MACROS, **default_macros, **DESKTOP_MACROS}

    if name not in all_macros:
        avail = ", ".join(sorted(all_macros.keys()))
        return f"Unknown desktop macro '{macro_name}'. Available macros: {avail}."

    cmd_or_spec = all_macros[name]
    commands = []
    if isinstance(cmd_or_spec, str):
        commands = [cmd_or_spec]
    elif isinstance(cmd_or_spec, list):
        commands = [c for c in cmd_or_spec if isinstance(c, str)]
    elif isinstance(cmd_or_spec, dict):
        if "command" in cmd_or_spec:
            commands = [cmd_or_spec["command"]]
        elif "commands" in cmd_or_spec:
            commands = cmd_or_spec["commands"]

    if not commands:
        return f"Macro '{macro_name}' has no executable commands defined."

    for cmd in commands:
        try:
            if isinstance(backend, HyprlandBackend) and name in default_macros and name not in {"lock", "suspend", "reboot", "restart", "poweroff", "shutdown"}:
                lua_macros = {
                    "toggle_floating": "hl.dsp.window.float()",
                    "fullscreen": 'hl.dsp.window.fullscreen({ mode = "maximized" })',
                    "pin": "hl.dsp.window.pin()",
                    "split": 'hl.dsp.layout("togglesplit")',
                }
                legacy = shlex.split(cmd)
                dispatcher = legacy[2] if len(legacy) > 2 else ""
                argument = " ".join(legacy[3:])
                res = _hyprland_dispatch(dispatcher, argument, lua_expression=lua_macros[name])
            else:
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10, env=os.environ)
            if res.returncode != 0:
                err = res.stderr.strip() or res.stdout.strip()
                return f"Macro '{macro_name}' failed at step '{cmd}': {err}"
        except Exception as e:
            return f"Macro '{macro_name}' failed: {e}"

    return f"Triggered desktop macro '{macro_name}'."


def list_desktop_macros() -> dict[str, str]:
    """Returns all available desktop macros (default for active DE + user-defined)."""
    ensure_gui_environment()
    backend = get_active_backend()
    default_macros = backend.get_default_macros()
    result = {}
    for k, v in {**UNIVERSAL_SYSTEM_MACROS, **default_macros, **DESKTOP_MACROS}.items():
        if isinstance(v, str):
            result[k] = v
        elif isinstance(v, dict):
            result[k] = v.get("description", str(v.get("command") or v.get("commands", "")))
        else:
            result[k] = str(v)
    return result


def show_desktop_notification(
    title: str,
    message: str,
    urgency: str = "normal",
    timeout_ms: int = 6000,
) -> str:
    """Displays an on-screen desktop notification popup via notify-send without speaking aloud."""
    ensure_gui_environment()
    urgency_val = urgency if urgency in ["low", "normal", "critical"] else "normal"
    cmd = [
        "notify-send",
        "-a", "Adam",
        "-u", urgency_val,
        "-t", str(timeout_ms),
        str(title),
        str(message)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, env=os.environ)
        if res.returncode == 0:
            return f"Notification displayed: '{title}'"
        return f"notify-send returned code {res.returncode}: {res.stderr.strip()}"
    except Exception as e:
        return f"Failed to show notification: {e}"
