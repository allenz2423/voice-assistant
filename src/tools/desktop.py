import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Any, Dict, List, Set

_HYPRLAND_LUA_DISPATCH: Optional[bool] = None


def _lua_string(value: str) -> str:
    """Quote a Python string as a Lua string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return f'"{escaped}"'


def _hyprland_dispatch(legacy_dispatch: str, legacy_args: str = "", *, lua_expression: str) -> subprocess.CompletedProcess:
    """Run a dispatcher on both legacy hyprlang and Lua-config Hyprland versions."""
    global _HYPRLAND_LUA_DISPATCH
    if _HYPRLAND_LUA_DISPATCH is None:
        try:
            version = subprocess.run(["hyprctl", "version"], capture_output=True, text=True, timeout=2).stdout
            match = re.search(r"Hyprland\s+(\d+)\.(\d+)", version)
            _HYPRLAND_LUA_DISPATCH = bool(match and tuple(map(int, match.groups())) >= (0, 55))
        except Exception:
            _HYPRLAND_LUA_DISPATCH = False

    if _HYPRLAND_LUA_DISPATCH:
        command = ["hyprctl", "dispatch", lua_expression]
    else:
        command = ["hyprctl", "dispatch", legacy_dispatch]
        if legacy_args:
            command.append(legacy_args)
    return subprocess.run(command, capture_output=True, text=True, timeout=3)

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


def detect_desktop_environment(refresh_env: bool = True) -> str:
    """Detects the currently running Desktop Environment or Window Manager."""
    if refresh_env:
        ensure_gui_environment()

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
    # If swaymsg is mocked or present and SWAYSOCK is set
    if os.environ.get("SWAYSOCK") and (shutil.which("swaymsg") or not shutil.which("hyprctl")):
        return "sway"
    if os.environ.get("I3SOCK") and (shutil.which("i3-msg") or not shutil.which("hyprctl")):
        return "i3"
    if os.environ.get("NIRI_SOCKET"):
        return "niri"
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") and (shutil.which("hyprctl") or not shutil.which("swaymsg")):
        return "hyprland"
    if os.environ.get("KDE_SESSION_VERSION"):
        return "kde_plasma"

    # 3. Session variables
    if "sway" in desktop_session and (shutil.which("swaymsg") or not shutil.which("hyprctl")):
        return "sway"
    if "i3" in desktop_session and (shutil.which("i3-msg") or not shutil.which("hyprctl")):
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

    # 4. Fallback: Binary presence check when display server is active
    if os.environ.get("WAYLAND_DISPLAY"):
        if shutil.which("swaymsg") and os.environ.get("SWAYSOCK"):
            return "sway"
        if shutil.which("hyprctl") and os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
            return "hyprland"
        if shutil.which("niri") and os.environ.get("NIRI_SOCKET"):
            return "niri"
        if shutil.which("hyprctl"):
            return "hyprland"
        if shutil.which("swaymsg"):
            return "sway"
        if shutil.which("cosmic-comp"):
            return "cosmic"

    if os.environ.get("DISPLAY"):
        if shutil.which("i3-msg"):
            return "i3"

    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return "hyprland"
    if os.environ.get("SWAYSOCK"):
        return "sway"
    if os.environ.get("I3SOCK"):
        return "i3"

    return "generic_desktop"


def _clean_str(s: str) -> str:
    """Removes invisible unicode characters (zero-width spaces) and whitespace."""
    if not s:
        return ""
    return s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "").strip()


def _get_app_directories() -> list[Path]:
    """Returns standard XDG application directories compliant with Freedesktop spec."""
    dirs = []
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        dirs.append(Path(data_home) / "applications")
    else:
        dirs.append(Path.home() / ".local" / "share" / "applications")

    data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")
    for d in data_dirs.split(":"):
        if d.strip():
            dirs.append(Path(d.strip()) / "applications")

    return [d for d in dirs if d.exists()]


def _scan_desktop_entries() -> dict[str, dict]:
    """Scans all standard desktop entries across the system."""
    apps = {}
    for app_dir in _get_app_directories():
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
                    clean_exec = re.sub(r"%[a-zA-Z0-9%]", "", clean_exec).strip()
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
    return apps


APPLICATION_ALIASES: dict[str, list[str]] = {}
WINDOW_ALIASES: dict[str, list[str]] = {}
DESKTOP_MACROS: dict[str, Any] = {}
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


def _resolve_browser_entry(apps: dict[str, dict]) -> Optional[dict]:
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
    for app_dir in _get_app_directories():
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
    bin_path = shutil.which(target_browser) or shutil.which(target_stem)
    if bin_path:
        return {
            "name": target_browser,
            "exec": bin_path,
            "comment": "Default Web Browser",
            "desktop_path": "",
            "desktop_id": f"{target_stem}.desktop",
        }

    return None


def _resolve_application_entry(target: str, apps: dict[str, dict]) -> Optional[dict]:
    """Resolves an app name to a desktop entry using exact matches, aliases, length-ranked substrings, and PATH fallback."""
    clean_target = (target or "").strip().lower()
    if not clean_target:
        return None

    # Check if target is a generic browser request or matches the default browser directly
    if clean_target in ["browser", "web browser", "web-browser", "default browser", "internet"]:
        browser_entry = _resolve_browser_entry(apps)
        if browser_entry:
            return browser_entry
    elif clean_target == DEFAULT_BROWSER.lower() or clean_target == Path(DEFAULT_BROWSER.lower()).stem:
        browser_entry = _resolve_browser_entry(apps)
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
    bin_path = shutil.which(clean_target)
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

    def get_default_macros(self) -> dict[str, str]:
        return {"lock": "loginctl lock-session"}


class HyprlandBackend(BaseDesktopBackend):
    name = "hyprland"

    def get_capabilities(self) -> set[str]:
        return {
            "window_focus", "window_move", "workspace_switch",
            "window_close", "window_list", "window_swap",
            "screenshot", "desktop_macro", "app_launch", "app_list"
        }

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
        result = subprocess.run(
            ["grim", "-"],
            capture_output=True,
            timeout=15,
            env=os.environ,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or "Screenshot capture failed.")
        if not result.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Screenshot capture returned invalid PNG data.")
        return result.stdout

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
            env=os.environ,
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
            if shutil.which(tool):
                res = subprocess.run(args, capture_output=True, timeout=15, env=os.environ)
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
        if shutil.which("grim"):
            res = subprocess.run(["grim", "-"], capture_output=True, timeout=15, env=os.environ)
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
        if shutil.which("spectacle"):
            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                res = subprocess.run(
                    ["spectacle", "-b", "-n", "-o", tmp_path],
                    capture_output=True,
                    timeout=15,
                    env=os.environ
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

        if shutil.which("grim"):
            res = subprocess.run(["grim", "-"], capture_output=True, timeout=15, env=os.environ)
            if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                return res.stdout

        for tool, args in [("maim", ["maim"]), ("scrot", ["scrot", "-"]), ("import", ["import", "-window", "root", "png:-"])]:
            if shutil.which(tool):
                res = subprocess.run(args, capture_output=True, timeout=15, env=os.environ)
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
        if shutil.which("gnome-screenshot"):
            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                res = subprocess.run(["gnome-screenshot", "-f", tmp_path], capture_output=True, timeout=15, env=os.environ)
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

        if shutil.which("grim"):
            res = subprocess.run(["grim", "-"], capture_output=True, timeout=15, env=os.environ)
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
        if shutil.which("grim"):
            res = subprocess.run(["grim", "-"], capture_output=True, timeout=15, env=os.environ)
            if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                return res.stdout
        if shutil.which("cosmic-screenshot"):
            res = subprocess.run(["cosmic-screenshot"], capture_output=True, timeout=15, env=os.environ)
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
            if shutil.which(tool):
                res = subprocess.run(args, capture_output=True, timeout=15, env=os.environ)
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


def get_active_backend(refresh_env: bool = True) -> BaseDesktopBackend:
    """Returns the backend instance for the active desktop environment."""
    ensure_gui_environment()
    xdg_current = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    desktop_session = os.environ.get("DESKTOP_SESSION", "").lower()

    # 1. Direct tool and socket matches
    # Hyprland
    if shutil.which("hyprctl") and (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or (os.environ.get("WAYLAND_DISPLAY") and "hyprland" in xdg_current)):
        return HyprlandBackend()

    # Sway
    if shutil.which("swaymsg") and (os.environ.get("SWAYSOCK") or os.environ.get("WAYLAND_DISPLAY")):
        return SwayBackend()

    # i3
    if shutil.which("i3-msg") and (os.environ.get("I3SOCK") or os.environ.get("DISPLAY")):
        return I3Backend()

    # Niri
    if shutil.which("niri") and (os.environ.get("NIRI_SOCKET") or "niri" in xdg_current):
        return NiriBackend()

    # KDE Plasma
    if "kde" in xdg_current or "plasma" in xdg_current or os.environ.get("KDE_SESSION_VERSION") or "kde" in desktop_session or shutil.which("kdotool"):
        return KdePlasmaBackend()

    # GNOME
    if "gnome" in xdg_current or "gnome" in desktop_session:
        return GnomeBackend()

    # COSMIC
    if "cosmic" in xdg_current or shutil.which("cosmic-comp"):
        return CosmicBackend()

    de = detect_desktop_environment(refresh_env=refresh_env)
    backend_cls = _BACKEND_MAP.get(de, GenericDesktopBackend)
    return backend_cls()


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

def capture_screenshot() -> bytes:
    """Captures the current desktop as PNG bytes without writing an image file."""
    ensure_gui_environment()
    backend = get_active_backend()
    return backend.capture_screenshot()


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

    exec_cmd = selected["exec"] if selected else target
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


def close_browser_tab(target: str = "browser") -> str:
    """Focuses the browser window and closes the active tab via Ctrl+W."""
    ensure_gui_environment()
    browser_target = (target or "browser").strip()

    focus_res = focus_window(browser_target)
    if "not found" in focus_res.lower() or "error" in focus_res.lower():
        if browser_target != "browser":
            focus_res = focus_window("browser")
            if "not found" in focus_res.lower() or "error" in focus_res.lower():
                return f"Could not find open browser window to close tab: {focus_res}"
        else:
            return f"No open browser window found: {focus_res}"

    time.sleep(0.08)

    # Wayland: wtype
    if shutil.which("wtype") and (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")):
        try:
            res = subprocess.run(["wtype", "-M", "ctrl", "w", "-m", "ctrl"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                return "Closed active browser tab."
        except Exception:
            pass

    # Generic Linux: ydotool
    if shutil.which("ydotool"):
        try:
            res = subprocess.run(["ydotool", "key", "29:1", "17:1", "17:0", "29:0"], capture_output=True, timeout=3)
            if res.returncode == 0:
                return "Closed active browser tab."
        except Exception:
            pass

    # X11 fallback: xdotool
    if shutil.which("xdotool") and os.environ.get("DISPLAY"):
        try:
            res = subprocess.run(["xdotool", "key", "ctrl+w"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                return "Closed active browser tab."
        except Exception:
            pass

    return "Failed to send close tab shortcut: no virtual keyboard input tool found (install wtype or ydotool)."



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
    all_macros = {**default_macros, **DESKTOP_MACROS}

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
            if isinstance(backend, HyprlandBackend) and name in default_macros and name != "lock":
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
    for k, v in {**default_macros, **DESKTOP_MACROS}.items():
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
