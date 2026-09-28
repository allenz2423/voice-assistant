import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

def check_system_updates() -> str:
    """Checks for pending package and system updates without taking locks."""
    # Arch Linux / Pacman checkupdates
    if shutil.which("checkupdates"):
        try:
            res = subprocess.run(["checkupdates"], capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                lines = [l for l in res.stdout.strip().splitlines() if l.strip()]
                count = len(lines)
                if count == 0:
                    return "System is completely up to date. No pending updates found."
                sample = ", ".join([l.split()[0] for l in lines[:8]])
                return f"There are {count} pending package updates available (including: {sample})."
            elif res.returncode == 2:
                return "System is completely up to date. No pending updates found."
        except Exception:
            pass

    # Debian / Ubuntu apt
    if shutil.which("apt"):
        try:
            res = subprocess.run(["apt", "list", "--upgradable"], capture_output=True, text=True, timeout=5)
            lines = [l for l in res.stdout.strip().splitlines() if "upgradable" in l]
            if lines:
                return f"There are {len(lines)} pending updates available via apt."
            return "No pending apt updates found."
        except Exception:
            pass

    return "Package update checker is not available for this distribution."

def manage_service(action: str, service_name: str) -> str:
    """Checks status or restarts/stops a systemd user or system service."""
    act = (action or "status").strip().lower()
    svc = (service_name or "").strip()
    if not svc:
        return "Please specify a service name (e.g., 'adam', 'pipewire', 'docker', 'sunadame')."

    if not shutil.which("systemctl"):
        return "systemctl is not available on this system."

    # Try user service first
    res_user = subprocess.run(["systemctl", "--user", "is-active", svc], capture_output=True, text=True)
    if res_user.returncode in [0, 3]:  # Active or inactive user service
        if act == "status":
            st = res_user.stdout.strip()
            return f"User service '{svc}' is currently {st}."
        elif act in ["restart", "start", "stop"]:
            subprocess.run(["systemctl", "--user", act, svc], check=False)
            return f"Executed '{act}' on user service '{svc}'."

    # Try system service
    res_sys = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True)
    if act == "status":
        st = res_sys.stdout.strip()
        return f"System service '{svc}' is currently {st}."
    elif act in ["restart", "start", "stop"]:
        # If restarting a system service, report status or note sudo
        proc = subprocess.run(["systemctl", act, svc], capture_output=True, text=True)
        if proc.returncode == 0:
            return f"Successfully executed '{act}' on system service '{svc}'."
        return f"Could not {act} system service '{svc}' (may require root privileges): {proc.stderr.strip()}"

    return f"Service '{svc}' not found or unrecognized action '{action}'."

def git_repo_status(repo_path: Optional[str] = "") -> str:
    """Summarizes git branch, modified files, and status in the current or specified repository."""
    if not shutil.which("git"):
        return "git utility is not installed."

    path = Path(repo_path).expanduser() if repo_path else Path.cwd()
    if not path.exists():
        return f"Path '{path}' does not exist."

    try:
        # Check current branch
        branch_res = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            capture_output=True, text=True, timeout=2
        )
        if branch_res.returncode != 0:
            return f"Directory '{path.name}' is not a git repository."
        branch = branch_res.stdout.strip() or "detached HEAD"

        # Check status
        status_res = subprocess.run(
            ["git", "-C", str(path), "status", "-s"],
            capture_output=True, text=True, timeout=2
        )
        status_lines = [l for l in status_res.stdout.strip().splitlines() if l.strip()]
        dirty_count = len(status_lines)

        if dirty_count == 0:
            return f"In repository '{path.name}', you are on branch '{branch}' with a clean working tree."
        else:
            sample = ", ".join([l.strip().split()[-1] for l in status_lines[:5]])
            return f"In repository '{path.name}', you are on branch '{branch}' with {dirty_count} uncommitted changes ({sample})."
    except Exception as e:
        return f"Error reading git status: {e}"

def docker_container_status() -> str:
    """Reports health and status of local Docker containers."""
    if not shutil.which("docker"):
        return "Docker is not installed on this system."

    try:
        res = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}: {{.Status}}"],
            capture_output=True, text=True, timeout=3
        )
        if res.returncode != 0:
            return f"Docker daemon is not running or accessible: {res.stderr.strip()}"

        lines = [l for l in res.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return "No Docker containers found on this system."
        return f"Docker containers ({len(lines)} total): " + ", ".join(lines)
    except Exception as e:
        return f"Error checking Docker status: {e}"
