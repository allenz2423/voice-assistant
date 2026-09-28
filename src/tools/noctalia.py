"""Thin wrappers for Noctalia's documented IPC commands."""

import shutil
import subprocess


def open_noctalia_calendar() -> str:
    """Open Noctalia's calendar tab through its IPC CLI."""
    executable = shutil.which("noctalia")
    if not executable:
        return "Noctalia is not available on this system."
    try:
        result = subprocess.run(
            [executable, "msg", "panel-open", "control-center", "calendar"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "Could not contact Noctalia."
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        return f"Noctalia could not open the calendar{': ' + detail if detail else '.'}"
    return "Opened the Noctalia calendar."
