"""Small adapter for the user's Remind configuration directory."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path


class ReminderManager:
    """Manage Adam-owned entries while leaving Remind files user-editable."""

    def __init__(self, config_dir: str | Path | None = None, executable: str = "remind"):
        self.config_dir = Path(config_dir or Path.home() / ".config" / "remind").expanduser()
        self.adam_file = self.config_dir / "adam.rem"
        self.executable = executable

    def _available(self) -> bool:
        return shutil.which(self.executable) is not None

    def _run(self, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.executable, *args], input=input_text, capture_output=True, text=True,
            timeout=10, check=False,
        )

    def _error(self, result: subprocess.CompletedProcess) -> str | None:
        if result.returncode == 0:
            return None
        detail = (result.stderr or result.stdout).strip().splitlines()
        return detail[0] if detail else "Remind rejected the reminder configuration."

    def create_reminder(self, message: str, when: str, repeat: str = "none") -> str:
        """Create a local-time reminder. `when` is an ISO date/time."""
        if not self._available():
            return "Remind is not installed. Install the system package 'remind' to use reminders."
        message = " ".join((message or "").replace("\\", " ").split())
        if not message:
            return "Please include a reminder message."
        try:
            moment = datetime.fromisoformat(when)
        except (TypeError, ValueError):
            return "Give the reminder time as a date and time, for example 2026-09-28T09:00."
        if moment.tzinfo is not None:
            moment = moment.astimezone()
        if moment <= datetime.now().astimezone().replace(tzinfo=None):
            return "That reminder time has already passed."

        date_text = moment.strftime("%-d %b %Y")
        clock_text = moment.strftime("%H:%M")
        date_rules = {
            "none": date_text,
            "daily": "",
            "weekly": moment.strftime("%a"),
            "monthly": moment.strftime("%-d"),
            "yearly": moment.strftime("%-d %b"),
        }
        if repeat not in date_rules:
            return "Repeat must be none, daily, weekly, monthly, or yearly."
        rule = date_rules[repeat]
        rem_line = f"REM {rule} AT {clock_text} MSG {message}".replace("REM  AT", "REM AT")
        # Ask Remind to parse the exact generated line before writing user data.
        check = self._run("-n", "-", input_text=rem_line + "\n")
        error = self._error(check)
        if error:
            return f"Could not create that reminder: {error}"

        reminder_id = uuid.uuid4().hex[:8]
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with self.adam_file.open("a", encoding="utf-8") as file:
            file.write(f"# ADAM-ID:{reminder_id}\n{rem_line}\n")
        return f"Reminder set for {moment.strftime('%A, %B %-d at %-I:%M %p')} (ID {reminder_id})."

    def list_reminders(self) -> str:
        if not self._available():
            return "Remind is not installed. Install the system package 'remind' to use reminders."
        if not self.config_dir.exists():
            return "There are no upcoming reminders."
        result = self._run("-n", str(self.config_dir))
        error = self._error(result)
        if error:
            return f"Could not read reminders: {error}"
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return "Upcoming reminders: " + "; ".join(lines[:20]) if lines else "There are no upcoming reminders."

    def show_calendar(self, months: int = 1) -> str:
        if not self._available():
            return "Remind is not installed. Install the system package 'remind' to use the calendar."
        months = max(1, min(3, int(months)))
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
        result = self._run(f"-c{months}", "-m", str(self.config_dir))
        error = self._error(result)
        return f"Could not show the calendar: {error}" if error else result.stdout.strip()

    def cancel_reminder(self, reminder_id: str) -> str:
        target = (reminder_id or "").strip().lower()
        if not target:
            return "Please specify a reminder ID."
        if not self.adam_file.exists():
            return f"No Adam reminder found with ID {target}."
        lines = self.adam_file.read_text(encoding="utf-8").splitlines()
        kept: list[str] = []
        removed = False
        skip_entry = False
        for line in lines:
            if line.startswith("# ADAM-ID:"):
                if line.partition(":")[2].strip().lower() == target:
                    removed = True
                    skip_entry = True
                    continue
                skip_entry = False
            if skip_entry and line.startswith("REM "):
                skip_entry = False
                continue
            kept.append(line)
        if not removed:
            return f"No Adam reminder found with ID {target}."
        content = "\n".join(kept).rstrip()
        self.adam_file.write_text(content + ("\n" if content else ""), encoding="utf-8")
        return f"Cancelled reminder {target}."
