"""Write Adam-owned iCalendar events to a Noctalia-watched local vdir."""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


class NoctaliaCalendar:
    def __init__(self, collection: str | Path | None = None):
        self.collection = Path(
            collection or Path.home() / ".local" / "share" / "noctalia-calendar" / "adam"
        ).expanduser()
        self.metadata = self.collection / "displayname"

    @staticmethod
    def _escape_ical(value: str) -> str:
        value = (value or "").replace("\\", "\\\\").replace("\r", " ").replace("\n", "\\n")
        return value.replace(",", "\\,").replace(";", "\\;")

    @staticmethod
    def _fold_line(line: str) -> str:
        chunks: list[str] = []
        current, limit = "", 75
        for char in line:
            if current and len((current + char).encode("utf-8")) > limit:
                chunks.append(current)
                current, limit = " " + char, 75
            else:
                current += char
        chunks.append(current)
        return "\r\n".join(chunks)

    @staticmethod
    def _parse_when(when: str) -> datetime:
        value = datetime.fromisoformat(when)
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return value.astimezone()

    def _ensure_collection(self) -> None:
        self.collection.mkdir(parents=True, exist_ok=True)
        if not self.metadata.exists():
            self.metadata.write_text("Adam\n", encoding="utf-8")

    def create_event(
        self,
        title: str,
        when: str,
        duration_minutes: int = 15,
        alarm_minutes: int = 10,
        repeat: str = "none",
    ) -> str:
        title = " ".join((title or "").split())
        if not title:
            return "Please include a calendar event title."
        try:
            start = self._parse_when(when)
        except (TypeError, ValueError):
            return "Give the event time as a date and time, for example 2026-09-28T09:00."
        if start <= datetime.now().astimezone():
            return "That event time has already passed."
        if repeat not in {"none", "daily", "weekly", "monthly", "yearly"}:
            return "Repeat must be none, daily, weekly, monthly, or yearly."
        try:
            duration_minutes = max(1, min(1440, int(duration_minutes)))
            alarm_minutes = max(0, min(10080, int(alarm_minutes)))
        except (TypeError, ValueError):
            return "Duration and alarm lead time must be whole minutes."

        event_id = uuid.uuid4().hex
        now_utc = datetime.now(timezone.utc)
        end = start + timedelta(minutes=duration_minutes)
        lines = [
            "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Adam//Noctalia Calendar//EN",
            "CALSCALE:GREGORIAN", "BEGIN:VEVENT", f"UID:{event_id}@adam.local",
            f"X-ADAM-ID:{event_id}", f"DTSTAMP:{now_utc:%Y%m%dT%H%M%SZ}",
            f"DTSTART:{start:%Y%m%dT%H%M%S}", f"DTEND:{end:%Y%m%dT%H%M%S}",
            f"SUMMARY:{self._escape_ical(title)}",
        ]
        if repeat != "none":
            lines.append(f"RRULE:FREQ={repeat.upper()}")
        lines.extend([
            "BEGIN:VALARM",
            f"TRIGGER:-PT{alarm_minutes}M" if alarm_minutes else "TRIGGER:PT0S",
            "ACTION:DISPLAY", f"DESCRIPTION:{self._escape_ical(title)}",
            "END:VALARM", "END:VEVENT", "END:VCALENDAR",
        ])
        self._ensure_collection()
        path = self.collection / f"{event_id}.ics"
        temp = path.with_suffix(".ics.tmp")
        content = "\r\n".join(self._fold_line(line) for line in lines) + "\r\n"
        temp.write_text(content, encoding="utf-8", newline="")
        os.replace(temp, path)
        alarm = "at the event time" if alarm_minutes == 0 else f"{alarm_minutes} minutes before"
        local_start = start.astimezone()
        return (
            f"Added '{title}' to the Noctalia calendar for "
            f"{local_start.strftime('%A, %B %-d at %-I:%M %p')}; alarm {alarm}. Event ID {event_id}"
        )

    def list_events(self) -> str:
        if not self.collection.is_dir():
            return "There are no Adam events in the Noctalia calendar."
        events = []
        for path in sorted(self.collection.glob("*.ics")):
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            content = re.sub(r"\r?\n[ \t]", "", content)
            event_id = re.search(r"(?m)^X-ADAM-ID:(.+)$", content)
            title = re.search(r"(?m)^SUMMARY:(.+)$", content)
            start = re.search(r"(?m)^DTSTART:(\d{8}T\d{6})$", content)
            if not (event_id and title and start):
                continue
            try:
                when = datetime.strptime(start.group(1), "%Y%m%dT%H%M%S").replace(
                    tzinfo=datetime.now().astimezone().tzinfo
                )
            except ValueError:
                continue
            if when >= datetime.now().astimezone():
                display_title = (title.group(1).replace("\\n", " ").replace("\\,", ",")
                                 .replace("\\;", ";").replace("\\\\", "\\"))
                events.append((when, display_title, event_id.group(1)))
        if not events:
            return "There are no upcoming Adam events in the Noctalia calendar."
        events.sort()
        return "Upcoming Noctalia events: " + "; ".join(
            f"{when.strftime('%A, %B %-d at %-I:%M %p')}: {title} (ID {event_id})"
            for when, title, event_id in events[:20]
        )

    def cancel_event(self, event_id: str) -> str:
        target = (event_id or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{32}", target):
            return "Please provide the event ID returned when it was created."
        path = self.collection / f"{target}.ics"
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"No Adam calendar event found with ID {target}."
        except OSError as exc:
            return f"Could not read that calendar event: {exc}"
        if f"X-ADAM-ID:{target}" not in content:
            return "That file is not a Adam-created calendar event; it was left untouched."
        path.unlink()
        return f"Removed event {target} from the Noctalia calendar."
