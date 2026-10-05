"""Small, local date parsing helpers for dated episodic memories."""

from __future__ import annotations

import calendar
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo, timezone as datetime_timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class EventTime:
    event_type: str
    date_start: str
    date_end: str
    start_at: str | None
    end_at: str | None
    original_expression: str
    timezone: str
    precision: str


_MONTHS_AGO = re.compile(r"\b(?P<n>\d{1,3})\s+months?\s+ago\b", re.I)
_YEARS_AGO = re.compile(r"\b(?P<n>\d{1,3})\s+years?\s+ago\b", re.I)
_WEEKS_AGO = re.compile(r"\b(?P<n>\d{1,3})\s+weeks?\s+ago\b", re.I)
_DAYS_AGO = re.compile(r"\b(?P<n>\d{1,4})\s+days?\s+ago\b", re.I)
_ABSOLUTE_DATE = re.compile(r"\b(?P<year>19\d{2}|20\d{2}|21\d{2})-(?P<month>\d{2})-(?P<day>\d{2})\b")

_TIME_RANGE = re.compile(
    r"\b(?:from\s+)?"
    r"(?P<start>\d{1,2}(?:[:.]\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)"
    r"\s+(?:to|until|through|[-–])\s+"
    r"(?P<end>\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)\b",
    re.I,
)
_TIME = re.compile(r"^(?P<hour>\d{1,2})(?:[:.](?P<minute>\d{2}))?\s*(?P<period>a\.?m\.?|p\.?m\.?)?$", re.I)


def _zone_info(name: str | None) -> ZoneInfo | None:
    if not name:
        return None
    try:
        return ZoneInfo(str(name))
    except (ZoneInfoNotFoundError, ValueError):
        return None


def local_timezone_name() -> str | None:
    """Return the system's IANA timezone key when the OS exposes one."""
    configured = str(os.environ.get("TZ", "")).strip().lstrip(":")
    if _zone_info(configured) is not None:
        return configured

    try:
        resolved = Path("/etc/localtime").resolve(strict=True)
        parts = resolved.parts
        marker = parts.index("zoneinfo")
        key = "/".join(parts[marker + 1:])
        if _zone_info(key) is not None:
            return key
    except (OSError, ValueError):
        pass

    try:
        key = Path("/etc/timezone").read_text(encoding="utf-8").strip()
        if _zone_info(key) is not None:
            return key
    except OSError:
        pass
    return None


def _parse_anchor(recorded_at: str | datetime, timezone_name: str | None = None) -> datetime:
    if isinstance(recorded_at, datetime):
        anchor = recorded_at
    else:
        value = str(recorded_at).strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        anchor = datetime.fromisoformat(value)
    timezone = _zone_info(timezone_name)
    if timezone is not None:
        return anchor.replace(tzinfo=timezone) if anchor.tzinfo is None else anchor.astimezone(timezone)
    if anchor.tzinfo is None:
        anchor = anchor.astimezone()
    return anchor


def _local_datetime(day: date, clock: time, timezone: tzinfo) -> datetime | None:
    """Resolve a wall clock only when its local-time instant is unambiguous."""
    wall_time = datetime.combine(day, clock)
    if not isinstance(timezone, ZoneInfo):
        return wall_time.replace(tzinfo=timezone)

    candidates: list[datetime] = []
    for fold in (0, 1):
        candidate = wall_time.replace(tzinfo=timezone, fold=fold)
        round_trip = candidate.astimezone(datetime_timezone.utc).astimezone(timezone)
        if round_trip.replace(tzinfo=None) == wall_time:
            candidates.append(candidate)

    if not candidates:
        # The clock time falls in a spring-forward gap.
        return None
    offsets = {candidate.utcoffset() for candidate in candidates}
    if len(offsets) > 1:
        # Fall-back repeats this local clock time; there is no evidence for which
        # occurrence the user meant, so keep the phrase but do not invent an offset.
        return None
    return candidates[0]


def _month_start(year: int, month: int) -> date:
    return date(year, month, 1)


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _resolve_date_expression(text: str, anchor: datetime) -> tuple[date, date, str, str] | None:
    day = anchor.date()

    absolute = _ABSOLUTE_DATE.search(text)
    if absolute:
        try:
            exact = date(*(int(absolute.group(key)) for key in ("year", "month", "day")))
            return exact, exact, absolute.group(0), "day"
        except ValueError:
            pass

    if re.search(r"\btoday\b", text, re.I):
        return day, day, "today", "day"
    if re.search(r"\bday before yesterday\b", text, re.I):
        value = day - timedelta(days=2)
        return value, value, "day before yesterday", "day"
    if re.search(r"\byesterday\b", text, re.I):
        value = day - timedelta(days=1)
        return value, value, "yesterday", "day"
    if re.search(r"\bthis week\b", text, re.I):
        start = day - timedelta(days=day.weekday())
        return start, start + timedelta(days=6), "this week", "week"
    if re.search(r"\blast week\b", text, re.I):
        end = day - timedelta(days=day.weekday() + 1)
        return end - timedelta(days=6), end, "last week", "week"
    if re.search(r"\bthis month\b", text, re.I):
        return _month_start(day.year, day.month), _month_end(day.year, day.month), "this month", "month"
    if re.search(r"\blast month\b", text, re.I):
        previous = _month_start(day.year, day.month) - timedelta(days=1)
        return _month_start(previous.year, previous.month), previous, "last month", "month"
    if re.search(r"\bthis year\b", text, re.I):
        return date(day.year, 1, 1), date(day.year, 12, 31), "this year", "year"
    if re.search(r"\blast year\b", text, re.I):
        year = day.year - 1
        return date(year, 1, 1), date(year, 12, 31), "last year", "year"

    for pattern, unit in ((_DAYS_AGO, "day"), (_WEEKS_AGO, "week"), (_MONTHS_AGO, "month"), (_YEARS_AGO, "year")):
        match = pattern.search(text)
        if not match:
            continue
        amount = int(match.group("n"))
        if amount < 1:
            continue
        expression = match.group(0)
        if unit == "day":
            value = day - timedelta(days=amount)
            return value, value, expression, "day_relative"
        if unit == "week":
            target = day - timedelta(weeks=amount)
            start = target - timedelta(days=target.weekday())
            return start, start + timedelta(days=6), expression, "week_relative"
        if unit == "month":
            month_index = day.year * 12 + day.month - 1 - amount
            year, month0 = divmod(month_index, 12)
            month = month0 + 1
            return _month_start(year, month), _month_end(year, month), expression, "month_relative"
        year = day.year - amount
        return date(year, 1, 1), date(year, 12, 31), expression, "year_relative"

    return None


def _parse_clock(value: str, inherited_period: str | None = None) -> tuple[time, str | None] | None:
    match = _TIME.fullmatch(value.strip())
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    period = re.sub(r"\W", "", match.group("period") or inherited_period or "").lower()
    if minute > 59 or hour < 1 or hour > 12:
        return None
    if period:
        if period.startswith("p") and hour < 12:
            hour += 12
        elif period.startswith("a") and hour == 12:
            hour = 0
    elif hour > 12:
        return None
    return time(hour, minute), period or None


def _time_range(text: str) -> tuple[time, time, str] | None:
    match = _TIME_RANGE.search(text)
    if not match:
        return None
    start_text, end_text = match.group("start"), match.group("end")
    start_period = re.search(r"([ap])\.?m\.?$", start_text.strip(), re.I)
    end_period = re.search(r"([ap])\.?m\.?$", end_text.strip(), re.I)
    inherited_start = end_period.group(1) if end_period else None
    inherited_end = start_period.group(1) if start_period else None
    start_value = _parse_clock(start_text, inherited_start)
    end_value = _parse_clock(end_text, inherited_end)
    if start_value is None or end_value is None:
        return None
    start_time, start_mark = start_value
    end_time, end_mark = end_value
    # If neither endpoint names AM/PM, the original clock values are ambiguous.
    if start_mark is None and end_mark is None:
        return None
    return start_time, end_time, match.group(0)


def parse_event_time(
    text: str,
    recorded_at: str | datetime,
    timezone_name: str | None = None,
) -> EventTime | None:
    """Resolve explicit relative or absolute event timing against its record time."""
    clean_text = str(text or "")
    anchor = _parse_anchor(recorded_at, timezone_name)
    resolved = _resolve_date_expression(clean_text, anchor)
    if resolved is None:
        return None
    date_start, date_end, date_expression, precision = resolved
    tzinfo = anchor.tzinfo
    tz_name = (
        timezone_name if _zone_info(timezone_name) is not None
        else getattr(tzinfo, "key", None) or anchor.tzname() or anchor.strftime("%z")
    )
    event_type = "work" if re.search(r"\b(?:work(?:ed|ing)?|shift|clocked\s+(?:in|out))\b", clean_text, re.I) else "event"
    clock_range = _time_range(clean_text)
    start_at = end_at = None
    expression = date_expression
    if clock_range and date_start == date_end:
        start_time, end_time, clock_expression = clock_range
        end_date = date_start + timedelta(days=1) if end_time <= start_time else date_start
        start_dt = _local_datetime(date_start, start_time, tzinfo)
        end_dt = _local_datetime(end_date, end_time, tzinfo)
        if start_dt is not None and end_dt is not None:
            start_at, end_at = start_dt.isoformat(), end_dt.isoformat()
        if end_date > date_end:
            date_end = end_date
        expression = f"{date_expression}; {clock_expression}"

    return EventTime(
        event_type=event_type,
        date_start=date_start.isoformat(),
        date_end=date_end.isoformat(),
        start_at=start_at,
        end_at=end_at,
        original_expression=expression,
        timezone=str(tz_name),
        precision=precision,
    )


def parse_temporal_query_range(query: str, reference_at: datetime | None = None) -> tuple[str, str] | None:
    """Return a local date window for common date-bounded memory questions."""
    anchor = reference_at or datetime.now().astimezone()
    resolved = _resolve_date_expression(str(query or ""), anchor)
    if resolved is None:
        return None
    return resolved[0].isoformat(), resolved[1].isoformat()


def event_type_for_query(query: str) -> str | None:
    if re.search(r"\b(?:work|worked|working|shift|clock(?:ed)?\s+(?:in|out)|hours)\b", str(query or ""), re.I):
        return "work"
    return None
