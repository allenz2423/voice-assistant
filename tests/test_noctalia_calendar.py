from datetime import datetime, timedelta

from src.tools.noctalia_calendar import NoctaliaCalendar


def future_local_time():
    return (datetime.now().astimezone() + timedelta(days=2)).replace(
        hour=9, minute=15, second=0, microsecond=0
    )


def test_create_list_cancel_vdir_event(tmp_path):
    manager = NoctaliaCalendar(tmp_path / "calendars" / "shin")
    result = manager.create_event(
        "Pick up groceries, please",
        future_local_time().isoformat(timespec="minutes"),
        alarm_minutes=5,
    )
    event_id = result.rsplit("Event ID ", 1)[1]
    event_file = manager.collection / f"{event_id}.ics"
    content = event_file.read_text(encoding="utf-8")
    assert "BEGIN:VCALENDAR" in content
    assert "TRIGGER:-PT5M" in content
    assert "SUMMARY:Pick up groceries\\, please" in content
    assert "Pick up groceries, please" in manager.list_events()

    assert "Removed event" in manager.cancel_event(event_id)
    assert not event_file.exists()
    assert "no upcoming shin events" in manager.list_events().lower()


def test_recurring_event_and_at_start_alarm(tmp_path):
    manager = NoctaliaCalendar(tmp_path / "shin")
    result = manager.create_event(
        "Weekly check-in",
        future_local_time().isoformat(timespec="minutes"),
        alarm_minutes=0,
        repeat="weekly",
    )
    event_id = result.rsplit("Event ID ", 1)[1]
    content = (manager.collection / f"{event_id}.ics").read_text(encoding="utf-8")
    assert "RRULE:FREQ=WEEKLY" in content
    assert "TRIGGER:PT0S" in content


def test_cancel_rejects_unowned_or_invalid_id():
    assert "provide the event ID" in NoctaliaCalendar().cancel_event("personal-event")
