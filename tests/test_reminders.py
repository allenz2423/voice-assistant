from datetime import datetime, timedelta

import pytest

from src.tools.reminders import ReminderManager


pytestmark = pytest.mark.skipif(
    __import__("shutil").which("remind") is None,
    reason="Remind system package is required",
)


def _future_local_time(days=2):
    return (datetime.now().astimezone() + timedelta(days=days)).replace(tzinfo=None, hour=9, minute=15, second=0, microsecond=0)


def test_create_list_cancel_reminder_and_preserve_user_config(tmp_path):
    (tmp_path / "personal.rem").write_text(
        "REM 1 Jan 2099 MSG User-owned event\n", encoding="utf-8"
    )
    manager = ReminderManager(tmp_path)
    result = manager.create_reminder(
        "Pick up groceries", _future_local_time().isoformat(timespec="minutes")
    )
    assert "ID " in result
    reminder_id = result.rsplit("ID ", 1)[1].rstrip(" ).")

    listing = manager.list_reminders()
    assert "Pick up groceries" in listing
    assert "User-owned event" in listing

    assert "Cancelled" in manager.cancel_reminder(reminder_id)
    assert "Pick up groceries" not in manager.list_reminders()
    assert "User-owned event" in manager.list_reminders()


@pytest.mark.parametrize("repeat", ["daily", "weekly", "monthly", "yearly"])
def test_create_recurring_reminder(tmp_path, repeat):
    manager = ReminderManager(tmp_path)
    result = manager.create_reminder(
        "Recurring task", _future_local_time().isoformat(timespec="minutes"), repeat
    )
    assert "Reminder set" in result
    assert "Recurring task" in manager.list_reminders()


def test_invalid_reminder_does_not_write(tmp_path):
    manager = ReminderManager(tmp_path)
    result = manager.create_reminder("", "not a date")
    assert "message" in result.lower()
    assert not manager.adam_file.exists()
