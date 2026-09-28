from unittest.mock import Mock

from src.tools import noctalia


def test_opens_noctalia_calendar_with_ipc(monkeypatch):
    run = Mock(return_value=Mock(returncode=0, stdout="ok", stderr=""))
    monkeypatch.setattr(noctalia.shutil, "which", lambda _: "/usr/bin/noctalia")
    monkeypatch.setattr(noctalia.subprocess, "run", run)

    assert noctalia.open_noctalia_calendar() == "Opened the Noctalia calendar."
    assert run.call_args.args[0] == [
        "/usr/bin/noctalia", "msg", "panel-open", "control-center", "calendar"
    ]


def test_reports_noctalia_ipc_failure(monkeypatch):
    monkeypatch.setattr(noctalia.shutil, "which", lambda _: "/usr/bin/noctalia")
    monkeypatch.setattr(
        noctalia.subprocess, "run",
        lambda *args, **kwargs: Mock(returncode=1, stdout="", stderr="No running instance"),
    )

    assert "No running instance" in noctalia.open_noctalia_calendar()
