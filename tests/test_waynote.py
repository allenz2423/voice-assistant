import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.tools.waynote import (
    generate_ulid,
    create_waynote,
    append_waynote,
    list_waynotes,
    manage_waynote,
    CROCKFORD_ALPHABET,
    VALID_COLORS,
)


def test_generate_ulid_format():
    ulid = generate_ulid()
    assert len(ulid) == 26
    for char in ulid:
        assert char in CROCKFORD_ALPHABET
        assert char not in {"I", "L", "O", "U"}


def test_create_waynote(tmp_path, monkeypatch):
    notes_dir = tmp_path / "notes"
    monkeypatch.setattr("src.tools.waynote.get_notes_dir", lambda: notes_dir)
    monkeypatch.setattr("src.tools.waynote._ensure_waynote_running", lambda: None)

    res = create_waynote(
        content="- Item 1\n- Item 2",
        title="Weekly Shopping",
        color="green",
    )

    assert "Created green" in res
    assert "Weekly Shopping" in res

    files = list(notes_dir.glob("*.md"))
    assert len(files) == 1
    note_file = files[0]
    assert note_file.name.endswith("-weekly-shopping.md")

    text = note_file.read_text(encoding="utf-8")
    assert "color: green" in text
    assert "# Weekly Shopping" in text
    assert "- Item 1" in text


def test_create_waynote_invalid_color_fallback(tmp_path, monkeypatch):
    notes_dir = tmp_path / "notes"
    monkeypatch.setattr("src.tools.waynote.get_notes_dir", lambda: notes_dir)
    monkeypatch.setattr("src.tools.waynote._ensure_waynote_running", lambda: None)

    res = create_waynote(content="Test content", color="neon-rainbow")
    assert "Created yellow" in res

    files = list(notes_dir.glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "color: yellow" in text


def test_append_waynote_existing(tmp_path, monkeypatch):
    notes_dir = tmp_path / "notes"
    monkeypatch.setattr("src.tools.waynote.get_notes_dir", lambda: notes_dir)
    monkeypatch.setattr("src.tools.waynote._ensure_waynote_running", lambda: None)

    create_waynote(content="- Buy milk", title="Groceries", color="yellow")

    res = append_waynote("Buy eggs", target="Groceries")
    assert "Appended to sticky note" in res
    assert "Buy eggs" in res

    files = list(notes_dir.glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "- Buy milk" in text
    assert "- Buy eggs" in text


def test_append_waynote_creates_if_empty(tmp_path, monkeypatch):
    notes_dir = tmp_path / "notes"
    monkeypatch.setattr("src.tools.waynote.get_notes_dir", lambda: notes_dir)
    monkeypatch.setattr("src.tools.waynote._ensure_waynote_running", lambda: None)

    res = append_waynote("Buy bread", target="Shopping")
    assert "Created" in res

    files = list(notes_dir.glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "Buy bread" in text


def test_list_waynotes(tmp_path, monkeypatch):
    notes_dir = tmp_path / "notes"
    monkeypatch.setattr("src.tools.waynote.get_notes_dir", lambda: notes_dir)
    monkeypatch.setattr("src.tools.waynote._ensure_waynote_running", lambda: None)

    assert list_waynotes() == "No sticky notes found."

    create_waynote(content="Email John about meeting", title="Todos", color="blue")
    create_waynote(content="Walk the dog", title="Personal", color="pink")

    listed = list_waynotes()
    assert "Todos" in listed
    assert "Personal" in listed
    assert "Email John" in listed
    assert "Walk the dog" in listed


def test_manage_waynote_invalid_action():
    res = manage_waynote("destroy-universe")
    assert "Invalid action" in res


def test_manage_waynote_dispatch(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/waynote")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res = manage_waynote("toggle")
        assert "executed" in res
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["/usr/bin/waynote", "toggle"]
