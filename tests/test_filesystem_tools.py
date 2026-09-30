import pytest

from src.tools.filesystem import create_file, read_file, write_file


def test_create_file_and_readback(tmp_path):
    path = tmp_path / "note.txt"
    assert "Created" in create_file(str(path), "first line\n")
    result = read_file(str(path))
    assert "first line" in result
    assert path.read_text() == "first line\n"


def test_create_file_refuses_to_overwrite_existing_file(tmp_path):
    path = tmp_path / "existing.txt"
    path.write_text("keep this")
    with pytest.raises(FileExistsError):
        create_file(str(path), "replace")
    assert path.read_text() == "keep this"


def test_write_file_requires_explicit_overwrite_and_replaces_atomically(tmp_path):
    path = tmp_path / "existing.txt"
    path.write_text("old")
    with pytest.raises(FileExistsError):
        write_file(str(path), "new")
    assert path.read_text() == "old"

    assert "Updated" in write_file(str(path), "new", overwrite=True)
    assert read_file(str(path)).endswith("new")
