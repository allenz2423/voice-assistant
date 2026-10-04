from pathlib import Path
from unittest.mock import patch

from src.tools import desktop


def _steam_library(home: Path) -> Path:
    root = home / ".local/share/Steam"
    (root / "steamapps").mkdir(parents=True)
    return root


def test_lists_steam_titles_from_default_and_added_library(tmp_path, monkeypatch):
    home = tmp_path / "home"
    main = _steam_library(home)
    added = tmp_path / "games/SteamLibrary"
    (added / "steamapps").mkdir(parents=True)
    (main / "steamapps/libraryfolders.vdf").write_text(
        '"libraryfolders" { "0" { "path" "' + str(main) + '" } '
        '"1" { "path" "' + str(added) + '" } }',
        encoding="utf-8",
    )
    (main / "steamapps/appmanifest_101.acf").write_text(
        '"AppState" { "appid" "101" "name" "Example App" }', encoding="utf-8"
    )
    (added / "steamapps/appmanifest_202.acf").write_text(
        '"AppState" { "appid" "202" "name" "Another Example" }', encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", lambda: home)

    apps = desktop._scan_desktop_entries()

    assert apps["example app"]["steam_app_id"] == "101"
    assert apps["another example"]["steam_app_id"] == "202"
    assert "another example" in desktop.list_applications("another example").lower()


def test_launches_steam_title_with_app_id_uri(tmp_path, monkeypatch):
    home = tmp_path / "home"
    steam = _steam_library(home)
    (steam / "steamapps/appmanifest_504210.acf").write_text(
        '"AppState" { "appid" "504210" "name" "SHENZHEN I/O" }', encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(desktop, "_scan_desktop_entries", lambda: desktop._scan_steam_entries(home))
    monkeypatch.setattr(desktop, "ensure_gui_environment", lambda: None)
    monkeypatch.setattr(desktop.shutil, "which", lambda name: "/usr/bin/steam" if name == "steam" else None)
    with patch.object(desktop.subprocess, "Popen") as popen:
        result = desktop.launch_application("SHENZHEN I/O")

    assert result == "Launched SHENZHEN I/O through Steam."
    popen.assert_called_once()
    assert popen.call_args.args[0] == ["steam", "steam://rungameid/504210"]


def test_rejects_custom_args_for_steam_title(tmp_path, monkeypatch):
    home = tmp_path / "home"
    steam = _steam_library(home)
    (steam / "steamapps/appmanifest_504210.acf").write_text(
        '"AppState" { "appid" "504210" "name" "SHENZHEN I/O" }', encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(desktop, "_scan_desktop_entries", lambda: desktop._scan_steam_entries(home))
    monkeypatch.setattr(desktop, "ensure_gui_environment", lambda: None)

    assert "Custom launch arguments" in desktop.launch_application("SHENZHEN I/O", "--unsafe")
