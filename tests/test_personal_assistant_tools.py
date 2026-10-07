import pytest
import asyncio
from src.tools.weather import get_weather_report
from src.tools.desktop import list_applications, launch_application, manage_clipboard
from src.tools.system_telemetry import get_system_status, list_processes, list_audio_devices, volume_control
from src.tools.media import get_now_playing
from src.tools.web import web_search
from src.tools.timers import TimerManager

@pytest.mark.asyncio
async def test_weather_report():
    report = await get_weather_report("")
    assert isinstance(report, str)
    assert len(report) > 10
    # Should report degrees or error message cleanly
    assert "degrees" in report.lower() or "unable" in report.lower()

def test_list_applications():
    res = list_applications()
    assert isinstance(res, str)
    assert "installed applications" in res.lower()

def test_list_applications_query():
    res = list_applications("alacritty")
    assert isinstance(res, str)
    # Alacritty is installed on this machine
    assert "alacritty" in res.lower() or "installed applications" in res.lower()

def test_get_system_status():
    status = get_system_status()
    assert isinstance(status, str)
    assert "cpu" in status.lower()
    assert "memory" in status.lower() or "gigabytes" in status.lower()

def test_get_system_status_includes_nvidia_gpu_utilization(monkeypatch):
    import subprocess
    from src.tools import system_telemetry

    monkeypatch.setattr(
        system_telemetry.shutil,
        "which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr(
        system_telemetry.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="0, NVIDIA RTX, 42, 7, 1024, 8192\n", stderr=""
        ),
    )

    status = system_telemetry.get_system_status()

    assert "utilization is 7 percent" in status
    assert "temperature is 42 degrees Celsius" in status

def test_get_system_status_does_not_report_rocm_gpu_on_driver_error(monkeypatch):
    import subprocess
    from src.tools import system_telemetry

    monkeypatch.setattr(
        system_telemetry.shutil,
        "which",
        lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None,
    )
    monkeypatch.setattr(
        system_telemetry.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="", stderr="ERROR: Driver not initialized"
        ),
    )

    status = system_telemetry.get_system_status()

    assert "AMD GPU active" not in status
    assert "GPU telemetry is unavailable" in status

def test_cpu_utilization_uses_idle_and_total_counter_deltas():
    from src.tools.system_telemetry import _cpu_utilization_percent

    assert _cpu_utilization_percent((1000, 600), (1100, 640)) == 60.0
    assert _cpu_utilization_percent((1000, 600), (1000, 600)) is None

def test_list_processes():
    procs = list_processes("cpu", limit=3)
    assert isinstance(procs, str)
    assert "top processes by cpu" in procs.lower()

def test_list_audio_devices():
    devs = list_audio_devices()
    assert isinstance(devs, str)
    assert "audio" in devs.lower()

def test_volume_control():
    from unittest.mock import patch, MagicMock
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Volume: 50%\n", stderr="")
        res = volume_control("up", 5)
        assert "volume" in res.lower()

def test_now_playing():
    res = get_now_playing()
    assert isinstance(res, str)
    assert len(res) > 0

def test_web_search():
    res = web_search("Linux")
    assert isinstance(res, str)
    assert len(res) > 20
    assert "linux" in res.lower() or "results" in res.lower()

def test_manage_clipboard():
    res = manage_clipboard("read")
    assert isinstance(res, str)
    assert len(res) > 0

def test_timer_manager():
    tm = TimerManager()
    res = tm.set_timer(60, "test timer")
    assert "timer set" in res.lower()

    listing = tm.list_timers()
    assert "test timer" in listing.lower()

    cancel = tm.cancel_timer("test timer")
    assert "cancelled" in cancel.lower()

    empty_listing = tm.list_timers()
    assert "no active timers" in empty_listing.lower()

def test_financial_quote():
    from src.tools.financial import get_financial_quote
    res = get_financial_quote("bitcoin")
    assert isinstance(res, str)
    assert "bitcoin" in res.lower() or "unable" in res.lower()

def test_calculate_math():
    from src.tools.math_calc import calculate_math
    assert "144" in calculate_math("12 * 12")
    assert "4" in calculate_math("sqrt(16)")
    assert "15" in calculate_math("10% of 150")

def test_check_system_updates():
    from src.tools.dev_sys import check_system_updates
    res = check_system_updates()
    assert isinstance(res, str)
    assert len(res) > 5

def test_manage_service():
    from src.tools.dev_sys import manage_service
    res = manage_service("status", "adam")
    assert isinstance(res, str)
    assert "service" in res.lower() or "adam" in res.lower()

def test_git_repo_status():
    from src.tools.dev_sys import git_repo_status
    res = git_repo_status()
    assert isinstance(res, str)
    assert "branch" in res.lower() or "repository" in res.lower()

def test_docker_container_status():
    from src.tools.dev_sys import docker_container_status
    res = docker_container_status()
    assert isinstance(res, str)
    assert len(res) > 5

def test_get_open_windows_prompt_context(monkeypatch):
    from types import SimpleNamespace
    from src.tools import desktop

    context = "Desktop State:\nWorkspace: 2 | Focused Window: Browser"
    environment_refreshes = []
    backend = SimpleNamespace(get_open_windows_prompt_context=lambda: context)
    monkeypatch.setattr(desktop, "ensure_gui_environment", lambda: environment_refreshes.append(True))
    monkeypatch.setattr(desktop, "get_active_backend", lambda: backend)

    state = desktop.get_open_windows_prompt_context()

    assert isinstance(state, str)
    assert state == context
    assert environment_refreshes == [True]

def test_resolve_application_entry_aliases():
    from src.tools.desktop import _resolve_application_entry, _scan_desktop_entries
    apps = _scan_desktop_entries()
    
    # Test discord alias -> vesktop
    discord_app = _resolve_application_entry("discord", apps)
    if discord_app:
        assert "vesktop" in discord_app["exec"].lower() or "discord" in discord_app["exec"].lower()

    # Test obs alias -> obs
    obs_app = _resolve_application_entry("obs", apps)
    if obs_app:
        assert "obs" in obs_app["exec"].lower()

    # Test edge alias -> edge
    edge_app = _resolve_application_entry("edge", apps)
    if edge_app:
        assert "edge" in edge_app["exec"].lower()

def test_close_application_active_pronoun():
    from unittest.mock import patch, MagicMock
    from src.tools.desktop import close_application
    # Empty or pronoun should target active without error and without touching real desktop windows
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        res = close_application("it")
        assert isinstance(res, str)
        assert len(res) > 0
