from pathlib import Path
import subprocess

import yaml

from tools.setup_wizard import NEMOTRON_DIARIZATION_MODEL, choose_nvidia_device
from src.config import SpeakerDiarizationConfig, STTConfig


def test_setup_uses_official_nemotron_transformers_model():
    assert NEMOTRON_DIARIZATION_MODEL == "nvidia/Nemotron-3-Diarization"


def test_legacy_gguf_config_migrates_to_official_model():
    config = SpeakerDiarizationConfig(model="assets/models/Nemotron-3-Diarization.q8_0.gguf", device="vulkan:0")

    assert config.model == NEMOTRON_DIARIZATION_MODEL
    assert config.device == "auto"


def test_setup_can_select_exact_cuda_device(monkeypatch):
    import tools.setup_wizard as wizard

    monkeypatch.setattr(wizard, "prompt_choice", lambda *_args: 1)
    assert choose_nvidia_device(
        "Select GPU", [("0", "GTX 1080 Ti"), ("1", "RTX 3070")], "0"
    ) == "1"


def test_qwen_defaults_to_automatic_language_detection():
    assert STTConfig().language == ""


def test_example_config_uses_portable_cpu_and_managed_browser_defaults():
    example_path = Path(__file__).resolve().parents[1] / "config.yaml.example"
    config = yaml.safe_load(example_path.read_text(encoding="utf-8"))

    assert config["stt"]["model_size"] == "small.en"
    assert config["stt"]["device"] == "cpu"
    assert config["stt"]["device_index"] == -1
    assert config["stt"]["compute_type"] == "int8"
    assert config["tts"]["device_id"] == -1
    assert config["browser_navigation"]["browser"] == "chromium"
    assert config["browser_navigation"]["profile_path"] == "~/.local/share/adam/browser-navigation"
    assert config["desktop"]["default_browser"] == "chromium"


def _write_wizard_config(project: Path, content: str) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / "config.yaml").write_text(content, encoding="utf-8")


def test_browser_navigation_honors_flag_default_and_preserves_isolated_profile(tmp_path, monkeypatch):
    import tools.setup_wizard as wizard

    project = tmp_path / "adam"
    _write_wizard_config(
        project,
        "browser_navigation:\n"
        "  enabled: false\n"
        "  browser: chromium\n"
        "  profile_path: ~/.local/share/adam/browser-navigation\n"
        "desktop:\n"
        "  default_browser: chromium\n",
    )
    monkeypatch.setattr(wizard, "PROJECT_DIR", project)
    monkeypatch.setenv("ADAM_BROWSER_NAVIGATION_DEFAULT", "1")
    monkeypatch.delenv("ADAM_SKIP_PYTHON", raising=False)
    monkeypatch.setattr(wizard.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(wizard.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(wizard.subprocess, "run", run)
    prompt_defaults = []
    monkeypatch.setattr(
        wizard,
        "prompt_yes_no",
        lambda _prompt, default_yes=True: prompt_defaults.append(default_yes) or True,
    )

    wizard.configure_browser_navigation()

    assert prompt_defaults == [True]
    assert wizard.get_current_config_value("enabled", section="browser_navigation") == "true"
    assert wizard.get_current_config_value("profile_path", section="browser_navigation") == "~/.local/share/adam/browser-navigation"
    assert calls == [["/usr/bin/uv", "run", "--no-sync", "playwright", "install", "chromium"]]


def test_browser_navigation_defers_when_python_dependencies_are_skipped(tmp_path, monkeypatch, capsys):
    import tools.setup_wizard as wizard

    project = tmp_path / "adam"
    _write_wizard_config(
        project,
        "browser_navigation:\n  enabled: false\n  browser: chromium\n"
        "  profile_path: ~/.local/share/adam/browser-navigation\n"
        "desktop:\n  default_browser: chromium\n",
    )
    monkeypatch.setattr(wizard, "PROJECT_DIR", project)
    monkeypatch.setenv("ADAM_SKIP_PYTHON", "1")
    monkeypatch.delenv("ADAM_BROWSER_NAVIGATION_DEFAULT", raising=False)
    monkeypatch.setattr(wizard.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(wizard, "prompt_yes_no", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        wizard.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must defer, not install")),
    )

    wizard.configure_browser_navigation()

    assert wizard.get_current_config_value("enabled", section="browser_navigation") == "false"
    assert "--skip-python-deps prevents installing it" in capsys.readouterr().out


def test_browser_navigation_uses_existing_enabled_setting_as_prompt_default(tmp_path, monkeypatch):
    import tools.setup_wizard as wizard

    project = tmp_path / "adam"
    _write_wizard_config(
        project,
        "browser_navigation:\n  enabled: true\n  browser: chromium\n"
        "  profile_path: ~/.local/share/adam/browser-navigation\n",
    )
    monkeypatch.setattr(wizard, "PROJECT_DIR", project)
    monkeypatch.delenv("ADAM_BROWSER_NAVIGATION_DEFAULT", raising=False)
    defaults = []
    monkeypatch.setattr(
        wizard,
        "prompt_yes_no",
        lambda _prompt, default_yes=True: defaults.append(default_yes) or False,
    )

    wizard.configure_browser_navigation()

    assert defaults == [True]
    assert wizard.get_current_config_value("enabled", section="browser_navigation") == "false"


def _systemd_fixture(project: Path, home: Path, *, tts_engine: str = "silent") -> None:
    _write_wizard_config(
        project,
        f"tts:\n  engine: {tts_engine}\n"
        "  model_path: assets/voices/kokoro/kokoro-v1.0.onnx\n"
        "  voices_path: assets/voices/kokoro/voices-v1.0.bin\n",
    )
    (project / "systemd").mkdir()
    (project / "systemd" / "adam.service.template").write_text(
        "[Service]\nWorkingDirectory={{PROJECT_DIR}}\n"
        "ExecStart={{PROJECT_DIR}}/.venv/bin/python -m src.main\n"
        "Environment=PATH={{PROJECT_DIR}}/.venv/bin:{{HOME}}/.local/bin:/usr/bin:/bin\n",
        encoding="utf-8",
    )
    home.mkdir(parents=True, exist_ok=True)


def _prepare_systemd_wizard(monkeypatch, wizard, project: Path, home: Path) -> None:
    monkeypatch.setattr(wizard, "PROJECT_DIR", project)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ADAM_SKIP_SERVICE", raising=False)
    monkeypatch.setattr(wizard, "systemd_user_manager_available", lambda: True)
    monkeypatch.setattr(wizard, "ensure_onnxruntime", lambda: True)
    monkeypatch.setattr(wizard, "ensure_torchaudio", lambda: True)


def test_systemd_requires_opt_in_before_inspecting_or_touching_legacy_service(tmp_path, monkeypatch):
    import tools.setup_wizard as wizard

    project, home = tmp_path / "adam", tmp_path / "home"
    _systemd_fixture(project, home)
    _prepare_systemd_wizard(monkeypatch, wizard, project, home)
    calls = []

    def run(*args):
        calls.append(args)
        if args == ("is-active", "adam-kev.service"):
            return subprocess.CompletedProcess(args, 0, "active", "")
        if args == ("is-enabled", "adam-kev.service"):
            return subprocess.CompletedProcess(args, 0, "enabled", "")
        raise AssertionError(f"unexpected systemctl call: {args}")

    monkeypatch.setattr(wizard, "_run_user_systemctl", run)
    prompt_defaults = []
    monkeypatch.setattr(
        wizard,
        "prompt_yes_no",
        lambda _prompt, default_yes=True: prompt_defaults.append(default_yes) or False,
    )

    wizard.configure_systemd()

    assert prompt_defaults == [False]
    assert calls == []
    assert not (home / ".config/systemd/user/adam.service").exists()


def test_systemd_renders_template_and_reports_confirmed_service_active(tmp_path, monkeypatch, capsys):
    import tools.setup_wizard as wizard

    project, home = tmp_path / "adam", tmp_path / "home"
    _systemd_fixture(project, home)
    _prepare_systemd_wizard(monkeypatch, wizard, project, home)
    commands = []

    def run(*args):
        commands.append(args)
        if args in (("is-active", "adam-kev.service"), ("is-enabled", "adam-kev.service")):
            return subprocess.CompletedProcess(args, 3, "", "inactive")
        return subprocess.CompletedProcess(args, 0, "active", "")

    monkeypatch.setattr(wizard, "_run_user_systemctl", run)
    monkeypatch.setattr(wizard, "prompt_yes_no", lambda *_args, **_kwargs: True)

    wizard.configure_systemd()

    service_path = home / ".config/systemd/user/adam.service"
    service = service_path.read_text(encoding="utf-8")
    assert str(project) in service
    assert str(home) in service
    assert "{{PROJECT_DIR}}" not in service and "{{HOME}}" not in service
    assert service_path.stat().st_mode & 0o777 == 0o644
    assert commands[-3:] == [
        ("daemon-reload",),
        ("enable", "--now", "adam.service"),
        ("is-active", "adam.service"),
    ]
    assert "adam.service is active." in capsys.readouterr().out


def test_systemd_refuses_to_start_when_kokoro_assets_are_missing(tmp_path, monkeypatch, capsys):
    import tools.setup_wizard as wizard

    project, home = tmp_path / "adam", tmp_path / "home"
    _systemd_fixture(project, home, tts_engine="kokoro")
    _prepare_systemd_wizard(monkeypatch, wizard, project, home)
    commands = []

    def run(*args):
        commands.append(args)
        if args in (("is-active", "adam-kev.service"), ("is-enabled", "adam-kev.service")):
            return subprocess.CompletedProcess(args, 0, "active", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(wizard, "_run_user_systemctl", run)
    prompts = []
    monkeypatch.setattr(
        wizard,
        "prompt_yes_no",
        lambda _prompt, default_yes=True: prompts.append(default_yes) or True,
    )

    wizard.configure_systemd()

    assert prompts == [False]
    assert commands == []
    assert not (home / ".config/systemd/user/adam.service").exists()
    assert "Kokoro assets are missing" in capsys.readouterr().out


def test_systemd_runtime_preflight_failure_does_not_inspect_or_stop_legacy_service(
    tmp_path, monkeypatch
):
    import tools.setup_wizard as wizard

    project, home = tmp_path / "adam", tmp_path / "home"
    _systemd_fixture(project, home)
    _prepare_systemd_wizard(monkeypatch, wizard, project, home)
    runtime_checks = []
    monkeypatch.setattr(
        wizard,
        "ensure_onnxruntime",
        lambda: runtime_checks.append("onnxruntime") or False,
    )
    monkeypatch.setattr(
        wizard,
        "ensure_torchaudio",
        lambda: runtime_checks.append("torchaudio") or True,
    )
    commands = []

    def run(*args):
        commands.append(args)
        if args in (("is-active", "adam-kev.service"), ("is-enabled", "adam-kev.service")):
            return subprocess.CompletedProcess(args, 0, "active", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(wizard, "_run_user_systemctl", run)
    prompts = []
    monkeypatch.setattr(
        wizard,
        "prompt_yes_no",
        lambda _prompt, default_yes=True: prompts.append(default_yes) or True,
    )

    wizard.configure_systemd()

    assert runtime_checks == ["onnxruntime"]
    assert prompts == [False]
    assert commands == []
    assert not (home / ".config/systemd/user/adam.service").exists()


def test_systemd_replacement_keeps_backup_and_checks_command_failures(tmp_path, monkeypatch, capsys):
    import tools.setup_wizard as wizard

    project, home = tmp_path / "adam", tmp_path / "home"
    _systemd_fixture(project, home)
    _prepare_systemd_wizard(monkeypatch, wizard, project, home)
    service_dir = home / ".config/systemd/user"
    service_dir.mkdir(parents=True)
    service_path = service_dir / "adam.service"
    service_path.write_text("user-managed unit\n", encoding="utf-8")
    monkeypatch.setattr(wizard, "prompt_yes_no", lambda *_args, **_kwargs: True)
    commands = []

    def run(*args):
        commands.append(args)
        if args in (("is-active", "adam-kev.service"), ("is-enabled", "adam-kev.service")):
            return subprocess.CompletedProcess(args, 3, "", "inactive")
        if args == ("enable", "--now", "adam.service"):
            return subprocess.CompletedProcess(args, 1, "", "start failed")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(wizard, "_run_user_systemctl", run)

    wizard.configure_systemd()

    assert service_path.read_text(encoding="utf-8").startswith("[Service]")
    assert (service_dir / "adam.service.backup").read_text(encoding="utf-8") == "user-managed unit\n"
    output = capsys.readouterr().out
    assert "Could not enable and start adam.service (start failed)" in output
    assert "adam.service is active" not in output


def test_systemd_keeps_existing_unit_when_replacement_is_declined(tmp_path, monkeypatch, capsys):
    import tools.setup_wizard as wizard

    project, home = tmp_path / "adam", tmp_path / "home"
    _systemd_fixture(project, home)
    _prepare_systemd_wizard(monkeypatch, wizard, project, home)
    service_dir = home / ".config/systemd/user"
    service_dir.mkdir(parents=True)
    service_path = service_dir / "adam.service"
    service_path.write_text("user-managed unit\n", encoding="utf-8")
    decisions = iter((True, False))  # Start Adam; retain the existing unit.
    monkeypatch.setattr(wizard, "prompt_yes_no", lambda *_args, **_kwargs: next(decisions))
    commands = []

    def run(*args):
        commands.append(args)
        if args in (("is-active", "adam-kev.service"), ("is-enabled", "adam-kev.service")):
            return subprocess.CompletedProcess(args, 3, "", "inactive")
        return subprocess.CompletedProcess(args, 0, "active", "")

    monkeypatch.setattr(wizard, "_run_user_systemctl", run)

    wizard.configure_systemd()

    assert service_path.read_text(encoding="utf-8") == "user-managed unit\n"
    assert not (service_dir / "adam.service.backup").exists()
    assert ("daemon-reload",) not in commands
    assert ("enable", "--now", "adam.service") in commands
    assert "Keeping the existing adam.service unchanged" in capsys.readouterr().out


def test_systemd_leaves_legacy_service_unchanged_when_user_declines(tmp_path, monkeypatch, capsys):
    import tools.setup_wizard as wizard

    project, home = tmp_path / "adam", tmp_path / "home"
    _systemd_fixture(project, home)
    _prepare_systemd_wizard(monkeypatch, wizard, project, home)
    commands = []

    def run(*args):
        commands.append(args)
        if args in (("is-active", "adam-kev.service"), ("is-enabled", "adam-kev.service")):
            return subprocess.CompletedProcess(args, 0, "active", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(wizard, "_run_user_systemctl", run)
    decisions = iter((True, False))  # Install Adam; keep the legacy service untouched.
    monkeypatch.setattr(wizard, "prompt_yes_no", lambda *_args, **_kwargs: next(decisions))

    wizard.configure_systemd()

    assert ("disable", "--now", "adam-kev.service") not in commands
    assert ("enable", "--now", "adam.service") in commands
    assert "Leaving adam-kev.service unchanged" in capsys.readouterr().out


def test_systemd_skips_unit_when_project_path_cannot_be_represented(tmp_path, monkeypatch, capsys):
    import tools.setup_wizard as wizard

    project, home = tmp_path / "adam project", tmp_path / "home"
    _systemd_fixture(project, home)
    _prepare_systemd_wizard(monkeypatch, wizard, project, home)
    commands = []

    def run(*args):
        commands.append(args)
        if args in (("is-active", "adam-kev.service"), ("is-enabled", "adam-kev.service")):
            return subprocess.CompletedProcess(args, 0, "active", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(wizard, "_run_user_systemctl", run)
    prompts = []
    monkeypatch.setattr(
        wizard,
        "prompt_yes_no",
        lambda _prompt, default_yes=True: prompts.append(default_yes) or True,
    )

    wizard.configure_systemd()

    assert not (home / ".config/systemd/user/adam.service").exists()
    assert prompts == [False]
    assert commands == []
    assert "cannot safely represent" in capsys.readouterr().out


def test_setup_script_help():
    setup_path = Path(__file__).resolve().parents[1] / "setup.sh"
    res = subprocess.run([str(setup_path), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Adam setup" in res.stdout
    assert "--dry-run" in res.stdout
    assert "--idea-routing" in res.stdout
    assert "--browser-navigation" in res.stdout
    assert "--diarization" in res.stdout
    assert "--omniparser" in res.stdout


def test_setup_script_dry_run_default():
    setup_path = Path(__file__).resolve().parents[1] / "setup.sh"
    res = subprocess.run([str(setup_path), "--dry-run"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Dry run: no commands will be executed and no files will be changed." in res.stdout
    assert "runtime-cpu" in res.stdout
    assert "speaker-verification" in res.stdout


def test_setup_script_dry_run_flags():
    setup_path = Path(__file__).resolve().parents[1] / "setup.sh"
    res = subprocess.run(
        [
            str(setup_path),
            "--dry-run",
            "--nvidia-runtime",
            "--idea-routing",
            "--browser-navigation",
            "--diarization",
            "--install-desktop-tools",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "runtime-nvidia" in res.stdout
    assert "intent-routing" in res.stdout
    assert "browser-control" in res.stdout
    assert "nemotron-diarization" in res.stdout
    assert "(required + optional desktop)" in res.stdout


def test_setup_script_mutual_exclusions():
    setup_path = Path(__file__).resolve().parents[1] / "setup.sh"
    # cpu-only and nvidia-runtime
    res = subprocess.run([str(setup_path), "--cpu-only", "--nvidia-runtime"], capture_output=True, text=True)
    assert res.returncode == 2
    assert "--cpu-only and --nvidia-runtime cannot be used together" in res.stderr

    # disable-computer-control and install-desktop-tools
    res = subprocess.run([str(setup_path), "--disable-computer-control", "--install-desktop-tools"], capture_output=True, text=True)
    assert res.returncode == 2
    assert "--disable-computer-control and --install-desktop-tools cannot be used together" in res.stderr

    # disable-computer-control and omniparser
    res = subprocess.run([str(setup_path), "--disable-computer-control", "--omniparser"], capture_output=True, text=True)
    assert res.returncode == 2
    assert "--disable-computer-control and --omniparser cannot be used together" in res.stderr
