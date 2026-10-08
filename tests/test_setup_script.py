from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _setup_copy(project: Path) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    source = (REPO_ROOT / "setup.sh").read_text(encoding="utf-8")
    invocation = '\nmain "$@"\n'
    assert source.endswith(invocation)
    functions = project / "setup-functions.sh"
    functions.write_text(source[: -len(invocation)] + "\n", encoding="utf-8")
    shutil.copy2(REPO_ROOT / "setup.sh", project / "setup.sh")
    return functions


def _run_function(functions: Path, body: str, *, env: dict[str, str] | None = None):
    child_env = {**os.environ, **(env or {}), "SETUP_FUNCTIONS": str(functions)}
    return subprocess.run(
        ["/bin/bash", "-c", 'source "$SETUP_FUNCTIONS"; ' + body],
        capture_output=True,
        text=True,
        env=child_env,
        check=False,
    )


def test_help_and_dry_run_are_read_only(tmp_path):
    project = tmp_path / "adam"
    project.mkdir()
    setup = project / "setup.sh"
    shutil.copy2(REPO_ROOT / "setup.sh", setup)
    shutil.copy2(REPO_ROOT / "config.yaml.example", project / "config.yaml.example")
    home = tmp_path / "home"
    home.mkdir()
    before = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mode & 0o777)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    help_result = subprocess.run(
        ["/bin/bash", str(setup), "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    dry_run = subprocess.run(
        ["/bin/bash", str(setup), "--dry-run", "--nvidia-runtime", "--skip-service"],
        cwd=tmp_path,
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
    )

    after = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mode & 0o777)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert help_result.returncode == 0
    assert "Adam setup" in help_result.stdout
    assert dry_run.returncode == 0
    assert "Dry run: no commands will be executed and no files will be changed." in dry_run.stdout
    assert "runtime-nvidia" in dry_run.stdout
    assert "Service: skipped (--skip-service)" in dry_run.stdout
    assert after == before
    assert not (project / "config.yaml").exists()
    assert not (project / ".venv").exists()
    assert not (home / ".config/systemd/user/adam.service").exists()


def test_prepare_config_stops_if_owner_only_permissions_cannot_be_set(tmp_path):
    project = tmp_path / "adam"
    functions = _setup_copy(project)
    config = project / "config.yaml"
    config.write_text("llm:\n  api_key: synthetic-test-secret\n", encoding="utf-8")
    config.chmod(0o644)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    chmod = fake_bin / "chmod"
    chmod.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
    chmod.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin"}

    result = _run_function(functions, "prepare_config || exit $?", env=env)

    assert result.returncode != 0
    assert "Could not restrict config.yaml permissions" in result.stderr
    assert config.stat().st_mode & 0o777 == 0o644


def test_prepare_config_preserves_existing_symlink_target_and_secures_it(tmp_path):
    project = tmp_path / "adam"
    functions = _setup_copy(project)
    target = tmp_path / "private-config.yaml"
    original = "speaker_verification:\n  enabled: true\n  users: {}\n"
    target.write_text(original, encoding="utf-8")
    target.chmod(0o644)
    (project / "config.yaml").symlink_to(target)

    result = _run_function(
        functions,
        "SKIP_SPEAKER_VERIFICATION=true; prepare_config || exit $?",
    )

    assert result.returncode == 0, result.stderr
    assert (project / "config.yaml").is_symlink()
    assert "  enabled: false" in target.read_text(encoding="utf-8")
    assert "  users: {}" in target.read_text(encoding="utf-8")
    assert target.stat().st_mode & 0o777 == 0o600


def test_service_renderer_preserves_existing_unit_and_never_enables_it(tmp_path):
    project = tmp_path / "adam"
    functions = _setup_copy(project)
    template_dir = project / "systemd"
    template_dir.mkdir()
    shutil.copy2(REPO_ROOT / "systemd/adam.service.template", template_dir)
    home = tmp_path / "home"
    user_unit = home / ".config/systemd/user/adam.service"
    systemctl_log = tmp_path / "systemctl.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    systemctl = fake_bin / "systemctl"
    systemctl.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$SYSTEMCTL_LOG"\nexit 0\n',
        encoding="utf-8",
    )
    systemctl.chmod(0o755)
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "SYSTEMCTL_LOG": str(systemctl_log),
    }

    installed = _run_function(functions, "render_service_unit || exit $?", env=env)

    assert installed.returncode == 0, installed.stderr
    assert user_unit.is_file()
    assert str(project) in user_unit.read_text(encoding="utf-8")
    assert user_unit.stat().st_mode & 0o777 == 0o644
    calls = systemctl_log.read_text(encoding="utf-8").splitlines()
    assert calls == ["--user show-environment", "--user daemon-reload"]

    user_unit.write_text("user-managed unit\n", encoding="utf-8")
    preserved = _run_function(functions, "render_service_unit || exit $?", env=env)

    assert preserved.returncode == 0, preserved.stderr
    assert user_unit.read_text(encoding="utf-8") == "user-managed unit\n"
    assert systemctl_log.read_text(encoding="utf-8").splitlines() == calls


def test_download_models_stops_when_first_asset_download_fails(tmp_path):
    project = tmp_path / "adam"
    functions = _setup_copy(project)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/bin/sh
output=
url=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output) output=$2; shift 2 ;;
        https://*) url=$1; shift ;;
        *) shift ;;
    esac
done
case "$url" in
    *kokoro-v1.0.onnx) printf partial-response > "$output"; exit 22 ;;
    *) printf synthetic-asset > "$output" ;;
esac
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin"}

    result = _run_function(
        functions,
        "ASSUME_YES=true; SKIP_MODELS=false; download_models || exit $?",
        env=env,
    )

    assert result.returncode != 0
    assert not (project / "assets/voices/kokoro/kokoro-v1.0.onnx").exists()
    assert not (project / "assets/voices/kokoro/voices-v1.0.bin").exists()


def test_optional_feature_download_failure_is_reported(tmp_path):
    project = tmp_path / "adam"
    functions = _setup_copy(project)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text("#!/bin/sh\nexit 17\n", encoding="utf-8")
    uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin"}

    result = _run_function(
        functions,
        "ENABLE_IDEA_ROUTING=true; install_optional_feature_assets || exit $?",
        env=env,
    )

    assert result.returncode != 0


def test_package_install_stops_when_apt_metadata_update_fails(tmp_path):
    project = tmp_path / "adam"
    functions = _setup_copy(project)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sudo = fake_bin / "sudo"
    sudo.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$SUDO_LOG"\nexit 23\n',
        encoding="utf-8",
    )
    sudo.chmod(0o755)
    log = tmp_path / "sudo.log"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "SUDO_LOG": str(log),
    }

    result = _run_function(
        functions,
        "PKG_MANAGER=apt; SUDO_CMD=(sudo); install_package_group required true synthetic-package || exit $?",
        env=env,
    )

    assert result.returncode != 0
    assert log.read_text(encoding="utf-8").splitlines() == ["apt-get update"]
