#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ASSUME_YES=false
DRY_RUN=false
SKIP_SYS_PKGS=false
SKIP_PYTHON=false
SKIP_MODELS=false
SKIP_OLLAMA=false
SKIP_SERVICE=false
SKIP_SPEAKER_VERIFICATION=false
SKIP_ENROLLMENT=false
CPU_ONLY=false
FORCE_NVIDIA_RUNTIME=false
ENABLE_IDEA_ROUTING=false
ENABLE_BROWSER_NAVIGATION=false
ENABLE_DIARIZATION=false
ENABLE_OMNIPARSER=false
DISABLE_COMPUTER_CONTROL=false
INSTALL_DESKTOP_TOOLS=false
RUNTIME_EXTRA=runtime-cpu
DISTRO_ID=unknown
DISTRO_NAME=Linux
DISTRO_LIKE=""
PKG_MANAGER=""
CORE_PKGS=()
DESKTOP_PKGS=()
TEMP_FILES=()
SYSTEMD_RUNTIME_DIR="${ADAM_SYSTEMD_RUNTIME_DIR:-/run/systemd/system}"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

cleanup() {
    local path
    if ((${#TEMP_FILES[@]})); then
        for path in "${TEMP_FILES[@]}"; do
            [[ -n "$path" ]] && rm -f -- "$path"
        done
    fi
    return 0
}
trap cleanup EXIT

usage() {
    cat <<'EOF'
Adam setup

Usage: ./setup.sh [OPTIONS]

Interactive mode installs dependencies and runs the configuration wizard.
Use --yes for a noninteractive bootstrap: it configures the requested flags,
installs dependencies and models, and never enables or starts a service.

Options:
  -y, --yes                    Noninteractive bootstrap; no service start
      --dry-run                Print the plan without running commands or writing files
      --skip-sys-pkgs           Do not call a system package manager
      --skip-python-deps        Use an already prepared .venv; do not run uv sync
      --skip-models             Skip Kokoro model and voice downloads
      --skip-ollama             Do not offer to install/pull Ollama in the wizard
      --skip-speaker-verification
                                Disable speaker verification and omit its extra
      --skip-enrollment         Skip microphone enrollment in the wizard
      --cpu-only                Use CPU runtime packages
      --nvidia-runtime          Use NVIDIA runtime packages
      --idea-routing            Enable local idea routing and install its extra
      --browser-navigation      Enable isolated browser navigation and install its extra
      --diarization             Enable Nemotron diarization and install its extra
      --omniparser              Enable OmniParser screenshot grounding and install runtime
      --disable-computer-control
                                Disable screenshot-guided desktop control in config
      --install-desktop-tools   Also install optional X11/Wayland desktop packages
      --skip-service            Skip systemd user-service setup
  -h, --help                    Show this help and exit

Requirements:
  Run as a regular user. Install Astral uv before starting setup:
  https://docs.astral.sh/uv/getting-started/installation/

System package support: Debian/Ubuntu, Fedora/RHEL, openSUSE, Arch, Gentoo,
Alpine, Void and Solus. NixOS users should install dependencies declaratively
and pass --skip-sys-pkgs.
EOF
}

log_info() { printf '[INFO] %s\n' "$*"; }
log_ok() { printf '[ OK ] %s\n' "$*"; }
log_warn() { printf '[WARN] %s\n' "$*" >&2; }
log_error() { printf '[ERROR] %s\n' "$*" >&2; }

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=true ;;
        -y|--yes) ASSUME_YES=true ;;
        --skip-sys-pkgs) SKIP_SYS_PKGS=true ;;
        --skip-python-deps) SKIP_PYTHON=true ;;
        --skip-models) SKIP_MODELS=true ;;
        --skip-ollama) SKIP_OLLAMA=true ;;
        --skip-service) SKIP_SERVICE=true ;;
        --skip-speaker-verification) SKIP_SPEAKER_VERIFICATION=true ;;
        --skip-enrollment) SKIP_ENROLLMENT=true ;;
        --cpu-only) CPU_ONLY=true ;;
        --nvidia-runtime) FORCE_NVIDIA_RUNTIME=true ;;
        --idea-routing) ENABLE_IDEA_ROUTING=true ;;
        --browser-navigation) ENABLE_BROWSER_NAVIGATION=true ;;
        --diarization|--nemotron-diarization) ENABLE_DIARIZATION=true ;;
        --omniparser) ENABLE_OMNIPARSER=true ;;
        --disable-computer-control) DISABLE_COMPUTER_CONTROL=true ;;
        --install-desktop-tools) INSTALL_DESKTOP_TOOLS=true ;;
        *) log_error "Unknown option: $1"; usage >&2; exit 2 ;;
    esac
    shift
done

if [[ "$CPU_ONLY" == true && "$FORCE_NVIDIA_RUNTIME" == true ]]; then
    log_error "--cpu-only and --nvidia-runtime cannot be used together."
    exit 2
fi
if [[ "$DISABLE_COMPUTER_CONTROL" == true && "$INSTALL_DESKTOP_TOOLS" == true ]]; then
    log_error "--disable-computer-control and --install-desktop-tools cannot be used together."
    exit 2
fi
if [[ "$DISABLE_COMPUTER_CONTROL" == true && "$ENABLE_OMNIPARSER" == true ]]; then
    log_error "--disable-computer-control and --omniparser cannot be used together."
    exit 2
fi
detect_distro() {
    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
        DISTRO_ID="${ID:-unknown}"
        DISTRO_NAME="${NAME:-Linux}"
        DISTRO_LIKE="${ID_LIKE:-}"
    fi
}

select_packages() {
    detect_distro
    local ids="${DISTRO_ID} ${DISTRO_LIKE}"

    if [[ "$ids" =~ (debian|ubuntu|linuxmint|pop|elementary|zorin|devuan|kali|parrot) ]]; then
        PKG_MANAGER=apt
        CORE_PKGS=(python3 libportaudio2 portaudio19-dev curl git jq pkg-config build-essential)
        DESKTOP_PKGS=(at-spi2-core gir1.2-atspi-2.0 python3-gi wtype ydotool xdotool wmctrl grim slurp brightnessctl libnotify-bin pipewire pipewire-pulse wireplumber alsa-utils pulseaudio-utils)
    elif [[ "$ids" =~ (fedora|rhel|centos|rocky|alma|amzn|openmandriva) ]]; then
        if command -v dnf >/dev/null 2>&1; then PKG_MANAGER=dnf; else PKG_MANAGER=yum; fi
        CORE_PKGS=(python3 portaudio portaudio-devel curl git jq pkgconf-pkg-config gcc gcc-c++ make)
        DESKTOP_PKGS=(at-spi2-core python3-gobject wtype ydotool xdotool wmctrl grim slurp brightnessctl libnotify pipewire pipewire-pulseaudio wireplumber alsa-utils pulseaudio-utils)
    elif [[ "$ids" =~ (opensuse|suse) ]]; then
        PKG_MANAGER=zypper
        CORE_PKGS=(python3 portaudio portaudio-devel curl git jq pkg-config gcc make)
        DESKTOP_PKGS=(at-spi2-core python-gobject wtype ydotool xdotool wmctrl grim slurp brightnessctl libnotify-tools pipewire pipewire-pulseaudio wireplumber alsa-utils pulseaudio-utils)
    elif [[ "$ids" =~ (arch|cachyos|manjaro|endeavouros|artix) ]]; then
        PKG_MANAGER=pacman
        CORE_PKGS=(python portaudio curl git jq pkgconf base-devel)
        DESKTOP_PKGS=(at-spi2-core python-gobject wtype ydotool xdotool wmctrl grim slurp brightnessctl libnotify pipewire pipewire-pulse wireplumber alsa-utils libpulse)
    elif [[ "$DISTRO_ID" == gentoo || "$DISTRO_LIKE" == *gentoo* ]]; then
        PKG_MANAGER=emerge
        CORE_PKGS=(dev-lang/python media-libs/portaudio net-misc/curl dev-vcs/git app-misc/jq dev-build/pkgconf sys-devel/gcc sys-devel/make)
        DESKTOP_PKGS=(app-accessibility/at-spi2-core dev-python/pygobject gui-apps/wtype app-misc/ydotool x11-misc/xdotool x11-misc/wmctrl gui-apps/grim gui-apps/slurp sys-power/brightnessctl x11-libs/libnotify media-video/pipewire media-video/wireplumber media-sound/alsa-utils)
    elif [[ "$DISTRO_ID" == alpine || "$DISTRO_LIKE" == *alpine* ]]; then
        PKG_MANAGER=apk
        CORE_PKGS=(python3 portaudio portaudio-dev curl git jq pkgconf build-base linux-headers)
        DESKTOP_PKGS=(at-spi2-core py3-gobject3 wtype ydotool xdotool wmctrl grim slurp brightnessctl libnotify pipewire pipewire-pulse wireplumber alsa-utils pulseaudio-utils)
    elif [[ "$DISTRO_ID" == void || "$DISTRO_LIKE" == *void* ]]; then
        PKG_MANAGER=xbps-install
        CORE_PKGS=(python3 portaudio-devel curl git jq pkg-config base-devel)
        DESKTOP_PKGS=(at-spi2-core python3-gobject wtype ydotool xdotool wmctrl grim slurp brightnessctl libnotify pipewire wireplumber alsa-utils pulseaudio)
    elif [[ "$DISTRO_ID" == solus || "$DISTRO_LIKE" == *solus* ]]; then
        PKG_MANAGER=eopkg
        CORE_PKGS=(python3 portaudio-devel curl git jq pkg-config system.devel)
        DESKTOP_PKGS=(at-spi2-core python-gobject wtype ydotool xdotool wmctrl grim slurp brightnessctl libnotify pipewire wireplumber alsa-utils pulseaudio)
    elif [[ "$DISTRO_ID" == nixos || "$DISTRO_LIKE" == *nixos* ]]; then
        PKG_MANAGER=nixos
        CORE_PKGS=(python3 portaudio curl git jq pkg-config gcc)
        DESKTOP_PKGS=(pipewire wireplumber libnotify at-spi2-core wtype ydotool xdotool wmctrl grim slurp brightnessctl)
    else
        # A manager fallback is useful for derivatives with incomplete os-release
        # data. Keep the package vocabulary conservative and explain uncertainty.
        if command -v apt-get >/dev/null 2>&1; then
            PKG_MANAGER=apt
            CORE_PKGS=(python3 libportaudio2 portaudio19-dev curl git jq pkg-config build-essential)
            DESKTOP_PKGS=(at-spi2-core gir1.2-atspi-2.0 python3-gi wtype ydotool xdotool wmctrl grim slurp brightnessctl libnotify-bin pipewire pipewire-pulse wireplumber alsa-utils pulseaudio-utils)
        elif command -v dnf >/dev/null 2>&1; then
            PKG_MANAGER=dnf
            CORE_PKGS=(python3 portaudio portaudio-devel curl git jq pkgconf-pkg-config gcc gcc-c++ make)
            DESKTOP_PKGS=(at-spi2-core python3-gobject wtype ydotool xdotool wmctrl grim slurp brightnessctl libnotify pipewire pipewire-pulseaudio wireplumber alsa-utils pulseaudio-utils)
        elif command -v pacman >/dev/null 2>&1; then
            PKG_MANAGER=pacman
            CORE_PKGS=(python portaudio curl git jq pkgconf base-devel)
            DESKTOP_PKGS=(at-spi2-core python-gobject wtype ydotool xdotool wmctrl grim slurp brightnessctl libnotify pipewire pipewire-pulse wireplumber alsa-utils libpulse)
        else
            log_error "Unsupported distribution '$DISTRO_NAME'. Install Python, PortAudio development files, curl, git, jq and a C build toolchain, then rerun with --skip-sys-pkgs."
            return 1
        fi
        log_warn "Unknown distribution ID '$DISTRO_ID'; using $PKG_MANAGER package names. Review them before continuing."
    fi
}

prompt_yes_no() {
    local prompt="$1" default_yes="$2" answer=""
    if [[ "$ASSUME_YES" == true ]]; then
        [[ "$default_yes" == true ]]
        return
    fi
    while true; do
        if [[ "$default_yes" == true ]]; then
            read -r -p "$prompt [Y/n] " answer || return 1
            [[ -z "$answer" || "$answer" =~ ^[Yy]([Ee][Ss])?$ ]] && return 0
            [[ "$answer" =~ ^[Nn]([Oo])?$ ]] && return 1
        else
            read -r -p "$prompt [y/N] " answer || return 1
            [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]] && return 0
            [[ -z "$answer" || "$answer" =~ ^[Nn]([Oo])?$ ]] && return 1
        fi
        printf 'Please answer yes or no.\n' >&2
    done
}

SUDO_CMD=()
find_privilege_tool() {
    if command -v sudo >/dev/null 2>&1; then
        SUDO_CMD=(sudo)
    elif command -v doas >/dev/null 2>&1; then
        SUDO_CMD=(doas)
    else
        log_error "System package installation requires sudo or doas. Use --skip-sys-pkgs if dependencies are already installed."
        return 1
    fi
}

pkg_install() {
    case "$PKG_MANAGER" in
        apt) "${SUDO_CMD[@]}" apt-get install -y -- "$@" ;;
        dnf|yum) "${SUDO_CMD[@]}" "$PKG_MANAGER" install -y -- "$@" ;;
        zypper) "${SUDO_CMD[@]}" zypper --non-interactive install -- "$@" ;;
        pacman) "${SUDO_CMD[@]}" pacman -S --needed --noconfirm -- "$@" ;;
        emerge) "${SUDO_CMD[@]}" emerge --ask=n --noreplace "$@" ;;
        apk) "${SUDO_CMD[@]}" apk add -- "$@" ;;
        xbps-install) "${SUDO_CMD[@]}" xbps-install -Sy "$@" ;;
        eopkg) "${SUDO_CMD[@]}" eopkg install -y "$@" ;;
        *) log_error "No installer for package manager '$PKG_MANAGER'."; return 1 ;;
    esac
}

install_package_group() {
    local label="$1" required="$2"
    shift 2
    local -a failures=()
    (($#)) || return 0
    log_info "Installing $label packages with $PKG_MANAGER."
    if [[ "$PKG_MANAGER" == apt ]]; then
        "${SUDO_CMD[@]}" apt-get update || return 1
    fi
    if pkg_install "$@"; then
        log_ok "$label packages installed."
        return 0
    fi
    log_warn "The batch install failed; retrying $label packages individually."
    local package
    for package in "$@"; do
        if ! pkg_install "$package"; then failures+=("$package"); fi
    done
    if ((${#failures[@]})); then
        if [[ "$required" == true ]]; then
            log_error "Required packages could not be installed: ${failures[*]}"
            return 1
        fi
        log_warn "Optional packages unavailable: ${failures[*]}"
        return 0
    fi
    log_ok "$label packages installed individually."
}

install_system_packages() {
    [[ "$SKIP_SYS_PKGS" == true ]] && { log_info "Skipping system package manager (--skip-sys-pkgs)."; return 0; }
    select_packages || return 1
    if [[ "$PKG_MANAGER" == nixos ]]; then
        log_error "NixOS package changes are declarative. Add the listed runtime packages to your Nix configuration, then rerun with --skip-sys-pkgs."
        return 1
    fi
    if [[ "$ASSUME_YES" != true ]] && ! prompt_yes_no "Install required packages for $DISTRO_NAME?" true; then
        log_error "Required system packages were not installed. Rerun with --skip-sys-pkgs only if they are already present."
        return 1
    fi
    find_privilege_tool || return 1
    install_package_group "required" true "${CORE_PKGS[@]}" || return 1

    local install_desktop=false
    if [[ "$DISABLE_COMPUTER_CONTROL" != true ]]; then
        if [[ "$INSTALL_DESKTOP_TOOLS" == true ]]; then
            install_desktop=true
        elif [[ "$ASSUME_YES" != true ]] && prompt_yes_no "Install optional X11/Wayland desktop-control packages?" false; then
            install_desktop=true
        fi
    fi
    if [[ "$install_desktop" == true ]]; then
        install_package_group "optional desktop-control" false "${DESKTOP_PKGS[@]}"
    else
        log_info "Skipping optional desktop-control packages. Use --install-desktop-tools to add them."
    fi
}

choose_runtime() {
    if [[ "$CPU_ONLY" == true ]]; then
        RUNTIME_EXTRA=runtime-cpu
    elif [[ "$FORCE_NVIDIA_RUNTIME" == true ]]; then
        RUNTIME_EXTRA=runtime-nvidia
    elif [[ "$ASSUME_YES" != true ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        if prompt_yes_no "Install NVIDIA ONNX/CUDA runtime libraries?" false; then
            RUNTIME_EXTRA=runtime-nvidia
        fi
    fi
    log_info "Python runtime selection: $RUNTIME_EXTRA. A detected GPU does not change this default."
}

ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        log_error "uv is required but is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/ and rerun setup."
        return 1
    fi
    log_info "Using $(uv --version)."
}

set_config_scalar() {
    local section="$1" key="$2" value="$3"
    python3 - "$CONFIG_FILE" "$section" "$key" "$value" <<'PY'
from pathlib import Path
import os
import re
import sys
import tempfile

path = Path(sys.argv[1])
section, key, value = sys.argv[2:]
text = path.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)
section_re = re.compile(rf"^{re.escape(section)}:\s*(?:#.*)?(?:\r?\n)?$")
top_re = re.compile(r"^[^\s#][^:\n]*:\s*(?:#.*)?(?:\r?\n)?$")
key_re = re.compile(rf"^(\s+{re.escape(key)}:\s*)(.*?)(\s+#.*)?(\r?\n)?$")

start = next((i for i, line in enumerate(lines) if section_re.match(line)), None)
if start is None:
    if text and not text.endswith("\n"):
        lines[-1] += "\n"
    if lines and lines[-1].strip():
        lines.append("\n")
    lines.extend([f"{section}:\n", f"  {key}: {value}\n"])
else:
    end = next((i for i in range(start + 1, len(lines)) if top_re.match(lines[i])), len(lines))
    found = False
    for i in range(start + 1, end):
        match = key_re.match(lines[i])
        if match:
            newline = match.group(4) or "\n"
            comment = match.group(3) or ""
            lines[i] = f"{match.group(1)}{value}{comment}{newline}"
            found = True
            break
    if not found:
        lines.insert(start + 1, f"  {key}: {value}\n")

contents = "".join(lines)
fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
        handle.write(contents)
    os.chmod(temp_name, 0o600)
    os.replace(temp_name, path)
except BaseException:
    try:
        os.unlink(temp_name)
    except FileNotFoundError:
        pass
    raise
PY
}

prepare_config() {
    local config="$SCRIPT_DIR/config.yaml"
    if [[ -L "$config" ]]; then
        if [[ ! -e "$config" ]]; then
            log_error "config.yaml is a broken symlink; repair it before running setup."
            return 1
        fi
        CONFIG_FILE="$(python3 - "$config" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)"
        if [[ ! -f "$CONFIG_FILE" ]]; then
            log_error "config.yaml must point to a regular file."
            return 1
        fi
    else
        CONFIG_FILE="$config"
    fi
    if [[ ! -e "$config" ]]; then
        if [[ ! -f "$SCRIPT_DIR/config.yaml.example" ]]; then
            log_error "Neither config.yaml nor config.yaml.example exists."
            return 1
        fi
        cp -- "$SCRIPT_DIR/config.yaml.example" "$config"
        log_info "Created config.yaml from config.yaml.example."
    fi
    [[ -f "$config" ]] || { log_error "config.yaml is not a regular file."; return 1; }

    if [[ "$SKIP_SPEAKER_VERIFICATION" == true ]]; then
        set_config_scalar speaker_verification enabled false || return 1
    fi
    if [[ "$ENABLE_IDEA_ROUTING" == true ]]; then
        set_config_scalar idea_routing enabled true || return 1
    fi
    if [[ "$ENABLE_BROWSER_NAVIGATION" == true ]]; then
        set_config_scalar browser_navigation enabled true || return 1
    fi
    if [[ "$ENABLE_DIARIZATION" == true ]]; then
        set_config_scalar speaker_diarization enabled true || return 1
    fi
    if [[ "$ENABLE_OMNIPARSER" == true ]]; then
        set_config_scalar computer_vision enabled true || return 1
        set_config_scalar computer_vision backend omniparser || return 1
    fi
    if [[ "$DISABLE_COMPUTER_CONTROL" == true ]]; then
        set_config_scalar computer_control enabled false || return 1
    fi
    chmod 600 -- "$config" || {
        log_error "Could not restrict config.yaml permissions to owner-only (0600)."
        return 1
    }
    log_ok "Configuration is private (mode 0600); existing settings were preserved except explicit flags."
}

download_asset() {
    local url="$1" destination="$2" label="$3"
    if [[ -s "$destination" ]]; then
        log_ok "$label already exists."
        return 0
    fi
    mkdir -p -- "$(dirname -- "$destination")"
    local temp="${destination}.part.$$"
    TEMP_FILES+=("$temp")
    log_info "Downloading $label."
    curl --fail --location --retry 3 --output "$temp" "$url" || return 1
    [[ -s "$temp" ]] || { log_error "$label download was empty."; return 1; }
    mv -- "$temp" "$destination"
    log_ok "Downloaded $label."
}

download_models() {
    if [[ "$SKIP_MODELS" == true ]]; then
        log_info "Skipping Kokoro downloads (--skip-models)."
        return 0
    fi
    if [[ "$ASSUME_YES" != true ]] && ! prompt_yes_no "Download the Kokoro model and voices (about 300 MB)?" true; then
        log_warn "Kokoro downloads skipped. Select another TTS engine or download these assets before starting Adam."
        return 0
    fi
    download_asset \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" \
        "$SCRIPT_DIR/assets/voices/kokoro/kokoro-v1.0.onnx" "Kokoro ONNX model" || return 1
    download_asset \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
        "$SCRIPT_DIR/assets/voices/kokoro/voices-v1.0.bin" "Kokoro voice embeddings" || return 1
}

sync_python_environment() {
    if [[ "$SKIP_PYTHON" == true ]]; then
        log_info "Skipping uv sync (--skip-python-deps)."
        return 0
    fi
    local -a args=(sync --inexact --extra "$RUNTIME_EXTRA")
    [[ "$SKIP_SPEAKER_VERIFICATION" == true ]] || args+=(--extra speaker-verification)

    local existing_extras=""
    if [[ -f "$CONFIG_FILE" ]]; then
        existing_extras="$(python3 - "$CONFIG_FILE" <<'PY'
from pathlib import Path
import re, sys

path = Path(sys.argv[1])
try:
    text = path.read_text(encoding="utf-8")
except Exception:
    sys.exit(0)

def is_enabled(section: str, key: str = "enabled") -> bool:
    sec_match = re.search(rf"(?m)^{re.escape(section)}:\s*(?:#.*)?$", text)
    if not sec_match:
        return False
    rest = text[sec_match.end():]
    next_sec = re.search(r"(?m)^[^\s#][^:\n]*:\s*(?:#.*)?$", rest)
    sec_body = rest[:next_sec.start()] if next_sec else rest
    val_match = re.search(rf"(?m)^\s+{re.escape(key)}:\s*([^\s#]+)", sec_body)
    if not val_match:
        return False
    return val_match.group(1).lower() in ("true", "yes", "1")

extras = []
if is_enabled("idea_routing"):
    extras.append("intent-routing")
if is_enabled("browser_navigation"):
    extras.append("browser-control")
if is_enabled("speaker_diarization"):
    extras.append("nemotron-diarization")
if is_enabled("computer_control", "ocr_only"):
    extras.append("computer-ocr")

print(" ".join(extras))
PY
)"
    fi

    if [[ "$ENABLE_IDEA_ROUTING" == true ]] || [[ " $existing_extras " =~ [[:space:]]intent-routing[[:space:]] ]]; then
        args+=(--extra intent-routing)
    fi
    if [[ "$ENABLE_BROWSER_NAVIGATION" == true ]] || [[ " $existing_extras " =~ [[:space:]]browser-control[[:space:]] ]]; then
        args+=(--extra browser-control)
    fi
    if [[ "$ENABLE_DIARIZATION" == true ]] || [[ " $existing_extras " =~ [[:space:]]nemotron-diarization[[:space:]] ]]; then
        args+=(--extra nemotron-diarization)
    fi
    if [[ " $existing_extras " =~ [[:space:]]computer-ocr[[:space:]] ]]; then
        args+=(--extra computer-ocr)
    fi

    log_info "Synchronizing the Python environment."
    uv "${args[@]}"
}

install_optional_feature_assets() {
    if [[ "$SKIP_PYTHON" == true ]]; then
        local python="$SCRIPT_DIR/.venv/bin/python"
        if { [[ "$ENABLE_IDEA_ROUTING" == true ]] || [[ "$ENABLE_BROWSER_NAVIGATION" == true ]] || [[ "$ENABLE_DIARIZATION" == true ]]; } && [[ ! -x "$python" ]]; then
            log_error "--skip-python-deps with an optional feature requires a prepared .venv."
            return 1
        fi
        if [[ "$ENABLE_IDEA_ROUTING" == true ]] && ! "$python" -c 'import huggingface_hub, tokenizers' >/dev/null 2>&1; then
            log_error "Idea routing dependencies are missing from .venv; remove --skip-python-deps or install the intent-routing extra first."
            return 1
        fi
        if [[ "$ENABLE_BROWSER_NAVIGATION" == true ]] && ! "$python" -c 'import playwright' >/dev/null 2>&1; then
            log_error "Browser navigation needs Playwright in .venv; remove --skip-python-deps or install the browser-control extra first."
            return 1
        fi
        if [[ "$ENABLE_DIARIZATION" == true ]] && ! "$python" -c 'import transformers, librosa' >/dev/null 2>&1; then
            log_error "Diarization dependencies are missing from .venv; remove --skip-python-deps or install the nemotron-diarization extra first."
            return 1
        fi
    fi
    if [[ "$ENABLE_IDEA_ROUTING" == true ]]; then
        uv run --no-sync python -m src.intent.idea_router --download || return 1
    fi
    if [[ "$ENABLE_DIARIZATION" == true && "$SKIP_MODELS" != true ]]; then
        log_info "Downloading Nemotron diarization model..."
        uv run --no-sync python -c \
            "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/Nemotron-3-Diarization', allow_patterns=['config.json', 'model.safetensors', 'processor_config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'special_tokens_map.json'])" \
            || return 1
    fi
    if [[ "$ENABLE_BROWSER_NAVIGATION" == true ]]; then
        local browser selection
        if [[ "$SKIP_PYTHON" == true ]]; then
            if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
                log_error "--skip-python-deps requires a prepared .venv before enabling browser navigation."
                return 1
            fi
            if ! "$SCRIPT_DIR/.venv/bin/python" -c 'import playwright' >/dev/null 2>&1; then
                log_error "Browser navigation needs Playwright in .venv; remove --skip-python-deps or install the browser-control extra first."
                return 1
            fi
        fi
        selection="$(uv run --no-sync python - "$CONFIG_FILE" <<'PY'
from pathlib import Path
import sys, yaml

path = Path(sys.argv[1])
try:
    c = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
except Exception:
    c = {}
b = c.get("browser_navigation", {})
browser = (b.get("browser") or "default").lower()
default_browser = (c.get("desktop", {}).get("default_browser") or "chromium").lower()
print(f"{browser}|{default_browser}")
PY
        )" || return 1
        browser="${selection%%|*}"
        local default_browser="${selection#*|}"
        [[ "$browser" == default ]] && browser="$default_browser"
        case "$browser" in
            firefox|*firefox*) uv run --no-sync playwright install firefox || return 1 ;;
            *)
                local channel_or_binary=""
                case "$browser" in
                    chromium|chromium-browser) channel_or_binary="" ;;
                    *edge*|*msedge*) channel_or_binary="$(command -v microsoft-edge-stable || command -v microsoft-edge || true)" ;;
                    *chrome*) channel_or_binary="$(command -v google-chrome-stable || command -v google-chrome || true)" ;;
                    *brave*) channel_or_binary="$(command -v brave-browser || command -v brave || true)" ;;
                    *vivaldi*) channel_or_binary="$(command -v vivaldi-stable || command -v vivaldi || true)" ;;
                    *zen*) channel_or_binary="$(command -v zen || command -v zen-browser || true)" ;;
                    *opera*) channel_or_binary="$(command -v opera || true)" ;;
                esac
                if [[ -n "$channel_or_binary" ]]; then
                    log_info "Using installed browser: $channel_or_binary"
                else
                    log_info "Installing Playwright Chromium build..."
                    uv run --no-sync playwright install chromium || return 1
                fi
                ;;
        esac
    fi
    if [[ "$ENABLE_OMNIPARSER" == true && "$SKIP_MODELS" != true ]]; then
        local device="cpu" gpu_uuid=""
        if [[ "$RUNTIME_EXTRA" == "runtime-nvidia" ]]; then
            gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -n 1 || true)"
            gpu_uuid="$(echo "$gpu_uuid" | tr -d '[:space:]')"
            if [[ -n "$gpu_uuid" ]]; then
                device="cuda"
            fi
        fi
        log_info "Installing OmniParser detector ($device)..."
        local -a omni_args=(python3 tools/install_omniparser.py --device "$device")
        if [[ -n "$gpu_uuid" ]]; then
            omni_args+=(--gpu-uuid "$gpu_uuid")
        fi
        "${omni_args[@]}" || {
            log_warn "OmniParser installation failed; leaving feature disabled."
            set_config_scalar computer_vision enabled false
        }
    fi
}

render_service_unit() {
    local template="$SCRIPT_DIR/systemd/adam.service.template"
    local destination="$HOME/.config/systemd/user/adam.service"
    [[ -f "$template" ]] || { log_error "Service template is missing: $template"; return 1; }
    if [[ -e "$destination" || -L "$destination" ]]; then
        log_info "Preserving existing service unit: $destination"
        return 0
    fi
    mkdir -p -- "$(dirname -- "$destination")"
    local temp
    temp="$(mktemp "$(dirname -- "$destination")/.adam.service.XXXXXX")"
    TEMP_FILES+=("$temp")
    local render_result=0
    python3 - "$template" "$temp" "$SCRIPT_DIR" "$HOME" <<'PY' || render_result=$?
import re
import sys
from pathlib import Path

template, output, project, home = sys.argv[1:]

def is_safe_path(val: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_./+-]+", val))

if not is_safe_path(project) or not is_safe_path(home):
    sys.exit(3)

text = Path(template).read_text(encoding="utf-8")
text = text.replace("{{PROJECT_DIR}}", project)
text = text.replace("{{HOME}}", home)
Path(output).write_text(text, encoding="utf-8")
PY
    if ((render_result == 3)); then
        rm -f -- "$temp"
        log_warn "The project or home path contains characters this service template cannot safely represent; skipping unit generation."
        return 0
    elif ((render_result != 0)); then
        rm -f -- "$temp"
        log_error "Failed to render systemd service unit."
        return 1
    fi
    chmod 644 -- "$temp"
    # Hard-link creation is atomic and refuses to clobber a concurrently created unit.
    if ln -- "$temp" "$destination" 2>/dev/null; then
        rm -f -- "$temp"
        log_ok "Installed user unit without enabling it: $destination"
        if systemctl --user show-environment >/dev/null 2>&1; then
            systemctl --user daemon-reload
        else
            log_warn "Could not reach the systemd user manager; run systemctl --user daemon-reload later."
        fi
    else
        rm -f -- "$temp"
        log_info "A service unit appeared during setup; preserving it: $destination"
    fi
}

print_dry_run() {
    select_packages >/dev/null 2>&1 || true
    local planned_runtime=runtime-cpu
    [[ "$FORCE_NVIDIA_RUNTIME" == true ]] && planned_runtime=runtime-nvidia
    local planned_extras=("$planned_runtime")
    [[ "$SKIP_SPEAKER_VERIFICATION" != true ]] && planned_extras+=(speaker-verification)
    [[ "$ENABLE_IDEA_ROUTING" == true ]] && planned_extras+=(intent-routing)
    [[ "$ENABLE_BROWSER_NAVIGATION" == true ]] && planned_extras+=(browser-control)
    [[ "$ENABLE_DIARIZATION" == true ]] && planned_extras+=(nemotron-diarization)
    local extras_str=""
    for e in "${planned_extras[@]}"; do
        extras_str+=" --extra $e"
    done
    local pkg_desc="${PKG_MANAGER:-unknown} (required)"
    if [[ "$INSTALL_DESKTOP_TOOLS" == true ]]; then
        pkg_desc="${PKG_MANAGER:-unknown} (required + optional desktop)"
    fi
    printf 'Dry run: no commands will be executed and no files will be changed.\n'
    printf 'Project: %s\nDistribution: %s (%s)\n' "$SCRIPT_DIR" "$DISTRO_NAME" "$DISTRO_ID"
    printf 'System packages: %s\n' "$([[ "$SKIP_SYS_PKGS" == true ]] && printf 'skipped' || printf '%s' "$pkg_desc")"
    printf 'Python dependencies: %s\n' "$([[ "$SKIP_PYTHON" == true ]] && printf 'skipped' || printf 'uv sync%s' "$extras_str")"
    printf 'Kokoro assets: %s\n' "$([[ "$SKIP_MODELS" == true ]] && printf 'skipped' || printf 'download if accepted/defaulted')"
    printf 'Config: initialize if absent; preserve existing values except explicit flags\n'
    local service_plan
    if [[ "$SKIP_SERVICE" == true ]]; then
        service_plan="skipped (--skip-service)"
    elif [[ "$ASSUME_YES" == true ]]; then
        service_plan="install only if absent; never enable or start"
    else
        service_plan="interactive wizard owns service setup"
    fi
    printf 'Service: %s\n' "$service_plan"
    printf 'Requested flags: idea-routing=%s browser-navigation=%s diarization=%s omniparser=%s disable-computer-control=%s install-desktop-tools=%s speaker-verification-skip=%s\n' \
        "$ENABLE_IDEA_ROUTING" "$ENABLE_BROWSER_NAVIGATION" "$ENABLE_DIARIZATION" "$ENABLE_OMNIPARSER" "$DISABLE_COMPUTER_CONTROL" "$INSTALL_DESKTOP_TOOLS" "$SKIP_SPEAKER_VERIFICATION"
}

main() {
    if [[ "$DRY_RUN" == true ]]; then
        print_dry_run
        return 0
    fi
    if ((EUID == 0)); then
        log_error "Run setup as the regular account that will use Adam, not as root."
        return 2
    fi
    if [[ "$ASSUME_YES" != true && ! -t 0 ]]; then
        log_error "Interactive setup needs a terminal. Use --yes for bootstrap or --dry-run to inspect the plan."
        return 2
    fi
    export ADAM_SKIP_SPEAKER_VERIFICATION="$([[ "$SKIP_SPEAKER_VERIFICATION" == true ]] && printf 1 || printf 0)"
    export ADAM_SKIP_ENROLLMENT="$([[ "$SKIP_ENROLLMENT" == true ]] && printf 1 || printf 0)"
    export ADAM_SKIP_OLLAMA="$([[ "$SKIP_OLLAMA" == true ]] && printf 1 || printf 0)"
    export ADAM_SKIP_SERVICE="$([[ "$SKIP_SERVICE" == true ]] && printf 1 || printf 0)"
    export ADAM_CPU_ONLY="$([[ "$CPU_ONLY" == true ]] && printf 1 || printf 0)"
    export ADAM_SKIP_PYTHON="$([[ "$SKIP_PYTHON" == true ]] && printf 1 || printf 0)"
    export ADAM_IDEA_ROUTING_DEFAULT="$([[ "$ENABLE_IDEA_ROUTING" == true ]] && printf 1 || printf 0)"
    export ADAM_BROWSER_NAVIGATION_DEFAULT="$([[ "$ENABLE_BROWSER_NAVIGATION" == true ]] && printf 1 || printf 0)"
    export ADAM_DIARIZATION_ENABLED="$([[ "$ENABLE_DIARIZATION" == true ]] && printf 1 || printf 0)"

    ensure_uv || return 1
    choose_runtime
    export ADAM_RUNTIME_EXTRA="$RUNTIME_EXTRA"
    install_system_packages || return 1
    prepare_config || return 1
    sync_python_environment || return 1
    download_models || return 1
    install_optional_feature_assets || return 1

    if [[ "$ASSUME_YES" == true ]]; then
        if [[ "$SKIP_SERVICE" != true ]] && [[ -d "$SYSTEMD_RUNTIME_DIR" ]] && command -v systemctl >/dev/null 2>&1; then
            render_service_unit || return 1
        elif [[ "$SKIP_SERVICE" != true ]]; then
            log_info "No active systemd user service setup detected; start Adam manually with uv run --no-sync python -m src.main."
        fi
        log_ok "Bootstrap complete. The service was not enabled or started."
        return 0
    fi

    log_info "Starting the interactive configuration wizard. It owns the service prompt and unit installation."
    uv run --no-sync python "$SCRIPT_DIR/tools/setup_wizard.py"
}

main "$@"
