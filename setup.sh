#!/usr/bin/env bash
# ==============================================================================
# Adam / Adam Voice Assistant - Universal Linux Setup Script
# Supports: Debian/Ubuntu, Fedora/RHEL, openSUSE, Arch, Gentoo, Alpine, Void, Solus
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color helpers
CLR_RESET="\033[0m"
CLR_INFO="\033[1;34m"
CLR_SUCCESS="\033[1;32m"
CLR_WARN="\033[1;33m"
CLR_ERR="\033[1;31m"
CLR_BOLD="\033[1m"
CLR_CYAN="\033[1;36m"

log_info()    { echo -e "${CLR_INFO}[INFO]${CLR_RESET} $*"; }
log_success() { echo -e "${CLR_SUCCESS}[OK]${CLR_RESET} $*"; }
log_warn()    { echo -e "${CLR_WARN}[WARN]${CLR_RESET} $*"; }
log_err()     { echo -e "${CLR_ERR}[ERROR]${CLR_RESET} $*" >&2; }

# Default configuration flags
ASSUME_YES=false
SKIP_SYS_PKGS=false
SKIP_MODELS=false
SKIP_OLLAMA=false
SKIP_SERVICE=false
SKIP_PYTHON=false
SKIP_SPEAKER_VERIFICATION=false
SKIP_ENROLLMENT=false
CPU_ONLY=false
FORCE_NVIDIA_RUNTIME=false
RUNTIME_EXTRA="runtime-cpu"

print_usage() {
    cat <<EOF
Usage: ./setup.sh [OPTIONS]

Interactive setup wizard for the Adam / Adam Voice Assistant.
Detects Linux distribution, installs dependencies, sets up Python virtualenv,
downloads neural models, and interactively configures audio, wake word, speech,
speaker verification, optional Nemotron diarization, and the user service.

Nemotron diarization uses NVIDIA's official Transformers model runtime. The wizard
lets you choose CPU or a specific CUDA GPU, installs the optional runtime, and can
download the model before starting the service. Meeting transcripts preserve mixed
audio and add speaker labels; they do not isolate or filter speakers.

Options:
  -y, --yes            Non-interactive / unattended mode (accept all defaults)
  --skip-sys-pkgs      Skip system package manager installation
  --skip-python-deps   Do not run uv sync (use an existing prepared environment)
  --skip-models        Skip downloading Kokoro TTS and Silero VAD neural models
  --skip-ollama        Skip the prompt to pull a local Ollama model
  --skip-speaker-verification  Do not install speaker verification or enroll a voice
  --skip-enrollment    Install speaker verification but skip microphone enrollment
  --cpu-only           Use CPU runtime packages and avoid CUDA libraries
  --nvidia-runtime     Install ONNX Runtime and CUDA support for NVIDIA GPUs
  --skip-service       Skip creating and enabling systemd user service
  -h, --help           Show this help message and exit

Supported Linux Distributions:
  - Debian / Ubuntu / Linux Mint / Pop!_OS (apt)
  - Fedora / RHEL / Rocky / Alma / CentOS (dnf)
  - openSUSE Leap / Tumbleweed (zypper)
  - Gentoo Linux (emerge)
  - Arch Linux / CachyOS / Manjaro (pacman)
  - Alpine Linux (apk; Python ML wheels may be limited by musl), Void (xbps), Solus (eopkg)
  - NixOS is recognized; install system packages declaratively with Nix
EOF
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)
            ASSUME_YES=true
            shift
            ;;
        --skip-sys-pkgs)
            SKIP_SYS_PKGS=true
            shift
            ;;
        --skip-python-deps)
            SKIP_PYTHON=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --skip-ollama)
            SKIP_OLLAMA=true
            shift
            ;;
        --skip-speaker-verification)
            SKIP_SPEAKER_VERIFICATION=true
            shift
            ;;
        --skip-enrollment)
            SKIP_ENROLLMENT=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --nvidia-runtime)
            FORCE_NVIDIA_RUNTIME=true
            shift
            ;;
        --skip-service)
            SKIP_SERVICE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_err "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ "$CPU_ONLY" == true && "$FORCE_NVIDIA_RUNTIME" == true ]]; then
    log_err "--cpu-only and --nvidia-runtime cannot be used together."
    exit 2
fi

# Ensure not running directly as root
if [[ $EUID -eq 0 ]]; then
    log_warn "Running setup.sh as root is NOT recommended."
    log_warn "The assistant runs as a systemd user service with user-specific audio sinks."
    if [[ "$ASSUME_YES" != true ]]; then
        read -r -p "Are you sure you want to continue as root? [y/N] " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            log_err "Setup aborted."
            exit 1
        fi
    fi
fi

# Detect privilege escalation tool (sudo or doas)
SUDO=""
if [[ $EUID -ne 0 ]]; then
    if command -v sudo &>/dev/null; then
        SUDO="sudo"
    elif command -v doas &>/dev/null; then
        SUDO="doas"
    else
        log_warn "Neither 'sudo' nor 'doas' found. System package installation might fail if root rights are required."
    fi
fi

# ------------------------------------------------------------------------------
# 1. Distro Detection & System Packages
# ------------------------------------------------------------------------------
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
        DISTRO_ID="${ID:-unknown}"
        DISTRO_NAME="${NAME:-Linux}"
        DISTRO_LIKE="${ID_LIKE:-}"
    else
        DISTRO_ID="unknown"
        DISTRO_NAME="Unknown Linux"
        DISTRO_LIKE=""
    fi
}

install_system_packages() {
    detect_distro
    log_info "Detected Linux distribution: ${CLR_BOLD}${DISTRO_NAME}${CLR_RESET} (${DISTRO_ID})"

    if [[ "$SKIP_SYS_PKGS" == true ]]; then
        log_info "Skipping system package installation (--skip-sys-pkgs)."
        return 0
    fi

    # Select both the package manager and distro-specific package names.
    local PKG_MANAGER=""
    local -a PKGS=()

    if [[ "$DISTRO_ID" == "gentoo" || "$DISTRO_LIKE" == *"gentoo"* ]]; then
        PKG_MANAGER="emerge"
        PKGS=(
            media-libs/portaudio
            gui-apps/wtype
            gui-apps/grim
            gui-apps/slurp
            sys-power/brightnessctl
            x11-libs/libnotify
            media-video/pipewire
            media-video/wireplumber
            media-sound/alsa-utils
            net-misc/curl
            dev-vcs/git
            app-misc/jq
            dev-build/pkgconf
            sys-devel/gcc
            sys-devel/make
        )
    elif [[ "$DISTRO_ID" == *"suse"* || "$DISTRO_LIKE" == *"suse"* ]]; then
        PKG_MANAGER="zypper"
        PKGS=(
            portaudio
            portaudio-devel
            wtype
            grim
            slurp
            brightnessctl
            libnotify-tools
            pipewire
            pipewire-pulseaudio
            wireplumber
            alsa-utils
            pulseaudio-utils
            curl
            git
            jq
            pkg-config
            gcc
            make
        )
    elif [[ "$DISTRO_ID" =~ (debian|ubuntu|linuxmint|pop|elementary|zorin|devuan|kali|parrot) || "$DISTRO_LIKE" == *"debian"* || "$DISTRO_LIKE" == *"ubuntu"* ]]; then
        PKG_MANAGER="apt"
        PKGS=(
            libportaudio2
            portaudio19-dev
            wtype
            grim
            slurp
            brightnessctl
            libnotify-bin
            pipewire
            pipewire-pulse
            wireplumber
            alsa-utils
            pulseaudio-utils
            curl
            git
            jq
            pkg-config
            build-essential
        )
    elif [[ "$DISTRO_ID" =~ (fedora|rhel|centos|rocky|alma|ol|amzn|openmandriva) || "$DISTRO_LIKE" == *"fedora"* || "$DISTRO_LIKE" == *"rhel"* ]]; then
        if command -v dnf &>/dev/null; then PKG_MANAGER="dnf"; else PKG_MANAGER="yum"; fi
        PKGS=(
            portaudio
            portaudio-devel
            wtype
            grim
            slurp
            brightnessctl
            libnotify
            pipewire
            pipewire-pulseaudio
            wireplumber
            alsa-utils
            pulseaudio-utils
            curl
            git
            jq
            pkgconf-pkg-config
            gcc
            gcc-c++
            make
        )
    elif [[ "$DISTRO_ID" =~ (arch|cachyos|manjaro|endeavouros|artix) || "$DISTRO_LIKE" == *"arch"* ]]; then
        PKG_MANAGER="pacman"
        PKGS=(
            portaudio
            wtype
            grim
            slurp
            brightnessctl
            libnotify
            pipewire
            pipewire-pulse
            wireplumber
            alsa-utils
            libpulse
            curl
            git
            jq
            pkgconf
            base-devel
        )
    elif [[ "$DISTRO_ID" == "alpine" || "$DISTRO_LIKE" == *"alpine"* ]]; then
        PKG_MANAGER="apk"
        PKGS=(
            portaudio portaudio-dev wtype grim slurp brightnessctl libnotify
            pipewire pipewire-pulse wireplumber alsa-utils pulseaudio-utils
            curl git jq pkgconf build-base linux-headers
        )
    elif [[ "$DISTRO_ID" == "void" || "$DISTRO_LIKE" == *"void"* ]]; then
        PKG_MANAGER="xbps-install"
        PKGS=(
            portaudio-devel wtype grim slurp brightnessctl libnotify
            pipewire wireplumber alsa-utils pulseaudio curl git jq
            pkg-config base-devel
        )
    elif [[ "$DISTRO_ID" == "solus" || "$DISTRO_LIKE" == *"solus"* ]]; then
        PKG_MANAGER="eopkg"
        PKGS=(
            portaudio-devel wtype grim slurp brightnessctl libnotify
            pipewire wireplumber alsa-utils pulseaudio curl git jq
            pkg-config system.devel
        )
    elif [[ "$DISTRO_ID" == "nixos" || "$DISTRO_LIKE" == *"nixos"* ]]; then
        log_warn "NixOS uses declarative system packages. Add portaudio, pipewire, wireplumber, libnotify, curl, git, jq, wtype, grim, slurp, and brightnessctl to your system configuration."
        log_warn "Continuing without changing your NixOS configuration."
        return 0
    elif command -v apt-get &>/dev/null; then
        PKG_MANAGER="apt"
        PKGS=(libportaudio2 portaudio19-dev pipewire pipewire-pulse wireplumber pulseaudio-utils alsa-utils curl git jq pkg-config build-essential)
    elif command -v dnf &>/dev/null; then
        PKG_MANAGER="dnf"
        PKGS=(portaudio portaudio-devel pipewire wireplumber pulseaudio-utils alsa-utils curl git jq pkgconf-pkg-config gcc gcc-c++ make)
    elif command -v yum &>/dev/null; then
        PKG_MANAGER="yum"
        PKGS=(portaudio portaudio-devel pipewire wireplumber pulseaudio-utils alsa-utils curl git jq pkgconfig gcc gcc-c++ make)
    elif command -v zypper &>/dev/null; then
        PKG_MANAGER="zypper"
        PKGS=(portaudio portaudio-devel pipewire pipewire-pulseaudio wireplumber pulseaudio-utils alsa-utils curl git jq pkg-config gcc make)
    elif command -v pacman &>/dev/null; then
        PKG_MANAGER="pacman"
        PKGS=(portaudio pipewire pipewire-pulse wireplumber pulseaudio alsa-utils curl git jq pkgconf base-devel)
    elif command -v apk &>/dev/null; then
        PKG_MANAGER="apk"
        PKGS=(portaudio portaudio-dev pipewire wireplumber alsa-utils pulseaudio curl git jq pkgconf build-base linux-headers)
    elif command -v xbps-install &>/dev/null; then
        PKG_MANAGER="xbps-install"
        PKGS=(portaudio-devel pipewire wireplumber alsa-utils pulseaudio curl git jq pkg-config base-devel)
    elif command -v eopkg &>/dev/null; then
        PKG_MANAGER="eopkg"
        PKGS=(portaudio-devel pipewire wireplumber alsa-utils pulseaudio curl git jq pkg-config system.devel)
    fi

    if [[ -z "$PKG_MANAGER" ]]; then
        log_warn "Could not identify a supported package manager for '${DISTRO_ID}'."
        log_warn "Install PortAudio development files, PipeWire, WirePlumber, PulseAudio utilities (pactl), curl, git, jq, and a C build toolchain, then rerun with --skip-sys-pkgs."
        return 0
    fi

    # Interactive confirmation prompt
    if [[ "$ASSUME_YES" != true ]]; then
        echo ""
        read -r -p "Install required system packages for ${DISTRO_NAME} using ${PKG_MANAGER}? [Y/n] " answer
        if [[ "$answer" =~ ^[Nn]$ ]]; then
            log_info "Skipping system package installation."
            return 0
        fi
    fi

    log_info "Installing system packages via ${PKG_MANAGER}..."
    if [[ "$PKG_MANAGER" == "apt" ]]; then
        if [[ -n "$SUDO" ]]; then "$SUDO" apt-get update -y; else apt-get update -y; fi
    fi

    install_package_batch() {
        case "$PKG_MANAGER" in
            apt) if [[ -n "$SUDO" ]]; then "$SUDO" apt-get install -y "$@"; else apt-get install -y "$@"; fi ;;
            dnf|yum) if [[ -n "$SUDO" ]]; then "$SUDO" "$PKG_MANAGER" install -y "$@"; else "$PKG_MANAGER" install -y "$@"; fi ;;
            zypper) if [[ -n "$SUDO" ]]; then "$SUDO" zypper --non-interactive install -y "$@"; else zypper --non-interactive install -y "$@"; fi ;;
            pacman) if [[ -n "$SUDO" ]]; then "$SUDO" pacman -S --needed --noconfirm "$@"; else pacman -S --needed --noconfirm "$@"; fi ;;
            emerge) if [[ -n "$SUDO" ]]; then "$SUDO" emerge --ask=n --noreplace "$@"; else emerge --ask=n --noreplace "$@"; fi ;;
            apk) if [[ -n "$SUDO" ]]; then "$SUDO" apk add "$@"; else apk add "$@"; fi ;;
            xbps-install) if [[ -n "$SUDO" ]]; then "$SUDO" xbps-install -Sy "$@"; else xbps-install -Sy "$@"; fi ;;
            eopkg) if [[ -n "$SUDO" ]]; then "$SUDO" eopkg install -y "$@"; else eopkg install -y "$@"; fi ;;
        esac
    }

    install_one_package() {
        local package="$1"
        case "$PKG_MANAGER" in
            apt) if [[ -n "$SUDO" ]]; then "$SUDO" apt-get install -y "$package"; else apt-get install -y "$package"; fi ;;
            dnf|yum) if [[ -n "$SUDO" ]]; then "$SUDO" "$PKG_MANAGER" install -y "$package"; else "$PKG_MANAGER" install -y "$package"; fi ;;
            zypper) if [[ -n "$SUDO" ]]; then "$SUDO" zypper --non-interactive install -y "$package"; else zypper --non-interactive install -y "$package"; fi ;;
            pacman) if [[ -n "$SUDO" ]]; then "$SUDO" pacman -S --needed --noconfirm "$package"; else pacman -S --needed --noconfirm "$package"; fi ;;
            emerge) if [[ -n "$SUDO" ]]; then "$SUDO" emerge --ask=n --noreplace "$package"; else emerge --ask=n --noreplace "$package"; fi ;;
            apk) if [[ -n "$SUDO" ]]; then "$SUDO" apk add "$package"; else apk add "$package"; fi ;;
            xbps-install) if [[ -n "$SUDO" ]]; then "$SUDO" xbps-install -Sy "$package"; else xbps-install -Sy "$package"; fi ;;
            eopkg) if [[ -n "$SUDO" ]]; then "$SUDO" eopkg install -y "$package"; else eopkg install -y "$package"; fi ;;
        esac
    }

    local -a FAILED_PKGS=()
    if ! install_package_batch "${PKGS[@]}"; then
        log_warn "The package manager rejected at least one package name; retrying packages individually."
        for package in "${PKGS[@]}"; do
            if ! install_one_package "$package"; then FAILED_PKGS+=("$package"); fi
        done
    fi
    if ((${#FAILED_PKGS[@]})); then
        log_warn "Some optional packages were unavailable: ${FAILED_PKGS[*]}"
        log_warn "Install missing tools later if a desktop action or audio device needs them."
    else
        log_success "System package installation completed."
    fi
}

# ------------------------------------------------------------------------------
# 2. uv Package Manager Installation
# ------------------------------------------------------------------------------
ensure_uv() {
    log_info "Checking for Astral 'uv' Python manager..."
    if ! command -v uv &>/dev/null; then
        if [[ -x "$HOME/.local/bin/uv" ]]; then
            export PATH="$HOME/.local/bin:$PATH"
        elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
            export PATH="$HOME/.cargo/bin:$PATH"
        else
            log_info "Installing uv via official Astral installer..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        fi
    fi

    if ! command -v uv &>/dev/null; then
        log_err "Failed to locate or install 'uv'. Please install uv manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    log_success "Using uv: $(uv --version)"
}

# ------------------------------------------------------------------------------
# 3. Python Environment & Dependencies
# ------------------------------------------------------------------------------
setup_python_env() {
    export ADAM_SKIP_SPEAKER_VERIFICATION="$([[ "$SKIP_SPEAKER_VERIFICATION" == true ]] && echo 1 || echo 0)"
    export ADAM_SKIP_ENROLLMENT="$([[ "$SKIP_ENROLLMENT" == true ]] && echo 1 || echo 0)"
    export ADAM_SKIP_OLLAMA="$([[ "$SKIP_OLLAMA" == true ]] && echo 1 || echo 0)"
    export ADAM_SKIP_SERVICE="$([[ "$SKIP_SERVICE" == true ]] && echo 1 || echo 0)"
    export ADAM_CPU_ONLY="$([[ "$CPU_ONLY" == true ]] && echo 1 || echo 0)"
    export ADAM_SKIP_PYTHON="$([[ "$SKIP_PYTHON" == true ]] && echo 1 || echo 0)"
    if [[ "$SKIP_PYTHON" == true ]]; then
        log_info "Skipping Python dependency sync (--skip-python-deps)."
        [[ "$CPU_ONLY" == true ]] && RUNTIME_EXTRA="runtime-cpu"
        [[ "$FORCE_NVIDIA_RUNTIME" == true ]] && RUNTIME_EXTRA="runtime-nvidia"
        export ADAM_RUNTIME_EXTRA="${RUNTIME_EXTRA}"
        return 0
    fi

    if [[ "$CPU_ONLY" == true ]]; then
        RUNTIME_EXTRA="runtime-cpu"
    elif [[ "$FORCE_NVIDIA_RUNTIME" == true ]]; then
        RUNTIME_EXTRA="runtime-nvidia"
    elif command -v nvidia-smi &>/dev/null && nvidia-smi -L 2>/dev/null | grep 'GPU' >/dev/null; then
        if [[ "$ASSUME_YES" == true ]]; then
            RUNTIME_EXTRA="runtime-nvidia"
        else
            echo ""
            read -r -p "Install NVIDIA CUDA runtime support? This only installs libraries; Adam will use only the GPU you select. [Y/n] " answer
            [[ "$answer" =~ ^[Nn]$ ]] && RUNTIME_EXTRA="runtime-cpu" || RUNTIME_EXTRA="runtime-nvidia"
        fi
    else
        RUNTIME_EXTRA="runtime-cpu"
        log_info "No NVIDIA GPU was detected; using the CPU ONNX Runtime."
    fi

    export ADAM_RUNTIME_EXTRA="$RUNTIME_EXTRA"

    log_info "Synchronizing Python virtual environment and dependencies..."
    local -a UV_ARGS=(sync --extra "$RUNTIME_EXTRA")
    if [[ "$SKIP_SPEAKER_VERIFICATION" != true ]]; then
        UV_ARGS+=(--extra speaker-verification)
    fi
    uv "${UV_ARGS[@]}"
    log_success "Python virtual environment configured in $SCRIPT_DIR/.venv"
}

# ------------------------------------------------------------------------------
# 4. Neural Models & Voice Assets Download
# ------------------------------------------------------------------------------
download_asset() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [[ -f "$dest" && -s "$dest" ]]; then
        log_success "$name already present: $dest"
        return 0
    fi

    log_info "Downloading $name..."
    mkdir -p "$(dirname "$dest")"
    local temp_dest="${dest}.part.$$"
    
    if curl -fSL --progress-bar "$url" -o "$temp_dest"; then
        mv "$temp_dest" "$dest"
        log_success "Successfully downloaded $name."
    else
        rm -f "$temp_dest"
        log_err "Failed to download $name from $url"
        return 1
    fi
}

download_models() {
    if [[ "$SKIP_MODELS" == true ]]; then
        log_info "Skipping model downloads (--skip-models)."
        return 0
    fi

    if [[ "$ASSUME_YES" != true ]]; then
        echo ""
        read -r -p "Download Kokoro TTS neural voice and Silero VAD weights (~350MB)? [Y/n] " answer
        if [[ "$answer" =~ ^[Nn]$ ]]; then
            log_info "Skipping model downloads."
            return 0
        fi
    fi

    log_info "Verifying required neural model assets..."
    mkdir -p assets/models assets/voices/kokoro assets/chimes

    # Silero VAD v5 ONNX
    download_asset \
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
        "assets/models/silero_vad.onnx" \
        "Silero VAD ONNX model"

    # Kokoro-82M TTS ONNX Model
    download_asset \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" \
        "assets/voices/kokoro/kokoro-v1.0.onnx" \
        "Kokoro-82M ONNX model weights"

    # Kokoro-82M Voice Binaries (am_adam, etc.)
    download_asset \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
        "assets/voices/kokoro/voices-v1.0.bin" \
        "Kokoro voice embeddings (voices-v1.0.bin)"

    log_success "All voice and VAD assets verified."
}

# ------------------------------------------------------------------------------
# 5. Systemd User Service Template Generation
# ------------------------------------------------------------------------------
generate_systemd_template() {
    if [[ ! -d /run/systemd/system ]] || ! command -v systemctl &>/dev/null || ! systemctl --user show-environment &>/dev/null; then
        log_warn "This Linux system does not use systemd; skipping the systemd user unit."
        log_info "Start Adam manually with: uv run python -m src.main"
        return 0
    fi
    local user_systemd_dir="$HOME/.config/systemd/user"
    mkdir -p "$user_systemd_dir"
    local service_dest="${user_systemd_dir}/adam.service"
    local template_file="${SCRIPT_DIR}/systemd/adam.service.template"

    if [[ -f "$template_file" ]]; then
        sed \
            -e "s|{{PROJECT_DIR}}|${SCRIPT_DIR}|g" \
            -e "s|{{HOME}}|${HOME}|g" \
            "$template_file" > "$service_dest"
    else
        cat <<EOF > "$service_dest"
[Unit]
Description=Adam: Voice-Activated Autonomous Terminal Agent
After=pipewire.service wireplumber.service pipewire-pulse.service
Wants=pipewire.service wireplumber.service

[Service]
Type=simple
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${SCRIPT_DIR}/.venv/bin/python -m src.main
Restart=on-failure
RestartSec=3s
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment="PATH=${SCRIPT_DIR}/.venv/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
PassEnvironment=DISPLAY WAYLAND_DISPLAY HYPRLAND_INSTANCE_SIGNATURE XDG_CURRENT_DESKTOP XDG_RUNTIME_DIR
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF
    fi

    systemctl --user daemon-reload
    log_success "Systemd user service written to: $service_dest"
}

# ------------------------------------------------------------------------------
# 6. Unattended / Automated Fallback
# ------------------------------------------------------------------------------
run_unattended_setup() {
    log_info "Running in automated non-interactive mode..."
    if [[ "$SKIP_SERVICE" != true ]] && [[ -d /run/systemd/system ]] && command -v systemctl &>/dev/null && systemctl --user show-environment &>/dev/null; then
        generate_systemd_template
        systemctl --user enable --now adam.service
        log_success "adam.service enabled and started."
    elif [[ "$SKIP_SERVICE" != true ]]; then
        log_warn "No systemd user service is available; start Adam manually with: uv run python -m src.main"
    fi
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
main() {
    echo -e "${CLR_CYAN}${CLR_BOLD}================================================================${CLR_RESET}"
    echo -e "${CLR_CYAN}${CLR_BOLD}          🎙️   Adam / Adam Voice Assistant Setup               ${CLR_RESET}"
    echo -e "${CLR_CYAN}${CLR_BOLD}================================================================${CLR_RESET}"

    # Base dependencies
    install_system_packages
    ensure_uv
    setup_python_env
    download_models

    # Initialize config.yaml from example if missing without touching existing settings.
    if [[ ! -f "${SCRIPT_DIR}/config.yaml" && -f "${SCRIPT_DIR}/config.yaml.example" ]]; then
        log_info "Initializing config.yaml from config.yaml.example..."
        cp "${SCRIPT_DIR}/config.yaml.example" "${SCRIPT_DIR}/config.yaml"
        chmod 600 "${SCRIPT_DIR}/config.yaml"
        if [[ "$RUNTIME_EXTRA" == "runtime-cpu" ]]; then
            log_info "Preparing laptop-friendly transcription defaults (small.en on CPU)."
            sed -i \
                -e '/^stt:/,/^[^ ]/ { s/^  model_size: "qwen3-asr-1.7b"/  model_size: "small.en"/; s/^  device: "Vulkan0"/  device: "cpu"/; s/^  compute_type: "int8_float32"/  compute_type: "int8"/; }' \
                -e 's/^  device_id: 0/  device_id: -1/' \
                "${SCRIPT_DIR}/config.yaml"
        fi
    fi

    if [[ "$ASSUME_YES" == true ]]; then
        run_unattended_setup
    else
        # Generate the unit only on systemd hosts and when the user wants one.
        if [[ "$SKIP_SERVICE" != true ]]; then generate_systemd_template; fi
        # Launch full interactive wizard for audio, persona, ollama, and enrollment
        uv run python "${SCRIPT_DIR}/tools/setup_wizard.py"
    fi
}

main
