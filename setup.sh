#!/usr/bin/env bash
# ==============================================================================
# Shin / Adam Voice Assistant - Universal Linux Setup Script
# Supports: Gentoo, openSUSE/SLES, Debian/Ubuntu, RHEL/Fedora/CentOS, Arch/CachyOS
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

print_usage() {
    cat <<EOF
Usage: ./setup.sh [OPTIONS]

Interactive setup wizard for the Shin / Adam Voice Assistant.
Detects Linux distribution, installs dependencies, sets up Python virtualenv,
downloads neural models, and interactively configures audio, wake word, and service.

Options:
  -y, --yes            Non-interactive / unattended mode (accept all defaults)
  --skip-sys-pkgs      Skip system package manager installation
  --skip-models        Skip downloading Kokoro TTS and Silero VAD neural models
  --skip-ollama        Skip Ollama check and LLM model pull
  --skip-service       Skip creating and enabling systemd user service
  -h, --help           Show this help message and exit

Supported Linux Distributions:
  - Debian / Ubuntu / Linux Mint / Pop!_OS (apt)
  - Fedora / RHEL / Rocky / Alma / CentOS (dnf)
  - openSUSE Leap / Tumbleweed (zypper)
  - Gentoo Linux (emerge)
  - Arch Linux / CachyOS / Manjaro (pacman)
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
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --skip-ollama)
            SKIP_OLLAMA=true
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

    # Determine package manager and package list
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
            curl
            git
            jq
        )
    elif [[ "$DISTRO_ID" =~ (debian|ubuntu|linuxmint|pop|elementary|zorin) || "$DISTRO_LIKE" == *"debian"* || "$DISTRO_LIKE" == *"ubuntu"* ]]; then
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
            curl
            git
            jq
        )
    elif [[ "$DISTRO_ID" =~ (fedora|rhel|centos|rocky|alma) || "$DISTRO_LIKE" == *"fedora"* || "$DISTRO_LIKE" == *"rhel"* ]]; then
        PKG_MANAGER="dnf"
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
            curl
            git
            jq
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
            curl
            git
            jq
        )
    fi

    if [[ -z "$PKG_MANAGER" ]]; then
        log_warn "Unrecognized Linux distribution '${DISTRO_ID}'."
        log_warn "Please ensure the following tools are installed: portaudio, wtype, grim, slurp, brightnessctl, notify-send, pipewire, wireplumber, curl, git, jq."
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
    case "$PKG_MANAGER" in
        emerge)
            if [[ -n "$SUDO" ]]; then
                $SUDO emerge --ask=n --noreplace "${PKGS[@]}"
            else
                emerge --ask=n --noreplace "${PKGS[@]}"
            fi
            ;;
        zypper)
            if [[ -n "$SUDO" ]]; then
                $SUDO zypper --non-interactive install -y "${PKGS[@]}"
            else
                zypper --non-interactive install -y "${PKGS[@]}"
            fi
            ;;
        apt)
            if [[ -n "$SUDO" ]]; then
                $SUDO apt-get update -y && $SUDO apt-get install -y "${PKGS[@]}"
            else
                apt-get update -y && apt-get install -y "${PKGS[@]}"
            fi
            ;;
        dnf)
            if [[ -n "$SUDO" ]]; then
                $SUDO dnf install -y "${PKGS[@]}"
            else
                dnf install -y "${PKGS[@]}"
            fi
            ;;
        pacman)
            if [[ -n "$SUDO" ]]; then
                $SUDO pacman -S --needed --noconfirm "${PKGS[@]}"
            else
                pacman -S --needed --noconfirm "${PKGS[@]}"
            fi
            ;;
    esac

    log_success "System package verification completed."
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
    log_info "Synchronizing Python virtual environment and dependencies..."
    uv sync --extra speaker-verification
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
    local user_systemd_dir="$HOME/.config/systemd/user"
    mkdir -p "$user_systemd_dir"
    local service_dest="${user_systemd_dir}/shin.service"
    local template_file="${SCRIPT_DIR}/systemd/shin.service.template"

    if [[ -f "$template_file" ]]; then
        sed \
            -e "s|{{PROJECT_DIR}}|${SCRIPT_DIR}|g" \
            -e "s|{{HOME}}|${HOME}|g" \
            "$template_file" > "$service_dest"
    else
        cat <<EOF > "$service_dest"
[Unit]
Description=Shin: Voice-Activated Autonomous Terminal Agent
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
    generate_systemd_template
    if [[ "$SKIP_SERVICE" != true ]]; then
        systemctl --user enable --now shin.service
        log_success "shin.service enabled and started."
    fi
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
main() {
    echo -e "${CLR_CYAN}${CLR_BOLD}================================================================${CLR_RESET}"
    echo -e "${CLR_CYAN}${CLR_BOLD}          🎙️   Shin / Adam Voice Assistant Setup               ${CLR_RESET}"
    echo -e "${CLR_CYAN}${CLR_BOLD}================================================================${CLR_RESET}"

    # Base dependencies
    install_system_packages
    ensure_uv
    setup_python_env
    download_models

    # Initialize config.yaml from example if missing
    if [[ ! -f "${SCRIPT_DIR}/config.yaml" && -f "${SCRIPT_DIR}/config.yaml.example" ]]; then
        log_info "Initializing config.yaml from config.yaml.example..."
        cp "${SCRIPT_DIR}/config.yaml.example" "${SCRIPT_DIR}/config.yaml"
    fi

    if [[ "$ASSUME_YES" == true ]]; then
        run_unattended_setup
    else
        # Always generate / update the systemd service file first
        generate_systemd_template
        # Launch full interactive wizard for audio, persona, ollama, and enrollment
        uv run python "${SCRIPT_DIR}/tools/setup_wizard.py"
    fi
}

main
