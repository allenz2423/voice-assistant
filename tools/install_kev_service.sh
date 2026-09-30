#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit_file="$unit_dir/adam-kev.service"

command -v systemctl >/dev/null 2>&1 || { echo "systemctl is required for the Kev user service." >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "uv is required to run Kev." >&2; exit 1; }
mkdir -p "$unit_dir"
cat >"$unit_file" <<EOF
[Unit]
Description=Adam local Kev-0.8B OCR computer-use decision model
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$project_dir
Environment=ADAM_KEV_ROOT=%h/.local/share/adam/kev-runtime
Environment=CUDA_VISIBLE_DEVICES=
Environment=KEV_DTYPE=fp32
Environment=KEV_CUDA_GRAPHS=0
Environment=KEV_FUSED=0
ExecStart=$project_dir/tools/run_kev_server.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now adam-kev.service
systemctl --user --no-pager --full status adam-kev.service
