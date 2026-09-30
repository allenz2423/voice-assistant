#!/usr/bin/env bash
set -euo pipefail

data_root="${XDG_DATA_HOME:-$HOME/.local/share}/adam"
kev_root="${ADAM_KEV_ROOT:-$data_root/kev-runtime}"
kev_repo="https://github.com/jaredpalmer/kev.git"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to run Kev. Install uv, then retry." >&2
  exit 1
fi

if [[ ! -f "$kev_root/pyproject.toml" ]]; then
  mkdir -p "$(dirname "$kev_root")"
  git clone --depth 1 "$kev_repo" "$kev_root"
fi

# Use a separate Python 3.13 environment. Adam's main runtime may use Python 3.14.
uv sync --project "$kev_root" --extra serve --python 3.13

# Keep inference on CPU unless the user explicitly selects one GPU. GTX 10-series
# cards also need fp32 because they do not support native BF16.
export KEV_DTYPE="${KEV_DTYPE:-fp32}"
export KEV_CUDA_GRAPHS="${KEV_CUDA_GRAPHS:-0}"
export KEV_FUSED="${KEV_FUSED:-0}"
if [[ -n "${ADAM_KEV_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$ADAM_KEV_CUDA_VISIBLE_DEVICES"
else
  export CUDA_VISIBLE_DEVICES=""
fi

exec uv run --project "$kev_root" --extra serve python -m kev.serve \
  --run jaredpalmer/kev-0.8b --host 127.0.0.1 --port "${ADAM_KEV_PORT:-8009}"
