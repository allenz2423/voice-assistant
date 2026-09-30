#!/usr/bin/env python3
"""Install the isolated OmniParser YOLOv8 Nano detector runtime and weights."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from urllib.error import URLError
from pathlib import Path


MODEL_URL = (
    "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/"
    "f55d0750e5b94db2125ef0b45b0fa4a85ddc59b4/icon_detect/model.pt"
)
MODEL_SHA256 = "dab3d4351ad00b035db829909a4db98354d5a90f6990e4ac00222a9a95d4bf57"


def install(device: str, gpu_uuid: str) -> None:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to create Adam's isolated OmniParser runtime.")
    if device == "cuda" and not gpu_uuid:
        raise RuntimeError("CUDA installation needs the selected NVIDIA GPU UUID.")

    data_home = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    runtime = data_home / "adam" / "omniparser-runtime"
    model_path = data_home / "adam" / "models" / "omniparser-yolov8n.pt"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not (runtime / "bin" / "python").is_file():
        subprocess.run([uv, "venv", "--python", "3.13", str(runtime)], check=True)
    index = (
        "https://download.pytorch.org/whl/cu124"
        if device == "cuda"
        else "https://download.pytorch.org/whl/cpu"
    )
    subprocess.run(
        [
            uv, "pip", "install", "--python", str(runtime / "bin" / "python"),
            "--index-url", index,
            "--extra-index-url", "https://pypi.org/simple",
            "torch==2.6.0", "torchvision==0.21.0", "numpy", "pillow",
            "ultralytics==8.3.70",
        ],
        check=True,
    )

    if not model_path.exists() or _sha256(model_path) != MODEL_SHA256:
        partial = model_path.with_suffix(".pt.part")
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Adam-Setup"})
        digest = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            expected = int(response.headers.get("Content-Length", "0"))
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if expected:
                    print(f"\rDownloading OmniParser YOLOv8 Nano detector: {total * 100 // expected}%", end="", flush=True)
        print()
        if digest.hexdigest() != MODEL_SHA256:
            partial.unlink(missing_ok=True)
            raise RuntimeError("Downloaded OmniParser YOLOv8 Nano weights failed SHA-256 verification.")
        partial.replace(model_path)

    print(f"OmniParser runtime: {runtime / 'bin' / 'python'}")
    print(f"OmniParser YOLOv8 Nano model: {model_path}")
    print(f"Selected backend:   {device}" + (f" ({gpu_uuid})" if gpu_uuid else ""))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--gpu-uuid", default="")
    args = parser.parse_args()
    try:
        install(args.device, args.gpu_uuid)
        return 0
    except (OSError, subprocess.CalledProcessError, RuntimeError, URLError) as exc:
        print(f"OmniParser installation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
