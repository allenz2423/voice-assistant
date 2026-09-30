"""Optional OmniParser screenshot region detection for computer control."""

from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


class OmniParserScreenshotGrounder:
    """Runs the detector in its isolated optional runtime, leaving Adam's env untouched."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        device: str = "cuda",
        gpu_uuid: str = "",
        python_path: str = "~/.local/share/adam/omniparser-runtime/bin/python",
        model_path: str = "~/.local/share/adam/models/omniparser-yolov8n.pt",
        confidence_threshold: float = 0.05,
        max_regions: int = 60,
        timeout_seconds: float = 15.0,
        worker_path: str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self.gpu_uuid = gpu_uuid.strip()
        self.python_path = Path(python_path).expanduser()
        self.model_path = Path(model_path).expanduser()
        self.confidence_threshold = confidence_threshold
        self.max_regions = max_regions
        self.timeout_seconds = timeout_seconds
        self.worker_path = Path(worker_path) if worker_path else (
            Path(__file__).resolve().parents[2] / "tools" / "omniparser_infer.py"
        )

    def _run(self, image: bytes) -> list[dict[str, Any]]:
        if not self.python_path.is_file():
            raise RuntimeError(f"OmniParser runtime not installed at {self.python_path}.")
        if not self.model_path.is_file():
            raise RuntimeError(f"OmniParser detector weights not found at {self.model_path}.")
        env = os.environ.copy()
        if self.device == "cuda":
            if not self.gpu_uuid:
                raise RuntimeError("Set computer_vision.gpu_uuid to the NVIDIA GPU Adam may use.")
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_uuid
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(
            [
                str(self.python_path), str(self.worker_path), str(self.model_path),
                self.device, str(self.confidence_threshold),
            ],
            input=image,
            capture_output=True,
            timeout=self.timeout_seconds,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail[-500:] or "OmniParser detector process failed.")
        try:
            result = json.loads(completed.stdout)
            regions = result["regions"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError("OmniParser returned invalid region data.") from exc
        return regions[: self.max_regions]

    def annotate(self, image: bytes) -> tuple[bytes, str]:
        """Return the screenshot overlaid with numbered boxes and pixel centers."""
        if not self.enabled:
            return image, ""
        regions = self._run(image)
        if not regions:
            return image, "OmniParser found no candidate clickable regions."

        screenshot = Image.open(io.BytesIO(image)).convert("RGB")
        draw = ImageDraw.Draw(screenshot)
        lines = [
            f"OmniParser candidate regions ({len(regions)} shown; IDs match the red screenshot labels). "
            "Boxes have no text labels; use visible UI text to choose the right one:"
        ]
        for region_id, region in enumerate(regions, start=1):
            x1, y1, x2, y2 = region["box"]
            cx, cy = region["center"]
            label = f"{region_id}"
            draw.rectangle((x1, y1, x2, y2), outline=(255, 45, 35), width=2)
            text_box = draw.textbbox((x1, y1), label)
            draw.rectangle(text_box, fill=(255, 45, 35))
            draw.text((x1, y1), label, fill=(255, 255, 255))
            lines.append(
                f"#{region_id} center=({cx},{cy}) box=({x1:g},{y1:g},{x2:g},{y2:g}) "
                f"confidence={region['score']:.2f}"
            )

        encoded = io.BytesIO()
        screenshot.save(encoded, format="PNG")
        return encoded.getvalue(), "\n".join(lines)
