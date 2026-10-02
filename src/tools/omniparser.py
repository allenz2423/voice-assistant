"""Optional OmniParser screenshot region detection for computer control."""

from __future__ import annotations

import atexit
import io
import json
import os
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from src.tools.desktop_timing import log_duration, log_elapsed, timed_stage


class OmniParserScreenshotGrounder:
    """Runs the detector in its isolated optional runtime as a persistent resident process."""

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
        preload: bool = False,
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
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        atexit.register(self.close)
        if preload and self.enabled:
            self.load()

    def _ensure_worker(self) -> subprocess.Popen:
        if self._process is not None and self._process.poll() is None:
            return self._process
        startup_started = time.perf_counter()
        self.close()
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

        proc = subprocess.Popen(
            [
                str(self.python_path), str(self.worker_path), str(self.model_path),
                self.device, str(self.confidence_threshold), "--server",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            line = proc.stdout.readline().decode("utf-8", errors="replace").strip() if proc.stdout else ""
            if line != "READY":
                err = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
                proc.kill()
                raise RuntimeError(err[-500:] or f"OmniParser worker failed to start (got {line!r}).")
        except Exception:
            log_duration("omniparser.worker_start", startup_started, status="error")
            proc.kill()
            raise
        self._process = proc
        log_duration("omniparser.worker_start", startup_started)
        print(f"[Vision] OmniParser resident worker ready on {self.device} ({self.gpu_uuid or 'CPU'}).", flush=True)
        return self._process

    def load(self) -> None:
        """Preload the resident OmniParser worker into memory/device."""
        if not self.enabled:
            return
        with self._lock:
            self._ensure_worker()

    def close(self) -> None:
        """Shut down the resident worker process."""
        with self._lock:
            if self._process is not None:
                proc = self._process
                self._process = None
                try:
                    if proc.poll() is None:
                        if proc.stdin:
                            try:
                                proc.stdin.write(struct.pack(">I", 0))
                                proc.stdin.flush()
                            except Exception:
                                pass
                        proc.terminate()
                        try:
                            proc.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                except Exception:
                    pass

    def _run(self, image: bytes) -> list[dict[str, Any]]:
        with self._lock:
            last_err = None
            for attempt in range(2):
                try:
                    proc = self._ensure_worker()
                    assert proc.stdin is not None and proc.stdout is not None
                    write_started = time.perf_counter()
                    proc.stdin.write(struct.pack(">I", len(image)))
                    proc.stdin.write(image)
                    proc.stdin.flush()
                    log_duration("omniparser.request_write", write_started, image_bytes=len(image))

                    response_started = time.perf_counter()
                    line = proc.stdout.readline().decode("utf-8", errors="replace").strip()
                    log_duration("omniparser.worker_response_wait", response_started)
                    if not line:
                        raise RuntimeError("OmniParser worker exited or returned empty response.")
                    result = json.loads(line)
                    if "error" in result:
                        raise RuntimeError(f"OmniParser inference failed: {result['error']}")
                    timings = result.get("timings_ms", {})
                    for stage in ("decode", "inference", "postprocess"):
                        elapsed_ms = timings.get(stage)
                        if isinstance(elapsed_ms, (int, float)):
                            log_elapsed(f"omniparser.worker.{stage}", elapsed_ms)
                    regions = result["regions"]
                    log_elapsed("omniparser.worker_roundtrip", (time.perf_counter() - write_started) * 1000,
                                region_count=len(regions))
                    return regions[: self.max_regions]
                except Exception as exc:
                    last_err = exc
                    self.close()
            raise RuntimeError(f"OmniParser resident worker error: {last_err}") from last_err

    def annotate(self, image: bytes) -> tuple[bytes, str]:
        """Return the screenshot overlaid with numbered boxes and pixel centers."""
        if not self.enabled:
            return image, ""
        with timed_stage("omniparser.annotate"):
            regions = self._run(image)
        if not regions:
            return image, "OmniParser found no candidate clickable regions."

        overlay_started = time.perf_counter()
        screenshot = Image.open(io.BytesIO(image)).convert("RGB")
        draw = ImageDraw.Draw(screenshot)
        lines = [
            f"OmniParser candidate regions ({len(regions)} shown; IDs match the red screenshot labels). "
            "Boxes have no text labels; use visible UI text to choose the right one:"
        ]
        for region_id, region in enumerate(regions, start=1):
            x1, y1, x2, y2 = region["box"]
            cx, cy = region.get("center") or [round((x1 + x2) / 2), round((y1 + y2) / 2)]
            score = region.get("score") or region.get("confidence", 0.0)
            label = f"{region_id}"
            draw.rectangle((x1, y1, x2, y2), outline=(255, 45, 35), width=2)
            text_box = draw.textbbox((x1, y1), label)
            draw.rectangle(text_box, fill=(255, 45, 35))
            draw.text((x1, y1), label, fill=(255, 255, 255))
            lines.append(
                f"#{region_id} center=({cx},{cy}) box=({x1:g},{y1:g},{x2:g},{y2:g}) "
                f"confidence={score:.2f}"
            )

        encoded = io.BytesIO()
        screenshot.save(encoded, format="PNG")
        log_duration("omniparser.overlay_encode", overlay_started, region_count=len(regions))
        return encoded.getvalue(), "\n".join(lines)
