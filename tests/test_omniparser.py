import io
import json
import struct
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.config import ComputerVisionConfig
from src.tools.omniparser import OmniParserScreenshotGrounder


def sample_png_bytes(width: int = 200, height: int = 150) -> bytes:
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_omniparser_disabled_returns_empty():
    grounder = OmniParserScreenshotGrounder(enabled=False)
    image = sample_png_bytes()
    annotated, text = grounder.annotate(image)
    assert annotated == image
    assert text == ""
    assert grounder._process is None


def test_omniparser_is_lazy_by_default():
    assert ComputerVisionConfig().preload_on_startup is False
    grounder = OmniParserScreenshotGrounder(enabled=True)
    assert grounder._process is None


def test_omniparser_mock_worker_lifecycle():
    fake_boxes = [
        {"box": [10.0, 20.0, 50.0, 60.0], "confidence": 0.85},
        {"box": [70.0, 80.0, 120.0, 110.0], "confidence": 0.92},
    ]

    class FakeProc:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(b"READY\n" + json.dumps({"regions": fake_boxes}).encode("utf-8") + b"\n")
            self.stderr = io.BytesIO()
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.terminated or self.killed else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            return 0

    fake_proc = FakeProc()

    with patch("subprocess.Popen", return_value=fake_proc), \
         patch.object(Path, "is_file", return_value=True):
        grounder = OmniParserScreenshotGrounder(
            enabled=True,
            device="cuda",
            gpu_uuid="GPU-TEST-UUID",
            python_path="/fake/bin/python",
            model_path="/fake/models/yolov8n.pt",
            preload=True,
        )

        assert grounder._process is fake_proc

        # Run inference
        image = sample_png_bytes()
        annotated, state_text = grounder.annotate(image)

        assert annotated != image
        assert "#1 center=" in state_text
        assert "#2 center=" in state_text

        # Verify stdin received the length header + image bytes
        fake_proc.stdin.seek(0)
        header = fake_proc.stdin.read(4)
        length = struct.unpack(">I", header)[0]
        assert length == len(image)
        payload = fake_proc.stdin.read(length)
        assert payload == image

        # Clean shutdown
        grounder.close()
        assert grounder._process is None
        assert fake_proc.terminated or fake_proc.killed


def test_omniparser_worker_handles_restart_on_failure():
    # Simulate first process dying and second process succeeding
    fake_boxes = [{"box": [5.0, 5.0, 25.0, 25.0], "confidence": 0.8}]

    class DyingProc:
        def __init__(self):
            self.stdin = MagicMock()
            self.stdin.write.side_effect = BrokenPipeError("Worker disconnected")
            self.stdout = io.BytesIO(b"READY\n")
            self.stderr = io.BytesIO()

        def poll(self):
            return 1

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 1

    class RecoveryProc:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(b"READY\n" + json.dumps({"regions": fake_boxes}).encode("utf-8") + b"\n")
            self.stderr = io.BytesIO()

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    procs = [DyingProc(), RecoveryProc()]

    with patch("subprocess.Popen", side_effect=procs), \
         patch.object(Path, "is_file", return_value=True):
        grounder = OmniParserScreenshotGrounder(
            enabled=True,
            device="cpu",
            gpu_uuid="",
            python_path="/fake/bin/python",
            model_path="/fake/models/yolov8n.pt",
        )

        image = sample_png_bytes()
        annotated, state_text = grounder.annotate(image)
        assert "#1 center=" in state_text
        grounder.close()


def test_omniparser_real_worker_if_installed():
    import os

    python_path = Path("~/.local/share/adam/omniparser-runtime/bin/python").expanduser()
    model_path = Path("~/.local/share/adam/models/omniparser-yolov8n.pt").expanduser()
    if not python_path.is_file() or not model_path.is_file():
        pytest.skip("OmniParser runtime or model not installed.")

    # GTX 1080 Ti UUID
    gpu_uuid = "GPU-1816d860-68b3-1b29-2ffd-c3d34d9e0673"
    cuda_env = os.environ.copy()
    cuda_env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    cuda_probe = subprocess.run(
        [str(python_path), "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True,
        text=True,
        env=cuda_env,
        timeout=30,
    )
    if cuda_probe.returncode != 0 or cuda_probe.stdout.strip() != "True":
        pytest.skip("OmniParser CUDA runtime is unavailable on this host.")

    grounder = OmniParserScreenshotGrounder(
        enabled=True,
        device="cuda",
        gpu_uuid=gpu_uuid,
        python_path=str(python_path),
        model_path=str(model_path),
        preload=True,
    )
    try:
        assert grounder._process is not None
        assert grounder._process.poll() is None

        image = sample_png_bytes(640, 480)
        annotated, text = grounder.annotate(image)
        assert isinstance(annotated, bytes)
        assert isinstance(text, str)
    finally:
        grounder.close()
        assert grounder._process is None
