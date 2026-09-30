"""Single-shot OmniParser YOLOv8 Nano worker; dependencies live in its own venv."""

from __future__ import annotations

import io
import json
import sys

import numpy as np
from PIL import Image
from ultralytics import YOLO


def detect(image_bytes: bytes, model_path: str, device_name: str, confidence: float) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    model = YOLO(model_path, task="detect")
    result = model.predict(
        source=image,
        imgsz=1280,
        conf=confidence,
        iou=0.45,
        device=device_name,
        verbose=False,
    )[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty((0,))

    regions = []
    for index in np.argsort(scores)[::-1]:
        x1, y1, x2, y2 = map(float, boxes[index])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue
        if (x2 - x1) * (y2 - y1) > width * height * 0.60:
            continue
        regions.append({
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center": [round((x1 + x2) / 2), round((y1 + y2) / 2)],
            "score": round(float(scores[index]), 3),
        })
    return {"width": width, "height": height, "device": device_name, "regions": regions}


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: omniparser_infer.py MODEL_PATH DEVICE CONFIDENCE", file=sys.stderr)
        return 2
    try:
        result = detect(sys.stdin.buffer.read(), sys.argv[1], sys.argv[2], float(sys.argv[3]))
        print(json.dumps(result, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(f"OmniParser inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
