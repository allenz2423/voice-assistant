"""Single-shot OmniParser YOLOv8 Nano worker; dependencies live in its own venv."""

from __future__ import annotations

import io
import json
import sys
import time

import numpy as np
from PIL import Image
from ultralytics import YOLO


def predict_image(image_bytes: bytes, model: YOLO, device_name: str, confidence: float) -> dict:
    decode_started = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    decode_ms = (time.perf_counter() - decode_started) * 1000
    inference_started = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=1280,
        conf=confidence,
        iou=0.45,
        device=device_name,
        verbose=False,
    )[0]
    inference_ms = (time.perf_counter() - inference_started) * 1000
    postprocess_started = time.perf_counter()
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
    postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
    return {
        "width": width,
        "height": height,
        "device": device_name,
        "regions": regions,
        "timings_ms": {
            "decode": round(decode_ms, 1),
            "inference": round(inference_ms, 1),
            "postprocess": round(postprocess_ms, 1),
        },
    }


def detect(image_bytes: bytes, model_path: str, device_name: str, confidence: float) -> dict:
    model = YOLO(model_path, task="detect")
    return predict_image(image_bytes, model, device_name, confidence)


def run_server(model_path: str, device_name: str, confidence: float) -> int:
    import struct

    try:
        model = YOLO(model_path, task="detect")
        dummy = Image.new("RGB", (64, 64), color="black")
        model.predict(source=dummy, imgsz=64, conf=confidence, device=device_name, verbose=False)
        sys.stdout.write("READY\n")
        sys.stdout.flush()
    except Exception as exc:
        print(f"OmniParser server initialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    stdin = sys.stdin.buffer
    stdout = sys.stdout

    while True:
        header = stdin.read(4)
        if not header or len(header) < 4:
            break
        (length,) = struct.unpack(">I", header)
        if length == 0:
            break
        payload = bytearray()
        while len(payload) < length:
            chunk = stdin.read(length - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) < length:
            break

        try:
            res = predict_image(bytes(payload), model, device_name, confidence)
            stdout.write(json.dumps(res, separators=(",", ":")) + "\n")
            stdout.flush()
        except Exception as exc:
            err = {"error": f"{type(exc).__name__}: {exc}", "regions": []}
            stdout.write(json.dumps(err, separators=(",", ":")) + "\n")
            stdout.flush()
    return 0


def main() -> int:
    args = [arg for arg in sys.argv[1:] if arg != "--server"]
    is_server = "--server" in sys.argv
    if len(args) != 3:
        print("usage: omniparser_infer.py MODEL_PATH DEVICE CONFIDENCE [--server]", file=sys.stderr)
        return 2
    model_path, device_name, confidence = args[0], args[1], float(args[2])
    if is_server:
        return run_server(model_path, device_name, confidence)
    try:
        result = detect(sys.stdin.buffer.read(), model_path, device_name, confidence)
        print(json.dumps(result, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(f"OmniParser inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
