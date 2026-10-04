# Paired wake detector replay review

The Desky checkout predates the wake gate and was not edited. Its Codex agent loaded the exact candidate copied to `/tmp/adam-wake-engine.py` and replayed the identical generated WAV used on the laptop. The temporary WAV was removed after the run. An earlier independently synthesized phrase did not trigger on Desky; this paired WAV replaces that non-comparable result.

- Input: `/tmp/adam-wake-paired.wav`, mono PCM16, 16,000 Hz, 106,128 samples; decoded directly to float32 as `int16 / 32768`.
- Candidate: `/tmp/adam-wake-engine.py`; fresh `WakeWordDetector` instances for each floor and scale.
- Frames: 1,280 samples, with the final 1,208 samples processed as the final partial frame.
- Detector mode: onnx (floor 0.0000), onnx (floor 0.0005).

## Paired replay

| Scale | Energy floor | First score ≥ 0.50 (frame; seconds) | Peak score | Eligible frames (scaled RMS ≥ 0.0005) | Maximum absolute paired score difference on eligible frames |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.0000 | 36 (2.880s) | 0.99308264 | 39 | 0.00000000 |
| 1.00 | 0.0005 | 36 (2.880s) | 0.99308264 | 39 | 0.00000000 |
| 0.10 | 0.0000 | 36 (2.880s) | 0.99312747 | 39 | 0.00000000 |
| 0.10 | 0.0005 | 36 (2.880s) | 0.99312747 | 39 | 0.00000000 |
| 0.05 | 0.0000 | 36 (2.880s) | 0.99318224 | 37 | 0.00000000 |
| 0.05 | 0.0005 | 36 (2.880s) | 0.99318224 | 37 | 0.00000000 |

## Zero-frame CPU timing

- Method: `time.process_time()`; each fresh detector received five 1,280-sample zero-frame warmups, then five trials of 100 zero frames.

| Energy floor | Trial CPU seconds (100 frames each) | Total CPU seconds (500 frames) |
|---:|---|---:|
| 0.0000 | 0.088986, 0.106499, 0.105200, 0.103494, 0.104066 | 0.508244 |
| 0.0005 | 0.000734, 0.000699, 0.000694, 0.000697, 0.000709 | 0.003534 |

## Detector mode by replay scale

| Scale | Floor 0.0000 | Floor 0.0005 |
|---:|---|---|
| 1.00 | onnx | onnx |
| 0.10 | onnx | onnx |
| 0.05 | onnx | onnx |
