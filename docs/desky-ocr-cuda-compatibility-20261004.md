# CUDA OCR benchmark and compatibility retest

Run date: 2026-10-04. No real screenshots or credentials were accessed. All OCR inputs were generated in memory.

## Diagnosis

The prior environment reported ONNX Runtime 1.30.0, CUDA 13.4 runtime, and a GTX 1080 Ti (compute capability 6.1). Its detector session failed during `cublasCreate` with `CUBLAS failure 8: the function requires an architectural feature absent from the device`. The RTX 3070 (8.6) succeeded.

This aligns with NVIDIA's CUDA 13.0 release notes: CUDA 13 removed library support for Maxwell, Pascal, and Volta, and cuBLAS 13 specifically removed support for compute capabilities earlier than Turing. ONNX Runtime's package table says 1.30.x PyPI GPU builds use CUDA 13.0. The 1080 Ti is Pascal/6.1, so the failure is an expected architecture incompatibility in the CUDA 13 stack, not a GPU UUID selection problem. [ONNX Runtime CUDA package matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

The CUDA 12 retest also exposed a separate cuDNN constraint. The first isolated install selected cuDNN 9.27.0.42; CUDA/cuBLAS initialized on the 1080 Ti, but its first convolution failed with `err 209 no kernel image is available for execution on the device` / `CUDNN_FE failure 11`. NVIDIA's cuDNN 9.27 CUDA 12 support matrix no longer lists Pascal. After pinning cuDNN 9.10.1.4, whose CUDA 12.x support matrix includes compute capability 6.1 and Pascal, the same OCR workload succeeded on both GPUs. [cuDNN 9.27 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.27.0/reference/support-matrix.html), [cuDNN 9.10.1 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.1/reference/support-matrix.html).

Conclusion: the prior cuBLAS initialization failure is consistent with the CUDA 13 removal of Pascal support. A CUDA 12 ONNX Runtime build can run this OCR workload on the 1080 Ti when paired with a cuDNN release that still supports Pascal. A broad `cudnn~=9.0` dependency can resolve to a newer cuDNN that no longer supports that GPU, so the CUDA/cuDNN build versions both matter.

## Isolated CUDA 12.x retest

The environment was created under `isolated/venv` (Python 3.13.14) without changing the existing project virtualenv, installed system packages, or repository source. Used official `onnxruntime-gpu==1.26.0` with its CUDA 12.x wheel, plus RapidOCR 3.9.2 and psutil 7.2.2. ORT build info: commit `8c546c37b4`, Release; `ort.cuda_version` reports 12.8. Installed NVIDIA runtime packages: CUDA runtime 12.9.79, cuBLAS 12.9.2.10, cuDNN 9.10.1.4, cuFFT 11.4.1.4, cuRAND 10.3.10.19, NVRTC and nvJitLink 12.9.86. CUDA 12.8 ORT needs CUDA 12.8 or newer, so these bundled CUDA 12.9 runtime libraries meet the documented minor-version requirement. [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

The detector (`PP-OCRv6_det_small.onnx`), recognizer (`PP-OCRv6_rec_small.onnx`), and classifier (`ch_ppocr_mobile_v2.0_cls_mobile.onnx`) were copied into the isolated environment. Their SHA-256 hashes match the corresponding small-model files in the prior environment:

- detector: `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f`
- recognizer: `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884`
- classifier: `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c`

The earlier report did not preserve its fixture-generation script or image bytes. The retest recreated an in-memory 1280×720 synthetic UI with the same 20 labels, using the existing `src/tools/ocr.py` path and same three model files. Thus model weights match exactly; fixture appearance may differ. Runs were separate and sequential, with `CUDA_VISIBLE_DEVICES` restricted to the GPU under test. Each device's memory was sampled with `nvidia-smi` immediately before and after its process; readings include other processes.

| Device | Result | CUDA EP session providers (detector/classifier/recognizer) | Cold read | Five warm reads (ms; median) | Recall / returned regions | Process RSS / USS | GPU memory before → after |
|---|---|---|---:|---|---|---:|---:|
| GTX 1080 Ti, CC 6.1 | **Success** | CUDA + CPU fallback / CUDA + CPU fallback / CUDA + CPU fallback | 4758.3 ms | 3395.2, 3507.5, 3179.9, 3210.8, 3323.9 (3323.9 ms) | 20/20; 21 regions | 1036.4 / 999.6 MiB | 492 → 492 MiB |
| RTX 3070, CC 8.6 | **Success** | CUDA + CPU fallback / CUDA + CPU fallback / CUDA + CPU fallback | 3142.5 ms | 1808.3, 1786.3, 1829.7, 1819.9, 1786.1 (1808.3 ms) | 20/20; 21 regions | 1196.1 / 1161.3 MiB | 1013 → 979 MiB |

Each model session's provider list included `CUDAExecutionProvider` and `CPUExecutionProvider`; the OCR wrapper verifies CUDA is present for all three. The measured GPU memory is whole-device `nvidia-smi` usage, not a per-process allocation. Timings include Python OCR decode/preprocessing and pipeline overhead; cold includes session/model setup. No CUDA event profiling or repeated-run statistics were collected.

## CPU control

Ran the copied `src/tools/ocr.py` with `ScreenOCR(device="cpu", model_size="small")` in both environments. Each run generated the same 45,389-byte 1280×720 PNG in memory using the same fixture code as `isolated/benchmark.py`, then performed one cold read followed by five warm reads. The detector, recognizer, and classifier hashes match the CUDA runs and project model files listed above. No files were installed or changed in the project virtualenv or system Python.

| Environment | Active providers (detector / classifier / recognizer) | Cold | Five warm reads (ms; median) | Recall / regions | RSS / USS after run | Process CPU % of one core (cold; warm 1–5) | Host CPU % during reads (cold; warm 1–5) |
|---|---|---:|---|---|---:|---|---|
| Isolated ORT 1.26.0 | CPU / CPU / CPU | 787.5 ms | 451.2, 449.4, 454.6, 441.7, 449.6 (449.6 ms) | 20/20; 21 | 152.6 / 132.0 MiB | 454.6%; 217.2%, 218.1%, 217.8%, 217.3%, 218.0% | 21.8%; 10.9%, 9.5%, 11.2%, 9.7%, 10.5% |
| Project ORT 1.30.0 | CPU / CPU / CPU | 811.7 ms | 444.9, 430.5, 433.2, 469.4, 441.4 (441.4 ms) | 20/20; 21 | 178.2 / 118.2 MiB | 478.0%; 215.8%, 218.3%, 219.3%, 215.2%, 217.5% | 20.2%; 11.2%, 8.5%, 8.2%, 12.0%, 9.2% |

Process CPU is elapsed process CPU time divided by each read's wall time (so 200% means about two cores); host CPU is the sampled whole-host utilization over each read interval. The copied wrapper caps ORT inference threads at two. `onnxruntime.get_available_providers()` also listed CUDA and TensorRT in these GPU-capable installations, but all three created OCR sessions reported only `CPUExecutionProvider`; no CUDA inference was requested or active.

On this one fixture, CPU warm medians were 7.4× faster than the CUDA 12 GTX 1080 Ti median (3,323.9 ms) and 4.1× faster than the CUDA 12 RTX 3070 median (1,808.3 ms). The CPU cold reads were also shorter than those CUDA cold reads. This workload therefore does not show an overall performance win for the older CUDA stack; CUDA 12's demonstrated benefit here is Pascal compatibility. These are single six-read samples, and they do not establish comparative performance across other images or application workloads.

Exact CPU run commands, from `/tmp/adam-cuda-ocr-bench`:

```bash
PYTHONPATH=/tmp/adam-cuda-ocr-bench BENCH_ENV=isolated-ort-1.26.0 isolated/venv/bin/python -u cpu_control.py | tee logs/cpu-isolated-run.txt
PYTHONPATH=/tmp/adam-cuda-ocr-bench BENCH_ENV=project-ort-1.30.0 /home/incoming/voice-assistant/.venv/bin/python -u cpu_control.py | tee logs/cpu-project-run.txt
```

`cpu_control.py` is scratch-only and uses the same fixture generation as `isolated/benchmark.py`; the `src/tools/ocr.py` imported by `PYTHONPATH` is the scratch copy. The first command used existing isolated packages without installing anything; the second used the pre-existing project environment read-only. Full stdout, including ORT/RapidOCR versions, provider lists, per-read timing and CPU data, is in `logs/cpu-isolated-run.txt` and `logs/cpu-project-run.txt`.

### Initial loader and cuDNN diagnostic failures

The first run with CUDA 12/cuDNN 9.27 also initially needed `LD_LIBRARY_PATH` pointed at the isolated venv's NVIDIA library directories; before setting it, ORT reported `libcublasLt.so.12: cannot open shared object file`. This was fixed only in the process environment, without changing system paths or libraries. With that path set, the remaining 1080 Ti failure was the cuDNN 9.27 no-kernel-image error described above. Pinning `nvidia-cudnn-cu12==9.10.1.4` resolved it. The final table records only successful runs under that pinned configuration.

## Reproducible commands

Run from `/tmp/adam-cuda-ocr-bench`. The file `isolated/benchmark.py` builds the synthetic fixture in memory, runs one cold and five warm reads, checks label recall, reports session providers and RSS/USS, and raises on OCR/CUDA failure. Model files are sourced read-only from the prior environment and copied into the isolated environment.

```bash
/home/incoming/voice-assistant/.venv/bin/python -m venv isolated/venv
isolated/venv/bin/python -m pip install 'onnxruntime-gpu[cuda,cudnn]==1.26.0' 'rapidocr==3.9.2' psutil
isolated/venv/bin/python -m pip install --no-deps --force-reinstall 'nvidia-cudnn-cu12==9.10.1.4'
mkdir -p isolated/models
cp /home/incoming/voice-assistant/.venv/lib/python3.13/site-packages/rapidocr/models/PP-OCRv6_det_small.onnx isolated/models/
cp /home/incoming/voice-assistant/.venv/lib/python3.13/site-packages/rapidocr/models/PP-OCRv6_rec_small.onnx isolated/models/
cp /home/incoming/voice-assistant/.venv/lib/python3.13/site-packages/rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx isolated/models/
cp isolated/models/*.onnx isolated/venv/lib/python3.13/site-packages/rapidocr/models/
LIBS=/tmp/adam-cuda-ocr-bench/isolated/venv/lib/python3.13/site-packages/nvidia
export LD_LIBRARY_PATH="$LIBS/cublas/lib:$LIBS/cuda_nvrtc/lib:$LIBS/cuda_runtime/lib:$LIBS/cudnn/lib:$LIBS/cufft/lib:$LIBS/curand/lib:$LIBS/nvjitlink/lib"
# Run one command at a time; wait for process exit before starting the other GPU.
CUDA_VISIBLE_DEVICES=GPU-1816d860-68b3-1b29-2ffd-c3d34d9e0673 BENCH_GPU_UUID=GPU-1816d860-68b3-1b29-2ffd-c3d34d9e0673 PYTHONPATH=/tmp/adam-cuda-ocr-bench isolated/venv/bin/python -u isolated/benchmark.py
CUDA_VISIBLE_DEVICES=GPU-e96c0d9d-4f51-3a0f-004b-9c85eee0d4d2 BENCH_GPU_UUID=GPU-e96c0d9d-4f51-3a0f-004b-9c85eee0d4d2 PYTHONPATH=/tmp/adam-cuda-ocr-bench isolated/venv/bin/python -u isolated/benchmark.py
```

## Limits and original baseline

These are single-process runs on one generated fixture, not repeated-run performance measurements or representative application usage. The exact prior fixture was unavailable, so only its dimensions, labels, and model weights were reproduced. GPU memory readings include unrelated device usage. Provider lists confirm CUDA provider registration/use at session level but do not identify placement of each individual ONNX node.

The original benchmark report's RTX 3070 run under ONNX Runtime 1.30.0 succeeded with 20/20 recall (cold 1641.8 ms; five warm reads 245.5, 249.8, 275.6, 313.2, 311.2 ms; RSS/USS 1468.8/1322.0 MiB). Its GTX 1080 Ti run failed before inference at `cublasCreate` with the CUBLAS failure 8 architecture message; therefore it had no OCR timings or Python memory result. Original host report: driver 580.178.04, CUDA runtime 13.4, GTX 1080 Ti CC 6.1, RTX 3070 CC 8.6. The isolated CUDA 12 results above provide the controlled compatibility retest.
