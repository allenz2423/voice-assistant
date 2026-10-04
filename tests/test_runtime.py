import os
import sys
from types import SimpleNamespace

from src.runtime import configure_native_threading


def test_native_threading_sets_conservative_defaults(monkeypatch):
    monkeypatch.delenv("ADAM_CPU_THREADS", raising=False)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    limit = configure_native_threading()

    assert limit == 4
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "4"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "4"


def test_native_threading_honors_user_limit_and_configures_loaded_pools(monkeypatch):
    torch_calls = []
    cv2_calls = []
    fake_torch = SimpleNamespace(
        set_num_threads=lambda value: torch_calls.append(("intra", value)),
        set_num_interop_threads=lambda value: torch_calls.append(("inter", value)),
    )
    fake_cv2 = SimpleNamespace(setNumThreads=lambda value: cv2_calls.append(value))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setenv("ADAM_CPU_THREADS", "8")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "16")

    limit = configure_native_threading()

    assert limit == 2
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    assert torch_calls == [("intra", 2), ("inter", 1)]
    assert cv2_calls == [2]


def test_native_threading_handles_invalid_overrides(monkeypatch):
    monkeypatch.setenv("ADAM_CPU_THREADS", "invalid")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "invalid")

    limit = configure_native_threading()

    assert limit == 4
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
