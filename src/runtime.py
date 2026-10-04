"""Small runtime defaults for keeping host-side CPU pools bounded."""

from __future__ import annotations

import os
import sys


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_native_threading() -> int:
    """Bound native thread pools while respecting a lower user-set limit.

    ``ADAM_CPU_THREADS`` raises or lowers the default cap (4). Pool variables
    are set before the application imports numeric runtimes, then this function
    can be called again after imports to update already-loaded libraries.
    """
    try:
        limit = max(1, min(256, int(os.environ.get("ADAM_CPU_THREADS", "4"))))
    except (TypeError, ValueError):
        limit = 4

    pool_limits = []
    for name in _THREAD_ENV_VARS:
        try:
            configured = int(os.environ.get(name, str(limit)))
            if configured > 0:
                limit_for_pool = min(limit, configured)
            else:
                limit_for_pool = limit
        except (TypeError, ValueError):
            limit_for_pool = limit
        os.environ[name] = str(limit_for_pool)
        pool_limits.append(limit_for_pool)

    effective_limit = min(pool_limits, default=limit)

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.set_num_threads(effective_limit)
        except (AttributeError, RuntimeError):
            pass
        try:
            torch.set_num_interop_threads(1)
        except (AttributeError, RuntimeError):
            pass

    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        try:
            cv2.setNumThreads(effective_limit)
        except (AttributeError, RuntimeError):
            pass

    return effective_limit
