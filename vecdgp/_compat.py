"""Optional-numba shim.

``numba`` compiles the hot loops (building the rows of the sparse Cholesky
factor ``U``, the forward solve, and the quadratic form).  It is strongly
recommended -- it is the Python analogue of the OpenMP-parallel C++ in the
``deepgp`` R package -- but the package remains importable and correct
without it, just slower.
"""

from __future__ import annotations

import os
import warnings

HAVE_NUMBA = False

if os.environ.get("VECDGP_DISABLE_NUMBA", "").lower() not in ("1", "true", "yes"):
    try:  # pragma: no cover - trivial
        from numba import njit as _njit, prange as _prange  # type: ignore

        HAVE_NUMBA = True
    except Exception:  # pragma: no cover
        pass

if HAVE_NUMBA:
    njit = _njit
    prange = _prange
else:  # pragma: no cover - fallback path

    warnings.warn(
        "numba is not available; vecdgp will fall back to pure Python loops, "
        "which are orders of magnitude slower. `pip install numba` is "
        "strongly recommended.",
        RuntimeWarning,
        stacklevel=2,
    )

    def njit(*args, **kwargs):
        """No-op stand-in for :func:`numba.njit`."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(func):
            return func

        return wrap

    prange = range


__all__ = ["njit", "prange", "HAVE_NUMBA"]


# ---------------------------------------------------------------------------
# nested-parallelism guard
# ---------------------------------------------------------------------------
import threading  # noqa: E402

_TLS = threading.local()


class in_worker_thread:
    """Mark a block as running inside vecdgp's own thread pool.

    Numba's ``parallel=True`` kernels must NOT be entered from several Python
    threads at once.  The ``workqueue`` threading layer -- numba's fallback,
    and what a stock macOS install typically gets -- detects this and aborts
    the process outright::

        Numba workqueue threading layer is terminating:
        Concurrent access has been detected.

    (``omp`` and ``tbb`` tolerate it, which is why this never showed up on
    Linux CI.)  Rather than depend on the threading layer, every parallel
    kernel here has a serial twin, and this flag selects it.  Draw-level
    parallelism then provides all the concurrency, with no nesting at all.
    """

    def __enter__(self):
        self.prev = getattr(_TLS, "worker", False)
        _TLS.worker = True
        return self

    def __exit__(self, *exc):
        _TLS.worker = self.prev
        return False


def in_worker():
    """True when the caller is inside :class:`in_worker_thread`."""
    return getattr(_TLS, "worker", False)


__all__ += ["in_worker", "in_worker_thread"]
