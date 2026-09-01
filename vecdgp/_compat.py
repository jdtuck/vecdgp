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
