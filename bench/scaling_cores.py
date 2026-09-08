"""Measure how vecdgp scales with cores *on your machine*.

There are two independent parallel axes and they behave differently:

* **Fitting** can only use numba's ``prange`` over the rows of ``U``, because
  MCMC sweeps are a Markov chain.  That parallelises almost perfectly.
* **Prediction** has a second, coarser axis: MCMC draws are independent, so
  they can be spread over a thread pool.  This matters because a large share
  of prediction is scipy KD-tree work and sequential sampling that numba's
  ``prange`` cannot reach at all.

Run::

    python bench/scaling_cores.py            # defaults: n=6000, d=2, m=25
    python bench/scaling_cores.py 20000 3    # n=20000, d=3
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from contextlib import contextmanager  # noqa: E402

from vecdgp import fit_two_layer  # noqa: E402
from vecdgp.predict import resolve_cores  # noqa: E402


@contextmanager
def pinned(c):
    """Hold the whole process to `c` cores, numba included.

    Without this, `cores=1` would still let numba's prange use every thread,
    and the prediction ladder would report only the *incremental* gain from
    draw-parallelism rather than the end-to-end effect of core count.
    """
    try:
        from numba import get_num_threads, set_num_threads
    except Exception:
        yield
        return
    prev = get_num_threads()
    try:
        set_num_threads(max(1, c))
        yield
    finally:
        set_num_threads(prev)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
D = int(sys.argv[2]) if len(sys.argv) > 2 else 2
M = int(sys.argv[3]) if len(sys.argv) > 3 else 25
N_TEST = 500
SWEEPS = 11


def core_ladder():
    top = resolve_cores(None)
    ladder, c = [], 1
    while c < top:
        ladder.append(c)
        c *= 2
    ladder.append(top)
    return ladder


def main():
    try:
        import numba

        print(f"numba {numba.__version__}, max threads {numba.config.NUMBA_NUM_THREADS}")
    except Exception:
        print("numba NOT INSTALLED -- everything below will be very slow")
    print(f"os.cpu_count() = {os.cpu_count()}   usable = {resolve_cores(None)}")
    print(f"n = {N}, d = {D}, m = {M}, {N_TEST} test locations\n")

    rng = np.random.default_rng(0)
    x = rng.random((N, D))
    y = rng.standard_normal(N)
    xp = rng.random((N_TEST, D))
    ladder = core_ladder()

    # ---- fitting: numba prange only --------------------------------------
    print("FIT  (numba prange over rows of U; sweeps are inherently serial)")
    print(f"  {'cores':>6} {'s/sweep':>10} {'speedup':>9} {'nmcmc=10000':>13}")
    base = None
    for c in ladder:
        t0 = time.time()
        fit = fit_two_layer(x, y, nmcmc=SWEEPS, true_g=1e-4, m=M, verb=False,
                            seed=1, cores=c)
        el = (time.time() - t0) / (SWEEPS - 1)
        base = base or el
        hrs = el * 10000 / 3600
        print(f"  {c:>6} {el:9.3f}s {base/el:8.2f}x {hrs:11.2f} h")

    # ---- prediction: draws spread over a thread pool ---------------------
    fit = fit_two_layer(x, y, nmcmc=41, true_g=1e-4, m=M, verb=False,
                        seed=1).trim(21)
    print(f"\nPREDICT  ({fit.nmcmc} retained draws; draws are independent)")
    for label, fn in [
        ("predict(lite=True)", lambda c: fit.predict(xp, cores=c)),
        ("post_sample", lambda c: fit.post_sample(
            xp, rng=np.random.default_rng(1), cores=c)),
    ]:
        print(f"  {label}")
        print(f"    {'cores':>6} {'seconds':>10} {'speedup':>9} {'ms/draw':>10}")
        base = None
        for c in ladder:
            with pinned(c):
                fn(c)  # warm the JIT
                t0 = time.time()
                fn(c)
                el = time.time() - t0
            base = base or el
            print(f"    {c:>6} {el:9.2f}s {base/el:8.2f}x "
                  f"{el/fit.nmcmc*1e3:9.1f}")

    print("\nIf the FIT table barely improves, numba is not using your cores:")
    print("  check NUMBA_NUM_THREADS and `python -m vecdgp.diagnose`.")


if __name__ == "__main__":
    main()
