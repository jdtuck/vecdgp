"""Does hoisting the per-row allocations out of u_entries pay?

v0  shipped: allocates pts, cov, out per row, plus an extra vector inside
    the back-substitution -- 4 heap allocations x n rows per call.
v1  back-substitution written straight into `out` (drops one allocation and
    one pass over the row).
v2  v1 plus thread-local scratch buffers allocated once per call.
"""
import sys
import time

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

sys.path.insert(0, "/home/claude/vecdgp")
from vecdgp.kernels import fill_cov_sym
from vecdgp.vecchia import _chol_lower, create_approx, create_U_values


@njit(cache=True, inline="always")
def _backsub_inplace(cov, n0, out):
    """M = R^-1 e_last, written reversed into `out` with no scratch vector.

    out[k] corresponds to M[last-k]; the recursion for M[i] only needs M[j]
    with j > i, i.e. out entries with a *smaller* index, so a single forward
    pass over `out` suffices.
    """
    last = n0 - 1
    out[0] = 1.0 / cov[last, last]
    for k in range(1, n0):
        mi = last - k
        s = 0.0
        for j in range(mi + 1, n0):
            s += cov[j, mi] * out[last - j]
        out[k] = -s / cov[mi, mi]


@njit(cache=True, parallel=True)
def u_entries_v1(x_ord, NN, NN_len, tau2, theta, g, v, sep):
    n, d = x_ord.shape
    mp1 = NN.shape[1]
    Uvals = np.zeros((n, mp1))
    for i in prange(n):
        n0 = NN_len[i]
        pts = np.empty((n0, d))
        for j in range(n0):
            idx = NN[i, n0 - 1 - j]
            for k in range(d):
                pts[j, k] = x_ord[idx, k]
        cov = np.empty((n0, n0))
        out = np.empty(n0)
        fill_cov_sym(cov, pts, n0, tau2, theta, g, v, sep)
        _chol_lower(cov, n0)
        _backsub_inplace(cov, n0, out)
        for k in range(n0):
            Uvals[i, k] = out[k]
    return Uvals


@njit(cache=True, parallel=True)
def u_entries_v2(x_ord, NN, NN_len, tau2, theta, g, v, sep):
    n, d = x_ord.shape
    mp1 = NN.shape[1]
    nt = get_num_threads()
    Uvals = np.zeros((n, mp1))
    pts_buf = np.empty((nt, mp1, d))
    cov_buf = np.empty((nt, mp1, mp1))
    out_buf = np.empty((nt, mp1))
    for i in prange(n):
        t = get_thread_id()
        n0 = NN_len[i]
        pts = pts_buf[t, :n0]
        cov = cov_buf[t, :n0, :n0]
        out = out_buf[t, :n0]
        for j in range(n0):
            idx = NN[i, n0 - 1 - j]
            for k in range(d):
                pts[j, k] = x_ord[idx, k]
        fill_cov_sym(cov, pts, n0, tau2, theta, g, v, sep)
        _chol_lower(cov, n0)
        _backsub_inplace(cov, n0, out)
        for k in range(n0):
            Uvals[i, k] = out[k]
    return Uvals


def timeit(fn, reps=7):
    best = 1e30
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{'n':>7} {'v0 shipped':>11} {'v1 no scratch':>14} {'v2 +per-thread':>15} {'max|diff|':>11}")
    for n in (2000, 6000, 20000):
        x = rng.random((n, 2))
        ap = create_approx(x, 25, rng=rng)
        args = (ap.x_ord, ap.NN, ap.NN_len, 1.0, np.array([0.1]), 1e-6, 2.5, False)
        a = create_U_values(ap, 1.0, 0.1, 1e-6, 2.5)
        b = u_entries_v1(*args)
        c = u_entries_v2(*args)
        t0 = timeit(lambda: create_U_values(ap, 1.0, 0.1, 1e-6, 2.5))
        t1 = timeit(lambda: u_entries_v1(*args))
        t2 = timeit(lambda: u_entries_v2(*args))
        d = max(np.abs(a - b).max(), np.abs(a - c).max())
        print(f"{n:7d} {t0*1e3:9.1f}ms {t1*1e3:12.1f}ms {t2*1e3:13.1f}ms {d:11.2e}")
