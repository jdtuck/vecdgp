"""How much of the numba-vs-C++ gap is C++, and how much is sloppy numba?

Three variants of the same u_entries computation:
  v0  the shipped version            (generic kernel, full square covariance)
  v1  + branch hoisted out of the inner loop (kernel specialised per v)
  v2  + symmetry (lower triangle only)  == what the C++ 'sym' column does
"""
import time
import numpy as np
from numba import njit, prange

import sys
sys.path.insert(0, "/home/claude/vecdgp")
from vecdgp.vecchia import create_approx, create_U_values, u_entries


@njit(cache=True, inline="always")
def _chol(A, n):
    for j in range(n):
        s = A[j, j]
        for k in range(j):
            s -= A[j, k] * A[j, k]
        A[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            t = A[i, j]
            for k in range(j):
                t -= A[i, k] * A[j, k]
            A[i, j] = t / A[j, j]


@njit(cache=True, inline="always")
def _backsub(cov, n0, out, mvec):
    last = n0 - 1
    mvec[last] = 1.0 / cov[last, last]
    for k in range(last - 1, -1, -1):
        s = 0.0
        for j in range(k + 1, n0):
            s += cov[j, k] * mvec[j]
        mvec[k] = -s / cov[k, k]
    for k in range(n0):
        out[k] = mvec[last - k]


@njit(cache=True, parallel=True)
def u_entries_v1(x_ord, NN, NN_len, theta, g):
    """Matern 5/2, isotropic: the v/sep branch never enters the inner loop."""
    n, d = x_ord.shape
    mp1 = NN.shape[1]
    U = np.zeros((n, mp1))
    for i in prange(n):
        n0 = NN_len[i]
        pts = np.empty((n0, d))
        for j in range(n0):
            idx = NN[i, n0 - 1 - j]
            for k in range(d):
                pts[j, k] = x_ord[idx, k]
        cov = np.empty((n0, n0))
        for a in range(n0):
            for b in range(n0):
                r2 = 0.0
                for k in range(d):
                    diff = pts[a, k] - pts[b, k]
                    r2 += diff * diff
                r2 = 5.0 * r2 / theta
                s = np.sqrt(r2)
                cov[a, b] = (1.0 + s + r2 / 3.0) * np.exp(-s)
            cov[a, a] += g
        _chol(cov, n0)
        out = np.empty(n0)
        _backsub(cov, n0, out, np.empty(n0))
        for k in range(n0):
            U[i, k] = out[k]
    return U


@njit(cache=True, parallel=True)
def u_entries_v2(x_ord, NN, NN_len, theta, g):
    """As v1 but only the lower triangle of the covariance is evaluated."""
    n, d = x_ord.shape
    mp1 = NN.shape[1]
    U = np.zeros((n, mp1))
    for i in prange(n):
        n0 = NN_len[i]
        pts = np.empty((n0, d))
        for j in range(n0):
            idx = NN[i, n0 - 1 - j]
            for k in range(d):
                pts[j, k] = x_ord[idx, k]
        cov = np.empty((n0, n0))
        for a in range(n0):
            for b in range(a + 1):
                r2 = 0.0
                for k in range(d):
                    diff = pts[a, k] - pts[b, k]
                    r2 += diff * diff
                r2 = 5.0 * r2 / theta
                s = np.sqrt(r2)
                cov[a, b] = (1.0 + s + r2 / 3.0) * np.exp(-s)
            cov[a, a] += g
        _chol(cov, n0)
        out = np.empty(n0)
        _backsub(cov, n0, out, np.empty(n0))
        for k in range(n0):
            U[i, k] = out[k]
    return U


def timeit(fn, reps):
    best = 1e30
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    THETA, G = 0.1, 1e-6
    print(f"{'n':>7} {'v0 shipped':>11} {'v1 hoisted':>11} {'v2 +symm':>10}  {'max|diff|':>10}")
    for n in (1000, 5000, 20000, 50000):
        x = rng.random((n, 2))
        ap = create_approx(x, 25, rng=rng)
        a = create_U_values(ap, 1.0, THETA, G, 2.5)
        b = u_entries_v1(ap.x_ord, ap.NN, ap.NN_len, THETA, G)
        c = u_entries_v2(ap.x_ord, ap.NN, ap.NN_len, THETA, G)
        reps = max(3, min(20, int(2e5 / n)))
        t0 = timeit(lambda: create_U_values(ap, 1.0, THETA, G, 2.5), reps)
        t1 = timeit(lambda: u_entries_v1(ap.x_ord, ap.NN, ap.NN_len, THETA, G), reps)
        t2 = timeit(lambda: u_entries_v2(ap.x_ord, ap.NN, ap.NN_len, THETA, G), reps)
        print(f"{n:7d} {t0*1e3:9.1f}ms {t1*1e3:9.1f}ms {t2*1e3:8.1f}ms  "
              f"{max(np.abs(a-b).max(), np.abs(a-c).max()):10.2e}")
