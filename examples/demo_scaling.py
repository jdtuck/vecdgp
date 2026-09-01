"""Checks the paper's headline claim: MCMC cost is O(n m^3), i.e. linear in n.

An un-approximated deep GP needs a dense Cholesky of an ``n x n`` matrix at
every likelihood evaluation, so a sweep costs ``O(n^3)``.  With the Vecchia
approximation each sweep instead assembles ``n`` independent ``(m+1)``-sized
Cholesky factors, so the cost should grow linearly in ``n`` for fixed ``m``.

This script times two-layer sweeps at several training sizes on the
Gramacy & Lee (2009) 2-D test function and fits a power law to the timings.

Run::

    python examples/demo_scaling.py
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vecdgp import crps, fit_two_layer, rmse

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (500, 1000, 2000, 4000, 8000)
NMCMC = 60  # short chains: we are timing, not inferring
M = 25  # the paper's default conditioning-set size


def gramacy_lee(u):
    """f(x) = x1 exp(-x1^2 - x2^2) on [-2, 6]^2, with inputs given on [0,1]^2."""
    z = -2.0 + 8.0 * np.asarray(u)
    return z[:, 0] * np.exp(-z[:, 0] ** 2 - z[:, 1] ** 2)


def main():
    rng = np.random.default_rng(0)
    rows = []
    for n in SIZES:
        x = rng.random((n, 2))
        y_raw = gramacy_lee(x)
        y = (y_raw - y_raw.mean()) / y_raw.std()

        t0 = time.time()
        fit = fit_two_layer(x, y, nmcmc=NMCMC, true_g=1e-4, m=M, verb=False,
                            seed=1)
        per_sweep = (time.time() - t0) / (NMCMC - 1)

        xp = rng.random((500, 2))
        yp = (gramacy_lee(xp) - y_raw.mean()) / y_raw.std()
        t0 = time.time()
        p = fit.trim(NMCMC // 2, 5).predict(xp)
        t_pred = time.time() - t0

        rows.append((n, per_sweep, t_pred, rmse(yp, p.mean), crps(yp, p.mean, p.s2)))
        print(
            f"n = {n:6d}   {per_sweep:7.3f} s / MCMC sweep   "
            f"predict {t_pred:6.2f}s   RMSE {rows[-1][3]:.4f}  CRPS {rows[-1][4]:.4f}"
        )

    n_arr = np.array([r[0] for r in rows], float)
    t_arr = np.array([r[1] for r in rows], float)
    slope, intercept = np.polyfit(np.log(n_arr), np.log(t_arr), 1)
    print(f"\nempirical cost ~ n^{slope:.2f}   (theory: n^1.00; dense DGP: n^3)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(n_arr, t_arr, "o-", label=f"vecdgp, m={M}  (fit: n^{slope:.2f})")
    ref = t_arr[0] * (n_arr / n_arr[0])
    ax.loglog(n_arr, ref, "k--", lw=1, label="linear reference")
    ref3 = t_arr[0] * (n_arr / n_arr[0]) ** 3
    ax.loglog(n_arr, ref3, ":", color="C3", lw=1, label="cubic (dense DGP)")
    ax.set_xlabel("n (training size)")
    ax.set_ylabel("seconds per MCMC sweep")
    ax.set_title("Vecchia-approximated two-layer DGP: cost scaling")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "scaling.png"), dpi=130)
    print("wrote scaling.png")


if __name__ == "__main__":
    main()
