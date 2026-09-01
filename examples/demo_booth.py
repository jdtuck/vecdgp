"""Reproduces the ``deepgp`` vignette's 1-D example with ``vecchia = TRUE``.

The "booth" function is piecewise: a wiggly trigonometric regime for
``x <= 0.58`` and a straight line after it.  That abrupt change of regime is
exactly the nonstationarity a stationary one-layer GP cannot represent, and
which the deep GP handles by warping the input space.

Run::

    python examples/demo_booth.py

Writes ``booth_fits.png`` and ``booth_latent.png`` next to the script.
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vecdgp import crps, fit_one_layer, fit_three_layer, fit_two_layer, rmse

HERE = os.path.dirname(os.path.abspath(__file__))
NMCMC, BURN, THIN = 4000, 2000, 2
M = 10  # conditioning-set size (n is only 20 here, so m must be < n)


def booth(x):
    x = np.ravel(x)
    return np.where(
        x <= 0.58,
        np.sin(np.pi * x * 6) + np.cos(np.pi * x * 12),
        5 * x - 4.9,
    )


def main():
    # ---- data, on the scale the priors are calibrated for -----------------
    n, np_test = 20, 100
    x = np.linspace(0, 1, n)[:, None]
    y_raw = booth(x)
    mu, sd = y_raw.mean(), y_raw.std()
    y = (y_raw - mu) / sd

    xp = np.linspace(0, 1, np_test)[:, None]
    yp = (booth(xp) - mu) / sd

    fits, preds = {}, {}
    for name, fn, kw in [
        ("1-layer GP", fit_one_layer, {}),
        ("2-layer DGP", fit_two_layer, {}),
        ("3-layer DGP", fit_three_layer, {}),
    ]:
        t0 = time.time()
        fit = fn(x, y, nmcmc=NMCMC, true_g=1e-6, m=M, verb=False, seed=1, **kw)
        fit = fit.trim(BURN, THIN)
        kwargs = {} if name == "1-layer GP" else {"store_latent": True}
        p = fit.predict(xp, **kwargs)
        fits[name], preds[name] = fit, p
        print(
            f"{name:12s}  fit {time.time() - t0:5.1f}s  "
            f"RMSE {rmse(yp, p.mean):.4f}  CRPS {crps(yp, p.mean, p.s2):.4f}"
        )

    # ---- fits -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, fits):
        p = preds[name]
        lo = p.mean - 1.96 * np.sqrt(p.s2)
        hi = p.mean + 1.96 * np.sqrt(p.s2)
        ax.fill_between(xp.ravel(), lo, hi, color="C0", alpha=0.25,
                        label="95% interval")
        ax.plot(xp, yp, "k--", lw=1, label="truth")
        ax.plot(xp, p.mean, "C0", lw=2, label="posterior mean")
        ax.plot(x, y, "ko", ms=4, label="training data")
        ax.set_title(f"{name}  (RMSE {rmse(yp, p.mean):.3f})")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Vecchia-approximated GP / DGP on the piecewise 'booth' function")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "booth_fits.png"), dpi=130)

    # ---- the learned warping ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    # W is identified only up to sign and shift (the outer kernel sees only
    # distances in W-space), so align the draws before plotting them.
    w = preds["2-layer DGP"].w_new[:, :, 0]
    w = w - w.mean(axis=1, keepdims=True)
    flip = np.sign(w @ (xp.ravel() - xp.mean()))
    w = w * np.where(flip == 0, 1.0, flip)[:, None]
    for t in range(0, w.shape[0], max(1, w.shape[0] // 60)):
        axes[0].plot(xp.ravel(), w[t], color="C1", alpha=0.2, lw=1)
    axes[0].plot(xp.ravel(), w.mean(axis=0), color="C3", lw=2)
    axes[0].set_title("latent warping W(x)  (sign/shift aligned)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("W")

    axes[1].plot(fits["2-layer DGP"].ll, lw=0.8)
    axes[1].set_title("outer log-likelihood trace (post burn-in)")
    axes[1].set_xlabel("MCMC iteration")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "booth_latent.png"), dpi=130)
    print("wrote booth_fits.png and booth_latent.png")


if __name__ == "__main__":
    main()
