"""Posterior sample paths, and why pointwise intervals are not enough.

``predict`` gives you a mean and a variance at each location independently.
That is all you need to draw an error band -- but it cannot answer any
question about the surface *as a whole*, because such questions depend on how
the uncertainty at one location is correlated with the next.

``post_sample`` draws whole functions from the posterior, so any functional of
the surface becomes a histogram over draws.  This script contrasts the two on
"where is the maximum?", a question the pointwise summary literally cannot
express.

Run::

    python examples/demo_post_sample.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vecdgp import fit_two_layer

HERE = os.path.dirname(os.path.abspath(__file__))


def booth(x):
    x = np.ravel(x)
    return np.where(
        x <= 0.58,
        np.sin(np.pi * x * 6) + np.cos(np.pi * x * 12),
        5 * x - 4.9,
    )


def main():
    # deliberately sparse, so there is real uncertainty to propagate
    n = 12
    x = np.linspace(0, 1, n)[:, None]
    y_raw = booth(x)
    mu, sd = y_raw.mean(), y_raw.std()
    y = (y_raw - mu) / sd
    xp = np.linspace(0, 1, 200)[:, None]
    truth = (booth(xp) - mu) / sd

    fit = fit_two_layer(x, y, nmcmc=6000, true_g=1e-6, m=10, verb=False, seed=0)
    fit = fit.trim(3000, 3)

    pred = fit.predict(xp)
    paths = fit.post_sample(xp, rng=np.random.default_rng(1))
    print(f"drew {paths.shape[0]} paths at {paths.shape[1]} locations")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # ---- paths inside the pointwise band ---------------------------------
    ax = axes[0]
    lo = pred.mean - 1.96 * np.sqrt(pred.s2)
    hi = pred.mean + 1.96 * np.sqrt(pred.s2)
    ax.fill_between(xp.ravel(), lo, hi, color="C0", alpha=0.18,
                    label="95% pointwise band")
    for p in paths[:: max(1, len(paths) // 40)]:
        ax.plot(xp.ravel(), p, color="C1", alpha=0.35, lw=0.8)
    ax.plot(xp, pred.mean, "C0", lw=2, label="posterior mean")
    ax.plot(xp, truth, "k--", lw=1, label="truth")
    ax.plot(x, y, "ko", ms=5, label="training data")
    ax.plot([], [], color="C1", lw=0.8, label="posterior draws")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Each path is a whole function, not a stack of margins")
    ax.legend(fontsize=8, loc="lower right")

    # ---- a functional the band cannot give you ---------------------------
    ax = axes[1]
    argmax_draws = xp.ravel()[paths.argmax(axis=1)]
    ax.hist(argmax_draws, bins=np.linspace(0, 1, 61), color="C1",
            alpha=0.75, label="posterior of argmax\n(from the draws)")
    ax.axvline(xp.ravel()[pred.mean.argmax()], color="C0", lw=2,
               label="argmax of the posterior mean\n(all the band can tell you)")
    ax.axvline(xp.ravel()[truth.argmax()], color="k", ls="--", lw=1.5,
               label="truth")
    ax.set_xlabel("location of the maximum")
    ax.set_ylabel("posterior draws")
    ax.set_title("Where is the maximum?")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "post_sample.png"), dpi=130)

    # the numbers behind the right-hand panel
    modes, counts = np.unique(np.round(argmax_draws, 2), return_counts=True)
    top = np.argsort(counts)[::-1][:3]
    print("\nposterior over the argmax location:")
    for i in top:
        print(f"  x = {modes[i]:.2f}   {100 * counts[i] / len(argmax_draws):5.1f}% "
              f"of draws")
    print(f"\nargmax of the posterior mean alone: "
          f"x = {xp.ravel()[pred.mean.argmax()]:.2f}  (a single number, with no "
          f"uncertainty attached)")
    print(f"truth: x = {xp.ravel()[truth.argmax()]:.2f}")
    print("\nwrote post_sample.png")


if __name__ == "__main__":
    main()
