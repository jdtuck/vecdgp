"""Gibbs schemes for one-, two- and three-layer Vecchia-approximated GPs.

These reproduce ``deepgp:::gibbs_*_vec``.  One sweep of the two-layer sampler
(Algorithm 1 of Sauer, Cooper & Gramacy 2023) is:

1. Metropolis-Hastings for the nugget ``g`` (skipped if ``true_g`` is given),
   using the outer likelihood ``L(Y | W)``.
2. Metropolis-Hastings for the outer lengthscale ``theta_y``.
3. Metropolis-Hastings for each inner lengthscale ``theta_w[k]``, using the
   inner likelihood ``L(W_k | X)``.
4. Elliptical slice sampling of the latent layer ``W`` given ``Y``.

Every likelihood evaluation goes through the Vecchia approximation, so the
per-sweep cost is ``O(n m^3)`` rather than ``O(n^3)``.
"""

from __future__ import annotations

import time

import numpy as np

from .mcmc import logl_vec, sample_g_vec, sample_theta_vec, sample_w_vec, sample_z_vec
from .vecchia import EPS, create_approx

__all__ = [
    "gibbs_one_layer_vec",
    "gibbs_two_layer_vec",
    "gibbs_three_layer_vec",
    "init_latent",
]


def init_latent(x, D):
    """Identity-mapping initialisation of a latent layer.

    Matches ``matrix(x, nrow = n, ncol = D)`` in R: the columns of ``x`` are
    recycled column-wise until ``D`` columns are filled.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.shape[0] == 1 and x.size != 1:
        x = x.T
    n = x.shape[0]
    flat = x.reshape(-1, order="F")
    return np.resize(flat, n * D).reshape(n, D, order="F").copy()


class _Progress:
    """Progress reporter with an ETA.

    A fit at n in the thousands runs for hours, so a bare iteration counter
    is not much use; this reports the measured sweep rate and a projected
    finish.  It also surfaces the thread count on the first tick, because a
    numba install that ends up single-threaded is the most common cause of a
    fit being unexpectedly slow (cost is very close to linear in cores).
    """

    def __init__(self, verb, nmcmc, every=100):
        self.verb = verb
        self.nmcmc = nmcmc
        self.every = every
        self.t0 = time.time()
        self.announced = False

    def __call__(self, j):
        if not self.verb or j % self.every != 0:
            return
        if not self.announced:
            self.announced = True
            try:
                from numba import get_num_threads

                nt = get_num_threads()
                note = f" on {nt} thread{'s' if nt != 1 else ''}"
                if nt == 1:
                    note += "  (cost is ~linear in cores -- see README)"
            except Exception:
                note = " without numba (much slower -- see README)"
            print(f"  mcmc{note}", flush=True)
        el = time.time() - self.t0
        rate = el / j
        left = rate * (self.nmcmc - j)
        print(
            f"  mcmc {j}/{self.nmcmc}  {rate:.3f} s/sweep  "
            f"elapsed {_hms(el)}  eta {_hms(left)}",
            flush=True,
        )


def _hms(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# one layer
# ---------------------------------------------------------------------------
def gibbs_one_layer_vec(x, y, nmcmc, verb, initial, true_g, settings, v, m,
                        order=None, rng=None, x_approx=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.atleast_2d(x)
    d = x.shape[1]
    sep = settings.sep

    if x_approx is None:
        x_approx = create_approx(x, m, order, rng=rng)

    est_g = true_g is None
    g = initial["g"] if est_g else true_g
    g_store = np.empty(nmcmc) if est_g else None
    if est_g:
        g_store[0] = g

    if sep:
        theta = np.empty((nmcmc, d))
        theta[0] = initial["theta"]
    else:
        theta = np.empty(nmcmc)
        theta[0] = np.atleast_1d(initial["theta"])[0]
    tau2 = np.full(nmcmc, np.nan)
    ll_store = np.full(nmcmc, np.nan)
    ll = None

    report = _Progress(verb, nmcmc)
    for j in range(1, nmcmc):
        report(j)
        tau2[j] = tau2[j - 1]  # carry forward until an acceptance updates it

        if est_g:
            g, ll, t2 = sample_g_vec(
                y, x_approx, theta=theta[j - 1], g=g, v=v,
                alpha=settings.g_alpha, beta=settings.g_beta,
                l=settings.l, u=settings.u, ll_prev=ll, sep=sep, rng=rng,
            )
            g_store[j] = g
            if t2 is not None:
                tau2[j] = t2

        if sep:
            theta[j] = theta[j - 1]
            for i in range(d):
                th, ll, t2 = sample_theta_vec(
                    y, x_approx, tau2=1.0, theta=theta[j], g=g, v=v,
                    alpha=settings.theta_alpha, beta=settings.theta_beta,
                    l=settings.l, u=settings.u, outer=True, ll_prev=ll,
                    sep=True, index=i, rng=rng,
                )
                theta[j, i] = th
                if t2 is not None:
                    tau2[j] = t2
        else:
            th, ll, t2 = sample_theta_vec(
                y, x_approx, tau2=1.0, theta=theta[j - 1], g=g, v=v,
                alpha=settings.theta_alpha, beta=settings.theta_beta,
                l=settings.l, u=settings.u, outer=True, ll_prev=ll, rng=rng,
            )
            theta[j] = th
            if t2 is not None:
                tau2[j] = t2
        ll_store[j] = ll

    return dict(g=g_store if est_g else g, theta=theta, tau2=tau2,
                x_approx=x_approx, ll=ll_store)


# ---------------------------------------------------------------------------
# two layers
# ---------------------------------------------------------------------------
def gibbs_two_layer_vec(x, y, nmcmc, D, verb, initial, true_g, settings, v, m,
                        order=None, rng=None, x_approx=None, w_approx=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.atleast_2d(x)
    n = len(y)

    est_g = true_g is None
    g = initial["g"] if est_g else true_g
    g_store = np.empty(nmcmc) if est_g else None
    if est_g:
        g_store[0] = g

    tau2_y = np.full(nmcmc, np.nan)
    theta_y = np.empty(nmcmc)
    theta_y[0] = initial["theta_y"]
    theta_w = np.empty((nmcmc, D))
    theta_w[0] = initial["theta_w"]
    w = np.empty((nmcmc, n, D))
    w[0] = initial["w"]
    ll_store = np.full(nmcmc, np.nan)
    ll_outer = None

    # a single random ordering shared by both layers (paper, Sec. 3.3)
    if order is None:
        order = rng.permutation(n)
    if x_approx is None:
        x_approx = create_approx(x, m, order, rng=rng)
    if w_approx is None:
        w_approx = create_approx(w[0], m, order, rng=rng)

    prior_mean = x if settings.pmx else None

    report = _Progress(verb, nmcmc)
    for j in range(1, nmcmc):
        report(j)

        if est_g:
            g, ll_outer, _ = sample_g_vec(
                y, w_approx, theta=theta_y[j - 1], g=g, v=v,
                alpha=settings.g_alpha, beta=settings.g_beta,
                l=settings.l, u=settings.u, ll_prev=ll_outer, rng=rng,
            )
            g_store[j] = g

        theta_y[j], ll_outer, _ = sample_theta_vec(
            y, w_approx, tau2=1.0, theta=theta_y[j - 1], g=g, v=v,
            alpha=settings.theta_y_alpha, beta=settings.theta_y_beta,
            l=settings.l, u=settings.u, outer=True, ll_prev=ll_outer, rng=rng,
        )

        # inner lengthscales: likelihood must be recomputed, W has changed
        for i in range(D):
            pm = x[:, i] if settings.pmx else 0.0
            theta_w[j, i], _, _ = sample_theta_vec(
                w[j - 1, :, i], x_approx, tau2=settings.tau2_w,
                theta=theta_w[j - 1, i], g=EPS, v=v,
                alpha=settings.theta_w_alpha, beta=settings.theta_w_beta,
                l=settings.l, u=settings.u, outer=False, ll_prev=None,
                prior_mean=pm, rng=rng,
            )

        w_approx, ll_outer, t2y = sample_w_vec(
            y, w_approx, x_approx, tau2_w=settings.tau2_w,
            theta_y=theta_y[j], theta_w=theta_w[j], g=g, v=v,
            ll_prev=ll_outer, prior_mean=prior_mean, rng=rng,
        )
        w[j] = w_approx.x_ord[w_approx.rev_ord_obs]
        ll_store[j] = ll_outer
        tau2_y[j] = t2y

    return dict(g=g_store if est_g else g, tau2_y=tau2_y, theta_y=theta_y,
                theta_w=theta_w, w=w, w_approx=w_approx, x_approx=x_approx,
                ll=ll_store)


# ---------------------------------------------------------------------------
# three layers
# ---------------------------------------------------------------------------
def gibbs_three_layer_vec(x, y, nmcmc, D, verb, initial, true_g, settings, v, m,
                          order=None, rng=None, x_approx=None, z_approx=None,
                          w_approx=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.atleast_2d(x)
    n = len(y)

    est_g = true_g is None
    g = initial["g"] if est_g else true_g
    g_store = np.empty(nmcmc) if est_g else None
    if est_g:
        g_store[0] = g

    tau2_y = np.full(nmcmc, np.nan)
    theta_y = np.empty(nmcmc)
    theta_y[0] = initial["theta_y"]
    theta_w = np.empty((nmcmc, D))
    theta_w[0] = initial["theta_w"]
    theta_z = np.empty((nmcmc, D))
    theta_z[0] = initial["theta_z"]
    w = np.empty((nmcmc, n, D))
    w[0] = initial["w"]
    z = np.empty((nmcmc, n, D))
    z[0] = initial["z"]
    ll_store = np.full(nmcmc, np.nan)
    ll_outer = None

    if order is None:
        order = rng.permutation(n)
    if x_approx is None:
        x_approx = create_approx(x, m, order, rng=rng)
    if z_approx is None:
        z_approx = create_approx(z[0], m, order, rng=rng)
    if w_approx is None:
        w_approx = create_approx(w[0], m, order, rng=rng)

    report = _Progress(verb, nmcmc)
    for j in range(1, nmcmc):
        report(j)

        if est_g:
            g, ll_outer, _ = sample_g_vec(
                y, w_approx, theta=theta_y[j - 1], g=g, v=v,
                alpha=settings.g_alpha, beta=settings.g_beta,
                l=settings.l, u=settings.u, ll_prev=ll_outer, rng=rng,
            )
            g_store[j] = g

        theta_y[j], ll_outer, _ = sample_theta_vec(
            y, w_approx, tau2=1.0, theta=theta_y[j - 1], g=g, v=v,
            alpha=settings.theta_y_alpha, beta=settings.theta_y_beta,
            l=settings.l, u=settings.u, outer=True, ll_prev=ll_outer, rng=rng,
        )

        ll_mid = 0.0  # recomputed each sweep: Z has changed
        for i in range(D):
            theta_w[j, i], ll_i, _ = sample_theta_vec(
                w[j - 1, :, i], z_approx, tau2=settings.tau2_w,
                theta=theta_w[j - 1, i], g=EPS, v=v,
                alpha=settings.theta_w_alpha, beta=settings.theta_w_beta,
                l=settings.l, u=settings.u, outer=False, rng=rng,
            )
            ll_mid += ll_i

        for i in range(D):
            theta_z[j, i], _, _ = sample_theta_vec(
                z[j - 1, :, i], x_approx, tau2=settings.tau2_z,
                theta=theta_z[j - 1, i], g=EPS, v=v,
                alpha=settings.theta_z_alpha, beta=settings.theta_z_beta,
                l=settings.l, u=settings.u, outer=False, rng=rng,
            )

        z_approx, _ = sample_z_vec(
            w[j - 1], z_approx, x_approx, tau2_w=settings.tau2_w,
            tau2_z=settings.tau2_z, theta_w=theta_w[j], theta_z=theta_z[j],
            v=v, ll_prev=ll_mid, rng=rng,
        )
        z[j] = z_approx.x_ord[z_approx.rev_ord_obs]

        w_approx, ll_outer, t2y = sample_w_vec(
            y, w_approx, z_approx, tau2_w=settings.tau2_w,
            theta_y=theta_y[j], theta_w=theta_w[j], g=g, v=v,
            ll_prev=ll_outer, rng=rng,
        )
        w[j] = w_approx.x_ord[w_approx.rev_ord_obs]
        ll_store[j] = ll_outer
        tau2_y[j] = t2y

    return dict(g=g_store if est_g else g, tau2_y=tau2_y, theta_y=theta_y,
                theta_w=theta_w, theta_z=theta_z, w=w, z=z,
                w_approx=w_approx, z_approx=z_approx, x_approx=x_approx,
                ll=ll_store)
