"""Posterior prediction, averaged over the MCMC draws.

For each retained draw ``t`` the latent layers are mapped forward to the test
locations and the outer layer is kriged, giving ``mu^(t)`` and either
point-wise ``s2^(t)`` or a full ``Sigma^(t)``.  Draws are then combined by the
law of total variance,

.. math::

   \\bar\\mu = \\frac1T \\sum_t \\mu^{(t)}, \\qquad
   \\bar\\Sigma = \\frac1T \\sum_t \\Sigma^{(t)} + \\widehat{\\mathrm{Cov}}_t(\\mu^{(t)}),

exactly as in ``deepgp:::predict_*_vec``.

``mean_map=True`` propagates the latent layers through their *conditional
mean*; ``mean_map=False`` draws a full sample from the latent predictive
distribution instead (slower, wider intervals).
"""

from __future__ import annotations

import numpy as np

from .krig import krig_vec
from .vecchia import EPS

__all__ = ["PredictResult", "predict_shallow_vec", "predict_deep_vec"]


class PredictResult(dict):
    """Dict of prediction outputs with attribute access."""

    __getattr__ = dict.__getitem__

    def __repr__(self):  # pragma: no cover - cosmetic
        keys = ", ".join(sorted(self.keys()))
        return f"PredictResult({keys})"


def _as2d(x):
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else np.ascontiguousarray(x)


def _combine(mu_t, s2_sum, sigma_sum, nmcmc, lite, return_all, s2_t=None):
    out = PredictResult()
    out["mean"] = mu_t.mean(axis=0)
    if lite:
        extra = np.var(mu_t, axis=0, ddof=1) if nmcmc > 1 else 0.0
        out["s2"] = s2_sum / nmcmc + extra
        if return_all:
            out["mean_all"] = mu_t
            out["s2_all"] = s2_t
    else:
        extra = np.cov(mu_t, rowvar=False) if nmcmc > 1 else 0.0
        out["Sigma"] = sigma_sum / nmcmc + extra
    return out


def _g_at(g, t):
    return float(g) if np.ndim(g) == 0 else float(g[t])


# ---------------------------------------------------------------------------
# one layer
# ---------------------------------------------------------------------------
def predict_shallow_vec(obj, x_new, m=None, lite=True, order_new=None,
                        return_all=False, rng=None, samples_only=False, nper=1):
    rng = np.random.default_rng() if rng is None else rng
    x_new = _as2d(x_new)
    if x_new.shape[1] != obj.x.shape[1]:
        raise ValueError("dimension of x_new does not match dimension of x")
    n_new = x_new.shape[0]
    n = obj.x.shape[0]
    sep = obj.settings.sep
    if samples_only:
        lite = False  # joint sampling needs the stacked ordering

    if m is None:
        m = min(n, 2 * obj.x_approx.m) if lite else min(n + n_new - 1, 2 * obj.x_approx.m)

    ap = obj.x_approx
    ap.clean_pred()
    ap.add_pred(x_new, m, lite=lite, order_new=order_new, rng=rng)

    samples = np.empty((nper * obj.nmcmc, n_new)) if samples_only else None
    mu_t = np.empty((obj.nmcmc, n_new))
    s2_sum = np.zeros(n_new)
    sigma_sum = np.zeros((n_new, n_new))
    s2_t = np.empty((obj.nmcmc, n_new)) if (lite and return_all) else None

    for t in range(obj.nmcmc):
        k = krig_vec(obj.y, ap, tau2=obj.tau2[t], theta=obj.theta[t],
                     g=_g_at(obj.g, t), v=obj.v, sep=sep,
                     s2=lite and not samples_only,
                     sigma=(not lite) and not samples_only,
                     nsamples=nper if samples_only else 0, rng=rng)
        if samples_only:
            samples[t * nper : (t + 1) * nper] = k["samples"]
            continue
        mu_t[t] = k["mean"]
        if lite:
            s2_sum += k["s2"]
            if return_all:
                s2_t[t] = k["s2"]
        else:
            sigma_sum += k["sigma"]

    if samples_only:
        return samples
    return _combine(mu_t, s2_sum, sigma_sum, obj.nmcmc, lite, return_all, s2_t)


# ---------------------------------------------------------------------------
# two and three layers
# ---------------------------------------------------------------------------
def predict_deep_vec(obj, x_new, m=None, lite=True, mean_map=True,
                     store_latent=False, order_new=None, return_all=False,
                     layers=2, rng=None, samples_only=False, nper=1):
    rng = np.random.default_rng() if rng is None else rng
    x_new = _as2d(x_new)
    if x_new.shape[1] != obj.x.shape[1]:
        raise ValueError("dimension of x_new does not match dimension of x")
    n_new = x_new.shape[0]
    n = len(obj.y)
    D = obj.w.shape[2]
    if samples_only:
        lite = False  # joint sampling needs the stacked ordering

    if m is None:
        m = min(n, 2 * obj.w_approx.m) if lite else min(n + n_new - 1, 2 * obj.w_approx.m)

    w_prior_mean = obj.x if obj.settings.pmx else None

    # x -> (z ->) w mapping approximation
    obj.x_approx.clean_pred()
    obj.x_approx.add_pred(x_new, min(n, m) if mean_map else m,
                          lite=mean_map, order_new=order_new, rng=rng)

    samples = np.empty((nper * obj.nmcmc, n_new)) if samples_only else None
    mu_t = np.empty((obj.nmcmc, n_new))
    s2_sum = np.zeros(n_new)
    sigma_sum = np.zeros((n_new, n_new))
    s2_t = np.empty((obj.nmcmc, n_new)) if (lite and return_all) else None
    w_new_store = np.empty((obj.nmcmc, n_new, D)) if store_latent else None
    z_new_store = (
        np.empty((obj.nmcmc, n_new, D)) if (store_latent and layers == 3) else None
    )

    for t in range(obj.nmcmc):
        g_t = _g_at(obj.g, t)

        if layers == 3:
            z_t = obj.z[t]
            z_new = np.empty((n_new, D))
            for i in range(D):
                k = krig_vec(z_t[:, i], obj.x_approx, tau2=obj.settings.tau2_z,
                             theta=obj.theta_z[t, i], g=EPS, v=obj.v,
                             nsamples=0 if mean_map else 1, rng=rng)
                z_new[:, i] = k["mean"] if mean_map else k["samples"][0]
            obj.z_approx.clean_pred()
            obj.z_approx.set_coords(z_t)
            obj.z_approx.add_pred(z_new, m, lite=mean_map,
                                  order_new=order_new, rng=rng)
            if store_latent:
                z_new_store[t] = z_new

            w_t = obj.w[t]
            w_new = np.empty((n_new, D))
            for i in range(D):
                k = krig_vec(w_t[:, i], obj.z_approx, tau2=obj.settings.tau2_w,
                             theta=obj.theta_w[t, i], g=EPS, v=obj.v,
                             nsamples=0 if mean_map else 1, rng=rng)
                w_new[:, i] = k["mean"] if mean_map else k["samples"][0]
        else:
            w_t = obj.w[t]
            w_new = np.empty((n_new, D))
            for i in range(D):
                pm = w_prior_mean[:, i] if w_prior_mean is not None else 0.0
                pmn = x_new[:, i] if w_prior_mean is not None else 0.0
                k = krig_vec(w_t[:, i], obj.x_approx, tau2=obj.settings.tau2_w,
                             theta=obj.theta_w[t, i], g=EPS, v=obj.v,
                             nsamples=0 if mean_map else 1,
                             prior_mean=pm, prior_mean_new=pmn, rng=rng)
                w_new[:, i] = k["mean"] if mean_map else k["samples"][0]

        obj.w_approx.clean_pred()
        obj.w_approx.set_coords(w_t)
        obj.w_approx.add_pred(w_new, m, lite=lite, order_new=order_new, rng=rng)
        if store_latent:
            w_new_store[t] = w_new

        k = krig_vec(obj.y, obj.w_approx, tau2=obj.tau2_y[t],
                     theta=obj.theta_y[t], g=g_t, v=obj.v,
                     s2=lite and not samples_only,
                     sigma=(not lite) and not samples_only,
                     nsamples=nper if samples_only else 0, rng=rng)
        if samples_only:
            samples[t * nper : (t + 1) * nper] = k["samples"]
            continue
        mu_t[t] = k["mean"]
        if lite:
            s2_sum += k["s2"]
            if return_all:
                s2_t[t] = k["s2"]
        else:
            sigma_sum += k["sigma"]

    if samples_only:
        return samples
    out = _combine(mu_t, s2_sum, sigma_sum, obj.nmcmc, lite, return_all, s2_t)
    if store_latent:
        out["w_new"] = w_new_store
        if layers == 3:
            out["z_new"] = z_new_store
    return out
