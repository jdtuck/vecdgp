"""Vecchia-approximated posterior prediction (kriging).

Section 3.4 of Sauer, Cooper & Gramacy (2023) offers two flavours:

**Point-wise ("lite").**  Each test location gets its own conditioning set of
``m`` nearest training neighbours (in the *warped* space ``W^{(t)}`` for a DGP,
recomputed at every MCMC draw).  The predictive distribution is the univariate

.. math:: Y_i \\mid Y, W \\sim N(\\mu_i(W), \\sigma_i^2(W)),

obtained from a single ``(m+1) x (m+1)`` Cholesky.  Embarrassingly parallel and
preferred for dense test sets.

**Joint.**  Test locations are appended to the ordering and a stacked sparse
factor is formed,

.. math::

   U_{stack} = \\begin{bmatrix} U_w & U_{w,*} \\\\ 0 & U_* \\end{bmatrix},

from which partitioned-inverse identities give

.. math::

   \\mu_* = -(U_*^\\top)^{-1} U_{w,*}^\\top Y, \\qquad
   \\Sigma_* = (U_* U_*^\\top)^{-1}.

A third mode draws sequential posterior samples one test location at a time,
conditioning on previously drawn values (used by ``post_sample``).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import spsolve_triangular

from ._compat import in_worker, njit, prange
from .kernels import _theta_vec, fill_cov_sym
from .vecchia import EPS, _chol_lower, create_U_sparse

__all__ = ["krig_vec"]


# ---------------------------------------------------------------------------
# point-wise ("lite") prediction
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _krig_lite_row(i, x_ord, x_new, NN_new, yo, tau2, theta, g, v, sep, want_s2,
                   mu, s2):
    """One test location.  Shared by the parallel and serial shells."""
    d = x_new.shape[1]
    m = NN_new.shape[1]
    n0 = m + 1
    pts = np.empty((n0, d))
    for j in range(m):
        idx = NN_new[i, j]
        for k in range(d):
            pts[j, k] = x_ord[idx, k]
    for k in range(d):
        pts[m, k] = x_new[i, k]

    K = np.empty((n0, n0))
    fill_cov_sym(K, pts, n0, 1.0, theta, g, v, sep)
    if _chol_lower(K, n0) != 0:
        trace = 0.0
        fill_cov_sym(K, pts, n0, 1.0, theta, g, v, sep)
        for k in range(n0):
            trace += K[k, k]
        jit = 1e-10 * trace / n0
        for _ in range(10):
            fill_cov_sym(K, pts, n0, 1.0, theta, g, v, sep)
            for k in range(n0):
                K[k, k] += jit
            if _chol_lower(K, n0) == 0:
                break
            jit *= 10.0

    # a = L[:m, :m]^{-1} y[NN]
    a = np.empty(m)
    for r in range(m):
        acc = yo[NN_new[i, r]]
        for c in range(r):
            acc -= K[r, c] * a[c]
        a[r] = acc / K[r, r]
    s = 0.0
    for r in range(m):
        s += K[m, r] * a[r]
    mu[i] = s
    if want_s2:
        s2[i] = tau2 * K[m, m] * K[m, m]


@njit(cache=True, parallel=True, nogil=True)
def _krig_lite(x_ord, x_new, NN_new, yo, tau2, theta, g, v, sep, want_s2):
    n_new = x_new.shape[0]
    mu = np.zeros(n_new)
    s2 = np.zeros(n_new)
    for i in prange(n_new):
        _krig_lite_row(i, x_ord, x_new, NN_new, yo, tau2, theta, g, v,
                       sep, want_s2, mu, s2)
    return mu, s2


@njit(cache=True, nogil=True)
def _krig_lite_serial(x_ord, x_new, NN_new, yo, tau2, theta, g, v, sep,
                      want_s2):
    """Serial twin of :func:`_krig_lite` -- see :mod:`vecdgp._compat`."""
    n_new = x_new.shape[0]
    mu = np.zeros(n_new)
    s2 = np.zeros(n_new)
    for i in range(n_new):
        _krig_lite_row(i, x_ord, x_new, NN_new, yo, tau2, theta, g, v,
                       sep, want_s2, mu, s2)
    return mu, s2


# ---------------------------------------------------------------------------
# sequential posterior sampling
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _krig_samples(x_ord, NN, NN_len, yo, z_norm, tau2, theta, g, v, sep, n_obs):
    """Sequential draws at the appended predictive locations.

    ``z_norm`` has shape ``(nsamples, n_new)`` and holds standard normals so
    that randomness stays with the caller's generator.
    """
    n_total = x_ord.shape[0]
    d = x_ord.shape[1]
    n_new = n_total - n_obs
    nsamples = z_norm.shape[0]
    out = np.zeros((nsamples, n_new))
    # zeros, not empty: slots n_obs.. are filled in as each location is drawn,
    # so a conditioning set that named a not-yet-drawn point would otherwise
    # read uninitialised memory -- silently zero on Linux, arbitrary on
    # Windows.  find_ordered_nn_appended guarantees that cannot happen; this
    # makes the consequence deterministic if that guarantee ever slips.
    work = np.zeros(n_total)
    for k in range(n_obs):
        work[k] = yo[k]

    for i in range(n_new):
        row = n_obs + i
        n0 = NN_len[row]
        ncond = n0 - 1
        pts = np.empty((n0, d))
        for j in range(n0):
            idx = NN[row, n0 - 1 - j]  # conditioning set first, target last
            for c in range(d):
                pts[j, c] = x_ord[idx, c]
        K = np.empty((n0, n0))
        fill_cov_sym(K, pts, n0, 1.0, theta, g, v, sep)
        if _chol_lower(K, n0) != 0:
            trace = 0.0
            fill_cov_sym(K, pts, n0, 1.0, theta, g, v, sep)
            for c in range(n0):
                trace += K[c, c]
            jit = 1e-10 * trace / n0
            ok = False
            for _ in range(12):
                fill_cov_sym(K, pts, n0, 1.0, theta, g, v, sep)
                for c in range(n0):
                    K[c, c] += jit
                if _chol_lower(K, n0) == 0:
                    ok = True
                    break
                jit *= 10.0
            if not ok:
                raise ValueError(
                    "post_sample: conditioning block not positive definite "
                    "even with jitter; check theta/tau2/g for non-finite "
                    "values"
                )
        sd = np.sqrt(tau2) * K[ncond, ncond]

        for s in range(nsamples):
            # refresh the working response with this chain's earlier draws
            for t in range(i):
                work[n_obs + t] = out[s, t]
            a = np.empty(ncond)
            for r in range(ncond):
                acc = work[NN[row, n0 - 1 - r]]
                for c in range(r):
                    acc -= K[r, c] * a[c]
                a[r] = acc / K[r, r]
            mean = 0.0
            for r in range(ncond):
                mean += K[ncond, r] * a[r]
            out[s, i] = mean + sd * z_norm[s, i]
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _check_finite(theta, tau2, g):
    """Reject non-finite hyperparameters at the boundary.

    ``tau2`` never enters the Cholesky -- it only scales the draw -- so a NaN
    there would sail past the factorisation's own guard and come back as a
    NaN sample.  Three scalars against a call that costs hundreds of
    microseconds; the check is free and the alternative is silent corruption.
    """
    if not np.all(np.isfinite(theta)):
        raise ValueError("non-finite lengthscale (theta) passed to prediction")
    if not np.isfinite(tau2) or tau2 < 0.0:
        raise ValueError("tau2 must be finite and non-negative")
    if not np.isfinite(g) or g < 0.0:
        raise ValueError("nugget (g) must be finite and non-negative")


def krig_vec(y, approx, tau2=1.0, theta=0.1, g=0.0, v=2.5, sep=False,
             s2=False, sigma=False, nsamples=0, prior_mean=0.0,
             prior_mean_new=0.0, rng=None):
    """Posterior predictive moments (or samples) at the stored test locations.

    Returns a dict with keys among ``mean``, ``s2``, ``sigma``, ``samples``.
    """
    y = np.asarray(y, dtype=np.float64)
    th = _theta_vec(theta, approx.x_ord.shape[1], sep)
    _check_finite(th, tau2, g)
    out = {}

    pm = np.asarray(prior_mean, dtype=np.float64)
    yo = y[approx.order] - (pm if pm.ndim == 0 else pm[approx.order])
    yo = np.ascontiguousarray(yo)

    if nsamples > 0:
        if approx.order_new is None:
            raise ValueError("posterior samples require an approx built with lite=False")
        rng = np.random.default_rng() if rng is None else rng
        n_new = approx.x_ord.shape[0] - approx.n_obs
        z_norm = rng.standard_normal((nsamples, n_new))
        samples = _krig_samples(
            approx.x_ord, approx.NN, approx.NN_len, yo, z_norm,
            float(tau2), th, float(g), float(v), bool(sep), approx.n_obs,
        )
        samples = samples[:, approx.rev_ord_new] + prior_mean_new
        out["samples"] = samples
        return out

    lite = approx.order_new is None
    if lite and sigma:
        raise ValueError("approx is lite; a full covariance is unavailable")
    if (not lite) and s2:
        raise ValueError("approx is not lite; request sigma instead of s2")

    if lite:
        kernel = _krig_lite_serial if in_worker() else _krig_lite
        mu, s2v = kernel(
            approx.x_ord, np.ascontiguousarray(approx.x_new), approx.NN_new,
            yo, float(tau2), th, float(g), float(v), bool(sep), bool(s2),
        )
        out["mean"] = mu + prior_mean_new
        if s2:
            out["s2"] = s2v
        return out

    # joint prediction via the stacked sparse factor
    U = create_U_sparse(approx, tau2=1.0, theta=th, g=g, v=v, sep=sep).tocsr()
    obs = approx.observed
    Upp = U[~obs][:, ~obs].tocsr()
    Uop = U[obs][:, ~obs].tocsr()
    idx = approx.rev_ord_new

    # mean: solve Upp^T mu = -(Uop^T y). Upp^T is sparse lower triangular.
    UopTy = Uop.T @ yo
    mu_ordered = -spsolve_triangular(Upp.T.tocsr(), UopTy, lower=True)
    out["mean"] = prior_mean_new + np.asarray(mu_ordered).ravel()[idx]

    if sigma:
        # Sigma* = (U* U*^T)^{-1} = Upp^{-T} Upp^{-1}
        n_new = Upp.shape[0]
        Upp_inv = solve_triangular(Upp.toarray(), np.eye(n_new), lower=False)
        Winv = Upp_inv.T @ Upp_inv
        out["sigma"] = tau2 * Winv[np.ix_(idx, idx)]
    return out
