"""Predictive scoring rules (ports of ``deepgp``'s ``rmse``/``crps``/``score``)."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import norm

__all__ = ["rmse", "crps", "score", "safe_cholesky"]


def safe_cholesky(a, name="matrix"):
    """Lower Cholesky of a symmetric positive-definite matrix, with jitter.

    Predictive covariances from a Vecchia GP are SPD in exact arithmetic but
    can lose definiteness numerically when the nugget is tiny and the kernel
    smooth.  Escalating jitter recovers a usable factor and reports how much
    was needed, following the same policy as the ``U``-entry construction.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be square")
    a = 0.5 * (a + a.T)  # kill any asymmetry from accumulated round-off
    try:
        return np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        pass
    jitter = 1e-10 * np.trace(a) / a.shape[0]
    for _ in range(10):
        try:
            L = np.linalg.cholesky(a + jitter * np.eye(a.shape[0]))
            warnings.warn(
                f"{name} was not numerically positive definite; "
                f"added jitter {jitter:.2e} to factorise it. Consider a "
                f"larger nugget (`true_g`) or a smaller `m`.",
                RuntimeWarning,
                stacklevel=3,
            )
            return L
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError(f"{name} is not positive definite")


def rmse(y, mu):
    """Root mean square error (lower is better)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return float(np.sqrt(np.mean((y - mu) ** 2)))


def crps(y, mu, s2):
    """Continuous ranked probability score (lower is better)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.sqrt(np.asarray(s2, dtype=float))
    z = (y - mu) / sigma
    return float(
        np.mean(
            sigma * (-(1.0 / np.sqrt(np.pi)) + 2.0 * norm.pdf(z)
                     + z * (2.0 * norm.cdf(z) - 1.0))
        )
    )


def score(y, mu, sigma):
    """Log-score, proportional to the MVN log-likelihood (higher is better).

    Requires the full predictive covariance (``lite=False``).

    Uses a Cholesky factorisation rather than ``slogdet`` + ``solve``.  The
    predictive covariance is symmetric positive definite, so Cholesky is both
    the faster and the numerically appropriate route: a general LU
    factorisation of a smooth, small-nugget covariance can hit a tiny or zero
    pivot and return ``-inf``/``nan`` (with divide-by-zero, overflow and
    invalid-value warnings) even when the matrix is perfectly usable.  The
    determinant itself routinely underflows -- ``det`` of a 50x50 predictive
    covariance is around 1e-92 -- which is exactly why only the *log*
    determinant should ever be formed.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    resid = y - mu
    L = safe_cholesky(sigma, name="predictive covariance")
    ldet = 2.0 * np.log(np.diag(L)).sum()
    a = solve_triangular(L, resid, lower=True)
    return float((-ldet - a @ a) / len(y))
