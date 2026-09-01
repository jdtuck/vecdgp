"""Predictive scoring rules (ports of ``deepgp``'s ``rmse``/``crps``/``score``)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["rmse", "crps", "score"]


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
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sign, ldet = np.linalg.slogdet(sigma)
    resid = y - mu
    quad = resid @ np.linalg.solve(sigma, resid)
    return float((-ldet - quad) / len(y))
