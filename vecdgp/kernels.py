"""Covariance kernels.

Parameterisation follows Sauer, Cooper & Gramacy (2023), *Vecchia-approximated
Deep Gaussian Processes for Computer Experiments* (arXiv:2204.02904) and the
``deepgp`` R package:

.. math::

    \\Sigma(x_i, x_j) = \\tau^2 \\left( k(\\|x_i - x_j\\|^2 / \\theta)
                                       + g \\mathbb{I}_{i=j} \\right)

Note that ``theta`` divides *squared* distances.

``v`` selects the kernel:

======  =========================================
``v``   kernel
======  =========================================
0.5     Matern nu = 1/2 (exponential)
1.5     Matern nu = 3/2
2.5     Matern nu = 5/2 (default)
999     squared exponential ("exp2")
======  =========================================

The scalar-kernel evaluations are written as ``numba``-compatible free
functions so that they can be inlined into the hot Vecchia loops in
:mod:`vecdgp.vecchia`.
"""

from __future__ import annotations

import numpy as np

from ._compat import njit, prange

EXP2 = 999.0
"""Sentinel value of ``v`` selecting the squared-exponential kernel."""

__all__ = [
    "EXP2", "sq_dist", "cov_matrix", "cross_cov", "kernel_from_d2",
    "fill_cov", "fill_cov_sym",
]


# ---------------------------------------------------------------------------
# scalar kernel, jit-compatible
# ---------------------------------------------------------------------------
@njit(cache=True, inline="always")
def kernel_from_d2(r, v):
    """Correlation from an *already scaled* squared distance ``r``.

    ``r`` must equal ``sum_k (x1_k - x2_k)^2 / theta_k`` (times the Matern
    constant 3 or 5 where applicable -- see :func:`_scaled_d2`).
    """
    if v == 999.0:  # squared exponential
        return np.exp(-r)
    s = np.sqrt(r)
    if v == 0.5:
        return np.exp(-s)
    if v == 1.5:
        return (1.0 + s) * np.exp(-s)
    # v == 2.5
    return (1.0 + s + r / 3.0) * np.exp(-s)


@njit(cache=True)
def _fill_d2(out, x1, x2, theta, v, sep, lower_only):
    """Fill ``out`` with theta-scaled squared distances.

    Includes the Matern constant (3 for nu=3/2, 5 for nu=5/2), so the result
    feeds straight into :func:`_apply_kernel`.  The ``sep`` branch is taken
    once, *outside* the loops -- keeping it out of the innermost loop is worth
    roughly a factor of two, since these loops are the hot path of the whole
    package.
    """
    c = 1.0
    if v == 1.5:
        c = 3.0
    elif v == 2.5:
        c = 5.0
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    d = x1.shape[1]
    if sep:
        for i in range(n1):
            jmax = i + 1 if lower_only else n2
            for j in range(jmax):
                r = 0.0
                for k in range(d):
                    diff = x1[i, k] - x2[j, k]
                    r += diff * diff / theta[k]
                out[i, j] = c * r
    else:
        th = theta[0]
        for i in range(n1):
            jmax = i + 1 if lower_only else n2
            for j in range(jmax):
                r = 0.0
                for k in range(d):
                    diff = x1[i, k] - x2[j, k]
                    r += diff * diff
                out[i, j] = c * r / th
    return out


@njit(cache=True)
def _apply_kernel(out, n1, n2, tau2, v, lower_only):
    """Map scaled squared distances in ``out`` to covariances, in place.

    The kernel branch is hoisted out of the loops for the same reason as in
    :func:`_fill_d2`.
    """
    if v == 999.0:  # squared exponential
        for i in range(n1):
            jmax = i + 1 if lower_only else n2
            for j in range(jmax):
                out[i, j] = tau2 * np.exp(-out[i, j])
    elif v == 0.5:
        for i in range(n1):
            jmax = i + 1 if lower_only else n2
            for j in range(jmax):
                s = np.sqrt(out[i, j])
                out[i, j] = tau2 * np.exp(-s)
    elif v == 1.5:
        for i in range(n1):
            jmax = i + 1 if lower_only else n2
            for j in range(jmax):
                r = out[i, j]
                s = np.sqrt(r)
                out[i, j] = tau2 * (1.0 + s) * np.exp(-s)
    else:  # v == 2.5
        for i in range(n1):
            jmax = i + 1 if lower_only else n2
            for j in range(jmax):
                r = out[i, j]
                s = np.sqrt(r)
                out[i, j] = tau2 * (1.0 + s + r / 3.0) * np.exp(-s)
    return out


@njit(cache=True)
def fill_cov(out, x1, x2, tau2, theta, g, v, sep, add_nugget):
    """Fill ``out`` with the full ``tau2 * (k(...) + g I)``.

    ``add_nugget`` should be True only when ``x1 is x2`` (square blocks); the
    R package likewise leaves the nugget off rectangular cross-covariances.
    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    _fill_d2(out, x1, x2, theta, v, sep, False)
    _apply_kernel(out, n1, n2, tau2, v, False)
    if add_nugget:
        for i in range(n1):
            out[i, i] += tau2 * g
    return out


@njit(cache=True)
def fill_cov_sym(out, x, n, tau2, theta, g, v, sep):
    """Fill only the **lower triangle** of the square covariance of ``x``.

    The covariance is symmetric and every consumer here feeds it straight to
    :func:`~vecdgp.vecchia._chol_lower`, which touches the lower triangle
    only -- so evaluating the upper triangle would double the number of
    ``exp``/``sqrt`` calls for nothing.  Those transcendentals dominate the
    runtime, so this halves the cost of the hot path.

    .. warning:: On return the strict upper triangle of ``out`` is
       **undefined**.  Do not hand the result to anything that reads it.
    """
    _fill_d2(out, x, x, theta, v, sep, True)
    _apply_kernel(out, n, n, tau2, v, True)
    for i in range(n):
        out[i, i] += tau2 * g
    return out


# ---------------------------------------------------------------------------
# python-level helpers (used in tests / dense reference implementations)
# ---------------------------------------------------------------------------
def _as2d(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.ascontiguousarray(x)


def _theta_vec(theta, d, sep):
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    if sep:
        if theta.size == 1:
            theta = np.repeat(theta, d)
        if theta.size != d:
            raise ValueError("length of theta must match ncol(x) when sep=True")
    return np.ascontiguousarray(theta)


def sq_dist(X1, X2=None):
    """Pairwise squared Euclidean distances (mirrors ``deepgp::sq_dist``)."""
    X1 = _as2d(X1)
    if X2 is None:
        X2 = X1
    else:
        X2 = _as2d(X2)
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("dimension of X1 & X2 do not match")
    d2 = (
        np.sum(X1 * X1, axis=1)[:, None]
        + np.sum(X2 * X2, axis=1)[None, :]
        - 2.0 * X1 @ X2.T
    )
    np.maximum(d2, 0.0, out=d2)
    return d2


def cov_matrix(x, tau2=1.0, theta=0.1, g=0.0, v=2.5, sep=False):
    """Dense covariance matrix ``tau2 * (k(x, x) + g I)``."""
    x = _as2d(x)
    th = _theta_vec(theta, x.shape[1], sep)
    out = np.empty((x.shape[0], x.shape[0]))
    fill_cov(out, x, x, float(tau2), th, float(g), float(v), bool(sep), True)
    return out


def cross_cov(x1, x2, tau2=1.0, theta=0.1, v=2.5, sep=False):
    """Dense rectangular cross-covariance (no nugget, as in ``deepgp``)."""
    x1 = _as2d(x1)
    x2 = _as2d(x2)
    th = _theta_vec(theta, x1.shape[1], sep)
    out = np.empty((x1.shape[0], x2.shape[0]))
    fill_cov(out, x1, x2, float(tau2), th, 0.0, float(v), bool(sep), False)
    return out
