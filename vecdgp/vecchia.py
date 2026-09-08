"""The Vecchia approximation: orderings, conditioning sets and the sparse
Cholesky factor ``U`` of the precision matrix.

Reference
---------
Sauer, Cooper & Gramacy (2023), *Vecchia-approximated Deep Gaussian Processes
for Computer Experiments*, Technometrics (arXiv:2204.02904); building on
Katzfuss & Guinness (2021) and Guinness (2018).

Notation
--------
Under a fixed ordering, the joint density factorises as

.. math::  p(Y) = \\prod_{i=1}^n p(y_i \\mid Y_{c(i)}),

with :math:`c(i) \\subset \\{1, \\dots, i-1\\}` of size :math:`\\min(m, i-1)`
taken to be the nearest neighbours of :math:`x_i` among its predecessors.
Each conditional is univariate Gaussian with

.. math::

   B_i = \\Sigma(x_i, X_{c(i)}) \\Sigma(X_{c(i)})^{-1}, \\quad
   \\mu_i = B_i Y_{c(i)}, \\quad
   \\sigma_i^2 = \\Sigma(x_i) - B_i \\Sigma(X_{c(i)}, x_i).

The induced precision matrix factorises as :math:`Q = U U^\\top` with ``U``
upper triangular and (Katzfuss & Guinness 2021, Prop. 1)

.. math::

   U_{ji} = \\begin{cases}
       1/\\sigma_i        & i = j \\\\
       -B_i[j] / \\sigma_i & j \\in c(i) \\\\
       0                  & \\text{otherwise.}
   \\end{cases}

Rather than forming :math:`B_i` explicitly we obtain the whole column of ``U``
from one small Cholesky decomposition: with
:math:`R = \\mathrm{chol}(\\Sigma(X_{c(i)} \\cup x_i))` (upper, point ``i``
ordered *last*), the column equals :math:`R^{-1} e_{last}`.  This is the
``U_entries`` trick of ``deepgp`` / ``GPvecchia``.

Storage
-------
``NN`` is an ``(n, m+1)`` integer array in the *ordered* index space: row ``i``
holds ``[i, c(i) sorted by distance...]``, padded with ``-1``.  ``Uvals`` has
matching shape, with ``Uvals[i, k]`` the value of ``U`` at row ``NN[i, k]``,
column ``i``; hence ``Uvals[i, 0]`` is the diagonal ``1/sigma_i``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp

from ._compat import njit, prange
from .kernels import fill_cov_sym, _as2d, _theta_vec

EPS = float(np.sqrt(np.finfo(float).eps))
"""Minimum nugget / jitter, ``sqrt(.Machine$double.eps)`` as in ``deepgp``."""

__all__ = [
    "EPS",
    "VecchiaApprox",
    "find_ordered_nn",
    "create_approx",
    "u_entries",
    "create_U_values",
    "create_U_sparse",
    "forward_solve_ut",
    "ut_mult",
    "rand_mvn_vec",
]


# ---------------------------------------------------------------------------
# small dense linear algebra, jit-friendly and exception-free
# ---------------------------------------------------------------------------
@njit(cache=True)
def _chol_lower(A, n):
    """In-place lower Cholesky. Returns 0 on success, 1 if not PD."""
    for j in range(n):
        s = A[j, j]
        for k in range(j):
            s -= A[j, k] * A[j, k]
        if s <= 0.0:
            return 1
        A[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            t = A[i, j]
            for k in range(j):
                t -= A[i, k] * A[j, k]
            A[i, j] = t / A[j, j]
    return 0


@njit(cache=True)
def _u_column(pts, n0, tau2, theta, g, v, sep, cov, out):
    """Column of ``U`` for one row: ``R^{-1} e_last`` reversed into ``out``.

    ``pts`` holds the conditioning set followed by the point itself.
    ``out[k]`` receives the entry belonging to ``NN[i, k]``.
    """
    # only the lower triangle is built: _chol_lower reads nothing else
    fill_cov_sym(cov, pts, n0, tau2, theta, g, v, sep)
    info = _chol_lower(cov, n0)
    if info != 0:
        # escalating jitter fallback (should be rare; g is >= EPS in practice)
        trace = 0.0
        fill_cov_sym(cov, pts, n0, tau2, theta, g, v, sep)
        for k in range(n0):
            trace += cov[k, k]
        jit = 1e-10 * trace / n0
        for _ in range(10):
            fill_cov_sym(cov, pts, n0, tau2, theta, g, v, sep)
            for k in range(n0):
                cov[k, k] += jit
            if _chol_lower(cov, n0) == 0:
                break
            jit *= 10.0

    # back-substitution for M = R^{-1} e_{n0-1}, R = L^T so R[k, j] = L[j, k]
    last = n0 - 1
    m_last = 1.0 / cov[last, last]
    out[0] = m_last  # diagonal entry -> NN[i, 0] == i
    # M[k] for k < last, walking upwards; store reversed into out
    m = np.zeros(n0)
    m[last] = m_last
    for k in range(last - 1, -1, -1):
        s = 0.0
        for j in range(k + 1, n0):
            s += cov[j, k] * m[j]
        m[k] = -s / cov[k, k]
    for k in range(n0):
        out[k] = m[last - k]


@njit(cache=True, parallel=True, nogil=True)
def u_entries(x_ord, NN, NN_len, tau2, theta, g, v, sep):
    """Entries of the sparse upper-triangular Cholesky factor ``U``.

    Returns ``Uvals`` of shape ``(n, m+1)``; ``Uvals[i, k]`` is ``U`` at row
    ``NN[i, k]``, column ``i``.
    """
    n = x_ord.shape[0]
    d = x_ord.shape[1]
    mp1 = NN.shape[1]
    Uvals = np.zeros((n, mp1))
    for i in prange(n):
        n0 = NN_len[i]
        pts = np.empty((n0, d))
        for j in range(n0):
            idx = NN[i, n0 - 1 - j]  # conditioning set first, point i last
            for k in range(d):
                pts[j, k] = x_ord[idx, k]
        cov = np.empty((n0, n0))
        out = np.empty(n0)
        _u_column(pts, n0, tau2, theta, g, v, sep, cov, out)
        for k in range(n0):
            Uvals[i, k] = out[k]
    return Uvals


@njit(cache=True, nogil=True)
def forward_solve_ut(Uvals, NN, NN_len, z):
    """Solve ``U^T y = z`` by forward substitution (``U^T`` is lower)."""
    n = z.shape[0]
    y = np.empty(n)
    for i in range(n):
        acc = z[i]
        for k in range(1, NN_len[i]):
            acc -= Uvals[i, k] * y[NN[i, k]]
        y[i] = acc / Uvals[i, 0]
    return y


@njit(cache=True, parallel=True, nogil=True)
def ut_mult(Uvals, NN, NN_len, vvec):
    """Compute ``U^T v`` exploiting the sparsity pattern."""
    n = vvec.shape[0]
    out = np.zeros(n)
    for i in prange(n):
        s = 0.0
        for k in range(NN_len[i]):
            s += Uvals[i, k] * vvec[NN[i, k]]
        out[i] = s
    return out


# ---------------------------------------------------------------------------
# ordered nearest neighbours
# ---------------------------------------------------------------------------
def find_ordered_nn(x, m):
    """Ordered nearest neighbours, equivalent to ``GpGp::find_ordered_nn``.

    Returns ``(NN, NN_len)``.  Row ``i`` of ``NN`` is ``i`` followed by the
    ``min(m, i)`` previous indices closest to ``x[i]``, sorted by increasing
    distance and padded with ``-1``.
    """
    x = _as2d(x)
    n = x.shape[0]
    m = int(min(m, max(n - 1, 0)))
    NN = np.full((n, m + 1), -1, dtype=np.int64)
    NN[:, 0] = np.arange(n)
    if n == 1 or m == 0:
        return NN, _nn_len(NN)

    # brute force for the first 2m+1 rows (cheap: prefixes are tiny)
    n_brute = int(min(n, 2 * m + 1))
    for i in range(1, n_brute):
        diff = x[:i] - x[i]
        d2 = np.einsum("ij,ij->i", diff, diff)
        k = min(m, i)
        nearest = np.argpartition(d2, k - 1)[:k] if k < i else np.arange(i)
        nearest = nearest[np.argsort(d2[nearest], kind="stable")]
        NN[i, 1 : 1 + k] = nearest

    # Doubling blocks for the rest.
    #
    # A conditioning set may only reference *predecessors*, but a KD-tree
    # knows nothing about the ordering, so neighbours have to be found by
    # distance and then filtered by index.  Under the random ordering the two
    # are independent, so among the k nearest neighbours of point i only about
    # k*i/n survive the filter -- for small i that is almost none, and a
    # single global sweep has to keep doubling k over *all* points to satisfy
    # its worst row.  That is what made this the dominant cost of joint
    # prediction (it is rebuilt at every MCMC draw).
    #
    # Processing in doubling blocks [a, 2a) instead bounds the damage: the
    # tree holds x[:b], of which at most half can fail the index filter, so a
    # modest k suffices and each point is queried once against a tree no
    # larger than it needs.  k = 2(m+1) measured fastest across n: a wider
    # first query costs more than the occasional doubling it avoids.
    # Exactness is unchanged -- every candidate is still the true nearest
    # predecessor set.
    a = n_brute
    while a < n:
        b = int(min(n, 2 * a))
        tree = _build_tree(x[:b])
        pending = np.arange(a, b)
        k = int(min(b, 2 * (m + 1)))
        while pending.size:
            _, idx = tree.query(x[pending], k=k, workers=-1)
            idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))

            # `idx` is distance-sorted, and a *stable* argsort on the validity
            # mask floats the predecessors to the front preserving that order.
            valid = idx <= pending[:, None]
            enough = valid.sum(axis=1) >= m + 1
            if enough.any():
                order = np.argsort(~valid[enough], axis=1, kind="stable")
                NN[pending[enough]] = np.take_along_axis(
                    idx[enough], order[:, : m + 1], axis=1
                )
                pending = pending[~enough]
            if pending.size:
                if k >= b:  # unreachable: every row here has >= 2m+1 predecessors
                    raise RuntimeError("failed to build conditioning sets")
                k = int(min(b, 2 * k))
        a = b
    return NN, _nn_len(NN)


def _nn_len(NN):
    return (NN >= 0).sum(axis=1).astype(np.int64)


def _build_tree(pts):
    """KD-tree tuned for build-once-query-once use.

    Both prediction paths rebuild a tree at *every* MCMC draw (the warped
    coordinates change), so construction cost matters as much as query cost.
    Sliding-midpoint splits skip the median-finding that ``balanced_tree``
    does, which is markedly cheaper to build for a negligible query penalty
    at these sizes.
    """
    return cKDTree(pts, compact_nodes=False, balanced_tree=False)


def _knnx(reference, query, k):
    """``k`` nearest neighbours of each query point among ``reference``."""
    k = int(min(k, reference.shape[0]))
    tree = _build_tree(reference)
    _, idx = tree.query(query, k=k, workers=-1)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx.reshape(-1, 1) if k == 1 else idx.reshape(1, -1)
    return idx


# ---------------------------------------------------------------------------
# the approximation object
# ---------------------------------------------------------------------------
@dataclass
class VecchiaApprox:
    """Ordering + conditioning sets for one layer of inputs.

    Mirrors the list produced by ``deepgp:::create_approx``.
    """

    m: int
    order: np.ndarray  # ord[k] = original index of ordered row k
    rev_ord_obs: np.ndarray  # inverse permutation
    NN: np.ndarray
    NN_len: np.ndarray
    x_ord: np.ndarray

    # prediction extras -----------------------------------------------------
    m_new: Optional[int] = None
    x_new: Optional[np.ndarray] = None  # lite = True
    NN_new: Optional[np.ndarray] = None  # lite = True
    order_new: Optional[np.ndarray] = None  # lite = False
    rev_ord_new: Optional[np.ndarray] = None  # lite = False
    observed: Optional[np.ndarray] = None  # lite = False
    n_obs: int = 0

    # -- construction -------------------------------------------------------
    def copy(self):
        return replace(self)

    @property
    def lite(self):
        return self.order_new is None

    def set_coords(self, x):
        """Replace the (latent) coordinates, keeping ordering & NN sets.

        Used during prediction, where each MCMC draw supplies new warped
        coordinates ``W^{(t)}`` for the same training locations.
        """
        x = _as2d(x)
        self.x_ord = np.ascontiguousarray(x[self.order])
        return self

    # -- prediction ---------------------------------------------------------
    def clean_pred(self):
        """Strip any predictive locations previously added."""
        if self.observed is not None:
            n = int(self.observed.sum())
            self.x_ord = np.ascontiguousarray(self.x_ord[:n])
            self.NN = self.NN[:n, : self.m + 1].copy()
            self.NN_len = _nn_len(self.NN)
            self.observed = None
            self.order_new = None
            self.rev_ord_new = None
        self.m_new = None
        self.x_new = None
        self.NN_new = None
        return self

    def add_pred(self, x_new, m, lite=True, order_new=None, rng=None):
        """Incorporate predictive locations.

        ``lite=True`` stores, for each test point, its ``m`` nearest training
        neighbours (independent point-wise prediction).  ``lite=False``
        appends the test points to the ordering and rebuilds the joint
        conditioning sets so that a stacked ``U`` can be formed.
        """
        x_new = _as2d(x_new)
        n_new = x_new.shape[0]
        self.m_new = int(m)
        if lite:
            self.x_new = x_new
            self.NN_new = _knnx(self.x_ord, x_new, m)
            self.m_new = self.NN_new.shape[1]
        else:
            rng = np.random.default_rng() if rng is None else rng
            if order_new is None:
                order_new = rng.permutation(n_new)
            order_new = np.asarray(order_new, dtype=np.int64)
            n_obs = self.x_ord.shape[0]
            self.n_obs = n_obs
            self.order_new = order_new
            self.rev_ord_new = np.argsort(order_new)
            self.observed = np.concatenate(
                [np.ones(n_obs, dtype=bool), np.zeros(n_new, dtype=bool)]
            )
            self.x_ord = np.ascontiguousarray(
                np.vstack([self.x_ord, x_new[order_new]])
            )
            self.NN, self.NN_len = find_ordered_nn(self.x_ord, m)
        return self


def create_approx(x, m, order=None, rng=None):
    """Build a :class:`VecchiaApprox` with a random ordering (Guinness 2018)."""
    x = _as2d(x)
    n = x.shape[0]
    m = int(min(m, n - 1))
    if order is None:
        rng = np.random.default_rng() if rng is None else rng
        order = rng.permutation(n)
    order = np.asarray(order, dtype=np.int64)
    x_ord = np.ascontiguousarray(x[order])
    NN, NN_len = find_ordered_nn(x_ord, m)
    return VecchiaApprox(
        m=m,
        order=order,
        rev_ord_obs=np.argsort(order),
        NN=NN,
        NN_len=NN_len,
        x_ord=x_ord,
        n_obs=n,
    )


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------
def create_U_values(approx, tau2=1.0, theta=0.1, g=0.0, v=2.5, sep=False):
    """Raw ``U`` entries for the current coordinates of ``approx``."""
    th = _theta_vec(theta, approx.x_ord.shape[1], sep)
    return u_entries(
        approx.x_ord, approx.NN, approx.NN_len,
        float(tau2), th, float(g), float(v), bool(sep),
    )


def create_U_sparse(approx, tau2=1.0, theta=0.1, g=0.0, v=2.5, sep=False):
    """``U`` as a ``scipy.sparse`` CSC matrix (used for joint prediction)."""
    Uvals = create_U_values(approx, tau2, theta, g, v, sep)
    n = approx.x_ord.shape[0]
    mask = approx.NN >= 0
    cols = np.repeat(np.arange(n), approx.NN_len)
    rows = approx.NN[mask]
    data = Uvals[mask]
    return sp.csc_matrix((data, (rows, cols)), shape=(n, n))


def rand_mvn_vec(approx, tau2=1.0, theta=0.1, g=EPS, v=2.5, sep=False,
                 prior_mean=0.0, rng=None):
    """Draw ``Y ~ N(prior_mean, (U U^T)^{-1})`` via one forward solve.

    This is the Vecchia prior draw of Section 3.2 of the paper: sample
    ``z ~ N(0, I)`` and solve ``U^T y = z``.  The result is returned in the
    *original* (un-ordered) index space, matching ``prior_mean``.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = approx.x_ord.shape[0]
    Uvals = create_U_values(approx, tau2, theta, g, v, sep)
    z = rng.standard_normal(n)
    y = forward_solve_ut(Uvals, approx.NN, approx.NN_len, z)
    return y[approx.rev_ord_obs] + prior_mean
