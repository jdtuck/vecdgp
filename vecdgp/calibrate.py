"""Fast repeated emulator draws, for calibration MCMC.

An outer calibration sampler evaluates the emulator once per iteration, at a
single proposed parameter, for a hundred thousand iterations or more.
:meth:`~vecdgp.fit.DGP2Vec.post_sample` is built for the opposite shape --
many locations, all retained draws at once -- and spends essentially all of
its time on per-call setup over the *training* set: rebuilding a KD-tree over
all ``n`` points, re-ordering the latent layer, allocating ``(n, m+1)``
conditioning arrays.  None of that depends on the location being evaluated.

This module hoists it.  Per posterior draw ``t`` the ordered latent layer and
its KD-tree are built once and memoised; a call then only queries the trees
and factorises one ``(m+1) x (m+1)`` block.  At n = 6000 that is the
difference between milliseconds and tens of microseconds per evaluation.

Typical use, propagating emulator uncertainty by drawing a fresh posterior
iteration each outer step::

    emu = fit.sampler()
    rng = np.random.default_rng(0)
    for step in range(100_000):
        t = rng.integers(emu.nmcmc)          # new posterior draw each step
        y = emu.sample(theta_proposed, draw=t, rng=rng)
        ...                                   # accept/reject on y

Holding ``t`` fixed instead gives a plug-in emulator: cheaper and lower
variance, but the calibration posterior will then be conditional on one
emulator fit rather than integrating over emulator uncertainty. That is a
statistical choice, not a performance one.
"""

from __future__ import annotations

import numpy as np

from ._compat import njit
from .kernels import _theta_vec, fill_cov_sym
from .krig import _krig_lite_serial
from .vecchia import EPS, _build_tree, _chol_lower

__all__ = ["PosteriorSampler"]


@njit(cache=True, nogil=True)
def _sample_appended(x_train, y_train, x_new, cond, cond_len, tau2, theta, g,
                     v, sep, z):
    """Sequential draws at ``x_new`` given a prebuilt conditioning set.

    ``cond[j, k]`` indexes the k-th conditioning point of new point ``j``:
    values below ``n_train`` are training points, values at or above it are
    earlier new points (offset by ``n_train``).  Conditioning points are
    ordered farthest-first with the target last, matching
    :func:`~vecdgp.krig._krig_samples` so the two agree numerically and not
    merely in distribution.
    """
    n_train = x_train.shape[0]
    d = x_train.shape[1]
    n_new = x_new.shape[0]
    nsamples = z.shape[0]
    out = np.zeros((nsamples, n_new))

    for s in range(nsamples):
        for j in range(n_new):
            nc = cond_len[j]
            n0 = nc + 1
            pts = np.empty((n0, d))
            vals = np.empty(nc)
            for k in range(nc):
                idx = cond[j, k]
                if idx < n_train:
                    for c in range(d):
                        pts[k, c] = x_train[idx, c]
                    vals[k] = y_train[idx]
                else:
                    jj = idx - n_train
                    for c in range(d):
                        pts[k, c] = x_new[jj, c]
                    vals[k] = out[s, jj]
            for c in range(d):
                pts[nc, c] = x_new[j, c]

            K = np.empty((n0, n0))
            fill_cov_sym(K, pts, n0, 1.0, theta, g, v, sep)
            if _chol_lower(K, n0) != 0:
                trace = 0.0
                fill_cov_sym(K, pts, n0, 1.0, theta, g, v, sep)
                for c in range(n0):
                    trace += K[c, c]
                jit = 1e-10 * trace / n0
                for _ in range(10):
                    fill_cov_sym(K, pts, n0, 1.0, theta, g, v, sep)
                    for c in range(n0):
                        K[c, c] += jit
                    if _chol_lower(K, n0) == 0:
                        break
                    jit *= 10.0

            a = np.empty(nc)
            for r in range(nc):
                acc = vals[r]
                for c in range(r):
                    acc -= K[r, c] * a[c]
                a[r] = acc / K[r, r]
            mean = 0.0
            for r in range(nc):
                mean += K[nc, r] * a[r]
            out[s, j] = mean + np.sqrt(tau2) * K[nc, nc] * z[s, j]
    return out


class _DrawState:
    """Per-posterior-draw cache: ordered latent coordinates and their tree."""

    __slots__ = ("w_ord", "tree", "z_ord", "z_tree", "w_cols", "z_cols")

    def __init__(self, w_ord, tree, z_ord=None, z_tree=None):
        self.w_ord = w_ord
        self.tree = tree
        self.z_ord = z_ord
        self.z_tree = z_tree
        # contiguous copies of each latent column, so the per-call kriging
        # does not re-copy n doubles every evaluation
        self.w_cols = [np.ascontiguousarray(w_ord[:, i])
                       for i in range(w_ord.shape[1])]
        self.z_cols = ([np.ascontiguousarray(z_ord[:, i])
                        for i in range(z_ord.shape[1])]
                       if z_ord is not None else None)


class PosteriorSampler:
    """Cached emulator sampler for repeated single-location evaluation.

    Built by :meth:`~vecdgp.fit.GPVec.sampler` and friends rather than
    directly.  Draws are bit-comparable with ``post_sample(draws=t)`` -- the
    same conditioning sets and the same conditional moments -- but skip the
    per-call setup that does not depend on the evaluation location.

    Per-draw state is built lazily on first use of that draw and kept, so a
    loop that visits every retained draw ends up holding one KD-tree per draw
    (a few hundred kB each at n = 6000).  ``max_cached`` bounds that if memory
    is tight; the cache then evicts in first-in order.
    """

    def __init__(self, fit, m=None, max_cached=None):
        self.fit = fit
        self.nmcmc = fit.nmcmc
        self.layers = 3 if hasattr(fit, "z") else (2 if hasattr(fit, "w") else 1)
        n = fit.x.shape[0]
        self.m = int(min(n, 2 * fit.m) if m is None else min(n, m))
        self.max_cached = max_cached
        self._cache = {}

        # X is fixed across draws and calls, so its ordering and tree are
        # built once for the whole loop.
        ap = fit.x_approx
        self.order = ap.order
        self.x_ord = np.ascontiguousarray(fit.x[ap.order])
        self.x_tree = _build_tree(self.x_ord)
        self.y_ord = np.ascontiguousarray(np.asarray(fit.y, float)[ap.order])
        self.v = float(fit.v)
        self.sep = bool(getattr(fit.settings, "sep", False))

    # -- per-draw state -----------------------------------------------------
    def _state(self, t):
        st = self._cache.get(t)
        if st is not None:
            return st
        fit = self.fit
        if self.layers == 1:
            st = _DrawState(self.x_ord, self.x_tree)
        else:
            w_ord = np.ascontiguousarray(fit.w[t][self.order])
            z_ord = z_tree = None
            if self.layers == 3:
                z_ord = np.ascontiguousarray(fit.z[t][self.order])
                z_tree = _build_tree(z_ord)
            st = _DrawState(w_ord, _build_tree(w_ord), z_ord, z_tree)
        if self.max_cached is not None and len(self._cache) >= self.max_cached:
            self._cache.pop(next(iter(self._cache)))
        self._cache[t] = st
        return st

    def _g_at(self, t):
        g = self.fit.g
        return float(g) if np.ndim(g) == 0 else float(g[t])

    # -- latent mapping -----------------------------------------------------
    def _map_forward(self, x_new, t, st):
        """Conditional-mean map of ``x_new`` through the latent layer(s)."""
        fit = self.fit
        if self.layers == 1:
            return x_new
        d_src, tree, src = x_new, self.x_tree, self.x_ord
        if self.layers == 3:
            k = int(min(self.m, src.shape[0]))
            nn = np.atleast_2d(np.asarray(tree.query(d_src, k=k)[1], np.int64))
            z_new = np.empty((x_new.shape[0], fit.z.shape[2]))
            for i in range(z_new.shape[1]):
                z_new[:, i] = _krig_lite_serial(
                    src, d_src, nn, st.z_cols[i], 1.0,
                    _theta_vec(fit.theta_z[t, i], src.shape[1], False),
                    EPS, self.v, False, False,
                )[0]
            d_src, tree, src = z_new, st.z_tree, st.z_ord

        k = int(min(self.m, src.shape[0]))
        nn = np.atleast_2d(np.asarray(tree.query(d_src, k=k)[1], np.int64))
        D = fit.w.shape[2]
        w_new = np.empty((x_new.shape[0], D))
        pmx = bool(getattr(fit.settings, "pmx", False))
        for i in range(D):
            src_vals = st.w_cols[i]
            if pmx:
                src_vals = np.ascontiguousarray(src_vals - self.x_ord[:, i])
            mu = _krig_lite_serial(
                src, d_src, nn, src_vals, 1.0,
                _theta_vec(fit.theta_w[t, i], src.shape[1], False),
                EPS, self.v, False, False,
            )[0]
            w_new[:, i] = mu + (x_new[:, i] if pmx else 0.0)
        return w_new

    # -- the call that runs inside the outer loop ---------------------------
    def _conditioning(self, w_new, st):
        """Nearest-predecessor sets for the evaluation points.

        Training points all precede the new ones, and new point ``j`` may also
        condition on new points before it -- brute-forced, since there are only
        a handful. Ordered farthest-first to match the reference sampler.
        """
        n_train = st.w_ord.shape[0]
        n_new = w_new.shape[0]
        m = int(min(self.m, n_train))
        dist, idx = st.tree.query(w_new, k=m)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(np.asarray(idx, np.int64))

        if n_new == 1:
            cond = idx[:, ::-1].copy()  # farthest first
            return np.ascontiguousarray(cond), np.full(1, m, np.int64)

        cond = np.empty((n_new, m), np.int64)
        clen = np.empty(n_new, np.int64)
        for j in range(n_new):
            cid, cd = idx[j], dist[j]
            if j:  # earlier new points are predecessors too
                dj = np.sqrt(((w_new[:j] - w_new[j]) ** 2).sum(axis=1))
                cid = np.concatenate([cid, np.arange(j, dtype=np.int64) + n_train])
                cd = np.concatenate([cd, dj])
            keep = np.argsort(cd, kind="stable")[:m]
            sel = cid[keep]
            clen[j] = sel.size
            cond[j, : sel.size] = sel[::-1]  # farthest first
        return cond, clen

    def sample(self, x_new, draw, rng=None, nper=1):
        """Draw from the emulator at ``x_new`` under posterior iteration ``draw``.

        Returns an array of shape ``(nper, len(x_new))``; for the common
        single-location single-sample case that is ``(1, 1)``.
        """
        fit = self.fit
        t = int(draw) % self.nmcmc
        rng = np.random.default_rng() if rng is None else rng
        x_new = np.ascontiguousarray(np.atleast_2d(np.asarray(x_new, float)))
        if x_new.shape[1] != fit.x.shape[1]:
            raise ValueError("dimension of x_new does not match dimension of x")

        st = self._state(t)
        w_new = np.ascontiguousarray(self._map_forward(x_new, t, st))
        cond, clen = self._conditioning(w_new, st)

        tau2 = float(fit.tau2_y[t] if self.layers > 1 else fit.tau2[t])
        theta = fit.theta_y[t] if self.layers > 1 else fit.theta[t]
        z = rng.standard_normal((nper, x_new.shape[0]))
        return _sample_appended(
            st.w_ord, self.y_ord, w_new, cond, clen, tau2,
            _theta_vec(theta, st.w_ord.shape[1], self.sep),
            self._g_at(t), self.v, self.sep, z,
        )

    def mean_sd(self, x_new, draw):
        """Conditional mean and standard deviation, without drawing.

        The moments the draw is taken from -- useful for a plug-in likelihood,
        or for checking against ``predict(draws=t)``.
        """
        fit = self.fit
        t = int(draw) % self.nmcmc
        x_new = np.ascontiguousarray(np.atleast_2d(np.asarray(x_new, float)))
        st = self._state(t)
        w_new = np.ascontiguousarray(self._map_forward(x_new, t, st))
        m = int(min(self.m, st.w_ord.shape[0]))
        nn = np.atleast_2d(np.asarray(st.tree.query(w_new, k=m)[1], np.int64))
        tau2 = float(fit.tau2_y[t] if self.layers > 1 else fit.tau2[t])
        theta = fit.theta_y[t] if self.layers > 1 else fit.theta[t]
        mu, s2 = _krig_lite_serial(
            st.w_ord, w_new, nn, self.y_ord, tau2,
            _theta_vec(theta, st.w_ord.shape[1], self.sep),
            self._g_at(t), self.v, self.sep, True,
        )
        return mu, np.sqrt(s2)
