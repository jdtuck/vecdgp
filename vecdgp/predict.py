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

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np

from ._compat import in_worker_thread
from .krig import krig_vec
from .vecchia import EPS

__all__ = ["PredictResult", "predict_shallow_vec", "predict_deep_vec",
           "resolve_cores"]


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
# parallelism over MCMC draws
# ---------------------------------------------------------------------------
def resolve_cores(cores):
    """Interpret the ``cores`` argument; ``None`` or ``-1`` means all of them.

    Capped by numba's hard thread maximum when numba is present, since
    ``NUMBA_NUM_THREADS`` may sit below the machine's core count and numba
    rejects any request above it.
    """
    avail = os.cpu_count() or 1
    try:
        from numba import config

        avail = min(avail, config.NUMBA_NUM_THREADS)
    except Exception:
        pass
    if cores is None or cores < 0:
        return avail
    return max(1, min(int(cores), avail))


def _spawn(rng, k):
    """``k`` independent generators, so a chunked run is still reproducible."""
    try:
        return rng.spawn(k)  # numpy >= 1.25
    except AttributeError:  # pragma: no cover - older numpy
        seeds = np.random.SeedSequence(rng.integers(2**63)).spawn(k)
        return [np.random.default_rng(s) for s in seeds]


def _chunks(nmcmc, cores):
    """Contiguous draw ranges, one per worker."""
    bounds = np.linspace(0, nmcmc, cores + 1).astype(int)
    return [range(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


@contextmanager
def _numba_threads(n):
    """Pin numba to ``n`` threads inside the block.

    When the draw loop is spread over a thread pool, leaving numba's own
    ``prange`` at full width would oversubscribe the machine several times
    over.  Draws are the coarser and better axis -- they parallelise the
    scipy KD-tree work too, which numba cannot touch -- so the inner width is
    narrowed to match.
    """
    try:
        from numba import get_num_threads, set_num_threads
    except Exception:
        yield
        return
    prev = get_num_threads()
    try:
        set_num_threads(max(1, min(n, prev)))
        yield
    finally:
        set_num_threads(prev)


def _run_draws(body, nmcmc, cores, rng, prefer_draws=False):
    """Apply ``body(t, rng, worker)`` for every draw, serially or across threads.

    One generator is spawned **per draw**, not per worker, so the random
    numbers consumed by draw ``t`` depend only on the seed and on ``t``.
    That makes results bit-identical whatever ``cores`` is set to -- worth
    the negligible spawn cost, since a parallel run that quietly returned
    different numbers would be a nasty thing to debug.

    There are two parallel axes and exactly one is used, never both (nesting
    them aborts under numba's ``workqueue`` threading layer):

    * **draws** -- a thread pool over MCMC draws, with serial numba kernels;
    * **numba** -- a serial draw loop with ``prange`` at full width.

    ``prefer_draws`` says the inner work is *not* prange-parallel, so the draw
    axis is the only one that buys anything.  That is the case for
    ``post_sample``, whose sampler is sequential over test locations by
    construction, and for joint prediction, which is dominated by scipy
    KD-tree and sparse-solve work numba cannot touch.  Without the flag, a
    machine with more cores than draws hands the work to numba and the
    parallelism silently collapses -- measured on a 40-core box, where
    ``post_sample`` peaked at 1.93x on 8 cores and fell back to 1.31x on 32.

    ``body`` must be free of shared mutable state: callers hand each worker
    its own copies of anything the draw loop writes to.
    """
    rngs = _spawn(rng, nmcmc)
    # With fewer draws than cores, splitting them leaves cores idle (each
    # worker runs the serial kernels), so prefer numba's prange -- unless the
    # inner work has no prange to offer.
    use_numba_axis = nmcmc < cores and not prefer_draws
    if cores <= 1 or nmcmc < 2 or use_numba_axis:
        for t in range(nmcmc):
            body(t, rngs[t], 0)
        return

    parts = _chunks(nmcmc, min(cores, nmcmc))

    def work(args):
        w, ts = args
        # inside the pool every numba kernel must take its serial path
        with in_worker_thread():
            for t in ts:
                body(t, rngs[t], w)

    inner = max(1, resolve_cores(None) // len(parts))
    with _numba_threads(inner):
        with ThreadPoolExecutor(max_workers=len(parts)) as ex:
            list(ex.map(work, enumerate(parts)))


# ---------------------------------------------------------------------------
# one layer
# ---------------------------------------------------------------------------
def predict_shallow_vec(obj, x_new, m=None, lite=True, order_new=None,
                        return_all=False, rng=None, samples_only=False, nper=1,
                        cores=1):
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
    ap.add_pred(x_new, m, lite=lite, order_new=order_new, rng=rng,
                pred_rows_only=samples_only)

    samples = np.empty((nper * obj.nmcmc, n_new)) if samples_only else None
    mu_t = np.empty((obj.nmcmc, n_new))
    s2_sum = np.zeros(n_new)
    sigma_sum = np.zeros((n_new, n_new))
    s2_t = np.empty((obj.nmcmc, n_new)) if (lite and return_all) else None

    # `ap` is fully built before the loop and only read inside it, so the
    # draws share it safely.
    sig_parts = {}

    def body(t, r, w):
        k = krig_vec(obj.y, ap, tau2=obj.tau2[t], theta=obj.theta[t],
                     g=_g_at(obj.g, t), v=obj.v, sep=sep,
                     s2=lite and not samples_only,
                     sigma=(not lite) and not samples_only,
                     nsamples=nper if samples_only else 0, rng=r)
        if samples_only:
            samples[t * nper : (t + 1) * nper] = k["samples"]
            return
        mu_t[t] = k["mean"]
        if lite:
            s2_t_all[t] = k["s2"]
        else:
            acc = sig_parts.get(w)
            sig_parts[w] = k["sigma"] if acc is None else acc + k["sigma"]

    s2_t_all = np.empty((obj.nmcmc, n_new)) if lite else None
    _run_draws(body, obj.nmcmc, cores, rng,
               prefer_draws=samples_only or not lite)
    if not samples_only:
        if lite:
            s2_sum = s2_t_all.sum(axis=0)
            if return_all:
                s2_t = s2_t_all
        else:
            for v in sig_parts.values():
                sigma_sum += v

    if samples_only:
        return samples
    return _combine(mu_t, s2_sum, sigma_sum, obj.nmcmc, lite, return_all, s2_t)


# ---------------------------------------------------------------------------
# two and three layers
# ---------------------------------------------------------------------------
def predict_deep_vec(obj, x_new, m=None, lite=True, mean_map=True,
                     store_latent=False, order_new=None, return_all=False,
                     layers=2, rng=None, samples_only=False, nper=1, cores=1):
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

    # The draw loop rewrites w_approx (and z_approx) at every iteration, so
    # each worker gets its own copy.  x_approx is finished above and only read
    # here, so it stays shared.  The copies are shallow by design: every
    # mutation in clean_pred / set_coords / add_pred rebinds an attribute to a
    # fresh array rather than writing into an existing one, so the arrays that
    # remain shared (order, rev_ord_obs, the training NN sets) are read-only.
    nworkers = 1 if cores <= 1 else min(resolve_cores(cores), obj.nmcmc)
    w_aps = [obj.w_approx if i == 0 else obj.w_approx.copy() for i in range(nworkers)]
    z_aps = (
        [obj.z_approx if i == 0 else obj.z_approx.copy() for i in range(nworkers)]
        if layers == 3
        else [None] * nworkers
    )
    s2_t_all = np.empty((obj.nmcmc, n_new)) if lite else None
    sig_parts = {}

    def body(t, r, w):
        g_t = _g_at(obj.g, t)
        w_ap, z_ap = w_aps[w], z_aps[w]

        if layers == 3:
            z_t = obj.z[t]
            z_new = np.empty((n_new, D))
            for i in range(D):
                k = krig_vec(z_t[:, i], obj.x_approx, tau2=obj.settings.tau2_z,
                             theta=obj.theta_z[t, i], g=EPS, v=obj.v,
                             nsamples=0 if mean_map else 1, rng=r)
                z_new[:, i] = k["mean"] if mean_map else k["samples"][0]
            z_ap.clean_pred()
            z_ap.set_coords(z_t)
            z_ap.add_pred(z_new, m, lite=mean_map, order_new=order_new, rng=r)
            if store_latent:
                z_new_store[t] = z_new

            w_t = obj.w[t]
            w_new = np.empty((n_new, D))
            for i in range(D):
                k = krig_vec(w_t[:, i], z_ap, tau2=obj.settings.tau2_w,
                             theta=obj.theta_w[t, i], g=EPS, v=obj.v,
                             nsamples=0 if mean_map else 1, rng=r)
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
                             prior_mean=pm, prior_mean_new=pmn, rng=r)
                w_new[:, i] = k["mean"] if mean_map else k["samples"][0]

        w_ap.clean_pred()
        w_ap.set_coords(w_t)
        w_ap.add_pred(w_new, m, lite=lite, order_new=order_new, rng=r,
                      pred_rows_only=samples_only)
        if store_latent:
            w_new_store[t] = w_new

        k = krig_vec(obj.y, w_ap, tau2=obj.tau2_y[t],
                     theta=obj.theta_y[t], g=g_t, v=obj.v,
                     s2=lite and not samples_only,
                     sigma=(not lite) and not samples_only,
                     nsamples=nper if samples_only else 0, rng=r)
        if samples_only:
            samples[t * nper : (t + 1) * nper] = k["samples"]
            return
        mu_t[t] = k["mean"]
        if lite:
            s2_t_all[t] = k["s2"]
        else:
            acc = sig_parts.get(w)
            sig_parts[w] = k["sigma"] if acc is None else acc + k["sigma"]

    _run_draws(body, obj.nmcmc, cores, rng,
               prefer_draws=samples_only or not lite)

    if samples_only:
        return samples
    if lite:
        s2_sum = s2_t_all.sum(axis=0)
        if return_all:
            s2_t = s2_t_all
    else:
        for part in sig_parts.values():
            sigma_sum += part
    out = _combine(mu_t, s2_sum, sigma_sum, obj.nmcmc, lite, return_all, s2_t)
    if store_latent:
        out["w_new"] = w_new_store
        if layers == 3:
            out["z_new"] = z_new_store
    return out
