"""User-facing fit functions and fitted-model objects.

Mirrors ``deepgp::fit_one_layer`` / ``fit_two_layer`` / ``fit_three_layer``
called with ``vecchia = TRUE``.

Inputs are expected on the same scale as the R package: ``x`` scaled to
:math:`[0,1]^d` and ``y`` centred and scaled to unit variance.  The default
priors are calibrated for that scaling.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .gibbs import (
    gibbs_one_layer_vec,
    gibbs_three_layer_vec,
    gibbs_two_layer_vec,
    init_latent,
)
from .kernels import EXP2
from .predict import predict_deep_vec, predict_shallow_vec
from .settings import Settings, default_settings
from .vecchia import EPS, VecchiaApprox

__all__ = [
    "fit_one_layer",
    "fit_two_layer",
    "fit_three_layer",
    "GPVec",
    "DGP2Vec",
    "DGP3Vec",
]


def _as2d(x):
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else np.ascontiguousarray(x)


def _resolve_v(cov, v):
    if cov == "exp2":
        return EXP2
    if cov != "matern":
        raise ValueError("cov must be 'matern' or 'exp2'")
    if v not in (0.5, 1.5, 2.5):
        raise ValueError("v must be one of 0.5, 1.5, 2.5")
    return float(v)


def _check_inputs(x, y, true_g, nmcmc):
    if np.ndim(y) != 1:
        raise ValueError("y must be a 1-d array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("dimensions of x and y do not match")
    if nmcmc <= 1:
        raise ValueError("nmcmc must be greater than 1")
    if x.min() < -5 or x.max() > 6:
        warnings.warn("this implementation is designed for x scaled to [0, 1]")
    if true_g is None and (abs(y.mean()) > 10 or not 0.1 <= y.var() <= 10):
        warnings.warn("designed for y scaled to mean zero and variance one")


def _thin_index(nmcmc, burn, thin):
    idx = np.arange(burn + 1, nmcmc + 1)
    idx = idx[idx % thin == 0]
    return idx - 1


# ---------------------------------------------------------------------------
# fitted objects
# ---------------------------------------------------------------------------
@dataclass
class _BaseFit:
    x: np.ndarray
    y: np.ndarray
    nmcmc: int
    settings: Settings
    v: float
    m: int
    ll: np.ndarray
    time: float = 0.0

    def _idx(self, burn, thin):
        if burn >= self.nmcmc:
            raise ValueError("burn must be less than nmcmc")
        return _thin_index(self.nmcmc, burn, thin)


@dataclass
class GPVec(_BaseFit):
    """One-layer GP fitted with the Vecchia approximation."""

    g: object = None
    theta: np.ndarray = None
    tau2: np.ndarray = None
    x_approx: VecchiaApprox = None

    def trim(self, burn, thin=1):
        i = self._idx(burn, thin)
        out = replace(self)
        out.nmcmc = len(i)
        out.theta = self.theta[i]
        out.tau2 = self.tau2[i]
        out.ll = self.ll[i]
        out.g = self.g if np.ndim(self.g) == 0 else self.g[i]
        return out

    def predict(self, x_new, m=None, lite=True, order_new=None,
                return_all=False, rng=None):
        return predict_shallow_vec(self, x_new, m=m, lite=lite,
                                   order_new=order_new,
                                   return_all=return_all, rng=rng)


@dataclass
class DGP2Vec(_BaseFit):
    """Two-layer deep GP fitted with the Vecchia approximation."""

    g: object = None
    theta_y: np.ndarray = None
    theta_w: np.ndarray = None
    tau2_y: np.ndarray = None
    w: np.ndarray = None
    x_approx: VecchiaApprox = None
    w_approx: VecchiaApprox = None

    def trim(self, burn, thin=1):
        i = self._idx(burn, thin)
        out = replace(self)
        out.nmcmc = len(i)
        out.theta_y = self.theta_y[i]
        out.theta_w = self.theta_w[i]
        out.tau2_y = self.tau2_y[i]
        out.w = self.w[i]
        out.ll = self.ll[i]
        out.g = self.g if np.ndim(self.g) == 0 else self.g[i]
        return out

    def predict(self, x_new, m=None, lite=True, mean_map=True,
                store_latent=False, order_new=None, return_all=False, rng=None):
        return predict_deep_vec(self, x_new, m=m, lite=lite, mean_map=mean_map,
                                store_latent=store_latent, order_new=order_new,
                                return_all=return_all, layers=2, rng=rng)


@dataclass
class DGP3Vec(_BaseFit):
    """Three-layer deep GP fitted with the Vecchia approximation."""

    g: object = None
    theta_y: np.ndarray = None
    theta_w: np.ndarray = None
    theta_z: np.ndarray = None
    tau2_y: np.ndarray = None
    w: np.ndarray = None
    z: np.ndarray = None
    x_approx: VecchiaApprox = None
    w_approx: VecchiaApprox = None
    z_approx: VecchiaApprox = None

    def trim(self, burn, thin=1):
        i = self._idx(burn, thin)
        out = replace(self)
        out.nmcmc = len(i)
        out.theta_y = self.theta_y[i]
        out.theta_w = self.theta_w[i]
        out.theta_z = self.theta_z[i]
        out.tau2_y = self.tau2_y[i]
        out.w = self.w[i]
        out.z = self.z[i]
        out.ll = self.ll[i]
        out.g = self.g if np.ndim(self.g) == 0 else self.g[i]
        return out

    def predict(self, x_new, m=None, lite=True, mean_map=True,
                store_latent=False, order_new=None, return_all=False, rng=None):
        return predict_deep_vec(self, x_new, m=m, lite=lite, mean_map=mean_map,
                                store_latent=store_latent, order_new=order_new,
                                return_all=return_all, layers=3, rng=rng)


# ---------------------------------------------------------------------------
# fit functions
# ---------------------------------------------------------------------------
def fit_one_layer(x, y, nmcmc=10000, sep=False, verb=True, theta_0=0.01,
                  g_0=0.001, true_g=None, v=2.5, settings=None, cov="matern",
                  m=None, order=None, seed=None, rng=None):
    """MCMC sampling for a one-layer (Vecchia-approximated) GP.

    Parameters
    ----------
    x, y : array
        Inputs (``n x d``, scaled to [0, 1]) and responses (length ``n``).
    nmcmc : int
        Number of MCMC iterations.
    sep : bool
        Separable (ARD) rather than isotropic lengthscales.
    theta_0, g_0 : float
        Initial lengthscale / nugget.
    true_g : float or None
        Fix the nugget at this value (use a small value such as ``1e-6`` for
        deterministic simulations); ``None`` estimates it.
    v : float
        Matern smoothness (0.5, 1.5 or 2.5); ignored when ``cov='exp2'``.
    m : int or None
        Size of the Vecchia conditioning sets; defaults to ``min(25, n-1)``.
    order : array or None
        Ordering of the observations; defaults to random (Guinness 2018).
    """
    import time

    tic = time.time()
    rng = np.random.default_rng(seed) if rng is None else rng
    x = _as2d(x)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = x.shape
    if sep and d == 1:
        sep = False
    _check_inputs(x, y, true_g, nmcmc)
    v = _resolve_v(cov, v)

    st = default_settings(layers=1, noisy=true_g is None, overrides=settings)
    st.sep = sep
    if m is None:
        m = min(25, n - 1)
    if m >= n:
        raise ValueError("m must be less than n")

    theta_0 = np.repeat(theta_0, d) if sep and np.ndim(theta_0) == 0 else theta_0
    initial = {"theta": theta_0, "g": g_0}

    out = gibbs_one_layer_vec(x, y, nmcmc, verb, initial, true_g, st, v, m,
                              order=order, rng=rng)
    return GPVec(x=x, y=y, nmcmc=nmcmc, settings=st, v=v, m=m, ll=out["ll"],
                 time=time.time() - tic, g=out["g"], theta=out["theta"],
                 tau2=out["tau2"], x_approx=out["x_approx"])


def fit_two_layer(x, y, nmcmc=10000, D=None, pmx=False, verb=True, w_0=None,
                  theta_y_0=0.01, theta_w_0=0.1, g_0=0.001, true_g=None,
                  v=2.5, settings=None, cov="matern", m=None, order=None,
                  seed=None, rng=None):
    """MCMC sampling for a two-layer deep GP with the Vecchia approximation.

    ``D`` is the width of the latent layer (defaults to ``ncol(x)``).  The
    latent layer is initialised at the identity mapping ``W = X`` unless
    ``w_0`` is supplied.  ``pmx=True`` gives the latent layer a prior mean of
    ``x`` instead of zero (requires ``D == ncol(x)``).
    """
    import time

    tic = time.time()
    rng = np.random.default_rng(seed) if rng is None else rng
    x = _as2d(x)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = x.shape
    D = d if D is None else int(D)
    _check_inputs(x, y, true_g, nmcmc)
    v = _resolve_v(cov, v)

    st = default_settings(layers=2, noisy=true_g is None, overrides=settings)
    if pmx and d != D:
        raise ValueError("pmx=True requires D == ncol(x)")
    st.pmx = pmx
    if m is None:
        m = min(25, n - 1)
    if m >= n:
        raise ValueError("m must be less than n")

    w0 = init_latent(x, D) if w_0 is None else _as2d(w_0)
    if w0.shape != (n, D):
        raise ValueError("w_0 must be n x D")
    theta_w_0 = (
        np.repeat(theta_w_0, D) if np.ndim(theta_w_0) == 0 else np.asarray(theta_w_0)
    )
    initial = {"w": w0, "theta_y": theta_y_0, "theta_w": theta_w_0, "g": g_0}

    out = gibbs_two_layer_vec(x, y, nmcmc, D, verb, initial, true_g, st, v, m,
                              order=order, rng=rng)
    return DGP2Vec(x=x, y=y, nmcmc=nmcmc, settings=st, v=v, m=m, ll=out["ll"],
                   time=time.time() - tic, g=out["g"], theta_y=out["theta_y"],
                   theta_w=out["theta_w"], tau2_y=out["tau2_y"], w=out["w"],
                   x_approx=out["x_approx"], w_approx=out["w_approx"])


def fit_three_layer(x, y, nmcmc=10000, D=None, verb=True, w_0=None, z_0=None,
                    theta_y_0=0.01, theta_w_0=0.1, theta_z_0=0.1, g_0=0.001,
                    true_g=None, v=2.5, settings=None, cov="matern", m=None,
                    order=None, seed=None, rng=None):
    """MCMC sampling for a three-layer deep GP with the Vecchia approximation."""
    import time

    tic = time.time()
    rng = np.random.default_rng(seed) if rng is None else rng
    x = _as2d(x)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = x.shape
    D = d if D is None else int(D)
    _check_inputs(x, y, true_g, nmcmc)
    v = _resolve_v(cov, v)

    st = default_settings(layers=3, noisy=true_g is None, overrides=settings)
    if m is None:
        m = min(25, n - 1)
    if m >= n:
        raise ValueError("m must be less than n")

    w0 = init_latent(x, D) if w_0 is None else _as2d(w_0)
    z0 = init_latent(x, D) if z_0 is None else _as2d(z_0)
    theta_w_0 = (
        np.repeat(theta_w_0, D) if np.ndim(theta_w_0) == 0 else np.asarray(theta_w_0)
    )
    theta_z_0 = (
        np.repeat(theta_z_0, D) if np.ndim(theta_z_0) == 0 else np.asarray(theta_z_0)
    )
    initial = {"w": w0, "z": z0, "theta_y": theta_y_0, "theta_w": theta_w_0,
               "theta_z": theta_z_0, "g": g_0}

    out = gibbs_three_layer_vec(x, y, nmcmc, D, verb, initial, true_g, st, v, m,
                                order=order, rng=rng)
    return DGP3Vec(x=x, y=y, nmcmc=nmcmc, settings=st, v=v, m=m, ll=out["ll"],
                   time=time.time() - tic, g=out["g"], theta_y=out["theta_y"],
                   theta_w=out["theta_w"], theta_z=out["theta_z"],
                   tau2_y=out["tau2_y"], w=out["w"], z=out["z"],
                   x_approx=out["x_approx"], w_approx=out["w_approx"],
                   z_approx=out["z_approx"])
