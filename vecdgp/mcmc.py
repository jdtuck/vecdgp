"""Vecchia log-likelihood and the MCMC transition kernels.

Implements Section 3 of Sauer, Cooper & Gramacy (2023):

* :func:`logl_vec` -- the sparse-Cholesky Gaussian log-likelihood (their
  Eq. 11), optionally profiled over the scale :math:`\\tau^2`;
* :func:`sample_theta_vec`, :func:`sample_g_vec` -- Metropolis-Hastings with
  uniform sliding-window proposals and Gamma priors;
* :func:`sample_w_vec`, :func:`sample_z_vec` -- elliptical slice sampling
  (Murray et al. 2010) for the latent layers, with prior draws produced by a
  single sparse forward solve.
"""

from __future__ import annotations

import math

import numpy as np

from .vecchia import EPS, create_U_values, rand_mvn_vec, ut_mult_auto

__all__ = [
    "logl_vec",
    "sample_theta_vec",
    "sample_g_vec",
    "sample_w_vec",
    "sample_z_vec",
    "log_dgamma",
]


def log_dgamma(x, alpha, beta):
    """Log density of Gamma(shape=alpha, rate=beta); ``-inf`` for x <= 0."""
    if x <= 0.0:
        return -np.inf
    return (
        alpha * math.log(beta)
        - math.lgamma(alpha)
        + (alpha - 1.0) * math.log(x)
        - beta * x
    )


# ---------------------------------------------------------------------------
# likelihood
# ---------------------------------------------------------------------------
def logl_vec(y, approx, tau2=1.0, theta=0.1, g=0.0, v=2.5, mu=0.0,
             sep=False, outer=False):
    """Vecchia-approximated Gaussian log-likelihood.

    ``log L = sum_i log U_ii - 0.5 * || U^T (y - mu) ||^2``

    With ``outer=True`` the scale is integrated out under a reference prior
    (Gramacy 2020, Ch. 5), giving the profile likelihood
    ``sum_i log U_ii - (n/2) log(quadratic form)``; the corresponding
    plug-in estimate ``tau2_hat = quad / n`` is returned alongside.

    Returns
    -------
    (ll, tau2_hat)
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    y_ord = y[approx.order]
    if np.ndim(mu) > 0:
        y_ord = y_ord - np.asarray(mu, dtype=np.float64)[approx.order]
    elif mu != 0.0:
        y_ord = y_ord - mu

    Uvals = create_U_values(approx, tau2=tau2, theta=theta, g=g, v=v, sep=sep)
    Uty = ut_mult_auto(Uvals, approx.NN, approx.NN_len, np.ascontiguousarray(y_ord))
    quad = float(np.dot(Uty, Uty))
    logdet = float(np.log(Uvals[:, 0]).sum())

    if outer:
        ll = logdet - 0.5 * n * math.log(quad)
    else:
        ll = logdet - 0.5 * quad
    return ll, quad / n


# ---------------------------------------------------------------------------
# Metropolis-Hastings, uniform sliding window
# ---------------------------------------------------------------------------
def _mh_propose(value, l, u, rng):
    return rng.uniform(l * value / u, u * value / l)


def sample_theta_vec(y, approx, tau2, theta, g, v, alpha, beta, l, u, outer,
                     ll_prev=None, prior_mean=0.0, sep=False, index=0,
                     rng=None):
    """One MH update of a lengthscale.

    Proposal ``theta* ~ Unif(l*theta/u, u*theta/l)``; prior
    ``Gamma(alpha, rate=beta)`` on ``theta - eps``.  ``theta`` may be a scalar
    (isotropic) or a vector with ``index`` selecting the coordinate updated
    (separable).  Returns ``(theta_new, ll, tau2_hat or None)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64)).copy()
    if not sep:
        index = 0

    if ll_prev is None:
        ll_prev, _ = logl_vec(y, approx, tau2=tau2, theta=theta, g=g, v=v,
                              mu=prior_mean, sep=sep, outer=outer)

    ru = rng.uniform()
    theta_star = theta.copy()
    theta_star[index] = _mh_propose(theta[index], l, u, rng)

    lpost_threshold = (
        ll_prev
        + log_dgamma(theta[index] - EPS, alpha, beta)
        + math.log(ru)
        - math.log(theta[index])
        + math.log(theta_star[index])
    )
    ll_new, tau2_new = logl_vec(y, approx, tau2=tau2, theta=theta_star, g=g,
                                v=v, mu=prior_mean, sep=sep, outer=outer)
    new = ll_new + log_dgamma(theta_star[index] - EPS, alpha, beta)
    if new > lpost_threshold:
        return float(theta_star[index]), ll_new, tau2_new
    return float(theta[index]), ll_prev, None


def sample_g_vec(y, approx, theta, g, v, alpha, beta, l, u, ll_prev=None,
                 sep=False, rng=None):
    """One MH update of the nugget (outer layer only: tau2 = 1, outer=True)."""
    rng = np.random.default_rng() if rng is None else rng
    if ll_prev is None:
        ll_prev, _ = logl_vec(y, approx, tau2=1.0, theta=theta, g=g, v=v,
                              sep=sep, outer=True)
    g_star = _mh_propose(g, l, u, rng)
    ru = rng.uniform()
    lpost_threshold = (
        ll_prev
        + log_dgamma(g - EPS, alpha, beta)
        + math.log(ru)
        - math.log(g)
        + math.log(g_star)
    )
    ll_new, tau2_new = logl_vec(y, approx, tau2=1.0, theta=theta, g=g_star,
                                v=v, sep=sep, outer=True)
    new = ll_new + log_dgamma(g_star - EPS, alpha, beta)
    if new > lpost_threshold:
        return float(g_star), ll_new, tau2_new
    return float(g), ll_prev, None


# ---------------------------------------------------------------------------
# elliptical slice sampling of the latent layers
# ---------------------------------------------------------------------------
_MAX_ESS = 100


def sample_w_vec(y, w_approx, x_approx, tau2_w, theta_y, theta_w, g, v,
                 ll_prev, prior_mean=None, rng=None):
    """ESS update of the latent layer ``W`` given the outer likelihood.

    Each column of ``W`` is updated in turn.  The prior draw comes from
    ``rand_mvn_vec`` on the *previous* layer's approximation (``x_approx`` in
    a two-layer model, ``z_approx`` in a three-layer model), and the slice
    criterion uses the outer, scale-profiled likelihood ``L(Y | W)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    D = w_approx.x_ord.shape[1]
    tau2_w = np.broadcast_to(np.atleast_1d(np.asarray(tau2_w, float)), (D,))
    theta_w = np.broadcast_to(np.atleast_1d(np.asarray(theta_w, float)), (D,))

    ll_new = ll_prev
    tau2_y = None
    for i in range(D):
        pm = 0.0 if prior_mean is None else np.asarray(prior_mean)[:, i]
        w_prior = rand_mvn_vec(x_approx, tau2=tau2_w[i], theta=theta_w[i],
                               g=EPS, v=v, prior_mean=pm, rng=rng)

        a = rng.uniform(0.0, 2.0 * np.pi)
        amin, amax = a - 2.0 * np.pi, a
        ll_threshold = ll_prev + math.log(rng.uniform())

        w_prev = w_approx.x_ord[w_approx.rev_ord_obs, i].copy()
        count = 0
        while True:
            count += 1
            w_proposal = w_prev * math.cos(a) + w_prior * math.sin(a)
            w_approx.x_ord[:, i] = w_proposal[w_approx.order]
            ll_new, tau2_y = logl_vec(y, w_approx, tau2=1.0, theta=theta_y,
                                      g=g, v=v, outer=True)
            if ll_new > ll_threshold:
                ll_prev = ll_new
                break
            if a < 0:
                amin = a
            else:
                amax = a
            a = rng.uniform(amin, amax)
            if count > _MAX_ESS:
                raise RuntimeError("reached maximum iterations of ESS")
    return w_approx, ll_new, tau2_y


def sample_z_vec(w, z_approx, x_approx, tau2_w, tau2_z, theta_w, theta_z, v,
                 ll_prev, rng=None):
    """ESS update of the inner latent layer ``Z`` of a three-layer DGP.

    The slice criterion is the (noise-free) middle-layer likelihood
    ``sum_j L(W_j | Z)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    w = np.atleast_2d(np.asarray(w, dtype=np.float64))
    if w.shape[0] == 1 and z_approx.x_ord.shape[0] != 1:
        w = w.T
    D = z_approx.x_ord.shape[1]
    tau2_w = np.broadcast_to(np.atleast_1d(np.asarray(tau2_w, float)), (D,))
    theta_w = np.broadcast_to(np.atleast_1d(np.asarray(theta_w, float)), (D,))
    tau2_z = np.broadcast_to(np.atleast_1d(np.asarray(tau2_z, float)), (D,))
    theta_z = np.broadcast_to(np.atleast_1d(np.asarray(theta_z, float)), (D,))

    for i in range(D):
        z_prior = rand_mvn_vec(x_approx, tau2=tau2_z[i], theta=theta_z[i],
                               g=EPS, v=v, rng=rng)
        a = rng.uniform(0.0, 2.0 * np.pi)
        amin, amax = a - 2.0 * np.pi, a
        ll_threshold = ll_prev + math.log(rng.uniform())

        z_prev = z_approx.x_ord[z_approx.rev_ord_obs, i].copy()
        count = 0
        while True:
            count += 1
            z_proposal = z_prev * math.cos(a) + z_prior * math.sin(a)
            z_approx.x_ord[:, i] = z_proposal[z_approx.order]
            ll_new = 0.0
            for j in range(D):
                ll_new += logl_vec(w[:, j], z_approx, tau2=tau2_w[j],
                                   theta=theta_w[j], g=EPS, v=v, outer=False)[0]
            if ll_new > ll_threshold:
                ll_prev = ll_new
                break
            if a < 0:
                amin = a
            else:
                amax = a
            a = rng.uniform(amin, amax)
            if count > _MAX_ESS:
                raise RuntimeError("reached maximum iterations of ESS")
    return z_approx, ll_prev
