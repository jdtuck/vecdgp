"""Self-contained correctness tests.

The strategy throughout: when the conditioning sets are complete
(``m = n - 1``) the Vecchia approximation is *exact*, so every approximated
quantity must reproduce the corresponding dense-GP calculation to machine
precision.  Additional tests check the approximation converges as ``m`` grows,
that prior draws have the right moments, and that an end-to-end fit recovers a
known function.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import multivariate_normal

from vecdgp import (
    cov_matrix,
    create_approx,
    create_U_sparse,
    create_U_values,
    cross_cov,
    fit_one_layer,
    fit_three_layer,
    fit_two_layer,
    find_ordered_nn,
    krig_vec,
    logl_vec,
    rand_mvn_vec,
    rmse,
    score,
)
from vecdgp.vecchia import forward_solve_ut, ut_mult

V_CASES = [0.5, 1.5, 2.5, 999.0]


def _data(n=60, d=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, d))
    y = rng.standard_normal(n)
    return x, y, rng


# ---------------------------------------------------------------------------
# nearest-neighbour construction
# ---------------------------------------------------------------------------
def test_find_ordered_nn_matches_brute_force():
    rng = np.random.default_rng(3)
    x = rng.random((200, 2))
    m = 12
    NN, NN_len = find_ordered_nn(x, m)
    assert NN[:, 0].tolist() == list(range(200))
    for i in range(200):
        k = min(m, i)
        assert NN_len[i] == k + 1
        cand = NN[i, 1 : 1 + k]
        assert np.all(cand < i), "conditioning sets must reference predecessors"
        assert len(set(cand.tolist())) == k
        d2 = np.sum((x[:i] - x[i]) ** 2, axis=1)
        expected = np.sort(np.sort(d2)[:k])
        assert np.allclose(np.sort(d2[cand]), expected)


# ---------------------------------------------------------------------------
# the sparse Cholesky factor U
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("v", V_CASES)
def test_U_factorises_the_precision_matrix(v):
    """With full conditioning sets, U U^T must equal Sigma^{-1} exactly."""
    x, _, rng = _data(n=50, d=2, seed=1)
    ap = create_approx(x, m=49, rng=rng)
    U = create_U_sparse(ap, tau2=2.0, theta=0.4, g=1e-4, v=v).toarray()
    Sigma = cov_matrix(ap.x_ord, tau2=2.0, theta=0.4, g=1e-4, v=v)
    assert np.abs(U @ U.T @ Sigma - np.eye(50)).max() < 1e-8
    assert np.allclose(np.triu(U), U), "U must be upper triangular"


def test_U_separable_lengthscales():
    x, _, rng = _data(n=45, d=3, seed=7)
    ap = create_approx(x, m=44, rng=rng)
    theta = np.array([0.2, 0.5, 1.0])
    U = create_U_sparse(ap, 1.0, theta, 1e-5, 2.5, sep=True).toarray()
    Sigma = cov_matrix(ap.x_ord, 1.0, theta, 1e-5, 2.5, sep=True)
    assert np.abs(U @ U.T @ Sigma - np.eye(45)).max() < 1e-8


def test_forward_solve_is_a_triangular_solve():
    x, _, rng = _data(n=40, seed=2)
    ap = create_approx(x, m=15, rng=rng)
    Uv = create_U_values(ap, 1.0, 0.3, 1e-6, 2.5)
    U = create_U_sparse(ap, 1.0, 0.3, 1e-6, 2.5).toarray()
    z = rng.standard_normal(40)
    y = forward_solve_ut(Uv, ap.NN, ap.NN_len, z)
    assert np.abs(U.T @ y - z).max() < 1e-9
    assert np.abs(ut_mult(Uv, ap.NN, ap.NN_len, y) - U.T @ y).max() < 1e-12


# ---------------------------------------------------------------------------
# likelihood
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("v", V_CASES)
def test_loglik_exact_when_conditioning_sets_are_full(v):
    x, y, rng = _data(n=55, d=2, seed=4)
    ap = create_approx(x, m=54, rng=rng)
    Sigma = cov_matrix(ap.x_ord, tau2=1.0, theta=0.35, g=1e-3, v=v)
    exact = multivariate_normal.logpdf(y[ap.order], mean=np.zeros(55), cov=Sigma)
    ll, _ = logl_vec(y, ap, tau2=1.0, theta=0.35, g=1e-3, v=v)
    # logl_vec drops the -n/2 log(2 pi) normalising constant
    assert ll == pytest.approx(exact + 0.5 * 55 * np.log(2 * np.pi), rel=1e-9)


def _chol_logdet_and_quad(Sigma, v):
    """Reference log|Sigma| and v' Sigma^-1 v for an SPD Sigma.

    Cholesky, not ``slogdet``/``solve``: the determinant of a GP covariance
    underflows hard (~1e-92 for n=50) and a general LU factorisation can trip
    divide-by-zero / overflow warnings on some LAPACK builds even when the
    matrix is well conditioned.  Only the log determinant is ever formed.
    """
    L = np.linalg.cholesky(Sigma)
    ldet = 2.0 * np.log(np.diag(L)).sum()
    a = solve_triangular(L, v, lower=True)
    return ldet, a @ a


def test_profile_loglik_and_tau2_hat():
    x, y, rng = _data(n=50, seed=5)
    ap = create_approx(x, m=49, rng=rng)
    ll, tau2 = logl_vec(y, ap, tau2=1.0, theta=0.3, g=1e-3, v=2.5, outer=True)
    Sigma = cov_matrix(ap.x_ord, 1.0, 0.3, 1e-3, 2.5)
    yo = y[ap.order]
    ldet, quad = _chol_logdet_and_quad(Sigma, yo)
    assert tau2 == pytest.approx(quad / 50, rel=1e-9)
    assert ll == pytest.approx(-0.5 * ldet - 25 * np.log(quad), rel=1e-9)


def test_loglik_converges_as_m_grows():
    x, y, rng = _data(n=120, d=2, seed=6)
    order = rng.permutation(120)
    Sigma = cov_matrix(x[order], 1.0, 0.5, 1e-4, 2.5)
    exact = multivariate_normal.logpdf(
        y[order], mean=np.zeros(120), cov=Sigma
    ) + 0.5 * 120 * np.log(2 * np.pi)
    errs = []
    for m in (3, 10, 30, 119):
        ap = create_approx(x, m=m, order=order)
        ll = logl_vec(y, ap, 1.0, 0.5, 1e-4, 2.5)[0]
        errs.append(abs(ll - exact) / abs(exact))
    assert errs[-1] < 1e-10, "full conditioning sets must be exact"
    assert errs[0] > errs[1] > errs[2] > errs[3], "error must shrink with m"


# ---------------------------------------------------------------------------
# prior sampling
# ---------------------------------------------------------------------------
def test_prior_draws_have_the_right_moments():
    x, _, rng = _data(n=30, d=1, seed=8)
    ap = create_approx(x, m=29, rng=rng)
    draws = np.array(
        [rand_mvn_vec(ap, tau2=1.5, theta=0.4, g=1e-6, v=2.5, rng=rng)
         for _ in range(40000)]
    )
    Sigma = cov_matrix(x, tau2=1.5, theta=0.4, g=1e-6, v=2.5)
    emp = np.cov(draws, rowvar=False)
    assert np.abs(draws.mean(axis=0)).max() < 0.05
    assert np.abs(emp - Sigma).max() < 0.05


def test_prior_draws_respect_a_prior_mean():
    x, _, rng = _data(n=25, d=1, seed=9)
    ap = create_approx(x, m=24, rng=rng)
    pm = x[:, 0]
    draws = np.array(
        [rand_mvn_vec(ap, 1.0, 0.3, 1e-6, 2.5, prior_mean=pm, rng=rng)
         for _ in range(5000)]
    )
    assert np.abs(draws.mean(axis=0) - pm).max() < 0.1


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
def _exact_krig(x, y, x_new, tau2, theta, g, v):
    K = cov_matrix(x, 1.0, theta, g, v)
    k = cross_cov(x_new, x, 1.0, theta, v)
    c = cho_factor(K, lower=True)
    mu = k @ cho_solve(c, y)
    s2 = tau2 * (1.0 + g - np.einsum("ij,ij->i", k, cho_solve(c, k.T).T))
    return mu, s2


@pytest.mark.parametrize("v", V_CASES)
def test_lite_prediction_matches_exact_kriging(v):
    x, y, rng = _data(n=40, d=2, seed=10)
    x_new = rng.random((17, 2))
    ap = create_approx(x, m=39, rng=rng)
    ap.add_pred(x_new, m=40, lite=True)  # condition on every training point
    out = krig_vec(y, ap, tau2=2.0, theta=0.4, g=1e-4, v=v, s2=True)
    mu, s2 = _exact_krig(x, y, x_new, 2.0, 0.4, 1e-4, v)
    assert np.abs(out["mean"] - mu).max() < 1e-8
    assert np.abs(out["s2"] - s2).max() < 1e-8


def test_joint_prediction_matches_exact_kriging():
    x, y, rng = _data(n=35, d=2, seed=11)
    x_new = rng.random((12, 2))
    ap = create_approx(x, m=34, rng=rng)
    ap.add_pred(x_new, m=35 + 12 - 1, lite=False, rng=rng)
    out = krig_vec(y, ap, tau2=1.7, theta=0.5, g=1e-4, v=2.5, sigma=True)

    K = cov_matrix(x, 1.0, 0.5, 1e-4, 2.5)
    k = cross_cov(x_new, x, 1.0, 0.5, 2.5)
    Knew = cov_matrix(x_new, 1.0, 0.5, 1e-4, 2.5)
    c = cho_factor(K, lower=True)
    mu = k @ cho_solve(c, y)
    Sig = 1.7 * (Knew - k @ cho_solve(c, k.T))
    assert np.abs(out["mean"] - mu).max() < 1e-7
    assert np.abs(out["sigma"] - Sig).max() < 1e-7


def test_lite_and_joint_prediction_agree_on_the_mean():
    x, y, rng = _data(n=60, d=2, seed=12)
    x_new = rng.random((25, 2))
    ap = create_approx(x, m=59, rng=rng)
    ap.add_pred(x_new, m=60, lite=True)
    lite = krig_vec(y, ap, 1.0, 0.4, 1e-4, 2.5, s2=True)
    ap.clean_pred()
    ap.add_pred(x_new, m=84, lite=False, rng=rng)
    joint = krig_vec(y, ap, 1.0, 0.4, 1e-4, 2.5, sigma=True)
    assert np.abs(lite["mean"] - joint["mean"]).max() < 1e-7
    assert np.abs(lite["s2"] - np.diag(joint["sigma"])).max() < 1e-7


def test_sequential_posterior_samples_have_the_right_moments():
    x, y, rng = _data(n=30, d=1, seed=13)
    x_new = np.linspace(0.05, 0.95, 8)[:, None]
    ap = create_approx(x, m=29, rng=rng)
    ap.add_pred(x_new, m=37, lite=False, rng=rng)
    samples = krig_vec(y, ap, tau2=1.0, theta=0.3, g=1e-4, v=2.5,
                       nsamples=30000, rng=rng)["samples"]
    joint = krig_vec(y, ap, tau2=1.0, theta=0.3, g=1e-4, v=2.5, sigma=True)
    assert np.abs(samples.mean(axis=0) - joint["mean"]).max() < 0.03
    assert np.abs(np.cov(samples, rowvar=False) - joint["sigma"]).max() < 0.02


# ---------------------------------------------------------------------------
# numerical robustness of the scoring rules
# ---------------------------------------------------------------------------
def test_score_is_stable_when_the_determinant_underflows():
    """A GP covariance has a vanishing determinant long before it is singular.

    ``det`` of a smooth, small-nugget covariance underflows to exactly 0 in
    float64 while the matrix is still perfectly well conditioned.  ``score``
    must therefore never form the determinant -- only its log -- and must not
    route an SPD matrix through a general LU factorisation, which can emit
    divide-by-zero / overflow / invalid warnings on some LAPACK builds.
    """
    rng = np.random.default_rng(77)
    n = 150
    x = rng.random((n, 2))
    S = cov_matrix(x, 1.0, 1.0, 1e-6, 2.5)

    # Establish the premise without ever calling np.linalg.det -- that is the
    # very trap under test, and it is not portable: it returns 0.0 under
    # OpenBLAS but warns and can return nan under MKL, so `det(S) == 0.0`
    # would itself be a platform-dependent assertion.
    ev = np.linalg.eigvalsh(S)
    ldet = np.log(ev).sum()
    assert ldet < np.log(np.nextafter(0, 1)), (
        "premise: the determinant must be unrepresentable in float64 "
        f"(log det = {ldet:.0f}, underflow below {np.log(np.nextafter(0, 1)):.0f})"
    )
    assert ev.min() > 0, "premise: but the matrix is still positive definite"

    y = rng.standard_normal(n)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any numerical warning fails the test
        s = score(y, np.zeros(n), S)
    assert np.isfinite(s)

    # independent cross-check via an eigendecomposition
    ev, Q = np.linalg.eigh(S)
    ldet = np.log(ev).sum()
    quad = ((Q.T @ y) ** 2 / ev).sum()
    assert s == pytest.approx((-ldet - quad) / n, rel=1e-6)


def test_score_matches_the_multivariate_normal_density():
    rng = np.random.default_rng(78)
    n = 60
    x = rng.random((n, 2))
    S = cov_matrix(x, 1.3, 0.4, 1e-3, 2.5)
    mu = rng.standard_normal(n)
    y = rng.standard_normal(n)
    logpdf = multivariate_normal.logpdf(y, mean=mu, cov=S)
    expected = (2.0 * logpdf + n * np.log(2 * np.pi)) / n
    assert score(y, mu, S) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def _booth(x):
    """Piecewise test function from the deepgp vignette (nonstationary)."""
    x = np.ravel(x)
    return np.where(x <= 0.58, np.sin(np.pi * x * 6) + np.cos(np.pi * x * 12),
                    5 * x - 4.9)


@pytest.fixture(scope="module")
def booth_data():
    x = np.linspace(0, 1, 20)[:, None]
    y = _booth(x)
    mu, sd = y.mean(), y.std()
    xp = np.linspace(0, 1, 100)[:, None]
    return x, (y - mu) / sd, xp, (_booth(xp) - mu) / sd


def test_two_layer_beats_one_layer_on_a_nonstationary_function(booth_data):
    x, y, xp, yp = booth_data
    gp = fit_one_layer(x, y, nmcmc=2000, true_g=1e-6, verb=False, seed=1,
                       m=10).trim(1000, 2)
    dgp = fit_two_layer(x, y, nmcmc=2000, true_g=1e-6, verb=False, seed=1,
                        m=10).trim(1000, 2)
    r_gp = rmse(yp, gp.predict(xp).mean)
    r_dgp = rmse(yp, dgp.predict(xp).mean)
    assert r_dgp < r_gp
    assert r_dgp < 0.1


def test_three_layer_runs_and_fits(booth_data):
    x, y, xp, yp = booth_data
    dgp3 = fit_three_layer(x, y, nmcmc=1000, true_g=1e-6, verb=False, seed=2,
                           m=10).trim(500, 2)
    p = dgp3.predict(xp)
    assert p.mean.shape == (100,)
    assert np.all(p.s2 > 0)
    assert rmse(yp, p.mean) < 0.2


def test_nugget_estimation_recovers_the_noise_level():
    rng = np.random.default_rng(21)
    n = 120
    x = np.sort(rng.random(n))[:, None]
    truth = np.sin(2 * np.pi * x).ravel()
    sd = 0.15
    y = truth + sd * rng.standard_normal(n)
    mu, s = y.mean(), y.std()
    y = (y - mu) / s
    fit = fit_one_layer(x, y, nmcmc=1500, verb=False, seed=3, m=15).trim(750, 2)
    # tau2 * g estimates the noise variance on the standardised scale
    est = np.nanmean(fit.tau2 * fit.g)
    assert est == pytest.approx((sd / s) ** 2, rel=0.6)


def test_trim_and_thin(booth_data):
    x, y, xp, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=200, true_g=1e-6, verb=False, seed=4, m=8)
    trimmed = fit.trim(100, 5)
    assert trimmed.nmcmc == 20
    assert trimmed.w.shape == (20, 20, 1)
    assert trimmed.theta_y.shape == (20,)
    assert fit.nmcmc == 200  # original untouched


def test_separable_lengthscales_run():
    rng = np.random.default_rng(31)
    x = rng.random((60, 2))
    y = np.sin(3 * x[:, 0]) + 0.2 * x[:, 1]
    y = (y - y.mean()) / y.std()
    fit = fit_one_layer(x, y, nmcmc=300, sep=True, true_g=1e-6, verb=False,
                        seed=5, m=10).trim(150)
    assert fit.theta.shape == (150, 2)
    p = fit.predict(rng.random((10, 2)))
    assert p.mean.shape == (10,)


def test_wide_latent_layer():
    rng = np.random.default_rng(41)
    x = rng.random((50, 2))
    y = np.sin(4 * x[:, 0] * x[:, 1])
    y = (y - y.mean()) / y.std()
    fit = fit_two_layer(x, y, D=3, nmcmc=300, true_g=1e-6, verb=False, seed=6,
                        m=10).trim(150)
    assert fit.w.shape == (150, 50, 3)
    p = fit.predict(rng.random((7, 2)), store_latent=True)
    assert p.w_new.shape == (150, 7, 3)
