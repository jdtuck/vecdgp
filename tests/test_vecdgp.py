"""Self-contained correctness tests.

The strategy throughout: when the conditioning sets are complete
(``m = n - 1``) the Vecchia approximation is *exact*, so every approximated
quantity must reproduce the corresponding dense-GP calculation to machine
precision.  Additional tests check the approximation converges as ``m`` grows,
that prior draws have the right moments, and that an end-to-end fit recovers a
known function.
"""

from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# posterior sample paths
# ---------------------------------------------------------------------------
def test_post_sample_moments_match_predict(booth_data):
    """Law of total variance: the paths must reproduce predict(lite=False).

    A path is drawn from N(mu^(t), Sigma^(t)) for each MCMC draw t, so across
    all paths the empirical covariance estimates E[Sigma^(t)] + Cov(mu^(t)) --
    which is exactly what predict assembles analytically.
    """
    x, y, xp, _ = booth_data
    xp = xp[::3]
    fit = fit_two_layer(x, y, nmcmc=1000, true_g=1e-6, verb=False, seed=1,
                        m=10).trim(500, 2)
    paths = fit.post_sample(xp, nper=25, rng=np.random.default_rng(7))
    pred = fit.predict(xp, lite=False, rng=np.random.default_rng(8))

    assert paths.shape == (25 * fit.nmcmc, len(xp))
    sd = np.sqrt(np.diag(pred.Sigma))
    assert np.abs(paths.mean(axis=0) - pred.mean).max() < 0.25 * sd.mean()
    emp = np.cov(paths, rowvar=False)
    assert np.abs(np.sqrt(np.diag(emp)) - sd).max() < 0.25 * sd.mean()
    # the whole point is the off-diagonal structure, not just the margins
    iu = np.triu_indices(len(xp), 1)
    assert np.corrcoef(emp[iu], pred.Sigma[iu])[0, 1] > 0.95


def test_post_sample_paths_match_exact_draws_from_the_joint_covariance(booth_data):
    """Paths must be genuine joint draws, not pointwise-independent ones.

    Marginal moments cannot tell those apart, so this compares a statistic
    that *is* sensitive to joint structure -- mean absolute successive
    difference along a path -- against exact multivariate-normal draws from
    the analytic covariance that ``predict(lite=False)`` returns.

    Note the reference must be the exact draws, not "smoother than
    independent": an interpolating GP has strongly *negative* predictive
    correlation between design points (here adjacent correlation ranges from
    +0.95 to -0.94), so joint paths are not uniformly smoother than
    independent ones and a roughness threshold alone would be misleading.
    """
    x, y, xp, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=600, true_g=1e-6, verb=False, seed=2,
                        m=10).trim(300, 2)
    paths = fit.post_sample(xp, rng=np.random.default_rng(11))
    pred = fit.predict(xp, lite=False, rng=np.random.default_rng(8))

    def roughness(a):
        return np.abs(np.diff(a, axis=1)).mean()

    rng = np.random.default_rng(12)
    L = np.linalg.cholesky(pred.Sigma + 1e-12 * np.eye(len(xp)))
    exact = pred.mean + (L @ rng.standard_normal((len(xp), paths.shape[0]))).T
    independent = pred.mean + rng.standard_normal(paths.shape) * np.sqrt(
        np.diag(pred.Sigma)
    )

    r_joint, r_exact, r_indep = roughness(paths), roughness(exact), roughness(independent)
    assert r_joint == pytest.approx(r_exact, rel=0.05)
    # and the statistic has the power to detect the failure mode it guards
    assert abs(r_joint - r_exact) < 0.25 * abs(r_indep - r_exact), (
        f"roughness {r_joint:.4f} vs exact {r_exact:.4f}, "
        f"pointwise-independent {r_indep:.4f}"
    )


def test_post_sample_is_available_on_every_model(booth_data):
    x, y, xp, _ = booth_data
    xp = xp[::10]
    for fn, kw in ((fit_one_layer, {}), (fit_two_layer, {}), (fit_three_layer, {})):
        fit = fn(x, y, nmcmc=200, true_g=1e-6, verb=False, seed=3, m=8).trim(100, 2)
        paths = fit.post_sample(xp, nper=3, rng=np.random.default_rng(4))
        assert paths.shape == (3 * fit.nmcmc, len(xp))
        assert np.all(np.isfinite(paths))


# ---------------------------------------------------------------------------
# parallelism over MCMC draws
# ---------------------------------------------------------------------------
def test_cores_does_not_change_results(booth_data):
    """`cores` must be a pure speed knob.

    One generator is spawned per draw rather than per worker, so the numbers
    consumed by draw t depend only on the seed and t.  A parallel run that
    quietly returned different answers would be a horrible thing to debug, so
    every path is pinned here -- including the stochastic ones.
    """
    x, y, xp, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=400, true_g=1e-6, verb=False, seed=1,
                        m=10).trim(200, 2)

    def both(fn):
        return fn(1), fn(4)

    a, b = both(lambda c: fit.predict(xp, rng=np.random.default_rng(9), cores=c))
    assert np.array_equal(a.mean, b.mean)
    assert np.array_equal(a.s2, b.s2)

    a, b = both(lambda c: fit.predict(xp[:40], lite=False,
                                      rng=np.random.default_rng(9), cores=c))
    assert np.abs(a.mean - b.mean).max() < 1e-12
    assert np.abs(a.Sigma - b.Sigma).max() < 1e-12  # summation order only

    a, b = both(lambda c: fit.post_sample(xp[:40], nper=3,
                                          rng=np.random.default_rng(9), cores=c))
    assert np.array_equal(a, b)


def test_cores_independence_for_three_layer(booth_data):
    x, y, xp, _ = booth_data
    fit = fit_three_layer(x, y, nmcmc=200, true_g=1e-6, verb=False, seed=2,
                          m=8).trim(100, 2)
    a = fit.predict(xp, rng=np.random.default_rng(3), cores=1)
    b = fit.predict(xp, rng=np.random.default_rng(3), cores=4)
    assert np.array_equal(a.mean, b.mean)
    assert np.array_equal(a.s2, b.s2)


def test_resolve_cores_respects_numba_limit():
    """Asking for more threads than numba allows must clamp, not raise."""
    from vecdgp.predict import resolve_cores

    assert resolve_cores(1) == 1
    assert resolve_cores(None) >= 1
    assert resolve_cores(-1) >= 1
    assert resolve_cores(10_000) == resolve_cores(None)  # capped at what exists
    # a fit must survive an over-large request rather than erroring
    rng = np.random.default_rng(5)
    x = rng.random((40, 1))
    y = np.sin(6 * x.ravel())
    y = (y - y.mean()) / y.std()
    fit = fit_one_layer(x, y, nmcmc=40, true_g=1e-6, verb=False, seed=0, m=8,
                        cores=10_000)
    assert fit.nmcmc == 40


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


# ---------------------------------------------------------------------------
# threading-layer safety
# ---------------------------------------------------------------------------
def test_parallel_prediction_survives_the_workqueue_threading_layer(tmp_path):
    """Guard against nested numba parallelism, which aborts the interpreter.

    Numba's ``parallel=True`` kernels may not be entered from several Python
    threads at once.  Under ``omp``/``tbb`` this is tolerated, but under
    ``workqueue`` -- numba's fallback, and what a stock macOS install
    typically gets -- it kills the process outright with "Concurrent access
    has been detected".  That is why this shipped green on Linux and died on
    macOS CI.

    The layer is fixed at import, so this has to run in a subprocess.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import numpy as np
        from vecdgp import fit_two_layer, fit_three_layer
        rng = np.random.default_rng(0)
        x = rng.random((200, 2)); y = rng.standard_normal(200)
        xp = rng.random((60, 2))
        fit = fit_two_layer(x, y, nmcmc=41, true_g=1e-4, m=8,
                            verb=False, seed=1).trim(21)
        fit.predict(xp, cores=4)
        fit.predict(xp[:25], lite=False, cores=4, rng=np.random.default_rng(2))
        fit.post_sample(xp[:25], nper=2, cores=4, rng=np.random.default_rng(2))
        f3 = fit_three_layer(x, y, nmcmc=21, true_g=1e-4, m=8,
                             verb=False, seed=1).trim(11)
        f3.predict(xp, cores=4)
        print("OK")
    """)
    env = dict(os.environ, NUMBA_THREADING_LAYER="workqueue")
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, env=env, timeout=900)
    assert r.returncode == 0, (
        f"parallel prediction aborted under the workqueue threading layer "
        f"(exit {r.returncode})\\nstdout: {r.stdout}\\nstderr: {r.stderr[-2000:]}"
    )
    assert "OK" in r.stdout


def test_worker_flag_selects_the_serial_kernels():
    """The dispatch itself, checkable on any threading layer."""
    from vecdgp._compat import in_worker, in_worker_thread
    from vecdgp.vecchia import create_U_values

    rng = np.random.default_rng(0)
    ap = create_approx(rng.random((80, 2)), m=12, rng=rng)

    assert not in_worker()
    parallel = create_U_values(ap, 1.0, 0.3, 1e-6, 2.5)
    with in_worker_thread():
        assert in_worker()
        serial = create_U_values(ap, 1.0, 0.3, 1e-6, 2.5)
    assert not in_worker(), "the flag must not leak out of the block"
    # the two kernels must agree exactly, not merely approximately
    assert np.array_equal(parallel, serial)


def test_axis_selection_does_not_collapse_on_many_core_machines():
    """With more cores than draws, only the useful axis may be chosen.

    ``post_sample``'s sampler is sequential over test locations, so it has no
    ``prange`` for numba to parallelise; joint prediction is likewise
    dominated by scipy work.  Handing either to the numba axis wastes the
    machine.  Measured on a 40-core box before this rule existed,
    ``post_sample`` peaked at 1.93x on 8 cores and fell to 1.31x on 32.

    The rule is asserted directly so it is covered on any machine, including
    CI boxes with far fewer cores than the case being guarded.
    """
    from vecdgp.predict import _run_draws

    def axis_for(nmcmc, cores, prefer_draws):
        seen = []
        _run_draws(lambda t, r, w: seen.append(w), nmcmc, cores,
                   np.random.default_rng(0), prefer_draws=prefer_draws)
        return len(set(seen))  # number of distinct workers actually used

    # more draws than cores: both kinds split over draws
    assert axis_for(20, 4, False) == 4
    assert axis_for(20, 4, True) == 4

    # fewer draws than cores: prange-friendly work goes to numba (1 worker),
    # sequential work still splits over the draws it has
    assert axis_for(20, 32, False) == 1, "lite prediction should use the numba axis"
    assert axis_for(20, 32, True) == 20, "post_sample must still use the draws"

    # degenerate cases stay serial
    assert axis_for(1, 32, True) == 1
    assert axis_for(20, 1, True) == 1


def test_post_sample_requests_the_draw_axis():
    """The flag must actually be passed, not just exist."""
    import vecdgp.predict as P

    seen = {}
    orig = P._run_draws

    def spy(body, nmcmc, cores, rng, prefer_draws=False):
        seen[len(seen)] = prefer_draws
        return orig(body, nmcmc, cores, rng, prefer_draws=prefer_draws)

    rng = np.random.default_rng(0)
    x = rng.random((60, 1))
    y = np.sin(6 * x.ravel())
    y = (y - y.mean()) / y.std()
    fit = fit_two_layer(x, y, nmcmc=40, true_g=1e-6, m=8, verb=False,
                        seed=1).trim(20)
    xp = rng.random((15, 1))

    P.predict_deep_vec.__globals__["_run_draws"] = spy
    try:
        seen.clear(); fit.predict(xp)
        assert seen[0] is False, "lite prediction is prange-friendly"
        seen.clear(); fit.predict(xp, lite=False, rng=np.random.default_rng(1))
        assert seen[0] is True, "joint prediction is scipy-bound"
        seen.clear(); fit.post_sample(xp, rng=np.random.default_rng(1))
        assert seen[0] is True, "post_sample is sequential inside"
    finally:
        P.predict_deep_vec.__globals__["_run_draws"] = orig


# ---------------------------------------------------------------------------
# selecting individual posterior draws (calibration-style use)
# ---------------------------------------------------------------------------
def test_draws_selects_individual_posterior_iterations(booth_data):
    """`draws` must pick out iterations without changing what they produce.

    A calibration MCMC needs one emulator sample per outer step, not one per
    retained draw.  Computing all of them and keeping one is pure waste --
    measured at 232 ms vs 3.7 ms for a single location at n = 6000.
    """
    x, y, xp, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=400, true_g=1e-6, verb=False, seed=1,
                        m=10).trim(200, 2)
    assert fit.nmcmc == 100
    xq = xp[:5]

    assert fit.post_sample(xq, draws=3).shape == (1, 5)
    assert fit.post_sample(xq, draws=[3, 8, 20]).shape == (3, 5)
    assert fit.post_sample(xq, draws=-1).shape == (1, 5)
    assert fit.post_sample(xq, nper=4, draws=3).shape == (4, 5)
    assert fit.post_sample(xq).shape == (100, 5)  # default unchanged

    with pytest.raises(IndexError):
        fit.post_sample(xq, draws=100)
    with pytest.raises(IndexError):
        fit.post_sample(xq, draws=-101)
    with pytest.raises(ValueError):
        fit.post_sample(xq, draws=[])


def test_select_is_consistent_with_trim(booth_data):
    """`select` and `trim` must agree where they overlap."""
    x, y, _, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=200, true_g=1e-6, verb=False, seed=2, m=8)
    idx = _thin_idx = np.arange(101, 201)[np.arange(101, 201) % 2 == 0] - 1
    a = fit.trim(100, 2)
    b = fit.select(idx)
    assert a.nmcmc == b.nmcmc
    assert np.array_equal(a.theta_y, b.theta_y)
    assert np.array_equal(a.w, b.w)
    assert np.array_equal(a.tau2_y, b.tau2_y)


def test_selected_draw_uses_that_draws_parameters(booth_data):
    """Picking draw t must actually use iteration t, not some other one."""
    x, y, xp, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=400, true_g=1e-6, verb=False, seed=3,
                        m=10).trim(200, 2)
    for t in (0, 11, 99):
        one = fit.select(t)
        assert one.nmcmc == 1
        assert one.theta_y[0] == fit.theta_y[t]
        assert np.array_equal(one.w[0], fit.w[t])
        assert one.tau2_y[0] == fit.tau2_y[t]


def test_appended_only_conditioning_sets_match_the_full_computation():
    """The post_sample shortcut must be exact, not merely close.

    Only the appended rows are built (the sampler reads nothing else), so
    those rows must equal what the full ordered-NN search produces.
    """
    from vecdgp.vecchia import find_ordered_nn, find_ordered_nn_appended

    rng = np.random.default_rng(0)
    for n_obs, n_new, m in [(800, 120, 25), (300, 300, 15), (500, 1, 30)]:
        x = rng.random((n_obs + n_new, 2))
        full, full_len = find_ordered_nn(x, m)
        app, app_len = find_ordered_nn_appended(x, n_obs, m)
        rows = slice(n_obs, n_obs + n_new)
        assert np.array_equal(full[rows], app[rows])
        assert np.array_equal(full_len[rows], app_len[rows])


def test_partial_conditioning_sets_cannot_be_used_to_build_U():
    """Guard against silent wrongness if the shortcut leaks into the U path."""
    from vecdgp.vecchia import create_U_sparse, create_U_values

    rng = np.random.default_rng(0)
    ap = create_approx(rng.random((200, 2)), m=10, rng=rng)
    ap.add_pred(rng.random((20, 2)), m=20, lite=False, rng=rng,
                pred_rows_only=True)
    for fn in (create_U_values, create_U_sparse):
        with pytest.raises(ValueError, match="predictive rows only"):
            fn(ap, 1.0, 0.3, 1e-6, 2.5)


# ---------------------------------------------------------------------------
# cached sampler for calibration MCMC
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fit_fn", [fit_one_layer, fit_two_layer, fit_three_layer])
def test_sampler_moments_are_identical_to_predict(booth_data, fit_fn):
    """The fast path must be the same calculation, not an approximation of it.

    `sampler` skips post_sample's per-call setup over the training set.  That
    is only legitimate if the conditioning sets and conditional moments come
    out the same, so this demands exact equality with predict(draws=t), not
    closeness.
    """
    x, y, xp, _ = booth_data
    fit = fit_fn(x, y, nmcmc=300, true_g=1e-6, verb=False, seed=1,
                 m=8).trim(150, 2)
    emu = fit.sampler()
    for t in (0, 7, fit.nmcmc - 1):
        mu, sd = emu.mean_sd(xp[:9], draw=t)
        p = fit.predict(xp[:9], draws=t, cores=1)
        assert np.array_equal(mu, p.mean)
        assert np.allclose(sd, np.sqrt(p.s2), rtol=0, atol=0)


def test_sampler_draws_have_the_right_moments(booth_data):
    x, y, xp, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=300, true_g=1e-6, verb=False, seed=2,
                        m=8).trim(150, 2)
    emu = fit.sampler()
    t, q = 5, xp[:1]
    mu, sd = emu.mean_sd(q, draw=t)
    s = emu.sample(q, draw=t, rng=np.random.default_rng(3), nper=60000)
    assert s.shape == (60000, 1)
    se = sd[0] / np.sqrt(60000)
    assert abs(s.mean() - mu[0]) < 5 * se
    assert abs(s.std() - sd[0]) < 0.05 * sd[0]


def test_sampler_matches_post_sample_distribution(booth_data):
    """Against the reference path, at the same posterior draw."""
    x, y, xp, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=300, true_g=1e-6, verb=False, seed=3,
                        m=8).trim(150, 2)
    emu = fit.sampler()
    t, q = 11, xp[:1]
    a = emu.sample(q, draw=t, rng=np.random.default_rng(4), nper=40000)
    b = fit.post_sample(q, draws=t, nper=40000, rng=np.random.default_rng(5),
                        cores=1)
    assert abs(a.mean() - b.mean()) < 0.05 * b.std()
    assert abs(a.std() - b.std()) < 0.05 * b.std()


def test_sampler_handles_several_locations_at_once(booth_data):
    """Multiple points condition on each other, as the sequential path does."""
    x, y, xp, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=200, true_g=1e-6, verb=False, seed=4,
                        m=8).trim(100, 2)
    emu = fit.sampler()
    q = xp[:6]
    s = emu.sample(q, draw=2, rng=np.random.default_rng(6), nper=20000)
    assert s.shape == (20000, 6)
    mu, sd = emu.mean_sd(q, draw=2)
    # marginals still match; the joint adds the cross-conditioning
    assert np.abs(s.mean(axis=0) - mu).max() < 0.1 * sd.mean()
    # neighbouring locations must come out correlated, not independent
    c = np.corrcoef(s, rowvar=False)
    assert np.abs(c[np.triu_indices(6, 1)]).max() > 0.2


def test_sampler_cache_is_lazy_and_bounded(booth_data):
    x, y, xp, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=200, true_g=1e-6, verb=False, seed=5,
                        m=8).trim(100, 2)
    emu = fit.sampler(max_cached=3)
    assert len(emu._cache) == 0, "state must not be built until a draw is used"
    for t in range(6):
        emu.sample(xp[:1], draw=t, rng=np.random.default_rng(t))
    assert len(emu._cache) <= 3, "max_cached must bound the cache"
    # evicted draws still work, and still give the same moments
    mu, _ = emu.mean_sd(xp[:1], draw=0)
    assert np.array_equal(mu, fit.predict(xp[:1], draws=0, cores=1).mean)


def test_sampler_draw_index_wraps_and_validates(booth_data):
    x, y, xp, _ = booth_data
    fit = fit_two_layer(x, y, nmcmc=200, true_g=1e-6, verb=False, seed=6,
                        m=8).trim(100, 2)
    emu = fit.sampler()
    a, _ = emu.mean_sd(xp[:1], draw=-1)
    b, _ = emu.mean_sd(xp[:1], draw=emu.nmcmc - 1)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        emu.sample(np.zeros((1, 5)), draw=0)  # wrong input dimension
