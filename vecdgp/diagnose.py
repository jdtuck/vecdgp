"""Environment and numerical self-check.

Run with::

    python -m vecdgp.diagnose

Prints library versions and the BLAS/LAPACK backend, then runs the core
exactness identities and reports the actual error magnitudes.  Any numerical
warning raised along the way is captured and shown rather than swallowed.

Useful when results or warnings differ between machines: the numerical
behaviour of a GP covariance depends on the LAPACK build underneath numpy,
and the speed depends on whether numba is present.
"""

from __future__ import annotations

import platform
import sys
import warnings

import numpy as np


def _versions():
    print("environment")
    print("-" * 60)
    print(f"  python  {sys.version.split()[0]}  ({platform.platform()})")
    print(f"  numpy   {np.__version__}")
    try:
        import scipy

        print(f"  scipy   {scipy.__version__}")
    except Exception as e:  # pragma: no cover
        print(f"  scipy   MISSING ({e})")
    try:
        import numba

        print(f"  numba   {numba.__version__}")
        from numba import config

        print(f"  numba threads: {numba.get_num_threads()}  "
              f"(of {config.NUMBA_NUM_THREADS} max)")
        _svml_report(config)
    except Exception:
        print("  numba   NOT INSTALLED  -> pure-Python fallback, ~100x slower")

    # which LAPACK numpy is linked against: this drives the numerical
    # differences people see in slogdet / solve on near-underflowing matrices
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        lapack = cfg.get("Build Dependencies", {}).get("lapack", {})
        print(f"  BLAS    {blas.get('name', '?')} {blas.get('version', '')}")
        print(f"  LAPACK  {lapack.get('name', '?')} {lapack.get('version', '')}")
    except Exception:
        print("  BLAS/LAPACK: could not determine (older numpy)")
    print()


def _svml_report(config):
    """Report SVML status, and say *why* when it is off.

    SVML gives LLVM vectorised ``exp``/``sqrt``.  Those dominate this
    package's hot loop -- a 26x26 conditioning block is ~350 transcendentals
    against ~5 900 flops of Cholesky -- so enabling it is the single largest
    remaining speedup available.  A bare "False" is not actionable, because
    there are two quite different reasons for it.
    """
    if config.USING_SVML:
        print("  numba SVML (vectorised exp/log): ENABLED")
        return
    print("  numba SVML (vectorised exp/log): disabled", end="")
    if config.DISABLE_INTEL_SVML:
        print("  <- NUMBA_DISABLE_INTEL_SVML is set")
        return
    try:
        import llvmlite.binding as llb

        has = getattr(llb.targets, "has_svml", None)
        if has is None:
            print("  <- llvmlite too old to report SVML support")
            return
        if not has():
            print()
            print("      reason: this llvmlite was built WITHOUT LLVM's SVML")
            print("      patch, so LLVM cannot emit vectorised calls no matter")
            print("      what is installed. Installing libsvml/intel-cmplr-lib-rt")
            print("      will NOT help. Check with:")
            print("        python -c \"import llvmlite.binding as b; "
                  "print(b.targets.has_svml())\"")
            print("      A numba/llvmlite build that ships the patch is needed.")
            return
        print()
        print("      llvmlite supports SVML but libsvml.so did not load;")
        print("      check LD_LIBRARY_PATH / run ldconfig.")
    except Exception as e:  # pragma: no cover
        print(f"  <- could not determine ({e})")


def _check(name, got, tol, fmt="{:.2e}"):
    ok = got < tol
    flag = "ok  " if ok else "FAIL"
    print(f"  [{flag}] {name:<52} {fmt.format(got)}  (tol {fmt.format(tol)})")
    return ok


def main():
    from . import (
        HAVE_NUMBA,
        cov_matrix,
        create_approx,
        create_U_sparse,
        krig_vec,
        logl_vec,
        score,
    )
    from .vecchia import create_U_values, forward_solve_ut

    _versions()
    print(f"vecdgp using numba: {HAVE_NUMBA}")
    print()

    rng = np.random.default_rng(0)
    n = 50
    x = rng.random((n, 2))
    y = rng.standard_normal(n)
    ok = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        print("numerical identities (exact when m = n-1)")
        print("-" * 60)

        # 1. U U^T = Sigma^{-1}
        ap = create_approx(x, m=n - 1, rng=rng)
        U = create_U_sparse(ap, 1.0, 0.3, 1e-3, 2.5).toarray()
        S = cov_matrix(ap.x_ord, 1.0, 0.3, 1e-3, 2.5)
        ok &= _check("U U^T Sigma = I", np.abs(U @ U.T @ S - np.eye(n)).max(), 1e-7)

        # 2. log-likelihood vs the dense Gaussian
        L = np.linalg.cholesky(S)
        yo = y[ap.order]
        from scipy.linalg import solve_triangular

        a = solve_triangular(L, yo, lower=True)
        exact = -np.log(np.diag(L)).sum() - 0.5 * (a @ a)
        ll, _ = logl_vec(y, ap, 1.0, 0.3, 1e-3, 2.5)
        ok &= _check("log-likelihood vs dense GP (relative)",
                     abs(ll - exact) / abs(exact), 1e-9)

        # 3. forward solve
        Uv = create_U_values(ap, 1.0, 0.3, 1e-3, 2.5)
        z = rng.standard_normal(n)
        sol = forward_solve_ut(Uv, ap.NN, ap.NN_len, z)
        ok &= _check("forward solve residual", np.abs(U.T @ sol - z).max(), 1e-8)

        # 4. prediction vs exact kriging
        x_new = rng.random((10, 2))
        ap.add_pred(x_new, m=n, lite=True)
        out = krig_vec(y, ap, 1.0, 0.3, 1e-3, 2.5, s2=True)
        from . import cross_cov
        from scipy.linalg import cho_factor, cho_solve

        K = cov_matrix(x, 1.0, 0.3, 1e-3, 2.5)
        k = cross_cov(x_new, x, 1.0, 0.3, 2.5)
        c = cho_factor(K, lower=True)
        ok &= _check("prediction vs exact kriging",
                     np.abs(out["mean"] - k @ cho_solve(c, y)).max(), 1e-7)

        # 5. the scoring rule on a covariance whose determinant underflows
        xb = rng.random((150, 2))
        Sb = cov_matrix(xb, 1.0, 1.0, 1e-6, 2.5)
        yb = rng.standard_normal(150)
        s = score(yb, np.zeros(150), Sb)
        ev, Q = np.linalg.eigh(Sb)
        ref = (-np.log(ev).sum() - ((Q.T @ yb) ** 2 / ev).sum()) / 150
        # deliberately NOT np.linalg.det: that is the trap being tested, and
        # calling it here would emit the very warnings this check exists to
        # rule out. The log determinant says the same thing, portably.
        ldet = np.log(ev).sum()
        print(f"         (that covariance has log det = {ldet:.0f}, i.e. det "
              f"underflows float64 below {np.log(np.nextafter(0, 1)):.0f}; "
              f"cond = {ev.max() / ev.min():.2e})")
        ok &= _check("score() vs eigendecomposition (relative)",
                     abs(s - ref) / abs(ref), 1e-5)

    print()
    if caught:
        print(f"warnings raised ({len(caught)})")
        print("-" * 60)
        for w in caught:
            print(f"  {w.category.__name__}: {w.message}")
            print(f"    at {w.filename}:{w.lineno}")
    else:
        print("no warnings raised")

    print()
    print("RESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
