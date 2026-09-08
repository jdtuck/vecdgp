"""vecdgp -- Vecchia-approximated deep Gaussian processes in Python.

A Python implementation of

    A. Sauer, A. Cooper and R. B. Gramacy (2023),
    *Vecchia-approximated Deep Gaussian Processes for Computer Experiments*,
    Journal of Computational and Graphical Statistics (arXiv:2204.02904),

following the reference R package ``deepgp`` (Sauer) with ``vecchia = TRUE``.

Quick start
-----------
>>> import numpy as np
>>> from vecdgp import fit_two_layer
>>> x = np.linspace(0, 1, 40)[:, None]
>>> y = np.sin(6 * np.pi * x).ravel()
>>> y = (y - y.mean()) / y.std()
>>> fit = fit_two_layer(x, y, nmcmc=500, true_g=1e-6, verb=False, seed=0)
>>> fit = fit.trim(250)
>>> p = fit.predict(np.linspace(0, 1, 100)[:, None])
>>> p.mean.shape, p.s2.shape
((100,), (100,))

``predict`` returns summarised moments; ``post_sample`` returns whole
posterior draws of the surface, for functionals a pointwise band cannot
express (argmax distributions, excursion probabilities, downstream
propagation):

>>> paths = fit.post_sample(np.linspace(0, 1, 100)[:, None])
>>> paths.shape
(250, 100)
"""

from ._compat import HAVE_NUMBA
from .fit import (
    DGP2Vec,
    DGP3Vec,
    GPVec,
    fit_one_layer,
    fit_three_layer,
    fit_two_layer,
)
from .kernels import EXP2, cov_matrix, cross_cov, sq_dist
from .krig import krig_vec
from .mcmc import logl_vec
from .metrics import crps, rmse, safe_cholesky, score
from .settings import Settings, default_settings
from .vecchia import (
    EPS,
    VecchiaApprox,
    create_approx,
    create_U_sparse,
    create_U_values,
    find_ordered_nn,
    rand_mvn_vec,
)

__version__ = "0.3.1"

__all__ = [
    "fit_one_layer",
    "fit_two_layer",
    "fit_three_layer",
    "GPVec",
    "DGP2Vec",
    "DGP3Vec",
    "krig_vec",
    "logl_vec",
    "create_approx",
    "create_U_values",
    "create_U_sparse",
    "find_ordered_nn",
    "rand_mvn_vec",
    "VecchiaApprox",
    "cov_matrix",
    "cross_cov",
    "sq_dist",
    "rmse",
    "crps",
    "score",
    "safe_cholesky",
    "Settings",
    "default_settings",
    "EPS",
    "EXP2",
    "HAVE_NUMBA",
    "__version__",
]
