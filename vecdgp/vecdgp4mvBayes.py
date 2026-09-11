"""
mvBayes-compatible wrapper for Vecchia deep Gaussian process emulators.

This wrapper is intended for a package exposing:
- fit_one_layer
- fit_two_layer
- fit_three_layer

and fitted objects supporting:
- .post_sample(...)
- .sampler(...)

Behavior
--------
- Supports 1-, 2-, and 3-layer models via `layers`
- Automatically scales X to [0, 1] by column
- Automatically centers/scales y to mean 0, variance 1
- Unscales predictions back to the original y scale
- Returns posterior-sample-like draws in shape (n_samples, n_obs), compatible with mvBayes
- Uses either:
    * post_sample(...)
    * cached sampler from calibrate.py
  depending on `sample_method`
- Sets .samples.residSD = 0, since predictive sampling is treated as already
  containing the full uncertainty to propagate
- Automatically manages MCMC length via retained-draw count, burn-in, and thinning

Default choices
---------------
- layers = 2
- sample_method = "auto"
- nper = 1
- nmcmc_keep = 1000
- burn = 5000
- thin = 5
- auto mode uses cached sampler when Xtest has one row, else post_sample
"""

import numpy as np

from .fit import fit_one_layer, fit_two_layer, fit_three_layer


class _MvBayesVecDGPSamples:
    """Simple container for mvBayes-compatible posterior sample attributes."""
    pass


class MvBayesVecDGPWrapper:
    """
    mvBayes-compatible wrapper around Vecchia GP / DGP fitted models.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix.
    y : np.ndarray
        Univariate response.
    layers : int, default=2
        Number of layers: 1, 2, or 3.
    sample_method : {"auto", "post_sample", "cached_sampler"}, default="auto"
        Method used to generate predictive draws.
    nper : int, default=1
        Draws per retained posterior iteration.
    sampler_kwargs : dict or None, default=None
        Optional keyword arguments passed to fit.sampler(...) when using cached sampling.
    nmcmc_keep : int, default=1000
        Number of retained posterior draws after burn-in and thinning.
    burn : int, default=5000
        Number of initial MCMC draws to discard.
    thin : int, default=5
        Keep every `thin`-th MCMC draw after burn-in.
    **kwargs
        Additional keyword arguments passed to the chosen fit_*_layer function,
        excluding `nmcmc`, which is managed by the wrapper.
    """

    def __init__(
        self,
        X,
        y,
        layers=2,
        sample_method="auto",
        nper=1,
        sampler_kwargs=None,
        nmcmc_keep=1000,
        burn=5000,
        thin=5,
        **kwargs,
    ):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 2D or coercible to 2D.")

        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            y = np.squeeze(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array or coercible to 1D.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if layers not in (1, 2, 3):
            raise ValueError("layers must be one of {1, 2, 3}.")

        if sample_method not in ("auto", "post_sample", "cached_sampler"):
            raise ValueError(
                "sample_method must be one of {'auto', 'post_sample', 'cached_sampler'}."
            )

        if not isinstance(nper, int) or nper <= 0:
            raise ValueError("nper must be a positive integer.")

        if not isinstance(nmcmc_keep, int) or nmcmc_keep <= 0:
            raise ValueError("nmcmc_keep must be a positive integer.")
        if not isinstance(burn, int) or burn < 0:
            raise ValueError("burn must be a nonnegative integer.")
        if not isinstance(thin, int) or thin <= 0:
            raise ValueError("thin must be a positive integer.")

        self.layers = layers
        self.sample_method = sample_method
        self.nper = nper
        self.sampler_kwargs = {} if sampler_kwargs is None else dict(sampler_kwargs)
        self.nmcmc_keep = nmcmc_keep
        self.burn = burn
        self.thin = thin

        # Scale X to [0, 1] by column
        self.X_min_ = X.min(axis=0)
        self.X_range_ = X.max(axis=0) - self.X_min_
        self.X_range_[self.X_range_ == 0] = 1.0
        X_scaled = (X - self.X_min_) / self.X_range_

        # Center/scale y
        self.y_mean_ = float(np.mean(y))
        self.y_sd_ = float(np.std(y))
        if self.y_sd_ == 0:
            self.y_sd_ = 1.0
        y_scaled = (y - self.y_mean_) / self.y_sd_

        # Prevent user from passing nmcmc directly; wrapper manages it
        if "nmcmc" in kwargs:
            raise ValueError(
                "Do not pass 'nmcmc' directly to MvBayesVecDGPWrapper. "
                "Use nmcmc_keep, burn, and thin instead."
            )

        # Total MCMC iterations needed so that after burn-in and thinning we retain
        # at least nmcmc_keep samples.
        nmcmc_total = burn + nmcmc_keep * thin

        fit_kwargs = dict(kwargs)
        fit_kwargs["nmcmc"] = nmcmc_total

        if layers == 1:
            fit_obj = fit_one_layer(X_scaled, y_scaled, **fit_kwargs)
        elif layers == 2:
            fit_obj = fit_two_layer(X_scaled, y_scaled, **fit_kwargs)
        else:
            fit_obj = fit_three_layer(X_scaled, y_scaled, **fit_kwargs)

        # Burn and thin automatically
        self.fit_obj = fit_obj.trim(burn=burn, thin=thin)

        # Cached sampler built lazily if needed
        self._cached_sampler = None

        # mvBayes-compatible samples object
        self.samples = _MvBayesVecDGPSamples()

        # Treat predictive draws as already containing full uncertainty
        self.samples.residSD = np.zeros(self.fit_obj.nmcmc)

        # Attach posterior draws for traceplots
        fit = self.fit_obj

        if hasattr(fit, "g"):
            g = np.asarray(fit.g)
            if g.ndim == 0:
                self.samples.g = np.repeat(float(g), fit.nmcmc)
            else:
                self.samples.g = g.copy()

        if self.layers == 1 and hasattr(fit, "tau2"):
            self.samples.tau2 = np.asarray(fit.tau2).copy()

        if self.layers in (2, 3) and hasattr(fit, "tau2_y"):
            self.samples.tau2_y = np.asarray(fit.tau2_y).copy()

    def _scale_X(self, Xtest):
        Xtest = np.asarray(Xtest, dtype=float)
        if Xtest.ndim == 1:
            Xtest = Xtest.reshape(-1, 1)
        if Xtest.ndim != 2:
            raise ValueError("Xtest must be 2D or coercible to 2D.")
        if Xtest.shape[1] != self.X_min_.shape[0]:
            raise ValueError("Xtest has wrong number of columns.")
        return (Xtest - self.X_min_) / self.X_range_

    def _unscale_draws(self, draws):
        return draws * self.y_sd_ + self.y_mean_

    def _get_cached_sampler(self):
        if self._cached_sampler is None:
            self._cached_sampler = self.fit_obj.sampler(**self.sampler_kwargs)
        return self._cached_sampler

    def _predict_post_sample(self, Xtest_scaled, draw_idx):
        """
        Use fit.post_sample(...) to generate predictive draws.

        Returns
        -------
        np.ndarray
            Shape (n_selected_draws * nper, n_obs)
        """
        return self.fit_obj.post_sample(
            Xtest_scaled,
            nper=self.nper,
            draws=draw_idx,
        )

    def _predict_cached_sampler(self, Xtest_scaled, draw_idx):
        """
        Use cached sampler from calibrate.py.

        Returns
        -------
        np.ndarray
            Shape (n_selected_draws * nper, n_obs)
        """
        sampler = self._get_cached_sampler()
        n_obs = Xtest_scaled.shape[0]

        out = np.empty((len(draw_idx) * self.nper, n_obs), dtype=float)

        row0 = 0
        rng = np.random.default_rng()
        for t in draw_idx:
            samp_t = sampler.sample(Xtest_scaled, draw=t, rng=rng, nper=self.nper)
            samp_t = np.asarray(samp_t)
            if samp_t.ndim != 2:
                raise ValueError(
                    f"Expected sampler.sample(...) to return 2D array, got shape {samp_t.shape}."
                )
            row1 = row0 + self.nper
            out[row0:row1, :] = samp_t
            row0 = row1

        return out

    def predict(self, Xtest, idxSamples=None):
        """
        Return predictive draws from the fitted emulator.

        Parameters
        ----------
        Xtest : array-like
            Test predictors.
        idxSamples : None or array-like of int
            Posterior draw indices to retain. If None, uses all retained MCMC draws.

        Returns
        -------
        np.ndarray
            Shape (n_samples_selected, n_obs), compatible with mvBayes.
            With nper=1, this is (n_selected_draws, n_obs).
        """
        Xtest_scaled = self._scale_X(Xtest)

        if idxSamples is None:
            draw_idx = np.arange(self.fit_obj.nmcmc, dtype=int)
        else:
            idxSamples = np.asarray(idxSamples, dtype=int).ravel()
            if idxSamples.size == 0:
                raise ValueError("idxSamples must not be empty.")
            draw_idx = idxSamples % self.fit_obj.nmcmc

        if self.sample_method == "post_sample":
            draws = self._predict_post_sample(Xtest_scaled, draw_idx)
        elif self.sample_method == "cached_sampler":
            draws = self._predict_cached_sampler(Xtest_scaled, draw_idx)
        else:  # auto
            if Xtest_scaled.shape[0] == 1:
                draws = self._predict_cached_sampler(Xtest_scaled, draw_idx)
            else:
                draws = self._predict_post_sample(Xtest_scaled, draw_idx)

        draws = np.asarray(draws)

        if draws.ndim != 2:
            raise ValueError(
                f"Expected predictive draws with 2 dimensions, got shape {draws.shape}."
            )

        return self._unscale_draws(draws)


def vecdgp4mvBayes(X, y, **kwargs):
    """
    Factory function for use as mvBayes(..., bayesModel=...).

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix.
    y : np.ndarray
        Univariate response.
    **kwargs
        Additional keyword arguments passed to MvBayesVecDGPWrapper.

    Returns
    -------
    MvBayesVecDGPWrapper
        mvBayes-compatible fitted model object.
    """
    return MvBayesVecDGPWrapper(X, y, **kwargs)
    