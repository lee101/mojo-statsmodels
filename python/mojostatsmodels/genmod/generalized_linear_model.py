from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import stats

from .._lib import addr, f64, lib
from . import families


class GLM:
    def __init__(
        self,
        endog,
        exog,
        family=None,
        offset=None,
        exposure=None,
        freq_weights=None,
        var_weights=None,
        missing: str = "none",
        **kwargs,
    ):
        if missing not in {"none", "drop", "raise"}:
            raise ValueError("missing must be 'none', 'drop', or 'raise'")
        y = np.asarray(endog, dtype=np.float64).reshape(-1)
        x = np.asarray(exog, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != y.size:
            raise ValueError("endog and exog matrices are different sizes")
        if y.size == 0 or x.shape[1] == 0:
            raise ValueError("endog and exog must be nonempty")
        self.family = families.Gaussian() if family is None else family
        if not isinstance(
            self.family, (families.Gaussian, families.Binomial, families.Poisson)
        ):
            raise NotImplementedError(
                "covered GLM families are Gaussian, Binomial, and Poisson"
            )
        n = y.size
        off = np.zeros(n) if offset is None else np.broadcast_to(
            np.asarray(offset, dtype=np.float64), (n,)
        ).copy()
        if exposure is not None:
            if not isinstance(self.family.link, families.links.Log):
                raise ValueError("exposure is only valid with the log link")
            exposed = np.broadcast_to(np.asarray(exposure, dtype=np.float64), (n,))
            if np.any(exposed <= 0):
                raise ValueError("exposure must be strictly positive")
            off += np.log(exposed)
        frequency = np.ones(n) if freq_weights is None else np.broadcast_to(
            np.asarray(freq_weights, dtype=np.float64), (n,)
        ).copy()
        variance = np.ones(n) if var_weights is None else np.broadcast_to(
            np.asarray(var_weights, dtype=np.float64), (n,)
        ).copy()
        finite = (
            np.isfinite(y)
            & np.isfinite(off)
            & np.isfinite(frequency)
            & np.isfinite(variance)
            & np.all(np.isfinite(x), axis=1)
        )
        if not np.all(finite):
            if missing == "drop":
                y, x, off, frequency, variance = (
                    value[finite] for value in (y, x, off, frequency, variance)
                )
            else:
                raise ValueError("NaN or infinity in model data")
        if np.any(frequency < 0) or np.any(variance < 0):
            raise ValueError("weights must be nonnegative")
        if isinstance(self.family, families.Binomial) and (
            np.any(y < 0) or np.any(y > 1)
        ):
            raise ValueError("Binomial endog must lie in [0, 1]")
        if isinstance(self.family, families.Poisson) and np.any(y < 0):
            raise ValueError("Poisson endog must be nonnegative")
        self.endog = f64(y, ndim=1)
        self.exog = f64(x, ndim=2)
        self.offset = f64(off, ndim=1)
        self.freq_weights = f64(frequency, ndim=1)
        self.var_weights = f64(variance, ndim=1)
        self.nobs = float(len(y))
        self.rank = int(np.linalg.matrix_rank(x))
        self.df_model = float(self.rank - int(np.any(
            (np.ptp(x, axis=0) == 0) & np.all(x != 0, axis=0)
        )))
        self.df_resid = float(np.sum(frequency) - self.rank)
        self.data = SimpleNamespace(endog=self.endog, exog=self.exog)

    def _starting_values(self):
        y = self.endog
        if isinstance(self.family, families.Gaussian):
            target = y - self.offset
        elif isinstance(self.family, families.Binomial):
            initial_mu = np.clip((y + 0.5) / 2.0, 1e-6, 1.0 - 1e-6)
            target = self.family.link(initial_mu) - self.offset
        else:
            target = np.log(np.maximum(y, 0.1)) - self.offset
        return np.linalg.lstsq(self.exog, target, rcond=None)[0]

    def fit(
        self,
        start_params=None,
        maxiter: int = 100,
        method: str = "IRLS",
        tol: float = 1e-8,
        scale=None,
        cov_type: str = "nonrobust",
        cov_kwds=None,
        use_t=None,
        full_output: bool = True,
        disp: bool = False,
        max_start_irls: int = 3,
        **kwargs,
    ):
        if method.upper() != "IRLS":
            raise NotImplementedError("the covered GLM fitting method is IRLS")
        if cov_type.lower() != "nonrobust":
            raise NotImplementedError("GLM currently covers nonrobust covariance")
        if isinstance(maxiter, (bool, np.bool_)) or int(maxiter) != maxiter:
            raise ValueError("maxiter must be a positive integer")
        maxiter = int(maxiter)
        if maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        n, k = self.exog.shape
        beta = f64(
            self._starting_values() if start_params is None else start_params,
            ndim=1,
            copy=True,
        )
        if beta.size != k:
            raise ValueError("start_params has the wrong length")
        covariance = np.empty((k, k))
        factor = np.empty((k, k))
        rhs = np.empty(k)
        next_beta = np.empty(k)
        iterations = lib().mst_glm_irls(
            addr(self.exog),
            addr(self.endog),
            addr(self.offset),
            addr(self.freq_weights),
            addr(self.var_weights),
            addr(beta),
            addr(covariance),
            addr(factor),
            addr(rhs),
            addr(next_beta),
            n,
            k,
            self.family.family_id,
            maxiter,
            float(tol),
        )
        if iterations < 0:
            raise np.linalg.LinAlgError("IRLS weighted design is singular")
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(covariance)):
            raise np.linalg.LinAlgError("IRLS produced non-finite results")
        return GLMResults(
            self,
            beta,
            covariance,
            iterations=iterations,
            converged=iterations <= maxiter,
            scale=scale,
            use_t=False if use_t is None else bool(use_t),
        )

    def predict(
        self,
        params,
        exog=None,
        exposure=None,
        offset=None,
        which: str = "mean",
        linear=None,
    ):
        x = self.exog if exog is None else f64(exog)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError("exog must be one- or two-dimensional")
        parameters = f64(params, ndim=1)
        if x.shape[1] != parameters.size:
            raise ValueError("exog and params have incompatible dimensions")
        if exog is None and offset is None:
            off = self.offset
        else:
            off = np.zeros(x.shape[0]) if offset is None else np.broadcast_to(
                np.asarray(offset, dtype=np.float64), (x.shape[0],)
            )
        if exposure is not None:
            exposed = np.broadcast_to(
                np.asarray(exposure, dtype=np.float64), (x.shape[0],)
            )
            if np.any(~np.isfinite(exposed)) or np.any(exposed <= 0):
                raise ValueError("exposure must be finite and strictly positive")
            off = off + np.log(exposed)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(parameters)):
            raise ValueError("exog and params must be finite")
        eta = np.asarray(x @ parameters + off)
        if linear is True or which == "linear":
            return eta
        return self.family.fitted(eta)


class GLMResults:
    def __init__(
        self,
        model: GLM,
        params: np.ndarray,
        normalized_cov_params: np.ndarray,
        *,
        iterations: int,
        converged: bool,
        scale,
        use_t: bool,
    ):
        self.model = model
        self.params = params
        self.normalized_cov_params = normalized_cov_params
        self.nobs = model.nobs
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.fit_history = {"iteration": iterations}
        self.converged = converged
        self.fittedvalues = model.predict(params)
        self.mu = self.fittedvalues
        self.resid_response = model.endog - self.mu
        self.resid_deviance = model.family.resid_dev(model.endog, self.mu)
        self.pearson_chi2 = float(np.sum(
            model.freq_weights
            * model.var_weights
            * self.resid_response**2
            / self._variance(self.mu)
        ))
        if scale is None:
            self.scale = (
                self.pearson_chi2 / self.df_resid
                if isinstance(model.family, families.Gaussian)
                else 1.0
            )
        else:
            self.scale = float(scale)
        self._cov_params = normalized_cov_params * self.scale
        self.use_t = use_t
        self.deviance = model.family.deviance(
            model.endog, self.mu, model.var_weights, model.freq_weights, self.scale
        )
        constant_mu = np.average(model.endog, weights=model.freq_weights)
        self.null_deviance = model.family.deviance(
            model.endog,
            np.full_like(model.endog, constant_mu),
            model.var_weights,
            model.freq_weights,
            self.scale,
        )
        likelihood_scale = (
            self.pearson_chi2 / np.sum(model.freq_weights)
            if isinstance(model.family, families.Gaussian)
            else self.scale
        )
        self.llf = model.family.loglike(
            model.endog,
            self.mu,
            model.var_weights,
            model.freq_weights,
            likelihood_scale,
        )
        self.aic = -2.0 * self.llf + 2.0 * len(params)
        self.bic_llf = -2.0 * self.llf + np.log(self.nobs) * len(params)

    def _variance(self, mu):
        if isinstance(self.model.family, families.Binomial):
            return np.clip(mu * (1.0 - mu), 1e-12, None)
        if isinstance(self.model.family, families.Poisson):
            return np.clip(mu, 1e-12, None)
        return np.ones_like(mu)

    @property
    def bse(self):
        return np.sqrt(np.diag(self._cov_params))

    @property
    def tvalues(self):
        return self.params / self.bse

    @property
    def pvalues(self):
        if self.use_t:
            return 2.0 * stats.t.sf(np.abs(self.tvalues), self.df_resid)
        return 2.0 * stats.norm.sf(np.abs(self.tvalues))

    def cov_params(self):
        return self._cov_params.copy()

    def conf_int(self, alpha: float = 0.05, cols=None):
        critical = (
            stats.t.ppf(1.0 - alpha / 2.0, self.df_resid)
            if self.use_t
            else stats.norm.ppf(1.0 - alpha / 2.0)
        )
        interval = np.column_stack(
            (self.params - critical * self.bse, self.params + critical * self.bse)
        )
        return interval if cols is None else interval[np.asarray(cols)]

    def predict(self, exog=None, transform: bool = True, *args, **kwargs):
        return self.model.predict(self.params, exog=exog, **kwargs)


GLMResultsWrapper = GLMResults
