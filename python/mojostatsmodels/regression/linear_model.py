from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
from scipy import stats

from .._lib import addr, f64, lib

_BLAS_PREDICT_MIN_ELEMENTS = 100_000


def _constant_columns(exog: np.ndarray) -> np.ndarray:
    return (np.ptp(exog, axis=0) == 0) & np.all(exog != 0, axis=0)


class RegressionModel:
    def __init__(
        self,
        endog,
        exog=None,
        *,
        weights=None,
        missing: str = "none",
        hasconst=None,
        **kwargs,
    ):
        if exog is None:
            raise ValueError("exog is required")
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
        w = np.ones(y.size) if weights is None else np.asarray(
            weights, dtype=np.float64
        ).reshape(-1)
        if w.size != y.size or np.any(w < 0):
            raise ValueError("weights must be nonnegative and match endog")
        finite = np.isfinite(y) & np.isfinite(w) & np.all(np.isfinite(x), axis=1)
        if not np.all(finite):
            if missing == "drop":
                y, x, w = y[finite], x[finite], w[finite]
            else:
                raise ValueError("NaN or infinity in model data")
        self.endog = f64(y, ndim=1)
        self.exog = f64(x, ndim=2)
        self.weights = f64(w, ndim=1)
        self.nobs = float(y.size)
        self.rank = None
        if hasconst is True:
            self.k_constant = 1
        elif hasconst is False:
            self.k_constant = 0
        else:
            self.k_constant = int(np.any(_constant_columns(x)))
        self.df_model = np.nan
        self.df_resid = np.nan
        self.data = SimpleNamespace(endog=self.endog, exog=self.exog)

    def fit(
        self,
        method: str = "pinv",
        cov_type: str = "nonrobust",
        cov_kwds=None,
        use_t=None,
        **kwargs,
    ):
        n, k = self.exog.shape
        params = np.empty(k)
        normalized_covariance = np.empty((k, k))
        factor = np.empty((k, k))
        rhs = np.empty(k)
        ok = lib().mst_wls_fit(
            addr(self.exog),
            addr(self.endog),
            addr(self.weights),
            addr(params),
            addr(normalized_covariance),
            addr(factor),
            addr(rhs),
            n,
            k,
        )
        if ok:
            singular_values = np.linalg.svd(np.tril(factor), compute_uv=False)
            tolerance = singular_values[0] * max(n, k) * np.finfo(float).eps
            self.rank = int(np.count_nonzero(singular_values > tolerance))
        if not ok or self.rank < k:
            root_weights = np.sqrt(self.weights)
            weighted_x = self.exog * root_weights[:, None]
            weighted_y = self.endog * root_weights
            self.rank = int(np.linalg.matrix_rank(weighted_x))
            params = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]
            normalized_covariance = np.linalg.pinv(weighted_x.T @ weighted_x)
        if not np.all(np.isfinite(params)) or not np.all(
            np.isfinite(normalized_covariance)
        ):
            raise np.linalg.LinAlgError("weighted least-squares solve was not finite")
        self.df_model = float(self.rank - self.k_constant)
        self.df_resid = float(n - self.rank)
        result = RegressionResults(self, params, normalized_covariance)
        if cov_type.lower() != "nonrobust":
            result = result.get_robustcov_results(cov_type=cov_type, use_t=use_t)
        elif use_t is not None:
            result.use_t = bool(use_t)
        return result

    def predict(self, params, exog=None):
        x = self.exog if exog is None else f64(exog)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError("exog must be one- or two-dimensional")
        parameters = f64(params, ndim=1)
        if x.shape[1] != parameters.size:
            raise ValueError("exog and params have incompatible dimensions")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(parameters)):
            raise ValueError("exog and params must be finite")
        prediction = np.empty(x.shape[0])
        if x.size >= _BLAS_PREDICT_MIN_ELEMENTS:
            np.matmul(x, parameters, out=prediction)
        else:
            lib().mst_linear_predict(
                addr(x),
                addr(parameters),
                addr(prediction),
                x.shape[0],
                x.shape[1],
            )
        return prediction


class OLS(RegressionModel):
    def __init__(
        self, endog, exog=None, missing: str = "none", hasconst=None, **kwargs
    ):
        super().__init__(
            endog, exog, weights=None, missing=missing, hasconst=hasconst, **kwargs
        )


class WLS(RegressionModel):
    def __init__(
        self,
        endog,
        exog,
        weights=1.0,
        missing: str = "none",
        hasconst=None,
        **kwargs,
    ):
        if np.ndim(weights) == 0:
            weights = np.full(np.asarray(endog).size, float(weights))
        super().__init__(
            endog,
            exog,
            weights=weights,
            missing=missing,
            hasconst=hasconst,
            **kwargs,
        )


class RegressionResults:
    def __init__(
        self,
        model: RegressionModel,
        params: np.ndarray,
        normalized_cov_params: np.ndarray,
    ):
        self.model = model
        self.params = params
        self.normalized_cov_params = normalized_cov_params
        self.nobs = model.nobs
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.fittedvalues = model.predict(params)
        self.resid = model.endog - self.fittedvalues
        self.wresid = np.sqrt(model.weights) * self.resid
        self.ssr = float(self.wresid @ self.wresid)
        self.centered_tss = float(
            np.sum(model.weights * (model.endog - np.average(
                model.endog, weights=model.weights
            )) ** 2)
        )
        self.uncentered_tss = float(np.sum(model.weights * model.endog**2))
        self.mse_resid = self.ssr / self.df_resid if self.df_resid > 0 else np.nan
        self.scale = self.mse_resid
        self.cov_type = "nonrobust"
        self.cov_kwds = {}
        self.use_t = True
        self._cov_params = normalized_cov_params * self.scale

    @property
    def rsquared(self):
        denominator = (
            self.centered_tss if self.model.k_constant else self.uncentered_tss
        )
        return 1.0 - self.ssr / denominator

    @property
    def rsquared_adj(self):
        if self.model.k_constant:
            return 1.0 - (self.nobs - 1.0) / self.df_resid * (
                1.0 - self.rsquared
            )
        return 1.0 - self.nobs / self.df_resid * (1.0 - self.rsquared)

    @property
    def llf(self):
        return -0.5 * self.nobs * (
            np.log(2.0 * np.pi) + 1.0 + np.log(self.ssr / self.nobs)
        ) + 0.5 * float(np.sum(np.log(self.model.weights)))

    @property
    def aic(self):
        return -2.0 * self.llf + 2.0 * (self.df_model + self.model.k_constant)

    @property
    def bic(self):
        return -2.0 * self.llf + np.log(self.nobs) * (
            self.df_model + self.model.k_constant
        )

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
        return self.model.predict(self.params, exog)

    def get_robustcov_results(self, cov_type: str = "HC1", use_t=None, **kwargs):
        kind = cov_type.upper()
        if kind not in {"HC0", "HC1", "HC2", "HC3"}:
            raise NotImplementedError("covered robust covariance types: HC0-HC3")
        result = copy.copy(self)
        x = self.model.exog
        weighted_x = x * np.sqrt(self.model.weights)[:, None]
        weighted_resid = self.wresid
        squared = weighted_resid**2
        if kind in {"HC2", "HC3"}:
            leverage = np.sum(
                (weighted_x @ self.normalized_cov_params) * weighted_x, axis=1
            )
            squared = squared / np.maximum(1.0 - leverage, 1e-12) ** (
                1 if kind == "HC2" else 2
            )
        elif kind == "HC1":
            squared *= self.nobs / self.df_resid
        meat = (weighted_x * squared[:, None]).T @ weighted_x
        result._cov_params = (
            self.normalized_cov_params @ meat @ self.normalized_cov_params
        )
        result.cov_type = kind
        result.use_t = False if use_t is None else bool(use_t)
        return result


RegressionResultsWrapper = RegressionResults
