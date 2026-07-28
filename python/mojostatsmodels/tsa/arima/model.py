from __future__ import annotations

from math import comb
from types import SimpleNamespace
import warnings

import numpy as np
from scipy import optimize, stats

from ..._lib import addr, f64, lib


def _stationary_from_raw(raw: np.ndarray) -> np.ndarray:
    coefficients = np.empty(0)
    for reflection in np.tanh(raw):
        previous = coefficients
        coefficients = np.empty(previous.size + 1)
        for i in range(previous.size):
            coefficients[i] = previous[i] - reflection * previous[-i - 1]
        coefficients[-1] = reflection
    return coefficients


def _raw_from_stationary(coefficients: np.ndarray) -> np.ndarray:
    current = np.asarray(coefficients, dtype=float).copy()
    reflections = np.empty(current.size)
    for size in range(current.size, 0, -1):
        reflection = float(np.clip(current[size - 1], -0.999, 0.999))
        reflections[size - 1] = reflection
        if size > 1:
            previous = np.empty(size - 1)
            denominator = max(1.0 - reflection * reflection, 1e-8)
            for i in range(size - 1):
                previous[i] = (
                    current[i] + reflection * current[size - i - 2]
                ) / denominator
            current = previous
    return np.arctanh(np.clip(reflections, -0.999999, 0.999999))


class ARIMA:
    def __init__(
        self,
        endog,
        exog=None,
        order=(0, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        concentrate_scale: bool = False,
        trend_offset: int = 1,
        dates=None,
        freq=None,
        missing: str = "none",
        validate_specification: bool = True,
    ):
        if exog is not None:
            raise NotImplementedError("ARIMA with exogenous regressors is not covered")
        if tuple(seasonal_order) != (0, 0, 0, 0):
            raise NotImplementedError("seasonal ARIMA is not covered")
        if len(order) != 3 or any(int(value) != value or value < 0 for value in order):
            raise ValueError("order must contain three nonnegative integers")
        p, d, q = map(int, order)
        if p > 8 or q > 8:
            raise ValueError("covered AR and MA orders are at most 8")
        y = np.asarray(endog, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(y)):
            raise ValueError("ARIMA requires finite observations")
        if y.size - d <= max(p, q) + 2:
            raise ValueError("too few observations for the requested order")
        if missing not in {"none", "raise"}:
            raise NotImplementedError("ARIMA missing-data handling is not covered")
        if trend is None:
            trend = "c" if d == 0 else "n"
        if trend not in {"n", "c"}:
            raise NotImplementedError("covered ARIMA trends are 'n' and 'c'")
        if d > 0 and trend == "c":
            raise ValueError("a constant trend is not covered in an integrated model")
        self.endog = f64(y, ndim=1)
        self.order = (p, d, q)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.enforce_stationarity = bool(enforce_stationarity)
        self.enforce_invertibility = bool(enforce_invertibility)
        self.concentrate_scale = bool(concentrate_scale)
        self.nobs = int(y.size)
        self.data = SimpleNamespace(endog=self.endog)

    def _differenced(self):
        result = self.endog.copy()
        for _ in range(self.order[1]):
            result = np.diff(result)
        return f64(result, ndim=1)

    def _initial_parameters(self, y: np.ndarray):
        p, _, q = self.order
        has_constant = int(self.trend == "c")
        burn = max(p, q)
        preliminary_order = min(max(burn + 10, 10), max(1, len(y) // 4))
        residual = np.zeros_like(y)
        if q and len(y) > preliminary_order + 2:
            rows = np.arange(preliminary_order, len(y))
            design = np.column_stack(
                [y[rows - lag] for lag in range(1, preliminary_order + 1)]
            )
            if has_constant:
                design = np.column_stack((np.ones(rows.size), design))
            preliminary = np.linalg.lstsq(design, y[rows], rcond=None)[0]
            residual[rows] = y[rows] - design @ preliminary
        rows = np.arange(max(burn, preliminary_order if q else burn), len(y))
        columns = []
        if has_constant:
            columns.append(np.ones(rows.size))
        columns.extend(y[rows - lag] for lag in range(1, p + 1))
        columns.extend(residual[rows - lag] for lag in range(1, q + 1))
        if not columns:
            return np.empty(0)
        design = np.column_stack(columns)
        return np.linalg.lstsq(design, y[rows], rcond=None)[0]

    def _physical_from_raw(self, raw: np.ndarray):
        p, _, q = self.order
        has_constant = int(self.trend == "c")
        physical = np.empty_like(raw)
        index = 0
        if has_constant:
            physical[0] = raw[0]
            index = 1
        if p:
            physical[index:index + p] = (
                _stationary_from_raw(raw[index:index + p])
                if self.enforce_stationarity
                else raw[index:index + p]
            )
        index += p
        if q:
            physical[index:index + q] = (
                -_stationary_from_raw(raw[index:index + q])
                if self.enforce_invertibility
                else raw[index:index + q]
            )
        return physical

    def _raw_from_physical(self, physical: np.ndarray):
        p, _, q = self.order
        has_constant = int(self.trend == "c")
        raw = np.empty_like(physical)
        index = 0
        if has_constant:
            raw[0] = physical[0]
            index = 1
        if p:
            raw[index:index + p] = (
                _raw_from_stationary(physical[index:index + p])
                if self.enforce_stationarity
                else physical[index:index + p]
            )
        index += p
        if q:
            raw[index:index + q] = (
                _raw_from_stationary(-physical[index:index + q])
                if self.enforce_invertibility
                else physical[index:index + q]
            )
        return raw

    def fit(
        self,
        start_params=None,
        transformed: bool = True,
        includes_fixed: bool = False,
        method=None,
        method_kwargs=None,
        gls=None,
        gls_kwargs=None,
        cov_type=None,
        cov_kwds=None,
        return_params: bool = False,
        low_memory: bool = False,
    ):
        if method not in {None, "css"}:
            raise NotImplementedError("the covered ARIMA estimator is conditional MLE")
        y = self._differenced()
        p, _, q = self.order
        has_constant = int(self.trend == "c")
        count = p + q + has_constant
        burn = max(p, q)
        residual = np.zeros(len(y))
        derivative = np.zeros((len(y), max(count, 1)))
        gradient = np.zeros(max(count, 1))

        def evaluate_physical(parameters):
            kernel_parameters = f64(parameters, ndim=1)
            if kernel_parameters.size == 0:
                kernel_parameters = np.zeros(1, dtype=np.float64)
            if has_constant:
                kernel_parameters = parameters.copy()
                kernel_parameters[0] *= 1.0 - np.sum(parameters[1:1 + p])
                kernel_parameters = f64(kernel_parameters, ndim=1)
            value = lib().mst_arma_objective(
                addr(y),
                addr(kernel_parameters),
                addr(residual),
                addr(derivative),
                addr(gradient),
                len(y),
                p,
                q,
                has_constant,
            )
            return float(value) if np.isfinite(value) else np.inf

        if count:
            initial = (
                self._initial_parameters(y)
                if start_params is None
                else np.asarray(start_params, dtype=np.float64)[:count]
            )
            if start_params is not None and np.asarray(start_params).size not in {
                count,
                count + 1,
            }:
                raise ValueError("start_params has the wrong length")
            if start_params is None and has_constant and p:
                initial[0] /= max(1.0 - np.sum(initial[1:1 + p]), 1e-6)
            if initial.size != count:
                raise ValueError("start_params has the wrong length")
            raw_initial = self._raw_from_physical(initial) if transformed else initial

            def objective(raw):
                physical = f64(self._physical_from_raw(raw), ndim=1)
                return evaluate_physical(physical)

            options = {"maxiter": 300, "ftol": 1e-11, "gtol": 1e-6}
            if method_kwargs:
                options.update(method_kwargs)
            optimized = optimize.minimize(
                objective, raw_initial, method="L-BFGS-B", options=options
            )
            if not optimized.success:
                optimized = optimize.minimize(
                    objective,
                    optimized.x,
                    method="Powell",
                    options={"maxiter": 300, "xtol": 1e-7, "ftol": 1e-10},
                )
            physical = f64(self._physical_from_raw(optimized.x), ndim=1)
            objective_value = evaluate_physical(physical)
            if not optimized.success:
                warnings.warn(
                    f"conditional likelihood optimization did not converge: {optimized.message}",
                    RuntimeWarning,
                )
        else:
            physical = np.empty(0, dtype=np.float64)
            objective_value = evaluate_physical(physical)
            optimized = SimpleNamespace(success=True, nit=0, message="no parameters")
        effective_n = len(y) - burn
        sigma2 = float(np.sum(residual[burn:] ** 2) / effective_n)
        all_params = np.r_[physical, sigma2]
        if return_params:
            return all_params
        return ARIMAResults(
            self,
            all_params,
            y,
            residual.copy(),
            objective_value=float(objective_value),
            converged=bool(optimized.success),
            iterations=int(optimized.nit),
        )


class ARIMAResults:
    def __init__(
        self,
        model: ARIMA,
        params: np.ndarray,
        differenced: np.ndarray,
        residual: np.ndarray,
        *,
        objective_value: float,
        converged: bool,
        iterations: int,
    ):
        self.model = model
        self.params = params
        self.nobs = model.nobs
        self._differenced = differenced
        self._innovations = residual
        self.sigma2 = float(params[-1])
        p, d, q = model.order
        index = int(model.trend == "c")
        self.arparams = params[index:index + p].copy()
        self.maparams = params[index + p:index + p + q].copy()
        self.param_names = (
            (["const"] if model.trend == "c" else [])
            + [f"ar.L{i}" for i in range(1, p + 1)]
            + [f"ma.L{i}" for i in range(1, q + 1)]
            + ["sigma2"]
        )
        burn = max(p, q)
        effective_n = len(differenced) - burn
        self.nobs_effective = effective_n
        self.llf = -0.5 * effective_n * (
            np.log(2.0 * np.pi) + 1.0 + np.log(self.sigma2)
        )
        self.aic = -2.0 * self.llf + 2.0 * len(params)
        self.bic = -2.0 * self.llf + np.log(effective_n) * len(params)
        self.hqic = -2.0 * self.llf + 2.0 * np.log(np.log(effective_n)) * len(params)
        self.mle_retvals = {
            "converged": converged,
            "iterations": iterations,
            "objective": objective_value,
        }
        self.fittedvalues = self._fitted_levels()
        self.resid = model.endog - self.fittedvalues

    def _fitted_levels(self):
        p, d, q = self.model.order
        burn = max(p, q)
        predicted_difference = self._differenced - self._innovations
        if d == 0:
            fitted = predicted_difference.copy()
            fitted[:burn] = np.nan
            return fitted
        fitted = np.full(self.model.nobs, np.nan)
        for z_index in range(burn, len(self._differenced)):
            time_index = z_index + d
            value = predicted_difference[z_index]
            for lag in range(1, d + 1):
                value -= (-1) ** lag * comb(d, lag) * self.model.endog[
                    time_index - lag
                ]
            fitted[time_index] = value
        return fitted

    def _forecast_differences(self, steps: int):
        p, _, q = self.model.order
        constant = (
            self.params[0] * (1.0 - np.sum(self.arparams))
            if self.model.trend == "c"
            else 0.0
        )
        history = list(self._differenced)
        innovations = list(self._innovations)
        predictions = np.empty(steps)
        for horizon in range(steps):
            value = constant
            for lag, coefficient in enumerate(self.arparams, 1):
                value += coefficient * history[-lag]
            for lag, coefficient in enumerate(self.maparams, 1):
                value += coefficient * innovations[-lag]
            predictions[horizon] = value
            history.append(value)
            innovations.append(0.0)
        return predictions

    def forecast(self, steps: int = 1, signal_only: bool = False, **kwargs):
        if int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer")
        predicted = self._forecast_differences(int(steps))
        d = self.model.order[1]
        if d == 0:
            return predicted
        last_by_level = []
        current = self.model.endog
        for _ in range(d):
            last_by_level.append(float(current[-1]))
            current = np.diff(current)
        levels = np.empty_like(predicted)
        for i, high_difference in enumerate(predicted):
            value = high_difference
            for level in range(d - 1, -1, -1):
                value = last_by_level[level] + value
                last_by_level[level] = value
            levels[i] = value
        return levels

    def predict(self, start=None, end=None, dynamic=False, **kwargs):
        if start is None:
            start = 0
        if end is None:
            end = self.nobs - 1
        if start < self.nobs and end < self.nobs:
            return self.fittedvalues[start:end + 1]
        if start < self.nobs:
            return np.r_[
                self.fittedvalues[start:],
                self.forecast(end - self.nobs + 1),
            ]
        forecasts = self.forecast(end - self.nobs + 1)
        return forecasts[start - self.nobs:]

    def _forecast_standard_error(self, steps: int):
        p, d, q = self.model.order
        psi = np.zeros(steps)
        psi[0] = 1.0
        for horizon in range(1, steps):
            if horizon <= q:
                psi[horizon] += self.maparams[horizon - 1]
            for lag in range(1, min(p, horizon) + 1):
                psi[horizon] += self.arparams[lag - 1] * psi[horizon - lag]
        for _ in range(d):
            psi = np.cumsum(psi)
        return np.sqrt(self.sigma2 * np.cumsum(psi**2))

    def get_forecast(self, steps: int = 1, signal_only: bool = False, **kwargs):
        return PredictionResults(
            self.forecast(steps, signal_only=signal_only),
            self._forecast_standard_error(int(steps)),
        )


class PredictionResults:
    def __init__(self, predicted_mean, se_mean):
        self.predicted_mean = predicted_mean
        self.se_mean = se_mean

    def conf_int(self, alpha: float = 0.05):
        critical = stats.norm.ppf(1.0 - alpha / 2.0)
        return np.column_stack(
            (
                self.predicted_mean - critical * self.se_mean,
                self.predicted_mean + critical * self.se_mean,
            )
        )
