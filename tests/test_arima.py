import numpy as np
import pytest
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA

from mojostatsmodels._lib import addr, f64, lib
from mojostatsmodels.tsa.arima.model import ARIMA


def simulate_arma(n, ar=(), ma=(), constant=0.0, seed=0):
    rng = np.random.default_rng(seed)
    innovation = rng.normal(size=n + 300)
    series = np.zeros(n + 300)
    for t in range(max(len(ar), len(ma)), len(series)):
        series[t] = constant
        series[t] += sum(value * series[t - lag] for lag, value in enumerate(ar, 1))
        series[t] += innovation[t]
        series[t] += sum(
            value * innovation[t - lag] for lag, value in enumerate(ma, 1)
        )
    return series[300:]


def test_arima_arma11_parameter_and_forecast_parity():
    y = simulate_arma(2400, ar=(0.65,), ma=(0.3,), constant=0.15, seed=21)
    ours = ARIMA(y, order=(1, 0, 1)).fit()
    theirs = StatsmodelsARIMA(y, order=(1, 0, 1)).fit()
    assert ours.mle_retvals["converged"]
    assert np.allclose(ours.params, theirs.params, atol=0.015)
    assert np.allclose(ours.forecast(8), theirs.forecast(8), atol=0.025)
    assert abs(ours.llf - theirs.llf) < 4.0


def test_arima_ar2_parity():
    y = simulate_arma(2200, ar=(0.5, -0.2), constant=-0.1, seed=81)
    ours = ARIMA(y, order=(2, 0, 0)).fit()
    theirs = StatsmodelsARIMA(y, order=(2, 0, 0)).fit()
    assert np.allclose(ours.params, theirs.params, atol=0.012)
    assert np.allclose(ours.forecast(5), theirs.forecast(5), atol=0.02)


def test_integrated_arima_parity():
    differences = simulate_arma(1800, ar=(0.55,), seed=4)
    y = np.cumsum(differences)
    ours = ARIMA(y, order=(1, 1, 0)).fit()
    theirs = StatsmodelsARIMA(y, order=(1, 1, 0)).fit()
    assert np.allclose(ours.params, theirs.params, atol=0.012)
    assert np.allclose(ours.forecast(10), theirs.forecast(10), atol=0.04)
    forecast = ours.get_forecast(4)
    assert forecast.conf_int().shape == (4, 2)
    assert np.all(forecast.se_mean > 0)


def test_random_walk_fit_and_forecast():
    rng = np.random.default_rng(7)
    y = np.cumsum(rng.normal(size=500))
    result = ARIMA(y, order=(0, 1, 0)).fit()
    assert result.params.shape == (1,)
    assert np.allclose(result.forecast(3), y[-1])
    assert result.param_names == ["sigma2"]


def test_mojo_arma_analytic_gradient():
    rng = np.random.default_rng(11)
    y = f64(rng.normal(size=300))
    parameters = f64(np.array([0.3, -0.15, 0.2]))
    residual = np.zeros(len(y))
    derivative = np.zeros((len(y), len(parameters)))
    gradient = np.zeros(len(parameters))

    def objective(values):
        values = f64(values)
        return lib().mst_arma_objective(
            addr(y),
            addr(values),
            addr(residual),
            addr(derivative),
            addr(gradient),
            len(y),
            2,
            1,
            0,
        )

    objective(parameters)
    analytic = gradient.copy()
    numerical = np.empty_like(parameters)
    step = 1e-6
    for i in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[i] += step
        minus[i] -= step
        numerical[i] = (objective(plus) - objective(minus)) / (2 * step)
    assert np.allclose(analytic, numerical, rtol=2e-5, atol=2e-5)


def test_arima_rejects_unsupported_structure():
    y = np.arange(100.0)
    with pytest.raises(NotImplementedError):
        ARIMA(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    with pytest.raises(NotImplementedError):
        ARIMA(y, exog=np.ones((100, 1)), order=(1, 0, 0))
