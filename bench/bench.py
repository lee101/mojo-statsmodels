from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojostatsmodels.api as msm  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA  # noqa: E402


def best_time(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def regression_data(n, k, seed=0):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(sm.add_constant(rng.normal(size=(n, k - 1))))
    beta = rng.normal(scale=0.3, size=k)
    return rng, x, beta


def simulate_arma(n, seed=0):
    rng = np.random.default_rng(seed)
    errors = rng.normal(size=n + 200)
    y = np.zeros(n + 200)
    for t in range(2, len(y)):
        y[t] = 0.1 + 0.62 * y[t - 1] - 0.18 * y[t - 2] + errors[t] + 0.25 * errors[t - 1]
    return y[200:]


CASES = []


def benchmark(name, repeat=3):
    def register(builder):
        CASES.append((name, repeat, builder))
        return builder
    return register


@benchmark("OLS.fit (200,000 x 12)")
def ols_fit():
    rng, x, beta = regression_data(200_000, 12)
    y = np.ascontiguousarray(x @ beta + rng.normal(scale=0.5, size=len(x)))
    ours = lambda: msm.OLS(y, x).fit()
    theirs = lambda: sm.OLS(y, x).fit()
    assert np.allclose(ours().params, theirs().params, atol=1e-9)
    return ours, theirs


@benchmark("OLS.predict (500,000 x 12)")
def ols_predict():
    rng, train_x, beta = regression_data(20_000, 12, 2)
    y = np.ascontiguousarray(train_x @ beta + rng.normal(scale=0.5, size=len(train_x)))
    _, test_x, _ = regression_data(500_000, 12, 3)
    our_result = msm.OLS(y, train_x).fit()
    their_result = sm.OLS(y, train_x).fit()
    assert np.allclose(our_result.predict(test_x), their_result.predict(test_x))
    return lambda: our_result.predict(test_x), lambda: their_result.predict(test_x)


@benchmark("Binomial GLM.fit (100,000 x 8)")
def binomial_fit():
    rng, x, beta = regression_data(100_000, 8, 4)
    probability = 1.0 / (1.0 + np.exp(-(x @ beta)))
    y = np.ascontiguousarray(rng.binomial(1, probability), dtype=np.float64)
    ours = lambda: msm.GLM(y, x, family=msm.families.Binomial()).fit()
    theirs = lambda: sm.GLM(y, x, family=sm.families.Binomial()).fit()
    assert np.allclose(ours().params, theirs().params, atol=1e-7)
    return ours, theirs


@benchmark("Poisson GLM.fit (100,000 x 8)")
def poisson_fit():
    rng, x, beta = regression_data(100_000, 8, 7)
    y = np.ascontiguousarray(rng.poisson(np.exp(x @ beta)), dtype=np.float64)
    ours = lambda: msm.GLM(y, x, family=msm.families.Poisson()).fit()
    theirs = lambda: sm.GLM(y, x, family=sm.families.Poisson()).fit()
    assert np.allclose(ours().params, theirs().params, atol=1e-7)
    return ours, theirs


@benchmark("ARIMA(2,0,1).fit (20,000)", repeat=2)
def arima_fit():
    y = simulate_arma(20_000, 9)
    ours = lambda: msm.ARIMA(y, order=(2, 0, 1)).fit()
    theirs = lambda: StatsmodelsARIMA(y, order=(2, 0, 1)).fit()
    assert np.allclose(ours().params, theirs().params, atol=0.01)
    return ours, theirs


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| benchmark | mojo-statsmodels | statsmodels | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, repeat, builder in CASES:
        ours, theirs = builder()
        ours()
        theirs()
        our_time = best_time(ours, repeat)
        their_time = best_time(theirs, repeat)
        ratio = their_time / our_time
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {our_time * 1000:.2f} ms | "
            f"{their_time * 1000:.2f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
