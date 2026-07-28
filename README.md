# mojo-statsmodels

`mojo-statsmodels` is a focused port of compute-heavy
[statsmodels](https://www.statsmodels.org/) fitting paths to Mojo. It exposes
NumPy-facing Python classes with statsmodels names and call patterns while the
normal equations, IRLS loop, small-prediction path, and conditional ARMA
likelihood run in one compiled Mojo shared library. Large prediction uses the
same zero-copy NumPy buffers with optimized BLAS.

The package is useful when repeated dense model fits dominate a workload. It
is not a replacement for all of statsmodels; diagnostics and model classes
outside the explicitly covered subset remain upstream's domain.

## Covered subset

| upstream API | coverage |
| --- | --- |
| `OLS`, `WLS` | dense float64 fitting, rank-deficient least-norm fallback, prediction, residual and fitted values, likelihood/AIC/BIC, R-squared, standard errors, tests, confidence intervals, and HC0-HC3 covariance |
| `GLM` | Gaussian-identity, Binomial-logit, and Poisson-log IRLS; offsets, Poisson exposure, frequency and variance weights; deviance, likelihood, covariance, tests, and prediction |
| `tsa.arima.model.ARIMA` | univariate nonseasonal `ARIMA(p,d,q)` with `p,q <= 8`, stationary/invertible transforms, conditional Gaussian maximum likelihood, level forecasts, approximate forecast intervals, and constant or no trend where valid |
| `add_constant` | NumPy arrays, including upstream-style existing-constant handling |

The main deliberate ARIMA difference is important: statsmodels defaults to an
exact state-space likelihood, while this package optimizes the conditional
Gaussian likelihood after a Hannan-Rissanen initialization. Conditional MLE
is a standard large-sample estimator and is substantially cheaper, but its
likelihood and short-sample estimates need not exactly equal the state-space
result. The parity suite compares parameters and forecasts on simulated AR,
ARMA, and integrated series.

Not covered are formulas or pandas-labelled results, GLM families and links
other than the three canonical pairs above, clustered/HAC GLM covariance,
seasonal ARIMA, ARIMA exogenous regressors, missing-value ARIMA, dynamic
in-sample ARIMA prediction, and the rest of statsmodels.

## Install

The repository pins its own Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` emits `dist/libmojo-statsmodels.so`. Imports also rebuild the
library when the Mojo source is newer, but an explicit build is recommended
for deployments.

## Usage

This complete example fits OLS, Poisson GLM, and ARIMA models:

```python
import numpy as np
import mojostatsmodels.api as sm

rng = np.random.default_rng(7)
x = sm.add_constant(rng.normal(size=(1_000, 3)))
y = x @ np.array([0.5, 1.0, -0.8, 0.2]) + rng.normal(scale=0.3, size=1_000)

ols = sm.OLS(y, x).fit(cov_type="HC3")
print(ols.params, ols.rsquared)

counts = rng.poisson(np.exp(x @ np.array([0.1, 0.2, -0.1, 0.05])))
poisson = sm.GLM(counts, x, family=sm.families.Poisson()).fit()
print(poisson.params, poisson.deviance)

series = np.cumsum(rng.normal(size=1_000))
arima = sm.ARIMA(series, order=(1, 1, 1)).fit()
print(arima.forecast(5))
```

Run it inside the environment with `pixi run python example.py`.

## Performance

Measured through `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, with statsmodels 0.14.6, NumPy 2.5.1, and SciPy 1.18.0. Times are
the best of repeated fits on the same arrays. The ARIMA comparison includes
the estimator difference described above; it is conditional MLE here versus
statsmodels' exact state-space MLE.

| benchmark | mojo-statsmodels | statsmodels | relative |
| --- | ---: | ---: | ---: |
| OLS.fit (200,000 x 12) | 75.36 ms | 247.59 ms | 3.29x faster |
| OLS.predict (500,000 x 12) | 28.95 ms | 22.64 ms | 1.28x slower |
| Binomial GLM.fit (100,000 x 8) | 162.23 ms | 823.14 ms | 5.07x faster |
| Poisson GLM.fit (100,000 x 8) | 193.26 ms | 532.87 ms | 2.76x faster |
| ARIMA(2,0,1).fit (20,000) | 88.94 ms | 10,969.37 ms | 123.34x faster |

Large prediction dispatches the zero-copy NumPy buffers to optimized BLAS;
small predictions stay in the Mojo SIMD kernel to avoid BLAS setup overhead.
The fitting wins come from native-width SIMD system updates, keeping the
complete IRLS loop in compiled code, and avoiding a redundant full-design rank
decomposition on the full-rank OLS path.

Measured CPU task-launch and reduction overhead outweighed parallel speedups at
these sizes, so the kernels remain serial. No GPU path is included: prediction
and system construction are streamed kernels at or below roughly 2 flops per
byte moved, while ARIMA is recursive, so device transfers and launches are not
justified. Run `pixi run bench` to reproduce the table under the repository's
machine-wide benchmark lock.

## How it works

All kernels live in `src/statsmodels.mojo`, so the fixed Mojo shared-library
build cost is paid once. The exported boundary uses C ABI functions with
`Int` buffer addresses. Python owns every input, result, and scratch allocation
and calls the library through `ctypes`; Mojo reconstructs mutable raw pointers
with `AnyOrigin[mut=True]`.

Arrays are C-contiguous row-major `float64`. A compatible NumPy array crosses
the boundary without a copy; incompatible dtype or stride layouts are
converted once. No Mojo export retains a pointer or allocates caller-visible
memory.

OLS/WLS forms a symmetric weighted Gram matrix, solves it by Cholesky, and
returns the inverse information matrix for inference. GLM performs the whole
IRLS iteration and final Fisher-information inversion in Mojo. ARIMA
differences the series in Python, then calls a recursive Mojo ARMA residual
kernel for each optimizer evaluation; that kernel also computes the analytic
conditional-likelihood gradient used for validation and lower-level callers.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite asserts numerical and behavioral parity against the installed
upstream statsmodels on the same generated data.

## License

MIT
