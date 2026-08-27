import numpy as np
import pytest
import statsmodels.api as sm

import mojostatsmodels.api as msm


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(184)
    x = rng.normal(size=(600, 7))
    x = sm.add_constant(x)
    beta = np.array([0.7, 1.5, -0.4, 0.0, 2.2, -1.1, 0.3, 0.8])
    y = x @ beta + rng.normal(scale=0.6, size=len(x))
    return x, y


def test_ols_full_result_parity(regression_data):
    x, y = regression_data
    ours = msm.OLS(y, x).fit()
    theirs = sm.OLS(y, x).fit()
    for name in ("params", "bse", "tvalues", "pvalues", "fittedvalues", "resid"):
        assert np.allclose(getattr(ours, name), getattr(theirs, name), rtol=1e-10)
    for name in (
        "ssr",
        "mse_resid",
        "scale",
        "rsquared",
        "rsquared_adj",
        "llf",
        "aic",
        "bic",
    ):
        assert getattr(ours, name) == pytest.approx(getattr(theirs, name), rel=1e-11)
    assert np.allclose(ours.conf_int(), theirs.conf_int(), rtol=1e-10)


def test_ols_without_constant(regression_data):
    x, y = regression_data
    x = x[:, 1:]
    ours = msm.OLS(y, x).fit()
    theirs = sm.OLS(y, x).fit()
    assert np.allclose(ours.params, theirs.params, atol=1e-11)
    assert ours.rsquared == pytest.approx(theirs.rsquared, rel=1e-12)
    assert ours.rsquared_adj == pytest.approx(theirs.rsquared_adj, rel=1e-12)


def test_wls_parity(regression_data):
    x, y = regression_data
    weights = np.linspace(0.2, 2.5, len(y))
    ours = msm.WLS(y, x, weights=weights).fit()
    theirs = sm.WLS(y, x, weights=weights).fit()
    assert np.allclose(ours.params, theirs.params, atol=1e-11)
    assert np.allclose(ours.bse, theirs.bse, rtol=1e-10)
    assert ours.llf == pytest.approx(theirs.llf, rel=1e-11)


@pytest.mark.parametrize("k", [5, 13])
def test_ols_simd_tail_parity(k):
    rng = np.random.default_rng(918 + k)
    x = np.ascontiguousarray(rng.normal(size=(257, k)))
    x[:, 0] = 1.0
    beta = rng.normal(size=k)
    y = np.ascontiguousarray(x @ beta + rng.normal(scale=0.2, size=len(x)))
    ours = msm.OLS(y, x).fit()
    theirs = sm.OLS(y, x).fit()
    assert np.allclose(ours.params, theirs.params, atol=1e-10)
    test_x = np.ascontiguousarray(rng.normal(size=(19, k)))
    test_x[:, 0] = 1.0
    assert np.allclose(ours.predict(test_x), theirs.predict(test_x), atol=1e-12)


def test_large_ols_prediction_parity(regression_data):
    x, y = regression_data
    ours = msm.OLS(y, x).fit()
    theirs = sm.OLS(y, x).fit()
    large_x = np.tile(x, (25, 1))
    assert np.allclose(ours.predict(large_x), theirs.predict(large_x))


@pytest.mark.parametrize("rows, k", [(12_500, 8), (5_882, 17), (5_883, 17)])
def test_ols_prediction_dispatch_threshold(rows, k):
    rng = np.random.default_rng(rows + k)
    x = np.ascontiguousarray(rng.normal(size=(200, k)))
    beta = rng.normal(size=k)
    y = np.ascontiguousarray(x @ beta + rng.normal(scale=0.2, size=len(x)))
    ours = msm.OLS(y, x).fit()
    theirs = sm.OLS(y, x).fit()
    test_x = np.ascontiguousarray(rng.normal(size=(rows, k)))
    assert np.allclose(ours.predict(test_x), theirs.predict(test_x), atol=1e-12)


def test_ols_prediction_propagates_nonfinite(regression_data):
    x, y = regression_data
    ours = msm.OLS(y, x).fit()
    theirs = sm.OLS(y, x).fit()
    test_x = x[:3].copy()
    test_x[1, 2] = np.nan
    assert np.allclose(
        ours.predict(test_x), theirs.predict(test_x), equal_nan=True
    )


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_ols_robust_covariance(regression_data, cov_type):
    x, y = regression_data
    ours = msm.OLS(y, x).fit(cov_type=cov_type)
    theirs = sm.OLS(y, x).fit(cov_type=cov_type)
    assert np.allclose(ours.cov_params(), theirs.cov_params(), rtol=1e-9)
    assert np.allclose(ours.bse, theirs.bse, rtol=1e-9)


def test_rank_deficient_ols_uses_least_norm_solution(regression_data):
    x, y = regression_data
    singular = np.column_stack((x, x[:, 2]))
    ours = msm.OLS(y, singular).fit()
    theirs = sm.OLS(y, singular).fit()
    assert np.allclose(ours.params, theirs.params, atol=1e-10)
    assert np.allclose(ours.fittedvalues, theirs.fittedvalues, atol=1e-10)


def test_add_constant_and_missing_drop(regression_data):
    x, y = regression_data
    assert np.array_equal(msm.add_constant(x), x)
    raw = x[:, 1:]
    assert np.array_equal(msm.add_constant(raw), x)
    damaged = y.copy()
    damaged[17] = np.nan
    ours = msm.OLS(damaged, x, missing="drop").fit()
    theirs = sm.OLS(damaged, x, missing="drop").fit()
    assert np.allclose(ours.params, theirs.params, atol=1e-11)


def test_ffi_prediction_dimension_and_layout_guards(regression_data):
    x, y = regression_data
    result = msm.OLS(y, x).fit()
    with pytest.raises(ValueError, match="incompatible"):
        result.predict(x[:, :-1])
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        result.predict(x[None, :, :])

    from mojostatsmodels._lib import addr

    with pytest.raises(TypeError, match="float64"):
        addr(np.ones(4, dtype=np.float32))
    with pytest.raises(TypeError, match="C-contiguous"):
        addr(np.ones((4, 4), dtype=np.float64)[:, ::2])


def test_empty_regression_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        msm.OLS(np.empty(0), np.empty((0, 1)))


def test_add_constant_modes():
    x = np.column_stack((np.arange(5.0), np.ones(5)))
    with pytest.raises(ValueError, match="already contains"):
        msm.add_constant(x, has_constant="raise")
    appended = msm.add_constant(np.arange(5.0), prepend=False)
    assert np.array_equal(appended[:, -1], np.ones(5))
