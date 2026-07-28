import numpy as np
import pytest
import statsmodels.api as sm

import mojostatsmodels.api as msm


@pytest.fixture(scope="module")
def design():
    rng = np.random.default_rng(771)
    x = sm.add_constant(rng.normal(size=(1200, 5)))
    return rng, x


def assert_glm_parity(ours, theirs, atol=2e-8):
    assert ours.converged
    assert np.allclose(ours.params, theirs.params, atol=atol)
    assert np.allclose(ours.bse, theirs.bse, atol=atol)
    assert np.allclose(ours.fittedvalues, theirs.fittedvalues, atol=atol)
    assert ours.deviance == pytest.approx(theirs.deviance, rel=2e-9)
    assert ours.llf == pytest.approx(theirs.llf, rel=2e-9)
    assert ours.aic == pytest.approx(theirs.aic, rel=2e-9)


def test_gaussian_glm_parity(design):
    rng, x = design
    y = x @ np.array([0.3, -0.8, 0.2, 1.1, 0.0, -0.4]) + rng.normal(
        scale=0.7, size=len(x)
    )
    ours = msm.GLM(y, x, family=msm.families.Gaussian()).fit()
    theirs = sm.GLM(y, x, family=sm.families.Gaussian()).fit()
    assert_glm_parity(ours, theirs)


def test_binomial_logit_glm_parity(design):
    rng, x = design
    eta = x @ np.array([-0.2, 0.7, -0.5, 0.3, 0.1, -0.6])
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    ours = msm.GLM(y, x, family=msm.families.Binomial()).fit()
    theirs = sm.GLM(y, x, family=sm.families.Binomial()).fit()
    assert_glm_parity(ours, theirs)
    assert np.allclose(ours.pvalues, theirs.pvalues, atol=2e-8)


def test_poisson_log_glm_parity(design):
    rng, x = design
    mu = np.exp(x @ np.array([0.1, 0.2, -0.15, 0.1, 0.05, -0.1]))
    y = rng.poisson(mu)
    ours = msm.GLM(y, x, family=msm.families.Poisson()).fit()
    theirs = sm.GLM(y, x, family=sm.families.Poisson()).fit()
    assert_glm_parity(ours, theirs)


def test_poisson_offset_exposure_and_frequency_weights(design):
    rng, x = design
    exposure = rng.uniform(0.5, 3.0, len(x))
    offset = rng.normal(scale=0.1, size=len(x))
    weights = rng.integers(1, 4, len(x)).astype(float)
    mu = exposure * np.exp(offset + x @ np.array([-0.2, 0.1, 0.05, -0.1, 0.2, 0.08]))
    y = rng.poisson(mu)
    ours = msm.GLM(
        y,
        x,
        family=msm.families.Poisson(),
        exposure=exposure,
        offset=offset,
        freq_weights=weights,
    ).fit()
    theirs = sm.GLM(
        y,
        x,
        family=sm.families.Poisson(),
        exposure=exposure,
        offset=offset,
        freq_weights=weights,
    ).fit()
    assert_glm_parity(ours, theirs, atol=5e-8)


def test_gaussian_variance_weights(design):
    rng, x = design
    weights = rng.uniform(0.25, 2.0, len(x))
    y = x @ np.array([0.2, -0.4, 0.1, 0.7, -0.2, 0.3])
    y += rng.normal(scale=1.0 / np.sqrt(weights), size=len(x))
    ours = msm.GLM(
        y, x, family=msm.families.Gaussian(), var_weights=weights
    ).fit()
    theirs = sm.GLM(
        y, x, family=sm.families.Gaussian(), var_weights=weights
    ).fit()
    assert_glm_parity(ours, theirs, atol=2e-8)
    assert np.allclose(ours.conf_int(), theirs.conf_int(), atol=2e-8)


def test_glm_prediction_contract(design):
    rng, x = design
    y = rng.poisson(np.exp(x @ np.array([0.0, 0.1, 0.1, -0.1, 0.2, 0.0])))
    result = msm.GLM(y, x, family=msm.families.Poisson()).fit()
    assert np.allclose(result.predict(x[:10]), result.fittedvalues[:10])
    linear = result.model.predict(result.params, x[:10], which="linear")
    assert np.allclose(np.exp(linear), result.predict(x[:10]))


def test_glm_rejects_noncanonical_link(design):
    _, x = design
    with pytest.raises(NotImplementedError):
        msm.families.Poisson(link=msm.families.links.Identity())


def test_glm_boundary_argument_validation(design):
    rng, x = design
    y = rng.poisson(np.ones(len(x)))
    model = msm.GLM(y, x, family=msm.families.Poisson())
    with pytest.raises(ValueError, match="positive integer"):
        model.fit(maxiter=1.5)
    with pytest.raises(ValueError, match="finite and positive"):
        model.fit(tol=np.nan)
    result = model.fit()
    with pytest.raises(ValueError, match="incompatible"):
        result.predict(x[:, :-1])
    with pytest.raises(ValueError, match="strictly positive"):
        result.predict(x[:2], exposure=[1.0, 0.0])
