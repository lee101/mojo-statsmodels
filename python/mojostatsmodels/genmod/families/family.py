from __future__ import annotations

import numpy as np
from scipy.special import gammaln, xlogy

from . import links


class Family:
    family_id = -1
    name = "Family"

    def fitted(self, linear_predictor):
        return self.link.inverse(linear_predictor)

    def deviance(self, endog, mu, var_weights=1.0, freq_weights=1.0, scale=1.0):
        return float(np.sum(self.resid_dev(endog, mu) ** 2 * var_weights * freq_weights))


class Gaussian(Family):
    family_id = 0
    name = "Gaussian"

    def __init__(self, link=None, check_link: bool = True):
        self.link = links.Identity() if link is None else link
        if not isinstance(self.link, links.Identity):
            raise NotImplementedError("Gaussian is covered with the identity link")

    def resid_dev(self, endog, mu):
        return np.asarray(endog) - np.asarray(mu)

    def loglike(self, endog, mu, var_weights=1.0, freq_weights=1.0, scale=1.0):
        error = np.asarray(endog) - np.asarray(mu)
        return float(
            -0.5
            * np.sum(
                np.asarray(freq_weights)
                * (np.log(2.0 * np.pi * scale / np.asarray(var_weights)) + error**2 * np.asarray(var_weights) / scale)
            )
        )


class Binomial(Family):
    family_id = 1
    name = "Binomial"

    def __init__(self, link=None, check_link: bool = True):
        self.link = links.Logit() if link is None else link
        if not isinstance(self.link, links.Logit):
            raise NotImplementedError("Binomial is covered with the logit link")

    def resid_dev(self, endog, mu):
        y = np.asarray(endog)
        mu = np.clip(mu, 1e-12, 1.0 - 1e-12)
        unit = 2.0 * (xlogy(y, y / mu) + xlogy(1.0 - y, (1.0 - y) / (1.0 - mu)))
        return np.sign(y - mu) * np.sqrt(np.maximum(unit, 0.0))

    def loglike(self, endog, mu, var_weights=1.0, freq_weights=1.0, scale=1.0):
        y = np.asarray(endog)
        mu = np.clip(mu, 1e-12, 1.0 - 1e-12)
        return float(np.sum(np.asarray(freq_weights) * (
            xlogy(y, mu) + xlogy(1.0 - y, 1.0 - mu)
        )))


class Poisson(Family):
    family_id = 2
    name = "Poisson"

    def __init__(self, link=None, check_link: bool = True):
        self.link = links.Log() if link is None else link
        if not isinstance(self.link, links.Log):
            raise NotImplementedError("Poisson is covered with the log link")

    def resid_dev(self, endog, mu):
        y = np.asarray(endog)
        mu = np.clip(mu, 1e-300, None)
        unit = 2.0 * (xlogy(y, y / mu) - (y - mu))
        return np.sign(y - mu) * np.sqrt(np.maximum(unit, 0.0))

    def loglike(self, endog, mu, var_weights=1.0, freq_weights=1.0, scale=1.0):
        y = np.asarray(endog)
        return float(np.sum(np.asarray(freq_weights) * (
            xlogy(y, mu) - mu - gammaln(y + 1.0)
        )))
