from __future__ import annotations

import numpy as np


class Identity:
    def __call__(self, mu):
        return np.asarray(mu)

    def inverse(self, z):
        return np.asarray(z)


class Logit:
    def __call__(self, mu):
        mu = np.clip(mu, 1e-12, 1.0 - 1e-12)
        return np.log(mu / (1.0 - mu))

    def inverse(self, z):
        z = np.asarray(z)
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


class Log:
    def __call__(self, mu):
        return np.log(np.clip(mu, 1e-300, None))

    def inverse(self, z):
        return np.exp(np.clip(z, -30.0, 30.0))


identity = Identity
logit = Logit
log = Log
