from __future__ import annotations

import numpy as np


def add_constant(data, prepend: bool = True, has_constant: str = "skip"):
    array = np.asarray(data)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("data must be one- or two-dimensional")
    constant = np.ptp(array, axis=0) == 0
    nonzero_constant = constant & np.all(array != 0, axis=0)
    if np.any(nonzero_constant):
        if has_constant == "raise":
            raise ValueError("data already contains a constant column")
        if has_constant == "skip":
            return array
    ones = np.ones((array.shape[0], 1), dtype=np.result_type(array, float))
    return np.concatenate((ones, array), axis=1) if prepend else np.concatenate(
        (array, ones), axis=1
    )
