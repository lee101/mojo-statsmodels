from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-statsmodels.so")
SOURCE = os.path.join(ROOT, "src", "statsmodels.mojo")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mst_wls_fit": ([I] * 9, I),
    "mst_linear_predict": ([I] * 5, None),
    "mst_glm_irls": ([I] * 14 + [F], I),
    "mst_arma_objective": ([I] * 9, F),
}

_library: ctypes.CDLL | None = None


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SOURCE)
    ):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip()[:5000])
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value, *, ndim: int | None = None, copy: bool = False) -> np.ndarray:
    if copy:
        array = np.array(value, dtype=np.float64, order="C", copy=True)
    else:
        array = np.ascontiguousarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected a {ndim}-dimensional array, got {array.ndim}")
    if not array.flags.c_contiguous or array.dtype != np.dtype(np.float64):
        raise TypeError("FFI buffers must be C-contiguous float64 arrays")
    return array


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != np.dtype(np.float64) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous float64 arrays")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffer has a null data pointer")
    return address
