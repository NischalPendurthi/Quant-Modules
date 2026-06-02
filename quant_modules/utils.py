from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


def ensure_2d(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def ensure_1d(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def ensure_same_shape(left: np.ndarray, right: np.ndarray, left_name: str, right_name: str) -> None:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left_name}{left.shape} != {right_name}{right.shape}")


def safe_float32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32, copy=False)


@njit(cache=True)
def rolling_mean_std_2d(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = values.shape
    means = np.full((rows, cols), np.nan, dtype=np.float32)
    stds = np.full((rows, cols), np.nan, dtype=np.float32)

    for c in range(cols):
        csum = 0.0
        csum_sq = 0.0
        for r in range(rows):
            x = float(values[r, c])
            csum += x
            csum_sq += x * x
            if r >= window:
                old = float(values[r - window, c])
                csum -= old
                csum_sq -= old * old
            if r >= window - 1:
                m = csum / window
                var = csum_sq / window - m * m
                if var < 0.0:
                    var = 0.0
                means[r, c] = m
                stds[r, c] = np.sqrt(var)
    return means, stds


@njit(cache=True)
def fractional_diff_weights(d: float, size: int, threshold: float = 1e-5) -> np.ndarray:
    """Adapted from fractional differencing recurrence commonly used in mlfinlab."""
    weights = np.empty(size, dtype=np.float64)
    weights[0] = 1.0
    for k in range(1, size):
        weights[k] = -weights[k - 1] * (d - k + 1.0) / k
        if abs(weights[k]) < threshold:
            return weights[: k + 1]
    return weights
