from __future__ import annotations

import numpy as np

from .interfaces import FeatureFactoryInterface
from .utils import ensure_2d, rolling_mean_std_2d, safe_float32, fractional_diff_weights


class FeatureFactory(FeatureFactoryInterface):
    def create_returns(self, prices: np.ndarray, period: int = 1) -> np.ndarray:
        x = ensure_2d("prices", safe_float32(prices))
        if period <= 0:
            raise ValueError("period must be > 0")
        out = np.full_like(x, np.nan, dtype=np.float32)
        out[period:, :] = x[period:, :] / (x[:-period, :] + 1e-12) - 1.0
        return out

    def create_zscores(self, values: np.ndarray, window: int = 60) -> np.ndarray:
        x = ensure_2d("values", safe_float32(values))
        if window < 2:
            raise ValueError("window must be >= 2")
        means, stds = rolling_mean_std_2d(x, window)
        return ((x - means) / (stds + 1e-6)).astype(np.float32)

    def create_spreads(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x = ensure_2d("a", safe_float32(a))
        y = ensure_2d("b", safe_float32(b))
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch: a{x.shape} != b{y.shape}")
        return (x - y).astype(np.float32)

    def create_momentum(self, prices: np.ndarray, window: int = 20) -> np.ndarray:
        x = ensure_2d("prices", safe_float32(prices))
        if window <= 0:
            raise ValueError("window must be > 0")
        out = np.full_like(x, np.nan, dtype=np.float32)
        out[window:, :] = x[window:, :] / (x[:-window, :] + 1e-12) - 1.0
        return out

    def fractional_differencing(self, series: np.ndarray, d: float, threshold: float = 1e-5) -> np.ndarray:
        x = ensure_2d("series", safe_float32(series))
        if not (0.0 <= d <= 2.0):
            raise ValueError("d must be in [0, 2]")
        w = fractional_diff_weights(d, x.shape[0], threshold)
        width = w.shape[0]
        out = np.full_like(x, np.nan, dtype=np.float32)
        for c in range(x.shape[1]):
            conv = np.convolve(x[:, c], w[::-1], mode="valid")
            out[width - 1 :, c] = conv.astype(np.float32)
        return out

    def alpha_volume_price_correlation(
        self, close: np.ndarray, volume: np.ndarray, window: int = 6, volume_delta_lag: int = 2
    ) -> np.ndarray:
        """Adapted from alpha-style formula using rolling correlation of price/volume deltas."""
        c = ensure_2d("close", safe_float32(close))
        v = ensure_2d("volume", safe_float32(volume))
        if c.shape != v.shape:
            raise ValueError(f"shape mismatch: close{c.shape} != volume{v.shape}")
        if window < 2:
            raise ValueError("window must be >= 2")
        if volume_delta_lag < 1:
            raise ValueError("volume_delta_lag must be >= 1")

        dlogv = np.full_like(v, np.nan, dtype=np.float32)
        # Default lag=2 is a common alpha-style choice to reduce single-bar microstructure noise.
        dlogv[volume_delta_lag:, :] = np.log(v[volume_delta_lag:, :] + 1e-12) - np.log(v[:-volume_delta_lag, :] + 1e-12)
        ret = np.full_like(c, np.nan, dtype=np.float32)
        ret[1:, :] = c[1:, :] / (c[:-1, :] + 1e-12) - 1.0

        ranks_v = np.argsort(np.argsort(dlogv, axis=0), axis=0).astype(np.float32)
        ranks_r = np.argsort(np.argsort(ret, axis=0), axis=0).astype(np.float32)

        out = np.full_like(c, np.nan, dtype=np.float32)
        for i in range(window - 1, c.shape[0]):
            xv = ranks_v[i - window + 1 : i + 1]
            yv = ranks_r[i - window + 1 : i + 1]
            xv_mu = np.nanmean(xv, axis=0)
            yv_mu = np.nanmean(yv, axis=0)
            cov = np.nanmean((xv - xv_mu) * (yv - yv_mu), axis=0)
            denom = np.nanstd(xv, axis=0) * np.nanstd(yv, axis=0) + 1e-12
            out[i, :] = -cov / denom
        return out
