from __future__ import annotations

import numpy as np

from .interfaces import StrategyBuilderInterface
from .feature_factory import FeatureFactory
from .utils import ensure_2d, safe_float32


class StrategyBuilder(StrategyBuilderInterface):
    def __init__(self):
        self._features = FeatureFactory()

    def threshold_strategy(self, signal: np.ndarray, upper: float, lower: float) -> np.ndarray:
        x = ensure_2d("signal", safe_float32(signal))
        if lower > upper:
            raise ValueError("lower must be <= upper")
        out = np.zeros_like(x, dtype=np.int8)
        out[x >= upper] = 1
        out[x <= lower] = -1
        return out

    def mean_reversion_strategy(self, signal: np.ndarray, entry_z: float, exit_z: float) -> np.ndarray:
        x = ensure_2d("signal", safe_float32(signal))
        if entry_z <= 0 or exit_z < 0:
            raise ValueError("entry_z must be > 0 and exit_z must be >= 0")
        z = self._features.create_zscores(x, window=60)
        out = np.zeros_like(x, dtype=np.int8)
        out[z >= entry_z] = -1
        out[z <= -entry_z] = 1
        out[np.abs(z) <= exit_z] = 0
        return out

    def momentum_strategy(self, signal: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        x = ensure_2d("signal", safe_float32(signal))
        out = np.zeros_like(x, dtype=np.int8)
        out[x > threshold] = 1
        out[x < -threshold] = -1
        return out

    def pairs_strategy(self, price_a: np.ndarray, price_b: np.ndarray, entry_z: float = 2.0, exit_z: float = 0.5) -> np.ndarray:
        a = ensure_2d("price_a", safe_float32(price_a))
        b = ensure_2d("price_b", safe_float32(price_b))
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: price_a{a.shape} != price_b{b.shape}")
        if a.shape[1] != 1:
            raise ValueError("pairs_strategy expects single-column price inputs")

        spread = self._features.create_spreads(a, b)
        z = self._features.create_zscores(spread, window=60)
        pos = np.zeros((a.shape[0], 2), dtype=np.int8)
        pos[z[:, 0] >= entry_z, 0] = -1
        pos[z[:, 0] >= entry_z, 1] = 1
        pos[z[:, 0] <= -entry_z, 0] = 1
        pos[z[:, 0] <= -entry_z, 1] = -1
        pos[np.abs(z[:, 0]) <= exit_z, :] = 0
        return pos
