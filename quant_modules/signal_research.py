from __future__ import annotations

import numpy as np

from .interfaces import SignalResearchInterface
from .signal_evaluator import SignalEvaluator
from .utils import ensure_2d, safe_float32


class SignalResearch(SignalResearchInterface):
    def __init__(self):
        self._evaluator = SignalEvaluator()

    def future_return_corr(self, signal: np.ndarray, future_returns: np.ndarray) -> float:
        x = ensure_2d("signal", safe_float32(signal))
        y = ensure_2d("future_returns", safe_float32(future_returns))
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch: signal{x.shape} != future_returns{y.shape}")
        ic = self._evaluator.compute_ic(x, y)
        return float(np.nanmean(np.diag(ic)))

    def information_coefficient(self, signal: np.ndarray, future_returns: np.ndarray) -> float:
        return self.future_return_corr(signal, future_returns)

    def rank_ic(self, signal: np.ndarray, future_returns: np.ndarray) -> float:
        x = ensure_2d("signal", safe_float32(signal))
        y = ensure_2d("future_returns", safe_float32(future_returns))
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch: signal{x.shape} != future_returns{y.shape}")
        ric = self._evaluator.compute_rank_ic(x, y)
        return float(np.nanmean(np.diag(ric)))

    def stability(self, signal: np.ndarray, future_returns: np.ndarray, window: int = 252) -> dict:
        x = ensure_2d("signal", safe_float32(signal))
        y = ensure_2d("future_returns", safe_float32(future_returns))
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch: signal{x.shape} != future_returns{y.shape}")
        if window < 2:
            raise ValueError("window must be >= 2")

        vals = []
        for i in range(window, x.shape[0] + 1):
            vals.append(self.future_return_corr(x[i - window : i], y[i - window : i]))
        arr = np.asarray(vals, dtype=np.float32)
        return {
            "rolling_ic": arr,
            "mean": float(np.nanmean(arr)) if arr.size else np.nan,
            "std": float(np.nanstd(arr)) if arr.size else np.nan,
            "positive_ratio": float(np.mean(arr > 0)) if arr.size else np.nan,
        }

    def decay_analysis(self, signal: np.ndarray, returns_by_horizon: np.ndarray) -> np.ndarray:
        x = ensure_2d("signal", safe_float32(signal))
        y = ensure_2d("returns_by_horizon", safe_float32(returns_by_horizon))
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"row mismatch: signal{x.shape} vs returns_by_horizon{y.shape}")
        if x.shape[1] != 1:
            raise ValueError("decay_analysis expects signal with a single column")
        return self._evaluator.compute_ic(x, y)[0]
