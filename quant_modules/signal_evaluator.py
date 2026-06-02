from __future__ import annotations

import numpy as np

from .interfaces import SignalEvaluatorInterface
from .utils import ensure_2d, safe_float32


class SignalEvaluator(SignalEvaluatorInterface):
    def compute_ic(self, signals: np.ndarray, forward_returns: np.ndarray) -> np.ndarray:
        x = ensure_2d("signals", safe_float32(signals))
        y = ensure_2d("forward_returns", safe_float32(forward_returns))
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"row mismatch: signals{x.shape} vs forward_returns{y.shape}")

        x_centered = x - np.nanmean(x, axis=0)
        y_centered = y - np.nanmean(y, axis=0)
        num = np.nan_to_num(x_centered).T @ np.nan_to_num(y_centered)

        x_ss = np.nansum(x_centered * x_centered, axis=0)
        y_ss = np.nansum(y_centered * y_centered, axis=0)
        denom = np.sqrt(x_ss[:, None] * y_ss[None, :]) + 1e-12
        return (num / denom).astype(np.float32)

    def compute_rank_ic(self, signals: np.ndarray, forward_returns: np.ndarray) -> np.ndarray:
        x = ensure_2d("signals", safe_float32(signals))
        y = ensure_2d("forward_returns", safe_float32(forward_returns))
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"row mismatch: signals{x.shape} vs forward_returns{y.shape}")
        xr = np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float32)
        yr = np.argsort(np.argsort(y, axis=0), axis=0).astype(np.float32)
        return self.compute_ic(xr, yr)

    def compute_turnover(self, positions: np.ndarray) -> np.ndarray:
        p = ensure_2d("positions", safe_float32(positions))
        if p.shape[0] < 2:
            return np.zeros(p.shape[1], dtype=np.float32)
        return np.nanmean(np.abs(np.diff(p, axis=0)), axis=0).astype(np.float32)

    def compute_sharpe(self, pnl: np.ndarray, periods_per_year: int = 252 * 390) -> np.ndarray:
        """Annualized Sharpe ratio.

        Expects minute-level PnL by default.
        The default 98,280 periods/year uses US equity regular-session minute bars
        (252 trading days * 390 minutes per day).
        """
        x = ensure_2d("pnl", safe_float32(pnl))
        mu = np.nanmean(x, axis=0)
        sigma = np.nanstd(x, axis=0) + 1e-12
        return (mu / sigma * np.sqrt(periods_per_year)).astype(np.float32)
