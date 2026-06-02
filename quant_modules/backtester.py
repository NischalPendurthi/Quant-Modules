from __future__ import annotations

import numpy as np

from .interfaces import BacktesterInterface, EventStrategyInterface
from .signal_evaluator import SignalEvaluator
from .utils import ensure_2d, safe_float32


class Backtester(BacktesterInterface):
    def __init__(self):
        self._evaluator = SignalEvaluator()

    def run_vectorized(self, prices: np.ndarray, signals: np.ndarray) -> dict:
        px = ensure_2d("prices", safe_float32(prices))
        sg = ensure_2d("signals", safe_float32(signals))
        if px.shape != sg.shape:
            raise ValueError(f"shape mismatch: prices{px.shape} != signals{sg.shape}")

        positions = np.sign(sg).astype(np.int8)
        pnl = self.compute_pnl(px, positions)
        return {
            "positions": positions,
            "pnl": pnl,
            "turnover": self._evaluator.compute_turnover(positions),
            "sharpe": self._evaluator.compute_sharpe(pnl),
            "total_pnl": np.nansum(pnl, axis=0).astype(np.float32),
        }

    def run_event_driven(self, data: np.ndarray, strategy: object) -> dict:
        arr = ensure_2d("data", safe_float32(data))
        if not isinstance(strategy, EventStrategyInterface):
            raise TypeError("strategy must implement generate_signal(row: np.ndarray) -> np.ndarray")

        signals = np.empty_like(arr, dtype=np.float32)
        for i in range(arr.shape[0]):
            signal = np.asarray(strategy.generate_signal(arr[i]))
            if signal.shape != (arr.shape[1],):
                raise ValueError(f"strategy signal must have shape ({arr.shape[1]},), got {signal.shape}")
            signals[i, :] = signal
        return self.run_vectorized(arr, signals)

    def compute_pnl(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        px = ensure_2d("prices", safe_float32(prices))
        pos = ensure_2d("positions", safe_float32(positions))
        if px.shape != pos.shape:
            raise ValueError(f"shape mismatch: prices{px.shape} != positions{pos.shape}")
        ret = np.full_like(px, 0.0, dtype=np.float32)
        ret[1:, :] = px[1:, :] / (px[:-1, :] + 1e-12) - 1.0

        lag_pos = np.zeros_like(pos, dtype=np.float32)
        lag_pos[1:, :] = pos[:-1, :]
        return (lag_pos * ret).astype(np.float32)
