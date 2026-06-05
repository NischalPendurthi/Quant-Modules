"""
backtest.py - Event-Driven Backtesting Engine
==============================================
Realistic simulation of trading strategies with:
- Position tracking and management
- Transaction cost implementation (10 bps)
- Long-only and long-short strategies
- Position sizing methods
- Risk constraints
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# Position Sizing Methods
# ══════════════════════════════════════════════════════════════════════

def equal_weight_sizing(
    signal: np.ndarray,
    n_assets: int = 1,
) -> np.ndarray:
    """
    Equal-weight position sizing:
    pos_t = sign(signal_t) / n_assets   [fully invested]
    """
    positions = np.sign(signal)
    return positions / max(n_assets, 1)


def volatility_adjusted_sizing(
    signal: np.ndarray,
    returns: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Volatility-adjusted (risk-parity) sizing:
    pos_t = signal_t / (volatility_t + ε)

    Reduces position size when volatility is high.
    """
    positions = np.zeros_like(signal)
    for t in range(window, len(signal)):
        vol_t = np.std(returns[t - window: t]) + 1e-8
        positions[t] = signal[t] / vol_t
    # Normalize to [-1, 1]
    max_pos = np.abs(positions).max() + 1e-12
    positions = positions / max_pos
    return positions


def signal_strength_sizing(
    signal: np.ndarray,
    percentile_bounds: Tuple[float, float] = (0.25, 0.75),
) -> np.ndarray:
    """
    Size based on signal strength (magnitude).
    pos_t = signal_t / max_signal   ∈ [-1, 1]

    Then apply percentile-based position limits.
    """
    max_sig = np.abs(signal).max() + 1e-12
    positions = signal / max_sig
    # Apply percentile bounds
    p_low, p_high = percentile_bounds
    lower_bound = np.percentile(np.abs(positions), p_low * 100)
    upper_bound = np.percentile(np.abs(positions), p_high * 100)
    mask = np.abs(positions) < lower_bound
    positions[mask] = 0
    return np.clip(positions, -1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
# Long-Only Strategy
# ══════════════════════════════════════════════════════════════════════

class LongOnlyBacktest:
    """
    Long-Only strategy:
        If signal > threshold:  Long (pos = 1)
        Else:                   Cash (pos = 0)

    Parameters
    ----------
    initial_capital : starting cash ($)
    position_size   : fraction of capital per trade (0.0-1.0)
    threshold       : signal threshold for entry
    transaction_cost: bps (basis points), default 10 bps
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        position_size: float = 1.0,
        threshold: float = 0.0,
        transaction_cost: float = 10.0,  # bps
    ):
        self.initial_capital    = initial_capital
        self.position_size      = position_size
        self.threshold          = threshold
        self.transaction_cost   = transaction_cost / 10000  # convert bps to fraction

    def backtest(
        self,
        timestamps: pd.Index,
        prices: np.ndarray,
        signal: np.ndarray,
        returns: np.ndarray,
    ) -> pd.DataFrame:
        """
        Run backtest.

        Parameters
        ----------
        timestamps : index of timestamps
        prices     : price series (for position sizing)
        signal     : predicted signal
        returns    : realized returns at each timestamp

        Returns
        -------
        backtest_df : detailed trade-by-trade output
            columns: [timestamp, price, signal, position, shares, 
                     cash, gross_pnl, realized_pnl, unrealized_pnl,
                     nav, exposure, turnover, costs]
        """
        n = len(signal)
        records = []

        cash        = self.initial_capital
        position    = 0  # shares held
        avg_price   = 0
        prev_pos    = 0

        for t in range(n):
            price_t   = prices[t]
            signal_t  = signal[t]
            ret_t     = returns[t]

            # ── Determine target position ────────────────────────────
            if signal_t > self.threshold:
                target_position = 1.0  # go long
            else:
                target_position = 0.0  # flat

            # ── Position change ─────────────────────────────────────
            pos_change = target_position - prev_pos
            nav_before = cash + position * price_t
            allocation = nav_before * self.position_size

            # ── Execute trade ────────────────────────────────────────
            if abs(pos_change) > 0.01:
                # Close old position
                if prev_pos != 0:
                    close_value = prev_pos * price_t
                    cash += close_value * (1 - self.transaction_cost)

                # Open new position
                if target_position > 0:
                    new_shares = allocation / price_t
                    cost = new_shares * price_t * (1 + self.transaction_cost)
                    if cash >= cost:
                        position = new_shares
                        avg_price = price_t
                        cash -= cost
                    else:
                        position = 0
                else:
                    position = 0

                turnover = abs(pos_change) * allocation
                costs = turnover * self.transaction_cost
            else:
                turnover = 0
                costs = 0

            # ── Mark-to-market ──────────────────────────────────────
            if position != 0:
                position_value = position * price_t
                unrealized_pnl = position * price_t * ret_t
                realized_pnl   = 0
            else:
                position_value = 0
                unrealized_pnl = 0
                realized_pnl   = 0

            nav = cash + position_value
            gross_pnl = nav - self.initial_capital

            records.append({
                "timestamp":      timestamps[t],
                "price":          round(price_t, 8),
                "signal":         round(signal_t, 8),
                "position":       round(position, 4),
                "shares":         round(position, 4),
                "cash":           round(cash, 2),
                "position_value": round(position_value, 2),
                "gross_pnl":      round(gross_pnl, 2),
                "realized_pnl":   round(realized_pnl, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "nav":            round(nav, 2),
                "turnover":       round(turnover, 2),
                "transaction_cost": round(costs, 2),
            })

            prev_pos = position

        return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════
# Long-Short Strategy
# ══════════════════════════════════════════════════════════════════════

class LongShortBacktest:
    """
    Long-Short strategy:
        If signal > upper_threshold:  Long (pos = +1)
        If signal < lower_threshold:  Short (pos = -1)
        Otherwise:                    Flat (pos = 0)

    Parameters
    ----------
    initial_capital   : starting cash ($)
    position_size     : fraction of capital per side
    upper_threshold   : entry threshold for long
    lower_threshold   : entry threshold for short
    transaction_cost  : bps
    """

    def __init__(
        self,
        initial_capital: float = 2_000_000,
        position_size: float = 0.5,
        upper_threshold: float = 0.5,
        lower_threshold: float = -0.5,
        transaction_cost: float = 10.0,
    ):
        self.initial_capital    = initial_capital
        self.position_size      = position_size
        self.upper_threshold    = upper_threshold
        self.lower_threshold    = lower_threshold
        self.transaction_cost   = transaction_cost / 10000

    def backtest(
        self,
        timestamps: pd.Index,
        prices: np.ndarray,
        signal: np.ndarray,
        returns: np.ndarray,
    ) -> pd.DataFrame:
        """
        Run long-short backtest.

        Returns
        -------
        backtest_df : detailed results
        """
        n = len(signal)
        records = []

        cash        = self.initial_capital
        position    = 0
        avg_price   = 0
        prev_pos    = 0

        for t in range(n):
            price_t   = prices[t]
            signal_t  = signal[t]
            ret_t     = returns[t]

            # ── Determine target position ────────────────────────────
            if signal_t > self.upper_threshold:
                target_position = 1.0
            elif signal_t < self.lower_threshold:
                target_position = -1.0
            else:
                target_position = 0.0

            # ── Position management ──────────────────────────────────
            pos_change = target_position - prev_pos
            nav_before = cash + position * price_t
            allocation = nav_before * self.position_size

            if abs(pos_change) > 0.01:
                # Close existing position
                if prev_pos != 0:
                    close_value = prev_pos * price_t
                    cash += close_value * (1 - self.transaction_cost * np.sign(close_value))

                # Open new position
                if target_position > 0:
                    new_shares = allocation / price_t
                    cost = new_shares * price_t * (1 + self.transaction_cost)
                    position = new_shares
                    cash -= cost
                elif target_position < 0:
                    new_shares = -allocation / price_t
                    cost = abs(new_shares) * price_t * (1 + self.transaction_cost)
                    position = new_shares
                    cash -= cost
                else:
                    position = 0

                turnover = abs(pos_change) * allocation
                costs = turnover * self.transaction_cost
            else:
                turnover = 0
                costs = 0

            # ── Mark-to-market ──────────────────────────────────────
            position_value = position * price_t
            unrealized_pnl = position * price_t * ret_t if position != 0 else 0
            nav = cash + position_value
            gross_pnl = nav - self.initial_capital

            records.append({
                "timestamp":      timestamps[t],
                "price":          round(price_t, 8),
                "signal":         round(signal_t, 8),
                "position":       round(position, 4),
                "shares":         round(position, 4),
                "cash":           round(cash, 2),
                "position_value": round(position_value, 2),
                "gross_pnl":      round(gross_pnl, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "nav":            round(nav, 2),
                "turnover":       round(turnover, 2),
                "transaction_cost": round(costs, 2),
            })

            prev_pos = position

        return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════
# Backtest Utilities
# ══════════════════════════════════════════════════════════════════════

def save_backtest_results(
    results_df: pd.DataFrame,
    output_path: str = "backtest_results.csv",
):
    """Save backtest results to CSV."""
    results_df.to_csv(output_path, index=False)
    print(f"[backtest] Results saved → {output_path}")
