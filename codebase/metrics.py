"""
metrics.py - Performance Analysis & Risk Metrics
================================================
Fixed:
  - sharpe_ratio() and volatility() now use ddof=1 (pandas convention)
    to match moccm_grader_modified.py which calls returns.std() (ddof=1 default)
  - Annual return calculation, hit ratio, trade-based metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")


# Constants
BARS_PER_DAY = 75
TRADING_DAYS_PER_YEAR = 252
BARS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR  # 18,900


# ══════════════════════════════════════════════════════════════════════
# Returns Metrics
# ══════════════════════════════════════════════════════════════════════

def annual_return(nav: np.ndarray, timestamps: pd.Index) -> float:
    """
    Compute annualized return using actual bar count.
    Annual Return = (nav_end / nav_start) ^ (BARS_PER_YEAR / n_bars) - 1
    """
    start_nav = nav[0]
    end_nav   = nav[-1]
    n_bars    = len(nav)

    if start_nav <= 0 or n_bars < 2:
        return 0.0

    years = n_bars / BARS_PER_YEAR
    if years <= 0:
        return 0.0

    return float((end_nav / start_nav) ** (1 / years) - 1)


def cagr(nav: np.ndarray, timestamps: pd.Index) -> float:
    """Compound Annual Growth Rate."""
    return annual_return(nav, timestamps)


def total_return(nav: np.ndarray) -> float:
    """Cumulative return from start to end."""
    if nav[0] <= 0:
        return 0.0
    return float((nav[-1] / nav[0]) - 1)


# ══════════════════════════════════════════════════════════════════════
# Risk Metrics
# ══════════════════════════════════════════════════════════════════════

def volatility(returns: np.ndarray) -> float:
    """
    Annualized volatility.
    σ_annual = σ_bar × √BARS_PER_YEAR

    Uses ddof=1 (unbiased / pandas convention) to match grader's returns.std().
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 2:
        return 0.0
    daily_vol = float(np.std(valid, ddof=1))   # FIX: was ddof=0 (numpy default)
    return daily_vol * np.sqrt(BARS_PER_YEAR)


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Sharpe Ratio = (μ_portfolio - r_f) / σ_portfolio × √BARS_PER_YEAR

    Uses ddof=1 for std to match moccm_grader_modified.py (returns.std() default).
    Note: for the official score you should use grader_sharpe() in main.py which
    also applies fee drag on Net_NAV. This function operates on whatever returns
    array you pass in.
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 2:
        return 0.0
    mean_ret = float(np.mean(valid))
    std_ret  = float(np.std(valid, ddof=1))    # FIX: was ddof=0 (numpy default)
    if std_ret <= 0:
        return 0.0
    return (mean_ret - risk_free_rate) * np.sqrt(BARS_PER_YEAR) / std_ret


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Sortino Ratio = (μ - r_f) / σ_downside × √BARS_PER_YEAR
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 2:
        return 0.0
    mean_ret = float(np.mean(valid))
    downside_returns = valid[valid < 0]
    if len(downside_returns) < 2:
        downside_vol = 0.0
    else:
        downside_vol = float(np.std(downside_returns, ddof=1))
    if downside_vol <= 0:
        return 0.0
    return (mean_ret - risk_free_rate) * np.sqrt(BARS_PER_YEAR) / downside_vol


def calmar_ratio(nav, timestamps):
    ann = annual_return(nav, timestamps)
    dd = maximum_drawdown(nav)
    if abs(dd) < 1e-12:
        return 0.0
    return ann / abs(dd)


# ══════════════════════════════════════════════════════════════════════
# Drawdown Analysis
# ══════════════════════════════════════════════════════════════════════

def running_maximum(nav: np.ndarray) -> np.ndarray:
    """Compute running (cumulative) maximum."""
    return np.maximum.accumulate(nav)


def drawdown(nav: np.ndarray) -> np.ndarray:
    """
    Drawdown at each point.
    DD_t = (nav_t - max_nav) / max_nav
    """
    running_max = running_maximum(nav)
    return (nav - running_max) / (running_max + 1e-12)


def maximum_drawdown(nav: np.ndarray) -> float:
    """Maximum (worst) drawdown."""
    dd = drawdown(nav)
    return float(np.min(dd)) if len(dd) > 0 else 0.0


def average_drawdown(nav: np.ndarray) -> float:
    """Average of all drawdowns."""
    dd = drawdown(nav)
    return float(np.mean(dd[dd < 0])) if np.any(dd < 0) else 0.0


# ══════════════════════════════════════════════════════════════════════
# Win/Loss Statistics (Trade-based, not bar-based)
# ══════════════════════════════════════════════════════════════════════

def _extract_trades(position_changes: np.ndarray, pnl_bar: np.ndarray):
    """
    Extract trades from position changes.
    Returns list of trade PnLs.
    """
    trades = []
    current_trade_pnl = 0.0
    in_trade = False

    for i in range(len(position_changes)):
        if position_changes[i] != 0 and not in_trade:
            in_trade = True
            current_trade_pnl = 0.0
        elif position_changes[i] != 0 and in_trade:
            current_trade_pnl += pnl_bar[i]
            trades.append(current_trade_pnl)
            in_trade = False
            current_trade_pnl = 0.0
        elif in_trade:
            current_trade_pnl += pnl_bar[i]

    if in_trade:
        trades.append(current_trade_pnl)

    return np.array(trades)


def profit_factor(pnl_trades: np.ndarray) -> float:
    """
    Profit Factor = Σ(wins) / |Σ(losses)|
    """
    wins   = pnl_trades[pnl_trades > 0].sum()
    losses = abs(pnl_trades[pnl_trades < 0].sum())
    if losses <= 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses)


def hit_ratio(pnl_trades: np.ndarray) -> float:
    """Percentage of profitable trades."""
    if len(pnl_trades) == 0:
        return 0.0
    return float((pnl_trades > 0).sum() / len(pnl_trades))


def avg_win(pnl_trades: np.ndarray) -> float:
    """Average profit per winning trade."""
    wins = pnl_trades[pnl_trades > 0]
    return float(wins.mean()) if len(wins) > 0 else 0.0


def avg_loss(pnl_trades: np.ndarray) -> float:
    """Average loss per losing trade."""
    losses = pnl_trades[pnl_trades < 0]
    return float(losses.mean()) if len(losses) > 0 else 0.0


def win_loss_ratio(pnl_trades: np.ndarray) -> float:
    """Ratio of average win to average loss."""
    aw = avg_win(pnl_trades)
    al = avg_loss(pnl_trades)
    if al >= 0 or al == 0:
        return 0.0
    return float(aw / abs(al))


# ══════════════════════════════════════════════════════════════════════
# Risk Tail Metrics
# ══════════════════════════════════════════════════════════════════════

def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Value at Risk (VaR):
    VaR_α = worst return at α percentile.
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 5:
        return 0.0
    return float(np.percentile(valid, (1 - confidence) * 100))


def expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR):
    CVaR_α = mean of returns worse than VaR_α.
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 5:
        return 0.0
    var  = value_at_risk(valid, confidence)
    tail = valid[valid <= var]
    if len(tail) == 0:
        return var
    return float(np.mean(tail))


def skewness(returns: np.ndarray) -> float:
    """Distribution skewness."""
    valid = returns[np.isfinite(returns)]
    if len(valid) < 3:
        return 0.0
    return float(stats.skew(valid))


def kurtosis_excess(returns: np.ndarray) -> float:
    """Excess kurtosis."""
    valid = returns[np.isfinite(returns)]
    if len(valid) < 4:
        return 0.0
    return float(stats.kurtosis(valid))


def tail_ratio(returns: np.ndarray) -> float:
    """
    Tail Ratio = |gains_99th| / |losses_1st|
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 10:
        return 1.0
    gains_tail = abs(np.percentile(valid, 99))
    loss_tail  = abs(np.percentile(valid, 1))
    if loss_tail <= 0:
        return 1.0
    return float(gains_tail / loss_tail)


# ══════════════════════════════════════════════════════════════════════
# Turnover & Trading Statistics
# ══════════════════════════════════════════════════════════════════════

def average_turnover(turnover_series: np.ndarray) -> float:
    """Average turnover per period."""
    valid = turnover_series[turnover_series > 0]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def avg_holding_period(turnover_series: np.ndarray, nav: np.ndarray) -> float:
    """
    Average holding period in bars.
    Holding Period ≈ (Avg NAV) / (Avg Daily Turnover) × (1/BARS_PER_DAY)
    """
    avg_nav      = np.mean(nav)
    avg_to_daily = average_turnover(turnover_series) * BARS_PER_DAY

    if avg_to_daily <= 0:
        return float('inf')

    return avg_nav / avg_to_daily


# ══════════════════════════════════════════════════════════════════════
# Comprehensive Metrics Summary
# ══════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    backtest_results: pd.DataFrame,
    timestamps: pd.Index,
) -> Dict:
    """
    Compute all performance metrics from backtest results.

    Parameters
    ----------
    backtest_results : DataFrame with columns [nav, turnover, position?]
    timestamps       : index of timestamps
    """
    nav = backtest_results["nav"].values

    # Calculate bar-level PnL
    pnl_bar = np.diff(nav, prepend=nav[0])

    # Get turnover
    if "turnover" in backtest_results.columns:
        turnover_series = backtest_results["turnover"].values
    elif "Interval_Turnover" in backtest_results.columns:
        turnover_series = backtest_results["Interval_Turnover"].values
    else:
        turnover_series = np.zeros(len(nav))

    # Bar-level returns for risk metrics
    returns_bar = pnl_bar / (nav + 1e-12)

    # Extract trades from position changes (if position column exists)
    if "position" in backtest_results.columns:
        position_changes = backtest_results["position"].diff().fillna(0).values
        trades = _extract_trades(position_changes, pnl_bar)
    else:
        trades = pnl_bar[pnl_bar != 0]

    # Returns
    ann_ret   = annual_return(nav, timestamps)
    total_ret = total_return(nav)
    cagr_val  = cagr(nav, timestamps)

    # Risk — sharpe_ratio() now uses ddof=1 internally
    vol     = volatility(returns_bar)
    sr      = sharpe_ratio(returns_bar)
    sortino = sortino_ratio(returns_bar)
    calmar  = calmar_ratio(nav, timestamps)

    # Drawdown
    max_dd = maximum_drawdown(nav)
    avg_dd = average_drawdown(nav)

    # Win/Loss (trade-based)
    pf  = profit_factor(trades)
    hr  = hit_ratio(trades) if len(trades) > 0 else 0.0
    aw  = avg_win(trades)
    al  = avg_loss(trades)
    wlr = win_loss_ratio(trades)

    # Tail
    var_95  = value_at_risk(returns_bar, 0.95)
    cvar_95 = expected_shortfall(returns_bar, 0.95)
    skew    = skewness(returns_bar)
    kurt    = kurtosis_excess(returns_bar)
    tail_r  = tail_ratio(returns_bar)

    # Trading
    avg_to = average_turnover(turnover_series)
    avg_hp = avg_holding_period(turnover_series, nav)

    return {
        "annual_return":      round(ann_ret,   6),
        "total_return":       round(total_ret, 6),
        "cagr":               round(cagr_val,  6),
        "volatility":         round(vol,       6),
        "sharpe_ratio":       round(sr,        4),
        "sortino_ratio":      round(sortino,   4),
        "calmar_ratio":       round(calmar,    4),
        "maximum_drawdown":   round(max_dd,    6),
        "average_drawdown":   round(avg_dd,    6),
        "profit_factor":      round(pf,        4),
        "hit_ratio":          round(hr,        4),
        "avg_win":            round(aw,        8),
        "avg_loss":           round(al,        8),
        "win_loss_ratio":     round(wlr,       4),
        "value_at_risk_95":   round(var_95,    6),
        "cvar_95":            round(cvar_95,   6),
        "skewness":           round(skew,      4),
        "kurtosis":           round(kurt,      4),
        "tail_ratio":         round(tail_r,    4),
        "average_turnover":   round(avg_to,    2),
        "avg_holding_period": round(avg_hp,    2),
        "num_trades":         len(trades),
    }


def print_metrics_report(metrics: Dict):
    """Pretty-print performance metrics."""
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS".center(70))
    print("=" * 70)

    print("\n[Returns]")
    print(f"  Annual Return       : {metrics['annual_return']:>8.2%}")
    print(f"  Total Return        : {metrics['total_return']:>8.2%}")
    print(f"  CAGR                : {metrics['cagr']:>8.2%}")

    print("\n[Risk]")
    print(f"  Volatility (Annual) : {metrics['volatility']:>8.2%}")
    print(f"  Sharpe Ratio        : {metrics['sharpe_ratio']:>8.4f}")
    print(f"  Sortino Ratio       : {metrics['sortino_ratio']:>8.4f}")
    print(f"  Calmar Ratio        : {metrics['calmar_ratio']:>8.4f}")

    print("\n[Drawdown]")
    print(f"  Maximum Drawdown    : {metrics['maximum_drawdown']:>8.2%}")
    print(f"  Average Drawdown    : {metrics['average_drawdown']:>8.2%}")

    print("\n[Win/Loss]")
    print(f"  Profit Factor       : {metrics['profit_factor']:>8.4f}")
    print(f"  Hit Ratio           : {metrics['hit_ratio']:>8.2%}")
    print(f"  Avg Win             : ${metrics['avg_win']:>8,.2f}")
    print(f"  Avg Loss            : ${metrics['avg_loss']:>8,.2f}")
    print(f"  Win/Loss Ratio      : {metrics['win_loss_ratio']:>8.4f}")
    print(f"  Number of Trades    : {metrics.get('num_trades', 0):>8}")

    print("\n[Tail Risk]")
    print(f"  VaR (95%)           : {metrics['value_at_risk_95']:>8.2%}")
    print(f"  CVaR (95%)          : {metrics['cvar_95']:>8.2%}")
    print(f"  Skewness            : {metrics['skewness']:>8.4f}")
    print(f"  Kurtosis            : {metrics['kurtosis']:>8.4f}")
    print(f"  Tail Ratio          : {metrics['tail_ratio']:>8.4f}")

    print("\n[Trading]")
    print(f"  Avg Turnover        : ${metrics['average_turnover']:>8,.2f}")
    print(f"  Avg Holding Period  : {metrics['avg_holding_period']:>8.2f} bars")

    print("=" * 70 + "\n")