"""
metrics.py - Performance Analysis & Risk Metrics
================================================
Comprehensive portfolio performance and risk measurement:
- Returns metrics (Annual Return, CAGR)
- Risk metrics (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Hit ratio and win/loss statistics
- Value at Risk (VaR) and Expected Shortfall (CVaR)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# Returns Metrics
# ══════════════════════════════════════════════════════════════════════

def annual_return(nav: np.ndarray, timestamps: pd.Index) -> float:
    """
    Compute annualized return.
    Return = (nav_end / nav_start) ^ (252 / n_days) - 1
    """
    start_nav = nav[0]
    end_nav   = nav[-1]
    n_days    = (timestamps[-1] - timestamps[0]).days
    years     = max(n_days / 365.25, 0.01)
    if start_nav <= 0:
        return 0.0
    return float((end_nav / start_nav) ** (1 / years) - 1)


def cagr(nav: np.ndarray, timestamps: pd.Index) -> float:
    """
    Compound Annual Growth Rate.
    CAGR = (final_value / initial_value) ^ (1 / years) - 1
    """
    return annual_return(nav, timestamps)


def total_return(nav: np.ndarray) -> float:
    """Cumulative return from start to end."""
    if nav[0] <= 0:
        return 0.0
    return float((nav[-1] / nav[0]) - 1)


# ══════════════════════════════════════════════════════════════════════
# Risk Metrics
# ══════════════════════════════════════════════════════════════════════

def volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Annualized volatility.
    σ_annual = σ_daily × √252
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 2:
        return 0.0
    daily_vol = float(np.std(valid))
    return daily_vol * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe Ratio = (μ_portfolio - r_f) / σ_portfolio × √252
    Measures excess return per unit of risk.
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 2:
        return 0.0
    mean_ret = float(np.mean(valid))
    vol      = volatility(valid, periods_per_year)
    if vol <= 0:
        return 0.0
    return (mean_ret - risk_free_rate) * np.sqrt(periods_per_year) / vol


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sortino Ratio = (μ - r_f) / σ_downside × √252
    Like Sharpe, but penalizes only downside volatility.
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 2:
        return 0.0
    mean_ret = float(np.mean(valid))
    downside_returns = valid[valid < 0]
    if len(downside_returns) < 2:
        downside_vol = 0.0
    else:
        downside_vol = float(np.std(downside_returns))
    if downside_vol <= 0:
        return 0.0
    return (mean_ret - risk_free_rate) * np.sqrt(periods_per_year) / downside_vol


def calmar_ratio(
    returns: np.ndarray,
    nav: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calmar Ratio = Annual Return / Maximum Drawdown
    Measures return relative to worst drawdown.
    """
    annual_ret = annual_return(nav, pd.date_range(start=0, periods=len(nav), freq="5min"))
    max_dd     = maximum_drawdown(nav)
    if max_dd >= 0 or max_dd == 0:  # no drawdown or invalid
        return 0.0
    return annual_ret / abs(max_dd)


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
# Win/Loss Statistics
# ══════════════════════════════════════════════════════════════════════

def profit_factor(pnl: np.ndarray) -> float:
    """
    Profit Factor = Σ(wins) / |Σ(losses)|
    Ratio of gross profit to gross loss.
    """
    wins   = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    if losses <= 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses)


def hit_ratio(pnl: np.ndarray) -> float:
    """Percentage of profitable trades."""
    if len(pnl) == 0:
        return 0.0
    return float((pnl > 0).sum() / len(pnl))


def avg_win(pnl: np.ndarray) -> float:
    """Average profit per winning trade."""
    wins = pnl[pnl > 0]
    return float(wins.mean()) if len(wins) > 0 else 0.0


def avg_loss(pnl: np.ndarray) -> float:
    """Average loss per losing trade."""
    losses = pnl[pnl < 0]
    return float(losses.mean()) if len(losses) > 0 else 0.0


def win_loss_ratio(pnl: np.ndarray) -> float:
    """Ratio of average win to average loss."""
    aw = avg_win(pnl)
    al = avg_loss(pnl)
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
    VaR_0.95 means the worst 5% loss.
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 5:
        return 0.0
    return float(np.percentile(valid, (1 - confidence) * 100))


def expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR):
    CVaR_α = mean of returns worse than VaR_α.
    More severe penalty for tail risk.
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 5:
        return 0.0
    var = value_at_risk(valid, confidence)
    return float(np.mean(valid[valid <= var]))


def skewness(returns: np.ndarray) -> float:
    """Distribution skewness. Negative = left tail (bad)."""
    valid = returns[np.isfinite(returns)]
    if len(valid) < 3:
        return 0.0
    return float(stats.skew(valid))


def kurtosis_excess(returns: np.ndarray) -> float:
    """Excess kurtosis. >0 = fatter tails (more extremes)."""
    valid = returns[np.isfinite(returns)]
    if len(valid) < 4:
        return 0.0
    return float(stats.kurtosis(valid))


def tail_ratio(returns: np.ndarray) -> float:
    """
    Tail Ratio = |gains_99th| / |losses_1st|
    Measures upside vs downside tail severity.
    >1 = more upside tail
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


def avg_holding_period(turnover_series: np.ndarray) -> float:
    """
    Approximate holding period.
    Assumes turnover ≈ 1/holding_period
    """
    avg_to = average_turnover(turnover_series)
    if avg_to <= 0:
        return float('inf')
    return 1.0 / avg_to


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
    backtest_results : DataFrame with columns [nav, realized_pnl, turnover]
    timestamps       : index of timestamps

    Returns
    -------
    metrics_dict : comprehensive performance summary
    """
    nav = backtest_results["nav"].values
    pnl = backtest_results["realized_pnl"].values
    turnover_series = backtest_results["turnover"].values
    returns = np.diff(nav) / (nav[:-1] + 1e-12)

    # Returns
    ann_ret = annual_return(nav, timestamps)
    total_ret = total_return(nav)
    cagr_val = cagr(nav, timestamps)

    # Risk
    vol = volatility(returns)
    sr = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    calmar = calmar_ratio(returns, nav)

    # Drawdown
    max_dd = maximum_drawdown(nav)
    avg_dd = average_drawdown(nav)

    # Win/Loss
    pf = profit_factor(pnl)
    hr = hit_ratio(pnl)
    aw = avg_win(pnl)
    al = avg_loss(pnl)
    wlr = win_loss_ratio(pnl)

    # Tail
    var_95 = value_at_risk(returns, 0.95)
    cvar_95 = expected_shortfall(returns, 0.95)
    skew = skewness(returns)
    kurt = kurtosis_excess(returns)
    tail_r = tail_ratio(returns)

    # Trading
    avg_to = average_turnover(turnover_series)
    avg_hp = avg_holding_period(turnover_series)

    return {
        "annual_return":        round(ann_ret, 6),
        "total_return":         round(total_ret, 6),
        "cagr":                 round(cagr_val, 6),
        "volatility":           round(vol, 6),
        "sharpe_ratio":         round(sr, 4),
        "sortino_ratio":        round(sortino, 4),
        "calmar_ratio":         round(calmar, 4),
        "maximum_drawdown":     round(max_dd, 6),
        "average_drawdown":     round(avg_dd, 6),
        "profit_factor":        round(pf, 4),
        "hit_ratio":            round(hr, 4),
        "avg_win":              round(aw, 8),
        "avg_loss":             round(al, 8),
        "win_loss_ratio":       round(wlr, 4),
        "value_at_risk_95":     round(var_95, 6),
        "cvar_95":              round(cvar_95, 6),
        "skewness":             round(skew, 4),
        "kurtosis":             round(kurt, 4),
        "tail_ratio":           round(tail_r, 4),
        "average_turnover":     round(avg_to, 2),
        "avg_holding_period":   round(avg_hp, 2),
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
    print(f"  Avg Win             : {metrics['avg_win']:>8.8f}")
    print(f"  Avg Loss            : {metrics['avg_loss']:>8.8f}")
    print(f"  Win/Loss Ratio      : {metrics['win_loss_ratio']:>8.4f}")

    print("\n[Tail Risk]")
    print(f"  VaR (95%)           : {metrics['value_at_risk_95']:>8.2%}")
    print(f"  CVaR (95%)          : {metrics['cvar_95']:>8.2%}")
    print(f"  Skewness            : {metrics['skewness']:>8.4f}")
    print(f"  Kurtosis            : {metrics['kurtosis']:>8.4f}")
    print(f"  Tail Ratio          : {metrics['tail_ratio']:>8.4f}")

    print("\n[Trading]")
    print(f"  Avg Turnover        : ${metrics['average_turnover']:>8,.2f}")
    print(f"  Avg Holding Period  : {metrics['avg_holding_period']:>8.2f} periods")

    print("=" * 70 + "\n")
