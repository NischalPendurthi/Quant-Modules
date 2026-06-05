"""
signal.py - Alpha & Signal Quality Analysis
=============================================
Comprehensive signal evaluation metrics:
- Information Coefficient (IC) analysis
- Signal quality metrics
- Stability and decay analysis
- Monthly performance attribution
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# Information Metrics
# ══════════════════════════════════════════════════════════════════════

def pearson_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Information Coefficient (Pearson):
    IC = Cov(signal, y) / [σ(signal) · σ(y)]
    Measures linear predictive power of the signal.
    """
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 5:
        return np.nan
    rho, _ = stats.pearsonr(y_true[valid], y_pred[valid])
    return float(rho)


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Information Coefficient (Spearman / Rank IC):
    RankIC = ρ(rank(signal), rank(y))
    More robust to outliers, measures monotonic relationship.
    """
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 5:
        return np.nan
    rho, _ = stats.spearmanr(y_true[valid], y_pred[valid])
    return float(rho)


def compute_ic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: Optional[pd.Index] = None,
) -> Dict:
    """
    Compute comprehensive IC metrics.

    Returns
    -------
    dict with:
        mean_ic, median_ic, ic_std, icir
        mean_rank_ic, rank_ic_std, rank_icir
    """
    # Fixed: Compute IC for the entire arrays, not creating single-element arrays
    ic = pearson_ic(y_true, y_pred)
    rank_ic = spearman_ic(y_true, y_pred)

    ic_mean      = ic if not np.isnan(ic) else 0.0
    ic_median    = ic if not np.isnan(ic) else 0.0
    ic_std       = 0.0  # Single value has no std
    icir         = ic_mean / (ic_std + 1e-12) if ic_std > 0 else ic_mean * 1e12

    rank_ic_mean = rank_ic if not np.isnan(rank_ic) else 0.0
    rank_ic_std  = 0.0
    rank_icir    = rank_ic_mean / (rank_ic_std + 1e-12) if rank_ic_std > 0 else rank_ic_mean * 1e12

    return {
        "mean_ic":       round(ic_mean,      6),
        "median_ic":     round(ic_median,    6),
        "ic_std":        round(ic_std,       6),
        "icir":          round(icir,         6),
        "mean_rank_ic":  round(rank_ic_mean, 6),
        "rank_ic_std":   round(rank_ic_std,  6),
        "rank_icir":     round(rank_icir,    6),
    }


def rolling_ic(
    y_true: pd.Series,
    y_pred: pd.Series,
    window: int = 252,
    method: str = "pearson",
) -> pd.Series:
    """
    Compute rolling Information Coefficient over a time window.

    Parameters
    ----------
    y_true : true target values (pd.Series with timestamp index)
    y_pred : predicted values
    window : rolling window in samples
    method : 'pearson' or 'spearman'

    Returns
    -------
    ic_series : rolling IC values
    """
    ic_vals = []
    idx     = []

    for i in range(window, len(y_true) + 1):
        y_w = y_true.iloc[i - window: i].values
        p_w = y_pred.iloc[i - window: i].values
        if method == "pearson":
            ic = pearson_ic(y_w, p_w)
        else:
            ic = spearman_ic(y_w, p_w)
        ic_vals.append(ic)
        idx.append(y_true.index[i - 1])

    return pd.Series(ic_vals, index=idx, name="rolling_ic")


def monthly_ic(
    y_true: pd.Series,
    y_pred: pd.Series,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Decompose IC by month.

    Returns
    -------
    df : monthly IC with columns [year, month, ic, n_samples]
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    df["year_month"] = df.index.to_period("M")

    records = []
    for ym, group in df.groupby("year_month"):
        y_w = group["y_true"].values
        p_w = group["y_pred"].values
        if method == "pearson":
            ic = pearson_ic(y_w, p_w)
        else:
            ic = spearman_ic(y_w, p_w)
        records.append({
            "period":    str(ym),
            "ic":        round(ic if ic is not None else 0.0, 6),
            "n_samples": len(group),
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════
# Signal Quality Metrics
# ══════════════════════════════════════════════════════════════════════

def hit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of predictions with the correct sign.
    Hit Ratio = Σ[sign(y_pred) = sign(y_true)] / n
    """
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t   = y_true[valid]
    y_p   = y_pred[valid]
    if len(y_t) == 0:
        return 0.0
    correct = np.sign(y_t) * np.sign(y_p) > 0
    return float(correct.sum() / len(y_t))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Decompose hit ratio into long and short accuracy.

    Returns
    -------
    dict with: overall_accuracy, long_accuracy, short_accuracy
    """
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t   = y_true[valid]
    y_p   = y_pred[valid]

    overall = hit_ratio(y_t, y_p)

    long_mask  = y_p > 0
    short_mask = y_p < 0

    long_correct = np.sum((y_t[long_mask] > 0)) if long_mask.sum() > 0 else 0
    long_acc     = long_correct / long_mask.sum() if long_mask.sum() > 0 else np.nan

    short_correct = np.sum((y_t[short_mask] < 0)) if short_mask.sum() > 0 else 0
    short_acc     = short_correct / short_mask.sum() if short_mask.sum() > 0 else np.nan

    return {
        "overall_accuracy": round(overall, 4),
        "long_accuracy":    round(long_acc if not np.isnan(long_acc) else 0.0, 4),
        "short_accuracy":   round(short_acc if not np.isnan(short_acc) else 0.0, 4),
        "long_count":       int(long_mask.sum()),
        "short_count":      int(short_mask.sum()),
    }


def signal_stability(y_pred: np.ndarray, window: int = 250) -> float:
    """
    Measure how consistent the signal is over time.
    Uses autocorrelation of the signal over rolling windows.

    Stability = mean(autocorr at lag 1) across rolling windows
    """
    acf_vals = []
    y_pred_flat = y_pred.flatten() if isinstance(y_pred, np.ndarray) else np.array(y_pred)
    
    for i in range(window, len(y_pred_flat)):
        w = y_pred_flat[i - window: i]
        if not np.isfinite(w).all():
            continue
        # Autocorrelation at lag 1
        if len(w) > 1 and np.var(w) > 0:
            c0 = np.cov(w[:-1], w[1:])[0, 1]
            v0 = np.var(w)
            if v0 > 0:
                acf_vals.append(c0 / v0)
    return float(np.mean(acf_vals)) if acf_vals else 0.0


def signal_decay(
    y_true: pd.Series,
    y_pred: pd.Series,
    max_lag: int = 10,
) -> pd.Series:
    """
    Measure how predictive power decays with lookahead.
    Signal_decay[lag] = IC(y_pred_t, y_true_{t+lag})

    Returns
    -------
    ic_by_lag : Series of IC values at different lookaheads
    """
    ic_vals = []
    for lag in range(max_lag + 1):
        y_t_shifted = y_true.shift(-lag)  # lookahead
        valid       = ~(y_t_shifted.isna() | y_pred.isna())
        if valid.sum() < 10:
            ic_vals.append(np.nan)
        else:
            ic = pearson_ic(
                y_t_shifted[valid].values,
                y_pred[valid].values,
            )
            ic_vals.append(ic if not np.isnan(ic) else 0.0)
    return pd.Series(ic_vals, index=range(max_lag + 1), name="ic_by_lag")


def prediction_distribution(y_pred: np.ndarray) -> Dict:
    """
    Analyze the distribution of predicted signals.

    Returns
    -------
    dict with: mean, std, skew, kurtosis, min, max, pct_positive
    """
    valid = np.isfinite(y_pred)
    y_p   = y_pred[valid]
    if len(y_p) == 0:
        return {
            "mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0,
            "min": 0.0, "max": 0.0, "pct_positive": 0.0, "pct_negative": 0.0
        }
    return {
        "mean":         round(float(np.mean(y_p)), 8),
        "std":          round(float(np.std(y_p)),  8),
        "skew":         round(float(stats.skew(y_p)), 4),
        "kurtosis":     round(float(stats.kurtosis(y_p)), 4),
        "min":          round(float(np.min(y_p)), 8),
        "max":          round(float(np.max(y_p)), 8),
        "pct_positive": round(float((y_p > 0).sum() / len(y_p) * 100), 2),
        "pct_negative": round(float((y_p < 0).sum() / len(y_p) * 100), 2),
    }


# ══════════════════════════════════════════════════════════════════════
# Comprehensive Signal Analysis
# ══════════════════════════════════════════════════════════════════════

def analyze_signal(
    y_true: pd.Series,
    y_pred: pd.Series,
    rolling_window: int = 252,
    signal_decay_max_lag: int = 10,
) -> Dict:
    """
    Comprehensive signal quality analysis.

    Returns
    -------
    analysis_dict : contains all signal quality metrics
    """
    y_t_arr = y_true.values
    y_p_arr = y_pred.values

    # IC metrics
    ic_metrics = compute_ic_metrics(y_t_arr, y_p_arr)

    # Signal quality
    da = directional_accuracy(y_t_arr, y_p_arr)
    hr = hit_ratio(y_t_arr, y_p_arr)

    # Signal characteristics
    pred_dist = prediction_distribution(y_p_arr)
    stability = signal_stability(y_p_arr, window=rolling_window)

    # Signal decay
    decay = signal_decay(y_true, y_pred, max_lag=signal_decay_max_lag)

    return {
        "ic_metrics":            ic_metrics,
        "directional_accuracy":  da,
        "hit_ratio":             round(hr, 4),
        "prediction_distribution": pred_dist,
        "signal_stability":      round(stability, 6),
        "signal_decay":          decay.to_dict(),
    }


# ══════════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════════

def print_signal_report(analysis: Dict):
    """Pretty-print signal analysis report."""
    print("\n" + "=" * 70)
    print("SIGNAL QUALITY ANALYSIS".center(70))
    print("=" * 70)

    print("\n[IC Metrics]")
    for key, val in analysis["ic_metrics"].items():
        print(f"  {key:20s}: {val:>8}")

    print("\n[Directional Accuracy]")
    da = analysis["directional_accuracy"]
    print(f"  Overall Accuracy    : {da['overall_accuracy']:>8.4f}")
    print(f"  Long Accuracy       : {da['long_accuracy']:>8.4f}  ({da['long_count']} trades)")
    print(f"  Short Accuracy      : {da['short_accuracy']:>8.4f}  ({da['short_count']} trades)")

    print("\n[Signal Characteristics]")
    print(f"  Hit Ratio           : {analysis['hit_ratio']:>8.4f}")
    print(f"  Signal Stability    : {analysis['signal_stability']:>8.6f}")

    print("\n[Prediction Distribution]")
    pd_info = analysis["prediction_distribution"]
    print(f"  Mean                : {pd_info['mean']:>8.8f}")
    print(f"  Std Dev             : {pd_info['std']:>8.8f}")
    print(f"  Skewness            : {pd_info['skew']:>8.4f}")
    print(f"  Kurtosis            : {pd_info['kurtosis']:>8.4f}")
    print(f"  Pct Positive        : {pd_info['pct_positive']:>8.2f}%")
    print(f"  Pct Negative        : {pd_info['pct_negative']:>8.2f}%")

    print("\n[Signal Decay (lookahead IC by lag)]")
    decay_dict = analysis["signal_decay"]
    for lag, ic in sorted(decay_dict.items(), key=lambda x: int(x[0]))[:6]:
        print(f"  Lag {int(lag):>2d}             : {ic:>8.6f}")

    print("=" * 70 + "\n")


