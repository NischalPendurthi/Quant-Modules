"""
features.py - Feature Engineering & Selection
==============================================
All transformations are mathematically documented.
No look-ahead bias: all computations use only past data.

Mathematical Notation
---------------------
    f_t        : raw feature value at time t
    μ_t(w)     : rolling mean over window w
    σ_t(w)     : rolling standard deviation over window w
    z_t(w)     = (f_t - μ_t(w)) / σ_t(w)            [Rolling Z-Score]
    β_t(w)     = Cov(f,y)_t(w) / Var(f)_t(w)         [Rolling Beta]
    IC_t       = Spearman(f_t, y_t)                   [Information Coefficient]
    ICIR       = Mean(IC) / Std(IC)                   [IC Information Ratio]
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 – FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

def _rolling(s: pd.Series, window: int, func: str, min_periods: int = None) -> pd.Series:
    """Helper: apply rolling function without look-ahead."""
    mp = min_periods if min_periods is not None else max(1, window // 2)
    r  = s.rolling(window=window, min_periods=mp)
    return getattr(r, func)()


# ── Rolling Statistics ──────────────────────────────────────────────

def rolling_mean(s: pd.Series, window: int) -> pd.Series:
    """μ_t(w) = (1/w) Σ_{i=0}^{w-1} f_{t-i}"""
    return _rolling(s, window, "mean").rename(f"{s.name}_rmean_{window}")


def rolling_median(s: pd.Series, window: int) -> pd.Series:
    """Median of last w observations."""
    return _rolling(s, window, "median").rename(f"{s.name}_rmed_{window}")


def rolling_std(s: pd.Series, window: int) -> pd.Series:
    """σ_t(w) = sqrt[ (1/(w-1)) Σ (f_{t-i} - μ)² ]"""
    return _rolling(s, window, "std").rename(f"{s.name}_rstd_{window}")


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """z_t(w) = (f_t - μ_t(w)) / σ_t(w)"""
    mu  = _rolling(s, window, "mean")
    sig = _rolling(s, window, "std")
    return ((s - mu) / sig.replace(0, np.nan)).rename(f"{s.name}_rzsc_{window}")


# ── Momentum / Lag Features ─────────────────────────────────────────

def lag_feature(s: pd.Series, lag: int) -> pd.Series:
    """f_{t - lag}  (simple lookback)"""
    return s.shift(lag).rename(f"{s.name}_lag{lag}")


def momentum(s: pd.Series, lag: int) -> pd.Series:
    """mom_t(lag) = f_t - f_{t-lag}  (raw change)"""
    return (s - s.shift(lag)).rename(f"{s.name}_mom{lag}")


def pct_change(s: pd.Series, lag: int) -> pd.Series:
    """pct_t(lag) = (f_t / f_{t-lag}) - 1"""
    return s.pct_change(lag).rename(f"{s.name}_pct{lag}")


# ── Cross-Sectional Features ─────────────────────────────────────────

def cross_sectional_rank(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    At each timestamp, rank each feature value across the feature universe.
    rank_t(f_i) = rank(f_i among f_1..f_n at time t) / n
    Returns a DataFrame with suffix '_csrank'.
    """
    ranked = df_features.rank(axis=1, pct=True)
    ranked.columns = [f"{c}_csrank" for c in df_features.columns]
    return ranked


def relative_spread(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    spread_t(f_i) = f_i - mean_t(f_1..f_n)
    Measures how much a feature deviates from the cross-sectional mean.
    """
    cs_mean   = df_features.mean(axis=1)
    spread    = df_features.subtract(cs_mean, axis=0)
    spread.columns = [f"{c}_spread" for c in df_features.columns]
    return spread


# ── Statistical Features ─────────────────────────────────────────────

def rolling_correlation(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    """
    ρ_t(w) = Cov(s1,s2)_t(w) / [σ(s1)_t(w) · σ(s2)_t(w)]
    """
    return s1.rolling(window).corr(s2).rename(f"corr_{s1.name}_{s2.name}_{window}")


def rolling_beta(s: pd.Series, target: pd.Series, window: int) -> pd.Series:
    """
    β_t(w) = Cov(f, y)_t(w) / Var(f)_t(w)
    Measures linear sensitivity of the target to a feature.
    """
    cov = s.rolling(window).cov(target)
    var = s.rolling(window).var()
    return (cov / var.replace(0, np.nan)).rename(f"{s.name}_beta_{window}")


# ── Regime Features ──────────────────────────────────────────────────

def volatility_regime(s: pd.Series, window: int = 24, n_regimes: int = 3) -> pd.Series:
    """
    Classify volatility into n_regimes (0=low, n-1=high).
    Uses rolling standard deviation quantiles estimated on expanding window
    (no look-ahead).
    vol_regime_t = qcut(σ_t(w), n_regimes)
    """
    vol  = _rolling(s, window, "std")
    # Use expanding quantiles to avoid look-ahead
    q_lo = vol.expanding().quantile(1 / n_regimes)
    q_hi = vol.expanding().quantile((n_regimes - 1) / n_regimes)
    regime = pd.Series(1, index=s.index, name=f"{s.name}_volreg")
    regime[vol <= q_lo] = 0
    regime[vol >= q_hi] = 2
    return regime


def trend_regime(s: pd.Series, fast: int = 6, slow: int = 24) -> pd.Series:
    """
    Binary trend regime:
        1 if rolling_mean(fast) > rolling_mean(slow)   → uptrend
        0 otherwise                                     → downtrend / flat
    """
    fast_ma = _rolling(s, fast,  "mean")
    slow_ma = _rolling(s, slow,  "mean")
    return (fast_ma > slow_ma).astype(int).rename(f"{s.name}_trendreg")


# ── Master Engineering Function ───────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    raw_feature_cols: Optional[List[str]] = None,
    windows: List[int] = (6, 12, 24),
    lags:    List[int] = (1, 3, 6, 12, 24),
    add_regime: bool = True,
    add_cross_sectional: bool = True,
) -> pd.DataFrame:
    """
    Apply all interpretable transformations to raw features.

    Parameters
    ----------
    df                  : DataFrame with columns y, f1..f49 (timestamp index)
    raw_feature_cols    : list of raw feature names; defaults to all f* columns
    windows             : rolling windows to apply
    lags                : lag periods for momentum
    add_regime          : whether to add regime features
    add_cross_sectional : whether to add cross-sectional features

    Returns
    -------
    df_out : original df augmented with engineered features (no look-ahead)
    """
    if raw_feature_cols is None:
        raw_feature_cols = [c for c in df.columns if c.startswith("f")]

    print(f"[features] Engineering features for {len(raw_feature_cols)} raw features ...")
    frames = [df.copy()]

    # ── Per-feature transformations ──────────────────────────────────
    for col in raw_feature_cols:
        s = df[col]

        for w in windows:
            frames.append(rolling_mean(s, w).to_frame())
            frames.append(rolling_std(s, w).to_frame())
            frames.append(rolling_zscore(s, w).to_frame())

        for lag in lags:
            frames.append(lag_feature(s, lag).to_frame())
            frames.append(momentum(s, lag).to_frame())

        if add_regime:
            frames.append(volatility_regime(s).to_frame())
            frames.append(trend_regime(s).to_frame())

    # ── Cross-sectional ───────────────────────────────────────────────
    if add_cross_sectional:
        raw_df = df[raw_feature_cols]
        frames.append(cross_sectional_rank(raw_df))
        frames.append(relative_spread(raw_df))

    df_out = pd.concat(frames, axis=1)
    # Remove duplicate columns (raw features already in df)
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    print(f"[features] Total columns after engineering: {df_out.shape[1]}")
    return df_out


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 – FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════

def _ic_series(feature: pd.Series, target: pd.Series, method: str = "spearman") -> float:
    """Information Coefficient: Spearman correlation of predictions vs target."""
    valid = (~feature.isna()) & (~target.isna())
    if valid.sum() < 10:
        return np.nan
    if method == "spearman":
        rho, _ = stats.spearmanr(feature[valid], target[valid])
    else:
        rho, _ = stats.pearsonr(feature[valid], target[valid])
    return rho


def _rolling_ic(feature: pd.Series, target: pd.Series, window: int = 250) -> pd.Series:
    """Rolling IC over a time window."""
    ic_vals = []
    idx     = []
    for i in range(window, len(feature) + 1):
        f_w = feature.iloc[i - window: i]
        y_w = target.iloc[i - window: i]
        ic  = _ic_series(f_w, y_w)
        ic_vals.append(ic)
        idx.append(feature.index[i - 1])
    return pd.Series(ic_vals, index=idx, name=feature.name)


def _mutual_info_approx(feature: pd.Series, target: pd.Series, n_bins: int = 20) -> float:
    """
    Approximate Mutual Information via histogram binning.
    MI(X; Y) = Σ p(x,y) log[ p(x,y) / (p(x)p(y)) ]
    """
    valid = (~feature.isna()) & (~target.isna())
    if valid.sum() < 20:
        return 0.0
    x = pd.qcut(feature[valid], q=n_bins, labels=False, duplicates="drop")
    y = pd.qcut(target[valid],  q=n_bins, labels=False, duplicates="drop")
    joint  = pd.crosstab(x, y, normalize=True) + 1e-12
    px     = joint.sum(axis=1)
    py     = joint.sum(axis=0)
    mi     = (joint * np.log(joint.div(px, axis=0).div(py, axis=1))).sum().sum()
    return max(float(mi), 0.0)


def _tstat(feature: pd.Series, target: pd.Series) -> float:
    """t-statistic from OLS regression of target on feature."""
    valid = (~feature.isna()) & (~target.isna())
    if valid.sum() < 10:
        return 0.0
    x = feature[valid].values
    y = target[valid].values
    x = np.column_stack([np.ones_like(x), x])
    try:
        beta, res, _, _ = np.linalg.lstsq(x, y, rcond=None)
        df_ = len(y) - 2
        if df_ < 1:
            return 0.0
        mse  = res[0] / df_ if len(res) > 0 else np.var(y - x @ beta)
        se   = np.sqrt(mse * np.linalg.inv(x.T @ x)[1, 1])
        return float(beta[1] / se) if se > 0 else 0.0
    except Exception:
        return 0.0


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 20,
    rolling_ic_window: int = 500,
    output_path: str = "selected_features.csv",
) -> Tuple[List[str], pd.DataFrame]:
    """
    Rank all features by multiple explainable criteria and return the top k.

    Criteria
    --------
    1. Pearson IC        : linear correlation with target
    2. Spearman IC       : rank correlation (non-linear, outlier-robust)
    3. Mutual Information: non-parametric dependency measure
    4. t-statistic       : statistical significance in OLS
    5. Rolling ICIR      : IC stability (IC mean / IC std over rolling window)

    Returns
    -------
    selected_cols : list of top-k feature names
    ranking_df    : full ranking table (saved to CSV)
    """
    feature_cols = X_train.columns.tolist()
    print(f"[features] Scoring {len(feature_cols)} features ...")

    records = []
    for col in feature_cols:
        f = X_train[col]
        pearson_ic  = _ic_series(f, y_train, method="pearson")
        spearman_ic = _ic_series(f, y_train, method="spearman")
        mi          = _mutual_info_approx(f, y_train)
        tstat       = _tstat(f, y_train)

        # Rolling IC → ICIR
        ric = _rolling_ic(f, y_train, window=rolling_ic_window)
        ric_mean = float(ric.mean()) if len(ric) > 0 else 0.0
        ric_std  = float(ric.std())  if len(ric) > 0 else 1.0
        icir     = ric_mean / ric_std if ric_std > 0 else 0.0

        records.append({
            "feature":     col,
            "pearson_ic":  round(pearson_ic  or 0.0, 6),
            "spearman_ic": round(spearman_ic or 0.0, 6),
            "mean_ic":     round(ric_mean,            6),
            "ic_std":      round(ric_std,             6),
            "icir":        round(icir,                6),
            "mutual_info": round(mi,                  6),
            "t_stat":      round(tstat,               6),
        })

    df_rank = pd.DataFrame(records)

    # ── Composite score ───────────────────────────────────────────────
    # Normalize each metric to [0,1] and combine
    for col in ["pearson_ic", "spearman_ic", "mean_ic", "icir", "mutual_info"]:
        abs_vals = df_rank[col].abs()
        mn, mx   = abs_vals.min(), abs_vals.max()
        df_rank[f"{col}_norm"] = (abs_vals - mn) / (mx - mn + 1e-12)

    norm_cols = [c for c in df_rank.columns if c.endswith("_norm")]
    df_rank["composite_score"] = df_rank[norm_cols].mean(axis=1)
    df_rank = df_rank.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df_rank["rank"] = df_rank.index + 1

    selected_cols = df_rank["feature"].iloc[:top_k].tolist()

    # ── Save ──────────────────────────────────────────────────────────
    out_cols = ["rank", "feature", "mean_ic", "ic_std", "icir",
                "pearson_ic", "spearman_ic", "mutual_info", "t_stat", "composite_score"]
    df_rank[out_cols].to_csv(output_path, index=False)
    print(f"[features] Feature ranking saved → {output_path}")
    print(f"[features] Top {top_k} selected features:\n  {selected_cols[:10]} ...")

    return selected_cols, df_rank[out_cols]