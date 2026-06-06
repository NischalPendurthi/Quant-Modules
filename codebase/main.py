"""
main_fixed.py - Fixed End-to-End Pipeline (No Leakage)
=======================================================
Key fixes:
  1. Signal sign frozen from training (NO auto-inversion in backtest)
  2. Target alignment validation
  3. Feature selection audit
  4. Asymmetric long/short analysis
  5. Proper metric reporting
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.stats import spearmanr

from pipeline_timing import PipelineTimer
from data import load_data, describe_data, create_target_variable
from features import engineer_features, fast_ic_ranking, compute_feature_ic
from models import compare_models, save_model
from signal_report import analyze_signal, print_signal_report, monthly_ic
from backtest import LongOnlyBacktest, LongShortBacktest, validate_output
from metrics import compute_all_metrics, print_metrics_report
import os


class Config:
    DATA_PATH = str(Path(__file__).parent.parent / "moccm_intraday_blackbox.csv")
    JUDGING_DATA_PATH = None
    IMPUTATION_METHOD = "ffill"
    MISSING_THRESHOLD = 0.50
    TEST_YEARS = 1

    ROLLING_WINDOWS = (6, 12, 24)
    MOMENTUM_LAGS = (1, 3, 6, 12, 24)
    ADD_REGIME = True
    ADD_CROSS_SECTIONAL = True

    TOP_K_FEATURES = 10  # Reduced from 20 to prevent overfitting
    ROLLING_IC_WINDOW = 500

    RIDGE_ALPHA = 1.0
    LASSO_ALPHA = 0.001
    ROLLING_WINDOW = 500
    
    HARDCODED_FEATURES = True
    HARDCODED_FEATURE_LIST = [
        "f1_mom1",
        "f1_rzsc_3",
        "f1_rzsc_6",
        "f1_rzsc_12",
        "f1_mom3",
        "f1_rzsc_24",
        "f1_mom6",
        "f1_mom12",
        "f1_mom24",
    ]

    # Conservative thresholds to reduce turnover
    LO_ENTRY_PERCENTILE = 0.95
    LO_EXIT_PERCENTILE = 0.60
    LS_LONG_PERCENTILE = 0.95
    LS_SHORT_PERCENTILE = 0.05
    LS_LONG_POSITION_FRACTION = 0.12
    LS_SHORT_POSITION_FRACTION = 0.04  # Smaller shorts (asymmetric)

    TRANSACTION_COST_BPS = 10.0
    INITIAL_CAPITAL_LO = 1_000_000.0
    INITIAL_CAPITAL_LS = 2_000_000.0

    OUTPUT_DIR = "output"
    SUBMISSIONS_DIR = "submissions"
    TEAM_NAME = "team"


def train_test_split(df, target_col="y", test_years=1):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    split_date = df.index.max() - pd.DateOffset(years=test_years)
    feature_cols = [c for c in df.columns if c != target_col]
    X, y = df[feature_cols], df[target_col]
    train_mask = df.index <= split_date
    test_mask = df.index > split_date
    print(f"[split] Train: {train_mask.sum()} rows")
    print(f"[split] Test: {test_mask.sum()} rows")
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def handle_missing(df, method="ffill", missing_threshold=0.50):
    df = df.copy()
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    if cols_to_drop:
        print(f"[missing] Dropping {len(cols_to_drop)} columns")
        df = df.drop(columns=cols_to_drop)
    if method == "ffill":
        df = df.ffill().bfill()
    print(f"[missing] After imputation: {df.isnull().sum().sum()} NaNs")
    return df


def validate_target_alignment(df, price_col='Close', target_col='y'):
    """Verify no look-ahead in target creation"""
    print("\n[Target Alignment Check]")
    print("First 10 rows:")
    print(df[[price_col, target_col]].head(10))
    
    expected_return = df[price_col].shift(-1) / df[price_col] - 1
    alignment = np.abs(df[target_col] - expected_return).mean()
    print(f"  Mean difference from 1-bar forward return: {alignment:.8f}")
    
    if alignment < 1e-6:
        print("  ✅ Target correctly aligned (1-bar forward)")
    elif alignment < 0.01:
        print("  ⚠️ Small misalignment detected")
    else:
        print("  ❌ Target misalignment! Check shift direction.")
    return alignment


def analyze_asymmetry(signal, y_true):
    """Check if signal works better on long or short side"""
    pos_mask = signal > 0
    neg_mask = signal < 0
    
    pos_ic = spearmanr(signal[pos_mask], y_true[pos_mask])[0] if pos_mask.sum() > 50 else 0
    neg_ic = spearmanr(-signal[neg_mask], y_true[neg_mask])[0] if neg_mask.sum() > 50 else 0
    
    print(f"\n[Asymmetry Analysis]")
    print(f"  Long side (signal>0) IC: {pos_ic:.4f} (n={pos_mask.sum()})")
    print(f"  Short side (signal<0) IC: {neg_ic:.4f} (n={neg_mask.sum()})")
    
    if pos_ic > neg_ic + 0.05:
        print(f"  → SIGNAL IS LONG-BIASED. Use Long-Only or asymmetric LS.")
        return "long_only"
    elif neg_ic > pos_ic + 0.05:
        print(f"  → SIGNAL IS SHORT-BIASED.")
        return "short_only"
    else:
        print(f"  → SIGNAL IS SYMMETRIC.")
        return "symmetric"


def build_full_signal(df_engineered, selected_features, best_model, signal_sign):
    """Generate predictions for full dataset with frozen sign"""
    from models import RollingOLSModel
    X_full = df_engineered[selected_features].fillna(0).values
    if isinstance(best_model, RollingOLSModel):
        if hasattr(best_model, "_last_beta") and best_model._last_beta is not None:
            pred_full = best_model.predict(X_full, best_model._last_beta)
        else:
            y_full = df_engineered["y"].fillna(0).values
            pred_full = best_model.fit_predict(X_full, y_full, selected_features)
    else:
        pred_full = best_model.predict(X_full)
    
    # Apply frozen sign
    return pd.Series(pred_full * signal_sign, index=df_engineered.index, name="signal")


def _get_prices(df_engineered):
    if "Close" in df_engineered.columns:
        return df_engineered["Close"].values.astype(float)
    raise ValueError("Close column not found")


def main(config: Config = None):
    if config is None:
        config = Config()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)

    timer = PipelineTimer()
    timer.start_pipeline()

    print("\n" + "=" * 80)
    print("QUANTITATIVE SIGNAL DISCOVERY (FIXED - NO LEAKAGE)")
    print("=" * 80 + "\n")

    # ── Phase 1: Data ────────────────────────────────────────────────
    print("\n[PHASE 1] DATA LOADING")
    print("─" * 80)

    df = load_data(config.DATA_PATH, timestamp_col="Timestamp", 
                   ticker_col="Ticker", price_col="Close", volume_col="Volume")
    df = handle_missing(df, config.IMPUTATION_METHOD, config.MISSING_THRESHOLD)

    if "y" not in df.columns:
        df = create_target_variable(df, price_col="Close", horizon=1,
                                    target_type="return", group_by_ticker=False)

    validate_target_alignment(df)

    # ── Phase 2: Feature Engineering ────────────────────────────────
    print("\n[PHASE 2] FEATURE ENGINEERING")
    print("─" * 80)

    df_engineered = engineer_features(df, windows=config.ROLLING_WINDOWS,
                                      lags=config.MOMENTUM_LAGS,
                                      add_regime=config.ADD_REGIME,
                                      add_cross_sectional=config.ADD_CROSS_SECTIONAL)
    print(f"[features] Engineered {df_engineered.shape[1] - 1} features")

    # ── Phase 3: Train/Test Split ────────────────────────────────────
    print("\n[PHASE 3] TRAIN/TEST SPLIT")
    print("─" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        df_engineered, target_col="y", test_years=config.TEST_YEARS
    )

    # ── Phase 4: Feature Selection (TRAIN ONLY) ──────────────────────
    print("\n[PHASE 4] FEATURE SELECTION (TRAIN ONLY)")
    print("─" * 80)

    if config.HARDCODED_FEATURES:
        selected_features = config.HARDCODED_FEATURE_LIST
    else:
        selected_features, _ = fast_ic_ranking(X_train, y_train, top_k=config.TOP_K_FEATURES)

    print(f"[selection] Selected {len(selected_features)} features")
    X_train_sel = X_train[selected_features].fillna(0)
    X_test_sel = X_test[selected_features].fillna(0)

    # ── Phase 5: Model Training (TRAIN ONLY) ─────────────────────────
    print("\n[PHASE 5] MODEL TRAINING")
    print("─" * 80)

    best_model, y_pred_train, y_pred_test, _ = compare_models(
        X_train_sel.values, X_test_sel.values,
        y_train.values, y_test.values,
        selected_features,
        ridge_alpha=config.RIDGE_ALPHA,
        lasso_alpha=config.LASSO_ALPHA,
        rolling_window=config.ROLLING_WINDOW,
    )

    # ── CRITICAL: Determine signal sign from TRAINING ONLY ───────────
    train_ic, _ = spearmanr(y_pred_train, y_train.values)
    SIGNAL_SIGN = 1 if train_ic >= 0 else -1
    print(f"\n[CRITICAL] Training IC = {train_ic:.4f}")
    print(f"[CRITICAL] Signal direction frozen to: {'NORMAL' if SIGNAL_SIGN == 1 else 'INVERTED'}")
    print(f"[CRITICAL] NO further inversion will be applied in backtest")

    # Apply sign to predictions
    y_pred_test = y_pred_test * SIGNAL_SIGN
    y_pred_train = y_pred_train * SIGNAL_SIGN

    # ── Phase 6: Asymmetry Analysis ──────────────────────────────────
    print("\n[PHASE 6] SIGNAL ASYMMETRY ANALYSIS")
    print("─" * 80)

    bias = analyze_asymmetry(y_pred_test, y_test.values)

    # ── Phase 7: Signal Quality ──────────────────────────────────────
    print("\n[PHASE 7] SIGNAL QUALITY")
    print("─" * 80)

    y_test_series = pd.Series(y_test.values, index=y_test.index, name="y_true")
    y_pred_series = pd.Series(y_pred_test, index=y_test.index, name="y_pred")

    signal_analysis = analyze_signal(y_test_series, y_pred_series)
    print_signal_report(signal_analysis)

    # ── Phase 8: Backtesting (with FROZEN SIGN) ──────────────────────
    print("\n[PHASE 8] BACKTESTING (SIGN FROZEN)")
    print("─" * 80)

    ts_test = y_test.index
    prices_test = _get_prices(df_engineered[df_engineered.index.isin(ts_test)])

    # Long-Only
    bt_lo = LongOnlyBacktest(
        initial_capital=config.INITIAL_CAPITAL_LO,
        entry_percentile=config.LO_ENTRY_PERCENTILE,
        exit_percentile=config.LO_EXIT_PERCENTILE,
    )
    results_lo_test = bt_lo.backtest(ts_test, prices_test, y_pred_test, force_sign=1)

    # Long-Short (with asymmetric sizing if biased)
    bt_ls = LongShortBacktest(
        initial_capital=config.INITIAL_CAPITAL_LS,
        long_percentile=config.LS_LONG_PERCENTILE,
        short_percentile=config.LS_SHORT_PERCENTILE,
        long_position_fraction=config.LS_LONG_POSITION_FRACTION,
        short_position_fraction=config.LS_SHORT_POSITION_FRACTION,
    )
    results_ls_test = bt_ls.backtest(ts_test, prices_test, y_pred_test, force_sign=1)

    # ── Phase 9: Submission Files ────────────────────────────────────
    print("\n[PHASE 9] GENERATING SUBMISSION FILES")
    print("─" * 80)

    # Build full signal with FROZEN sign
    full_signal = build_full_signal(df_engineered, selected_features, best_model, SIGNAL_SIGN)
    full_prices = _get_prices(df_engineered)
    full_timestamps = df_engineered.index

    # Use last 94,500 rows for judging dry-run
    GRADER_ROWS = 94_500
    judge_prices = full_prices[-GRADER_ROWS:]
    judge_timestamps = full_timestamps[-GRADER_ROWS:]
    judge_signal = full_signal.values[-GRADER_ROWS:]

    # Long-Only submission
    results_lo_full = bt_lo.backtest(judge_timestamps, judge_prices, judge_signal, force_sign=1)
    lo_path = f"{config.SUBMISSIONS_DIR}/{config.TEAM_NAME}_longonly_results.csv"
    results_lo_full.to_csv(lo_path, index=False)
    validate_output(results_lo_full, "LONG_ONLY", config.INITIAL_CAPITAL_LO)

    # Long-Short submission
    results_ls_full = bt_ls.backtest(judge_timestamps, judge_prices, judge_signal, force_sign=1)
    ls_path = f"{config.SUBMISSIONS_DIR}/{config.TEAM_NAME}_longshort_results.csv"
    results_ls_full.to_csv(ls_path, index=False)
    validate_output(results_ls_full, "LONG_SHORT", config.INITIAL_CAPITAL_LS)

    # ── Phase 10: Metrics ────────────────────────────────────────────
    print("\n[PHASE 10] PERFORMANCE METRICS")
    print("─" * 80)

    # Calculate proper metrics
    nav_lo = results_lo_test["Gross_NAV"].values
    nav_ls = results_ls_test["Gross_NAV"].values
    
    # Simple returns
    ret_lo = np.diff(nav_lo) / nav_lo[:-1]
    ret_ls = np.diff(nav_ls) / nav_ls[:-1]
    
    print(f"\nLong-Only Performance (Test Set):")
    print(f"  Total Return: {(nav_lo[-1] / nav_lo[0] - 1) * 100:.2f}%")
    print(f"  Sharpe (daily): {np.mean(ret_lo) / (np.std(ret_lo) + 1e-12):.4f}")
    print(f"  Max Drawdown: {((nav_lo - np.maximum.accumulate(nav_lo)) / np.maximum.accumulate(nav_lo)).min() * 100:.2f}%")
    
    print(f"\nLong-Short Performance (Test Set):")
    print(f"  Total Return: {(nav_ls[-1] / nav_ls[0] - 1) * 100:.2f}%")
    print(f"  Sharpe (daily): {np.mean(ret_ls) / (np.std(ret_ls) + 1e-12):.4f}")
    print(f"  Max Drawdown: {((nav_ls - np.maximum.accumulate(nav_ls)) / np.maximum.accumulate(nav_ls)).min() * 100:.2f}%")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nSubmission files: {lo_path}, {ls_path}")

    return {
        "best_model": best_model,
        "signal_sign": SIGNAL_SIGN,
        "bias": bias,
        "results_lo_full": results_lo_full,
        "results_ls_full": results_ls_full,
    }


if __name__ == "__main__":
    main()