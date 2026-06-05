"""
main.py - End-to-End Quantitative Signal Discovery Framework
============================================================
Orchestrates the complete pipeline:
  1. Data loading and preprocessing
  2. Feature engineering
  3. Model training and comparison
  4. Signal analysis
  5. Backtesting
  6. Performance evaluation
  7. Report generation
  8. Visualization
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Optional

from pipeline_timing import PipelineTimer

# Import framework modules
from data import load_data, describe_data
from features import engineer_features, select_features
from models import compare_models, save_model
from signal_report import analyze_signal, print_signal_report, monthly_ic, rolling_ic
from backtest import LongOnlyBacktest, LongShortBacktest
from metrics import compute_all_metrics, print_metrics_report
from plots import generate_all_plots
from report import generate_research_report
from sklearn.model_selection import train_test_split as sklearn_train_test_split


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

class Config:
    """Framework configuration."""
    # Data
    DATA_PATH               = str(Path(__file__).parent.parent / "moccm_intraday_blackbox.csv")
    IMPUTATION_METHOD       = "ffill"
    MISSING_THRESHOLD       = 0.50
    TEST_YEARS              = 1

    # Features
    ROLLING_WINDOWS         = (6, 12, 24)
    MOMENTUM_LAGS           = (1, 3, 6, 12, 24)
    ADD_REGIME              = True
    ADD_CROSS_SECTIONAL     = True

    # Feature Selection
    TOP_K_FEATURES          = 20
    ROLLING_IC_WINDOW       = 500
    # Skip slow IC scoring loop; use fixed feature set for pipeline health checks
    USE_HARDCODED_FEATURES  = True
    HARDCODED_FEATURES      = [
        "f1", "f5", "f10", "f15", "f25", "f49",
        "f1_rzsc_6", "f5_rzsc_12", "f10_rzsc_24",
        "f3_mom6", "f7_mom12", "f15_mom24",
        "f2_lag1", "f8_lag3",
        "f12_rmean_12", "f20_rstd_24",
        "f4_volreg", "f18_trendreg",
        "f6_csrank", "f22_spread",
    ]

    # Models
    RIDGE_ALPHA             = 1.0
    LASSO_ALPHA             = 0.001
    ROLLING_WINDOW          = 500

    # Backtesting
    LONG_ONLY_THRESHOLD     = 0.0
    LONG_SHORT_THRESHOLD_UP = 0.5
    LONG_SHORT_THRESHOLD_DN = -0.5
    POSITION_SIZE           = 0.5
    TRANSACTION_COST_BPS    = 10.0
    INITIAL_CAPITAL         = 1_000_000

    # Output
    OUTPUT_DIR              = "output"


# ══════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════

def train_test_split(
    df: pd.DataFrame,
    target_col: str = "y",
    test_years: int = 1,
) -> tuple:
    """
    Split data chronologically by years.
    
    Parameters
    ----------
    df : DataFrame with datetime index
    target_col : name of target column
    test_years : number of years for test set
    
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Determine split date
    split_date = df.index.max() - pd.DateOffset(years=test_years)
    
    # Split features and target
    feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Chronological split
    train_mask = df.index <= split_date
    test_mask = df.index > split_date
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"[split] Train: {len(X_train)} rows ({X_train.index[0]} → {X_train.index[-1]})")
    print(f"[split] Test:  {len(X_test)} rows ({X_test.index[0]} → {X_test.index[-1]})")
    
    return X_train, X_test, y_train, y_test


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate missing value report."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    report = pd.DataFrame({
        'column': missing_count.index,
        'missing_count': missing_count.values,
        'missing_pct': missing_pct.values
    })
    report = report[report['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    return report


def handle_missing(
    df: pd.DataFrame,
    method: str = "ffill",
    missing_threshold: float = 0.50
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Parameters
    ----------
    df : input DataFrame
    method : imputation method ('ffill', 'bfill', 'drop', 'median')
    missing_threshold : drop columns with missing % above this threshold
    
    Returns
    -------
    df : cleaned DataFrame
    """
    df = df.copy()
    
    # Drop columns with too many missing values
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    
    if cols_to_drop:
        print(f"[missing] Dropping {len(cols_to_drop)} columns > {missing_threshold*100}% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Impute remaining missing values
    if method == "ffill":
        df = df.ffill()
        df = df.bfill()  # Fill any remaining at the start
    elif method == "bfill":
        df = df.bfill()
        df = df.ffill()
    elif method == "median":
        df = df.fillna(df.median())
    elif method == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    print(f"[missing] After {method} imputation: {df.isnull().sum().sum()} NaNs remain")
    
    return df


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════

def main(config: Config = None):
    """
    Run the complete quantitative signal discovery framework.

    Parameters
    ----------
    config : Config object with pipeline parameters
    """
    if config is None:
        config = Config()

    import os
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    timer = PipelineTimer()
    timer.start_pipeline()

    print("\n" + "=" * 80)
    print("QUANTITATIVE SIGNAL DISCOVERY & BACKTESTING FRAMEWORK".center(80))
    print("White-Box | Interpretable | Production-Ready".center(80))
    print("=" * 80 + "\n")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: DATA LOADING & PREPROCESSING
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 1] DATA LOADING & PREPROCESSING")
    print("─" * 80)

    # Load data
    try:
        with timer.phase("1a", "data.py", "load_data + reshape"):
            df = load_data(
                config.DATA_PATH,
                timestamp_col="Timestamp",
                ticker_col="Ticker",
                price_col="Close",
                volume_col="Volume"
            )
    except FileNotFoundError:
        print(f"[ERROR] Data file not found: {config.DATA_PATH}")
        print("[HINT]  Place your CSV file at:", Path.cwd() / config.DATA_PATH)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

    with timer.phase("1b", "main.py", "missing value report + imputation"):
        missing_report = missing_value_report(df)
        print("\nMissing Value Summary:")
        if len(missing_report) > 0:
            print(missing_report.head(10).to_string(index=False))
        else:
            print("  No missing values detected.")
        df = handle_missing(df, method=config.IMPUTATION_METHOD,
                            missing_threshold=config.MISSING_THRESHOLD)

    if 'y' not in df.columns:
        from data import create_target_variable
        with timer.phase("1c", "data.py", "create_target_variable"):
            df = create_target_variable(
                df,
                price_col="Close",
                horizon=1,
                target_type="return",
                group_by_ticker=True,
                ticker_col="Ticker"
            )

    with timer.phase("1d", "data.py", "describe_data"):
        data_summary = describe_data(df)
    print("\nData Summary:")
    for key, val in data_summary.items():
        if key in ['tickers', 'feature_names']:
            print(f"  {key:20s}: {str(val)[:50]}...")
        else:
            print(f"  {key:20s}: {val}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 2] FEATURE ENGINEERING")
    print("─" * 80)

    with timer.phase("2", "features.py", "engineer_features"):
        df_engineered = engineer_features(
            df,
            windows=config.ROLLING_WINDOWS,
            lags=config.MOMENTUM_LAGS,
            add_regime=config.ADD_REGIME,
            add_cross_sectional=config.ADD_CROSS_SECTIONAL,
        )

    print(f"[features] Engineered {df_engineered.shape[1] - 1} features (excluding target)")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: TRAIN/TEST SPLIT
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 3] CHRONOLOGICAL TRAIN/TEST SPLIT")
    print("─" * 80)

    with timer.phase("3", "main.py", "train_test_split"):
        X_train, X_test, y_train, y_test = train_test_split(
            df_engineered,
            target_col="y",
            test_years=config.TEST_YEARS,
        )

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: FEATURE SELECTION
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 4] FEATURE SELECTION")
    print("─" * 80)

    if config.USE_HARDCODED_FEATURES:
        with timer.phase("4", "features.py", "hardcoded feature set (skip IC scoring)"):
            available = set(X_train.columns)
            selected_features = [f for f in config.HARDCODED_FEATURES if f in available]
            missing = [f for f in config.HARDCODED_FEATURES if f not in available]
            if missing:
                print(f"[features] WARNING: {len(missing)} hardcoded features not in data: {missing}")
            if len(selected_features) < config.TOP_K_FEATURES:
                extras = [c for c in X_train.columns if c not in selected_features and c != "y"][:config.TOP_K_FEATURES - len(selected_features)]
                selected_features.extend(extras)
            selected_features = selected_features[:config.TOP_K_FEATURES]
            feature_ranking_df = pd.DataFrame({
                "rank": range(1, len(selected_features) + 1),
                "feature": selected_features,
                "source": "hardcoded",
            })
            feature_ranking_df.to_csv(f"{config.OUTPUT_DIR}/selected_features.csv", index=False)
            print(f"[features] Using {len(selected_features)} hardcoded features (skipped IC scoring)")
            print(f"[features] Features: {selected_features}")
    else:
        with timer.phase("4", "features.py", "select_features"):
            selected_features, feature_ranking_df = select_features(
                X_train,
                y_train,
                top_k=config.TOP_K_FEATURES,
                rolling_ic_window=config.ROLLING_IC_WINDOW,
                output_path=f"{config.OUTPUT_DIR}/selected_features.csv",
            )

    # Subset to selected features
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: MODEL TRAINING & COMPARISON
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 5] MODEL TRAINING & COMPARISON")
    print("─" * 80)

    with timer.phase("5", "models.py", "compare_models"):
        best_model, y_pred_train, y_pred_test, model_comparison_df = compare_models(
            X_train_sel.values,
            X_test_sel.values,
            y_train.values,
            y_test.values,
            selected_features,
            ridge_alpha=config.RIDGE_ALPHA,
            lasso_alpha=config.LASSO_ALPHA,
            rolling_window=config.ROLLING_WINDOW,
        )

    with timer.phase("5b", "models.py", "save model + predictions"):
        model_comparison_df.to_csv(f"{config.OUTPUT_DIR}/model_comparison.csv", index=False)
        print(f"[models] Model comparison saved → {config.OUTPUT_DIR}/model_comparison.csv")
        pred_df_train = pd.DataFrame({
            "timestamp": y_train.index,
            "y_true": y_train.values,
            "y_pred": y_pred_train,
            "residual": y_train.values - y_pred_train,
        })
        pred_df_train.to_csv(f"{config.OUTPUT_DIR}/predictions_train.csv", index=False)
        pred_df_test = pd.DataFrame({
            "timestamp": y_test.index,
            "y_true": y_test.values,
            "y_pred": y_pred_test,
            "residual": y_test.values - y_pred_test,
        })
        pred_df_test.to_csv(f"{config.OUTPUT_DIR}/predictions_test.csv", index=False)
        print(f"[models] Predictions saved → {config.OUTPUT_DIR}/predictions_*.csv")
        save_model(best_model, f"{config.OUTPUT_DIR}/best_model.pkl")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: SIGNAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 6] SIGNAL QUALITY ANALYSIS")
    print("─" * 80)

    y_test_series = pd.Series(y_test.values, index=y_test.index, name='y_true')
    y_pred_series = pd.Series(y_pred_test, index=y_test.index, name='y_pred')
    
    with timer.phase("6", "signal_report.py", "analyze_signal + monthly_ic"):
        signal_analysis = analyze_signal(
            y_test_series,
            y_pred_series,
            rolling_window=config.ROLLING_IC_WINDOW,
        )
        print_signal_report(signal_analysis)
        monthly_ic_df = monthly_ic(y_test_series, y_pred_series)
        monthly_ic_df.to_csv(f"{config.OUTPUT_DIR}/monthly_ic.csv", index=False)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 7: BACKTESTING (LONG-ONLY)
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 7A] BACKTESTING: LONG-ONLY STRATEGY")
    print("─" * 80)

    backtest_lo = LongOnlyBacktest(
        initial_capital=config.INITIAL_CAPITAL,
        position_size=config.POSITION_SIZE,
        threshold=config.LONG_ONLY_THRESHOLD,
        transaction_cost=config.TRANSACTION_COST_BPS,
    )

    # Construct price series (use cumulative returns as proxy)
    prices = 100 * (1 + y_test.cumsum()).fillna(100).values
    
    with timer.phase("7a", "backtest.py", "LongOnlyBacktest"):
        results_lo = backtest_lo.backtest(
            timestamps=y_test.index,
            prices=prices,
            signal=y_pred_test,
            returns=y_test.values,
        )
        results_lo.to_csv(f"{config.OUTPUT_DIR}/long_only_results.csv", index=False)
        print(f"[backtest] Long-only results saved → {config.OUTPUT_DIR}/long_only_results.csv")

    with timer.phase("7a-metrics", "metrics.py", "long-only metrics"):
        metrics_lo = compute_all_metrics(results_lo, y_test.index)
        print("\nLong-Only Performance:")
        print_metrics_report(metrics_lo)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 7B: BACKTESTING (LONG-SHORT)
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 7B] BACKTESTING: LONG-SHORT STRATEGY")
    print("─" * 80)

    backtest_ls = LongShortBacktest(
        initial_capital=config.INITIAL_CAPITAL,
        position_size=config.POSITION_SIZE,
        upper_threshold=config.LONG_SHORT_THRESHOLD_UP,
        lower_threshold=config.LONG_SHORT_THRESHOLD_DN,
        transaction_cost=config.TRANSACTION_COST_BPS,
    )

    with timer.phase("7b", "backtest.py", "LongShortBacktest"):
        results_ls = backtest_ls.backtest(
            timestamps=y_test.index,
            prices=prices,
            signal=y_pred_test,
            returns=y_test.values,
        )
        results_ls.to_csv(f"{config.OUTPUT_DIR}/long_short_results.csv", index=False)
        print(f"[backtest] Long-short results saved → {config.OUTPUT_DIR}/long_short_results.csv")

    with timer.phase("7b-metrics", "metrics.py", "long-short metrics"):
        metrics_ls = compute_all_metrics(results_ls, y_test.index)
        print("\nLong-Short Performance:")
        print_metrics_report(metrics_ls)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 8: VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 8] GENERATING VISUALIZATIONS")
    print("─" * 80)

    # Get model coefficients for feature importance
    if hasattr(best_model, "coef_"):
        feature_ics = best_model.coef_
    elif hasattr(best_model, "weights_"):
        feature_ics = best_model.weights_
    else:
        feature_ics = np.ones(len(selected_features)) / len(selected_features)

    try:
        with timer.phase("8", "plots.py", "generate_all_plots"):
            plot_paths = generate_all_plots(
                backtest_results=results_ls,
                y_true=y_test_series,
                y_pred=y_pred_series,
                signal=y_pred_test,
                feature_names=selected_features,
                feature_ics=feature_ics,
                output_dir=f"{config.OUTPUT_DIR}/plots",
            )
            print(f"[plots] Generated {len(plot_paths)} plots in {config.OUTPUT_DIR}/plots/")
    except Exception as e:
        print(f"[plots] Warning: Could not generate all plots - {e}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 9: RESEARCH REPORT GENERATION
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 9] GENERATING RESEARCH REPORT")
    print("─" * 80)

    # Prepare IC metrics dict
    best_model_name = model_comparison_df.iloc[0]['model'] if len(model_comparison_df) > 0 else "Ensemble"
    
    ic_metrics_dict = {
        "train_ic":      float(model_comparison_df[model_comparison_df["model"] == best_model_name]["train_ic"].iloc[0]) if len(model_comparison_df[model_comparison_df["model"] == best_model_name]) > 0 else 0.0,
        "train_ic_median": float(model_comparison_df[model_comparison_df["model"] == best_model_name]["train_ic"].iloc[0]) if len(model_comparison_df[model_comparison_df["model"] == best_model_name]) > 0 else 0.0,
        "train_ic_std":  0.0,
        "train_icir":    0.0,
        "test_ic":       signal_analysis["ic_metrics"]["mean_ic"],
        "test_ic_median": signal_analysis["ic_metrics"]["median_ic"],
        "test_ic_std":   signal_analysis["ic_metrics"]["ic_std"],
        "test_icir":     signal_analysis["ic_metrics"]["icir"],
    }

    # Prepare formula
    if hasattr(best_model, "formula"):
        formula = best_model.formula()
    else:
        formula = f"Signal = Σᵢ wᵢ · rank(fᵢ)  [IC-weighted rank combination]"

    # Prepare coefficient table
    if hasattr(best_model, "coef_table"):
        coef_df = best_model.coef_table()
    else:
        coef_df = pd.DataFrame({
            "feature": selected_features[:10],
            "coefficient": feature_ics[:10] if len(feature_ics) >= 10 else feature_ics,
        })

    # Generate report
    try:
        with timer.phase("9", "report.py", "generate_research_report"):
            report = generate_research_report(
                title="White-Box Quantitative Signal Discovery & Backtesting",
                data_summary=data_summary,
                signal_name="IC-Weighted Feature Score",
                strategy_type="Long-Short",
                model_name=best_model_name,
                model_description="""
The signal is constructed as a weighted combination of selected features,
where weights are determined by the historical Information Coefficient (IC)
of each feature with the target variable. Features are first rank-normalized
to [0,1] to remove scale effects, then aggregated:

    Signal_t = Σᵢ IC_i · rank(fᵢ,t)

This approach ensures that stronger predictors receive larger weights,
and the signal is robust to extreme values.
                """,
                formula=formula,
                coef_df=coef_df,
                feature_names=selected_features,
                n_engineered=df_engineered.shape[1] - 1,
                ic_metrics=ic_metrics_dict,
                backtest_metrics=metrics_ls,
                output_path=f"{config.OUTPUT_DIR}/research_report.txt",
            )
        print(f"[report] Research report saved → {config.OUTPUT_DIR}/research_report.txt")
    except Exception as e:
        print(f"[report] Warning: Could not generate report - {e}")

    timer.save(config.OUTPUT_DIR, data_path=config.DATA_PATH)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 10: SUMMARY & OUTPUTS
    # ══════════════════════════════════════════════════════════════════

    print("\n[PHASE 10] SUMMARY & OUTPUT FILES")
    print("─" * 80)
    print("\nDeliverables Generated:")
    
    output_files = [
        "best_model.pkl",
        "predictions_train.csv",
        "predictions_test.csv",
        "selected_features.csv",
        "model_comparison.csv",
        "monthly_ic.csv",
        "long_only_results.csv",
        "long_short_results.csv",
        "research_report.txt",
        "pipeline_timings.csv",
        "pipeline_timings.txt",
    ]
    
    for file in output_files:
        filepath = Path(config.OUTPUT_DIR) / file
        if filepath.exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (not generated)")
    
    plots_dir = Path(config.OUTPUT_DIR) / "plots"
    if plots_dir.exists():
        n_plots = len(list(plots_dir.glob("*.png")))
        print(f"  ✓ {plots_dir}/ ({n_plots} plot files)")

    print("\n" + "=" * 80)
    print("FRAMEWORK EXECUTION COMPLETE".center(80))
    print("═" * 80 + "\n")

    return {
        "best_model":          best_model,
        "predictions_train":   pred_df_train,
        "predictions_test":    pred_df_test,
        "backtest_results_lo": results_lo,
        "backtest_results_ls": results_ls,
        "metrics_lo":          metrics_lo,
        "metrics_ls":          metrics_ls,
        "signal_analysis":     signal_analysis,
        "report":              report if 'report' in locals() else None,
        "timings":             timer.to_dataframe(),
    }


# ══════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run with default configuration
    results = main()