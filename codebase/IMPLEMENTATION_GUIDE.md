# Implementation Guide: Complete White-Box Quantitative Framework

## ✅ Completed Deliverables

Your quantitative trading research framework is now **100% complete** with the following components:

### Core Modules (8 files)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| **data.py** | Data loading & preprocessing | `load_data()`, `handle_missing()`, `train_test_split()`, `missing_value_report()` |
| **features.py** | Feature engineering & selection | `engineer_features()`, `select_features()`, rolling statistics, momentum, regime features |
| **models.py** | 6 interpretable models | `OLSModel`, `RidgeModel`, `LassoModel`, `RollingOLSModel`, `FactorScoreModel`, `RankAlphaModel`, `compare_models()` |
| **signal.py** | Signal quality analysis | `analyze_signal()`, `pearson_ic()`, `spearman_ic()`, `rolling_ic()`, `directional_accuracy()` |
| **backtest.py** | Event-driven backtesting | `LongOnlyBacktest`, `LongShortBacktest`, realistic P&L tracking |
| **metrics.py** | Performance metrics | `sharpe_ratio()`, `sortino_ratio()`, `maximum_drawdown()`, `compute_all_metrics()` (21 metrics) |
| **plots.py** | Visualizations | 11 publication-quality plots (cumulative PnL, drawdown, rolling IC, etc.) |
| **report.py** | Research report generation | `ResearchReport`, `generate_research_report()` |
| **main.py** | End-to-end orchestration | `Config`, `main()` – runs entire pipeline in one command |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /home/harshi/Desktop/Quant-Modules/backtesting
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place a CSV file in the backtesting directory with this format:
```
Date,Time,y,f1,f2,...,f49
2018-01-01,09:30:00,-0.0005,1.2,3.4,...
2018-01-01,09:35:00,0.0003,1.5,3.2,...
```

### 3. Run the Framework
```python
from main import Config, main

config = Config()
config.DATA_PATH = "your_data.csv"
results = main(config)
```

This automatically:
- ✅ Loads and cleans data
- ✅ Engineers 100+ interpretable features
- ✅ Selects top-k features by IC ranking
- ✅ Trains 6 different models
- ✅ Analyzes signal quality (IC, hit ratio, decay)
- ✅ Runs long-only & long-short backtests
- ✅ Computes 21 performance metrics
- ✅ Generates 11 publication-quality plots
- ✅ Creates research report for jury presentation

---

## 📊 What Each File Does

### data.py
**Responsibility**: Data pipeline foundation

```python
from data import load_data, handle_missing, train_test_split

# Load CSV and align timestamps
df = load_data("data.csv")
# Output: Timestamp index, numeric columns, sorted chronologically

# Handle missing values (configurable: ffill, bfill, rolling_median, remove)
df = handle_missing(df, method="ffill", missing_threshold=0.50)

# Chronological train/test split (last year = test)
X_train, X_test, y_train, y_test = train_test_split(df, test_years=1)
```

**Mathematical Operations**:
- Timestamp construction: `pd.to_datetime(Date + Time)`
- Chronological validation: Ensure no duplicates, properly sorted
- Missing value analysis: Count, percentage, per-feature statistics

---

### features.py
**Responsibility**: Interpretable feature transformation pipeline

```python
from features import engineer_features, select_features

# Transform raw features → 100+ engineered features
df_eng = engineer_features(df, 
    windows=(6, 12, 24),      # Rolling window sizes
    lags=(1, 3, 6, 12, 24),   # Lag periods
    add_regime=True,           # Volatility & trend regimes
    add_cross_sectional=True   # Ranking, spread
)

# Select top-20 features by IC ranking
features, ranking_df = select_features(
    X_train, y_train, 
    top_k=20,
    output_path="selected_features.csv"
)
```

**Features Created**:
1. **Rolling Statistics** (3 windows × 3 types = 9 per raw feature)
   - `f1_rmean_6`, `f1_rmean_12`, `f1_rmean_24`
   - `f1_rstd_6`, `f1_rstd_12`, `f1_rstd_24`
   - `f1_rzsc_6`, `f1_rzsc_12`, `f1_rzsc_24`

2. **Momentum** (5 lags × 2 types = 10 per raw feature)
   - `f1_lag1`, `f1_lag3`, `f1_lag6`, `f1_lag12`, `f1_lag24`
   - `f1_mom1`, `f1_mom3`, `f1_mom6`, `f1_mom12`, `f1_mom24`

3. **Regime** (2 per raw feature)
   - `f1_volreg`: 0=low, 1=med, 2=high volatility
   - `f1_trendreg`: 1=uptrend (sma6 > sma24), 0=downtrend

4. **Cross-Sectional** (2 types × 49 features = 98 total)
   - `f1_csrank`: rank of f1 relative to other features [0,1]
   - `f1_spread`: f1 - mean(all features)

**Selection Criteria**:
- Pearson IC (linear correlation with target)
- Spearman IC (rank correlation, outlier-robust)
- Rolling ICIR (IC stability)
- Mutual Information
- t-statistic (OLS significance)
→ Composite score combines all criteria

---

### models.py
**Responsibility**: 6 interpretable white-box models

```python
from models import compare_models, OLSModel, RidgeModel, LassoModel

# Train all 6 models and compare
best_model, y_pred_train, y_pred_test, comparison_df = compare_models(
    X_train.values, X_test.values,
    y_train.values, y_test.values,
    selected_features,
    ridge_alpha=1.0,        # L2 regularization strength
    lasso_alpha=0.001,      # L1 regularization strength
    rolling_window=500      # For rolling OLS
)

# Best model selected by: Test Rank IC (most important metric)
```

**Model Details**:

| Model | Formula | Closed-Form | Interpretation |
|-------|---------|-------------|-----------------|
| **OLS** | ŷ = Xβ | β=(X'X)⁻¹X'y | Baseline; no regularization |
| **Ridge** | ŷ = Xβ_ridge | β=(X'X+λI)⁻¹X'y | Shrinks all coefficients equally (L2) |
| **Lasso** | ŷ = Xβ_lasso | Coordinate descent | Sparse; removes weak features (L1) |
| **Rolling OLS** | ŷ_t = x_t'β_{t-1} | Time-varying β | Adaptive to regime shifts |
| **Factor Score** | Signal = Σ IC_i·rank(f_i) | Weights = historical IC | Stronger features weighted more |
| **Rank Alpha** | Signal = Σ RankIC_i·pct_rank(f_i) | Rank-based IC | More robust to outliers |

**Output Metrics** (per model):
- Train/Test IC (Pearson & Spearman)
- Train/Test MSE, RMSE, MAE
- Train/Test R²
→ Best model selected by Test Rank IC

---

### signal.py
**Responsibility**: Comprehensive signal quality analysis

```python
from signal import analyze_signal, print_signal_report, rolling_ic, monthly_ic

# Analyze signal quality
analysis = analyze_signal(y_test, y_pred_test, rolling_window=252)

# Contents of analysis dict:
# {
#   "ic_metrics": {"mean_ic": 0.0847, "icir": 1.24, ...},
#   "directional_accuracy": {"overall": 0.532, "long_acc": 0.548, "short_acc": 0.516},
#   "hit_ratio": 0.532,
#   "prediction_distribution": {"mean": 0.0001, "std": 0.0523, "skew": 0.12, ...},
#   "signal_stability": 0.82,
#   "signal_decay": {0: 0.0847, 1: 0.0621, 2: 0.0384, ...}  # lookahead IC
# }

print_signal_report(analysis)

# Decompose IC by month (to spot seasonal patterns)
monthly_ic_df = monthly_ic(y_test, y_pred_test)

# Rolling IC (to detect regime shifts)
rolling_ic_series = rolling_ic(y_test, y_pred_test, window=252)
```

**Key Metrics**:
- **IC Metrics**: Mean IC, ICIR (stability ratio), Rank IC
- **Directional Accuracy**: % correct sign (overall, long-only, short-only)
- **Hit Ratio**: Same as directional accuracy
- **Signal Stability**: Autocorrelation of signal (robust = high)
- **Signal Decay**: IC by lookahead periods (should decay to zero)

---

### backtest.py
**Responsibility**: Realistic event-driven portfolio simulation

```python
from backtest import LongOnlyBacktest, LongShortBacktest

# Long-Only: signal > 0 → long, else flat
backtest_lo = LongOnlyBacktest(
    initial_capital=1_000_000,
    position_size=1.0,
    threshold=0.0,
    transaction_cost=10.0  # bps
)
results_lo = backtest_lo.backtest(timestamps, prices, signal, returns)

# Long-Short: signal > 0.5 → long, signal < -0.5 → short, else flat
backtest_ls = LongShortBacktest(
    initial_capital=2_000_000,
    position_size=0.5,
    upper_threshold=0.5,
    lower_threshold=-0.5,
    transaction_cost=10.0
)
results_ls = backtest_ls.backtest(timestamps, prices, signal, returns)

# Output: DataFrame with columns:
# [timestamp, price, signal, position, shares, cash, nav, realized_pnl,
#  unrealized_pnl, turnover, transaction_cost]
```

**Features**:
- ✅ No look-ahead bias (signal from previous bar)
- ✅ Realistic position management (buy/sell with transaction costs)
- ✅ Mark-to-market (daily P&L calculation)
- ✅ Turnover tracking (for cost analysis)
- ✅ Exposure constraints (max leverage, cash balance)

---

### metrics.py
**Responsibility**: 21 comprehensive performance metrics

```python
from metrics import compute_all_metrics, print_metrics_report

metrics = compute_all_metrics(backtest_results, timestamps)
print_metrics_report(metrics)

# Computes 21 metrics across 6 categories:
```

**Return Metrics**:
- Annual Return: (nav_end/nav_start)^(1/years) - 1
- Total Return: (nav_end/nav_start) - 1
- CAGR: Same as annual return

**Risk Metrics**:
- Volatility: σ_daily × √252
- Sharpe: (mean_ret - rf) / σ × √252
- Sortino: (mean_ret - rf) / σ_downside × √252
- Calmar: annual_ret / |max_drawdown|

**Drawdown**:
- Max DD: min(nav_t - running_max_nav) / running_max_nav
- Avg DD: mean of all negative drawdowns

**Win/Loss**:
- Profit Factor: Σ(wins) / |Σ(losses)|
- Hit Ratio: % profitable trades
- Avg Win / Avg Loss
- Win/Loss Ratio: avg_win / |avg_loss|

**Tail Risk**:
- VaR (95%): worst 5% loss
- CVaR (95%): mean of worst 5% losses
- Skewness: distribution asymmetry
- Kurtosis: tail fatness
- Tail Ratio: upside tail / downside tail

**Trading**:
- Avg Turnover: average daily position changes
- Avg Holding Period: 1 / avg_turnover

---

### plots.py
**Responsibility**: 11 publication-quality visualizations

```python
from plots import generate_all_plots

plot_paths = generate_all_plots(
    backtest_results=results_ls,
    y_true=y_test,
    y_pred=y_pred_series,
    signal=y_pred_test,
    feature_names=selected_features,
    feature_ics=model.coef_,
    output_dir="plots"
)
```

**Plots Generated**:
1. **01_cumulative_pnl.png**: NAV over time (line + fill)
2. **02_daily_pnl.png**: Realized P&L by day (bar chart, green/red)
3. **03_drawdown.png**: Underwater plot (running drawdown)
4. **04_rolling_sharpe.png**: Rolling Sharpe ratio (252-bar window)
5. **05_rolling_ic.png**: Rolling IC (predictive power stability)
6. **06_signal_distribution.png**: Histogram + box plot of signal
7. **07_actual_vs_predicted.png**: Scatter (actual vs pred) with diagonal
8. **08_feature_importance.png**: Bar chart (top-15 features by IC)
9. **09_monthly_returns_heatmap.png**: Year × Month P&L matrix
10. **10_exposure.png**: Position size over time
11. **11_turnover.png**: Daily turnover + cumulative transaction costs

All plots have:
- Proper date formatting (month-year on x-axis)
- Grid lines & legends
- Title, axis labels
- Publication-quality DPI (300)

---

### report.py
**Responsibility**: Automatic research report generation for jury

```python
from report import generate_research_report

report = generate_research_report(
    title="White-Box Quantitative Signal Discovery",
    data_summary={"start_date": "...", "n_rows": 100000, ...},
    signal_name="IC-Weighted Feature Score",
    strategy_type="long-short",
    model_name="Ridge Regression",
    model_description="Mathematical formulation...",
    formula="Signal = 0.45·mom6 + 0.32·lag12 - ...",
    coef_df=pd.DataFrame({"feature": [...], "coefficient": [...]}),
    feature_names=selected_features,
    n_engineered=150,
    ic_metrics={"mean_ic": 0.0847, ...},
    backtest_metrics={"annual_return": 0.125, ...},
    output_path="research_report.txt"
)
```

**Report Sections** (11 sections, ~3000 lines):
1. Executive Summary (key metrics, signal source)
2. Data Overview (dates, frequency, statistics)
3. Feature Engineering (transformations, selection)
4. Signal Model Specification (math formulation, formula)
5. Coefficients & Significance (table of weights)
6. IC Analysis (in/out-of-sample, interpretation)
7. Backtesting Results (all 21 metrics)
8. Portfolio Analysis (returns, risk, drawdown)
9. Risk Analysis (tail risk, VaR, CVaR)
10. Trading Formula (closed-form representation)
11. Conclusions & Next Steps

**Jury-Ready**: Fully formatted, comprehensive, self-contained document

---

### main.py
**Responsibility**: End-to-end orchestration

```python
from main import Config, main

# Default config
config = Config()
config.DATA_PATH = "data.csv"
config.TOP_K_FEATURES = 20

# Run entire pipeline
results = main(config)

# Returns dict with all outputs:
# {
#   "best_model": model_object,
#   "predictions_train": DataFrame,
#   "predictions_test": DataFrame,
#   "backtest_results_lo": DataFrame (long-only),
#   "backtest_results_ls": DataFrame (long-short),
#   "metrics_lo": dict,
#   "metrics_ls": dict,
#   "signal_analysis": dict,
#   "report": ResearchReport
# }
```

**Pipeline Steps** (10 phases):
1. Data loading & preprocessing
2. Feature engineering
3. Train/test split (chronological)
4. Feature selection (IC ranking)
5. Model training & comparison
6. Signal quality analysis
7. Long-only backtesting
8. Long-short backtesting
9. Visualization generation
10. Research report generation

**Output Directory Structure**:
```
output/
├── best_model.pkl
├── predictions_train.csv
├── predictions_test.csv
├── selected_features.csv
├── model_comparison.csv
├── monthly_ic.csv
├── long_only_results.csv
├── long_short_results.csv
├── research_report.txt
└── plots/
    ├── 01_cumulative_pnl.png
    ├── ...
    └── 11_turnover.png
```

---

## 🎯 Usage Examples

### Example 1: Run with Default Configuration
```python
from main import main
results = main()  # Uses all defaults
```

### Example 2: Custom Configuration
```python
from main import Config, main

config = Config()
config.DATA_PATH = "intraday_5min.csv"
config.TOP_K_FEATURES = 15
config.RIDGE_ALPHA = 0.5
config.ROLLING_WINDOWS = (4, 8, 12)
config.MOMENTUM_LAGS = (1, 2, 4, 8)
config.LONG_SHORT_THRESHOLD_UP = 0.3
config.LONG_SHORT_THRESHOLD_DN = -0.3

results = main(config)
```

### Example 3: Step-by-Step Manual Pipeline
```python
import pandas as pd
import numpy as np

# Step 1: Load & clean data
from data import load_data, handle_missing, train_test_split
df = load_data("data.csv")
df = handle_missing(df, method="ffill")
X_train, X_test, y_train, y_test = train_test_split(df, test_years=1)

# Step 2: Engineer & select features
from features import engineer_features, select_features
df_eng = engineer_features(df)
features, _ = select_features(X_train, y_train, top_k=20)

# Step 3: Train best model
from models import compare_models
best_model, y_pred_train, y_pred_test, _ = compare_models(
    X_train[features].values, X_test[features].values,
    y_train.values, y_test.values, features
)

# Step 4: Backtest
from backtest import LongShortBacktest
from metrics import compute_all_metrics

backtest = LongShortBacktest()
results = backtest.backtest(
    y_test.index, np.ones(len(y_test))*100,
    y_pred_test, y_test.values
)

metrics = compute_all_metrics(results, y_test.index)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {metrics['maximum_drawdown']:.2%}")

# Step 5: Visualize
from plots import generate_all_plots
plots = generate_all_plots(results, y_test, pd.Series(y_pred_test, index=y_test.index),
                            y_pred_test, features, best_model.coef_)

# Step 6: Report
from report import generate_research_report
report = generate_research_report(
    data_summary={...}, signal_name="My Signal", ...
)
```

---

## 🔧 Configuration Reference

```python
class Config:
    # Data
    DATA_PATH               = "data.csv"
    IMPUTATION_METHOD       = "ffill"  # ffill, bfill, rolling_median, remove
    MISSING_THRESHOLD       = 0.50     # Drop features >50% missing
    TEST_YEARS              = 1        # Last 1 year = test set

    # Features
    ROLLING_WINDOWS         = (6, 12, 24)    # Rolling window sizes
    MOMENTUM_LAGS           = (1, 3, 6, 12, 24)
    ADD_REGIME              = True
    ADD_CROSS_SECTIONAL     = True

    # Feature Selection
    TOP_K_FEATURES          = 20       # Select top-20
    ROLLING_IC_WINDOW       = 500      # For IC stability calc

    # Models
    RIDGE_ALPHA             = 1.0      # L2 regularization
    LASSO_ALPHA             = 0.001    # L1 regularization
    ROLLING_WINDOW          = 500      # Window for rolling OLS

    # Backtesting
    LONG_ONLY_THRESHOLD     = 0.0      # signal > 0 → long
    LONG_SHORT_THRESHOLD_UP = 0.5      # signal > 0.5 → long
    LONG_SHORT_THRESHOLD_DN = -0.5     # signal < -0.5 → short
    POSITION_SIZE           = 0.5      # Fraction of capital per trade
    TRANSACTION_COST_BPS    = 10.0     # 10 basis points
    INITIAL_CAPITAL         = 1_000_000

    # Output
    OUTPUT_DIR              = "output"
```

---

## 📋 Checklist for Jury Defense

- [ ] **Data**: Verified 10 years of clean 5-minute data
- [ ] **Features**: 100+ interpretable features, no look-ahead bias
- [ ] **Model**: Chose Ridge/OLS/Lasso based on test IC (not overfitting)
- [ ] **Signal**: Out-of-sample IC > 0.05 (or your benchmark)
- [ ] **Backtest**: Realistic costs (10 bps), proper position sizing
- [ ] **Metrics**: Sharpe > 0.5, Max DD < 10%, Hit Ratio > 50%
- [ ] **Plots**: 11 publication-quality visualizations
- [ ] **Report**: Jury-ready explanation of signal economics
- [ ] **Formula**: Can write signal as one-line formula
- [ ] **Robustness**: Tested on different regimes, time periods, parameters

---

## 🚨 Common Pitfalls to Avoid

1. **Look-ahead bias**: ❌ Use future data in signal  
   → ✅ Only use data available at time T

2. **Overfitting**: ❌ Cherry-pick features based on test set  
   → ✅ Use cross-validation; select features on train set only

3. **Data snooping**: ❌ Run 100 models, pick the best  
   → ✅ Pre-specify model, stick with it

4. **Unrealistic costs**: ❌ Ignore slippage, assume instant execution  
   → ✅ Use 10–20 bps transaction costs

5. **Black-box models**: ❌ Use neural networks or random forests  
   → ✅ Use interpretable models (linear, ridge, lasso)

6. **In-sample metrics only**: ❌ Report training R² and IC  
   → ✅ Always show test set metrics

7. **No risk limits**: ❌ Allow unlimited leverage  
   → ✅ Implement position limits, cash constraints

---

## 📞 Support

**Problem**: Framework runs but finds weak signal (low IC)
- **Solution**: 
  1. Check data quality (missing values, outliers)
  2. Try different feature windows (6,12,24 vs 4,8,12)
  3. Try different lags (1,3,6,12,24 vs 1,2,4,8)
  4. Increase TOP_K_FEATURES (20 → 30)
  5. Try Lasso for sparse selection

**Problem**: Memory error on large datasets
- **Solution**:
  1. Process in chunks
  2. Reduce ROLLING_IC_WINDOW (500 → 250)
  3. Drop unnecessary features before modeling

**Problem**: Backtest profits look suspicious (too good)
- **Solution**:
  1. Check for look-ahead bias (use signal from t-1, not t)
  2. Increase transaction costs (10 → 20 bps)
  3. Reduce position size (0.5 → 0.2)
  4. Add more constraints (max exposure, stop-loss)

---

## 🎓 Educational Value

This framework teaches:
- **Quantitative Trading**: Signal discovery, backtesting, risk management
- **Interpretability**: Building models that explain themselves (no black boxes)
- **Software Engineering**: Modular design, clean code, documentation
- **Statistical Analysis**: IC, rolling metrics, regime detection, tail risk
- **Python Skills**: NumPy, pandas, scipy, matplotlib, visualization

Perfect for:
- Hackathon submissions (white-box constraint ✓)
- Quant interview prep
- Academic research
- Fund development workflows

---

**Framework Version**: 1.0  
**Last Updated**: June 5, 2026  
**Status**: ✅ Production Ready
