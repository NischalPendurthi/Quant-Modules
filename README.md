# White-Box Quantitative Signal Discovery & Backtesting Framework

A complete, production-ready Python framework for developing **mathematically interpretable** quantitative trading signals suitable for presentation and defense before a quantitative trading jury.

## Key Features

✅ **White-Box Only**: No neural networks, random forests, gradient boosting, or other black-box models  
✅ **Fully Interpretable**: Every signal component is mathematically explainable  
✅ **No Look-Ahead Bias**: All calculations use only information available at time T  
✅ **Efficient**: Handles millions of rows with 5-minute frequency data  
✅ **Production-Ready**: Modular, clean code with comprehensive documentation  
✅ **Complete Pipeline**: Data loading → Feature engineering → Model training → Backtesting → Reporting  

---

## Architecture

The framework is organized into 8 modular Python files:

```
backtesting/
├── data.py              # Data loading, preprocessing, missing values
├── features.py          # Feature engineering & selection
├── models.py            # 6 interpretable models (OLS, Ridge, Lasso, Rolling, Factor Score, Rank Alpha)
├── signal.py            # Signal quality analysis (IC, hit ratio, stability, decay)
├── backtest.py          # Event-driven backtesting engine (long-only & long-short)
├── metrics.py           # Performance metrics (Sharpe, Sortino, Calmar, VaR, CVaR, etc.)
├── plots.py             # 11 publication-quality visualizations
├── report.py            # Automatic research report generation
├── main.py              # End-to-end pipeline orchestration
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Pipeline Phases

### Phase 1: Data Loading & Preprocessing
- **Input**: CSV with columns `Date`, `Time`, `y`, `f1`–`f49`
- **Output**: Clean, aligned time series with proper handling of missing values

**Key functions**:
```python
from data import load_data, handle_missing, missing_value_report, train_test_split

df = load_data("data.csv")                           # Load and align timestamps
missing_report = missing_value_report(df)            # Analyze gaps
df = handle_missing(df, method="ffill")              # Impute strategically
X_train, X_test, y_train, y_test = train_test_split(df, test_years=1)  # Chronological split
```

---

### Phase 2: Feature Engineering
- Transforms 49 raw features into 100+ interpretable derived features
- No look-ahead bias; all calculations use lagged data

**Transformations**:
- **Rolling Statistics**: Mean, median, std, z-score (windows: 6, 12, 24 bars)
- **Momentum**: Lags, raw momentum, pct change (lags: 1, 3, 6, 12, 24)
- **Cross-Sectional**: Relative ranking, relative spread
- **Regime**: Volatility regime, trend regime

**Key functions**:
```python
from features import engineer_features, select_features

df_eng = engineer_features(df, windows=(6, 12, 24), lags=(1, 3, 6, 12, 24))
selected_features, ranking_df = select_features(
    X_train, y_train, top_k=20,
    output_path="selected_features.csv"
)
```

---

### Phase 3: Feature Selection
- Scores all features by multiple explainable criteria:
  - Pearson & Spearman Information Coefficient (IC)
  - Rolling IC stability (ICIR)
  - Mutual Information
  - t-statistics
- Selects top-k features by composite score

**Output**: `selected_features.csv` with ranking table

---

### Phase 4: Model Training & Comparison
Implements 6 interpretable models from scratch (NumPy only):

#### **Model 1: OLS Linear Regression**
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
β = (X'X)⁻¹ X'y
```
Baseline linear model; closed-form solution.

#### **Model 2: Ridge Regression**
```
β_ridge = (X'X + λI)⁻¹ X'y
```
L2 regularization; shrinks all coefficients toward zero.

#### **Model 3: Lasso Regression**
```
min ||y - Xβ||² + λ||β||₁
```
L1 regularization; produces sparse coefficients (feature selection).

#### **Model 4: Rolling OLS (Adaptive)**
```
β_t = (X_{t-w:t}'X_{t-w:t})⁻¹ X_{t-w:t}'y_{t-w:t}
ŷ_t = x_t' β_{t-1}   (no look-ahead)
```
Time-varying coefficients; adapts to market regime shifts.

#### **Model 5: Factor Score (IC-Weighted)**
```
Signal_t = Σᵢ IC_i · rank(fᵢ,t)
```
Weights determined by historical IC; rank-normalized for robustness.

#### **Model 6: Rank-Based Alpha**
```
Signal_t = Σᵢ RankIC_i · pct_rank(fᵢ,t)
```
Uses Spearman IC; more robust to outliers than raw values.

**Comparison metrics** (for each model):
- Train/Test IC (Pearson & Spearman)
- Train/Test MSE, RMSE, MAE
- Train/Test R²

**Selection criterion**: Best out-of-sample Rank IC

**Key functions**:
```python
from models import compare_models, save_model

best_model, y_pred_train, y_pred_test, comparison_df = compare_models(
    X_train, X_test, y_train, y_test, selected_features,
    ridge_alpha=1.0, lasso_alpha=0.001, rolling_window=500
)
save_model(best_model, "best_model.pkl")
```

---

### Phase 5: Signal Quality Analysis
Comprehensive evaluation of the signal:

**Information Coefficient (IC) Metrics**:
- Mean IC, Median IC, IC Std, ICIR
- Rolling IC (with time decay analysis)
- Monthly IC decomposition

**Signal Quality Metrics**:
- Hit Ratio (directional accuracy)
- Long/Short accuracy
- Signal stability
- Signal decay (lookahead analysis)
- Prediction distribution (mean, std, skew, kurtosis)

**Key functions**:
```python
from signal import analyze_signal, monthly_ic, rolling_ic, print_signal_report

analysis = analyze_signal(y_test, y_pred_test, rolling_window=252)
print_signal_report(analysis)

monthly_ic_df = monthly_ic(y_test, y_pred_test)
rolling_ic_series = rolling_ic(y_test, y_pred_test, window=252)
```

---

### Phase 6: Backtesting Engine
Realistic event-driven simulation with transaction costs.

#### **Long-Only Strategy**
```
if signal > threshold:
    Long (position = +1)
else:
    Flat (position = 0)
```

#### **Long-Short Strategy**
```
if signal > upper_threshold:
    Long (position = +1)
elif signal < lower_threshold:
    Short (position = -1)
else:
    Flat (position = 0)
```

**Position Sizing Methods**:
- Equal-weight
- Volatility-adjusted
- Signal-strength weighted

**Constraints**:
- Long-only max exposure: $1,000,000
- Long-short max exposure: $2,000,000
- Cash balance > 0
- Start flat, end flat

**Transactions**:
- 10 bps (0.1%) per trade
- Tracked individually for P&L attribution

**Key functions**:
```python
from backtest import LongOnlyBacktest, LongShortBacktest

backtest_lo = LongOnlyBacktest(
    initial_capital=1_000_000,
    position_size=1.0,
    threshold=0.0,
    transaction_cost=10.0
)
results_lo = backtest_lo.backtest(timestamps, prices, signal, returns)

backtest_ls = LongShortBacktest(
    initial_capital=2_000_000,
    position_size=0.5,
    upper_threshold=0.5,
    lower_threshold=-0.5,
    transaction_cost=10.0
)
results_ls = backtest_ls.backtest(timestamps, prices, signal, returns)
```

---

### Phase 7: Performance Metrics
Comprehensive evaluation across all key dimensions.

**Return Metrics**:
- Annual Return
- Total Return
- CAGR

**Risk Metrics**:
- Volatility (annualized)
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

**Drawdown**:
- Maximum Drawdown
- Average Drawdown

**Win/Loss Statistics**:
- Profit Factor
- Hit Ratio
- Average Win / Average Loss
- Win/Loss Ratio

**Tail Risk**:
- Value at Risk (VaR, 95%)
- Expected Shortfall (CVaR, 95%)
- Skewness
- Kurtosis
- Tail Ratio

**Trading Activity**:
- Average Turnover
- Average Holding Period

**Key functions**:
```python
from metrics import compute_all_metrics, print_metrics_report

metrics = compute_all_metrics(backtest_results, timestamps)
print_metrics_report(metrics)
```

---

### Phase 8: Visualizations
11 publication-quality plots (matplotlib + seaborn):

1. **Cumulative NAV** – Portfolio value over time with fill
2. **Daily PnL** – Realized P&L by day (green/red bars)
3. **Drawdown Curve** – Underwater plot; worst peak-to-trough
4. **Rolling Sharpe Ratio** – Risk-adjusted returns over time
5. **Rolling IC** – Predictive power stability
6. **Signal Distribution** – Histogram + box plot
7. **Actual vs Predicted** – Scatter with reference diagonal
8. **Feature Importance** – Bar chart of top features by IC weight
9. **Monthly Returns Heatmap** – Year × Month matrix of P&L
10. **Market Exposure** – Position sizing over time
11. **Turnover & Costs** – Daily turnover + cumulative transaction costs

**Key functions**:
```python
from plots import generate_all_plots

plot_paths = generate_all_plots(
    backtest_results, y_true, y_pred, signal,
    feature_names, feature_ics,
    output_dir="plots"
)
```

---

### Phase 9: Research Report
Automatic generation of a professional, jury-ready report:

**Sections**:
1. Executive Summary (key metrics)
2. Data Overview (source, frequency, statistics)
3. Feature Engineering (transformations, selection methodology)
4. Signal Model Specification (mathematical formulation, formula)
5. Model Coefficients & Significance
6. IC Analysis (in-sample and out-of-sample)
7. Backtesting Results (all metrics)
8. Portfolio Analysis (returns, risk, drawdown)
9. Risk Analysis (tail metrics, VaR, CVaR)
10. Trading Formula (closed-form representation)
11. Conclusions & Next Steps

**Key functions**:
```python
from report import generate_research_report

report = generate_research_report(
    title="White-Box Signal Discovery",
    data_summary=data_summary,
    signal_name="IC-Weighted Feature Score",
    strategy_type="long-short",
    model_name="Ridge Regression",
    model_description="...",
    formula="Signal = ...",
    coef_df=coef_dataframe,
    feature_names=selected_features,
    n_engineered=150,
    ic_metrics=ic_dict,
    backtest_metrics=metrics,
    output_path="research_report.txt"
)
```

---

## Usage

### Quick Start

```python
from main import Config, main

# Default configuration
config = Config()
config.DATA_PATH = "my_data.csv"
config.TOP_K_FEATURES = 20
config.OUTPUT_DIR = "results"

# Run entire pipeline
results = main(config)
```

### Custom Configuration

```python
from main import Config, main

config = Config()
config.DATA_PATH = "data.csv"

# Feature engineering
config.ROLLING_WINDOWS = (6, 12, 24)
config.MOMENTUM_LAGS = (1, 3, 6, 12, 24)

# Model selection
config.RIDGE_ALPHA = 0.5
config.LASSO_ALPHA = 0.001
config.ROLLING_WINDOW = 500

# Backtesting
config.LONG_SHORT_THRESHOLD_UP = 0.3
config.LONG_SHORT_THRESHOLD_DN = -0.3
config.POSITION_SIZE = 0.6
config.TRANSACTION_COST_BPS = 10.0

# Run
results = main(config)
```

### Step-by-Step Workflow

```python
import numpy as np
import pandas as pd
from data import load_data, handle_missing, train_test_split
from features import engineer_features, select_features
from models import compare_models, save_model
from backtest import LongOnlyBacktest, LongShortBacktest
from metrics import compute_all_metrics, print_metrics_report
from signal import analyze_signal, print_signal_report
from plots import generate_all_plots

# 1. Load & preprocess
df = load_data("data.csv")
df = handle_missing(df, method="ffill")

# 2. Engineer features
df = engineer_features(df, windows=(6, 12, 24), lags=(1, 3, 6, 12, 24))

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(df, test_years=1)

# 4. Select features
features, ranking = select_features(X_train, y_train, top_k=20)
X_train = X_train[features]
X_test = X_test[features]

# 5. Train models
best_model, y_pred_train, y_pred_test, comparison = compare_models(
    X_train.values, X_test.values, y_train.values, y_test.values,
    features, ridge_alpha=1.0
)

# 6. Analyze signal
signal_analysis = analyze_signal(y_test, pd.Series(y_pred_test, index=y_test.index))
print_signal_report(signal_analysis)

# 7. Backtest
backtest = LongShortBacktest()
results = backtest.backtest(y_test.index, np.ones(len(y_test))*100, y_pred_test, y_test.values)

# 8. Evaluate
metrics = compute_all_metrics(results, y_test.index)
print_metrics_report(metrics)

# 9. Visualize
plot_paths = generate_all_plots(
    results, y_test, pd.Series(y_pred_test, index=y_test.index),
    y_pred_test, features, best_model.coef_
)
```

---

## Input Data Format

**CSV file with columns**:
```
Date,Time,y,f1,f2,...,f49
2018-01-01,09:30:00,-0.0005,1.2,3.4,...
2018-01-01,09:35:00,0.0003,1.5,3.2,...
...
```

- **Date**: YYYY-MM-DD format
- **Time**: HH:MM:SS format (5-minute intervals)
- **y**: Target variable (future return)
- **f1–f49**: Raw explanatory features (numeric)

---

## Output Files

All outputs saved to `output/` directory:

```
output/
├── best_model.pkl                      # Serialized model
├── predictions_train.csv               # Training predictions (y_true, y_pred, residuals)
├── predictions_test.csv                # Test predictions
├── selected_features.csv               # Feature ranking (IC, t-stat, mutual info)
├── model_comparison.csv                # All 6 models' metrics
├── monthly_ic.csv                      # IC decomposed by month
├── long_only_results.csv               # Trade-by-trade long-only backtest
├── long_short_results.csv              # Trade-by-trade long-short backtest
├── research_report.txt                 # Publication-ready report
└── plots/
    ├── 01_cumulative_pnl.png
    ├── 02_daily_pnl.png
    ├── 03_drawdown.png
    ├── 04_rolling_sharpe.png
    ├── 05_rolling_ic.png
    ├── 06_signal_distribution.png
    ├── 07_actual_vs_predicted.png
    ├── 08_feature_importance.png
    ├── 09_monthly_returns_heatmap.png
    ├── 10_exposure.png
    └── 11_turnover.png
```

---

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run framework
python main.py
```

---

## Dependencies

- **NumPy** ≥ 1.20: Numerical computations
- **Pandas** ≥ 1.3: Data manipulation & time series
- **SciPy** ≥ 1.7: Statistical functions (IC, correlation)
- **Matplotlib** ≥ 3.4: Publication-quality plots
- **Seaborn** ≥ 0.11: Statistical visualization
- **Scikit-Learn** ≥ 0.24: Helper functions (rankdata, etc.)

---

## Mathematical Foundations

### Information Coefficient (IC)

The IC measures the predictive power of a signal:

$$IC = \text{Cov}(\text{signal}, y) / (\sigma_{\text{signal}} \cdot \sigma_y)$$

- **IC = 0**: No predictive power
- **IC = 0.05**: Weak predictive power (typical for intraday signals)
- **IC ≥ 0.10**: Strong predictive power
- **IC > 0.20**: Exceptional predictive power

### Rolling Coefficient (time-varying OLS)

Allows the model to adapt to non-stationary markets:

$$\beta_t = (X_{t-w:t}' X_{t-w:t})^{-1} X_{t-w:t}' y_{t-w:t}$$

where $w$ is the window length (typically 500 bars = ~2 trading days).

### Rank Normalization

Converts raw features to [0,1] percentile scale:

$$\text{rank}_{\text{norm}} = \frac{\text{rank}(f_i) - 1}{n - 1}$$

This removes scale effects and makes the model robust to outliers.

---

## Performance Standards

**Typical Results** (on liquid intraday data):

| Metric | Benchmark | Good | Excellent |
|--------|-----------|------|-----------|
| Sharpe Ratio | 0.0 | 0.5–1.0 | > 1.5 |
| Hit Ratio | 50% | 51–55% | > 55% |
| CAGR | 0% | 5–20% | > 20% |
| Max Drawdown | - | -10% to -5% | < -5% |
| Profit Factor | 1.0 | 1.5–2.0 | > 2.0 |

---

## Best Practices for Jury Defense

1. **Explain the economics**: Why should this signal work? (Mean reversion? Momentum? Regime effects?)
2. **Show out-of-sample results**: The test period IC is the key metric
3. **Decompose P&L**: Show monthly breakdown; highlight consistency
4. **Discuss risks**: Transaction costs, slippage, market impact, regime change
5. **Demonstrate robustness**: Rolling IC stability; parameter sensitivity
6. **Quantify tail risk**: VaR, CVaR, max drawdown—jury wants to see you understand downside
7. **Provide formula**: The trading rule should fit on one line

---

## Example Jury Presentation

```
Title: "Volatility-Regime-Aware Momentum Signal"

1. Signal Formula:
   Signal_t = 0.45·momz(6) + 0.32·lagmean(12) - 0.18·trendRev(24)
   
2. Test Period Performance:
   • Annual Return: 12.5%
   • Sharpe: 0.85
   • Hit Ratio: 53.2%
   • Max Drawdown: -8.3%
   
3. Information Coefficient:
   • Out-of-Sample IC: 0.0847
   • Rolling IC Stability (ICIR): 1.24
   
4. Economics:
   "This signal captures mean-reversion in short-term momentum
   when volatility regime is LOW, and pure momentum when HIGH.
   The feature weighting is justified by rolling IC analysis."
   
5. Robustness:
   "Signal remains profitable across:
   - Different market regimes (bull/bear)
   - Different volatility regimes
   - Time-varying transaction costs (5–15 bps)
   - Rolling window re-estimation (quarterly)"
```

---

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'pandas'"
```bash
pip install -r requirements.txt
```

**Issue**: "Data file not found"
```python
# Check the DATA_PATH in main.py or config
# File should be in the same directory as main.py
```

**Issue**: "Insufficient data for feature selection"
```python
# Reduce TOP_K_FEATURES or increase DATA_PATH file size
config.TOP_K_FEATURES = 10
```

**Issue**: "NaN in predictions"
```python
# Increase rolling windows or adjust ROLLING_IC_WINDOW
# or use method='bfill' for missing value handling
```

---

## Literature & References

- **Information Coefficient**: Grinold & Kahn (1999), "Active Portfolio Management"
- **Mean Reversion & Momentum**: Jegadeesh & Titman (1993)
- **Sharpe Ratio**: Sharpe (1994), "The Sharpe Ratio"
- **Sortino Ratio**: Sortino & Price (1994), "Performance Measurement in a Downside Risk Framework"
- **Rolling Regression**: Hamilton & Lin (1996), "Stock Market Volatility and Business Cycles"

---

## License

This framework is provided as-is for educational and research purposes.

---

## Contact & Support

For questions or improvements, please open an issue or submit a pull request.

---

**Last Updated**: 2026-06-05  
**Framework Version**: 1.0  
**Author**: Quantitative Research Team
