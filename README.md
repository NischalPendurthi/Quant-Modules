# Quant-Modules

Reusable, interface-driven quantitative research framework with strict shape checks,
memory-efficient ingestion, and NumPy/optional-Numba implementations.

## Modules

- `ColumnAnalyzer`: distribution, stationarity, lead/lag, and missingness diagnostics.
- `FeatureFactory`: returns, z-scores, spreads, momentum, fractional differencing, and alpha-style features.
- `SignalResearch`: IC/rank-IC analysis, stability, and decay diagnostics.
- `SignalEvaluator`: matrix-based IC, rank-IC, turnover, and Sharpe for many signals/horizons.
- `StrategyBuilder`: threshold, mean-reversion, momentum, and pairs position builders.
- `Backtester`: vectorized and event-driven execution plus PnL computation.

## Quick test

```bash
python -m unittest discover -s tests -v
```
