# quant_research_framework/core/cointegration.py
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class CointegrationAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def johansen_test(self, tickers, det_order=0, k_ar_diff=1):
        """Johansen cointegration test. Returns (result, rank) or (None, 0)."""
        subset = self.data[list(tickers)].dropna()
        if len(subset) < 30:
            return None, 0
        try:
            result = coint_johansen(subset.values, det_order, k_ar_diff)
        except Exception:
            return None, 0
        rank = self._trace_rank(result)
        return result, rank

    def _trace_rank(self, result, signif_col=1):
        """Infer cointegration rank from Johansen trace test (95%: signif_col=1)."""
        rank = 0
        for i, stat in enumerate(result.lr1):
            if stat > result.cvt[i, signif_col]:
                rank = len(result.lr1) - i
        return rank

    def build_spread(self, tickers, det_order=0, k_ar_diff=1):
        """Cointegrating spread using the first eigenvector."""
        subset = self.data[list(tickers)].dropna()
        result = coint_johansen(subset.values, det_order, k_ar_diff)
        weights = result.evec[:, 0]
        spread = pd.Series(subset.values @ weights, index=subset.index, name="spread")
        return spread, weights

    def is_stationary(self, spread: pd.Series) -> dict:
        series = spread.dropna()
        if len(series) < 20:
            return {"stationary": False, "adf_pvalue": 1.0, "adf_stat": np.nan}
        adf_stat, pvalue, _, _, _, _ = adfuller(series)
        return {
            "stationary": pvalue < 0.05,
            "adf_pvalue": pvalue,
            "adf_stat": adf_stat,
        }
