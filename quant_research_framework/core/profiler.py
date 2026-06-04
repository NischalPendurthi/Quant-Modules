# quant_research_framework/core/profiler.py
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

class SeriesProfiler:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def profile_all(self) -> pd.DataFrame:
        profiles = []
        for col in self.data.columns:
            series = self.data[col].dropna()
            profile = self._profile_series(series)
            profile['column'] = col
            profiles.append(profile)
        return pd.DataFrame(profiles).set_index('column')
    
    def _profile_series(self, s: pd.Series) -> dict:
        return {
            'mean': s.mean(),
            'std': s.std(),
            'adf_stat': adfuller(s)[0],
            'adf_pvalue': adfuller(s)[1],
            'kpss_stat': kpss(s, regression='c')[0],
            'kpss_pvalue': kpss(s, regression='c')[1],
            'hurst': self._hurst(s),
            'autocorr_lag1': s.autocorr(lag=1),
            'is_stationary': adfuller(s)[1] < 0.05,
        }
    
    def _hurst(self, s: pd.Series) -> float:
        """Simple Hurst exponent estimation"""
        lags = range(2, 100)
        tau = [np.std(np.diff(s, lag)) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]