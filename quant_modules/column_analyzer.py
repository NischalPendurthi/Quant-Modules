from __future__ import annotations

import numpy as np

from .interfaces import ColumnAnalyzerInterface
from .ingestion import ingest_matrix
from .utils import ensure_1d


class ColumnAnalyzer(ColumnAnalyzerInterface):
    def __init__(self, values: np.ndarray):
        self.values = ingest_matrix(values)

    def analyze_distribution(self) -> dict:
        x = self.values
        mu = np.nanmean(x, axis=0)
        sigma = np.nanstd(x, axis=0)
        centered = x - mu
        with np.errstate(invalid="ignore", divide="ignore"):
            skew = np.nanmean(centered**3, axis=0) / (sigma**3 + 1e-12)
            kurt = np.nanmean(centered**4, axis=0) / (sigma**4 + 1e-12)

        return {
            "mean": mu.astype(np.float32),
            "std": sigma.astype(np.float32),
            "min": np.nanmin(x, axis=0).astype(np.float32),
            "max": np.nanmax(x, axis=0).astype(np.float32),
            "skew": skew.astype(np.float32),
            "kurtosis": kurt.astype(np.float32),
            "q01": np.nanquantile(x, 0.01, axis=0).astype(np.float32),
            "q99": np.nanquantile(x, 0.99, axis=0).astype(np.float32),
        }

    def analyze_stationarity(self) -> dict:
        x = self.values
        dx = np.diff(x, axis=0)
        var_level = np.nanvar(x, axis=0) + 1e-12
        var_diff = np.nanvar(dx, axis=0)
        lagged = x[:-1]
        current = x[1:]
        cov = np.nanmean((lagged - np.nanmean(lagged, axis=0)) * (current - np.nanmean(current, axis=0)), axis=0)
        auto_corr_1 = cov / (np.nanstd(lagged, axis=0) * np.nanstd(current, axis=0) + 1e-12)

        return {
            "variance_ratio": (var_diff / var_level).astype(np.float32),
            "autocorr_lag1": auto_corr_1.astype(np.float32),
        }

    def analyze_lead_lag(self, reference: np.ndarray, max_lag: int = 5) -> dict:
        ref = ensure_1d("reference", reference)
        if ref.shape[0] != self.values.shape[0]:
            raise ValueError("reference length must match number of rows in analyzer values")
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative")

        cols = self.values.shape[1]
        lags = np.arange(-max_lag, max_lag + 1)
        corr = np.full((lags.size, cols), np.nan, dtype=np.float32)
        for i, lag in enumerate(lags):
            if lag < 0:
                x = ref[-lag:]
                y = self.values[:lag, :]
            elif lag > 0:
                x = ref[:-lag]
                y = self.values[lag:, :]
            else:
                x = ref
                y = self.values
            x_mu = np.nanmean(x)
            x_std = np.nanstd(x) + 1e-12
            y_mu = np.nanmean(y, axis=0)
            y_std = np.nanstd(y, axis=0) + 1e-12
            cov = np.nanmean((x[:, None] - x_mu) * (y - y_mu), axis=0)
            corr[i, :] = (cov / (x_std * y_std)).astype(np.float32)
        return {"lags": lags.astype(np.int16), "corr": corr}

    def analyze_missingness(self) -> dict:
        mask = np.isnan(self.values)
        missing_rate = mask.mean(axis=0).astype(np.float32)
        missing_count = mask.sum(axis=0).astype(np.int32)

        max_run = np.zeros(mask.shape[1], dtype=np.int32)
        for c in range(mask.shape[1]):
            run = 0
            best = 0
            for r in range(mask.shape[0]):
                if mask[r, c]:
                    run += 1
                    if run > best:
                        best = run
                else:
                    run = 0
            max_run[c] = best

        return {
            "missing_rate": missing_rate,
            "missing_count": missing_count,
            "max_consecutive_missing": max_run,
        }
