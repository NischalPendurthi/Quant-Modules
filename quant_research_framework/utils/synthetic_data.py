# quant_research_framework/utils/synthetic_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    """
    Generates realistic anonymous time series for framework testing.
    One hidden target + correlated predictors + noise.
    """
    
    def __init__(self, n_series=50, n_days=1000, seed=42):
        np.random.seed(seed)
        self.n_series = n_series
        self.n_days = n_days
        self.dates = pd.date_range('2020-01-01', periods=n_days)
    
    def generate(self) -> pd.DataFrame:
        # Hidden target: random walk with drift
        target = np.cumsum(np.random.randn(self.n_days) * 0.5 + 0.01)
        target = (target - target.mean()) / target.std()
        
        data = {'TICKER_00': target}  # Hidden tradable asset
        
        # Generate predictors with varying lead-lag and correlation
        for i in range(1, self.n_series):
            col_name = f'TICKER_{i:02d}'
            
            # Some are leaders, some followers, some noise
            if i % 5 == 0:  # Strong leader
                lag = np.random.randint(3, 8)
                noise = np.random.randn(self.n_days) * 0.3
                series = np.roll(target, -lag) + noise
            elif i % 5 == 1:  # Follower
                lag = np.random.randint(3, 8)
                noise = np.random.randn(self.n_days) * 0.3
                series = np.roll(target, lag) + noise
            elif i % 7 == 0:  # Sector-related
                series = target * 0.7 + np.random.randn(self.n_days) * 0.6
            else:  # Pure noise
                series = np.random.randn(self.n_days) * 1.2
            
            data[col_name] = (series - series.mean()) / series.std()
        
        df = pd.DataFrame(data, index=self.dates)
        print(f"✅ Generated synthetic dataset: {df.shape}")
        print(f"   Hidden target: TICKER_00")
        return df