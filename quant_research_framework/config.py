# quant_research_framework/config.py
from pathlib import Path

class Config:
    # Data
    DATA_PATH = Path("data")
    OUTPUT_PATH = Path("output")
    
    # Analysis
    TARGET_PERCENTILE = 0.20          # Top/Bottom % for leaders/followers
    LOOKBACK_WINDOW = 30              # Days for lead-lag / cointegration
    MIN_ASSETS = 40                   # Minimum assets with no NaNs
    Z_ENTRY = 2.0                     # Z-score entry threshold
    Z_EXIT = 0.0                      # Z-score exit threshold
    
    # Backtesting
    INITIAL_CAPITAL = 1_000_000
    TRANSACTION_COST = 0.001          # 10 bps
    
    # Visualization
    FIGSIZE = (14, 8)
    
    @classmethod
    def setup_dirs(cls):
        cls.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        (cls.OUTPUT_PATH / "profiles").mkdir(exist_ok=True)
        (cls.OUTPUT_PATH / "graphs").mkdir(exist_ok=True)
        (cls.OUTPUT_PATH / "backtests").mkdir(exist_ok=True)