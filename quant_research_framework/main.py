# quant_research_framework/main.py
from pathlib import Path
import pandas as pd
from config import Config
from core.profiler import SeriesProfiler
from core.relationships import RelationshipAnalyzer
from core.target_discovery import TargetDiscovery
from core.cointegration import CointegrationAnalyzer
from utils.synthetic_data import SyntheticDataGenerator

def main():
    Config.setup_dirs()
    
    print("=== Anonymous Quant Research Framework ===\n")
    
    # === Option 1: Use Synthetic Data (Recommended for testing) ===
    generator = SyntheticDataGenerator(n_series=50, n_days=1500)
    data = generator.generate()
    
    # === Option 2: Use real competition data ===
    # data = pd.read_csv(Config.DATA_PATH / "competition_data.csv", index_col=0, parse_dates=True)
    
    # 1. Profile all series
    profiler = SeriesProfiler(data)
    profiles = profiler.profile_all()
    print("Series Stationarity Summary:")
    print(profiles['is_stationary'].value_counts())
    
    # 2. Relationship Analysis
    rel = RelationshipAnalyzer(data)
    relationships = rel.compute_all()
    
    # 3. Target Discovery
    discovery = TargetDiscovery(data)
    discovery_results = discovery.run(correlation_threshold=0.35)
    
    print("\n=== Top 5 Likely Target Candidates ===")
    print(discovery_results['likely_target_candidates'].head(5).round(4))
    
    # 4. Cointegration Analysis on top candidates
    top_candidates = discovery_results['likely_target_candidates'].head(8).index.tolist()
    coin = CointegrationAnalyzer(data)
    
    print("\n=== Cointegration Analysis ===")
    for i in range(len(top_candidates)-1):
        for j in range(i+1, len(top_candidates)):
            tickers = [top_candidates[i], top_candidates[j]]
            johansen_result, rank = coin.johansen_test(tickers)
            if rank and rank > 0:
                spread, weights = coin.build_spread(tickers)
                stat = coin.is_stationary(spread)
                print(f"{tickers[0]} & {tickers[1]} → Rank {rank} | Stationary: {stat['stationary']}")
    
    # Save results
    profiles.to_csv(Config.OUTPUT_PATH / "profiles" / "series_profiles.csv")
    print(f"\n✅ Analysis complete. Results saved to: {Config.OUTPUT_PATH}")

if __name__ == "__main__":
    main()