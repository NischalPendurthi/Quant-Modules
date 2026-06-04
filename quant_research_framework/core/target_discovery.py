# quant_research_framework/core/target_discovery.py
import pandas as pd
from core.graph_builder import DependencyGraph

class TargetDiscovery:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.graph = None
    
    def run(self, correlation_threshold=0.3):
        # Build relationship matrix
        corr = self.data.corr()
        
        # Build graph
        self.graph = DependencyGraph(corr, threshold=correlation_threshold)
        
        # Get leadership scores
        scores = self.graph.get_lead_lag_scores()
        
        return {
            'likely_target_candidates': scores,
            'likely_followers': scores.tail(5),
            'graph': self.graph.G
        }