# quant_research_framework/core/graph_builder.py
import networkx as nx
import pandas as pd

class DependencyGraph:
    def __init__(self, relationship_matrix: pd.DataFrame, threshold=0.3):
        self.G = nx.DiGraph()
        self._build_graph(relationship_matrix, threshold)
    
    def _build_graph(self, matrix: pd.DataFrame, threshold: float):
        for i in matrix.columns:
            for j in matrix.columns:
                if i != j:
                    val = matrix.loc[i, j]
                    if abs(val) > threshold:
                        self.G.add_edge(i, j, weight=val)
    
    def get_lead_lag_scores(self) -> pd.Series:
        """Row mean = leadership score"""
        scores = {}
        for node in self.G.nodes():
            scores[node] = sum(data['weight'] for _, _, data in self.G.out_edges(node, data=True))
        return pd.Series(scores).sort_values(ascending=False)