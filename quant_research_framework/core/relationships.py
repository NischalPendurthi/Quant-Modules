# quant_research_framework/core/relationships.py
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests

class RelationshipAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def compute_all(self) -> pd.DataFrame:
        """Compute correlation, spearman, granger, cross-corr"""
        corr = self.data.corr()
        spear = self.data.corr(method='spearman')
        
        # Simple Granger (example for one lag)
        granger_matrix = pd.DataFrame(index=self.data.columns, columns=self.data.columns)
        for col1 in self.data.columns:
            for col2 in self.data.columns:
                if col1 != col2:
                    try:
                        gr = grangercausalitytests(self.data[[col2, col1]], maxlag=1, verbose=False)
                        granger_matrix.loc[col1, col2] = gr[1][0][0]  # p-value
                    except:
                        granger_matrix.loc[col1, col2] = np.nan
        return {
            'pearson': corr,
            'spearman': spear,
            'granger_pvalue': granger_matrix
        }