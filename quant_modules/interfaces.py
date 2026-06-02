from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class ColumnAnalyzerInterface(ABC):
    @abstractmethod
    def analyze_distribution(self) -> dict: ...

    @abstractmethod
    def analyze_stationarity(self) -> dict: ...

    @abstractmethod
    def analyze_lead_lag(self, reference: np.ndarray, max_lag: int = 5) -> dict: ...

    @abstractmethod
    def analyze_missingness(self) -> dict: ...


class FeatureFactoryInterface(ABC):
    @abstractmethod
    def create_returns(self, prices: np.ndarray, period: int = 1) -> np.ndarray: ...

    @abstractmethod
    def create_zscores(self, values: np.ndarray, window: int = 60) -> np.ndarray: ...

    @abstractmethod
    def create_spreads(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def create_momentum(self, prices: np.ndarray, window: int = 20) -> np.ndarray: ...


class SignalResearchInterface(ABC):
    @abstractmethod
    def future_return_corr(self, signal: np.ndarray, future_returns: np.ndarray) -> float: ...

    @abstractmethod
    def information_coefficient(self, signal: np.ndarray, future_returns: np.ndarray) -> float: ...

    @abstractmethod
    def rank_ic(self, signal: np.ndarray, future_returns: np.ndarray) -> float: ...

    @abstractmethod
    def stability(self, signal: np.ndarray, future_returns: np.ndarray, window: int = 252) -> dict: ...

    @abstractmethod
    def decay_analysis(self, signal: np.ndarray, returns_by_horizon: np.ndarray) -> np.ndarray: ...


class SignalEvaluatorInterface(ABC):
    @abstractmethod
    def compute_ic(self, signals: np.ndarray, forward_returns: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def compute_rank_ic(self, signals: np.ndarray, forward_returns: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def compute_turnover(self, positions: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def compute_sharpe(self, pnl: np.ndarray, periods_per_year: int = 252 * 390) -> np.ndarray: ...


class StrategyBuilderInterface(ABC):
    @abstractmethod
    def threshold_strategy(self, signal: np.ndarray, upper: float, lower: float) -> np.ndarray: ...

    @abstractmethod
    def mean_reversion_strategy(self, signal: np.ndarray, entry_z: float, exit_z: float) -> np.ndarray: ...

    @abstractmethod
    def momentum_strategy(self, signal: np.ndarray, threshold: float = 0.0) -> np.ndarray: ...

    @abstractmethod
    def pairs_strategy(self, price_a: np.ndarray, price_b: np.ndarray, entry_z: float = 2.0, exit_z: float = 0.5) -> np.ndarray: ...


class BacktesterInterface(ABC):
    @abstractmethod
    def run_vectorized(self, prices: np.ndarray, signals: np.ndarray) -> dict: ...

    @abstractmethod
    def run_event_driven(self, data: np.ndarray, strategy: object) -> dict: ...

    @abstractmethod
    def compute_pnl(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray: ...
