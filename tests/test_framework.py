import unittest
import numpy as np

from quant_modules import (
    Backtester,
    ColumnAnalyzer,
    FeatureFactory,
    SignalEvaluator,
    SignalResearch,
    StrategyBuilder,
    ingest_matrix,
)


class TestFramework(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.prices = np.cumsum(rng.normal(0.0, 0.1, size=(300, 3)).astype(np.float32), axis=0) + 100.0
        self.signals = rng.normal(0.0, 1.0, size=(300, 3)).astype(np.float32)

    def test_ingest_downcasts_float64(self):
        x = np.ones((10, 2), dtype=np.float64)
        out = ingest_matrix(x)
        self.assertEqual(out.dtype, np.float32)

    def test_column_analyzer_shapes(self):
        analyzer = ColumnAnalyzer(self.signals)
        dist = analyzer.analyze_distribution()
        self.assertEqual(dist["mean"].shape, (3,))
        miss = analyzer.analyze_missingness()
        self.assertEqual(miss["missing_count"].shape, (3,))

    def test_feature_factory_and_shape_checks(self):
        factory = FeatureFactory()
        ret = factory.create_returns(self.prices, period=1)
        self.assertEqual(ret.shape, self.prices.shape)
        with self.assertRaises(ValueError):
            factory.create_spreads(self.prices, self.prices[:, :2])

    def test_signal_evaluator_matrix_ic(self):
        evaluator = SignalEvaluator()
        fwd = np.roll(self.signals, -1, axis=0)
        ic = evaluator.compute_ic(self.signals, fwd)
        rank_ic = evaluator.compute_rank_ic(self.signals, fwd)
        self.assertEqual(ic.shape, (3, 3))
        self.assertEqual(rank_ic.shape, (3, 3))

    def test_signal_research_decay(self):
        research = SignalResearch()
        one_signal = self.signals[:, :1]
        horizons = np.column_stack([
            np.roll(one_signal[:, 0], -h) for h in range(1, 4)
        ]).astype(np.float32)
        decay = research.decay_analysis(one_signal, horizons)
        self.assertEqual(decay.shape, (3,))

    def test_strategy_and_backtester(self):
        builder = StrategyBuilder()
        backtester = Backtester()
        pos = builder.momentum_strategy(self.signals, threshold=0.0)
        result = backtester.run_vectorized(self.prices, pos.astype(np.float32))
        self.assertIn("pnl", result)
        self.assertEqual(result["pnl"].shape, self.prices.shape)


if __name__ == "__main__":
    unittest.main()
