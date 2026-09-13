import os
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
FACTOR_LAYER_PATH = os.path.join(PROJECT_ROOT, "src", "Factors_Layer")
SELECTION_LAYER_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
    "Factor_Selection_Layer",
)

for path in (FACTOR_LAYER_PATH, SELECTION_LAYER_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from forward_returns import compute_forward_returns
from quantile_analysis import (
    aggregate_quantile_returns,
    prepare_factor_quantiles,
)


class FactorLayerTest(unittest.TestCase):
    def test_forward_return_requires_quality_at_both_endpoints(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0], "B": [50.0, 55.0, 60.0]},
            index=dates,
        )
        quality = pd.DataFrame(True, index=dates, columns=prices.columns)
        quality.loc[dates[1], "B"] = False

        result = compute_forward_returns(prices, quality, horizon=1)

        self.assertAlmostEqual(result.loc[dates[0], "A"], 0.10)
        self.assertAlmostEqual(result.loc[dates[1], "A"], 0.10)
        self.assertTrue(pd.isna(result.loc[dates[0], "B"]))
        self.assertTrue(pd.isna(result.loc[dates[1], "B"]))
        self.assertTrue(result.loc[dates[2]].isna().all())


class FactorSelectionLayerTest(unittest.TestCase):
    def test_lagged_factor_is_ranked_before_future_returns_are_used(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        tickers = [f"T{number:02d}" for number in range(30)]
        factor = pd.DataFrame(
            [np.arange(30), np.arange(30) + 100],
            index=dates,
            columns=tickers,
            dtype=float,
        )
        membership = pd.DataFrame(
            True,
            index=dates,
            columns=tickers,
        )
        forward_returns = pd.DataFrame(
            [np.arange(30), np.arange(30)],
            index=dates,
            columns=tickers,
            dtype=float,
        )

        quantiles = prepare_factor_quantiles(factor, membership)
        means, counts, signal_count, return_count = (
            aggregate_quantile_returns(quantiles, forward_returns)
        )

        self.assertEqual(signal_count[0], 0)
        self.assertEqual(return_count[0], 0)
        self.assertTrue(np.isnan(means[0]).all())
        np.testing.assert_array_equal(counts[1], np.repeat(6, 5))
        self.assertEqual(signal_count[1], 30)
        self.assertEqual(return_count[1], 30)
        self.assertAlmostEqual(means[1, 0], 2.5)
        self.assertAlmostEqual(means[1, 4], 26.5)


if __name__ == "__main__":
    unittest.main()
