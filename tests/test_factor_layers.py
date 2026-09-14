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
    aggregate_quantile_relationships,
    compute_daily_relationship_metrics,
    prepare_factor_quantiles,
)
from hypothesis_analysis import daily_effects
from pattern_classification import classify_pattern


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

        signal, quantiles = prepare_factor_quantiles(factor, membership)
        statistics = aggregate_quantile_relationships(
            signal,
            quantiles,
            forward_returns,
        )
        relationships = compute_daily_relationship_metrics(
            signal,
            forward_returns,
        )

        self.assertEqual(statistics["signal_asset_count"][0], 0)
        self.assertEqual(statistics["return_asset_count"][0], 0)
        self.assertTrue(np.isnan(statistics["return_means"][0]).all())
        np.testing.assert_array_equal(
            statistics["counts"][1],
            np.repeat(3, 10),
        )
        self.assertEqual(statistics["signal_asset_count"][1], 30)
        self.assertEqual(statistics["return_asset_count"][1], 30)
        self.assertAlmostEqual(statistics["signal_means"][1, 0], 1.0)
        self.assertAlmostEqual(statistics["signal_means"][1, 9], 28.0)
        self.assertAlmostEqual(statistics["signal_medians"][1, 0], 1.0)
        self.assertAlmostEqual(statistics["return_means"][1, 0], 1.0)
        self.assertAlmostEqual(statistics["return_means"][1, 9], 28.0)
        self.assertAlmostEqual(statistics["return_medians"][1, 9], 28.0)
        self.assertAlmostEqual(relationships["spearman_ic"][1], 1.0)
        self.assertAlmostEqual(
            relationships["pearson_correlation"][1],
            1.0,
        )
        self.assertAlmostEqual(relationships["factor_beta"][1], 1.0)

    def test_daily_effects_keep_monotonic_and_tail_contrasts_separate(self):
        frame = pd.DataFrame(
            {
                "spearman_ic": [0.2],
                **{
                    f"q{quantile}_return_mean": [quantile / 100]
                    for quantile in range(1, 11)
                },
            }
        )

        effects = daily_effects(frame)

        self.assertAlmostEqual(effects["q10_minus_q1"].iloc[0], 0.09)
        self.assertAlmostEqual(
            effects["q10_minus_middle"].iloc[0],
            0.045,
        )
        self.assertAlmostEqual(
            effects["middle_minus_q1"].iloc[0],
            0.045,
        )
        self.assertAlmostEqual(
            effects["edges_minus_middle"].iloc[0],
            0.0,
        )

    def test_pattern_classification_accepts_negative_monotonic_shape(self):
        row = pd.Series(
            {
                "quantile_curve_rho": -0.95,
                "quantile_step_ratio": 0.80,
                "q10_minus_q1_mean": -0.03,
                "q10_minus_q1_hac_tstat": -3.0,
                "q10_minus_middle_mean": -0.02,
                "q10_minus_middle_hac_tstat": -2.5,
                "middle_minus_q1_mean": -0.01,
                "middle_minus_q1_hac_tstat": -2.0,
            }
        )

        pattern, direction, primary = classify_pattern(row)

        self.assertEqual(pattern, "negative_monotonic")
        self.assertEqual(direction, "negative")
        self.assertEqual(primary, "q10_minus_q1")


if __name__ == "__main__":
    unittest.main()
