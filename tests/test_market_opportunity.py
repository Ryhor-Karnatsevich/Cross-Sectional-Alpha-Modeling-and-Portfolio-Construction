import os
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "Factors_Layer"))

from market_opportunity_research import (
    HORIZON,
    absolute_turnover,
    apply_thresholds,
    compound_path,
    training_thresholds,
)


class MarketOpportunityTests(unittest.TestCase):
    def test_training_thresholds_purge_forward_horizon(self):
        index = pd.bdate_range("2010-01-01", "2015-12-31")
        indicators = pd.DataFrame(
            {
                "risk_free_rate_pct": 3.0,
                "median_stock_volatility": np.arange(len(index), dtype=float),
                "cross_sectional_dispersion": np.arange(len(index), dtype=float),
                "average_correlation_proxy": np.arange(len(index), dtype=float),
            },
            index=index,
        )

        result = training_thresholds(indicators, index, 2015)
        selection_position = index.searchsorted("2014-12-31", side="right") - 1

        self.assertEqual(
            pd.Timestamp(result["purged_train_end"]),
            index[selection_position - HORIZON],
        )
        self.assertLess(result["purged_train_end"], pd.Timestamp("2014-12-31"))

    def test_opportunity_score_and_exposure_rules(self):
        indicators = pd.DataFrame(
            {
                "risk_free_rate_pct": [3.0, 1.0, 3.0],
                "median_stock_volatility": [0.25, 0.25, 0.50],
                "cross_sectional_dispersion": [0.30, 0.10, 0.30],
                "average_correlation_proxy": [0.10, 0.40, 0.10],
            },
            index=pd.bdate_range("2020-01-01", periods=3),
        )
        thresholds = {
            "risk_free_rate_threshold_pct": 2.0,
            "volatility_lower": 0.20,
            "volatility_upper": 0.40,
            "dispersion_threshold": 0.20,
            "correlation_threshold": 0.20,
        }

        _, score, exposures = apply_thresholds(indicators, thresholds)

        self.assertEqual(score.tolist(), [4.0, 1.0, 3.0])
        self.assertEqual(exposures["binary_opportunity"].tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(exposures["tiered_opportunity"].tolist(), [1.0, 0.0, 1.0])

    def test_absolute_turnover_handles_exposure_change(self):
        weights = pd.Series({"A": 0.5, "B": 0.5})

        self.assertAlmostEqual(
            absolute_turnover(weights, 1.0, weights, 0.5),
            0.5,
        )
        replacement = pd.Series({"A": 0.5, "C": 0.5})
        self.assertAlmostEqual(
            absolute_turnover(weights, 1.0, replacement, 1.0),
            1.0,
        )

    def test_transaction_costs_reduce_compounded_return(self):
        path = pd.DataFrame(
            {
                "spread_net_0bps_return": [0.02, 0.01],
                "spread_net_10bps_return": [0.018, 0.008],
                "spread_net_25bps_return": [0.015, 0.005],
                "exposure": [1.0, 1.0],
            }
        )

        gross = compound_path(path, 0)
        net_10 = compound_path(path, 10)
        net_25 = compound_path(path, 25)

        self.assertGreater(gross, net_10)
        self.assertGreater(net_10, net_25)


if __name__ == "__main__":
    unittest.main()
