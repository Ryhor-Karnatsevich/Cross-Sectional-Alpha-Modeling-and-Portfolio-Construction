import os
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "Factors_Layer"))

from trend_slope_conditional_research import (
    apply_position_capacity,
    build_condition_exposures,
    conditional_ic_summary,
    estimate_selected_spread_volatility,
    select_baskets,
)
from portfolio_implementation_research import estimate_spread_volatility


class TrendSlopeConditionalTests(unittest.TestCase):
    def test_condition_exposures_use_only_declared_rules(self):
        states = pd.DataFrame(
            {
                "low_correlation": [True, True, False],
                "high_dispersion": [True, False, True],
                "exposure_binary_opportunity": [1.0, 0.0, 1.0],
            }
        )

        result = build_condition_exposures(states)

        self.assertEqual(result["always_active"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(result["low_correlation"].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(
            result["low_corr_high_dispersion"].tolist(),
            [1.0, 0.0, 0.0],
        )
        self.assertEqual(result["binary_opportunity"].tolist(), [1.0, 0.0, 1.0])

    def test_quantile_buffer_retains_only_names_inside_exit_band(self):
        ranks = pd.Series(
            {
                "A": 1.0,
                "B": 0.9,
                "C": 0.8,
                "D": 0.7,
                "E": 0.6,
                "F": 0.5,
                "G": 0.4,
                "H": 0.3,
                "I": 0.2,
                "J": 0.1,
            }
        )

        long_names, short_names = select_baskets(
            ranks,
            previous_long=["C", "D"],
            previous_short=["G", "H"],
            entry_quantile=0.20,
            exit_quantile=0.30,
            buffered=True,
        )

        self.assertEqual(long_names, ["A", "B", "C", "D"])
        self.assertEqual(short_names, ["H", "I", "J"])

    def test_position_capacity_preserves_leg_ratio(self):
        long_exposure, short_exposure = apply_position_capacity(
            long_exposure=1.2,
            short_exposure=0.8,
            long_assets=50,
            short_assets=50,
        )

        self.assertAlmostEqual(long_exposure, 1.0)
        self.assertAlmostEqual(short_exposure, 2 / 3)
        self.assertAlmostEqual(long_exposure / short_exposure, 1.5)

    def test_optimized_volatility_matches_full_matrix_calculation(self):
        index = pd.bdate_range("2020-01-01", periods=70)
        values = np.arange(210, dtype=float).reshape(70, 3) / 100_000
        returns = pd.DataFrame(values, index=index, columns=["A", "B", "C"])
        weights = pd.Series({"A": 0.6, "C": -0.4})

        optimized = estimate_selected_spread_volatility(
            returns,
            index[-1],
            weights,
        )
        original = estimate_spread_volatility(returns, index[-1], weights)

        self.assertAlmostEqual(optimized, original)

    def test_conditional_ic_reports_arithmetic_mean_not_standard_error(self):
        index = pd.bdate_range("2020-01-01", periods=10)
        ic = pd.Series([0.01, 0.03] * 5, index=index)
        exposures = pd.DataFrame(
            {
                "always_active": 1.0,
                "low_correlation": [1.0] * 5 + [0.0] * 5,
                "low_corr_high_dispersion": [1.0] * 4 + [0.0] * 6,
                "binary_opportunity": [0.0] * 4 + [1.0] * 6,
            },
            index=index,
        )

        result = conditional_ic_summary(ic, exposures)
        always = result[result["condition"] == "always_active"].iloc[0]

        self.assertAlmostEqual(always["conditional_mean_ic"], ic.mean())
        self.assertNotAlmostEqual(
            always["conditional_mean_ic"],
            always["conditional_hac_standard_error"],
        )


if __name__ == "__main__":
    unittest.main()
