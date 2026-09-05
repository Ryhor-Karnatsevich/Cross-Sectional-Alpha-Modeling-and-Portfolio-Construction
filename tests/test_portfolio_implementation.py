import os
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "Factors_Layer"))

from portfolio_implementation_research import (
    MAX_ABSOLUTE_POSITION,
    beta_neutral_leg_exposures,
    build_trailing_betas,
    capped_equal_weights,
    liquidity_cost,
    select_buffered_baskets,
    volatility_scale,
)


class PortfolioImplementationTests(unittest.TestCase):
    def test_buffer_retains_names_inside_exit_band(self):
        scores = pd.Series(
            {
                "A": 10.0,
                "B": 9.0,
                "C": 8.0,
                "D": 7.0,
                "E": 6.0,
                "F": 5.0,
                "G": 4.0,
                "H": 3.0,
                "I": 2.0,
                "J": 1.0,
            }
        )
        eligible = pd.Series(True, index=scores.index)

        long_names, short_names, _ = select_buffered_baskets(
            scores,
            eligible,
            previous_long=["C"],
            previous_short=["H"],
            buffered=True,
        )

        self.assertEqual(long_names, ["A", "B", "C"])
        self.assertEqual(short_names, ["H", "I", "J"])

    def test_beta_neutral_leg_scaling(self):
        long = pd.Series({"A": 0.5, "B": 0.5})
        short = pd.Series({"C": 0.5, "D": 0.5})
        beta = pd.Series({"A": 1.2, "B": 1.2, "C": 0.8, "D": 0.8})

        long_exposure, short_exposure = beta_neutral_leg_exposures(
            long,
            short,
            beta,
            gross_exposure=2.0,
        )

        self.assertAlmostEqual(long_exposure + short_exposure, 2.0)
        self.assertAlmostEqual(long_exposure * 1.2, short_exposure * 0.8)

    def test_beta_market_proxy_uses_point_in_time_membership(self):
        index = pd.bdate_range("2020-01-01", periods=140)
        member_returns = pd.Series(
            np.sin(np.arange(len(index)) / 7) / 100,
            index=index,
        )
        returns = pd.DataFrame(
            {
                "MEMBER": member_returns,
                "OUTSIDER": member_returns * -4,
            }
        )
        availability = pd.DataFrame(True, index=index, columns=returns.columns)
        membership = pd.DataFrame(
            {"MEMBER": True, "OUTSIDER": False},
            index=index,
        )

        betas = build_trailing_betas(returns, availability, membership)

        self.assertAlmostEqual(betas.iloc[-1]["MEMBER"], 1.0)
        self.assertAlmostEqual(betas.iloc[-1]["OUTSIDER"], -4.0)

    def test_position_cap_and_total_exposure(self):
        tickers = [f"S{i}" for i in range(100)]
        weights = capped_equal_weights(tickers, total_exposure=1.2)

        self.assertAlmostEqual(weights.sum(), 1.2)
        self.assertLessEqual(weights.abs().max(), MAX_ABSOLUTE_POSITION)

    def test_volatility_target_only_reduces_exposure(self):
        self.assertEqual(volatility_scale(0.05), 1.0)
        self.assertAlmostEqual(volatility_scale(0.20), 0.5)
        self.assertEqual(volatility_scale(1.0), 0.25)

    def test_liquidity_cost_increases_for_less_liquid_assets(self):
        changes = pd.Series({"A": 0.5})
        liquid = liquidity_cost(changes, pd.Series({"A": 100_000_000}))
        illiquid = liquidity_cost(changes, pd.Series({"A": 1_000_000}))

        self.assertGreater(illiquid, liquid)
        self.assertGreater(liquid, 0)


if __name__ == "__main__":
    unittest.main()
