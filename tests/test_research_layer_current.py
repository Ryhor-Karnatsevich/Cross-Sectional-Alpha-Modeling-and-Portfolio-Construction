import os
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESEARCH_LAYER_PATH = os.path.join(PROJECT_ROOT, "src", "Research_Layer")
if RESEARCH_LAYER_PATH not in sys.path:
    sys.path.insert(0, RESEARCH_LAYER_PATH)

from multiple_testing_research import bh_values, simes_p_value
from portfolio_construction import build_target_weights
from portfolio_evaluation import apply_transaction_costs
from portfolio_engine import (
    MATRIX_NAMES,
    calendar_offsets,
    simulate_factor_paths,
    simulate_path,
)
from research_report import build_research_report
from risk_exposure import build_market_regimes
from walk_forward_research import oriented_net_returns, run_walk_forward


class MultipleTestingResearchTests(unittest.TestCase):
    def test_simes_uses_ordered_family_p_values(self):
        self.assertAlmostEqual(simes_p_value([0.01, 0.04, 0.50]), 0.03)

    def test_bh_keeps_original_index(self):
        values = pd.Series([0.001, 0.20, np.nan], index=[5, 8, 13])
        result = bh_values(values)
        self.assertEqual(result.index.tolist(), [5, 8, 13])
        self.assertTrue(bool(result.loc[5, "rejected"]))
        self.assertFalse(bool(result.loc[13, "rejected"]))


class PortfolioConstructionTests(unittest.TestCase):
    def setUp(self):
        self.scores = pd.Series(
            np.arange(100, dtype=float),
            index=[f"S{number:03d}" for number in range(100)],
        )
        self.eligible = pd.Series(True, index=self.scores.index)
        self.betas = pd.Series(1.0, index=self.scores.index)

    def test_q10_minus_q1_is_dollar_neutral(self):
        weights, _ = build_target_weights(
            self.scores,
            self.eligible,
            "q10_minus_q1",
            self.betas,
            False,
        )
        self.assertAlmostEqual(weights.sum(), 0.0)
        self.assertAlmostEqual(weights.abs().sum(), 2.0)
        self.assertEqual((weights > 0).sum(), 10)
        self.assertEqual((weights < 0).sum(), 10)

    def test_continuous_weights_use_the_whole_cross_section(self):
        weights, _ = build_target_weights(
            self.scores,
            self.eligible,
            "continuous_high_minus_low",
            self.betas,
            False,
        )
        self.assertAlmostEqual(weights.sum(), 0.0)
        self.assertAlmostEqual(weights.abs().sum(), 2.0)
        self.assertGreater((weights != 0).sum(), 90)

    def test_beta_neutralization_drops_unknown_betas(self):
        betas = self.betas.copy()
        betas.iloc[0] = np.nan
        weights, applied = build_target_weights(
            self.scores,
            self.eligible,
            "q10_minus_q1",
            betas,
            True,
        )
        self.assertTrue(applied)
        self.assertNotIn(self.scores.index[0], weights.index)
        self.assertAlmostEqual((weights * betas.reindex(weights.index)).sum(), 0.0)


class PortfolioMechanicsTests(unittest.TestCase):
    def test_transaction_costs_are_charged_on_turnover(self):
        gross = pd.Series([0.01, 0.01])
        turnover = pd.Series([1.0, 0.0])
        net = apply_transaction_costs(gross, turnover, 10)
        self.assertAlmostEqual(net.iloc[0], (1.01 * 0.999) - 1)
        self.assertAlmostEqual(net.iloc[1], 0.01)

    def test_reversing_a_spread_does_not_reverse_costs(self):
        gross = pd.Series([0.02])
        turnover = pd.Series([1.0])
        positive = oriented_net_returns(gross, turnover, 1).iloc[0]
        negative = oriented_net_returns(gross, turnover, -1).iloc[0]
        self.assertLess(negative, -0.02)
        self.assertLess(positive, 0.02)

    def test_calendar_phases_are_evenly_spread(self):
        self.assertEqual(calendar_offsets(1), (0,))
        self.assertEqual(calendar_offsets(5), (0, 2, 4))
        self.assertEqual(calendar_offsets(21), (0, 10, 20))

    def test_batched_engine_matches_single_path_engine(self):
        dates = pd.bdate_range("2020-01-01", periods=40)
        tickers = [f"S{number:02d}" for number in range(40)]
        factor = pd.DataFrame(
            np.tile(np.arange(40, dtype=float), (40, 1)),
            index=dates,
            columns=tickers,
        )
        returns = pd.DataFrame(
            np.tile(np.linspace(-0.01, 0.01, 40), (40, 1)),
            index=dates,
            columns=tickers,
        )
        membership = pd.DataFrame(True, index=dates, columns=tickers)
        availability = membership.copy()
        betas = pd.DataFrame(1.0, index=dates, columns=tickers)
        specifications = pd.DataFrame(
            [
                {
                    "path_key": "test",
                    "method": "q10_minus_q1",
                    "beta_neutral": False,
                    "rebalance_days": 5,
                    "calendar_offset": 0,
                }
            ]
        )
        single, _ = simulate_path(
            factor,
            returns,
            membership,
            availability,
            betas,
            None,
            "q10_minus_q1",
            False,
            5,
            0,
        )
        batched, _ = simulate_factor_paths(
            factor,
            specifications,
            returns,
            membership,
            availability,
            betas,
        )
        for name in MATRIX_NAMES:
            np.testing.assert_allclose(
                single[name].to_numpy(),
                batched[name]["test"].to_numpy(),
            )

    def test_previous_day_factor_earns_current_daily_return(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        tickers = [f"S{number:03d}" for number in range(100)]
        factor = pd.DataFrame(
            np.tile(np.arange(100, dtype=float), (3, 1)),
            index=dates,
            columns=tickers,
        )
        returns = pd.DataFrame(0.0, index=dates, columns=tickers)
        returns.loc[dates[1], tickers[:10]] = -0.01
        returns.loc[dates[1], tickers[-10:]] = 0.01
        eligible = pd.DataFrame(True, index=dates, columns=tickers)
        betas = pd.DataFrame(1.0, index=dates, columns=tickers)
        output, _ = simulate_path(
            factor,
            returns,
            eligible,
            eligible,
            betas,
            None,
            "q10_minus_q1",
            False,
            1,
            0,
        )
        self.assertAlmostEqual(output["gross_returns"].loc[dates[1]], 0.02)

    def test_market_regime_uses_previous_known_risk_free_rate(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        returns = pd.DataFrame(
            {"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]},
            index=dates,
        )
        eligible = pd.DataFrame(True, index=dates, columns=returns.columns)
        market = returns.mean(axis=1)
        risk_free = pd.DataFrame(
            {
                "annual_rate_pct": [1.0, 3.0, 3.0],
                "daily_rate": [0.0, 0.0, 0.0],
            },
            index=dates,
        )
        regimes = build_market_regimes(
            returns,
            eligible,
            eligible,
            market,
            risk_free,
        )
        self.assertEqual(regimes.loc[dates[0], "risk_free_state"], "unavailable")
        self.assertEqual(regimes.loc[dates[1], "risk_free_state"], "at_or_below_2pct")
        self.assertEqual(regimes.loc[dates[2], "risk_free_state"], "above_2pct")

    def test_walk_forward_keeps_both_period_schemes(self):
        dates = pd.bdate_range("2010-01-01", "2018-12-31")
        gross = pd.DataFrame({"path": 0.0005}, index=dates)
        turnover = pd.DataFrame({"path": 0.0}, index=dates)
        zeros = pd.DataFrame({"path": 0.0}, index=dates)
        matrices = {
            "gross_returns": gross,
            "turnover": turnover,
            "beta_exposure": zeros,
            "gross_exposure": zeros,
            "net_exposure": zeros,
            "unpriced_weight": zeros,
            "max_sector_exposure": zeros,
            "unknown_sector_weight": zeros,
            "max_position": zeros,
            "holding_count": zeros,
        }
        metadata = pd.DataFrame(
            [
                {
                    "path_key": "path",
                    "method": "q10_minus_q1",
                    "portfolio_type": "market_neutral",
                }
            ]
        )
        risk_free = pd.Series(0.0, index=dates)
        periods, summary, paths = run_walk_forward(
            matrices,
            metadata,
            risk_free,
        )
        self.assertEqual(
            set(summary["walk_forward_scheme"]),
            {"short", "long"},
        )
        self.assertEqual(paths.shape[1], 2)
        self.assertTrue(periods["complete_oos_period"].any())


class ResearchReportTests(unittest.TestCase):
    def test_report_uses_portfolio_and_walk_forward_tables_separately(self):
        candidates = pd.DataFrame(
            [{
                "factor_key": "factor|variant",
                "selection_horizon": 5,
                "selection_evidence": "lead",
                "selection_pattern": "positive_monotonic",
                "selection_status": "exploratory",
            }]
        )
        multiple_testing = pd.DataFrame(
            [{
                "method": "global_bh",
                "rejections": 1,
                "economic_rejections": 0,
                "ic_rejections": 1,
                "interpretation": "confirmatory",
            }]
        )
        statistics = pd.DataFrame(
            [{
                "path_key": "path",
                "factor_key": "factor|variant",
                "method": "q10_minus_q1",
                "beta_neutral": False,
                "rebalance_days": 21,
                "transaction_cost_bps": 10,
                "annualized_return": 0.01,
                "sharpe": 0.1,
                "maximum_drawdown": -0.1,
                "annualized_turnover": 2.0,
                "alpha_hac_tstat": 0.2,
            }]
        )
        phase_stability = pd.DataFrame(
            [{
                "path_key": "path",
                "phase_count": 3,
                "phase_sharpe_mean": 0.1,
                "phase_sharpe_min": 0.0,
                "phase_sharpe_max": 0.2,
                "phase_sharpe_std": 0.1,
            }]
        )
        walk_forward = pd.DataFrame(
            [{
                "walk_key": "short|path",
                "factor_key": "factor|variant",
                "method": "q10_minus_q1",
                "beta_neutral": False,
                "rebalance_days": 21,
                "complete_oos_periods": 2,
                "positive_oos_period_rate": 0.5,
                "stitched_oos_annualized_return": 0.01,
                "stitched_oos_sharpe": 0.1,
                "orientation_changes": 0,
            }]
        )
        risk = pd.DataFrame(
            [{
                "factor_key": "factor|variant",
                "method": "q10_minus_q1",
                "beta_neutral": False,
                "rebalance_days": 21,
                "average_estimated_beta": 0.0,
                "maximum_absolute_estimated_beta": 0.1,
                "average_gross_exposure": 2.0,
                "average_net_exposure": 0.0,
                "maximum_position_weight": 0.1,
                "maximum_unpriced_weight": 0.0,
            }]
        )
        sector_status = pd.DataFrame(
            [{"status": "NOT_AVAILABLE", "reason": "No PIT sectors."}]
        )
        overlap = pd.DataFrame(
            columns=[
                "factor_left",
                "factor_right",
                "mean_daily_rank_correlation",
                "mean_absolute_daily_rank_correlation",
                "high_absolute_overlap_rate",
            ]
        )
        report = build_research_report(
            candidates,
            multiple_testing,
            statistics,
            phase_stability,
            walk_forward,
            risk,
            sector_status,
            pd.DataFrame(),
            overlap,
            [],
        )
        self.assertIn("Full-History Portfolio Implementations", report)
        self.assertIn("Walk-Forward Behaviour", report)


if __name__ == "__main__":
    unittest.main()
