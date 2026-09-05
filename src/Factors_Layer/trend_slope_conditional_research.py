"""Focused robustness study of the frozen conditional Trend Slope hypothesis.

This is the final bounded research extension. Trend Slope was identified after
the development sample had already been inspected, so even a positive result
here is not independent validation. Signal selection, market conditions and
portfolio variants are declared below and are not selected from this run.
"""

import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from composite_alpha_research import (
    FIRST_OOS_YEAR,
    factor_cache_path,
    score_training_candidates,
)
from market_opportunity_research import (
    OUTPUT_DIR as OPPORTUNITY_OUTPUT_DIR,
    state_difference_test,
)
from pipeline import load_data, load_membership
from portfolio_implementation_research import (
    FIXED_COSTS_BPS,
    MAX_ABSOLUTE_POSITION,
    MAX_GROSS_EXPOSURE,
    VOLATILITY_LOOKBACK,
    beta_neutral_leg_exposures,
    build_trailing_betas,
    capped_equal_weights,
    compound_annual,
    drift_absolute_weights,
    liquidity_cost,
    realized_returns,
    volatility_scale,
    weight_changes,
)
from statistical_research import hac_mean_test
from walk_forward import IC_CACHE_PATH, METADATA_PATH


data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data_System")
)
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import BASE_DIR, LIQUIDITY_PATH


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "Factor_Research",
    "trend_slope_conditional_stage",
)

SIGNAL_FAMILY = "trend_slope"
PRIMARY_CONDITION = "low_correlation"
PRIMARY_PORTFOLIO = "controlled_quintile"
PRIMARY_HOLDING_PERIOD = 21
HOLDING_PERIODS = (21, 42)
CALENDAR_PHASES = 21
MIN_ASSETS = 50

CONDITIONS = (
    "always_active",
    "low_correlation",
    "low_corr_high_dispersion",
    "binary_opportunity",
)

PORTFOLIOS = {
    "equal_quintile": {
        "entry_quantile": 0.20,
        "exit_quantile": 0.20,
        "buffered": False,
        "beta_neutral": False,
        "risk_controlled": False,
    },
    "controlled_quintile": {
        "entry_quantile": 0.20,
        "exit_quantile": 0.30,
        "buffered": True,
        "beta_neutral": True,
        "risk_controlled": True,
    },
    "controlled_decile": {
        "entry_quantile": 0.10,
        "exit_quantile": 0.20,
        "buffered": True,
        "beta_neutral": True,
        "risk_controlled": True,
    },
}

# Seven declared comparisons instead of a redundant full Cartesian grid:
# four market conditions, three implementations, and one horizon sensitivity.
EXPERIMENTS = (
    ("always_active", "controlled_quintile", 21),
    ("low_correlation", "controlled_quintile", 21),
    ("low_corr_high_dispersion", "controlled_quintile", 21),
    ("binary_opportunity", "controlled_quintile", 21),
    ("low_correlation", "equal_quintile", 21),
    ("low_correlation", "controlled_decile", 21),
    ("low_correlation", "controlled_quintile", 42),
)


def build_condition_exposures(states):
    """Build only the four market rules declared before this run."""
    exposures = pd.DataFrame(index=states.index)
    exposures["always_active"] = 1.0
    exposures["low_correlation"] = states["low_correlation"].astype(float)
    exposures["low_corr_high_dispersion"] = (
        states["low_correlation"] & states["high_dispersion"]
    ).astype(float)
    exposures["binary_opportunity"] = states[
        "exposure_binary_opportunity"
    ].astype(float)
    return exposures.fillna(0.0)


def select_baskets(
    ranks,
    previous_long,
    previous_short,
    entry_quantile,
    exit_quantile,
    buffered,
):
    """Select tails and optionally retain names inside an exit buffer."""
    ranks = ranks.dropna()
    long_entries = set(ranks.index[ranks >= 1 - entry_quantile])
    short_entries = set(ranks.index[ranks <= entry_quantile])

    if not buffered:
        return sorted(long_entries), sorted(short_entries)

    retained_long = {
        ticker
        for ticker in previous_long
        if ticker in ranks.index and ranks[ticker] >= 1 - exit_quantile
    }
    retained_short = {
        ticker
        for ticker in previous_short
        if ticker in ranks.index and ranks[ticker] <= exit_quantile
    }
    return (
        sorted(long_entries | retained_long),
        sorted(short_entries | retained_short),
    )


def apply_position_capacity(
    long_exposure,
    short_exposure,
    long_assets,
    short_assets,
):
    """De-risk both legs equally when a 2% position cap makes gross infeasible."""
    scales = [1.0]
    if long_exposure > 0:
        scales.append(long_assets * MAX_ABSOLUTE_POSITION / long_exposure)
    if short_exposure > 0:
        scales.append(short_assets * MAX_ABSOLUTE_POSITION / short_exposure)
    scale = min(scales)
    return long_exposure * scale, short_exposure * scale


def estimate_selected_spread_volatility(returns, date, signed_weights):
    """Estimate the same trailing volatility using only selected columns."""
    position = returns.index.get_loc(date)
    start = max(0, position - VOLATILITY_LOOKBACK)
    window = returns.iloc[start:position].reindex(columns=signed_weights.index)
    if len(window) < 40:
        return np.nan
    portfolio_returns = window.fillna(0).dot(signed_weights)
    return float(portfolio_returns.std() * np.sqrt(252))


def build_year_ranks(factor, prices, membership, year):
    """Create t-1 signal ranks for one OOS calendar year."""
    year_dates = prices.loc[f"{year}-01-01":f"{year}-12-31"].index
    if year_dates.empty:
        raise ValueError(f"No price dates for OOS year {year}")

    first_position = prices.index.get_loc(year_dates[0])
    slice_start = max(0, first_position - 1)
    date_slice = prices.index[slice_start : prices.index.get_loc(year_dates[-1]) + 1]
    sliced_prices = prices.loc[date_slice]
    sliced_membership = membership.loc[date_slice]
    signal = factor.loc[date_slice].shift(1)
    eligible_signal = signal.where(
        sliced_membership
        & sliced_prices.notna()
        & (sliced_prices > 0)
    )
    ranks = eligible_signal.rank(axis=1, method="first", pct=True)
    return ranks, sliced_prices, sliced_membership, 1, len(date_slice) - 1


def simulate_trend_path(
    ranks,
    prices,
    returns,
    betas,
    average_dollar_volume,
    exposure,
    holding_period,
    offset,
    portfolio,
    start_position,
    end_position,
):
    """Simulate one annual calendar phase of the conditional Trend strategy."""
    settings = PORTFOLIOS[portfolio]
    previous_long = pd.Series(dtype=float)
    previous_short = pd.Series(dtype=float)
    previous_active = False
    rows = []

    for position in range(
        start_position + offset,
        end_position + 1,
        holding_period,
    ):
        actual_horizon = min(holding_period, end_position - position)
        if actual_horizon <= 0 or position + actual_horizon >= len(prices):
            break

        date = prices.index[position]
        active = bool(exposure.reindex([date]).fillna(0).iloc[0] > 0)
        current_ranks = ranks.iloc[position].dropna()
        if len(current_ranks) < MIN_ASSETS:
            continue

        previous_long_names = previous_long.index if previous_active else []
        previous_short_names = previous_short.index if previous_active else []
        long_names, short_names = select_baskets(
            current_ranks,
            previous_long_names,
            previous_short_names,
            settings["entry_quantile"],
            settings["exit_quantile"],
            settings["buffered"],
        )
        unit_long = pd.Series(1 / len(long_names), index=long_names)
        unit_short = pd.Series(1 / len(short_names), index=short_names)

        beta = betas.loc[date]
        beta = beta.fillna(beta.median()).fillna(1.0)
        unit_signed = pd.concat([unit_long, -unit_short]).groupby(level=0).sum()
        estimated_volatility = (
            estimate_selected_spread_volatility(
                returns,
                date,
                unit_signed,
            )
            if settings["risk_controlled"]
            else np.nan
        )
        risk_scale = (
            volatility_scale(estimated_volatility)
            if settings["risk_controlled"]
            else 1.0
        )
        gross_exposure = MAX_GROSS_EXPOSURE * risk_scale if active else 0.0

        if settings["beta_neutral"] and active:
            long_exposure, short_exposure = beta_neutral_leg_exposures(
                unit_long,
                unit_short,
                beta,
                gross_exposure,
            )
        else:
            long_exposure = gross_exposure / 2
            short_exposure = gross_exposure / 2

        long_exposure, short_exposure = apply_position_capacity(
            long_exposure,
            short_exposure,
            len(long_names),
            len(short_names),
        )

        target_long = capped_equal_weights(long_names, long_exposure)
        target_short = capped_equal_weights(short_names, short_exposure)
        long_changes = weight_changes(previous_long, target_long)
        short_changes = weight_changes(previous_short, target_short)
        dollar_turnover = float(long_changes.sum() + short_changes.sum())
        adv = average_dollar_volume.loc[date]
        modeled_cost = liquidity_cost(long_changes, adv) + liquidity_cost(
            short_changes,
            adv,
        )

        long_asset_returns = realized_returns(
            prices,
            position,
            actual_horizon,
            long_names,
        )
        short_asset_returns = realized_returns(
            prices,
            position,
            actual_horizon,
            short_names,
        )
        unit_long_return = float((unit_long * long_asset_returns).sum())
        unit_short_return = float((unit_short * short_asset_returns).sum())
        long_contribution = float((target_long * long_asset_returns).sum())
        short_contribution = -float((target_short * short_asset_returns).sum())

        if not previous_active and active:
            transition = "entry"
        elif previous_active and not active:
            transition = "exit"
        elif active:
            transition = "stay_active"
        else:
            transition = "stay_inactive"

        row = {
            "date": date,
            "end_date": prices.index[position + actual_horizon],
            "holding_days": actual_horizon,
            "active": active,
            "transition": transition,
            "long_assets": len(long_names),
            "short_assets": len(short_names),
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "gross_exposure": long_exposure + short_exposure,
            "net_exposure": long_exposure - short_exposure,
            "estimated_beta": float(
                (target_long * beta.reindex(target_long.index)).sum()
                - (target_short * beta.reindex(target_short.index)).sum()
            ),
            "estimated_spread_volatility": estimated_volatility,
            "volatility_scale": risk_scale,
            "underlying_spread_return": unit_long_return - unit_short_return,
            "inactive_foregone_return": (
                unit_long_return - unit_short_return if not active else 0.0
            ),
            "long_contribution": long_contribution,
            "short_contribution": short_contribution,
            "spread_gross_return": long_contribution + short_contribution,
            "dollar_turnover": dollar_turnover,
            "modeled_liquidity_cost": modeled_cost,
        }
        for cost_bps in FIXED_COSTS_BPS:
            row[f"fixed_{cost_bps}bps_cost"] = (
                dollar_turnover * cost_bps / 10_000
            )
        rows.append(row)

        if active:
            previous_long = drift_absolute_weights(
                target_long,
                long_asset_returns,
            )
            previous_short = drift_absolute_weights(
                target_short,
                short_asset_returns,
            )
        else:
            previous_long = pd.Series(dtype=float)
            previous_short = pd.Series(dtype=float)
        previous_active = active

    return pd.DataFrame(rows), previous_long, previous_short


def conditional_ic_summary(ic_history, exposures):
    """Test the frozen Trend IC inside each declared market condition."""
    rows = []
    for condition in CONDITIONS:
        state = exposures[condition].astype(bool)
        aligned = pd.concat([ic_history.rename("ic"), state.rename("state")], axis=1)
        aligned = aligned.dropna()
        active_ic = aligned.loc[aligned["state"], "ic"]
        standard_error, tstat, pvalue = hac_mean_test(
            active_ic,
            PRIMARY_HOLDING_PERIOD - 1,
        )
        difference = (
            state_difference_test(aligned["ic"], aligned["state"])
            if condition != "always_active"
            else {
                "state_0_mean_ic": np.nan,
                "difference": np.nan,
                "hac_tstat": np.nan,
                "pvalue": np.nan,
            }
        )
        rows.append(
            {
                "condition": condition,
                "observations": len(active_ic),
                "active_rate": aligned["state"].mean(),
                "conditional_mean_ic": active_ic.mean(),
                "conditional_hac_standard_error": standard_error,
                "conditional_hac_tstat": tstat,
                "conditional_pvalue": pvalue,
                "inactive_mean_ic": difference["state_0_mean_ic"],
                "difference": difference["difference"],
                "difference_hac_tstat": difference["hac_tstat"],
                "difference_pvalue": difference["pvalue"],
            }
        )

    result = pd.DataFrame(rows)
    for source, target in (
        ("conditional_pvalue", "conditional_pvalue_fdr_bh"),
        ("difference_pvalue", "difference_pvalue_fdr_bh"),
    ):
        result[target] = np.nan
        valid = result[source].notna()
        if valid.any():
            result.loc[valid, target] = multipletests(
                result.loc[valid, source],
                method="fdr_bh",
            )[1]
    return result


def summarize_grid(annual, final_data_year):
    """Calculate phase-level annual-return evidence and multiple-test control."""
    complete = annual[annual["oos_year"] < final_data_year]
    rows = []
    return_columns = (
        "gross_return",
        "net_fixed_10bps_return",
        "net_fixed_25bps_return",
        "net_liquidity_cost_return",
    )

    keys = ["condition", "portfolio", "holding_period", "offset"]
    for values, data in complete.groupby(keys):
        row = dict(zip(keys, values))
        row.update(
            {
                "complete_oos_years": len(data),
                "active_period_rate": data["active_period_rate"].mean(),
                "average_turnover": data["average_turnover"].mean(),
                "average_gross_exposure": data["average_gross_exposure"].mean(),
                "average_abs_beta": data["average_abs_beta"].mean(),
            }
        )
        for column in return_columns:
            returns = data[column]
            test = ttest_1samp(returns, 0)
            row[f"{column}_mean"] = returns.mean()
            row[f"{column}_volatility"] = returns.std()
            row[f"{column}_sharpe"] = (
                returns.mean() / returns.std() if returns.std() else np.nan
            )
            row[f"{column}_positive_year_rate"] = (returns > 0).mean()
            row[f"{column}_pvalue"] = float(test.pvalue)
        rows.append(row)

    summary = pd.DataFrame(rows)
    for column in return_columns:
        summary[f"{column}_pvalue_fdr_bh"] = multipletests(
            summary[f"{column}_pvalue"],
            method="fdr_bh",
        )[1]

    always = complete[complete["condition"] == "always_active"][
        [
            "oos_year",
            "portfolio",
            "holding_period",
            "offset",
            "net_liquidity_cost_return",
        ]
    ].rename(
        columns={
            "net_liquidity_cost_return": "always_net_liquidity_cost_return"
        }
    )
    comparison = complete.merge(
        always,
        on=["oos_year", "portfolio", "holding_period", "offset"],
        how="left",
    )
    comparison["return_delta_vs_always"] = (
        comparison["net_liquidity_cost_return"]
        - comparison["always_net_liquidity_cost_return"]
    )
    delta_rows = []
    for values, data in comparison.groupby(keys):
        delta = data["return_delta_vs_always"].dropna()
        test = ttest_1samp(delta, 0) if len(delta) >= 3 else None
        delta_rows.append(
            {
                **dict(zip(keys, values)),
                "mean_return_delta_vs_always": delta.mean(),
                "delta_vs_always_tstat": (
                    float(test.statistic) if test is not None else np.nan
                ),
                "delta_vs_always_pvalue": (
                    float(test.pvalue) if test is not None else np.nan
                ),
            }
        )
    deltas = pd.DataFrame(delta_rows)
    deltas["delta_vs_always_pvalue_fdr_bh"] = np.nan
    valid = (
        (deltas["condition"] != "always_active")
        & deltas["delta_vs_always_pvalue"].notna()
    )
    if valid.any():
        deltas.loc[valid, "delta_vs_always_pvalue_fdr_bh"] = multipletests(
            deltas.loc[valid, "delta_vs_always_pvalue"],
            method="fdr_bh",
        )[1]
    summary = summary.merge(deltas, on=keys, how="left")
    return summary


def aggregate_phases(grid):
    """Aggregate the 21 timing phases without treating them as independent."""
    rows = []
    keys = ["condition", "portfolio", "holding_period"]
    for values, data in grid.groupby(keys):
        rows.append(
            {
                **dict(zip(keys, values)),
                "calendar_phases": len(data),
                "median_active_period_rate": data["active_period_rate"].median(),
                "median_turnover": data["average_turnover"].median(),
                "median_gross_exposure": data["average_gross_exposure"].median(),
                "median_abs_beta": data["average_abs_beta"].median(),
                "median_net_annual_return": data[
                    "net_liquidity_cost_return_mean"
                ].median(),
                "median_return_per_gross_exposure": (
                    data["net_liquidity_cost_return_mean"]
                    / data["average_gross_exposure"].replace(0, np.nan)
                ).median(),
                "worst_phase_net_annual_return": data[
                    "net_liquidity_cost_return_mean"
                ].min(),
                "median_sharpe": data[
                    "net_liquidity_cost_return_sharpe"
                ].median(),
                "positive_phase_rate": (
                    data["net_liquidity_cost_return_mean"] > 0
                ).mean(),
                "median_return_delta_vs_always": data[
                    "mean_return_delta_vs_always"
                ].median(),
                "minimum_portfolio_fdr_pvalue": data[
                    "net_liquidity_cost_return_pvalue_fdr_bh"
                ].min(),
                "minimum_delta_vs_always_fdr_pvalue": data[
                    "delta_vs_always_pvalue_fdr_bh"
                ].min(),
            }
        )
    return pd.DataFrame(rows)


def annual_decomposition(annual):
    primary = annual[
        (annual["condition"] == PRIMARY_CONDITION)
        & (annual["portfolio"] == PRIMARY_PORTFOLIO)
        & (annual["holding_period"] == PRIMARY_HOLDING_PERIOD)
    ]
    return (
        primary.groupby("oos_year")
        .agg(
            median_gross_return=("gross_return", "median"),
            median_net_return=("net_liquidity_cost_return", "median"),
            median_long_contribution=("long_contribution", "median"),
            median_short_contribution=("short_contribution", "median"),
            median_turnover=("average_turnover", "median"),
            median_active_period_rate=("active_period_rate", "median"),
            median_gross_exposure=("average_gross_exposure", "median"),
            median_abs_beta=("average_abs_beta", "median"),
        )
        .reset_index()
    )


def concentration_statistics(decomposition, final_data_year):
    complete = decomposition[decomposition["oos_year"] < final_data_year]
    returns = complete["median_net_return"]
    positive = returns.clip(lower=0)
    total_positive = positive.sum()
    leave_one_out = [returns.drop(index).mean() for index in returns.index]
    return pd.DataFrame(
        [
            {
                "complete_oos_years": len(returns),
                "positive_year_rate": (returns > 0).mean(),
                "largest_positive_year_share": (
                    positive.max() / total_positive if total_positive > 0 else np.nan
                ),
                "largest_two_positive_year_share": (
                    positive.nlargest(2).sum() / total_positive
                    if total_positive > 0
                    else np.nan
                ),
                "leave_one_year_out_mean_min": min(leave_one_out),
                "leave_one_year_out_mean_max": max(leave_one_out),
            }
        ]
    )


def stop_go_decision(
    phase_summary,
    grid,
    ic_summary,
    concentration,
    selection_eligibility_rate,
):
    """Apply criteria declared before reading this stage's portfolio results."""
    primary = phase_summary[
        (phase_summary["condition"] == PRIMARY_CONDITION)
        & (phase_summary["portfolio"] == PRIMARY_PORTFOLIO)
        & (phase_summary["holding_period"] == PRIMARY_HOLDING_PERIOD)
    ].iloc[0]
    primary_grid = grid[
        (grid["condition"] == PRIMARY_CONDITION)
        & (grid["portfolio"] == PRIMARY_PORTFOLIO)
        & (grid["holding_period"] == PRIMARY_HOLDING_PERIOD)
    ]
    primary_ic = ic_summary[ic_summary["condition"] == PRIMARY_CONDITION].iloc[0]
    concentration = concentration.iloc[0]

    tests = [
        ("conditional_mean_ic_positive", primary_ic["conditional_mean_ic"] > 0,
         primary_ic["conditional_mean_ic"], 0.0, True),
        ("ic_difference_fdr_below_0_05",
         primary_ic["difference_pvalue_fdr_bh"] < 0.05,
         primary_ic["difference_pvalue_fdr_bh"], 0.05, False),
        ("median_net_return_positive", primary["median_net_annual_return"] > 0,
         primary["median_net_annual_return"], 0.0, True),
        ("median_sharpe_at_least_0_5", primary["median_sharpe"] >= 0.5,
         primary["median_sharpe"], 0.5, True),
        ("positive_year_rate_at_least_0_6",
         concentration["positive_year_rate"] >= 0.6,
         concentration["positive_year_rate"], 0.6, True),
        ("positive_phase_rate_at_least_0_8",
         primary["positive_phase_rate"] >= 0.8,
         primary["positive_phase_rate"], 0.8, True),
        ("largest_year_below_0_4_positive_profit",
         concentration["largest_positive_year_share"] <= 0.4,
         concentration["largest_positive_year_share"], 0.4, True),
        ("portfolio_fdr_below_0_05",
         primary_grid["net_liquidity_cost_return_pvalue_fdr_bh"].min() < 0.05,
         primary_grid["net_liquidity_cost_return_pvalue_fdr_bh"].min(), 0.05, True),
        ("active_period_rate_at_least_0_15",
         primary["median_active_period_rate"] >= 0.15,
         primary["median_active_period_rate"], 0.15, False),
        ("training_selection_eligibility_at_least_0_5",
         selection_eligibility_rate >= 0.5,
         selection_eligibility_rate, 0.5, True),
    ]
    result = pd.DataFrame(
        [
            {
                "criterion": name,
                "passed": bool(passed),
                "observed": observed,
                "threshold": threshold,
                "required_for_paper_candidate": required,
            }
            for name, passed, observed, threshold, required in tests
        ]
    )
    required = result[result["required_for_paper_candidate"]]
    if required["passed"].all():
        decision = "paper_candidate"
    elif result["passed"].sum() >= 5:
        decision = "research_watchlist"
    else:
        decision = "reject_mini_strategy"
    return result, decision


def plot_results(phase_summary, decomposition, ic_summary):
    primary = phase_summary[
        (phase_summary["portfolio"] == PRIMARY_PORTFOLIO)
        & (phase_summary["holding_period"] == PRIMARY_HOLDING_PERIOD)
    ]
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(primary["condition"], primary["median_net_annual_return"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Trend Slope: frozen market-condition comparison")
    axis.set_ylabel("Median annual return after modeled costs")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "condition_comparison.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(
        decomposition["oos_year"],
        decomposition["median_long_contribution"],
        label="Long contribution",
    )
    axis.bar(
        decomposition["oos_year"],
        decomposition["median_short_contribution"],
        bottom=decomposition["median_long_contribution"],
        label="Short contribution",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Primary conditional Trend Slope: annual decomposition")
    axis.set_ylabel("Annual contribution")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "annual_long_short_decomposition.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(ic_summary["condition"], ic_summary["conditional_mean_ic"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Trend Slope Mean IC inside frozen market conditions")
    axis.set_ylabel("Conditional Mean IC")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "conditional_ic.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def run_trend_slope_conditional_research():
    required = [
        os.path.join(OPPORTUNITY_OUTPUT_DIR, "stitched_oos_states.parquet"),
        IC_CACHE_PATH,
        METADATA_PATH,
        LIQUIDITY_PATH,
    ]
    if not all(os.path.exists(path) for path in required):
        raise FileNotFoundError(
            "Run walk_forward.py and market_opportunity_research.py first"
        )

    states = pd.read_parquet(required[0])
    exposures = build_condition_exposures(states)
    daily_ic = pd.read_parquet(IC_CACHE_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    returns, availability, _, prices = load_data()
    membership = load_membership().reindex(index=prices.index, columns=prices.columns)
    liquidity = pd.read_parquet(LIQUIDITY_PATH).reindex(
        index=prices.index,
        columns=prices.columns,
    )
    average_dollar_volume = np.expm1(liquidity).shift(1)
    betas = build_trailing_betas(returns, availability, membership)

    factor_cache = {}
    selection_rows = []
    ic_rows = []
    annual_rows = []
    primary_paths = []

    for year in range(FIRST_OOS_YEAR, prices.index.max().year + 1):
        _, selected = score_training_candidates(
            daily_ic,
            metadata,
            SIGNAL_FAMILY,
            year,
        )
        selection_rows.append(selected.to_dict())
        if selected.variant not in factor_cache:
            factor_cache[selected.variant] = pd.read_parquet(
                factor_cache_path(SIGNAL_FAMILY, selected.variant)
            ).astype(float)
        factor = factor_cache[selected.variant]

        year_ic = daily_ic[selected.key].loc[f"{year}-01-01":f"{year}-12-31"]
        for date, value in year_ic.items():
            ic_rows.append(
                {
                    "date": date,
                    "oos_year": year,
                    "variant": selected.variant,
                    "ic": value,
                }
            )

        ranks, year_prices, _, start_position, end_position = build_year_ranks(
            factor,
            prices,
            membership,
            year,
        )

        for condition, portfolio, holding_period in EXPERIMENTS:
            exposure = exposures[condition]
            for offset in range(CALENDAR_PHASES):
                path, final_long, final_short = simulate_trend_path(
                    ranks,
                    year_prices,
                    returns,
                    betas,
                    average_dollar_volume,
                    exposure,
                    holding_period,
                    offset,
                    portfolio,
                    start_position,
                    end_position,
                )
                if path.empty:
                    continue
                final_adv = average_dollar_volume.loc[path.iloc[-1]["end_date"]]
                compounded = compound_annual(
                    path,
                    final_long,
                    final_short,
                    final_adv,
                )
                annual_rows.append(
                    {
                        "oos_year": year,
                        "condition": condition,
                        "portfolio": portfolio,
                        "holding_period": holding_period,
                        "offset": offset,
                        "holding_periods": len(path),
                        "active_period_rate": path["active"].mean(),
                        "average_turnover": path["dollar_turnover"].mean(),
                        "average_gross_exposure": path["gross_exposure"].mean(),
                        "average_abs_beta": path["estimated_beta"].abs().mean(),
                        "long_contribution": path["long_contribution"].sum(),
                        "short_contribution": path["short_contribution"].sum(),
                        **compounded,
                    }
                )
                if (
                    condition == PRIMARY_CONDITION
                    and portfolio == PRIMARY_PORTFOLIO
                    and holding_period == PRIMARY_HOLDING_PERIOD
                ):
                    path = path.copy()
                    path["oos_year"] = year
                    path["offset"] = offset
                    primary_paths.append(path)
        print(f"Conditional Trend Slope OOS year complete: {year}")

    selections = pd.DataFrame(selection_rows)
    ic_history = pd.DataFrame(ic_rows).set_index("date").sort_index()
    annual = pd.DataFrame(annual_rows)
    paths = pd.concat(primary_paths, ignore_index=True)
    ic_summary = conditional_ic_summary(ic_history["ic"], exposures)
    grid = summarize_grid(annual, int(prices.index.max().year))
    phase_summary = aggregate_phases(grid)
    decomposition = annual_decomposition(annual)
    concentration = concentration_statistics(
        decomposition,
        int(prices.index.max().year),
    )
    complete_selections = selections[
        selections["oos_year"] < int(prices.index.max().year)
    ]
    selection_eligibility_rate = complete_selections["eligible"].mean()
    decision_tests, decision = stop_go_decision(
        phase_summary,
        grid,
        ic_summary,
        concentration,
        selection_eligibility_rate,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    selections.to_csv(os.path.join(OUTPUT_DIR, "annual_selections.csv"), index=False)
    ic_history.to_parquet(os.path.join(OUTPUT_DIR, "stitched_trend_slope_ic.parquet"))
    ic_summary.to_csv(os.path.join(OUTPUT_DIR, "conditional_ic.csv"), index=False)
    annual.to_parquet(os.path.join(OUTPUT_DIR, "annual_phase_returns.parquet"))
    paths.to_parquet(os.path.join(OUTPUT_DIR, "primary_paths.parquet"))
    grid.to_csv(os.path.join(OUTPUT_DIR, "grid_statistics.csv"), index=False)
    phase_summary.to_csv(os.path.join(OUTPUT_DIR, "phase_summary.csv"), index=False)
    decomposition.to_csv(os.path.join(OUTPUT_DIR, "annual_decomposition.csv"), index=False)
    concentration.to_csv(os.path.join(OUTPUT_DIR, "profit_concentration.csv"), index=False)
    decision_tests.to_csv(os.path.join(OUTPUT_DIR, "stop_go_criteria.csv"), index=False)
    plot_results(phase_summary, decomposition, ic_summary)

    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "signal_family": SIGNAL_FAMILY,
                "signal_selection": "previous 5 purged years, 21-day IC only",
                "conditions": list(CONDITIONS),
                "portfolios": PORTFOLIOS,
                "holding_periods": list(HOLDING_PERIODS),
                "experiments": [list(experiment) for experiment in EXPERIMENTS],
                "calendar_phases": CALENDAR_PHASES,
                "primary_condition": PRIMARY_CONDITION,
                "primary_portfolio": PRIMARY_PORTFOLIO,
                "primary_holding_period": PRIMARY_HOLDING_PERIOD,
                "complete_year_selection_eligibility_rate": (
                    float(selection_eligibility_rate)
                ),
                "decision": decision,
                "development_backtest": True,
                "independent_validation": False,
                "parameters_selected_from_this_stage_results": False,
            },
            file,
            indent=2,
        )

    return phase_summary, ic_summary, decomposition, decision_tests, decision


def print_results(phase_summary, ic_summary, decomposition, decision_tests, decision):
    primary_comparison = phase_summary[
        (phase_summary["portfolio"] == PRIMARY_PORTFOLIO)
        & (phase_summary["holding_period"] == PRIMARY_HOLDING_PERIOD)
    ]
    columns = [
        "condition",
        "median_active_period_rate",
        "median_turnover",
        "median_net_annual_return",
        "worst_phase_net_annual_return",
        "median_sharpe",
        "positive_phase_rate",
        "minimum_portfolio_fdr_pvalue",
    ]
    print("\nFROZEN CONDITION COMPARISON")
    print(
        primary_comparison[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nCONDITIONAL TREND SLOPE IC")
    print(
        ic_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nPRIMARY ANNUAL DECOMPOSITION")
    print(
        decomposition.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nSTOP / GO")
    print(decision_tests.to_string(index=False))
    print(f"Decision: {decision}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    output = run_trend_slope_conditional_research()
    print_results(*output)
