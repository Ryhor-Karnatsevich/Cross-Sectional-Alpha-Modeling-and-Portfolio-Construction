"""Turnover-aware and risk-controlled implementation of the frozen alpha candidate.

The signal and Market Opportunity rule are not re-selected here. This stage
only tests whether transparent portfolio implementation can preserve the fixed
conditional signal after turnover, liquidity-dependent costs, and risk limits.
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
    COMPONENT_FAMILIES,
    FIRST_OOS_YEAR,
    build_year_composites,
    factor_cache_path,
    score_training_candidates,
)
from market_opportunity_research import (
    OUTPUT_DIR as OPPORTUNITY_OUTPUT_DIR,
    state_difference_test,
)
from pipeline import load_data, load_membership
from statistical_research import compute_daily_spearman_ic, compute_forward_returns
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
    "portfolio_implementation_stage",
)

HOLDING_PERIODS = (21, 42, 63)
CALENDAR_PHASES = 21
VARIANTS = (
    "conditional_equal",
    "conditional_buffered",
    "conditional_buffered_beta_neutral",
    "conditional_risk_controlled",
)
PRIMARY_VARIANT = "conditional_risk_controlled"
PRIMARY_HOLDING_PERIOD = 21

ENTRY_QUANTILE = 0.20
EXIT_QUANTILE = 0.30
MAX_ABSOLUTE_POSITION = 0.02
MAX_GROSS_EXPOSURE = 2.0
BETA_WINDOW = 252
BETA_MIN_PERIODS = 126
VOLATILITY_LOOKBACK = 63
VOLATILITY_TARGET = 0.10
MIN_VOLATILITY_SCALE = 0.25
REFERENCE_AUM = 10_000_000
BASE_COST_BPS = 5.0
IMPACT_BPS_AT_FULL_ADV = 25.0
FALLBACK_COST_BPS = 25.0
MAX_COST_BPS = 100.0
FIXED_COSTS_BPS = (10, 25)
MIN_ASSETS = 50


def build_trailing_betas(returns, availability, membership):
    """Estimate stock betas against a point-in-time S&P 500 return proxy."""
    asset_returns = returns.where(availability)
    market_return = returns.where(availability & membership).mean(axis=1)
    covariance = asset_returns.rolling(
        BETA_WINDOW,
        min_periods=BETA_MIN_PERIODS,
    ).cov(market_return)
    market_variance = market_return.rolling(
        BETA_WINDOW,
        min_periods=BETA_MIN_PERIODS,
    ).var()
    return covariance.div(market_variance, axis=0).shift(1)


def capped_equal_weights(tickers, total_exposure=1.0):
    """Create equal weights while enforcing the absolute position cap."""
    tickers = pd.Index(tickers)
    if len(tickers) == 0 or total_exposure <= 0:
        return pd.Series(dtype=float)
    if len(tickers) * MAX_ABSOLUTE_POSITION + 1e-12 < total_exposure:
        raise ValueError("Position cap is infeasible for the selected basket")
    return pd.Series(total_exposure / len(tickers), index=tickers)


def select_buffered_baskets(scores, eligible, previous_long, previous_short, buffered):
    """Enter at 20% tails and, with a buffer, exit only beyond 30% tails."""
    ranked = scores.loc[eligible].rank(method="first", pct=True)
    long_entries = set(ranked.index[ranked >= 1 - ENTRY_QUANTILE])
    short_entries = set(ranked.index[ranked <= ENTRY_QUANTILE])

    if not buffered:
        return sorted(long_entries), sorted(short_entries), ranked

    eligible_names = set(ranked.index)
    retained_long = {
        ticker
        for ticker in previous_long
        if ticker in eligible_names and ranked[ticker] >= 1 - EXIT_QUANTILE
    }
    retained_short = {
        ticker
        for ticker in previous_short
        if ticker in eligible_names and ranked[ticker] <= EXIT_QUANTILE
    }
    return (
        sorted(long_entries | retained_long),
        sorted(short_entries | retained_short),
        ranked,
    )


def beta_neutral_leg_exposures(long_weights, short_weights, beta, gross_exposure):
    """Scale long and short books to zero estimated beta at fixed gross."""
    long_beta = float((long_weights * beta.reindex(long_weights.index)).sum())
    short_beta = float((short_weights * beta.reindex(short_weights.index)).sum())

    if (
        not np.isfinite(long_beta)
        or not np.isfinite(short_beta)
        or long_beta <= 0
        or short_beta <= 0
        or long_beta + short_beta <= 0
    ):
        return gross_exposure / 2, gross_exposure / 2

    long_exposure = gross_exposure * short_beta / (long_beta + short_beta)
    short_exposure = gross_exposure * long_beta / (long_beta + short_beta)
    return long_exposure, short_exposure


def estimate_spread_volatility(returns, date, signed_weights):
    """Estimate trailing annualized volatility of the proposed spread."""
    position = returns.index.get_loc(date)
    start = max(0, position - VOLATILITY_LOOKBACK)
    window = returns.iloc[start:position]
    available_weights = signed_weights.reindex(window.columns).fillna(0)
    portfolio_returns = window.fillna(0).dot(available_weights)
    if portfolio_returns.notna().sum() < 40:
        return np.nan
    return float(portfolio_returns.std() * np.sqrt(252))


def volatility_scale(estimated_volatility):
    if not np.isfinite(estimated_volatility) or estimated_volatility <= 0:
        return 1.0
    return float(
        np.clip(
            VOLATILITY_TARGET / estimated_volatility,
            MIN_VOLATILITY_SCALE,
            1.0,
        )
    )


def drift_absolute_weights(weights, asset_returns):
    """Carry absolute book weights to the next rebalance."""
    gross_value = weights * (1 + asset_returns.reindex(weights.index).fillna(0))
    total = gross_value.sum()
    if not np.isfinite(total) or total <= 0:
        return weights
    return gross_value / total * weights.sum()


def weight_changes(previous, target):
    tickers = previous.index.union(target.index)
    return (
        target.reindex(tickers, fill_value=0)
        - previous.reindex(tickers, fill_value=0)
    ).abs()


def liquidity_cost(changes, average_dollar_volume):
    """Apply spread plus square-root market impact to each dollar traded."""
    if changes.empty:
        return 0.0
    adv = average_dollar_volume.reindex(changes.index)
    participation = (changes * REFERENCE_AUM).div(adv).replace([np.inf, -np.inf], np.nan)
    modeled_bps = BASE_COST_BPS + IMPACT_BPS_AT_FULL_ADV * np.sqrt(participation)
    modeled_bps = modeled_bps.clip(upper=MAX_COST_BPS).fillna(FALLBACK_COST_BPS)
    return float((changes * modeled_bps / 10_000).sum())


def realized_returns(prices, position, horizon, tickers):
    start = prices.iloc[position].reindex(tickers)
    future = prices.iloc[position + 1 : position + horizon + 1][tickers]
    end = future.ffill().iloc[-1].fillna(start)
    return end / start - 1


def simulate_path(
    factor,
    prices,
    membership,
    returns,
    betas,
    average_dollar_volume,
    exposure,
    holding_period,
    offset,
    variant,
    start_position,
    end_position,
):
    """Simulate one annual phase of the frozen conditional strategy."""
    buffered = variant != "conditional_equal"
    beta_neutral = variant in {
        "conditional_buffered_beta_neutral",
        "conditional_risk_controlled",
    }
    risk_controlled = variant == "conditional_risk_controlled"
    signal = factor.shift(1)
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
        scores = signal.iloc[position]
        eligible = (
            scores.notna()
            & membership.iloc[position].fillna(False)
            & prices.iloc[position].notna()
            & (prices.iloc[position] > 0)
        )
        if eligible.sum() < MIN_ASSETS:
            continue

        previous_long_names = previous_long.index if previous_active else []
        previous_short_names = previous_short.index if previous_active else []
        long_names, short_names, _ = select_buffered_baskets(
            scores,
            eligible,
            previous_long_names,
            previous_short_names,
            buffered,
        )
        unit_long = capped_equal_weights(long_names)
        unit_short = capped_equal_weights(short_names)

        beta = betas.loc[date]
        beta = beta.fillna(beta.median()).fillna(1.0)
        unit_signed = pd.concat([unit_long, -unit_short]).groupby(level=0).sum()
        risk_scale = 1.0
        estimated_volatility = estimate_spread_volatility(
            returns,
            date,
            unit_signed,
        )
        if risk_controlled:
            risk_scale = volatility_scale(estimated_volatility)

        gross_exposure = MAX_GROSS_EXPOSURE * risk_scale if active else 0.0
        if beta_neutral and active:
            long_exposure, short_exposure = beta_neutral_leg_exposures(
                unit_long,
                unit_short,
                beta,
                gross_exposure,
            )
        else:
            long_exposure = gross_exposure / 2
            short_exposure = gross_exposure / 2

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
        underlying_spread = unit_long_return - unit_short_return
        long_contribution = float((target_long * long_asset_returns).sum())
        short_contribution = -float((target_short * short_asset_returns).sum())
        gross_return = long_contribution + short_contribution

        if not previous_active and active:
            transition = "entry"
        elif previous_active and not active:
            transition = "exit"
        elif active:
            transition = "stay_active"
        else:
            transition = "stay_inactive"

        residual_beta = float(
            (target_long * beta.reindex(target_long.index)).sum()
            - (target_short * beta.reindex(target_short.index)).sum()
        )
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
            "estimated_beta": residual_beta,
            "estimated_spread_volatility": estimated_volatility,
            "volatility_scale": risk_scale,
            "underlying_spread_return": underlying_spread,
            "inactive_foregone_return": underlying_spread if not active else 0.0,
            "long_contribution": long_contribution,
            "short_contribution": short_contribution,
            "spread_gross_return": gross_return,
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


def liquidation_cost(weights, adv, fixed_bps=None):
    changes = weights.abs()
    if fixed_bps is not None:
        return float(changes.sum() * fixed_bps / 10_000)
    return liquidity_cost(changes, adv)


def compound_annual(path, final_long, final_short, final_adv):
    result = {"gross_return": (1 + path["spread_gross_return"]).prod() - 1}
    for cost_bps in FIXED_COSTS_BPS:
        period_return = path["spread_gross_return"] - path[f"fixed_{cost_bps}bps_cost"]
        closing = liquidation_cost(final_long, final_adv, cost_bps) + liquidation_cost(
            final_short,
            final_adv,
            cost_bps,
        )
        result[f"net_fixed_{cost_bps}bps_return"] = (
            (1 + period_return).prod() * (1 - closing) - 1
        )

    modeled_period = path["spread_gross_return"] - path["modeled_liquidity_cost"]
    modeled_closing = liquidity_cost(final_long.abs(), final_adv) + liquidity_cost(
        final_short.abs(),
        final_adv,
    )
    result["net_liquidity_cost_return"] = (
        (1 + modeled_period).prod() * (1 - modeled_closing) - 1
    )
    return result


def summarize_grid(annual, final_data_year):
    complete = annual[annual["oos_year"] < final_data_year]
    rows = []
    return_columns = [
        "gross_return",
        "net_fixed_10bps_return",
        "net_fixed_25bps_return",
        "net_liquidity_cost_return",
    ]

    for (variant, horizon, offset), data in complete.groupby(
        ["variant", "holding_period", "offset"]
    ):
        row = {
            "variant": variant,
            "holding_period": horizon,
            "offset": offset,
            "complete_oos_years": len(data),
            "average_turnover": data["average_turnover"].mean(),
            "average_gross_exposure": data["average_gross_exposure"].mean(),
            "average_abs_beta": data["average_abs_beta"].mean(),
            "active_period_rate": data["active_period_rate"].mean(),
        }
        for column in return_columns:
            values = data[column]
            test = ttest_1samp(values, 0)
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_volatility"] = values.std()
            row[f"{column}_sharpe"] = (
                values.mean() / values.std() if values.std() else np.nan
            )
            row[f"{column}_positive_year_rate"] = (values > 0).mean()
            row[f"{column}_pvalue"] = float(test.pvalue)
        rows.append(row)

    summary = pd.DataFrame(rows)
    for column in return_columns:
        pvalue = f"{column}_pvalue"
        summary[f"{column}_pvalue_fdr_bh"] = multipletests(
            summary[pvalue],
            method="fdr_bh",
        )[1]
    return summary


def aggregate_phases(summary):
    rows = []
    for (variant, horizon), data in summary.groupby(["variant", "holding_period"]):
        rows.append(
            {
                "variant": variant,
                "holding_period": horizon,
                "calendar_phases": len(data),
                "median_turnover": data["average_turnover"].median(),
                "median_gross_exposure": data["average_gross_exposure"].median(),
                "median_abs_beta": data["average_abs_beta"].median(),
                "median_active_period_rate": data["active_period_rate"].median(),
                "gross_median_annual_return": data["gross_return_mean"].median(),
                "fixed_10bps_median_annual_return": data[
                    "net_fixed_10bps_return_mean"
                ].median(),
                "fixed_10bps_median_sharpe": data[
                    "net_fixed_10bps_return_sharpe"
                ].median(),
                "fixed_10bps_positive_phase_rate": (
                    data["net_fixed_10bps_return_mean"] > 0
                ).mean(),
                "liquidity_cost_median_annual_return": data[
                    "net_liquidity_cost_return_mean"
                ].median(),
                "liquidity_cost_worst_phase_annual_return": data[
                    "net_liquidity_cost_return_mean"
                ].min(),
                "liquidity_cost_median_sharpe": data[
                    "net_liquidity_cost_return_sharpe"
                ].median(),
                "liquidity_cost_positive_phase_rate": (
                    data["net_liquidity_cost_return_mean"] > 0
                ).mean(),
                "minimum_fdr_pvalue": data[
                    "net_liquidity_cost_return_pvalue_fdr_bh"
                ].min(),
            }
        )
    return pd.DataFrame(rows)


def component_conditional_ic(selected_by_year, states, prices, membership):
    """Identify which fixed composite components benefit from low correlation."""
    forward_returns = compute_forward_returns(prices, 21)
    cache = {}
    rows = []

    for year, selections in selected_by_year.items():
        for selection in selections:
            key = (selection.family, selection.variant)
            if key not in cache:
                factor = pd.read_parquet(factor_cache_path(*key)).astype(float)
                cache[key] = compute_daily_spearman_ic(
                    factor,
                    forward_returns,
                    membership,
                )["ic"]
            values = cache[key].loc[f"{year}-01-01":f"{year}-12-31"]
            for date, value in values.items():
                rows.append(
                    {
                        "date": date,
                        "family": selection.family,
                        "variant": selection.variant,
                        "ic": value,
                    }
                )

    history = pd.DataFrame(rows)
    results = []
    low_correlation = states["low_correlation"].astype("boolean")
    for family, data in history.groupby("family"):
        ic = data.groupby("date")["ic"].first().sort_index()
        results.append(
            {
                "family": family,
                **state_difference_test(ic, low_correlation),
            }
        )
    summary = pd.DataFrame(results)
    summary["pvalue_fdr_bh"] = multipletests(
        summary["pvalue"],
        method="fdr_bh",
    )[1]
    return history, summary


def transition_summary(paths):
    rows = []
    for transition, data in paths.groupby("transition"):
        rows.append(
            {
                "transition": transition,
                "observations": len(data),
                "mean_underlying_spread_return": data[
                    "underlying_spread_return"
                ].mean(),
                "mean_realized_gross_return": data["spread_gross_return"].mean(),
                "mean_dollar_turnover": data["dollar_turnover"].mean(),
                "mean_fixed_10bps_cost": data["fixed_10bps_cost"].mean(),
                "mean_liquidity_cost": data["modeled_liquidity_cost"].mean(),
                "total_inactive_foregone_return": data[
                    "inactive_foregone_return"
                ].sum(),
            }
        )
    return pd.DataFrame(rows)


def annual_decomposition(annual):
    primary = annual[
        (annual["variant"] == PRIMARY_VARIANT)
        & (annual["holding_period"] == PRIMARY_HOLDING_PERIOD)
    ]
    return (
        primary.groupby("oos_year")
        .agg(
            median_gross_return=("gross_return", "median"),
            median_net_liquidity_cost_return=("net_liquidity_cost_return", "median"),
            median_long_contribution=("long_contribution", "median"),
            median_short_contribution=("short_contribution", "median"),
            median_turnover=("average_turnover", "median"),
            median_gross_exposure=("average_gross_exposure", "median"),
            median_abs_beta=("average_abs_beta", "median"),
        )
        .reset_index()
    )


def concentration_statistics(annual_decomp, final_data_year):
    complete = annual_decomp[annual_decomp["oos_year"] < final_data_year]
    returns = complete["median_net_liquidity_cost_return"]
    positive = returns.clip(lower=0)
    total_positive = positive.sum()
    top_share = positive.max() / total_positive if total_positive > 0 else np.nan
    top_two_share = (
        positive.nlargest(2).sum() / total_positive if total_positive > 0 else np.nan
    )
    leave_one_out = [returns.drop(index).mean() for index in returns.index]
    return pd.DataFrame(
        [
            {
                "complete_oos_years": len(returns),
                "positive_year_rate": (returns > 0).mean(),
                "largest_positive_year_share": top_share,
                "largest_two_positive_year_share": top_two_share,
                "leave_one_year_out_mean_min": min(leave_one_out),
                "leave_one_year_out_mean_max": max(leave_one_out),
            }
        ]
    )


def stop_go_decision(phase_summary, annual_decomp, grid_summary, final_data_year):
    primary = phase_summary[
        (phase_summary["variant"] == PRIMARY_VARIANT)
        & (phase_summary["holding_period"] == PRIMARY_HOLDING_PERIOD)
    ].iloc[0]
    concentration = concentration_statistics(annual_decomp, final_data_year).iloc[0]
    primary_grid = grid_summary[
        (grid_summary["variant"] == PRIMARY_VARIANT)
        & (grid_summary["holding_period"] == PRIMARY_HOLDING_PERIOD)
    ]
    tests = [
        (
            "positive_net_return",
            primary["liquidity_cost_median_annual_return"] > 0,
            primary["liquidity_cost_median_annual_return"],
            0.0,
        ),
        (
            "median_sharpe_at_least_0_4",
            primary["liquidity_cost_median_sharpe"] >= 0.4,
            primary["liquidity_cost_median_sharpe"],
            0.4,
        ),
        (
            "positive_year_rate_above_half",
            concentration["positive_year_rate"] > 0.5,
            concentration["positive_year_rate"],
            0.5,
        ),
        (
            "positive_phase_rate_at_least_0_8",
            primary["liquidity_cost_positive_phase_rate"] >= 0.8,
            primary["liquidity_cost_positive_phase_rate"],
            0.8,
        ),
        (
            "largest_year_below_half_positive_profit",
            concentration["largest_positive_year_share"] <= 0.5,
            concentration["largest_positive_year_share"],
            0.5,
        ),
        (
            "portfolio_fdr_below_0_05",
            primary_grid["net_liquidity_cost_return_pvalue_fdr_bh"].min() < 0.05,
            primary_grid["net_liquidity_cost_return_pvalue_fdr_bh"].min(),
            0.05,
        ),
    ]
    result = pd.DataFrame(
        [
            {
                "criterion": name,
                "passed": bool(passed),
                "observed": observed,
                "threshold": threshold,
            }
            for name, passed, observed, threshold in tests
        ]
    )
    passed = int(result["passed"].sum())
    decision = "paper_candidate" if passed >= 5 else "research_candidate_only"
    return result, decision


def plot_results(phase_summary, annual_decomp):
    labels = (
        phase_summary["variant"].str.replace("conditional_", "", regex=False)
        + "\n"
        + phase_summary["holding_period"].astype(str)
        + "d"
    )
    figure, axis = plt.subplots(figsize=(14, 7))
    colors = np.where(
        phase_summary["liquidity_cost_median_annual_return"] >= 0,
        "#2f7d32",
        "#a33a3a",
    )
    axis.bar(labels, phase_summary["liquidity_cost_median_annual_return"], color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Median annual return after modeled liquidity costs")
    axis.set_title("Frozen conditional composite: implementation sensitivity")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "implementation_comparison.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        annual_decomp["oos_year"],
        annual_decomp["median_long_contribution"],
        label="Long contribution",
    )
    axis.bar(
        annual_decomp["oos_year"],
        annual_decomp["median_short_contribution"],
        bottom=annual_decomp["median_long_contribution"],
        label="Short contribution",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Primary implementation: annual long and short contribution")
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


def run_portfolio_implementation_research():
    required = [
        os.path.join(OPPORTUNITY_OUTPUT_DIR, "stitched_oos_states.parquet"),
        IC_CACHE_PATH,
        METADATA_PATH,
    ]
    if not all(os.path.exists(path) for path in required):
        raise FileNotFoundError(
            "Run walk_forward.py and market_opportunity_research.py first"
        )

    states = pd.read_parquet(required[0])
    exposure = states["exposure_binary_opportunity"]
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

    selected_by_year = {}
    annual_rows = []
    primary_paths = []

    for year in range(FIRST_OOS_YEAR, prices.index.max().year + 1):
        selected_rows = []
        for family in COMPONENT_FAMILIES:
            _, selected = score_training_candidates(
                daily_ic,
                metadata,
                family,
                year,
            )
            selected_rows.append(selected)
        selected_by_year[year] = selected_rows

        composites, sliced_prices, sliced_membership, start_position, end_position = (
            build_year_composites(
                selected_rows,
                prices,
                returns,
                availability,
                membership,
                year,
            )
        )
        factor = composites["raw_equal"]

        for variant in VARIANTS:
            for holding_period in HOLDING_PERIODS:
                for offset in range(CALENDAR_PHASES):
                    if start_position + offset > end_position:
                        continue
                    path, final_long, final_short = simulate_path(
                        factor,
                        sliced_prices,
                        sliced_membership,
                        returns,
                        betas,
                        average_dollar_volume,
                        exposure,
                        holding_period,
                        offset,
                        variant,
                        start_position,
                        end_position,
                    )
                    if path.empty:
                        continue
                    final_adv = average_dollar_volume.loc[path.iloc[-1]["end_date"]]
                    annual_result = compound_annual(
                        path,
                        final_long,
                        final_short,
                        final_adv,
                    )
                    annual_rows.append(
                        {
                            "oos_year": year,
                            "variant": variant,
                            "holding_period": holding_period,
                            "offset": offset,
                            "holding_periods": len(path),
                            "active_period_rate": path["active"].mean(),
                            "average_turnover": path["dollar_turnover"].mean(),
                            "average_gross_exposure": path["gross_exposure"].mean(),
                            "average_abs_beta": path["estimated_beta"].abs().mean(),
                            "long_contribution": path["long_contribution"].sum(),
                            "short_contribution": path["short_contribution"].sum(),
                            **annual_result,
                        }
                    )
                    if (
                        variant == PRIMARY_VARIANT
                        and holding_period == PRIMARY_HOLDING_PERIOD
                    ):
                        path = path.copy()
                        path["oos_year"] = year
                        path["offset"] = offset
                        primary_paths.append(path)
        print(f"Portfolio implementation OOS year complete: {year}")

    annual = pd.DataFrame(annual_rows)
    paths = pd.concat(primary_paths, ignore_index=True)
    grid_summary = summarize_grid(annual, int(prices.index.max().year))
    phase_summary = aggregate_phases(grid_summary)
    component_history, component_summary = component_conditional_ic(
        selected_by_year,
        states,
        prices,
        membership,
    )
    transitions = transition_summary(paths)
    annual_decomp = annual_decomposition(annual)
    concentration = concentration_statistics(
        annual_decomp,
        int(prices.index.max().year),
    )
    decision_tests, decision = stop_go_decision(
        phase_summary,
        annual_decomp,
        grid_summary,
        int(prices.index.max().year),
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    annual.to_parquet(os.path.join(OUTPUT_DIR, "annual_phase_returns.parquet"))
    paths.to_parquet(os.path.join(OUTPUT_DIR, "primary_portfolio_paths.parquet"))
    grid_summary.to_csv(os.path.join(OUTPUT_DIR, "grid_statistics.csv"), index=False)
    phase_summary.to_csv(os.path.join(OUTPUT_DIR, "phase_summary.csv"), index=False)
    component_history.to_parquet(
        os.path.join(OUTPUT_DIR, "component_oos_ic.parquet")
    )
    component_summary.to_csv(
        os.path.join(OUTPUT_DIR, "component_low_correlation_ic.csv"),
        index=False,
    )
    transitions.to_csv(os.path.join(OUTPUT_DIR, "transition_summary.csv"), index=False)
    annual_decomp.to_csv(
        os.path.join(OUTPUT_DIR, "annual_decomposition.csv"),
        index=False,
    )
    concentration.to_csv(
        os.path.join(OUTPUT_DIR, "profit_concentration.csv"),
        index=False,
    )
    decision_tests.to_csv(
        os.path.join(OUTPUT_DIR, "stop_go_criteria.csv"),
        index=False,
    )
    plot_results(phase_summary, annual_decomp)

    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "frozen_signal": "raw_equal composite",
                "frozen_market_rule": "binary: at least 3 of 4 conditions",
                "holding_periods": list(HOLDING_PERIODS),
                "calendar_phases": CALENDAR_PHASES,
                "variants": list(VARIANTS),
                "primary_variant": PRIMARY_VARIANT,
                "primary_holding_period": PRIMARY_HOLDING_PERIOD,
                "entry_quantile": ENTRY_QUANTILE,
                "exit_quantile": EXIT_QUANTILE,
                "beta_window": BETA_WINDOW,
                "volatility_target": VOLATILITY_TARGET,
                "maximum_absolute_position": MAX_ABSOLUTE_POSITION,
                "maximum_gross_exposure": MAX_GROSS_EXPOSURE,
                "reference_aum": REFERENCE_AUM,
                "base_cost_bps": BASE_COST_BPS,
                "impact_bps_at_full_adv": IMPACT_BPS_AT_FULL_ADV,
                "decision": decision,
                "development_backtest": True,
                "parameters_selected_from_this_stage_results": False,
            },
            file,
            indent=2,
        )

    return phase_summary, component_summary, transitions, decision_tests, decision


def print_results(phase_summary, component_summary, transitions, decision_tests, decision):
    display = [
        "variant",
        "holding_period",
        "median_turnover",
        "median_gross_exposure",
        "median_abs_beta",
        "liquidity_cost_median_annual_return",
        "liquidity_cost_median_sharpe",
        "liquidity_cost_positive_phase_rate",
        "minimum_fdr_pvalue",
    ]
    print("\nPORTFOLIO IMPLEMENTATION GRID")
    print(
        phase_summary[display].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nCOMPONENT IC IN LOW-CORRELATION STATES")
    print(
        component_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nPRIMARY IMPLEMENTATION TRANSITIONS")
    print(
        transitions.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nSTOP / GO")
    print(decision_tests.to_string(index=False))
    print(f"Decision: {decision}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    output = run_portfolio_implementation_research()
    print_results(*output)
