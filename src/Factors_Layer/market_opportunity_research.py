"""Point-in-time market-opportunity research for the fixed composite signal.

This stage asks whether the existing raw equal-weight composite is more useful
when rates, single-stock volatility, cross-sectional dispersion, and stock
co-movement create a better environment for relative-value equity selection.
It does not use the current OOS year to choose thresholds or factor parameters.
"""

import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from composite_alpha_research import (
    COMPONENT_FAMILIES,
    FIRST_OOS_YEAR,
    HORIZON,
    LOOKBACK_YEARS,
    build_year_composites,
    score_training_candidates,
)
from pipeline import load_data, load_membership
from quantile_research import drift_weights, realized_asset_returns
from statistical_research import (
    compute_daily_spearman_ic,
    compute_forward_returns,
    hac_mean_test,
)
from walk_forward import IC_CACHE_PATH, METADATA_PATH


data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data_System")
)
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import BASE_DIR
from risk_free_rate import ensure_risk_free_rate


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "Factor_Research",
    "market_opportunity_stage",
)

VOLATILITY_WINDOW = 63
VOLATILITY_MIN_PERIODS = 50
DISPERSION_SMOOTHING = 21
CORRELATION_SMOOTHING = 21
RATE_THRESHOLD_PCT = 2.0
VOLATILITY_LOWER_QUANTILE = 0.50
VOLATILITY_UPPER_QUANTILE = 0.90
MIN_TRAINING_OBSERVATIONS = 252
MIN_PORTFOLIO_ASSETS = 50
CALENDAR_PHASES = 21
COSTS_BPS = (0, 10, 25)
SCHEMES = ("always_active", "binary_opportunity", "tiered_opportunity")


def build_opportunity_indicators(returns, availability, risk_free_rate):
    """Build lagged indicators using only the point-in-time investable universe."""
    eligible_returns = returns.where(availability)

    daily_dispersion = eligible_returns.std(axis=1)
    dispersion = (
        daily_dispersion.rolling(
            DISPERSION_SMOOTHING,
            min_periods=15,
        ).mean()
        * np.sqrt(252)
    )

    stock_volatility = (
        eligible_returns.rolling(
            VOLATILITY_WINDOW,
            min_periods=VOLATILITY_MIN_PERIODS,
        ).std()
        * np.sqrt(252)
    )
    median_stock_volatility = stock_volatility.median(axis=1)

    # Efficient realized average-correlation proxy. Each return is scaled by
    # volatility known before that return. The cross-sectional pair products
    # are then averaged over one month. This avoids thousands of daily 500x500
    # correlation matrices while retaining the required co-movement signal.
    prior_volatility = eligible_returns.rolling(
        VOLATILITY_WINDOW,
        min_periods=VOLATILITY_MIN_PERIODS,
    ).std().shift(1)
    standardized = (eligible_returns / prior_volatility).clip(-5, 5)
    count = standardized.notna().sum(axis=1)
    total = standardized.sum(axis=1, min_count=2)
    squared_total = standardized.pow(2).sum(axis=1, min_count=2)
    pair_count = count * (count - 1)
    daily_pair_product = ((total.pow(2) - squared_total) / pair_count).where(
        count >= MIN_PORTFOLIO_ASSETS
    )
    average_correlation_proxy = daily_pair_product.rolling(
        CORRELATION_SMOOTHING,
        min_periods=15,
    ).mean().clip(-1, 1)

    aligned_rate = (
        risk_free_rate["annual_rate_pct"]
        .reindex(returns.index)
        .ffill()
    )

    # Every value used for a decision at t was completely observable by t-1.
    return pd.DataFrame(
        {
            "risk_free_rate_pct": aligned_rate,
            "median_stock_volatility": median_stock_volatility,
            "cross_sectional_dispersion": dispersion,
            "average_correlation_proxy": average_correlation_proxy,
        }
    ).shift(1)


def training_thresholds(indicators, index, year):
    """Estimate one year's thresholds from the prior five purged years."""
    selection_date = pd.Timestamp(f"{year - 1}-12-31")
    selection_position = index.searchsorted(selection_date, side="right") - 1
    purged_position = selection_position - HORIZON
    if purged_position < 0:
        raise ValueError(f"Insufficient history for OOS year {year}")

    purged_end = index[purged_position]
    train_start = pd.Timestamp(f"{year - LOOKBACK_YEARS}-01-01")
    training = indicators.loc[train_start:purged_end]

    required = [
        "median_stock_volatility",
        "cross_sectional_dispersion",
        "average_correlation_proxy",
    ]
    if any(training[column].notna().sum() < MIN_TRAINING_OBSERVATIONS for column in required):
        raise ValueError(f"Insufficient opportunity history for OOS year {year}")

    return {
        "oos_year": year,
        "train_start": train_start,
        "purged_train_end": purged_end,
        "risk_free_rate_threshold_pct": RATE_THRESHOLD_PCT,
        "volatility_lower": training["median_stock_volatility"].quantile(
            VOLATILITY_LOWER_QUANTILE
        ),
        "volatility_upper": training["median_stock_volatility"].quantile(
            VOLATILITY_UPPER_QUANTILE
        ),
        "dispersion_threshold": training["cross_sectional_dispersion"].median(),
        "correlation_threshold": training["average_correlation_proxy"].median(),
    }


def apply_thresholds(indicators, thresholds):
    """Create transparent opportunity conditions and two exposure rules."""
    valid = indicators.notna().all(axis=1)
    conditions = pd.DataFrame(index=indicators.index)
    conditions["rate_above_2pct"] = (
        indicators["risk_free_rate_pct"]
        > thresholds["risk_free_rate_threshold_pct"]
    )
    conditions["moderately_high_stock_volatility"] = (
        indicators["median_stock_volatility"] >= thresholds["volatility_lower"]
    ) & (
        indicators["median_stock_volatility"] <= thresholds["volatility_upper"]
    )
    conditions["high_dispersion"] = (
        indicators["cross_sectional_dispersion"]
        >= thresholds["dispersion_threshold"]
    )
    conditions["low_correlation"] = (
        indicators["average_correlation_proxy"]
        <= thresholds["correlation_threshold"]
    )
    conditions = conditions.where(valid)
    score = conditions.astype(float).sum(axis=1, min_count=4).rename(
        "opportunity_score"
    )

    exposures = pd.DataFrame(index=indicators.index)
    exposures["always_active"] = 1.0
    exposures["binary_opportunity"] = (score >= 3).astype(float).where(valid, 0.0)
    exposures["tiered_opportunity"] = score.map(
        {0.0: 0.0, 1.0: 0.0, 2.0: 0.5, 3.0: 1.0, 4.0: 1.0}
    ).fillna(0.0)

    return conditions, score, exposures


def absolute_turnover(previous_weights, previous_exposure, target_weights, exposure):
    """Return dollars traded for one long or short book."""
    tickers = previous_weights.index.union(target_weights.index)
    previous = previous_weights.reindex(tickers, fill_value=0) * previous_exposure
    target = target_weights.reindex(tickers, fill_value=0) * exposure
    return float((target - previous).abs().sum())


def run_exposure_path(
    factor,
    prices,
    membership,
    exposure,
    start_position,
    end_position,
    offset,
):
    """Trade Q5-Q1 at a point-in-time exposure with exact scaled turnover."""
    signal = factor.shift(1)
    previous_weights = {
        1: pd.Series(dtype=float),
        5: pd.Series(dtype=float),
    }
    previous_exposure = 0.0
    rows = []

    for position in range(start_position + offset, end_position + 1, HORIZON):
        actual_horizon = min(HORIZON, end_position - position)
        if actual_horizon <= 0 or position + actual_horizon >= len(prices):
            break

        date = prices.index[position]
        current_exposure = float(exposure.reindex([date]).fillna(0).iloc[0])
        scores = signal.iloc[position]
        eligible = (
            scores.notna()
            & membership.iloc[position].fillna(False)
            & prices.iloc[position].notna()
            & (prices.iloc[position] > 0)
        )
        if eligible.sum() < MIN_PORTFOLIO_ASSETS:
            continue

        ranked = scores.loc[eligible].rank(method="first", pct=True)
        labels = np.ceil(ranked * 5).clip(1, 5).astype(int)
        row = {
            "date": date,
            "end_date": prices.index[position + actual_horizon],
            "holding_days": actual_horizon,
            "exposure": current_exposure,
            "eligible_assets": int(eligible.sum()),
        }
        dollar_turnover = 0.0

        for quantile in (1, 5):
            tickers = labels.index[labels == quantile]
            target_weights = pd.Series(1 / len(tickers), index=tickers)
            asset_returns, _, _ = realized_asset_returns(
                prices,
                position,
                actual_horizon,
                tickers,
            )
            row[f"q{quantile}_return"] = float(
                (target_weights * asset_returns).sum()
            )
            dollar_turnover += absolute_turnover(
                previous_weights[quantile],
                previous_exposure,
                target_weights,
                current_exposure,
            )

            if current_exposure > 0:
                previous_weights[quantile] = drift_weights(
                    target_weights,
                    asset_returns,
                )
            else:
                previous_weights[quantile] = pd.Series(dtype=float)

        previous_exposure = current_exposure
        row["spread_gross_return"] = current_exposure * (
            row["q5_return"] - row["q1_return"]
        )
        row["dollar_turnover"] = dollar_turnover
        for cost_bps in COSTS_BPS:
            row[f"spread_net_{cost_bps}bps_return"] = (
                row["spread_gross_return"]
                - dollar_turnover * cost_bps / 10_000
            )
        rows.append(row)

    return pd.DataFrame(rows)


def compound_path(path, cost_bps):
    column = f"spread_net_{cost_bps}bps_return"
    growth = (1 + path[column]).prod()
    final_exposure = float(path.iloc[-1]["exposure"])
    liquidation_cost = 2 * final_exposure * cost_bps / 10_000
    return growth * (1 - liquidation_cost) - 1


def state_difference_test(ic, state):
    data = pd.concat([ic.rename("ic"), state.rename("state")], axis=1).dropna()
    if len(data) < 30 or data["state"].nunique() < 2:
        return {
            "observations": len(data),
            "state_0_mean_ic": np.nan,
            "state_1_mean_ic": np.nan,
            "difference": np.nan,
            "hac_tstat": np.nan,
            "pvalue": np.nan,
        }

    state_values = data["state"].astype(float)
    fitted = sm.OLS(
        data["ic"].astype(float),
        sm.add_constant(state_values),
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HORIZON - 1, "use_correction": True},
    )
    return {
        "observations": len(data),
        "state_0_mean_ic": data.loc[state_values == 0, "ic"].mean(),
        "state_1_mean_ic": data.loc[state_values == 1, "ic"].mean(),
        "difference": float(fitted.params.iloc[1]),
        "hac_tstat": float(fitted.tvalues.iloc[1]),
        "pvalue": float(fitted.pvalues.iloc[1]),
    }


def summarize_conditional_ic(ic, conditions, score):
    tests = []
    states = {
        **{column: conditions[column] for column in conditions},
        "high_opportunity_score": score >= 3,
    }
    for name, state in states.items():
        tests.append({"condition": name, **state_difference_test(ic, state)})

    summary = pd.DataFrame(tests)
    summary["pvalue_holm"] = multipletests(summary["pvalue"], method="holm")[1]
    summary["pvalue_fdr_bh"] = multipletests(
        summary["pvalue"], method="fdr_bh"
    )[1]
    return summary


def opportunity_regression(ic, indicators):
    data = pd.concat([ic.rename("ic"), indicators], axis=1).dropna()
    predictors = list(indicators.columns)
    standardized = (data[predictors] - data[predictors].mean()) / data[
        predictors
    ].std()
    fitted = sm.OLS(data["ic"], sm.add_constant(standardized)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HORIZON - 1, "use_correction": True},
    )
    rows = []
    for predictor in predictors:
        rows.append(
            {
                "predictor": predictor,
                "standardized_coefficient": float(fitted.params[predictor]),
                "hac_tstat": float(fitted.tvalues[predictor]),
                "pvalue": float(fitted.pvalues[predictor]),
                "observations": int(fitted.nobs),
                "model_r_squared": float(fitted.rsquared),
            }
        )
    result = pd.DataFrame(rows)
    result["pvalue_holm"] = multipletests(result["pvalue"], method="holm")[1]
    result["pvalue_fdr_bh"] = multipletests(
        result["pvalue"], method="fdr_bh"
    )[1]
    return result


def summarize_portfolios(annual, final_data_year):
    complete = annual[annual["oos_year"] < final_data_year]
    rows = []

    for (scheme, offset), data in complete.groupby(["scheme", "offset"]):
        row = {
            "scheme": scheme,
            "offset": offset,
            "complete_oos_years": len(data),
            "average_exposure": data["average_exposure"].mean(),
            "active_period_rate": data["active_period_rate"].mean(),
            "average_dollar_turnover": data["average_dollar_turnover"].mean(),
        }
        for cost_bps in COSTS_BPS:
            values = data[f"net_{cost_bps}bps_return"]
            test = ttest_1samp(values, popmean=0)
            row[f"net_{cost_bps}bps_mean_annual_return"] = values.mean()
            row[f"net_{cost_bps}bps_annual_volatility"] = values.std()
            row[f"net_{cost_bps}bps_sharpe"] = (
                values.mean() / values.std() if values.std() else np.nan
            )
            row[f"net_{cost_bps}bps_tstat"] = float(test.statistic)
            row[f"net_{cost_bps}bps_pvalue"] = float(test.pvalue)
            row[f"net_{cost_bps}bps_positive_year_rate"] = (values > 0).mean()
        rows.append(row)

    summary = pd.DataFrame(rows)
    for cost_bps in COSTS_BPS:
        pvalue_column = f"net_{cost_bps}bps_pvalue"
        summary[f"net_{cost_bps}bps_pvalue_fdr_bh"] = multipletests(
            summary[pvalue_column], method="fdr_bh"
        )[1]

    always = complete[complete["scheme"] == "always_active"][
        ["oos_year", "offset", "net_10bps_return"]
    ].rename(columns={"net_10bps_return": "always_net_10bps_return"})
    comparison = complete.merge(always, on=["oos_year", "offset"], how="left")
    comparison["delta_vs_always"] = (
        comparison["net_10bps_return"]
        - comparison["always_net_10bps_return"]
    )
    delta_rows = []
    for (scheme, offset), data in comparison.groupby(["scheme", "offset"]):
        delta = data["delta_vs_always"].dropna()
        test = ttest_1samp(delta, popmean=0) if len(delta) >= 3 else None
        delta_rows.append(
            {
                "scheme": scheme,
                "offset": offset,
                "net_10bps_mean_delta_vs_always": delta.mean(),
                "delta_tstat": float(test.statistic) if test else np.nan,
                "delta_pvalue": float(test.pvalue) if test else np.nan,
            }
        )
    deltas = pd.DataFrame(delta_rows)
    valid = (deltas["scheme"] != "always_active") & deltas["delta_pvalue"].notna()
    deltas["delta_pvalue_fdr_bh"] = np.nan
    deltas.loc[valid, "delta_pvalue_fdr_bh"] = multipletests(
        deltas.loc[valid, "delta_pvalue"], method="fdr_bh"
    )[1]
    summary = summary.merge(deltas, on=["scheme", "offset"], how="left")
    return summary


def aggregate_phases(summary):
    rows = []
    for scheme, data in summary.groupby("scheme"):
        rows.append(
            {
                "scheme": scheme,
                "calendar_phases": len(data),
                "median_average_exposure": data["average_exposure"].median(),
                "median_active_period_rate": data["active_period_rate"].median(),
                "median_dollar_turnover": data["average_dollar_turnover"].median(),
                "gross_median_annual_return": data[
                    "net_0bps_mean_annual_return"
                ].median(),
                "gross_median_sharpe": data["net_0bps_sharpe"].median(),
                "net_10bps_median_annual_return": data[
                    "net_10bps_mean_annual_return"
                ].median(),
                "net_10bps_worst_phase_annual_return": data[
                    "net_10bps_mean_annual_return"
                ].min(),
                "net_10bps_median_sharpe": data["net_10bps_sharpe"].median(),
                "net_10bps_positive_phase_rate": (
                    data["net_10bps_mean_annual_return"] > 0
                ).mean(),
                "net_10bps_median_delta_vs_always": data[
                    "net_10bps_mean_delta_vs_always"
                ].median(),
                "net_25bps_median_annual_return": data[
                    "net_25bps_mean_annual_return"
                ].median(),
            }
        )
    return pd.DataFrame(rows)


def plot_results(indicators, phase_summary):
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    columns = [
        ("risk_free_rate_pct", "3-month Treasury rate (%)"),
        ("median_stock_volatility", "Median stock volatility"),
        ("cross_sectional_dispersion", "Cross-sectional dispersion"),
        ("average_correlation_proxy", "Average correlation proxy"),
    ]
    for axis, (column, title) in zip(axes.flat, columns):
        axis.plot(indicators.index, indicators[column], linewidth=0.8)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.suptitle("Lagged point-in-time market-opportunity indicators")
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "opportunity_indicators.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    data = phase_summary.set_index("scheme")
    figure, axis = plt.subplots(figsize=(11, 6))
    x = np.arange(len(data))
    width = 0.36
    axis.bar(
        x - width / 2,
        data["gross_median_annual_return"],
        width,
        label="Gross",
    )
    axis.bar(
        x + width / 2,
        data["net_10bps_median_annual_return"],
        width,
        label="Net 10 bps",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [label.replace("_", " ").title() for label in data.index]
    )
    axis.set_ylabel("Median mean annual return across 21 phases")
    axis.set_title("Fixed composite with point-in-time opportunity exposure")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "opportunity_portfolio_comparison.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def run_market_opportunity_research(force_rate_download=False):
    if not os.path.exists(IC_CACHE_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Run walk_forward.py before market opportunity research")

    daily_ic_cache = pd.read_parquet(IC_CACHE_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    returns, availability, _, prices = load_data()
    membership = load_membership().reindex(index=prices.index, columns=prices.columns)
    risk_free_rate = ensure_risk_free_rate(
        prices.index.min(),
        prices.index.max(),
        force_download=force_rate_download,
    )
    indicators = build_opportunity_indicators(
        returns,
        availability,
        risk_free_rate,
    )
    forward_returns = compute_forward_returns(prices, HORIZON)

    threshold_rows = []
    state_rows = []
    ic_rows = []
    annual_rows = []

    for year in range(FIRST_OOS_YEAR, prices.index.max().year + 1):
        selected_rows = []
        for family in COMPONENT_FAMILIES:
            _, selected = score_training_candidates(
                daily_ic_cache,
                metadata,
                family,
                year,
            )
            selected_rows.append(selected)

        thresholds = training_thresholds(indicators, prices.index, year)
        threshold_rows.append(thresholds)
        year_indicators = indicators.loc[f"{year}-01-01":f"{year}-12-31"]
        conditions, score, exposures = apply_thresholds(year_indicators, thresholds)
        year_states = pd.concat(
            [
                year_indicators,
                conditions,
                score,
                exposures.add_prefix("exposure_"),
            ],
            axis=1,
        )
        year_states["oos_year"] = year
        state_rows.append(year_states)

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
        year_ic = compute_daily_spearman_ic(
            factor,
            forward_returns.loc[sliced_prices.index],
            sliced_membership,
        )["ic"].loc[f"{year}-01-01":f"{year}-12-31"]
        for date, value in year_ic.items():
            ic_rows.append({"date": date, "oos_year": year, "ic": value})

        for scheme in SCHEMES:
            exposure = exposures[scheme]
            for offset in range(CALENDAR_PHASES):
                if start_position + offset > end_position:
                    continue
                path = run_exposure_path(
                    factor,
                    sliced_prices,
                    sliced_membership,
                    exposure,
                    start_position,
                    end_position,
                    offset,
                )
                if path.empty:
                    continue
                row = {
                    "oos_year": year,
                    "scheme": scheme,
                    "offset": offset,
                    "holding_periods": len(path),
                    "average_exposure": path["exposure"].mean(),
                    "active_period_rate": (path["exposure"] > 0).mean(),
                    "average_dollar_turnover": path["dollar_turnover"].mean(),
                }
                for cost_bps in COSTS_BPS:
                    row[f"net_{cost_bps}bps_return"] = compound_path(
                        path,
                        cost_bps,
                    )
                annual_rows.append(row)

        print(f"Market-opportunity OOS year complete: {year}")

    thresholds = pd.DataFrame(threshold_rows)
    states = pd.concat(state_rows).sort_index()
    ic_history = pd.DataFrame(ic_rows).set_index("date").sort_index()
    annual = pd.DataFrame(annual_rows)
    conditional_ic = summarize_conditional_ic(
        ic_history["ic"],
        states[
            [
                "rate_above_2pct",
                "moderately_high_stock_volatility",
                "high_dispersion",
                "low_correlation",
            ]
        ],
        states["opportunity_score"],
    )
    regression = opportunity_regression(
        ic_history["ic"],
        states[
            [
                "risk_free_rate_pct",
                "median_stock_volatility",
                "cross_sectional_dispersion",
                "average_correlation_proxy",
            ]
        ],
    )
    portfolio_summary = summarize_portfolios(
        annual,
        final_data_year=int(prices.index.max().year),
    )
    phase_summary = aggregate_phases(portfolio_summary)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    indicators.to_parquet(os.path.join(OUTPUT_DIR, "opportunity_indicators.parquet"))
    thresholds.to_csv(os.path.join(OUTPUT_DIR, "annual_thresholds.csv"), index=False)
    states.to_parquet(os.path.join(OUTPUT_DIR, "stitched_oos_states.parquet"))
    ic_history.to_parquet(os.path.join(OUTPUT_DIR, "stitched_raw_equal_oos_ic.parquet"))
    conditional_ic.to_csv(os.path.join(OUTPUT_DIR, "conditional_ic.csv"), index=False)
    regression.to_csv(os.path.join(OUTPUT_DIR, "opportunity_ic_regression.csv"), index=False)
    annual.to_parquet(os.path.join(OUTPUT_DIR, "annual_phase_returns.parquet"))
    portfolio_summary.to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_statistics.csv"), index=False
    )
    phase_summary.to_csv(os.path.join(OUTPUT_DIR, "phase_summary.csv"), index=False)
    plot_results(indicators, phase_summary)

    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "factor": "raw_equal composite selected walk-forward",
                "component_families": list(COMPONENT_FAMILIES),
                "horizon_days": HORIZON,
                "lookback_years": LOOKBACK_YEARS,
                "purge_days": HORIZON,
                "calendar_phases": CALENDAR_PHASES,
                "rate_series": "FRED DGS3MO",
                "rate_threshold_pct": RATE_THRESHOLD_PCT,
                "volatility_window": VOLATILITY_WINDOW,
                "volatility_training_quantile_range": [
                    VOLATILITY_LOWER_QUANTILE,
                    VOLATILITY_UPPER_QUANTILE,
                ],
                "dispersion_smoothing_days": DISPERSION_SMOOTHING,
                "correlation_proxy_smoothing_days": CORRELATION_SMOOTHING,
                "schemes": list(SCHEMES),
                "costs_bps_per_dollar_traded": list(COSTS_BPS),
                "cash_carry_included": False,
                "development_backtest": True,
                "current_oos_year_used_for_thresholds": False,
            },
            file,
            indent=2,
        )

    return conditional_ic, regression, portfolio_summary, phase_summary


def print_results(conditional_ic, regression, phase_summary):
    print("\nMARKET OPPORTUNITY: CONDITIONAL OOS IC")
    print(
        conditional_ic.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nMARKET OPPORTUNITY: MULTIVARIATE OOS IC REGRESSION")
    print(
        regression.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nMARKET OPPORTUNITY: PORTFOLIO RESULTS ACROSS 21 PHASES")
    print(
        phase_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    results = run_market_opportunity_research()
    print_results(results[0], results[1], results[3])
