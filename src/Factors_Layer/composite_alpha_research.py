"""Final composite-alpha test using only training-time signal decisions.

Three economically distinct families are combined on one common 21-day horizon:
Trend Slope, Short-Term Reversal, and Liquidity Change. Every OOS year gets its
own purged parameter selection and shrunk IC weights. Results remain a
development backtest because the component families were chosen after earlier
sample inspection.
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

from factor_independence import residualize_cross_sectionally
from pipeline import build_factor, load_data, load_membership
from quantile_research import run_one_quantile_path
from statistical_research import compute_daily_spearman_ic, compute_forward_returns, hac_mean_test
from walk_forward import (
    FACTOR_CACHE_DIR,
    IC_CACHE_PATH,
    METADATA_PATH,
    purged_training_metrics,
)


data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data_System")
)
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import BASE_DIR


OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "Factor_Research", "composite_stage")
COMPONENT_FAMILIES = ("trend_slope", "short_term_reversal", "liquidity_change")
HORIZON = 21
LOOKBACK_YEARS = 5
FIRST_OOS_YEAR = 2015
CALENDAR_PHASES = 21
IC_SHRINKAGE_TO_EQUAL = 0.50
COSTS_BPS = (10, 25)

COMPOSITE_VARIANTS = (
    "raw_equal",
    "residual_equal",
    "residual_ic_shrunk",
    "raw_equal_inverse_vol",
    "residual_equal_inverse_vol",
    "residual_ic_shrunk_inverse_vol",
)


def factor_cache_path(family, variant):
    return os.path.join(FACTOR_CACHE_DIR, f"{family}__{variant}.parquet")


def score_training_candidates(daily_ic, metadata, family, year):
    """Select one 21-day specification using only information known then."""
    hypotheses = metadata[
        (metadata["family"] == family)
        & (metadata["horizon_days"] == HORIZON)
    ]
    selection_date = pd.Timestamp(f"{year - 1}-12-31")
    train_start = pd.Timestamp(f"{year - LOOKBACK_YEARS}-01-01")
    rows = []

    for hypothesis in hypotheses.itertuples():
        metrics = purged_training_metrics(
            daily_ic[hypothesis.key],
            daily_ic.index,
            selection_date,
            HORIZON,
            train_start,
        )
        if metrics is None:
            continue
        rows.append(
            {
                "oos_year": year,
                "family": family,
                "variant": hypothesis.variant,
                "key": hypothesis.key,
                **metrics,
            }
        )

    candidates = pd.DataFrame(rows)
    candidates["selection_score"] = (
        0.25 * candidates["late_mean_ic"].rank(pct=True)
        + 0.20 * candidates["late_nw_tstat"].rank(pct=True)
        + 0.15 * candidates["mean_ic"].rank(pct=True)
        + 0.15 * candidates["nw_tstat"].rank(pct=True)
        + 0.10 * candidates["positive_rate"].rank(pct=True)
        + 0.05 * candidates["early_mean_ic"].rank(pct=True)
        + 0.10 * (-candidates["stability_gap"]).rank(pct=True)
    )
    selected = candidates.sort_values(
        ["eligible", "selection_score"],
        ascending=[False, False],
    ).iloc[0]
    return candidates, selected


def shrunk_ic_weights(selected_rows):
    """Shrink positive historical Mean IC weights halfway toward equal."""
    equal = pd.Series(
        1 / len(COMPONENT_FAMILIES),
        index=COMPONENT_FAMILIES,
    )
    strengths = pd.Series(
        {
            row.family: (
                max(float(row.mean_ic), 0)
                if bool(row.eligible)
                else 0
            )
            for row in selected_rows
        }
    ).reindex(COMPONENT_FAMILIES)

    if strengths.sum() <= 0:
        data_weight = equal
    else:
        data_weight = strengths / strengths.sum()

    return (
        IC_SHRINKAGE_TO_EQUAL * equal
        + (1 - IC_SHRINKAGE_TO_EQUAL) * data_weight
    )


def combine_factors(factors, weights, availability):
    combined = None

    for family, weight in weights.items():
        contribution = factors[family] * weight
        combined = contribution if combined is None else combined + contribution

    return build_factor(combined, availability)


def build_year_composites(
    selected_rows,
    prices,
    returns,
    availability,
    membership,
    year,
):
    year_dates = prices.loc[f"{year}-01-01" : f"{year}-12-31"].index
    if year_dates.empty:
        return {}, None, None
    start_position = prices.index.get_loc(year_dates[0])
    end_position = prices.index.get_loc(year_dates[-1])
    slice_start = max(0, start_position - 1)
    date_slice = prices.index[slice_start : end_position + 1]
    component_factors = {}

    for row in selected_rows:
        factor = pd.read_parquet(factor_cache_path(row.family, row.variant))
        component_factors[row.family] = factor.loc[date_slice].astype(float)

    sliced_membership = membership.loc[date_slice]
    sliced_availability = availability.loc[date_slice]
    trend = component_factors["trend_slope"]
    residual_factors = {
        "trend_slope": trend,
        "short_term_reversal": residualize_cross_sectionally(
            component_factors["short_term_reversal"],
            [trend],
            sliced_membership,
        ),
        "liquidity_change": residualize_cross_sectionally(
            component_factors["liquidity_change"],
            [trend],
            sliced_membership,
        ),
    }
    equal_weights = pd.Series(
        1 / len(COMPONENT_FAMILIES),
        index=COMPONENT_FAMILIES,
    )
    ic_weights = shrunk_ic_weights(selected_rows)
    raw_equal = combine_factors(
        component_factors,
        equal_weights,
        sliced_availability,
    )
    residual_equal = combine_factors(
        residual_factors,
        equal_weights,
        sliced_availability,
    )
    residual_ic_shrunk = combine_factors(
        residual_factors,
        ic_weights,
        sliced_availability,
    )

    volatility = returns.rolling(60, min_periods=40).std() * np.sqrt(252)
    volatility = volatility.loc[date_slice].where(sliced_availability)

    def inverse_vol_scale(factor):
        inverse_volatility = 1 / volatility.replace(0, np.nan)
        lower = inverse_volatility.quantile(0.10, axis=1)
        upper = inverse_volatility.quantile(0.90, axis=1)
        capped = inverse_volatility.clip(lower=lower, upper=upper, axis=0)
        return build_factor(factor * capped, sliced_availability)

    sliced_composites = {
        "raw_equal": raw_equal,
        "residual_equal": residual_equal,
        "residual_ic_shrunk": residual_ic_shrunk,
        "raw_equal_inverse_vol": inverse_vol_scale(raw_equal),
        "residual_equal_inverse_vol": inverse_vol_scale(residual_equal),
        "residual_ic_shrunk_inverse_vol": inverse_vol_scale(residual_ic_shrunk),
    }
    return (
        sliced_composites,
        prices.loc[date_slice],
        membership.loc[date_slice],
        1,
        len(date_slice) - 1,
    )


def compound_net_return(path, cost_bps):
    growth = (
        1
        + path["spread_gross_return"]
        - 2 * path["spread_turnover"] * cost_bps / 10_000
    ).prod()
    liquidation_cost = 2 * cost_bps / 10_000
    return growth * (1 - liquidation_cost) - 1


def evaluate_year(
    year,
    composites,
    start_position,
    end_position,
    prices,
    membership,
    forward_returns,
):
    ic_rows = []
    portfolio_rows = []

    for composite_name, factor in composites.items():
        ic = compute_daily_spearman_ic(
            factor,
            forward_returns,
            membership,
        )["ic"].loc[f"{year}-01-01" : f"{year}-12-31"]
        for date, value in ic.items():
            ic_rows.append(
                {
                    "date": date,
                    "oos_year": year,
                    "composite": composite_name,
                    "ic": value,
                }
            )

        for offset in range(CALENDAR_PHASES):
            phase_start = start_position + offset
            if phase_start > end_position:
                continue
            path = run_one_quantile_path(
                factor,
                prices,
                membership,
                HORIZON,
                0,
                composite_name,
                start_position=phase_start,
                end_position=end_position,
                allow_partial_final_period=True,
            )
            if path.empty:
                continue
            row = {
                "oos_year": year,
                "composite": composite_name,
                "offset": offset,
                "holding_periods": len(path),
                "calendar_days_covered": (
                    path["end_date"].max() - path["date"].min()
                ).days,
                "average_turnover": path["spread_turnover"].mean(),
                "gross_return": compound_net_return(path, 0),
            }
            for cost_bps in COSTS_BPS:
                row[f"net_{cost_bps}bps_return"] = compound_net_return(
                    path,
                    cost_bps,
                )
            portfolio_rows.append(row)

    return ic_rows, portfolio_rows


def summarize_stitched_ic(ic_history):
    rows = []

    for composite, data in ic_history.groupby("composite"):
        values = data.sort_values("date").set_index("date")["ic"].dropna()
        _, tstat, pvalue = hac_mean_test(values, HORIZON - 1)
        yearly = data.groupby("oos_year")["ic"].mean()
        rows.append(
            {
                "composite": composite,
                "observations": len(values),
                "mean_oos_ic": values.mean(),
                "hac_tstat": tstat,
                "pvalue": pvalue,
                "positive_rate": (values > 0).mean(),
                "positive_year_rate": (yearly > 0).mean(),
                "worst_year_mean_ic": yearly.min(),
                "best_year_mean_ic": yearly.max(),
            }
        )

    summary = pd.DataFrame(rows)
    summary["pvalue_holm"] = multipletests(summary["pvalue"], method="holm")[1]
    summary["pvalue_fdr_bh"] = multipletests(
        summary["pvalue"],
        method="fdr_bh",
    )[1]
    return summary


def summarize_portfolios(annual_portfolios):
    complete = annual_portfolios[annual_portfolios["oos_year"] < 2026]
    rows = []

    for (composite, offset), data in complete.groupby(["composite", "offset"]):
        row = {
            "composite": composite,
            "offset": offset,
            "complete_oos_years": len(data),
            "average_turnover": data["average_turnover"].mean(),
        }
        for cost_bps in COSTS_BPS:
            values = data[f"net_{cost_bps}bps_return"]
            test = ttest_1samp(values, popmean=0)
            row[f"net_{cost_bps}bps_mean_annual_return"] = values.mean()
            row[f"net_{cost_bps}bps_annual_volatility"] = values.std()
            row[f"net_{cost_bps}bps_sharpe"] = (
                values.mean() / values.std() if values.std() else np.nan
            )
            row[f"net_{cost_bps}bps_positive_year_rate"] = (values > 0).mean()
            row[f"net_{cost_bps}bps_tstat"] = float(test.statistic)
            row[f"net_{cost_bps}bps_pvalue"] = float(test.pvalue)
        rows.append(row)

    summary = pd.DataFrame(rows)
    for cost_bps in COSTS_BPS:
        pvalue_column = f"net_{cost_bps}bps_pvalue"
        summary[f"net_{cost_bps}bps_pvalue_holm"] = multipletests(
            summary[pvalue_column],
            method="holm",
        )[1]
        summary[f"net_{cost_bps}bps_pvalue_fdr_bh"] = multipletests(
            summary[pvalue_column],
            method="fdr_bh",
        )[1]
    return summary


def aggregate_phases(portfolio_summary):
    rows = []

    for composite, data in portfolio_summary.groupby("composite"):
        row = {
            "composite": composite,
            "calendar_phases": len(data),
            "median_turnover": data["average_turnover"].median(),
        }
        for cost_bps in COSTS_BPS:
            return_column = f"net_{cost_bps}bps_mean_annual_return"
            sharpe_column = f"net_{cost_bps}bps_sharpe"
            row[f"net_{cost_bps}bps_median_annual_return"] = data[
                return_column
            ].median()
            row[f"net_{cost_bps}bps_worst_phase_annual_return"] = data[
                return_column
            ].min()
            row[f"net_{cost_bps}bps_median_sharpe"] = data[
                sharpe_column
            ].median()
            row[f"net_{cost_bps}bps_positive_phase_rate"] = (
                data[return_column] > 0
            ).mean()
            row[f"net_{cost_bps}bps_significant_fdr_phase_rate"] = (
                data[f"net_{cost_bps}bps_pvalue_fdr_bh"] < 0.05
            ).mean()
        rows.append(row)

    return pd.DataFrame(rows)


def plot_composite_results(phase_summary):
    data = phase_summary.sort_values("net_10bps_median_sharpe")
    labels = [name.replace("_", " ").title() for name in data["composite"]]
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    axes[0].barh(labels, data["net_10bps_median_annual_return"])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Median annual Q5-Q1 return")
    axes[0].set_xlabel("Net return after 10 bps costs")
    axes[1].barh(labels, data["net_10bps_median_sharpe"])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Median annual Sharpe")
    axes[1].set_xlabel("Net Sharpe after 10 bps costs")
    figure.suptitle("Walk-forward composite alpha across 21 calendar phases")
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "composite_phase_summary.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def run_composite_research():
    if not os.path.exists(IC_CACHE_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Run walk_forward.py before composite research")
    daily_ic = pd.read_parquet(IC_CACHE_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    returns, availability, _, prices = load_data()
    membership = load_membership().reindex(index=prices.index, columns=prices.columns)
    forward_returns = compute_forward_returns(prices, HORIZON)
    candidate_rows = []
    selection_rows = []
    weight_rows = []
    ic_rows = []
    portfolio_rows = []

    for year in range(FIRST_OOS_YEAR, prices.index.max().year + 1):
        selected_rows = []
        for family in COMPONENT_FAMILIES:
            candidates, selected = score_training_candidates(
                daily_ic,
                metadata,
                family,
                year,
            )
            candidate_rows.append(candidates)
            selection_rows.append(selected.to_frame().T)
            selected_rows.append(selected)

        weights = shrunk_ic_weights(selected_rows)
        for family in COMPONENT_FAMILIES:
            weight_rows.append(
                {
                    "oos_year": year,
                    "family": family,
                    "equal_weight": 1 / len(COMPONENT_FAMILIES),
                    "shrunk_ic_weight": weights[family],
                }
            )
        (
            composites,
            sliced_prices,
            sliced_membership,
            start_position,
            end_position,
        ) = build_year_composites(
            selected_rows,
            prices,
            returns,
            availability,
            membership,
            year,
        )
        year_ic, year_portfolios = evaluate_year(
            year,
            composites,
            start_position,
            end_position,
            sliced_prices,
            sliced_membership,
            forward_returns.loc[sliced_prices.index],
        )
        ic_rows.extend(year_ic)
        portfolio_rows.extend(year_portfolios)
        print(f"Composite OOS year complete: {year}")

    candidates = pd.concat(candidate_rows, ignore_index=True)
    selections = pd.concat(selection_rows, ignore_index=True)
    weights = pd.DataFrame(weight_rows)
    ic_history = pd.DataFrame(ic_rows)
    annual_portfolios = pd.DataFrame(portfolio_rows)
    ic_summary = summarize_stitched_ic(ic_history)
    portfolio_summary = summarize_portfolios(annual_portfolios)
    phase_summary = aggregate_phases(portfolio_summary)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    candidates.to_parquet(os.path.join(OUTPUT_DIR, "training_candidate_scores.parquet"))
    selections.to_csv(os.path.join(OUTPUT_DIR, "annual_component_selections.csv"), index=False)
    weights.to_csv(os.path.join(OUTPUT_DIR, "annual_composite_weights.csv"), index=False)
    ic_history.to_parquet(os.path.join(OUTPUT_DIR, "stitched_composite_oos_ic.parquet"))
    ic_summary.to_csv(os.path.join(OUTPUT_DIR, "composite_oos_ic_summary.csv"), index=False)
    annual_portfolios.to_parquet(
        os.path.join(OUTPUT_DIR, "composite_annual_phase_returns.parquet")
    )
    portfolio_summary.to_csv(
        os.path.join(OUTPUT_DIR, "composite_portfolio_statistics.csv"),
        index=False,
    )
    phase_summary.to_csv(
        os.path.join(OUTPUT_DIR, "composite_phase_summary.csv"),
        index=False,
    )
    plot_composite_results(phase_summary)

    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "component_families": list(COMPONENT_FAMILIES),
                "common_horizon_days": HORIZON,
                "lookback_years": LOOKBACK_YEARS,
                "purge_days": HORIZON,
                "calendar_phases": CALENDAR_PHASES,
                "ic_shrinkage_to_equal": IC_SHRINKAGE_TO_EQUAL,
                "costs_bps": list(COSTS_BPS),
                "test_used_for_parameter_or_weight_selection": False,
                "development_backtest": True,
                "composite_variants": list(COMPOSITE_VARIANTS),
            },
            file,
            indent=2,
        )

    return selections, weights, ic_summary, portfolio_summary, phase_summary


def print_composite_results(ic_summary, phase_summary):
    print("\nCOMPOSITE: STITCHED WALK-FORWARD OOS IC")
    print(
        ic_summary.sort_values("mean_oos_ic", ascending=False).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    columns = [
        "composite",
        "net_10bps_median_annual_return",
        "net_10bps_worst_phase_annual_return",
        "net_10bps_median_sharpe",
        "net_10bps_positive_phase_rate",
        "net_25bps_median_annual_return",
        "net_25bps_median_sharpe",
    ]
    print("\nCOMPOSITE: PORTFOLIO RESULTS ACROSS 21 PHASES")
    print(
        phase_summary[columns].sort_values(
            "net_10bps_median_sharpe",
            ascending=False,
        ).to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    _, _, ic_summary, _, phase_summary = run_composite_research()
    print_composite_results(ic_summary, phase_summary)
