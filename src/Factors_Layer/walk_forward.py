"""Stage 5: purged annual walk-forward selection and OOS factor portfolios."""

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

from candidate_research import CANDIDATE_GRID, build_candidate, parameter_text
from quantile_research import run_one_quantile_path
from research import PARAMETER_GRID, build_factor_variant
from statistical_research import compute_daily_spearman_ic, compute_forward_returns, hac_mean_test
from pipeline import load_data, load_membership


data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data_System")
)
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import BASE_DIR, VOLUME_PATH


OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "Factor_Research", "walk_forward_stage")
FACTOR_CACHE_DIR = os.path.join(OUTPUT_DIR, "factor_cache")
IC_CACHE_PATH = os.path.join(OUTPUT_DIR, "daily_ic_cache.parquet")
METADATA_PATH = os.path.join(OUTPUT_DIR, "hypothesis_metadata.csv")
HORIZONS = (5, 21, 63, 126)
LOOKBACK_YEARS = 5
FIRST_OOS_YEAR = 2015
TRANSACTION_COST_BPS = 10


def hypothesis_key(family, variant, horizon):
    return f"{family}|{variant}|h{horizon}"


def factor_cache_path(family, variant):
    safe_name = f"{family}__{variant}.parquet"
    return os.path.join(FACTOR_CACHE_DIR, safe_name)


def all_specifications():
    specifications = []

    for family, variants in PARAMETER_GRID.items():
        for parameters in variants:
            specifications.append(
                {
                    "source": "baseline_grid",
                    "family": family,
                    "variant": parameters["name"],
                    "parameters": parameters,
                }
            )

    for family, variants in CANDIDATE_GRID.items():
        for parameters in variants:
            specifications.append(
                {
                    "source": "candidate_grid",
                    "family": family,
                    "variant": parameters["variant"],
                    "parameters": parameters,
                }
            )

    return specifications


def build_specification(
    specification,
    returns,
    prices,
    volume,
    availability,
):
    if specification["source"] == "baseline_grid":
        return build_factor_variant(
            specification["family"],
            specification["parameters"],
            returns,
            prices,
            availability,
        )

    return build_candidate(
        specification["family"],
        specification["parameters"],
        returns,
        prices,
        volume,
        availability,
    )


def build_research_cache(returns, prices, volume, availability, membership):
    """Calculate every factor matrix once and all 224 daily IC histories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FACTOR_CACHE_DIR, exist_ok=True)
    forward_returns = {
        horizon: compute_forward_returns(prices, horizon)
        for horizon in HORIZONS
    }
    ic_columns = {}
    metadata_rows = []

    for specification in all_specifications():
        family = specification["family"]
        variant = specification["variant"]
        print(f"Walk-forward cache: {family} | {variant}")
        factor = build_specification(
            specification,
            returns,
            prices,
            volume,
            availability,
        )
        factor.astype("float32").to_parquet(factor_cache_path(family, variant))

        for horizon in HORIZONS:
            key = hypothesis_key(family, variant, horizon)
            ic_columns[key] = compute_daily_spearman_ic(
                factor,
                forward_returns[horizon],
                membership,
            )["ic"].astype("float32")
            parameters = specification["parameters"]
            metadata_rows.append(
                {
                    "key": key,
                    "source": specification["source"],
                    "family": family,
                    "variant": variant,
                    "horizon_days": horizon,
                    "parameters": (
                        parameter_text(parameters)
                        if specification["source"] == "candidate_grid"
                        else ", ".join(
                            f"{name}={value}"
                            for name, value in parameters.items()
                            if name != "name"
                        )
                    ),
                }
            )

    daily_ic = pd.DataFrame(ic_columns)
    metadata = pd.DataFrame(metadata_rows)
    daily_ic.to_parquet(IC_CACHE_PATH)
    metadata.to_csv(METADATA_PATH, index=False)
    return daily_ic, metadata


def load_or_build_cache(returns, prices, volume, availability, membership):
    expected_hypotheses = len(all_specifications()) * len(HORIZONS)

    if os.path.exists(IC_CACHE_PATH) and os.path.exists(METADATA_PATH):
        daily_ic = pd.read_parquet(IC_CACHE_PATH)
        metadata = pd.read_csv(METADATA_PATH)
        factor_files_exist = all(
            os.path.exists(factor_cache_path(row.family, row.variant))
            for row in metadata.drop_duplicates(["family", "variant"]).itertuples()
        )

        if daily_ic.shape[1] == expected_hypotheses and factor_files_exist:
            print("Valid walk-forward cache found -> loading")
            return daily_ic, metadata

    return build_research_cache(
        returns,
        prices,
        volume,
        availability,
        membership,
    )


def purged_training_metrics(series, index, selection_date, horizon, train_start):
    selection_position = index.searchsorted(selection_date, side="right") - 1
    purged_position = selection_position - horizon

    if purged_position < 0:
        return None

    purged_end = index[purged_position]
    values = series.loc[train_start:purged_end].dropna()

    if len(values) < 126:
        return None

    midpoint = train_start + (purged_end - train_start) / 2
    early = values.loc[:midpoint]
    late = values.loc[midpoint:]
    _, full_tstat, _ = hac_mean_test(values, horizon - 1)
    _, late_tstat, _ = hac_mean_test(late, horizon - 1)

    return {
        "observations": len(values),
        "purged_train_end": purged_end,
        "mean_ic": values.mean(),
        "nw_tstat": full_tstat,
        "positive_rate": (values > 0).mean(),
        "early_mean_ic": early.mean(),
        "late_mean_ic": late.mean(),
        "late_nw_tstat": late_tstat,
        "stability_gap": abs(early.mean() - late.mean()),
        "eligible": early.mean() > 0 and late.mean() > 0,
    }


def select_for_year(daily_ic, metadata, year):
    selection_date = pd.Timestamp(f"{year - 1}-12-31")
    train_start = pd.Timestamp(f"{year - LOOKBACK_YEARS}-01-01")
    rows = []

    for hypothesis in metadata.itertuples():
        metrics = purged_training_metrics(
            daily_ic[hypothesis.key],
            daily_ic.index,
            selection_date,
            int(hypothesis.horizon_days),
            train_start,
        )
        if metrics is None:
            continue
        rows.append(
            {
                "oos_year": year,
                "selection_date": selection_date,
                "key": hypothesis.key,
                "source": hypothesis.source,
                "family": hypothesis.family,
                "variant": hypothesis.variant,
                "horizon_days": int(hypothesis.horizon_days),
                "parameters": hypothesis.parameters,
                **metrics,
            }
        )

    candidates = pd.DataFrame(rows)
    candidates["selection_score"] = np.nan

    for _, family_data in candidates.groupby("family"):
        index = family_data.index
        score = (
            0.20 * family_data["late_mean_ic"].rank(pct=True)
            + 0.20 * family_data["late_nw_tstat"].rank(pct=True)
            + 0.15 * family_data["mean_ic"].rank(pct=True)
            + 0.15 * family_data["nw_tstat"].rank(pct=True)
            + 0.10 * family_data["positive_rate"].rank(pct=True)
            + 0.10 * family_data["early_mean_ic"].rank(pct=True)
            + 0.10 * (-family_data["stability_gap"]).rank(pct=True)
        )
        candidates.loc[index, "selection_score"] = score

    selected_rows = []
    for _, family_data in candidates.groupby("family"):
        selected_rows.append(
            family_data.sort_values(
                ["eligible", "selection_score"],
                ascending=[False, False],
            ).iloc[0]
        )

    return candidates, pd.DataFrame(selected_rows).reset_index(drop=True)


def build_oos_ic_history(daily_ic, selections):
    rows = []

    for selection in selections.itertuples():
        values = daily_ic[selection.key].loc[
            f"{selection.oos_year}-01-01" : f"{selection.oos_year}-12-31"
        ]
        for date, value in values.items():
            rows.append(
                {
                    "date": date,
                    "oos_year": selection.oos_year,
                    "family": selection.family,
                    "variant": selection.variant,
                    "horizon_days": selection.horizon_days,
                    "ic": value,
                }
            )

    return pd.DataFrame(rows)


def summarize_oos_ic(oos_ic):
    rows = []

    for family, data in oos_ic.groupby("family"):
        values = data.sort_values("date").set_index("date")["ic"].dropna()
        _, tstat, pvalue = hac_mean_test(values, max(HORIZONS) - 1)
        yearly_means = data.groupby("oos_year")["ic"].mean()
        rows.append(
            {
                "family": family,
                "observations": len(values),
                "mean_oos_ic": values.mean(),
                "conservative_hac_tstat": tstat,
                "pvalue": pvalue,
                "positive_rate": (values > 0).mean(),
                "positive_year_rate": (yearly_means > 0).mean(),
                "worst_year_mean_ic": yearly_means.min(),
                "best_year_mean_ic": yearly_means.max(),
            }
        )

    summary = pd.DataFrame(rows)
    pvalues = summary["pvalue"].to_numpy()
    summary["pvalue_holm"] = multipletests(pvalues, method="holm")[1]
    summary["pvalue_fdr_bh"] = multipletests(pvalues, method="fdr_bh")[1]
    return summary


def build_walk_forward_portfolios(selections, prices, membership):
    factor_memory = {}
    rows = []

    for selection in selections.itertuples():
        factor_key = (selection.family, selection.variant)
        if factor_key not in factor_memory:
            factor_memory[factor_key] = pd.read_parquet(
                factor_cache_path(*factor_key)
            ).astype(float)
        factor = factor_memory[factor_key]
        year_dates = prices.loc[
            f"{selection.oos_year}-01-01" : f"{selection.oos_year}-12-31"
        ].index
        if year_dates.empty:
            continue
        start_position = prices.index.get_loc(year_dates[0])
        end_position = prices.index.get_loc(year_dates[-1])
        path = run_one_quantile_path(
            factor,
            prices,
            membership,
            int(selection.horizon_days),
            0,
            selection.family,
            start_position=start_position,
            end_position=end_position,
            allow_partial_final_period=True,
        )
        if path.empty:
            continue
        path["oos_year"] = selection.oos_year
        path["variant"] = selection.variant
        rows.append(path)

    return pd.concat(rows, ignore_index=True)


def summarize_walk_forward_portfolios(paths):
    annual_rows = []

    for (family, year), data in paths.groupby(["factor", "oos_year"]):
        gross = (1 + data["spread_net_0bps_return"]).prod() - 1
        net_growth = (
            1
            + data["spread_gross_return"]
            - 2 * data["spread_turnover"] * TRANSACTION_COST_BPS / 10_000
        ).prod()
        # Close both fully invested books at the annual model-reset boundary.
        liquidation_cost = 2 * TRANSACTION_COST_BPS / 10_000
        net = net_growth * (1 - liquidation_cost) - 1
        annual_rows.append(
            {
                "family": family,
                "oos_year": year,
                "gross_return": gross,
                "net_10bps_return": net,
                "holding_periods": len(data),
                "average_turnover": data["spread_turnover"].mean(),
                "calendar_days_covered": (
                    data["end_date"].max() - data["date"].min()
                ).days,
            }
        )

    annual = pd.DataFrame(annual_rows)
    summary_rows = []
    complete_years = annual[annual["oos_year"] < 2026]

    for family, data in complete_years.groupby("family"):
        net = data["net_10bps_return"]
        test = ttest_1samp(net, popmean=0)
        tstat = float(test.statistic)
        pvalue = float(test.pvalue)
        summary_rows.append(
            {
                "family": family,
                "complete_oos_years": len(data),
                "mean_annual_net_return": net.mean(),
                "annual_volatility": net.std(),
                "annual_sharpe": net.mean() / net.std() if net.std() else np.nan,
                "annual_return_tstat": tstat,
                "pvalue": pvalue,
                "positive_year_rate": (net > 0).mean(),
                "worst_year": net.min(),
                "best_year": net.max(),
                "average_turnover": data["average_turnover"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["pvalue_holm"] = multipletests(summary["pvalue"], method="holm")[1]
    summary["pvalue_fdr_bh"] = multipletests(
        summary["pvalue"],
        method="fdr_bh",
    )[1]
    return annual, summary


def run_walk_forward():
    returns, availability, _, prices = load_data()
    membership = load_membership().reindex(index=prices.index, columns=prices.columns)
    volume = pd.read_parquet(VOLUME_PATH).reindex(
        index=prices.index,
        columns=prices.columns,
    )
    daily_ic, metadata = load_or_build_cache(
        returns,
        prices,
        volume,
        availability,
        membership,
    )
    candidate_tables = []
    selections = []

    for year in range(FIRST_OOS_YEAR, prices.index.max().year + 1):
        candidates, selected = select_for_year(daily_ic, metadata, year)
        candidate_tables.append(candidates)
        selections.append(selected)
        print(f"Walk-forward selection complete: {year}")

    all_candidates = pd.concat(candidate_tables, ignore_index=True)
    selections = pd.concat(selections, ignore_index=True)
    oos_ic = build_oos_ic_history(daily_ic, selections)
    oos_ic_summary = summarize_oos_ic(oos_ic)
    portfolio_paths = build_walk_forward_portfolios(
        selections,
        prices,
        membership,
    )
    annual_portfolios, portfolio_summary = summarize_walk_forward_portfolios(
        portfolio_paths
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_candidates.to_parquet(os.path.join(OUTPUT_DIR, "annual_candidate_scores.parquet"))
    selections.to_csv(os.path.join(OUTPUT_DIR, "annual_selections.csv"), index=False)
    oos_ic.to_parquet(os.path.join(OUTPUT_DIR, "stitched_oos_ic.parquet"))
    oos_ic_summary.to_csv(os.path.join(OUTPUT_DIR, "stitched_oos_ic_summary.csv"), index=False)
    portfolio_paths.to_parquet(os.path.join(OUTPUT_DIR, "oos_quantile_paths.parquet"))
    annual_portfolios.to_csv(
        os.path.join(OUTPUT_DIR, "oos_annual_portfolio_returns.csv"),
        index=False,
    )
    portfolio_summary.to_csv(
        os.path.join(OUTPUT_DIR, "oos_portfolio_summary.csv"),
        index=False,
    )
    plot_annual_oos_returns(annual_portfolios)

    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "lookback_years": LOOKBACK_YEARS,
                "first_oos_year": FIRST_OOS_YEAR,
                "last_oos_year": int(prices.index.max().year),
                "purge_days": "equal to each hypothesis forward horizon",
                "candidate_variants": len(all_specifications()),
                "factor_horizon_hypotheses": len(metadata),
                "selection_uses_future_data": False,
                "portfolio_cost_bps": TRANSACTION_COST_BPS,
                "portfolio_phase": 0,
                "portfolio_restarts_each_year": True,
            },
            file,
            indent=2,
        )

    return selections, oos_ic_summary, annual_portfolios, portfolio_summary


def plot_annual_oos_returns(annual_portfolios):
    complete = annual_portfolios[annual_portfolios["oos_year"] < 2026]
    matrix = complete.pivot(
        index="family",
        columns="oos_year",
        values="net_10bps_return",
    )
    color_limit = np.nanquantile(np.abs(matrix.to_numpy()), 0.98)
    figure, axis = plt.subplots(figsize=(14, 8))
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu",
        vmin=-color_limit,
        vmax=color_limit,
    )
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns)
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels(
        [name.replace("_", " ").title() for name in matrix.index]
    )
    axis.set_xlabel("Out-of-sample year")
    axis.set_title("Purged walk-forward Q5-Q1 annual returns, net 10 bps")
    figure.colorbar(image, ax=axis, label="Annual net return", fraction=0.03, pad=0.02)
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "oos_annual_returns_heatmap.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_walk_forward_results(oos_ic_summary, portfolio_summary):
    print("\nSTAGE 5: STITCHED OOS IC")
    print(
        oos_ic_summary.sort_values("mean_oos_ic", ascending=False).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print("\nSTAGE 5: WALK-FORWARD Q5-Q1 PORTFOLIOS, NET 10 BPS")
    print(
        portfolio_summary.sort_values("annual_sharpe", ascending=False).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    selections, ic_summary, annual_returns, portfolio_summary = run_walk_forward()
    print_walk_forward_results(ic_summary, portfolio_summary)
