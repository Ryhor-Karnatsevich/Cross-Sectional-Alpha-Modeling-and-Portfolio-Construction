"""Stage 4: implementability diagnostics with factor quantile portfolios.

This is not the final portfolio-construction engine. It builds transparent,
equal-weight factor portfolios to test monotonicity, long-short returns,
turnover, and simple transaction-cost sensitivity.
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from candidate_research import OUTPUT_DIR as CANDIDATE_OUTPUT_DIR
from factor_independence import build_baseline_factors, load_selected_factors
from pipeline import load_data, load_membership
from statistical_research import PERIODS


data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data_System")
)
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import BASE_DIR


OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "Factor_Research", "quantile_stage")
QUANTILES = 5
MIN_ASSETS = 50
COST_SCENARIOS_BPS = (0, 5, 10, 25)


def selected_factor_horizons(selected):
    horizons = {
        "baseline_momentum": 21,
        "baseline_low_vol": 21,
        "baseline_trend": 21,
    }

    for row in selected.itertuples():
        horizons[f"{row.family}__{row.variant}"] = int(row.horizon_days)

    return horizons


def realized_asset_returns(prices, position, horizon, tickers):
    """Use the last observable price within a completed holding period."""
    start_prices = prices.iloc[position].reindex(tickers)
    future_window = prices.iloc[position + 1 : position + horizon + 1][tickers]
    exact_end = future_window.iloc[-1]
    last_observable = future_window.ffill().iloc[-1]
    no_future_price = last_observable.isna()
    end_prices = last_observable.fillna(start_prices)
    returns = end_prices / start_prices - 1
    early_exit = exact_end.isna() & last_observable.notna()

    return returns, early_exit, no_future_price


def drift_weights(target_weights, asset_returns):
    gross_value = target_weights * (1 + asset_returns)
    total_value = gross_value.sum()

    if not np.isfinite(total_value) or total_value <= 0:
        return target_weights

    return gross_value / total_value


def half_l1_turnover(previous_weights, target_weights):
    all_tickers = previous_weights.index.union(target_weights.index)
    previous = previous_weights.reindex(all_tickers, fill_value=0)
    target = target_weights.reindex(all_tickers, fill_value=0)
    return 0.5 * (target - previous).abs().sum()


def run_one_quantile_path(
    factor,
    prices,
    membership,
    horizon,
    offset,
    factor_name,
    start_position=None,
    end_position=None,
    allow_partial_final_period=False,
):
    """Build one non-overlapping holding-period path for one phase."""
    signal = factor.shift(1)
    previous_end_weights = {
        quantile: pd.Series(dtype=float)
        for quantile in range(1, QUANTILES + 1)
    }
    rows = []

    first_position = offset if start_position is None else start_position + offset
    last_position = len(prices) - 1 if end_position is None else end_position

    for position in range(first_position, last_position + 1, horizon):
        actual_horizon = min(horizon, last_position - position)
        if actual_horizon < horizon and not allow_partial_final_period:
            break
        if actual_horizon <= 0 or position + actual_horizon >= len(prices):
            break
        date = prices.index[position]
        scores = signal.iloc[position]
        eligible = (
            scores.notna()
            & membership.iloc[position].fillna(False)
            & prices.iloc[position].notna()
            & (prices.iloc[position] > 0)
        )

        if eligible.sum() < MIN_ASSETS:
            continue

        ranked = scores.loc[eligible].rank(method="first", pct=True)
        labels = np.ceil(ranked * QUANTILES).clip(1, QUANTILES).astype(int)
        period_row = {
            "factor": factor_name,
            "horizon_days": horizon,
            "holding_days": actual_horizon,
            "offset": offset,
            "date": date,
            "end_date": prices.index[position + actual_horizon],
            "eligible_assets": int(eligible.sum()),
        }

        for quantile in range(1, QUANTILES + 1):
            tickers = labels.index[labels == quantile]
            target_weights = pd.Series(1 / len(tickers), index=tickers)
            asset_returns, early_exit, no_future = realized_asset_returns(
                prices,
                position,
                actual_horizon,
                tickers,
            )
            quantile_return = float((target_weights * asset_returns).sum())
            turnover = half_l1_turnover(
                previous_end_weights[quantile],
                target_weights,
            )
            previous_end_weights[quantile] = drift_weights(
                target_weights,
                asset_returns,
            )

            period_row[f"q{quantile}_return"] = quantile_return
            period_row[f"q{quantile}_turnover"] = turnover
            period_row[f"q{quantile}_assets"] = len(tickers)
            period_row[f"q{quantile}_early_exit_rate"] = early_exit.mean()
            period_row[f"q{quantile}_no_future_price_rate"] = no_future.mean()

        period_row["spread_gross_return"] = (
            period_row["q5_return"] - period_row["q1_return"]
        )
        period_row["spread_turnover"] = (
            period_row["q5_turnover"] + period_row["q1_turnover"]
        )

        for cost_bps in COST_SCENARIOS_BPS:
            # half-L1 turnover multiplied by two equals dollars traded.
            cost = 2 * period_row["spread_turnover"] * cost_bps / 10_000
            period_row[f"spread_net_{cost_bps}bps_return"] = (
                period_row["spread_gross_return"] - cost
            )

        rows.append(period_row)

    return pd.DataFrame(rows)


def annualized_statistics(returns, horizon):
    returns = pd.Series(returns).dropna()
    periods_per_year = 252 / horizon

    if len(returns) < 3:
        return {
            "observations": len(returns),
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "tstat": np.nan,
            "pvalue": np.nan,
            "win_rate": np.nan,
        }

    mean = returns.mean()
    std = returns.std()
    test = ttest_1samp(returns, popmean=0)
    tstat = float(test.statistic)
    pvalue = float(test.pvalue)

    return {
        "observations": len(returns),
        "annualized_return": mean * periods_per_year,
        "annualized_volatility": std * np.sqrt(periods_per_year),
        "sharpe": mean / std * np.sqrt(periods_per_year) if std else np.nan,
        "tstat": tstat,
        "pvalue": pvalue,
        "win_rate": (returns > 0).mean(),
    }


def summarize_quantile_path(path, period_name, start_date, end_date):
    period = path[(path["date"] >= start_date) & (path["date"] <= end_date)]
    if period.empty:
        return None

    horizon = int(period["horizon_days"].iloc[0])
    row = {
        "period": period_name,
        "factor": period["factor"].iloc[0],
        "horizon_days": horizon,
        "offset": int(period["offset"].iloc[0]),
        "average_spread_turnover": period["spread_turnover"].mean(),
        "average_early_exit_rate": period[
            ["q1_early_exit_rate", "q5_early_exit_rate"]
        ].mean(axis=1).mean(),
        "average_no_future_price_rate": period[
            ["q1_no_future_price_rate", "q5_no_future_price_rate"]
        ].mean(axis=1).mean(),
    }

    quantile_means = []
    for quantile in range(1, QUANTILES + 1):
        mean_return = period[f"q{quantile}_return"].mean()
        row[f"q{quantile}_mean_return"] = mean_return
        quantile_means.append(mean_return)

    row["quantile_monotonicity"] = (
        spearmanr(range(1, QUANTILES + 1), quantile_means).statistic
        if pd.Series(quantile_means).nunique() > 1
        else np.nan
    )

    for cost_bps in COST_SCENARIOS_BPS:
        column = f"spread_net_{cost_bps}bps_return"
        metrics = annualized_statistics(period[column], horizon)

        for metric, value in metrics.items():
            row[f"net_{cost_bps}bps_{metric}"] = value

    row["gross_pvalue"] = row["net_0bps_pvalue"]
    return row


def add_spread_multiple_testing(summary):
    summary = summary.copy()
    summary["gross_pvalue_holm"] = np.nan
    summary["gross_pvalue_fdr_bh"] = np.nan

    for _, group in summary.groupby("period"):
        valid = group["gross_pvalue"].notna()
        index = group.index[valid]
        if index.empty:
            continue
        pvalues = summary.loc[index, "gross_pvalue"].to_numpy()
        summary.loc[index, "gross_pvalue_holm"] = multipletests(
            pvalues,
            method="holm",
        )[1]
        summary.loc[index, "gross_pvalue_fdr_bh"] = multipletests(
            pvalues,
            method="fdr_bh",
        )[1]

    return summary


def run_quantile_research():
    returns, availability, _, prices = load_data()
    membership = load_membership().reindex(index=prices.index, columns=prices.columns)
    selected, selected_factors = load_selected_factors()
    factors = {
        **build_baseline_factors(returns, prices, availability),
        **selected_factors,
    }
    horizons = selected_factor_horizons(selected)
    paths = []
    summary_rows = []

    for factor_name, factor in factors.items():
        horizon = horizons[factor_name]
        phase_count = min(21, horizon)
        print(f"Quantile portfolios: {factor_name} | h={horizon}")

        for offset in range(phase_count):
            path = run_one_quantile_path(
                factor,
                prices,
                membership,
                horizon,
                offset,
                factor_name,
            )
            paths.append(path)

            for period_name, (start_date, end_date) in PERIODS.items():
                row = summarize_quantile_path(
                    path,
                    period_name,
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date),
                )
                if row is not None:
                    summary_rows.append(row)

    portfolio_paths = pd.concat(paths, ignore_index=True)
    summary = add_spread_multiple_testing(pd.DataFrame(summary_rows))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    portfolio_paths.to_parquet(os.path.join(OUTPUT_DIR, "quantile_portfolio_paths.parquet"))
    summary.to_csv(os.path.join(OUTPUT_DIR, "quantile_portfolio_statistics.csv"), index=False)
    plot_quantile_results(summary)

    assumptions = pd.Series(
        {
            "signal_lag_days": 1,
            "quantiles": QUANTILES,
            "minimum_assets": MIN_ASSETS,
            "holding_period_equals_selected_horizon": True,
            "rebalance_paths_are_non_overlapping": True,
            "maximum_calendar_phases": 21,
            "weighting": "equal weight inside every quantile",
            "turnover": "half L1 distance; spread is Q5 plus Q1 turnover",
            "transaction_costs_bps_per_dollar_traded": list(COST_SCENARIOS_BPS),
            "missing_endpoint": "last observable price inside completed horizon",
            "no_future_price": "zero return fallback and separately reported",
        }
    )
    assumptions.to_json(
        os.path.join(OUTPUT_DIR, "run_config.json"),
        indent=2,
    )

    return portfolio_paths, summary


def short_factor_name(name):
    if name.startswith("baseline_"):
        return name.replace("baseline_", "Baseline ").replace("_", " ").title()
    return name.split("__", maxsplit=1)[0].replace("_", " ").title()


def plot_quantile_results(summary):
    full = summary[summary["period"] == "full"]
    aggregate = full.groupby("factor").agg(
        net_return=("net_10bps_annualized_return", "median"),
        net_sharpe=("net_10bps_sharpe", "median"),
    ).sort_values("net_sharpe")
    labels = [short_factor_name(name) for name in aggregate.index]
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    axes[0].barh(labels, aggregate["net_return"])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Median annualized Q5-Q1 return")
    axes[0].set_xlabel("Net return after 10 bps costs")
    axes[1].barh(labels, aggregate["net_sharpe"])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Median Q5-Q1 Sharpe")
    axes[1].set_xlabel("Net Sharpe after 10 bps costs")
    figure.suptitle("Stage 4: quantile portfolio results across calendar phases")
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "quantile_net_results.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_quantile_results(summary):
    full = summary[summary["period"] == "full"]
    aggregate = full.groupby(["factor", "horizon_days"]).agg(
        phase_median_gross_sharpe=("net_0bps_sharpe", "median"),
        phase_median_net_10bps_sharpe=("net_10bps_sharpe", "median"),
        phase_median_annual_net_10bps=("net_10bps_annualized_return", "median"),
        phase_median_turnover=("average_spread_turnover", "median"),
        phase_median_monotonicity=("quantile_monotonicity", "median"),
    )
    print("\nSTAGE 4: QUANTILE PORTFOLIO SUMMARY ACROSS PHASES")
    print(aggregate.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    _, quantile_statistics = run_quantile_research()
    print_quantile_results(quantile_statistics)
