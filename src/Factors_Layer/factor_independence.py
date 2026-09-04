"""Stage 3: factor overlap, residualization, and Low Vol role diagnosis."""

import os
import sys
from itertools import combinations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from candidate_research import CACHE_DIR, OUTPUT_DIR as CANDIDATE_OUTPUT_DIR
from factors import compute_momentum, compute_trend, compute_volatility
from pipeline import build_factor, load_data, load_membership
from statistical_research import (
    HORIZONS,
    PERIODS,
    add_multiple_testing_corrections,
    compute_daily_spearman_ic,
    compute_forward_returns,
    summarize_ic,
)
from transforms import zscore


data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data_System")
)
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import BASE_DIR


OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "Factor_Research", "independence_stage")
MIN_ASSETS = 30


def build_baseline_factors(returns, prices, availability):
    momentum = build_factor(compute_momentum(returns), availability)
    low_vol = -build_factor(compute_volatility(returns), availability)
    trend = build_factor(compute_trend(prices), availability)
    return {
        "baseline_momentum": momentum,
        "baseline_low_vol": low_vol,
        "baseline_trend": trend,
    }


def load_selected_factors():
    selected_path = os.path.join(CANDIDATE_OUTPUT_DIR, "selected_candidates.csv")

    if not os.path.exists(selected_path):
        raise FileNotFoundError(
            "Run candidate_research.py before factor_independence.py"
        )

    selected = pd.read_csv(selected_path)
    factors = {}

    for row in selected.itertuples():
        filename = f"{row.family}__{row.variant}.parquet"
        path = os.path.join(CACHE_DIR, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing selected factor cache: {path}")

        factors[f"{row.family}__{row.variant}"] = pd.read_parquet(path).astype(float)

    return selected, factors


def rowwise_spearman(left, right, membership, min_assets=MIN_ASSETS):
    left, right = left.align(right, join="inner")
    membership = membership.reindex(index=left.index, columns=left.columns)
    valid = left.notna() & right.notna() & membership.fillna(False)
    count = valid.sum(axis=1)
    left_rank = left.where(valid).rank(axis=1)
    right_rank = right.where(valid).rank(axis=1)
    left_centered = left_rank.sub(left_rank.mean(axis=1), axis=0)
    right_centered = right_rank.sub(right_rank.mean(axis=1), axis=0)
    numerator = (left_centered * right_centered).sum(axis=1, min_count=1)
    denominator = np.sqrt(
        left_centered.pow(2).sum(axis=1, min_count=1)
        * right_centered.pow(2).sum(axis=1, min_count=1)
    )
    return (numerator / denominator).where(count >= min_assets)


def calculate_score_correlations(factors, membership):
    rows = []

    for left_name, right_name in combinations(factors, 2):
        daily_correlation = rowwise_spearman(
            factors[left_name],
            factors[right_name],
            membership,
        )

        for period_name, (start_date, end_date) in PERIODS.items():
            values = daily_correlation.loc[start_date:end_date].dropna()
            rows.append(
                {
                    "period": period_name,
                    "left_factor": left_name,
                    "right_factor": right_name,
                    "observations": len(values),
                    "mean_daily_spearman": values.mean(),
                    "median_daily_spearman": values.median(),
                    "mean_absolute_spearman": values.abs().mean(),
                }
            )

    return pd.DataFrame(rows)


def calculate_ic_correlations(factors, prices, membership, horizon=21):
    forward_returns = compute_forward_returns(prices, horizon)
    ic_series = {}

    for name, factor in factors.items():
        ic_series[name] = compute_daily_spearman_ic(
            factor,
            forward_returns,
            membership,
        )["ic"]

    ic_frame = pd.DataFrame(ic_series)
    rows = []

    for period_name, (start_date, end_date) in PERIODS.items():
        matrix = ic_frame.loc[start_date:end_date].corr()

        for left_name, right_name in combinations(matrix.columns, 2):
            rows.append(
                {
                    "period": period_name,
                    "horizon_days": horizon,
                    "left_factor": left_name,
                    "right_factor": right_name,
                    "ic_series_correlation": matrix.loc[left_name, right_name],
                }
            )

    return ic_frame, pd.DataFrame(rows)


def residualize_cross_sectionally(target, controls, membership):
    """Remove same-date linear exposure to controls with cross-sectional OLS."""
    residual = pd.DataFrame(np.nan, index=target.index, columns=target.columns)

    for date in target.index:
        y = target.loc[date]
        x = pd.concat([control.loc[date] for control in controls], axis=1)
        valid = y.notna() & x.notna().all(axis=1) & membership.loc[date].fillna(False)

        if valid.sum() < max(MIN_ASSETS, len(controls) + 2):
            continue

        design = np.column_stack(
            [np.ones(valid.sum()), x.loc[valid].to_numpy(dtype=float)]
        )
        response = y.loc[valid].to_numpy(dtype=float)
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        residual.loc[date, valid] = response - design @ coefficients

    return zscore(residual).where(membership)


def calculate_incremental_ic(
    selected,
    selected_factors,
    baselines,
    prices,
    membership,
):
    rows = []
    controls = list(baselines.values())

    for row in selected.itertuples():
        name = f"{row.family}__{row.variant}"
        raw_factor = selected_factors[name]
        residual_factor = residualize_cross_sectionally(
            raw_factor,
            controls,
            membership,
        )
        forward_returns = compute_forward_returns(prices, int(row.horizon_days))

        for version, factor in [("raw", raw_factor), ("residual", residual_factor)]:
            ic_data = compute_daily_spearman_ic(
                factor,
                forward_returns,
                membership,
            )

            for period_name, (start_date, end_date) in PERIODS.items():
                result = summarize_ic(
                    ic_data,
                    name,
                    int(row.horizon_days),
                    period_name,
                    start_date,
                    end_date,
                )
                result["family"] = row.family
                result["variant"] = row.variant
                result["version"] = version
                result["controls"] = "baseline_momentum + baseline_low_vol + baseline_trend"
                rows.append(result)

    # Explicitly answer whether the original Trend adds information over Momentum.
    trend_residual = residualize_cross_sectionally(
        baselines["baseline_trend"],
        [baselines["baseline_momentum"]],
        membership,
    )
    forward_returns = compute_forward_returns(prices, 21)

    for version, factor in [
        ("raw", baselines["baseline_trend"]),
        ("residual", trend_residual),
    ]:
        ic_data = compute_daily_spearman_ic(factor, forward_returns, membership)

        for period_name, (start_date, end_date) in PERIODS.items():
            result = summarize_ic(
                ic_data,
                "baseline_trend_over_momentum",
                21,
                period_name,
                start_date,
                end_date,
            )
            result["family"] = "baseline_trend_over_momentum"
            result["variant"] = "trend_sma50"
            result["version"] = version
            result["controls"] = "baseline_momentum"
            rows.append(result)

    return pd.DataFrame(rows)


def future_realized_volatility(returns, horizon):
    """Annualized volatility of returns t+1 through t+h, aligned at t."""
    minimum = max(2, int(np.ceil(horizon * 0.80)))
    return (
        returns.shift(-horizon)
        .rolling(horizon, min_periods=minimum)
        .std()
        * np.sqrt(252)
    )


def evaluate_low_vol_roles(returns, prices, availability, membership):
    volatility_descriptor = build_factor(
        compute_volatility(returns, window=60, min_obs=40),
        availability,
    )
    low_vol_alpha = -volatility_descriptor
    rows = []

    for horizon in HORIZONS:
        targets = {
            "alpha_return": (
                low_vol_alpha,
                compute_forward_returns(prices, horizon),
            ),
            "risk_descriptor": (
                volatility_descriptor,
                future_realized_volatility(returns, horizon),
            ),
        }

        for role, (factor, target) in targets.items():
            ic_data = compute_daily_spearman_ic(factor, target, membership)

            for period_name, (start_date, end_date) in PERIODS.items():
                result = summarize_ic(
                    ic_data,
                    "low_vol_60d",
                    horizon,
                    period_name,
                    start_date,
                    end_date,
                )
                result["role"] = role
                rows.append(result)

    return pd.DataFrame(rows)


def correlation_matrix_from_long(table, value_column, period="full"):
    names = sorted(set(table["left_factor"]) | set(table["right_factor"]))
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    selected = table[table["period"] == period]

    for row in selected.itertuples():
        value = getattr(row, value_column)
        matrix.loc[row.left_factor, row.right_factor] = value
        matrix.loc[row.right_factor, row.left_factor] = value

    return matrix


def plot_correlation_matrix(matrix, title, filename):
    def short_label(name):
        if name.startswith("baseline_"):
            return name.replace("baseline_", "Baseline ").replace("_", " ").title()
        return name.split("__", maxsplit=1)[0].replace("_", " ").title()

    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1)
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels([short_label(name) for name in matrix.columns], rotation=90)
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels([short_label(name) for name in matrix.index])
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Correlation", fraction=0.035, pad=0.02)
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT_DIR, filename), dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_independence_research():
    returns, availability, _, prices = load_data()
    membership = load_membership().reindex(index=prices.index, columns=prices.columns)
    selected, selected_factors = load_selected_factors()
    baselines = build_baseline_factors(returns, prices, availability)
    all_factors = {**baselines, **selected_factors}

    score_correlations = calculate_score_correlations(all_factors, membership)
    _, ic_correlations = calculate_ic_correlations(
        all_factors,
        prices,
        membership,
        horizon=21,
    )
    incremental_ic = calculate_incremental_ic(
        selected,
        selected_factors,
        baselines,
        prices,
        membership,
    )
    low_vol_roles = evaluate_low_vol_roles(
        returns,
        prices,
        availability,
        membership,
    )
    incremental_ic = add_multiple_testing_corrections(incremental_ic)
    low_vol_roles = add_multiple_testing_corrections(low_vol_roles)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    score_correlations.to_csv(
        os.path.join(OUTPUT_DIR, "factor_score_correlations.csv"),
        index=False,
    )
    ic_correlations.to_csv(
        os.path.join(OUTPUT_DIR, "ic_series_correlations_21d.csv"),
        index=False,
    )
    incremental_ic.to_csv(
        os.path.join(OUTPUT_DIR, "incremental_residual_ic.csv"),
        index=False,
    )
    low_vol_roles.to_csv(
        os.path.join(OUTPUT_DIR, "low_vol_role_comparison.csv"),
        index=False,
    )

    score_matrix = correlation_matrix_from_long(
        score_correlations,
        "mean_daily_spearman",
    )
    ic_matrix = correlation_matrix_from_long(
        ic_correlations,
        "ic_series_correlation",
    )
    score_matrix.to_csv(os.path.join(OUTPUT_DIR, "factor_score_matrix_full.csv"))
    ic_matrix.to_csv(os.path.join(OUTPUT_DIR, "ic_series_matrix_full_21d.csv"))
    plot_correlation_matrix(
        score_matrix,
        "Mean daily cross-sectional factor-score correlation",
        "factor_score_correlation.png",
    )
    plot_correlation_matrix(
        ic_matrix,
        "Correlation between daily 21-day IC series",
        "ic_series_correlation.png",
    )

    return score_correlations, ic_correlations, incremental_ic, low_vol_roles


def print_independence_results(score_correlations, incremental_ic, low_vol_roles):
    full_scores = score_correlations[score_correlations["period"] == "full"]
    strongest = full_scores.iloc[
        full_scores["mean_daily_spearman"].abs().argsort()[::-1]
    ].head(10)
    print("\nSTAGE 3: STRONGEST FACTOR-SCORE OVERLAPS")
    print(
        strongest[
            ["left_factor", "right_factor", "mean_daily_spearman"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    full_incremental = incremental_ic[
        incremental_ic["period"] == "full"
    ][["family", "version", "horizon_days", "mean_ic", "nw_tstat"]]
    print("\nRAW VERSUS RESIDUAL FACTOR IC")
    print(full_incremental.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    full_low_vol = low_vol_roles[low_vol_roles["period"] == "full"]
    print("\nLOW VOL: ALPHA VERSUS RISK DESCRIPTOR")
    print(
        full_low_vol[
            ["role", "horizon_days", "mean_ic", "nw_tstat", "nw_pvalue"]
        ].to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    outputs = run_independence_research()
    print_independence_results(outputs[0], outputs[2], outputs[3])
