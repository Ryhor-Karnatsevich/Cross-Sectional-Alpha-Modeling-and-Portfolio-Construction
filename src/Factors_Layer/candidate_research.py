"""Stage 2: economically motivated price and volume factor candidates.

The parameter grid below is deliberately small and declared before evaluation.
Candidate selection uses train and validation only. Test results are reported,
but never enter the selection score.
"""

import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import build_factor, load_data, load_membership
from statistical_research import (
    HORIZONS,
    PERIODS,
    add_multiple_testing_corrections,
    compute_daily_spearman_ic,
    compute_forward_returns,
    summarize_calendar_offsets,
    summarize_ic,
)


data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data_System")
)
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import BASE_DIR, VOLUME_PATH


OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "Factor_Research", "candidate_stage")
CACHE_DIR = os.path.join(OUTPUT_DIR, "selected_factor_cache")
MIN_OBSERVATION_RATIO = 0.80


# This is the complete Stage 2 search family. Do not expand it after seeing test.
CANDIDATE_GRID = {
    "short_term_reversal": [
        {"variant": "reversal_5d", "window": 5},
        {"variant": "reversal_10d", "window": 10},
        {"variant": "reversal_21d", "window": 21},
    ],
    "residual_momentum": [
        {"variant": "resmom_6m_1m", "window": 126, "skip": 21},
        {"variant": "resmom_9m_1m", "window": 189, "skip": 21},
        {"variant": "resmom_12m_1m", "window": 252, "skip": 21},
    ],
    "volatility_scaled_momentum": [
        {
            "variant": "vsmom_6m_1m_vol60",
            "window": 126,
            "skip": 21,
            "vol_window": 60,
        },
        {
            "variant": "vsmom_9m_1m_vol60",
            "window": 189,
            "skip": 21,
            "vol_window": 60,
        },
        {
            "variant": "vsmom_12m_1m_vol60",
            "window": 252,
            "skip": 21,
            "vol_window": 60,
        },
    ],
    "high_proximity": [
        {"variant": "high_6m", "window": 126},
        {"variant": "high_12m", "window": 252},
        {"variant": "high_18m", "window": 378},
    ],
    "trend_slope": [
        {"variant": "slope_50d", "window": 50},
        {"variant": "slope_100d", "window": 100},
        {"variant": "slope_200d", "window": 200},
        {"variant": "slope_250d", "window": 250},
    ],
    "risk_adjusted_trend": [
        {"variant": "risk_trend_50d", "window": 50},
        {"variant": "risk_trend_100d", "window": 100},
        {"variant": "risk_trend_200d", "window": 200},
        {"variant": "risk_trend_250d", "window": 250},
    ],
    "liquidity_change": [
        {"variant": "liq_5d_60d", "short_window": 5, "long_window": 60},
        {"variant": "liq_20d_126d", "short_window": 20, "long_window": 126},
        {"variant": "liq_20d_252d", "short_window": 20, "long_window": 252},
    ],
    "price_volume_confirmation": [
        {
            "variant": "pvc_6m_1m_liq5_60",
            "window": 126,
            "skip": 21,
            "short_window": 5,
            "long_window": 60,
        },
        {
            "variant": "pvc_9m_1m_liq20_126",
            "window": 189,
            "skip": 21,
            "short_window": 20,
            "long_window": 126,
        },
        {
            "variant": "pvc_12m_1m_liq20_252",
            "window": 252,
            "skip": 21,
            "short_window": 20,
            "long_window": 252,
        },
    ],
}


def required_observations(window):
    return max(2, int(np.ceil(window * MIN_OBSERVATION_RATIO)))


def cumulative_log_return(returns, window, skip=0):
    formation_window = window - skip

    if formation_window <= 0:
        raise ValueError("window must be greater than skip")

    return (
        np.log1p(returns)
        .shift(skip)
        .rolling(
            formation_window,
            min_periods=required_observations(formation_window),
        )
        .sum()
    )


def residual_market_returns(returns, availability):
    """Remove the contemporaneous equal-weight universe return each day."""
    eligible_returns = returns.where(availability)
    market_return = eligible_returns.mean(axis=1)
    return eligible_returns.sub(market_return, axis=0)


def rolling_log_price_slope(prices, window):
    """Vectorized rolling OLS slope that remains correct with missing values."""
    log_prices = np.log(prices)
    observation_number = pd.Series(
        np.arange(len(prices), dtype=float),
        index=prices.index,
    )
    observed = log_prices.notna().astype(float)
    x = observed.mul(observation_number, axis=0)
    x_squared = observed.mul(observation_number.pow(2), axis=0)
    xy = log_prices.mul(observation_number, axis=0)
    minimum = required_observations(window)

    count = observed.rolling(window, min_periods=minimum).sum()
    sum_x = x.rolling(window, min_periods=minimum).sum()
    sum_y = log_prices.rolling(window, min_periods=minimum).sum()
    sum_x_squared = x_squared.rolling(window, min_periods=minimum).sum()
    sum_xy = xy.rolling(window, min_periods=minimum).sum()
    denominator = count * sum_x_squared - sum_x.pow(2)
    slope = (count * sum_xy - sum_x * sum_y) / denominator

    return (slope * 252).where(denominator > 0)


def liquidity_change(prices, volume, short_window, long_window):
    """Log ratio of recent to slow average dollar trading volume."""
    dollar_volume = prices * volume
    short_average = dollar_volume.rolling(
        short_window,
        min_periods=required_observations(short_window),
    ).mean()
    long_average = dollar_volume.rolling(
        long_window,
        min_periods=required_observations(long_window),
    ).mean()
    valid = (short_average > 0) & (long_average > 0)
    ratio = (short_average / long_average).where(valid)
    return np.log(ratio)


def build_candidate(
    family,
    parameters,
    returns,
    prices,
    volume,
    availability,
):
    """Build one candidate without using any future observations."""
    window = parameters.get("window")
    skip = parameters.get("skip", 0)

    if family == "short_term_reversal":
        raw = -cumulative_log_return(returns, window)
        return build_factor(raw, availability)

    if family == "residual_momentum":
        residual_returns = residual_market_returns(returns, availability)
        raw = cumulative_log_return(residual_returns, window, skip)
        return build_factor(raw, availability)

    if family == "volatility_scaled_momentum":
        momentum = cumulative_log_return(returns, window, skip)
        volatility = returns.rolling(
            parameters["vol_window"],
            min_periods=required_observations(parameters["vol_window"]),
        ).std()
        raw = momentum / volatility.replace(0, np.nan)
        return build_factor(raw, availability)

    if family == "high_proximity":
        rolling_high = prices.rolling(
            window,
            min_periods=required_observations(window),
        ).max()
        raw = prices / rolling_high - 1
        return build_factor(raw, availability)

    if family == "trend_slope":
        raw = rolling_log_price_slope(prices, window)
        return build_factor(raw, availability)

    if family == "risk_adjusted_trend":
        slope = rolling_log_price_slope(prices, window)
        volatility = returns.rolling(
            window,
            min_periods=required_observations(window),
        ).std() * np.sqrt(252)
        raw = slope / volatility.replace(0, np.nan)
        return build_factor(raw, availability)

    if family == "liquidity_change":
        raw = liquidity_change(
            prices,
            volume,
            parameters["short_window"],
            parameters["long_window"],
        )
        return build_factor(raw, availability)

    if family == "price_volume_confirmation":
        momentum = build_factor(
            cumulative_log_return(returns, window, skip),
            availability,
        )
        liquidity = build_factor(
            liquidity_change(
                prices,
                volume,
                parameters["short_window"],
                parameters["long_window"],
            ),
            availability,
        )
        # A bounded multiplier strengthens momentum when liquidity is increasing,
        # without allowing volume alone to reverse the direction of the signal.
        raw = momentum * (1 + 0.25 * liquidity.clip(-2, 2))
        return build_factor(raw, availability)

    raise ValueError(f"Unknown candidate family: {family}")


def parameter_text(parameters):
    return ", ".join(
        f"{key}={value}"
        for key, value in parameters.items()
        if key != "variant"
    )


def add_selection_scores(statistics, calendar_offsets):
    """Rank each family using train and validation, never test."""
    metrics = statistics.pivot_table(
        index=["family", "variant", "parameters", "horizon_days"],
        columns="period",
        values=["mean_ic", "nw_tstat", "positive_rate"],
    )
    metrics.columns = [f"{period}_{metric}" for metric, period in metrics.columns]
    metrics = metrics.reset_index()

    offset_consistency = (
        calendar_offsets[
            calendar_offsets["period"].isin(["train", "validation"])
        ]
        .assign(positive_offset=lambda frame: frame["mean_ic"] > 0)
        .groupby(["family", "variant", "horizon_days", "period"])[
            "positive_offset"
        ]
        .mean()
        .unstack("period")
        .rename(
            columns={
                "train": "train_positive_offsets",
                "validation": "validation_positive_offsets",
            }
        )
        .reset_index()
    )
    metrics = metrics.merge(
        offset_consistency,
        on=["family", "variant", "horizon_days"],
        how="left",
    )
    metrics["stability_gap"] = (
        metrics["train_mean_ic"] - metrics["validation_mean_ic"]
    ).abs()
    metrics["selection_eligible"] = (
        (metrics["train_mean_ic"] > 0)
        & (metrics["validation_mean_ic"] > 0)
    )
    metrics["selection_score"] = np.nan

    for _, family_results in metrics.groupby("family"):
        index = family_results.index
        score = (
            0.20 * family_results["validation_mean_ic"].rank(pct=True)
            + 0.15 * family_results["validation_nw_tstat"].rank(pct=True)
            + 0.10 * family_results["validation_positive_rate"].rank(pct=True)
            + 0.10
            * family_results["validation_positive_offsets"].rank(pct=True)
            + 0.15 * family_results["train_mean_ic"].rank(pct=True)
            + 0.10 * family_results["train_nw_tstat"].rank(pct=True)
            + 0.05 * family_results["train_positive_rate"].rank(pct=True)
            + 0.05 * family_results["train_positive_offsets"].rank(pct=True)
            + 0.10 * (-family_results["stability_gap"]).rank(pct=True)
        )
        metrics.loc[index, "selection_score"] = score

    return metrics


def select_one_per_family(selection_table):
    selected = []

    for _, family_results in selection_table.groupby("family"):
        ranked = family_results.sort_values(
            ["selection_eligible", "selection_score"],
            ascending=[False, False],
        )
        selected.append(ranked.iloc[0])

    return pd.DataFrame(selected).reset_index(drop=True)


def find_parameters(family, variant):
    for parameters in CANDIDATE_GRID[family]:
        if parameters["variant"] == variant:
            return parameters

    raise ValueError(f"Unknown variant: {family} | {variant}")


def safe_cache_name(family, variant):
    return f"{family}__{variant}.parquet"


def save_selected_factor_cache(
    selected,
    returns,
    prices,
    volume,
    availability,
):
    os.makedirs(CACHE_DIR, exist_ok=True)

    for filename in os.listdir(CACHE_DIR):
        if filename.endswith(".parquet"):
            os.remove(os.path.join(CACHE_DIR, filename))

    for row in selected.itertuples():
        parameters = find_parameters(row.family, row.variant)
        factor = build_candidate(
            row.family,
            parameters,
            returns,
            prices,
            volume,
            availability,
        )
        path = os.path.join(CACHE_DIR, safe_cache_name(row.family, row.variant))
        factor.astype("float32").to_parquet(path)


def plot_candidate_heatmap(statistics):
    """Show full-period Mean IC for every candidate and horizon."""
    full = statistics[statistics["period"] == "full"].copy()
    full["candidate"] = full["family"] + " | " + full["variant"]
    heatmap = full.pivot(index="candidate", columns="horizon_days", values="mean_ic")
    color_limit = np.nanquantile(np.abs(heatmap.to_numpy()), 0.98)
    figure, axis = plt.subplots(figsize=(10, 12))
    image = axis.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu",
        vmin=-color_limit,
        vmax=color_limit,
    )
    axis.set_xticks(range(len(heatmap.columns)))
    axis.set_xticklabels(heatmap.columns)
    axis.set_yticks(range(len(heatmap.index)))
    axis.set_yticklabels(heatmap.index)
    axis.set_xlabel("Forward horizon, trading days")
    axis.set_title("Stage 2 candidates: full-period Mean IC")
    figure.colorbar(image, ax=axis, label="Mean IC", fraction=0.03, pad=0.02)
    figure.tight_layout()
    figure.savefig(
        os.path.join(OUTPUT_DIR, "candidate_mean_ic_heatmap.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def run_candidate_research():
    returns, availability, _, prices = load_data()
    membership = load_membership().reindex(index=prices.index, columns=prices.columns)
    volume = pd.read_parquet(VOLUME_PATH).reindex(
        index=prices.index,
        columns=prices.columns,
    )

    statistic_rows = []
    offset_rows = []

    for family, variants in CANDIDATE_GRID.items():
        print(f"Building family: {family}")

        for parameters in variants:
            factor = build_candidate(
                family,
                parameters,
                returns,
                prices,
                volume,
                availability,
            )

            for horizon in HORIZONS:
                forward_returns = compute_forward_returns(prices, horizon)
                ic_data = compute_daily_spearman_ic(
                    factor,
                    forward_returns,
                    membership,
                )

                for period_name, (start_date, end_date) in PERIODS.items():
                    row = summarize_ic(
                        ic_data,
                        parameters["variant"],
                        horizon,
                        period_name,
                        start_date,
                        end_date,
                    )
                    row["family"] = family
                    row["variant"] = parameters["variant"]
                    row["parameters"] = parameter_text(parameters)
                    statistic_rows.append(row)

                    rows = summarize_calendar_offsets(
                        ic_data,
                        parameters["variant"],
                        horizon,
                        period_name,
                        start_date,
                        end_date,
                    )
                    for offset_row in rows:
                        offset_row["family"] = family
                        offset_row["variant"] = parameters["variant"]
                    offset_rows.extend(rows)

    statistics = add_multiple_testing_corrections(pd.DataFrame(statistic_rows))
    calendar_offsets = pd.DataFrame(offset_rows)
    selection_table = add_selection_scores(statistics, calendar_offsets)
    selected = select_one_per_family(selection_table)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    statistics.to_csv(os.path.join(OUTPUT_DIR, "candidate_statistics.csv"), index=False)
    calendar_offsets.to_csv(
        os.path.join(OUTPUT_DIR, "candidate_calendar_offsets.csv"),
        index=False,
    )
    selection_table.to_csv(
        os.path.join(OUTPUT_DIR, "candidate_selection_table.csv"),
        index=False,
    )
    selected.to_csv(os.path.join(OUTPUT_DIR, "selected_candidates.csv"), index=False)
    save_selected_factor_cache(
        selected,
        returns,
        prices,
        volume,
        availability,
    )
    plot_candidate_heatmap(statistics)

    config = {
        "candidate_families": len(CANDIDATE_GRID),
        "candidate_variants": sum(len(items) for items in CANDIDATE_GRID.values()),
        "horizons": list(HORIZONS),
        "selection_data": ["train", "validation"],
        "test_used_in_selection": False,
        "minimum_observation_ratio": MIN_OBSERVATION_RATIO,
        "grid": CANDIDATE_GRID,
    }
    with open(
        os.path.join(OUTPUT_DIR, "run_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    return statistics, calendar_offsets, selection_table, selected


def print_selected_candidates(selected):
    columns = [
        "family",
        "variant",
        "horizon_days",
        "train_mean_ic",
        "validation_mean_ic",
        "selection_eligible",
        "selection_score",
    ]
    print("\nSTAGE 2: SELECTED USING TRAIN + VALIDATION ONLY")
    print(selected[columns].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    _, _, _, selected_candidates = run_candidate_research()
    print_selected_candidates(selected_candidates)
