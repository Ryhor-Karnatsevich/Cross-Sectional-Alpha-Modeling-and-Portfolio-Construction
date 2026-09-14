import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

selection_layer_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Factor_Selection_Layer")
)
if selection_layer_path not in sys.path:
    sys.path.insert(0, selection_layer_path)

from ic_analysis import summarize_ic
from selection_config import (
    DAILY_QUANTILE_RESULTS_PATH,
    FORWARD_HORIZONS,
    QUANTILE_COUNT,
)
from selection_storage import temporary_path
from research_config import RESEARCH_RESULTS_DIR


# -------------------------
# EXAMPLE FACTOR
FACTOR_KEY = "momentum|12m-1m"
LEGACY_RESULT_DIR = os.path.join(
    RESEARCH_RESULTS_DIR,
    "Legacy",
    "Single_Factor_Screening",
)
SCREENING_RESULT_PATH = os.path.join(
    LEGACY_RESULT_DIR,
    "single_factor_screening.csv",
)
SCREENING_FIGURE_DIR = os.path.join(
    LEGACY_RESULT_DIR,
    "Figures",
)
ROLLING_WINDOW = 126


# -------------------------
# INITIAL FILTER SETTINGS
# These are starting research thresholds, not final universal rules.
MIN_VALID_RATIO = 0.90
MIN_ABSOLUTE_MEAN_IC = 0.01
MIN_ABSOLUTE_HAC_TSTAT = 1.96
MIN_DIRECTION_CONSISTENCY = 0.52
MIN_ORIENTED_CURVE_MONOTONICITY = 0.50


# -------------------------
# DATA LOADING
def required_columns():
    columns = [
        "date",
        "hypothesis_key",
        "factor_key",
        "family",
        "variant",
        "horizon_days",
        "parameters",
        "research_eligible",
        "return_asset_count",
        "spearman_ic",
        "pearson_correlation",
        "factor_beta",
        "raw_spread",
        "median_spread",
    ]

    for quantile in range(1, QUANTILE_COUNT + 1):
        columns.extend(
            [
                f"q{quantile}_count",
                f"q{quantile}_factor_score_mean",
                f"q{quantile}_return_mean",
                f"q{quantile}_return_median",
            ]
        )

    return columns


def load_factor_results(factor_key):
    if not os.path.exists(DAILY_QUANTILE_RESULTS_PATH):
        raise FileNotFoundError(
            "Run the Factor Selection pipeline before screening"
        )

    table = pq.read_table(
        DAILY_QUANTILE_RESULTS_PATH,
        columns=required_columns(),
        filters=[("factor_key", "=", factor_key)],
    )
    factor_results = table.to_pandas().sort_values(
        ["horizon_days", "date"]
    )

    if factor_results.empty:
        raise ValueError(f"Factor not found: {factor_key}")

    found_horizons = set(
        factor_results["horizon_days"].drop_duplicates().astype(int)
    )
    expected_horizons = set(FORWARD_HORIZONS)

    if found_horizons != expected_horizons:
        raise ValueError(
            "Factor horizons do not match the configured horizons"
        )

    return factor_results


# -------------------------
# RELATIONSHIP SUMMARY
def relationship_direction(mean_ic):
    if pd.isna(mean_ic) or mean_ic == 0:
        return 0

    return int(np.sign(mean_ic))


def quantile_curve(frame, value_name):
    return np.array(
        [
            frame[f"q{quantile}_{value_name}"].mean()
            for quantile in range(1, QUANTILE_COUNT + 1)
        ],
        dtype=float,
    )


def curve_monotonicity(return_curve):
    if not np.isfinite(return_curve).all():
        return np.nan
    if np.unique(return_curve).size < 2:
        return np.nan

    return float(
        spearmanr(
            np.arange(1, QUANTILE_COUNT + 1),
            return_curve,
        ).statistic
    )


def summarize_horizon(frame):
    frame = frame[frame["research_eligible"]].copy()
    horizon = int(frame["horizon_days"].iloc[0])
    ic_statistics = summarize_ic(frame["spearman_ic"], horizon)
    mean_ic = ic_statistics["mean_ic"]
    direction = relationship_direction(mean_ic)
    valid_ic = frame["spearman_ic"].dropna()
    valid_spread = frame["raw_spread"].dropna()
    mean_return_curve = quantile_curve(frame, "return_mean")
    median_return_curve = quantile_curve(frame, "return_median")
    mean_score_curve = quantile_curve(frame, "factor_score_mean")
    mean_curve_monotonicity = curve_monotonicity(mean_return_curve)
    median_curve_monotonicity = curve_monotonicity(median_return_curve)

    if direction:
        direction_consistency = (
            np.sign(valid_ic) == direction
        ).mean()
    else:
        direction_consistency = np.nan

    result = {
        "factor_key": frame["factor_key"].iloc[0],
        "family": frame["family"].iloc[0],
        "variant": frame["variant"].iloc[0],
        "horizon_days": horizon,
        "parameters": frame["parameters"].iloc[0],
        "research_dates": len(frame),
        "valid_ic_dates": len(valid_ic),
        "valid_ic_ratio": len(valid_ic) / len(frame),
        "valid_spread_dates": len(valid_spread),
        "valid_spread_ratio": len(valid_spread) / len(frame),
        "median_daily_assets": frame["return_asset_count"].median(),
        "mean_ic": mean_ic,
        "median_ic": valid_ic.median(),
        "ic_std": ic_statistics["std_ic"],
        "hac_tstat": ic_statistics["tstat"],
        "direction": direction,
        "direction_consistency": direction_consistency,
        "mean_pearson_correlation": frame[
            "pearson_correlation"
        ].mean(),
        "mean_factor_beta": frame["factor_beta"].mean(),
        "mean_raw_spread": valid_spread.mean(),
        "median_raw_spread": valid_spread.median(),
        "mean_median_spread": frame["median_spread"].mean(),
        "mean_curve_monotonicity": mean_curve_monotonicity,
        "median_curve_monotonicity": median_curve_monotonicity,
    }

    for position, quantile in enumerate(
        range(1, QUANTILE_COUNT + 1)
    ):
        result[f"q{quantile}_average_count"] = frame[
            f"q{quantile}_count"
        ].mean()
        result[f"q{quantile}_average_factor_score"] = mean_score_curve[
            position
        ]
        result[f"q{quantile}_average_return"] = mean_return_curve[position]
        result[f"q{quantile}_median_return"] = median_return_curve[position]

    return result


def summarize_factor(factor_results):
    rows = []

    for _, horizon_results in factor_results.groupby(
        "horizon_days",
        sort=True,
    ):
        rows.append(summarize_horizon(horizon_results))

    return pd.DataFrame(rows).sort_values("horizon_days")


# -------------------------
# PRIMARY FILTERS
def add_data_filter(screening):
    screening["pass_data"] = (
        (screening["valid_ic_ratio"] >= MIN_VALID_RATIO)
        & (screening["valid_spread_ratio"] >= MIN_VALID_RATIO)
    )


def add_relationship_strength_filter(screening):
    screening["pass_relationship_strength"] = (
        (screening["mean_ic"].abs() >= MIN_ABSOLUTE_MEAN_IC)
        & (screening["hac_tstat"].abs() >= MIN_ABSOLUTE_HAC_TSTAT)
    )


def add_direction_filter(screening):
    oriented_median_ic = (
        screening["direction"] * screening["median_ic"]
    )
    screening["pass_direction"] = (
        (screening["direction"] != 0)
        & (oriented_median_ic > 0)
        & (
            screening["direction_consistency"]
            >= MIN_DIRECTION_CONSISTENCY
        )
    )


def add_metric_agreement_filter(screening):
    oriented_pearson = (
        screening["direction"]
        * screening["mean_pearson_correlation"]
    )
    oriented_beta = (
        screening["direction"] * screening["mean_factor_beta"]
    )
    screening["pass_metric_agreement"] = (
        (oriented_pearson > 0)
        & (oriented_beta > 0)
    )


def add_quantile_shape_filter(screening):
    direction = screening["direction"]
    oriented_mean_spread = direction * screening["mean_raw_spread"]
    oriented_median_spread = direction * screening["median_raw_spread"]
    oriented_mean_curve = (
        direction * screening["mean_curve_monotonicity"]
    )
    oriented_median_curve = (
        direction * screening["median_curve_monotonicity"]
    )

    screening["oriented_mean_spread"] = oriented_mean_spread
    screening["oriented_median_spread"] = oriented_median_spread
    screening["oriented_mean_curve_monotonicity"] = oriented_mean_curve
    screening["oriented_median_curve_monotonicity"] = (
        oriented_median_curve
    )
    screening["pass_quantile_shape"] = (
        (oriented_mean_spread > 0)
        & (oriented_median_spread > 0)
        & (
            oriented_mean_curve
            >= MIN_ORIENTED_CURVE_MONOTONICITY
        )
        & (oriented_median_curve > 0)
    )


def apply_primary_filters(screening):
    screening = screening.copy()
    add_data_filter(screening)
    add_relationship_strength_filter(screening)
    add_direction_filter(screening)
    add_metric_agreement_filter(screening)
    add_quantile_shape_filter(screening)

    filter_columns = [
        "pass_data",
        "pass_relationship_strength",
        "pass_direction",
        "pass_metric_agreement",
        "pass_quantile_shape",
    ]
    screening["filters_passed"] = screening[filter_columns].sum(axis=1)
    screening["pass_primary_screen"] = screening[
        filter_columns
    ].all(axis=1)
    return screening


# -------------------------
# FIGURE HELPERS
def research_results(factor_results):
    return factor_results[factor_results["research_eligible"]].copy()


def horizon_axes(title, sharex=False, sharey=False):
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(16, 18),
        sharex=sharex,
        sharey=sharey,
    )
    figure.suptitle(f"{FACTOR_KEY}: {title}", fontsize=16)
    return figure, axes.ravel()


def save_figure(figure, file_name):
    os.makedirs(SCREENING_FIGURE_DIR, exist_ok=True)
    path = os.path.join(SCREENING_FIGURE_DIR, file_name)
    temporary = temporary_path(path)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(
        temporary,
        dpi=160,
        bbox_inches="tight",
        format="png",
    )
    plt.close(figure)
    os.replace(temporary, path)
    return path


# -------------------------
# DAILY FIGURES
def plot_daily_ic(factor_results):
    research = research_results(factor_results)
    figure, axes = horizon_axes("daily Spearman IC")

    for axis, horizon in zip(axes, FORWARD_HORIZONS):
        data = research[research["horizon_days"] == horizon]
        rolling = data["spearman_ic"].rolling(
            ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW // 2,
        ).mean()
        axis.plot(
            data["date"],
            data["spearman_ic"],
            color="steelblue",
            alpha=0.18,
            linewidth=0.6,
            label="Daily IC",
        )
        axis.plot(
            data["date"],
            rolling,
            color="navy",
            linewidth=1.5,
            label=f"{ROLLING_WINDOW}-day mean",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"Horizon {horizon} days")
        axis.set_ylabel("Spearman IC")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")

    return save_figure(figure, "01_daily_ic.png")


def plot_ic_distributions(factor_results):
    research = research_results(factor_results)
    figure, axes = horizon_axes(
        "Spearman IC distributions",
        sharex=True,
        sharey=True,
    )

    for axis, horizon in zip(axes, FORWARD_HORIZONS):
        values = research.loc[
            research["horizon_days"] == horizon,
            "spearman_ic",
        ].dropna()
        axis.hist(values, bins=50, color="steelblue", alpha=0.8)
        axis.axvline(values.mean(), color="darkred", label="Mean")
        axis.axvline(
            values.median(),
            color="darkorange",
            linestyle="--",
            label="Median",
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(f"Horizon {horizon} days")
        axis.set_xlabel("Daily Spearman IC")
        axis.set_ylabel("Dates")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")

    return save_figure(figure, "02_ic_distributions.png")


def plot_daily_spread(factor_results):
    research = research_results(factor_results)
    figure, axes = horizon_axes("daily Q10 - Q1 mean-return spread")

    for axis, horizon in zip(axes, FORWARD_HORIZONS):
        data = research[research["horizon_days"] == horizon]
        rolling = data["raw_spread"].rolling(
            ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW // 2,
        ).mean()
        axis.plot(
            data["date"],
            data["raw_spread"],
            color="mediumseagreen",
            alpha=0.18,
            linewidth=0.6,
            label="Daily spread",
        )
        axis.plot(
            data["date"],
            rolling,
            color="darkgreen",
            linewidth=1.5,
            label=f"{ROLLING_WINDOW}-day mean",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"Horizon {horizon} days")
        axis.set_ylabel("Forward return")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")

    return save_figure(figure, "03_daily_spread.png")


# -------------------------
# QUANTILE FIGURES
def plot_quantile_return_curves(factor_results):
    research = research_results(factor_results)
    quantiles = np.arange(1, QUANTILE_COUNT + 1)
    figure, axes = horizon_axes("Q1-Q10 future-return curves")

    for axis, horizon in zip(axes, FORWARD_HORIZONS):
        data = research[research["horizon_days"] == horizon]
        mean_curve = quantile_curve(data, "return_mean")
        median_curve = quantile_curve(data, "return_median")
        axis.plot(
            quantiles,
            mean_curve,
            marker="o",
            linewidth=1.8,
            label="Mean return",
        )
        axis.plot(
            quantiles,
            median_curve,
            marker="o",
            linestyle="--",
            linewidth=1.4,
            label="Median return",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"Horizon {horizon} days")
        axis.set_xticks(quantiles)
        axis.set_xlabel("Factor-score quantile")
        axis.set_ylabel("Forward return")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.2)
        axis.legend(loc="best")

    return save_figure(figure, "04_quantile_return_curves.png")


def plot_factor_score_return_curves(factor_results):
    research = research_results(factor_results)
    figure, axes = horizon_axes(
        "normalized factor score versus future return"
    )

    for axis, horizon in zip(axes, FORWARD_HORIZONS):
        data = research[research["horizon_days"] == horizon]
        scores = quantile_curve(data, "factor_score_mean")
        mean_returns = quantile_curve(data, "return_mean")
        median_returns = quantile_curve(data, "return_median")
        axis.plot(
            scores,
            mean_returns,
            marker="o",
            linewidth=1.8,
            label="Mean return",
        )
        axis.plot(
            scores,
            median_returns,
            marker="o",
            linestyle="--",
            linewidth=1.4,
            label="Median return",
        )

        for quantile, score, value in zip(
            range(1, QUANTILE_COUNT + 1),
            scores,
            mean_returns,
        ):
            axis.annotate(
                f"Q{quantile}",
                (score, value),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
            )

        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"Horizon {horizon} days")
        axis.set_xlabel("Average normalized factor score")
        axis.set_ylabel("Forward return")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.2)
        axis.legend(loc="best")

    return save_figure(figure, "05_factor_score_return_curves.png")


# -------------------------
# SUMMARY FIGURES
def plot_horizon_summary(screening):
    horizons = screening["horizon_days"].astype(str)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    figure.suptitle(f"{FACTOR_KEY}: horizon summary", fontsize=16)

    axes[0, 0].plot(
        horizons,
        screening["mean_ic"],
        marker="o",
        label="Mean IC",
    )
    axes[0, 0].plot(
        horizons,
        screening["median_ic"],
        marker="o",
        linestyle="--",
        label="Median IC",
    )
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("IC by horizon")
    axes[0, 0].set_xlabel("Horizon days")
    axes[0, 0].legend()

    axes[0, 1].bar(horizons, screening["hac_tstat"], color="steelblue")
    axes[0, 1].axhline(
        MIN_ABSOLUTE_HAC_TSTAT,
        color="darkred",
        linestyle="--",
    )
    axes[0, 1].axhline(
        -MIN_ABSOLUTE_HAC_TSTAT,
        color="darkred",
        linestyle="--",
    )
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("HAC t-stat")
    axes[0, 1].set_xlabel("Horizon days")

    axes[1, 0].bar(
        horizons,
        screening["direction_consistency"],
        color="darkorange",
    )
    axes[1, 0].axhline(
        MIN_DIRECTION_CONSISTENCY,
        color="darkred",
        linestyle="--",
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Daily IC direction consistency")
    axes[1, 0].set_xlabel("Horizon days")

    axes[1, 1].plot(
        horizons,
        screening["oriented_mean_curve_monotonicity"],
        marker="o",
        label="Mean-return curve",
    )
    axes[1, 1].plot(
        horizons,
        screening["oriented_median_curve_monotonicity"],
        marker="o",
        linestyle="--",
        label="Median-return curve",
    )
    axes[1, 1].axhline(
        MIN_ORIENTED_CURVE_MONOTONICITY,
        color="darkred",
        linestyle="--",
    )
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_ylim(-1, 1)
    axes[1, 1].set_title("Oriented Q1-Q10 monotonicity")
    axes[1, 1].set_xlabel("Horizon days")
    axes[1, 1].legend()

    for axis in axes.ravel():
        axis.grid(alpha=0.2)

    return save_figure(figure, "06_horizon_summary.png")


def plot_data_coverage(screening):
    horizons = screening["horizon_days"].astype(str)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    figure.suptitle(f"{FACTOR_KEY}: data coverage", fontsize=16)

    positions = np.arange(len(screening))
    width = 0.38
    axes[0].bar(
        positions - width / 2,
        screening["valid_ic_ratio"],
        width,
        label="Valid IC",
    )
    axes[0].bar(
        positions + width / 2,
        screening["valid_spread_ratio"],
        width,
        label="Valid Q10-Q1 spread",
    )
    axes[0].axhline(MIN_VALID_RATIO, color="darkred", linestyle="--")
    axes[0].set_xticks(positions, horizons)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Valid research dates")
    axes[0].set_xlabel("Horizon days")
    axes[0].set_ylabel("Valid-date ratio")
    axes[0].legend()

    axes[1].bar(
        horizons,
        screening["median_daily_assets"],
        color="slateblue",
    )
    axes[1].set_title("Median stocks used per date")
    axes[1].set_xlabel("Horizon days")
    axes[1].set_ylabel("Stocks")

    for axis in axes:
        axis.grid(alpha=0.2)

    return save_figure(figure, "07_data_coverage.png")


def plot_filter_map(screening):
    filter_columns = [
        "pass_data",
        "pass_relationship_strength",
        "pass_direction",
        "pass_metric_agreement",
        "pass_quantile_shape",
    ]
    labels = [
        "Data",
        "Relationship strength",
        "Direction",
        "Metric agreement",
        "Quantile shape",
    ]
    values = screening[filter_columns].to_numpy(dtype=int)
    figure, axis = plt.subplots(figsize=(11, 6))
    image = axis.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    axis.set_yticks(
        np.arange(len(screening)),
        [f"h{horizon}" for horizon in screening["horizon_days"]],
    )
    axis.set_title(f"{FACTOR_KEY}: primary filter map")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                "PASS" if values[row, column] else "FAIL",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    figure.colorbar(image, ax=axis, ticks=[0, 1])
    return save_figure(figure, "08_filter_map.png")


def create_figures(factor_results, screening):
    return [
        plot_daily_ic(factor_results),
        plot_ic_distributions(factor_results),
        plot_daily_spread(factor_results),
        plot_quantile_return_curves(factor_results),
        plot_factor_score_return_curves(factor_results),
        plot_horizon_summary(screening),
        plot_data_coverage(screening),
        plot_filter_map(screening),
    ]


# -------------------------
# RESULT SAVING
def save_screening(screening):
    os.makedirs(LEGACY_RESULT_DIR, exist_ok=True)
    temporary = temporary_path(SCREENING_RESULT_PATH)
    screening.to_csv(temporary, index=False)
    os.replace(temporary, SCREENING_RESULT_PATH)


# -------------------------
# COMPLETE SINGLE-FACTOR SCREENING
def run_single_factor_screening():
    factor_results = load_factor_results(FACTOR_KEY)
    summary = summarize_factor(factor_results)
    screening = apply_primary_filters(summary)
    save_screening(screening)
    figure_paths = create_figures(factor_results, screening)

    display_columns = [
        "horizon_days",
        "mean_ic",
        "hac_tstat",
        "direction_consistency",
        "oriented_mean_spread",
        "oriented_mean_curve_monotonicity",
        "filters_passed",
        "pass_primary_screen",
    ]
    print(f"Factor: {FACTOR_KEY}")
    print(screening[display_columns].to_string(index=False))
    print(f"Result: {SCREENING_RESULT_PATH}")
    print(f"Figures: {SCREENING_FIGURE_DIR}")
    print(f"Figure files: {len(figure_paths)}")
    return screening


if __name__ == "__main__":
    run_single_factor_screening()
