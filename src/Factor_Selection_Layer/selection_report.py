import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from selection_config import (
    FORWARD_HORIZONS,
    MAX_FACTOR_FIGURES,
    QUANTILE_COUNT,
    SELECTION_FIGURES_DIR,
)
from selection_storage import safe_factor_name, temporary_path


# -------------------------
# FIGURE SAVING
def save_figure(figure, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = temporary_path(path)
    figure.tight_layout()
    figure.savefig(
        temporary,
        dpi=160,
        bbox_inches="tight",
        format="png",
    )
    plt.close(figure)
    os.replace(temporary, path)
    return path


def factor_figure_path(factor_key, suffix):
    family, variant = factor_key.split("|", maxsplit=1)
    name = safe_factor_name(family, variant)
    return os.path.join(SELECTION_FIGURES_DIR, f"{name}__{suffix}.png")


# -------------------------
# COMPLETE-SCOPE OVERVIEW
def heatmap_panel(axis, table, value, title, color_map, center=None):
    pivot = table.pivot(
        index="factor_key",
        columns="horizon_days",
        values=value,
    ).reindex(columns=FORWARD_HORIZONS)
    matrix = pivot.to_numpy(dtype=float)
    finite = matrix[np.isfinite(matrix)]
    kwargs = {}

    if center is not None and finite.size:
        limit = max(abs(finite.min() - center), abs(finite.max() - center))
        kwargs = {"vmin": center - limit, "vmax": center + limit}

    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        **kwargs,
    )
    axis.set_title(title)
    axis.set_xticks(
        np.arange(len(FORWARD_HORIZONS)),
        [str(horizon) for horizon in FORWARD_HORIZONS],
    )
    axis.set_xlabel("Forward horizon, trading days")
    axis.set_yticks(
        np.arange(len(pivot.index)),
        pivot.index,
        fontsize=7,
    )
    plt.colorbar(image, ax=axis, fraction=0.025, pad=0.02)


def plot_hypothesis_overview(cards):
    ordered = cards.sort_values(["factor_key", "horizon_days"])
    figure, axes = plt.subplots(2, 2, figsize=(18, max(8, 0.35 * ordered["factor_key"].nunique() + 6)))
    figure.suptitle("Factor hypothesis overview", fontsize=16)
    heatmap_panel(
        axes[0, 0],
        ordered,
        "spearman_ic_mean",
        "Mean daily Spearman IC",
        "RdBu_r",
        center=0,
    )
    heatmap_panel(
        axes[0, 1],
        ordered,
        "strongest_economic_effect_abs_tstat",
        "Strongest economic-effect |HAC t-stat|",
        "magma",
    )
    heatmap_panel(
        axes[1, 0],
        ordered,
        "strongest_economic_effect_monthly_direction_rate",
        "Strongest economic-effect monthly consistency",
        "YlGn",
    )
    heatmap_panel(
        axes[1, 1],
        ordered,
        "mean_rank_autocorrelation",
        "Factor rank autocorrelation",
        "viridis",
    )
    return save_figure(
        figure,
        os.path.join(SELECTION_FIGURES_DIR, "hypothesis_overview.png"),
    )


# -------------------------
# FACTOR-SPECIFIC FIGURES
def plot_quantile_curves(factor_key, curves, cards):
    factor_curves = curves[curves["factor_key"] == factor_key]
    factor_cards = cards[cards["factor_key"] == factor_key].set_index(
        "horizon_days"
    )
    figure, axes = plt.subplots(4, 2, figsize=(15, 18))
    figure.suptitle(f"{factor_key}: quantile return shape", fontsize=16)

    for axis, horizon in zip(axes.ravel(), FORWARD_HORIZONS):
        data = factor_curves[
            factor_curves["horizon_days"] == horizon
        ].sort_values("quantile")
        middle = data.loc[
            data["quantile"].isin((4, 5, 6, 7)),
            "average_return",
        ].mean()
        relative_return = data["average_return"] - middle
        axis.plot(
            data["quantile"],
            relative_return,
            marker="o",
            linewidth=1.7,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        pattern = factor_cards.loc[horizon, "pattern"]
        axis.set_title(f"h{horizon}: {pattern}")
        axis.set_xticks(range(1, QUANTILE_COUNT + 1))
        axis.set_xlabel("Factor-score quantile")
        axis.set_ylabel("Return relative to Q4-Q7")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.2)

    return save_figure(
        figure,
        factor_figure_path(factor_key, "quantile_curves"),
    )


def plot_factor_dashboard(factor_key, cards):
    data = cards[cards["factor_key"] == factor_key].sort_values(
        "horizon_days"
    )
    horizons = data["horizon_days"].astype(str)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    figure.suptitle(f"{factor_key}: evidence by horizon", fontsize=16)

    axes[0, 0].plot(horizons, data["spearman_ic_mean"], marker="o")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Mean daily Spearman IC")

    axes[0, 1].bar(
        horizons,
        data["strongest_economic_effect_abs_tstat"],
        color="steelblue",
    )
    axes[0, 1].axhline(1.96, color="darkred", linestyle="--")
    axes[0, 1].set_title("Strongest economic-effect |HAC t-stat|")

    axes[1, 0].plot(
        horizons,
        data["strongest_economic_effect_monthly_direction_rate"],
        marker="o",
        label="Monthly",
    )
    axes[1, 0].plot(
        horizons,
        data["strongest_economic_effect_annual_direction_rate"],
        marker="o",
        label="Annual",
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Direction consistency")
    axes[1, 0].legend()

    axes[1, 1].plot(
        horizons,
        data["mean_rank_autocorrelation"],
        marker="o",
        color="darkorange",
    )
    axes[1, 1].set_ylim(-0.1, 1)
    axes[1, 1].set_title("Rank autocorrelation at matching horizon")

    for axis in axes.ravel():
        axis.set_xlabel("Forward horizon, trading days")
        axis.grid(alpha=0.2)

    return save_figure(
        figure,
        factor_figure_path(factor_key, "evidence_dashboard"),
    )


def plot_monthly_ic(factor_key, time_stability):
    data = time_stability[
        (time_stability["factor_key"] == factor_key)
        & (time_stability["effect"] == "spearman_ic")
        & (time_stability["frequency"] == "monthly")
    ].copy()
    data["period_date"] = pd.to_datetime(data["period"])
    data["year"] = data["period_date"].dt.year
    data["month"] = data["period_date"].dt.month
    limit = data["mean_effect"].abs().quantile(0.98)
    limit = max(float(limit), 0.01) if pd.notna(limit) else 0.05
    figure, axes = plt.subplots(4, 2, figsize=(16, 18))
    figure.suptitle(f"{factor_key}: monthly mean IC", fontsize=16)

    for axis, horizon in zip(axes.ravel(), FORWARD_HORIZONS):
        horizon_data = data[data["horizon_days"] == horizon]
        pivot = horizon_data.pivot(
            index="year",
            columns="month",
            values="mean_effect",
        ).reindex(columns=range(1, 13))
        image = axis.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(f"Horizon {horizon} days")
        axis.set_xticks(np.arange(12), range(1, 13))
        axis.set_yticks(np.arange(len(pivot.index)), pivot.index, fontsize=7)
        axis.set_xlabel("Month")
        axis.set_ylabel("Year")
        plt.colorbar(image, ax=axis, fraction=0.025, pad=0.02)

    return save_figure(
        figure,
        factor_figure_path(factor_key, "monthly_ic"),
    )


def factor_keys_for_figures(cards):
    factor_scores = (
        cards.groupby("factor_key", sort=False)["evidence_score"]
        .max()
        .sort_values(ascending=False)
    )
    return factor_scores.head(MAX_FACTOR_FIGURES).index.tolist()


def create_selection_figures(cards, curves, time_stability):
    paths = [plot_hypothesis_overview(cards)]

    for factor_key in factor_keys_for_figures(cards):
        paths.extend(
            [
                plot_factor_dashboard(factor_key, cards),
                plot_quantile_curves(factor_key, curves, cards),
                plot_monthly_ic(factor_key, time_stability),
            ]
        )

    return paths


# -------------------------
# MARKDOWN REPORT
def format_number(value, digits=4):
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def markdown_cell(value):
    return str(value).replace("|", "\\|")


def markdown_table(cards):
    columns = [
        "Horizon",
        "Pattern",
        "Direction",
        "Mean IC",
        "IC HAC t",
        "Primary effect",
        "Effect mean",
        "Effect HAC t",
        "Monthly stability",
        "Annual stability",
        "Rank autocorr",
        "Status",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in cards.sort_values(["factor_key", "horizon_days"]).itertuples():
        values = [
            f"{row.factor_key} / h{row.horizon_days}",
            row.pattern,
            row.pattern_direction,
            format_number(row.spearman_ic_mean),
            format_number(row.spearman_ic_hac_tstat, 2),
            row.primary_effect,
            format_number(row.primary_effect_mean),
            format_number(row.primary_effect_hac_tstat, 2),
            format_number(row.primary_effect_monthly_direction_rate, 2),
            format_number(row.primary_effect_annual_direction_rate, 2),
            format_number(row.mean_rank_autocorrelation, 2),
            row.evidence_status,
        ]
        lines.append("| " + " | ".join(map(markdown_cell, values)) + " |")

    return "\n".join(lines)


def economic_markdown_table(cards):
    columns = [
        "Hypothesis",
        "Pattern",
        "Strongest economic effect",
        "Q10-Q1 mean / t",
        "Q10-middle mean / t",
        "Middle-Q1 mean / t",
        "Edges-middle mean / t",
        "Status",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in cards.itertuples():
        values = [
            f"{row.factor_key} / h{row.horizon_days}",
            row.pattern,
            row.strongest_economic_effect,
            (
                f"{format_number(row.q10_minus_q1_mean)} / "
                f"{format_number(row.q10_minus_q1_hac_tstat, 2)}"
            ),
            (
                f"{format_number(row.q10_minus_middle_mean)} / "
                f"{format_number(row.q10_minus_middle_hac_tstat, 2)}"
            ),
            (
                f"{format_number(row.middle_minus_q1_mean)} / "
                f"{format_number(row.middle_minus_q1_hac_tstat, 2)}"
            ),
            (
                f"{format_number(row.edges_minus_middle_mean)} / "
                f"{format_number(row.edges_minus_middle_hac_tstat, 2)}"
            ),
            row.evidence_status,
        ]
        lines.append("| " + " | ".join(map(markdown_cell, values)) + " |")

    return "\n".join(lines)


def build_selection_report(cards, effect_tests, figure_paths):
    factor_count = cards["factor_key"].nunique()
    hypothesis_count = cards["hypothesis_key"].nunique()
    full_scope = bool(effect_tests["full_research_scope"].iloc[0])
    pattern_counts = cards["pattern"].value_counts()
    status_counts = cards["evidence_status"].value_counts()
    visible_cards = cards[
        cards["pattern"] != "no_stable_structure"
    ].head(50)
    visible_scope = "detected patterns"

    if visible_cards.empty:
        visible_cards = cards.head(25)
        visible_scope = "strongest available economic effects"
    lines = [
        "# Factor Selection Report",
        "",
        "## Scope",
        "",
        f"- Factor configurations analyzed: `{factor_count}`.",
        f"- Factor-horizon hypotheses analyzed: `{hypothesis_count}`.",
        f"- Effect tests analyzed: `{len(effect_tests)}`.",
        f"- Full 448-hypothesis scope: `{'yes' if full_scope else 'no'}`.",
        "",
    ]

    if not full_scope:
        lines.extend(
            [
                "> The current run is intentionally restricted. Benjamini-Hochberg values describe only the active run and are not final full-universe evidence.",
                "",
            ]
        )

    lines.extend(
        [
            "## Pattern Classification",
            "",
        ]
    )
    for pattern, count in pattern_counts.items():
        lines.append(f"- `{pattern}`: {count} hypotheses.")

    lines.extend(["", "## Evidence Status", ""])
    for status, count in status_counts.items():
        lines.append(f"- `{status}`: {count} hypotheses.")

    lines.extend(
        [
            "",
            "## Economic Pattern Cards",
            "",
            f"The table shows up to 50 {visible_scope}. Complete results remain in `hypothesis_cards.csv` and `effect_tests.csv`.",
            "",
            economic_markdown_table(visible_cards),
            "",
            "## IC and Stability Diagnostics",
            "",
            "IC is reported as supporting information. It does not create a candidate when no economic pattern is detected.",
            "",
            markdown_table(visible_cards),
            "",
            "## Figures",
            "",
        ]
    )
    report_directory = os.path.dirname(
        os.path.join(SELECTION_FIGURES_DIR, "placeholder")
    )

    for path in figure_paths:
        relative = os.path.relpath(path, os.path.dirname(report_directory))
        relative = relative.replace("\\", "/")
        label = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
        lines.extend([f"### {label.title()}", "", f"![{label}]({relative})", ""])

    lines.extend(
        [
            "## Interpretation Rules",
            "",
            "- `positive_monotonic` and `negative_monotonic` describe ordered Q1-Q10 relationships.",
            "- `upper_tail` and `lower_tail` describe an effect concentrated in one extreme quantile.",
            "- `both_tails_vs_middle` describes a U-shaped or inverted-U relationship.",
            "- `no_stable_structure` means that no supported shape was identified by the current descriptive rules.",
            "- A provisional candidate is not a validated trading strategy.",
            "",
        ]
    )
    return "\n".join(lines)
