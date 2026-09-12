import numpy as np
import pandas as pd

from factor_config import (
    MIN_OOS_IC_OBSERVATIONS,
    MIN_SELECTION_IC_OBSERVATIONS,
    RESEARCH_END_DATE,
    RESEARCH_START_DATE,
    ROBUSTNESS_CONFIGS,
    SELECTION_WEIGHTS,
)
from factor_storage import (
    load_factor_matrix,
    save_selected_factor_matrices,
)
from sensitivity import summarize_ic
from transforms import percentile_rank


# -------------------------
# ROBUSTNESS WINDOWS
def window_offsets(configuration):
    if "selection_months" in configuration:
        return (
            pd.DateOffset(months=configuration["selection_months"]),
            pd.DateOffset(months=configuration["oos_months"]),
            pd.DateOffset(months=configuration["step_months"]),
        )

    return (
        pd.DateOffset(years=configuration["selection_years"]),
        pd.DateOffset(years=configuration["oos_years"]),
        pd.DateOffset(years=configuration["step_years"]),
    )


def generate_robustness_windows(index, layer_name, configuration):
    index = pd.DatetimeIndex(index).sort_values().unique()
    research_end = pd.Timestamp(RESEARCH_END_DATE or index.max())
    selection_offset, oos_offset, step_offset = window_offsets(configuration)
    selection_start_boundary = pd.Timestamp(RESEARCH_START_DATE)
    rows = []
    window_number = 1

    while True:
        oos_start_boundary = selection_start_boundary + selection_offset
        oos_end_boundary = oos_start_boundary + oos_offset

        if oos_end_boundary - pd.Timedelta(days=1) > research_end:
            break

        selection_dates = index[
            (index >= selection_start_boundary)
            & (index < oos_start_boundary)
        ]
        oos_dates = index[
            (index >= oos_start_boundary)
            & (index < oos_end_boundary)
        ]

        if len(selection_dates) and len(oos_dates):
            rows.append(
                {
                    "robustness_layer": layer_name,
                    "window": window_number,
                    "selection_start": selection_dates[0],
                    "selection_end": selection_dates[-1],
                    "oos_start": oos_dates[0],
                    "oos_end": oos_dates[-1],
                }
            )
            window_number += 1

        selection_start_boundary += step_offset

    return pd.DataFrame(rows)


# -------------------------
# SELECTION METRICS
def purged_selection_values(
    series,
    index,
    selection_start,
    oos_start,
    horizon,
):
    selection_start_position = index.searchsorted(selection_start, side="left")
    oos_start_position = index.searchsorted(oos_start, side="left")
    purged_end_position = oos_start_position - horizon - 1

    if purged_end_position < selection_start_position:
        return pd.Series(dtype=float)

    positions = index[selection_start_position : purged_end_position + 1]
    return series.reindex(positions).dropna()


def selection_metrics(values, horizon):
    values = pd.Series(values).dropna()
    midpoint = len(values) // 2
    early = values.iloc[:midpoint]
    late = values.iloc[midpoint:]
    full_statistics = summarize_ic(values, horizon)
    late_statistics = summarize_ic(late, horizon)

    return {
        **full_statistics,
        "early_mean_ic": early.mean() if not early.empty else np.nan,
        "late_mean_ic": late.mean() if not late.empty else np.nan,
        "late_tstat": late_statistics["tstat"],
        "stability": (
            abs(early.mean() - late.mean())
            if not early.empty and not late.empty
            else np.nan
        ),
    }


def add_selection_scores(candidates):
    candidates = candidates.copy()
    candidates["selection_score"] = np.nan
    candidates["eligible"] = (
        (candidates["observations"] >= MIN_SELECTION_IC_OBSERVATIONS)
        & (candidates["early_mean_ic"] > 0)
        & (candidates["late_mean_ic"] > 0)
    )

    for _, family_candidates in candidates.groupby("family"):
        rows = family_candidates.index
        score = (
            SELECTION_WEIGHTS["late_mean_ic"]
            * family_candidates["late_mean_ic"].rank(pct=True)
            + SELECTION_WEIGHTS["late_tstat"]
            * family_candidates["late_tstat"].rank(pct=True)
            + SELECTION_WEIGHTS["mean_ic"]
            * family_candidates["mean_ic"].rank(pct=True)
            + SELECTION_WEIGHTS["tstat"]
            * family_candidates["tstat"].rank(pct=True)
            + SELECTION_WEIGHTS["positive_rate"]
            * family_candidates["positive_rate"].rank(pct=True)
            + SELECTION_WEIGHTS["early_mean_ic"]
            * family_candidates["early_mean_ic"].rank(pct=True)
            + SELECTION_WEIGHTS["stability"]
            * (-family_candidates["stability"]).rank(pct=True)
        )
        candidates.loc[rows, "selection_score"] = score

    return candidates


# -------------------------
# ONE WINDOW
def evaluate_window(daily_ic, metadata, index, window):
    rows = []

    for hypothesis in metadata.itertuples():
        horizon = int(hypothesis.horizon_days)
        selection_values = purged_selection_values(
            daily_ic[hypothesis.key],
            index,
            window.selection_start,
            window.oos_start,
            horizon,
        )
        metrics = selection_metrics(selection_values, horizon)
        rows.append(
            {
                "robustness_layer": window.robustness_layer,
                "window": window.window,
                "selection_start": window.selection_start,
                "selection_end": window.selection_end,
                "selection_ic_start": (
                    selection_values.index.min()
                    if not selection_values.empty
                    else pd.NaT
                ),
                "selection_ic_end": (
                    selection_values.index.max()
                    if not selection_values.empty
                    else pd.NaT
                ),
                "oos_start": window.oos_start,
                "oos_end": window.oos_end,
                "key": hypothesis.key,
                "family": hypothesis.family,
                "variant": hypothesis.variant,
                "horizon_days": horizon,
                "parameters": hypothesis.parameters,
                **metrics,
            }
        )

    candidates = add_selection_scores(pd.DataFrame(rows))
    candidates["selected"] = False

    for _, family_candidates in candidates.groupby("family"):
        eligible_candidates = family_candidates[
            family_candidates["eligible"]
        ]

        if eligible_candidates.empty:
            continue

        winner = eligible_candidates.sort_values(
            "selection_score",
            ascending=False,
        ).iloc[0]
        candidates.loc[winner.name, "selected"] = True
        oos_values = daily_ic[winner["key"]].loc[
            window.oos_start : window.oos_end
        ].dropna()
        oos_statistics = summarize_ic(oos_values, int(winner["horizon_days"]))

        for metric, value in oos_statistics.items():
            candidates.loc[winner.name, f"oos_{metric}"] = value

        candidates.loc[winner.name, "oos_eligible"] = (
            len(oos_values) >= MIN_OOS_IC_OBSERVATIONS
        )

    return candidates


# -------------------------
# COMPLETE ROBUSTNESS
def run_robustness(daily_ic, metadata, trading_index):
    all_results = []

    for layer_name, configuration in ROBUSTNESS_CONFIGS.items():
        windows = generate_robustness_windows(
            trading_index,
            layer_name,
            configuration,
        )
        print(f"Robustness {layer_name}: {len(windows)} windows")

        for window in windows.itertuples(index=False):
            all_results.append(
                evaluate_window(
                    daily_ic,
                    metadata,
                    trading_index,
                    window,
                )
            )

    if not all_results:
        raise ValueError("No complete robustness windows were created")

    results = pd.concat(all_results, ignore_index=True)
    selected = results[results["selected"]].copy().reset_index(drop=True)
    return results, selected


# -------------------------
# SELECTED FACTOR MATRICES
def save_selected_signals(selected, trading_index, ticker_columns):
    saved_files = []

    for (robustness_layer, family), selections in selected.groupby(
        ["robustness_layer", "family"]
    ):
        scores = pd.DataFrame(
            np.nan,
            index=trading_index,
            columns=ticker_columns,
            dtype="float32",
        )

        for selection in selections.itertuples():
            factor = load_factor_matrix(selection.family, selection.variant)
            dates = scores.loc[selection.oos_start : selection.oos_end].index
            scores.loc[dates] = factor.reindex(
                index=dates,
                columns=ticker_columns,
            ).to_numpy()

        ranks = percentile_rank(scores)
        score_path, rank_path = save_selected_factor_matrices(
            robustness_layer,
            family,
            scores,
            ranks,
        )
        saved_files.append(
            {
                "robustness_layer": robustness_layer,
                "family": family,
                "score_path": score_path,
                "rank_path": rank_path,
            }
        )

    return saved_files
