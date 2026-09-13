import numpy as np
import pandas as pd

from ic_analysis import summarize_ic
from selection_config import (
    MIN_OOS_IC_OBSERVATIONS,
    MIN_SELECTION_IC_OBSERVATIONS,
    RESEARCH_END_DATE,
    RESEARCH_START_DATE,
    ROBUSTNESS_CONFIGS,
)


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
# PERIOD PREPARATION
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

    dates = index[selection_start_position: purged_end_position + 1]
    return series.reindex(dates).dropna()


def oos_is_complete(index, oos_end, horizon):
    oos_end_position = index.searchsorted(oos_end, side="right") - 1

    if oos_end_position < 0:
        return False

    return oos_end_position + horizon < len(index)


# -------------------------
# WINDOW METRICS
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
        "within_selection_change": (
            late.mean() - early.mean()
            if not early.empty and not late.empty
            else np.nan
        ),
    }


def evaluate_window(daily_ic, metadata, index, window):
    rows = []

    for hypothesis in metadata.itertuples(index=False):
        horizon = int(hypothesis.horizon_days)
        selection_values = purged_selection_values(
            daily_ic[hypothesis.key],
            index,
            window.selection_start,
            window.oos_start,
            horizon,
        )
        oos_values = daily_ic[hypothesis.key].loc[
            window.oos_start:window.oos_end
        ].dropna()
        sample_statistics = selection_metrics(selection_values, horizon)
        oos_statistics = summarize_ic(oos_values, horizon)
        sample_eligible = (
            sample_statistics["observations"]
            >= MIN_SELECTION_IC_OBSERVATIONS
        )
        complete_oos = oos_is_complete(index, window.oos_end, horizon)
        oos_eligible = (
            complete_oos
            and oos_statistics["observations"] >= MIN_OOS_IC_OBSERVATIONS
        )
        sample_mean = sample_statistics["mean_ic"]
        oos_mean = oos_statistics["mean_ic"]
        comparable = (
            sample_eligible
            and oos_eligible
            and pd.notna(sample_mean)
            and pd.notna(oos_mean)
        )

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
                **sample_statistics,
                "sample_eligible": sample_eligible,
                "oos_complete": complete_oos,
                "oos_observations": oos_statistics["observations"],
                "oos_mean_ic": oos_mean,
                "oos_std_ic": oos_statistics["std_ic"],
                "oos_tstat": oos_statistics["tstat"],
                "oos_positive_rate": oos_statistics["positive_rate"],
                "oos_eligible": oos_eligible,
                "ic_change": oos_mean - sample_mean if comparable else np.nan,
                "absolute_ic_change": (
                    abs(oos_mean - sample_mean) if comparable else np.nan
                ),
                "sign_consistent": (
                    np.sign(sample_mean) == np.sign(oos_mean)
                    if comparable
                    else pd.NA
                ),
            }
        )

    return pd.DataFrame(rows)


# -------------------------
# COMPLETE OPTIONAL ROBUSTNESS
def run_robustness(daily_ic, metadata, trading_index):
    trading_index = pd.DatetimeIndex(trading_index).sort_values().unique()
    all_results = []

    for layer_name, configuration in ROBUSTNESS_CONFIGS.items():
        horizons = set(configuration["horizons"])
        layer_metadata = metadata[
            metadata["horizon_days"].isin(horizons)
        ]
        windows = generate_robustness_windows(
            trading_index,
            layer_name,
            configuration,
        )
        print(
            f"Optional robustness {layer_name}: {len(windows)} windows x "
            f"{len(layer_metadata)} hypotheses"
        )

        for window in windows.itertuples(index=False):
            all_results.append(
                evaluate_window(
                    daily_ic,
                    layer_metadata,
                    trading_index,
                    window,
                )
            )

    if not all_results:
        raise ValueError("No complete robustness windows were created")

    return pd.concat(all_results, ignore_index=True)


# -------------------------
# OPTIONAL ROBUSTNESS SUMMARY
def weighted_mean(frame, value_column, weight_column):
    valid = frame[value_column].notna() & frame[weight_column].gt(0)

    if not valid.any():
        return np.nan

    return np.average(
        frame.loc[valid, value_column],
        weights=frame.loc[valid, weight_column],
    )


def aggregate_robustness(robustness_results):
    rows = []

    for (layer, key), group in robustness_results.groupby(
        ["robustness_layer", "key"],
        sort=False,
    ):
        first = group.iloc[0]
        sample = group[group["sample_eligible"]]
        oos = group[group["oos_eligible"]]
        comparable = group[
            group["sample_eligible"] & group["oos_eligible"]
        ].copy()
        comparable["sample_absolute_mean_ic"] = comparable["mean_ic"].abs()
        comparable["oos_absolute_mean_ic"] = comparable["oos_mean_ic"].abs()
        comparable["oriented_oos_mean_ic"] = (
            np.sign(comparable["mean_ic"]) * comparable["oos_mean_ic"]
        )
        comparable["ic_magnitude_change"] = (
            comparable["oos_absolute_mean_ic"]
            - comparable["sample_absolute_mean_ic"]
        )

        weighted_sample_absolute_mean_ic = weighted_mean(
            comparable,
            "sample_absolute_mean_ic",
            "observations",
        )
        weighted_oos_absolute_mean_ic = weighted_mean(
            comparable,
            "oos_absolute_mean_ic",
            "oos_observations",
        )
        magnitude_retention_ratio = (
            weighted_oos_absolute_mean_ic
            / weighted_sample_absolute_mean_ic
            if weighted_sample_absolute_mean_ic > 0
            else np.nan
        )

        rows.append(
            {
                "robustness_layer": layer,
                "key": key,
                "family": first["family"],
                "variant": first["variant"],
                "horizon_days": first["horizon_days"],
                "parameters": first["parameters"],
                "total_windows": len(group),
                "sample_eligible_windows": len(sample),
                "oos_complete_windows": int(group["oos_complete"].sum()),
                "oos_eligible_windows": len(oos),
                "weighted_sample_mean_ic": weighted_mean(
                    sample,
                    "mean_ic",
                    "observations",
                ),
                "median_sample_mean_ic": sample["mean_ic"].median(),
                "weighted_oos_mean_ic": weighted_mean(
                    oos,
                    "oos_mean_ic",
                    "oos_observations",
                ),
                "median_oos_mean_ic": oos["oos_mean_ic"].median(),
                "oos_ic_std_across_windows": oos["oos_mean_ic"].std(),
                "positive_sample_window_rate": (
                    (sample["mean_ic"] > 0).mean()
                    if not sample.empty
                    else np.nan
                ),
                "positive_oos_window_rate": (
                    (oos["oos_mean_ic"] > 0).mean()
                    if not oos.empty
                    else np.nan
                ),
                "weighted_sample_absolute_mean_ic": (
                    weighted_sample_absolute_mean_ic
                ),
                "weighted_oos_absolute_mean_ic": (
                    weighted_oos_absolute_mean_ic
                ),
                "weighted_oriented_oos_mean_ic": weighted_mean(
                    comparable,
                    "oriented_oos_mean_ic",
                    "oos_observations",
                ),
                "median_oriented_oos_mean_ic": comparable[
                    "oriented_oos_mean_ic"
                ].median(),
                "magnitude_retention_ratio": magnitude_retention_ratio,
                "weighted_ic_magnitude_change": weighted_mean(
                    comparable,
                    "ic_magnitude_change",
                    "oos_observations",
                ),
                "sign_consistency_rate": (
                    comparable["sign_consistent"].astype(float).mean()
                    if not comparable.empty
                    else np.nan
                ),
                "mean_ic_change": comparable["ic_change"].mean(),
                "mean_absolute_ic_change": comparable[
                    "absolute_ic_change"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)
