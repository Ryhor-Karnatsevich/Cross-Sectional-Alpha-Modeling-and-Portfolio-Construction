import os

import pandas as pd

from factor_config import (
    FACTOR_SCREENING_FUNNEL_PATH,
    FACTOR_SCREENING_PATH,
    PASSED_FACTOR_CANDIDATES_PATH,
    ROBUSTNESS_CONFIGS,
    ROBUSTNESS_SUMMARY_PATH,
    SCREENING_MIN_COMPLETE_WINDOW_RATIO,
    SCREENING_MIN_POSITIVE_WINDOW_RATE,
    SCREENING_MIN_SIGN_CONSISTENCY_RATE,
)


IDENTITY_COLUMNS = (
    "key",
    "family",
    "variant",
    "horizon_days",
    "parameters",
)


# -------------------------
# LAYER PREPARATION
def prepare_layer_summary(robustness_summary, layer_name):
    layer = robustness_summary[
        robustness_summary["robustness_layer"] == layer_name
    ].copy()

    if layer["key"].duplicated().any():
        raise ValueError(f"Duplicated hypotheses in {layer_name} summary")

    layer = layer.drop(columns="robustness_layer")
    metric_columns = [
        column
        for column in layer.columns
        if column not in IDENTITY_COLUMNS
    ]
    return layer.rename(
        columns={
            column: f"{layer_name}_{column}"
            for column in metric_columns
        }
    )


def store_condition(screening, layer_name, condition_name, condition, required):
    column = f"pass_{layer_name}_{condition_name}"
    screening[column] = condition.astype("boolean")
    screening.loc[~required, column] = pd.NA
    return column


# ================================================================================
# SCREENING CONDITIONS
# Only the functions in this section define which hypotheses pass the screening.
# ================================================================================
def check_data_coverage(screening, layer_name, required):
    total_windows = screening[f"{layer_name}_total_windows"]
    sample_ratio = (
        screening[f"{layer_name}_sample_eligible_windows"] / total_windows
    )
    oos_ratio = (
        screening[f"{layer_name}_oos_eligible_windows"] / total_windows
    )
    condition = (
        (sample_ratio >= SCREENING_MIN_COMPLETE_WINDOW_RATIO)
        & (oos_ratio >= SCREENING_MIN_COMPLETE_WINDOW_RATIO)
    )
    return store_condition(
        screening, layer_name, "data", condition, required
    )


def check_positive_weighted_ic(screening, layer_name, required):
    condition = (
        (screening[f"{layer_name}_weighted_sample_mean_ic"] > 0)
        & (screening[f"{layer_name}_weighted_oos_mean_ic"] > 0)
    )
    return store_condition(
        screening, layer_name, "weighted_ic", condition, required
    )


def check_positive_median_ic(screening, layer_name, required):
    condition = (
        (screening[f"{layer_name}_median_sample_mean_ic"] > 0)
        & (screening[f"{layer_name}_median_oos_mean_ic"] > 0)
    )
    return store_condition(
        screening, layer_name, "median_ic", condition, required
    )


def check_positive_window_rate(screening, layer_name, required):
    condition = (
        (
            screening[f"{layer_name}_positive_sample_window_rate"]
            >= SCREENING_MIN_POSITIVE_WINDOW_RATE
        )
        & (
            screening[f"{layer_name}_positive_oos_window_rate"]
            >= SCREENING_MIN_POSITIVE_WINDOW_RATE
        )
    )
    return store_condition(
        screening, layer_name, "positive_windows", condition, required
    )


def check_sign_consistency(screening, layer_name, required):
    condition = (
        screening[f"{layer_name}_sign_consistency_rate"]
        >= SCREENING_MIN_SIGN_CONSISTENCY_RATE
    )
    return store_condition(
        screening, layer_name, "sign_consistency", condition, required
    )


def apply_layer_conditions(screening, layer_name, required):
    condition_columns = [
        check_data_coverage(screening, layer_name, required),
        check_positive_weighted_ic(screening, layer_name, required),
        check_positive_median_ic(screening, layer_name, required),
        check_positive_window_rate(screening, layer_name, required),
        check_sign_consistency(screening, layer_name, required),
    ]
    screening[f"pass_{layer_name}"] = screening[
        condition_columns
    ].fillna(True).all(axis=1)
    return condition_columns


# ================================================================================
# SCREENING ASSEMBLY AND OUTPUT
# ================================================================================
def build_factor_screening(robustness_summary):
    short = prepare_layer_summary(robustness_summary, "short")
    long = prepare_layer_summary(robustness_summary, "long")
    screening = long.merge(
        short,
        on=list(IDENTITY_COLUMNS),
        how="outer",
        validate="one_to_one",
    )
    short_horizons = set(ROBUSTNESS_CONFIGS["short"]["horizons"])
    long_horizons = set(ROBUSTNESS_CONFIGS["long"]["horizons"])
    screening["short_required"] = screening["horizon_days"].isin(
        short_horizons
    )
    screening["long_required"] = screening["horizon_days"].isin(
        long_horizons
    )
    short_conditions = apply_layer_conditions(
        screening,
        "short",
        screening["short_required"],
    )
    long_conditions = apply_layer_conditions(
        screening,
        "long",
        screening["long_required"],
    )
    condition_names = (
        "data",
        "weighted_ic",
        "median_ic",
        "positive_windows",
        "sign_consistency",
    )

    for name in condition_names:
        screening[f"pass_{name}"] = (
            screening[f"pass_short_{name}"].fillna(True)
            & screening[f"pass_long_{name}"].fillna(True)
        )

    screening["passed_all_basic_conditions"] = screening[
        [f"pass_{name}" for name in condition_names]
    ].all(axis=1)
    screening = screening.sort_values(
        ["family", "variant", "horizon_days"]
    ).reset_index(drop=True)

    return screening, short_conditions + long_conditions


# -------------------------
# FILTER FUNNEL
def build_screening_funnel(screening):
    conditions = (
        ("complete_sample_and_oos_coverage", "pass_data"),
        ("positive_weighted_sample_and_oos_ic", "pass_weighted_ic"),
        ("positive_median_sample_and_oos_ic", "pass_median_ic"),
        ("positive_sample_and_oos_window_rate", "pass_positive_windows"),
        ("sample_oos_sign_consistency", "pass_sign_consistency"),
    )
    remaining = pd.Series(True, index=screening.index)
    rows = [
        {
            "step": 0,
            "condition": "all_hypotheses",
            "remaining_hypotheses": int(remaining.sum()),
        }
    ]

    for step, (name, column) in enumerate(conditions, start=1):
        remaining &= screening[column]
        rows.append(
            {
                "step": step,
                "condition": name,
                "remaining_hypotheses": int(remaining.sum()),
            }
        )

    return pd.DataFrame(rows)


# -------------------------
# FACTOR SCREENING
def run_factor_screening(robustness_summary):
    screening, _ = build_factor_screening(robustness_summary)
    passed = screening[
        screening["passed_all_basic_conditions"]
    ].copy().reset_index(drop=True)
    funnel = build_screening_funnel(screening)

    print(
        f"Factor screening: {len(passed)} of {len(screening)} "
        "hypotheses passed"
    )

    return screening, passed, funnel


# -------------------------
# INDEPENDENT SCREENING RUN
def main():
    if not os.path.exists(ROBUSTNESS_SUMMARY_PATH):
        raise FileNotFoundError(
            "Robustness summary is missing. Run the Factor Layer pipeline first: "
            f"{ROBUSTNESS_SUMMARY_PATH}"
        )

    robustness_summary = pd.read_csv(ROBUSTNESS_SUMMARY_PATH)
    required_columns = {
        *IDENTITY_COLUMNS,
        "robustness_layer",
        "total_windows",
        "sample_eligible_windows",
        "oos_eligible_windows",
        "weighted_sample_mean_ic",
        "weighted_oos_mean_ic",
        "median_sample_mean_ic",
        "median_oos_mean_ic",
        "positive_sample_window_rate",
        "positive_oos_window_rate",
        "sign_consistency_rate",
    }
    missing_columns = sorted(
        required_columns.difference(robustness_summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Robustness summary is incompatible. Missing columns: "
            + ", ".join(missing_columns)
        )

    screening, passed, funnel = run_factor_screening(robustness_summary)

    from factor_storage import save_csv

    save_csv(screening, FACTOR_SCREENING_PATH)
    save_csv(passed, PASSED_FACTOR_CANDIDATES_PATH)
    save_csv(funnel, FACTOR_SCREENING_FUNNEL_PATH)

    print("Independent factor screening is ready")
    print(f"Passed factor candidates: {len(passed)}")
    print(f"Screening results: {FACTOR_SCREENING_PATH}")


if __name__ == "__main__":
    main()
