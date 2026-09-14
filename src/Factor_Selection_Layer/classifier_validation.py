import pandas as pd

from pattern_classification import classify_pattern
from selection_config import CLASSIFIER_VALIDATION_PATH
from selection_storage import save_csv


# -------------------------
# KNOWN PATTERN EXAMPLES
def validation_cases():
    return {
        "positive_monotonic": {
            "quantile_curve_rho": 0.95,
            "quantile_step_ratio": 0.80,
            "q10_minus_q1_mean": 0.020,
            "q10_minus_q1_hac_tstat": 3.00,
            "q10_minus_middle_mean": 0.012,
            "q10_minus_middle_hac_tstat": 2.50,
            "middle_minus_q1_mean": 0.008,
            "middle_minus_q1_hac_tstat": 2.10,
        },
        "negative_monotonic": {
            "quantile_curve_rho": -0.95,
            "quantile_step_ratio": 0.80,
            "q10_minus_q1_mean": -0.020,
            "q10_minus_q1_hac_tstat": -3.00,
            "q10_minus_middle_mean": -0.012,
            "q10_minus_middle_hac_tstat": -2.50,
            "middle_minus_q1_mean": -0.008,
            "middle_minus_q1_hac_tstat": -2.10,
        },
        "upper_tail": {
            "quantile_curve_rho": 0.45,
            "quantile_step_ratio": 0.55,
            "q10_minus_q1_mean": 0.014,
            "q10_minus_q1_hac_tstat": 2.20,
            "q10_minus_middle_mean": 0.015,
            "q10_minus_middle_hac_tstat": 3.00,
            "middle_minus_q1_mean": -0.001,
            "middle_minus_q1_hac_tstat": -0.20,
        },
        "lower_tail": {
            "quantile_curve_rho": 0.40,
            "quantile_step_ratio": 0.50,
            "q10_minus_q1_mean": 0.014,
            "q10_minus_q1_hac_tstat": 2.20,
            "q10_minus_middle_mean": -0.001,
            "q10_minus_middle_hac_tstat": -0.20,
            "middle_minus_q1_mean": 0.015,
            "middle_minus_q1_hac_tstat": 3.00,
        },
        "both_tails_vs_middle": {
            "quantile_curve_rho": 0.05,
            "quantile_step_ratio": 0.45,
            "q10_minus_q1_mean": 0.001,
            "q10_minus_q1_hac_tstat": 0.10,
            "q10_minus_middle_mean": 0.015,
            "q10_minus_middle_hac_tstat": 3.00,
            "middle_minus_q1_mean": -0.014,
            "middle_minus_q1_hac_tstat": -2.80,
        },
        "no_stable_structure": {
            "quantile_curve_rho": 0.10,
            "quantile_step_ratio": 0.40,
            "q10_minus_q1_mean": 0.001,
            "q10_minus_q1_hac_tstat": 0.20,
            "q10_minus_middle_mean": 0.001,
            "q10_minus_middle_hac_tstat": 0.30,
            "middle_minus_q1_mean": 0.000,
            "middle_minus_q1_hac_tstat": 0.00,
        },
    }


# -------------------------
# CLASSIFIER SELF-CHECK
def run_classifier_validation():
    rows = []

    for expected_pattern, metrics in validation_cases().items():
        actual_pattern, direction, primary_effect = classify_pattern(
            pd.Series(metrics)
        )
        rows.append(
            {
                "expected_pattern": expected_pattern,
                "actual_pattern": actual_pattern,
                "pattern_direction": direction,
                "primary_effect": primary_effect,
                "passed": actual_pattern == expected_pattern,
            }
        )

    validation = pd.DataFrame(rows)
    save_csv(validation, CLASSIFIER_VALIDATION_PATH)

    if not validation["passed"].all():
        failed = validation.loc[
            ~validation["passed"],
            ["expected_pattern", "actual_pattern"],
        ]
        raise ValueError(
            "Pattern classifier validation failed:\n"
            + failed.to_string(index=False)
        )

    return validation


if __name__ == "__main__":
    result = run_classifier_validation()
    print(result.to_string(index=False))
