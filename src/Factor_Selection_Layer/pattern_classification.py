import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from selection_config import (
    EFFECT_NAMES,
    FORWARD_HORIZONS,
    MIN_MONTHLY_DIRECTION_RATE,
    MIN_VALID_DATE_RATIO,
    MIN_YEARLY_DIRECTION_RATE,
    MONOTONIC_ABSOLUTE_RHO,
    MONOTONIC_STEP_RATIO,
    MULTIPLE_TESTING_ALPHA,
    PATTERN_ABSOLUTE_TSTAT,
    TAIL_DOMINANCE_RATIO,
)


# -------------------------
# MULTIPLE TESTING
def apply_multiple_testing(effect_tests, factor_count):
    effect_tests = effect_tests.copy()
    valid = effect_tests["raw_p_value"].notna()
    effect_tests["bh_q_value_active_scope"] = np.nan
    effect_tests["reject_active_scope_fdr"] = False

    if valid.any():
        rejected, q_values, _, _ = multipletests(
            effect_tests.loc[valid, "raw_p_value"],
            alpha=MULTIPLE_TESTING_ALPHA,
            method="fdr_bh",
        )
        effect_tests.loc[valid, "bh_q_value_active_scope"] = q_values
        effect_tests.loc[valid, "reject_active_scope_fdr"] = rejected

    expected_hypotheses = 56 * len(FORWARD_HORIZONS)
    expected_effect_tests = expected_hypotheses * len(EFFECT_NAMES)
    active_hypotheses = effect_tests["hypothesis_key"].nunique()
    full_scope = (
        factor_count == 56
        and active_hypotheses == expected_hypotheses
        and len(effect_tests) == expected_effect_tests
    )
    effect_tests["active_hypothesis_count"] = active_hypotheses
    effect_tests["active_effect_test_count"] = len(effect_tests)
    effect_tests["full_research_scope"] = full_scope
    effect_tests["planned_bonferroni_p_value"] = np.minimum(
        effect_tests["raw_p_value"] * expected_effect_tests,
        1.0,
    )
    return effect_tests


# -------------------------
# PATTERN RULES
def statistically_visible(tstat):
    return pd.notna(tstat) and abs(tstat) >= PATTERN_ABSOLUTE_TSTAT


def classify_pattern(row):
    rho = row["quantile_curve_rho"]
    step_ratio = row["quantile_step_ratio"]
    spread_mean = row["q10_minus_q1_mean"]
    spread_tstat = row["q10_minus_q1_hac_tstat"]
    upper_mean = row["q10_minus_middle_mean"]
    upper_tstat = row["q10_minus_middle_hac_tstat"]
    lower_mean = row["middle_minus_q1_mean"]
    lower_tstat = row["middle_minus_q1_hac_tstat"]
    upper_visible = statistically_visible(upper_tstat)
    lower_visible = statistically_visible(lower_tstat)
    spread_visible = statistically_visible(spread_tstat)
    monotonic = (
        pd.notna(rho)
        and pd.notna(step_ratio)
        and abs(rho) >= MONOTONIC_ABSOLUTE_RHO
        and step_ratio >= MONOTONIC_STEP_RATIO
        and spread_visible
        and np.sign(rho) == np.sign(spread_mean)
    )

    if monotonic and rho > 0:
        return "positive_monotonic", "positive", "q10_minus_q1"
    if monotonic and rho < 0:
        return "negative_monotonic", "negative", "q10_minus_q1"

    opposite_tail_directions = (
        upper_visible
        and lower_visible
        and np.sign(upper_mean) == -np.sign(lower_mean)
    )
    if opposite_tail_directions:
        direction = (
            "tails_above_middle" if upper_mean > 0 else "tails_below_middle"
        )
        return "both_tails_vs_middle", direction, "edges_minus_middle"

    upper_dominates = (
        upper_visible
        and (
            not lower_visible
            or abs(upper_mean) >= TAIL_DOMINANCE_RATIO * abs(lower_mean)
        )
    )
    if upper_dominates:
        direction = "q10_above_middle" if upper_mean > 0 else "q10_below_middle"
        return "upper_tail", direction, "q10_minus_middle"

    lower_dominates = (
        lower_visible
        and (
            not upper_visible
            or abs(lower_mean) >= TAIL_DOMINANCE_RATIO * abs(upper_mean)
        )
    )
    if lower_dominates:
        direction = "q1_below_middle" if lower_mean > 0 else "q1_above_middle"
        return "lower_tail", direction, "middle_minus_q1"

    return "no_stable_structure", "none", "none"


def strongest_economic_effect(row):
    effect_names = (
        "q10_minus_q1",
        "q10_minus_middle",
        "middle_minus_q1",
        "edges_minus_middle",
    )
    available = {
        effect: row[f"{effect}_hac_tstat"]
        for effect in effect_names
        if pd.notna(row[f"{effect}_hac_tstat"])
    }

    if not available:
        return pd.Series(
            {
                "strongest_economic_effect": "none",
                "strongest_economic_effect_mean": np.nan,
                "strongest_economic_effect_hac_tstat": np.nan,
                "strongest_economic_effect_abs_tstat": np.nan,
                "strongest_economic_effect_monthly_direction_rate": np.nan,
                "strongest_economic_effect_annual_direction_rate": np.nan,
            }
        )

    effect = max(available, key=lambda name: abs(available[name]))
    return pd.Series(
        {
            "strongest_economic_effect": effect,
            "strongest_economic_effect_mean": row[f"{effect}_mean"],
            "strongest_economic_effect_hac_tstat": row[
                f"{effect}_hac_tstat"
            ],
            "strongest_economic_effect_abs_tstat": abs(
                row[f"{effect}_hac_tstat"]
            ),
            "strongest_economic_effect_monthly_direction_rate": row[
                f"{effect}_monthly_direction_rate"
            ],
            "strongest_economic_effect_annual_direction_rate": row[
                f"{effect}_annual_direction_rate"
            ],
        }
    )


def primary_effect_values(row, effect_lookup):
    hypothesis_effects = effect_lookup.loc[row["hypothesis_key"]]

    if row["primary_effect"] == "none":
        scope = hypothesis_effects.iloc[0]
        return pd.Series(
            {
                "primary_effect_mean": np.nan,
                "primary_effect_median": np.nan,
                "primary_effect_hac_tstat": np.nan,
                "primary_effect_raw_p_value": np.nan,
                "primary_effect_bh_q_active_scope": np.nan,
                "primary_effect_planned_bonferroni_p": np.nan,
                "primary_effect_monthly_direction_rate": np.nan,
                "primary_effect_annual_direction_rate": np.nan,
                "primary_effect_valid_date_ratio": row["valid_ic_ratio"],
                "full_research_scope": bool(scope["full_research_scope"]),
                "active_hypothesis_count": int(
                    scope["active_hypothesis_count"]
                ),
                "active_effect_test_count": int(
                    scope["active_effect_test_count"]
                ),
            }
        )

    effect = hypothesis_effects.loc[row["primary_effect"]]
    return pd.Series(
        {
            "primary_effect_mean": effect["mean"],
            "primary_effect_median": effect["median"],
            "primary_effect_hac_tstat": effect["hac_tstat"],
            "primary_effect_raw_p_value": effect["raw_p_value"],
            "primary_effect_bh_q_active_scope": effect[
                "bh_q_value_active_scope"
            ],
            "primary_effect_planned_bonferroni_p": effect[
                "planned_bonferroni_p_value"
            ],
            "primary_effect_monthly_direction_rate": effect[
                "monthly_direction_rate"
            ],
            "primary_effect_annual_direction_rate": effect[
                "annual_direction_rate"
            ],
            "primary_effect_valid_date_ratio": effect["valid_date_ratio"],
            "full_research_scope": bool(effect["full_research_scope"]),
            "active_hypothesis_count": int(
                effect["active_hypothesis_count"]
            ),
            "active_effect_test_count": int(
                effect["active_effect_test_count"]
            ),
        }
    )


# -------------------------
# EVIDENCE STATUS
def evidence_status(row):
    if row["valid_ic_ratio"] < MIN_VALID_DATE_RATIO:
        return "insufficient_data"
    if row["pattern"] == "no_stable_structure":
        return "no_stable_structure"
    if abs(row["primary_effect_hac_tstat"]) < PATTERN_ABSOLUTE_TSTAT:
        return "weak_pattern"

    stable_time = (
        row["primary_effect_monthly_direction_rate"]
        >= MIN_MONTHLY_DIRECTION_RATE
        and row["primary_effect_annual_direction_rate"]
        >= MIN_YEARLY_DIRECTION_RATE
    )
    if not stable_time:
        return "time_unstable_pattern"

    if not row["full_research_scope"]:
        return "provisional_candidate"
    if row["primary_effect_bh_q_active_scope"] <= MULTIPLE_TESTING_ALPHA:
        return "candidate_after_full_fdr"
    return "rejected_by_multiple_testing"


def classification_reason(row):
    pattern = row["pattern"]

    if pattern == "positive_monotonic":
        return "Higher factor deciles generally have higher future returns."
    if pattern == "negative_monotonic":
        return "Higher factor deciles generally have lower future returns."
    if pattern == "upper_tail":
        return "Q10 differs from the middle more clearly than Q1 does."
    if pattern == "lower_tail":
        return "Q1 differs from the middle more clearly than Q10 does."
    if pattern == "both_tails_vs_middle":
        return "Both extreme deciles differ from the middle in the same way."
    return "No stable monotonic or tail structure passed the descriptive rules."


# -------------------------
# COMPLETE CLASSIFICATION
def classify_hypotheses(cards, effect_tests, factor_count):
    effect_tests = apply_multiple_testing(effect_tests, factor_count)
    cards = cards.copy()
    classifications = cards.apply(classify_pattern, axis=1)
    cards[["pattern", "pattern_direction", "primary_effect"]] = pd.DataFrame(
        classifications.tolist(),
        index=cards.index,
    )
    strongest_effects = cards.apply(strongest_economic_effect, axis=1)
    cards = pd.concat([cards, strongest_effects], axis=1)
    effect_lookup = effect_tests.set_index(
        ["hypothesis_key", "effect"],
        verify_integrity=True,
    )
    primary = cards.apply(
        primary_effect_values,
        axis=1,
        effect_lookup=effect_lookup,
    )
    cards = pd.concat([cards, primary], axis=1)
    cards["evidence_status"] = cards.apply(evidence_status, axis=1)
    cards["classification_reason"] = cards.apply(
        classification_reason,
        axis=1,
    )
    cards["evidence_score"] = (
        (cards["pattern"] != "no_stable_structure").astype(int) * 100
        + cards["primary_effect_hac_tstat"].abs().fillna(0).clip(upper=10)
        + cards["primary_effect_monthly_direction_rate"].fillna(0)
        + cards["primary_effect_annual_direction_rate"].fillna(0)
        + cards["strongest_economic_effect_abs_tstat"].fillna(0) / 100
    )
    cards = cards.sort_values(
        ["evidence_score", "factor_key", "horizon_days"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return cards, effect_tests
