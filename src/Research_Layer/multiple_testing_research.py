import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from research_config import (
    MULTIPLE_TESTING_ALPHA,
    MULTIPLE_TESTING_GROUP_COLUMNS,
)


def bh_values(p_values, alpha=MULTIPLE_TESTING_ALPHA):
    p_values = pd.Series(p_values, dtype=float)
    result = pd.DataFrame(
        {"q_value": np.nan, "rejected": False},
        index=p_values.index,
    )
    valid = p_values.notna()
    if valid.any():
        rejected, q_values, _, _ = multipletests(
            p_values.loc[valid],
            alpha=alpha,
            method="fdr_bh",
        )
        result.loc[valid, "q_value"] = q_values
        result.loc[valid, "rejected"] = rejected
    return result


def simes_p_value(p_values):
    valid = np.sort(pd.Series(p_values, dtype=float).dropna().to_numpy())
    if not len(valid):
        return np.nan
    ranks = np.arange(1, len(valid) + 1)
    return float(np.minimum(1.0, np.min(valid * len(valid) / ranks)))


def add_group_bh(frame, group_columns, prefix):
    frame = frame.copy()
    frame[f"{prefix}_q_value"] = np.nan
    frame[f"{prefix}_rejected"] = False

    for _, group in frame.groupby(list(group_columns), dropna=False):
        adjusted = bh_values(group["raw_p_value"])
        frame.loc[group.index, f"{prefix}_q_value"] = adjusted["q_value"]
        frame.loc[group.index, f"{prefix}_rejected"] = adjusted[
            "rejected"
        ].astype(bool)
    return frame


def add_hierarchical_fdr(frame):
    frame = frame.copy()
    group_columns = list(MULTIPLE_TESTING_GROUP_COLUMNS)
    group_rows = []

    for key, group in frame.groupby(group_columns, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key))
        row["group_simes_p_value"] = simes_p_value(group["raw_p_value"])
        row["group_test_count"] = len(group)
        group_rows.append(row)

    groups = pd.DataFrame(group_rows)
    group_bh = bh_values(groups["group_simes_p_value"])
    groups["hierarchical_group_q_value"] = group_bh["q_value"]
    groups["hierarchical_group_selected"] = group_bh["rejected"].astype(bool)
    selected_count = int(groups["hierarchical_group_selected"].sum())
    group_count = len(groups)
    within_alpha = (
        MULTIPLE_TESTING_ALPHA * selected_count / group_count
        if group_count
        else 0.0
    )
    groups["hierarchical_within_alpha"] = within_alpha

    frame = frame.merge(
        groups,
        on=group_columns,
        how="left",
        validate="many_to_one",
    )
    frame["hierarchical_within_q_value"] = np.nan
    frame["hierarchical_rejected"] = False

    selected_groups = groups.loc[groups["hierarchical_group_selected"]]
    for row in selected_groups.itertuples(index=False):
        mask = pd.Series(True, index=frame.index)
        for column in group_columns:
            mask &= frame[column].eq(getattr(row, column))
        adjusted = bh_values(
            frame.loc[mask, "raw_p_value"],
            alpha=within_alpha,
        )
        frame.loc[mask, "hierarchical_within_q_value"] = adjusted["q_value"]
        frame.loc[mask, "hierarchical_rejected"] = adjusted[
            "rejected"
        ].astype(bool)
    return frame


def compare_multiple_testing(effect_tests):
    required = {
        "hypothesis_key",
        "factor_key",
        "family",
        "effect",
        "raw_p_value",
    }
    missing = required.difference(effect_tests.columns)
    if missing:
        raise ValueError(
            "Selection effect tests miss columns: "
            + ", ".join(sorted(missing))
        )

    result = effect_tests.copy()
    global_bh = bh_values(result["raw_p_value"])
    result["research_global_bh_q_value"] = global_bh["q_value"]
    result["research_global_rejected"] = global_bh["rejected"].astype(bool)
    result = add_group_bh(result, ("effect",), "within_effect_bh")
    result = add_group_bh(result, ("family",), "within_family_bh")
    result = add_group_bh(
        result,
        MULTIPLE_TESTING_GROUP_COLUMNS,
        "within_family_effect_bh",
    )
    result = add_hierarchical_fdr(result)

    summaries = []
    methods = {
        "raw_p_below_0_05": result["raw_p_value"]
        <= MULTIPLE_TESTING_ALPHA,
        "global_bh": result["research_global_rejected"],
        "within_effect_bh_exploratory": result["within_effect_bh_rejected"],
        "within_family_bh_exploratory": result["within_family_bh_rejected"],
        "within_family_effect_bh_exploratory": result[
            "within_family_effect_bh_rejected"
        ],
        "hierarchical_fdr": result["hierarchical_rejected"],
    }
    for method, rejected in methods.items():
        rejected = rejected.fillna(False).astype(bool)
        summaries.append(
            {
                "method": method,
                "tests": len(result),
                "rejections": int(rejected.sum()),
                "economic_rejections": int(
                    (rejected & result["effect"].ne("spearman_ic")).sum()
                ),
                "ic_rejections": int(
                    (rejected & result["effect"].eq("spearman_ic")).sum()
                ),
                "interpretation": (
                    "confirmatory"
                    if method == "global_bh"
                    else (
                        "structured_sensitivity"
                        if method == "hierarchical_fdr"
                        else "exploratory"
                    )
                ),
            }
        )
    return result, pd.DataFrame(summaries)
