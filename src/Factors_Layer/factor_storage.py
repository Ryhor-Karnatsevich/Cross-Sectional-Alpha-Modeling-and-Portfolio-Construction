import json
import os
import sys

import pandas as pd

from factor_config import (
    DAILY_IC_CACHE_PATH,
    FACTOR_CACHE_DIR,
    FACTOR_DATA_DIR,
    FACTOR_FIGURES_DIR,
    FACTOR_MATRIX_CACHE_DIR,
    FACTOR_METADATA_CACHE_PATH,
    FACTOR_RESULTS_DIR,
    FACTOR_RUN_METADATA_PATH,
    ROBUSTNESS_RESULTS_PATH,
    ROBUSTNESS_SUMMARY_PATH,
    SELECTED_FACTOR_CONFIGS_PATH,
    SELECTED_FACTOR_RANKS_DIR,
    SELECTED_FACTOR_SCORES_DIR,
    SENSITIVITY_RESULTS_PATH,
    SENSITIVITY_SUMMARY_PATH,
)


data_system_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data_System")
)
if data_system_path not in sys.path:
    sys.path.insert(0, data_system_path)

from config import (
    AVAILABILITY_PATH,
    MEMBERSHIP_PATH,
    QUALITY_PATH,
    RAW_PRICES_PATH,
    RETURNS_PATH,
    VOLUME_PATH,
    VOLUME_QUALITY_PATH,
)


# -------------------------
# DIRECTORIES
def prepare_factor_directories():
    directories = (
        FACTOR_DATA_DIR,
        FACTOR_CACHE_DIR,
        FACTOR_MATRIX_CACHE_DIR,
        SELECTED_FACTOR_SCORES_DIR,
        SELECTED_FACTOR_RANKS_DIR,
        FACTOR_RESULTS_DIR,
        FACTOR_FIGURES_DIR,
    )

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# -------------------------
# INPUT DATA
def load_factor_inputs():
    prices = pd.read_parquet(RAW_PRICES_PATH).sort_index()
    reference_index = prices.index
    reference_columns = prices.columns

    returns = pd.read_parquet(RETURNS_PATH).reindex(
        index=reference_index,
        columns=reference_columns,
    )
    volume = pd.read_parquet(VOLUME_PATH).reindex(
        index=reference_index,
        columns=reference_columns,
    )
    availability = pd.read_parquet(AVAILABILITY_PATH).reindex(
        index=reference_index,
        columns=reference_columns,
        fill_value=False,
    )
    membership = pd.read_parquet(MEMBERSHIP_PATH).reindex(
        index=reference_index,
        columns=reference_columns,
        fill_value=False,
    )
    price_quality = pd.read_parquet(QUALITY_PATH).reindex(
        index=reference_index,
        columns=reference_columns,
        fill_value=False,
    )
    volume_quality = pd.read_parquet(VOLUME_QUALITY_PATH).reindex(
        index=reference_index,
        columns=reference_columns,
        fill_value=False,
    )

    availability = availability.astype(bool)
    membership = membership.astype(bool)
    price_quality = price_quality.astype(bool)
    volume_quality = volume_quality.astype(bool)
    volume = volume.where(volume_quality)

    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must have unique sorted dates")
    if not prices.columns.is_unique:
        raise ValueError("Prices must have unique ticker columns")
    if not availability.equals(prices.notna() & membership & price_quality):
        raise ValueError("Availability does not match Data System inputs")

    return {
        "prices": prices,
        "returns": returns,
        "volume": volume,
        "availability": availability,
        "membership": membership,
        "price_quality": price_quality,
    }


# -------------------------
# FILE NAMES
def safe_factor_name(family, variant):
    return f"{family}__{variant}".replace("/", "-").replace(" ", "_")


def factor_matrix_cache_path(family, variant):
    filename = f"{safe_factor_name(family, variant)}.parquet"
    return os.path.join(FACTOR_MATRIX_CACHE_DIR, filename)


def selected_matrix_path(directory, robustness_layer, family):
    filename = f"{robustness_layer}__{family}.parquet"
    return os.path.join(directory, filename)


# -------------------------
# ATOMIC SAVING
def temporary_path(path):
    root, extension = os.path.splitext(path)
    return f"{root}.temporary{extension}"


def save_parquet(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = temporary_path(path)
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def save_csv(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = temporary_path(path)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = temporary_path(path)

    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    os.replace(temporary, path)


# -------------------------
# FACTOR CACHE
def save_factor_matrix(family, variant, factor):
    path = factor_matrix_cache_path(family, variant)
    save_parquet(factor.astype("float32"), path)
    return path


def load_factor_matrix(family, variant):
    return pd.read_parquet(factor_matrix_cache_path(family, variant))


def save_sensitivity_cache(daily_ic, metadata):
    save_parquet(daily_ic.astype("float32"), DAILY_IC_CACHE_PATH)
    save_csv(metadata, FACTOR_METADATA_CACHE_PATH)


# -------------------------
# FINAL OUTPUTS
def save_factor_results(
    sensitivity_results,
    robustness_results,
    selected_configs,
    run_metadata,
):
    save_parquet(sensitivity_results, SENSITIVITY_RESULTS_PATH)
    save_parquet(robustness_results, ROBUSTNESS_RESULTS_PATH)
    save_csv(selected_configs, SELECTED_FACTOR_CONFIGS_PATH)
    save_csv(sensitivity_results, SENSITIVITY_SUMMARY_PATH)
    save_csv(
        robustness_results[robustness_results["selected"]].copy(),
        ROBUSTNESS_SUMMARY_PATH,
    )
    save_json(run_metadata, FACTOR_RUN_METADATA_PATH)


def save_selected_factor_matrices(
    robustness_layer,
    family,
    scores,
    ranks,
):
    score_path = selected_matrix_path(
        SELECTED_FACTOR_SCORES_DIR,
        robustness_layer,
        family,
    )
    rank_path = selected_matrix_path(
        SELECTED_FACTOR_RANKS_DIR,
        robustness_layer,
        family,
    )
    save_parquet(scores.astype("float32"), score_path)
    save_parquet(ranks.astype("float32"), rank_path)
    return score_path, rank_path
