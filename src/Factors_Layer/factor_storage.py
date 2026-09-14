import hashlib
import json
import os
import sys

import pandas as pd

from factor_config import (
    ANNUALIZATION_FACTOR,
    APPLY_WINSORIZATION,
    FACTOR_CACHE_DIR,
    FACTOR_CACHE_MANIFEST_PATH,
    FACTOR_CONFIGS,
    FACTOR_DATA_DIR,
    FACTOR_MATRIX_CACHE_DIR,
    FACTOR_METADATA_CACHE_PATH,
    FACTOR_RESULTS_DIR,
    FACTOR_RUN_METADATA_PATH,
    FORWARD_HORIZONS,
    FORWARD_RETURN_MATRIX_CACHE_DIR,
    MIN_OBSERVATION_RATIO,
    WINSOR_LOWER,
    WINSOR_UPPER,
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


INPUT_PATHS = (
    RAW_PRICES_PATH,
    RETURNS_PATH,
    VOLUME_PATH,
    AVAILABILITY_PATH,
    MEMBERSHIP_PATH,
    QUALITY_PATH,
    VOLUME_QUALITY_PATH,
)


# -------------------------
# DIRECTORIES
def prepare_factor_directories():
    directories = (
        FACTOR_DATA_DIR,
        FACTOR_CACHE_DIR,
        FACTOR_MATRIX_CACHE_DIR,
        FORWARD_RETURN_MATRIX_CACHE_DIR,
        FACTOR_RESULTS_DIR,
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
        "price_quality": price_quality,
    }


# -------------------------
# FILE NAMES
def safe_factor_name(family, variant):
    return f"{family}__{variant}".replace("/", "-").replace(" ", "_")


def factor_matrix_cache_path(family, variant):
    filename = f"{safe_factor_name(family, variant)}.parquet"
    return os.path.join(FACTOR_MATRIX_CACHE_DIR, filename)


def forward_return_matrix_cache_path(horizon):
    return os.path.join(
        FORWARD_RETURN_MATRIX_CACHE_DIR,
        f"forward_returns_h{int(horizon)}.parquet",
    )


def expected_factor_matrix_paths():
    return tuple(
        factor_matrix_cache_path(family, configuration["variant"])
        for family, configurations in FACTOR_CONFIGS.items()
        for configuration in configurations
    )


def expected_forward_return_matrix_paths():
    return tuple(
        forward_return_matrix_cache_path(horizon)
        for horizon in FORWARD_HORIZONS
    )


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
# CACHE FINGERPRINT
def file_state(path):
    state = os.stat(path)
    return {
        "path": os.path.relpath(path, start=os.path.dirname(FACTOR_DATA_DIR)),
        "size": state.st_size,
        "modified_ns": state.st_mtime_ns,
    }


def source_hash(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def cache_signature_payload():
    source_directory = os.path.dirname(__file__)
    source_files = (
        os.path.join(source_directory, "factors.py"),
        os.path.join(source_directory, "transforms.py"),
        os.path.join(source_directory, "factor_builder.py"),
        os.path.join(source_directory, "forward_returns.py"),
    )
    configuration = {
        "factor_configs": FACTOR_CONFIGS,
        "forward_horizons": FORWARD_HORIZONS,
        "min_observation_ratio": MIN_OBSERVATION_RATIO,
        "apply_winsorization": APPLY_WINSORIZATION,
        "winsor_lower": WINSOR_LOWER,
        "winsor_upper": WINSOR_UPPER,
        "annualization_factor": ANNUALIZATION_FACTOR,
    }

    return {
        "input_files": [file_state(path) for path in INPUT_PATHS],
        "source_hashes": {
            os.path.basename(path): source_hash(path)
            for path in source_files
        },
        "configuration": configuration,
    }


def cache_signature(payload):
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_cache_manifest():
    payload = cache_signature_payload()
    return {
        "signature": cache_signature(payload),
        "payload": payload,
    }


def save_cache_manifest():
    manifest = current_cache_manifest()
    save_json(manifest, FACTOR_CACHE_MANIFEST_PATH)
    return manifest


def factor_metadata_is_valid():
    try:
        metadata = pd.read_csv(FACTOR_METADATA_CACHE_PATH)
    except (OSError, ValueError):
        return False

    required_columns = {"key", "family", "variant", "parameters", "path"}
    return (
        required_columns.issubset(metadata.columns)
        and len(metadata) == 56
        and not metadata["key"].duplicated().any()
    )


def factor_cache_is_valid():
    required_paths = (
        FACTOR_METADATA_CACHE_PATH,
        FACTOR_CACHE_MANIFEST_PATH,
        *expected_factor_matrix_paths(),
        *expected_forward_return_matrix_paths(),
    )

    if not all(os.path.exists(path) for path in required_paths):
        return False
    if not factor_metadata_is_valid():
        return False

    try:
        with open(FACTOR_CACHE_MANIFEST_PATH, "r", encoding="utf-8") as file:
            saved_manifest = json.load(file)

        return saved_manifest.get("signature") == current_cache_manifest()[
            "signature"
        ]
    except (OSError, ValueError, TypeError):
        return False


# -------------------------
# FACTOR CACHE
def save_factor_matrix(family, variant, factor):
    path = factor_matrix_cache_path(family, variant)
    save_parquet(factor.astype("float32"), path)
    return path


def save_forward_return_matrix(horizon, forward_returns):
    path = forward_return_matrix_cache_path(horizon)
    save_parquet(forward_returns.astype("float32"), path)
    return path


def save_factor_metadata(metadata):
    save_csv(metadata, FACTOR_METADATA_CACHE_PATH)


def load_factor_metadata():
    metadata = pd.read_csv(FACTOR_METADATA_CACHE_PATH)

    if len(metadata) != 56 or metadata["key"].duplicated().any():
        raise ValueError("Factor metadata must contain 56 unique matrices")

    return metadata


# -------------------------
# RUN METADATA
def save_run_metadata(run_metadata):
    save_json(run_metadata, FACTOR_RUN_METADATA_PATH)
