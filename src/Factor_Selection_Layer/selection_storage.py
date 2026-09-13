import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from selection_config import (
    FACTOR_MATRIX_DIR,
    FACTOR_METADATA_PATH,
    FORWARD_RETURN_MATRIX_DIR,
    MEMBERSHIP_PATH,
    SELECTION_CACHE_DIR,
    SELECTION_DATA_DIR,
    SELECTION_FIGURES_DIR,
    SELECTION_RESULTS_DIR,
)


# -------------------------
# DIRECTORIES
def prepare_selection_directories():
    for directory in (
        SELECTION_DATA_DIR,
        SELECTION_CACHE_DIR,
        SELECTION_RESULTS_DIR,
        SELECTION_FIGURES_DIR,
    ):
        os.makedirs(directory, exist_ok=True)


# -------------------------
# FILE NAMES
def safe_factor_name(family, variant):
    return f"{family}__{variant}".replace("/", "-").replace(" ", "_")


def factor_matrix_path(family, variant):
    return os.path.join(
        FACTOR_MATRIX_DIR,
        f"{safe_factor_name(family, variant)}.parquet",
    )


def forward_return_matrix_path(horizon):
    return os.path.join(
        FORWARD_RETURN_MATRIX_DIR,
        f"forward_returns_h{int(horizon)}.parquet",
    )


# -------------------------
# FACTOR LAYER INPUTS
def load_factor_metadata():
    metadata = pd.read_csv(FACTOR_METADATA_PATH)
    required = {"key", "family", "variant", "parameters", "path"}
    missing = sorted(required.difference(metadata.columns))

    if missing:
        raise ValueError(
            "Factor metadata is incompatible. Missing columns: "
            + ", ".join(missing)
        )
    if len(metadata) != 56 or metadata["key"].duplicated().any():
        raise ValueError("Factor metadata must contain 56 unique matrices")

    return metadata


def load_membership():
    membership = pd.read_parquet(MEMBERSHIP_PATH).sort_index().astype(bool)

    if not membership.index.is_unique or not membership.index.is_monotonic_increasing:
        raise ValueError("Membership must have unique sorted dates")
    if not membership.columns.is_unique:
        raise ValueError("Membership must have unique ticker columns")

    return membership


def validate_matrix_axes(matrix, reference, name):
    if not matrix.index.equals(reference.index):
        raise ValueError(f"{name} dates do not match membership")
    if not matrix.columns.equals(reference.columns):
        raise ValueError(f"{name} ticker columns do not match membership")


def load_factor_matrix(family, variant, membership):
    path = factor_matrix_path(family, variant)
    factor = pd.read_parquet(path).sort_index()
    validate_matrix_axes(factor, membership, f"Factor {family}|{variant}")
    return factor


def load_forward_return_matrices(horizons, membership):
    matrices = {}

    for horizon in horizons:
        path = forward_return_matrix_path(horizon)
        matrix = pd.read_parquet(path).sort_index()
        validate_matrix_axes(matrix, membership, f"Forward return h{horizon}")
        matrices[int(horizon)] = matrix

    return matrices


# -------------------------
# ATOMIC SAVING
def temporary_path(path):
    root, extension = os.path.splitext(path)
    return f"{root}.temporary{extension}"


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = temporary_path(path)

    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    os.replace(temporary, path)


def save_parquet_chunks(chunks, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = temporary_path(path)

    if os.path.exists(temporary):
        os.remove(temporary)

    writer = None
    row_count = 0
    chunk_count = 0

    try:
        for chunk in chunks:
            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )

            writer.write_table(table)
            row_count += len(chunk)
            chunk_count += 1

        if writer is None:
            raise ValueError("No quantile-result chunks were created")
    except BaseException:
        if writer is not None:
            writer.close()
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    else:
        writer.close()
        os.replace(temporary, path)

    return row_count, chunk_count
