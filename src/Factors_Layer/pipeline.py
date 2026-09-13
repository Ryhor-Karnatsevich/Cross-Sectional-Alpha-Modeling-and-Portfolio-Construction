from datetime import datetime, timezone

from factor_builder import build_factor_matrices
from factor_config import (
    APPLY_WINSORIZATION,
    FACTOR_RUN_METADATA_PATH,
    FACTOR_VARIANT_COUNT,
    FORWARD_HORIZONS,
)
from factor_storage import (
    factor_cache_is_valid,
    load_factor_inputs,
    load_factor_metadata,
    prepare_factor_directories,
    save_cache_manifest,
    save_factor_metadata,
    save_run_metadata,
)
from forward_returns import build_forward_return_matrices


# -------------------------
# FACTOR CACHE
def prepare_factor_cache(inputs):
    if factor_cache_is_valid():
        print("Factor cache found -> reusing factor and forward-return matrices")
        return load_factor_metadata(), True

    print("Factor cache missing or outdated -> rebuilding")
    metadata = build_factor_matrices(inputs)
    build_forward_return_matrices(inputs)
    save_factor_metadata(metadata)
    save_cache_manifest()
    return metadata, False


# -------------------------
# COMPLETE FACTOR LAYER
def run_pipeline():
    print("Preparing Factor Layer directories...")
    prepare_factor_directories()

    print("Loading Data System matrices...")
    inputs = load_factor_inputs()
    metadata, cache_reused = prepare_factor_cache(inputs)
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_start": inputs["prices"].index.min().date().isoformat(),
        "data_end": inputs["prices"].index.max().date().isoformat(),
        "trading_dates": len(inputs["prices"]),
        "ticker_columns": len(inputs["prices"].columns),
        "factor_families": metadata["family"].nunique(),
        "factor_variants": FACTOR_VARIANT_COUNT,
        "forward_horizons": list(FORWARD_HORIZONS),
        "winsorization_applied": APPLY_WINSORIZATION,
        "factor_cache_reused": cache_reused,
    }
    save_run_metadata(run_metadata)

    print("Factor Layer is ready")
    print(f"Factor cache reused: {cache_reused}")
    print(f"Factor-score matrices: {FACTOR_VARIANT_COUNT}")
    print(f"Forward-return matrices: {len(FORWARD_HORIZONS)}")
    print(f"Run metadata: {FACTOR_RUN_METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    run_pipeline()
