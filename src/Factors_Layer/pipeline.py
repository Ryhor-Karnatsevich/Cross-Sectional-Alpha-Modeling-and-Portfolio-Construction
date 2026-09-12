from datetime import datetime, timezone

from factor_config import (
    APPLY_WINSORIZATION,
    FACTOR_RUN_METADATA_PATH,
    FACTOR_VARIANT_COUNT,
    FORWARD_HORIZONS,
    RESEARCH_END_DATE,
    RESEARCH_START_DATE,
    ROBUSTNESS_CONFIGS,
)
from factor_storage import (
    factor_cache_is_valid,
    load_factor_inputs,
    load_sensitivity_cache,
    prepare_factor_directories,
    save_cache_manifest,
    save_factor_results,
)
from robustness import aggregate_robustness, run_robustness
from sensitivity import run_sensitivity, summarize_sensitivity


# -------------------------
# FACTOR CACHE
def prepare_factor_cache(inputs):
    if factor_cache_is_valid():
        print("Factor cache found -> loading daily IC")

        try:
            daily_ic, metadata = load_sensitivity_cache()
            sensitivity_results = summarize_sensitivity(daily_ic, metadata)
            return sensitivity_results, daily_ic, metadata, True
        except (OSError, ValueError):
            print("Factor cache is unreadable -> rebuilding")

    else:
        print("Factor cache missing or outdated -> rebuilding")

    sensitivity_results, daily_ic, metadata = run_sensitivity(inputs)
    save_cache_manifest()
    return sensitivity_results, daily_ic, metadata, False


# -------------------------
# COMPLETE FACTOR LAYER
def run_pipeline():
    print("Preparing Factor Layer directories...")
    prepare_factor_directories()

    print("Loading Data System matrices...")
    inputs = load_factor_inputs()
    sensitivity_results, daily_ic, metadata, cache_reused = (
        prepare_factor_cache(inputs)
    )

    robustness_results = run_robustness(
        daily_ic,
        metadata,
        inputs["prices"].index,
    )
    robustness_summary = aggregate_robustness(robustness_results)
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_start": inputs["prices"].index.min().date().isoformat(),
        "data_end": inputs["prices"].index.max().date().isoformat(),
        "research_start": RESEARCH_START_DATE,
        "research_end": RESEARCH_END_DATE,
        "trading_dates": len(inputs["prices"]),
        "ticker_columns": len(inputs["prices"].columns),
        "factor_families": metadata["family"].nunique(),
        "factor_variants": FACTOR_VARIANT_COUNT,
        "forward_horizons": list(FORWARD_HORIZONS),
        "hypotheses": len(metadata),
        "winsorization_applied": APPLY_WINSORIZATION,
        "factor_cache_reused": cache_reused,
        "robustness_windows": {
            layer: int(
                robustness_results.loc[
                    robustness_results["robustness_layer"] == layer,
                    "window",
                ].nunique()
            )
            for layer in ROBUSTNESS_CONFIGS
        },
        "robustness_hypotheses": {
            layer: int(
                robustness_results.loc[
                    robustness_results["robustness_layer"] == layer,
                    "key",
                ].nunique()
            )
            for layer in ROBUSTNESS_CONFIGS
        },
        "robustness_rows": len(robustness_results),
        "robustness_summary_rows": len(robustness_summary),
    }

    save_factor_results(
        sensitivity_results,
        robustness_results,
        robustness_summary,
        run_metadata,
    )

    print("Factor Layer is ready")
    print(f"Factor cache reused: {cache_reused}")
    print(f"Factor variants: {FACTOR_VARIANT_COUNT}")
    print(f"Daily IC hypotheses: {len(metadata)}")
    print(f"Robustness rows: {len(robustness_results)}")
    print(f"Run metadata: {FACTOR_RUN_METADATA_PATH}")

    return sensitivity_results, robustness_results, robustness_summary


if __name__ == "__main__":
    run_pipeline()
