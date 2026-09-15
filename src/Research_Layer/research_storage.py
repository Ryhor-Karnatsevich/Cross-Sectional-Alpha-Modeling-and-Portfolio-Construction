import hashlib
import json
import os
import shutil

import pandas as pd

from research_config import (
    PORTFOLIO_BETA_PATH,
    PORTFOLIO_GROSS_EXPOSURE_PATH,
    PORTFOLIO_METADATA_PATH,
    PORTFOLIO_MAX_SECTOR_EXPOSURE_PATH,
    PORTFOLIO_MAX_POSITION_PATH,
    PORTFOLIO_HOLDING_COUNT_PATH,
    PORTFOLIO_NET_EXPOSURE_PATH,
    PORTFOLIO_PATHS_PATH,
    PORTFOLIO_TURNOVER_PATH,
    PORTFOLIO_UNPRICED_WEIGHT_PATH,
    PORTFOLIO_UNKNOWN_SECTOR_WEIGHT_PATH,
    PHASE_STABILITY_PATH,
    RESEARCH_CACHE_DIR,
    RESEARCH_CACHE_MANIFEST_PATH,
    RESEARCH_CANDIDATES,
    RESEARCH_FIGURES_DIR,
    PROJECT_ROOT,
    RESEARCH_RESULTS_DIR,
)


PORTFOLIO_CACHE_PATHS = {
    "gross_returns": PORTFOLIO_PATHS_PATH,
    "turnover": PORTFOLIO_TURNOVER_PATH,
    "beta_exposure": PORTFOLIO_BETA_PATH,
    "gross_exposure": PORTFOLIO_GROSS_EXPOSURE_PATH,
    "net_exposure": PORTFOLIO_NET_EXPOSURE_PATH,
    "unpriced_weight": PORTFOLIO_UNPRICED_WEIGHT_PATH,
    "max_sector_exposure": PORTFOLIO_MAX_SECTOR_EXPOSURE_PATH,
    "unknown_sector_weight": PORTFOLIO_UNKNOWN_SECTOR_WEIGHT_PATH,
    "max_position": PORTFOLIO_MAX_POSITION_PATH,
    "holding_count": PORTFOLIO_HOLDING_COUNT_PATH,
}


def prepare_research_directories():
    os.makedirs(RESEARCH_CACHE_DIR, exist_ok=True)
    os.makedirs(RESEARCH_RESULTS_DIR, exist_ok=True)
    os.makedirs(RESEARCH_FIGURES_DIR, exist_ok=True)


def save_csv(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)


def save_parquet(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_parquet(path)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def save_text(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def input_signature(paths, settings):
    files = {}
    for path in paths:
        if not os.path.exists(path):
            files[path] = None
            continue
        stat = os.stat(path)
        files[path] = {
            "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }

    payload = {
        "files": files,
        "settings": settings,
        "candidates": list(RESEARCH_CANDIDATES),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def load_cache_manifest():
    if not os.path.exists(RESEARCH_CACHE_MANIFEST_PATH):
        return None
    with open(RESEARCH_CACHE_MANIFEST_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def portfolio_cache_is_valid(signature):
    manifest = load_cache_manifest()
    if manifest is None or manifest.get("signature") != signature:
        return False
    if not os.path.exists(PORTFOLIO_METADATA_PATH):
        return False
    if not os.path.exists(PHASE_STABILITY_PATH):
        return False
    return all(os.path.exists(path) for path in PORTFOLIO_CACHE_PATHS.values())


def save_portfolio_cache(matrices, metadata, signature, signature_payload):
    for name, path in PORTFOLIO_CACHE_PATHS.items():
        save_parquet(matrices[name].astype("float32"), path)
    save_csv(metadata, PORTFOLIO_METADATA_PATH)
    save_json(
        {
            "signature": signature,
            "inputs": signature_payload,
            "path_count": int(metadata["path_key"].nunique()),
            "dates": int(len(matrices["gross_returns"])),
        },
        RESEARCH_CACHE_MANIFEST_PATH,
    )


def load_portfolio_cache():
    matrices = {
        name: pd.read_parquet(path)
        for name, path in PORTFOLIO_CACHE_PATHS.items()
    }
    metadata = pd.read_csv(PORTFOLIO_METADATA_PATH)
    return matrices, metadata


def clear_current_research_outputs():
    for directory in (RESEARCH_CACHE_DIR, RESEARCH_FIGURES_DIR):
        resolved = os.path.abspath(directory)
        if (
            resolved == os.path.abspath(PROJECT_ROOT)
            or os.path.commonpath([resolved, PROJECT_ROOT])
            != os.path.abspath(PROJECT_ROOT)
        ):
            raise ValueError(f"Unsafe Research Layer path: {resolved}")
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    if os.path.isdir(RESEARCH_RESULTS_DIR):
        for name in os.listdir(RESEARCH_RESULTS_DIR):
            path = os.path.join(RESEARCH_RESULTS_DIR, name)
            if name == "Figures":
                continue
            if os.path.isfile(path):
                os.remove(path)
