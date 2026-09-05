"""Download and validate the daily US three-month Treasury yield from FRED."""

from io import StringIO
import os

import pandas as pd
import requests

from config import FRED_DGS3MO_CSV_URL, RISK_FREE_RATE_PATH


def download_dgs3mo(start_date, end_date):
    """Download the official FRED DGS3MO series without requiring an API key."""
    response = requests.get(
        FRED_DGS3MO_CSV_URL,
        params={"cosd": start_date, "coed": end_date},
        timeout=30,
    )
    response.raise_for_status()

    data = pd.read_csv(StringIO(response.text))
    if data.shape[1] != 2:
        raise ValueError(f"Unexpected DGS3MO columns: {list(data.columns)}")

    data.columns = ["date", "annual_rate_pct"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["annual_rate_pct"] = pd.to_numeric(
        data["annual_rate_pct"],
        errors="coerce",
    )
    data = data.dropna(subset=["date"]).set_index("date").sort_index()
    data = data.loc[start_date:end_date]

    if data.empty or data["annual_rate_pct"].notna().sum() < 100:
        raise RuntimeError("FRED returned insufficient DGS3MO observations")
    if data.index.duplicated().any():
        raise ValueError("Duplicate DGS3MO dates detected")
    if (data["annual_rate_pct"].dropna() < 0).any():
        raise ValueError("Unexpected negative DGS3MO observation")

    return data


def ensure_risk_free_rate(start_date, end_date, force_download=False):
    """Load a valid local DGS3MO file or refresh it from FRED."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if os.path.exists(RISK_FREE_RATE_PATH) and not force_download:
        data = pd.read_parquet(RISK_FREE_RATE_PATH)
        data.index = pd.to_datetime(data.index)
        coverage_start = data["annual_rate_pct"].dropna().index.min()
        coverage_end = data["annual_rate_pct"].dropna().index.max()
        if coverage_start <= start and coverage_end >= end:
            return data.loc[start:end]

    data = download_dgs3mo(start.date().isoformat(), end.date().isoformat())
    os.makedirs(os.path.dirname(RISK_FREE_RATE_PATH), exist_ok=True)
    data.to_parquet(RISK_FREE_RATE_PATH)
    return data
