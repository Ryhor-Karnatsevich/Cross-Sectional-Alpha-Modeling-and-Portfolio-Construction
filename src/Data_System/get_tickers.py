from io import BytesIO
import os

import numpy as np
import pandas as pd
import requests

from config import (
    HISTORICAL_COMPONENTS_PATH,
    HISTORICAL_COMPONENTS_URL,
    START_DATE,
)


def normalize_ticker(ticker):
    return ticker.strip().upper().replace(".", "-")


def get_sp500_history(refresh=True):
    if refresh:
        try:
            response = requests.get(
                HISTORICAL_COMPONENTS_URL,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=60,
            )
            response.raise_for_status()

            history = pd.read_csv(BytesIO(response.content))

            os.makedirs(os.path.dirname(HISTORICAL_COMPONENTS_PATH), exist_ok=True)
            with open(HISTORICAL_COMPONENTS_PATH, "wb") as file:
                file.write(response.content)
        except requests.RequestException:
            if not os.path.exists(HISTORICAL_COMPONENTS_PATH):
                raise
            print("Historical universe download failed -> using local snapshot")
            history = pd.read_csv(HISTORICAL_COMPONENTS_PATH)
    else:
        history = pd.read_csv(HISTORICAL_COMPONENTS_PATH)

    required_columns = {"date", "tickers"}
    if not required_columns.issubset(history.columns):
        raise ValueError("Historical components file must contain date and tickers columns")

    history = history.loc[:, ["date", "tickers"]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="raise")
    history = history.sort_values("date").drop_duplicates("date", keep="last")

    history["tickers"] = history["tickers"].apply(
        lambda value: ",".join(normalize_ticker(ticker) for ticker in value.split(","))
    )

    component_counts = history["tickers"].str.split(",").str.len()
    relevant_counts = component_counts.loc[history["date"] >= START_DATE]

    if relevant_counts.empty:
        raise ValueError(f"No historical S&P 500 snapshots found after {START_DATE}")
    if not relevant_counts.between(450, 550).all():
        raise ValueError("Unexpected number of S&P 500 components in source file")

    duplicates = history["tickers"].apply(
        lambda value: len(value.split(",")) != len(set(value.split(",")))
    )
    if duplicates.any():
        raise ValueError("Duplicate tickers found inside a historical snapshot")

    return history


def get_sp500_tickers(history=None, start=START_DATE):
    if history is None:
        history = get_sp500_history()

    start = pd.Timestamp(start)
    start_position = history["date"].searchsorted(start, side="right") - 1
    if start_position < 0:
        raise ValueError("Requested start date is earlier than source history")

    relevant_history = history.iloc[start_position:]
    tickers = set()

    for values in relevant_history["tickers"]:
        tickers.update(values.split(","))

    return sorted(tickers)


def get_sp500_tickers_by_date(date, history=None):
    if history is None:
        history = get_sp500_history()

    date = pd.Timestamp(date)
    position = history["date"].searchsorted(date, side="right") - 1

    if position < 0:
        raise ValueError("Requested date is earlier than source history")

    return history.iloc[position]["tickers"].split(",")


def build_membership_matrix(history, trading_dates, tickers):
    trading_dates = pd.DatetimeIndex(trading_dates)
    ticker_positions = {ticker: position for position, ticker in enumerate(tickers)}
    snapshot_values = np.zeros((len(history), len(tickers)), dtype=bool)

    for row_number, values in enumerate(history["tickers"]):
        positions = [
            ticker_positions[ticker]
            for ticker in values.split(",")
            if ticker in ticker_positions
        ]
        snapshot_values[row_number, positions] = True

    snapshots = pd.DataFrame(
        snapshot_values,
        index=history["date"],
        columns=tickers,
    )

    membership = snapshots.reindex(trading_dates, method="ffill")

    if membership.isna().any().any():
        raise ValueError("Membership history does not cover the first trading date")

    membership = membership.astype(bool)

    latest_snapshot = history["date"].max()
    if trading_dates.max() > latest_snapshot:
        print(
            "Warning: membership after "
            f"{latest_snapshot.date()} uses the latest available repository snapshot"
        )

    return membership


if __name__ == "__main__":
    history = get_sp500_history()
    tickers = get_sp500_tickers(history)

    print(f"Snapshots: {len(history)}")
    print(f"Source period: {history['date'].min().date()} -> {history['date'].max().date()}")
    print(f"Historical tickers since {START_DATE}: {len(tickers)}")
    print(f"Members on {START_DATE}: {len(get_sp500_tickers_by_date(START_DATE, history))}")
