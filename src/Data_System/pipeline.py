import os

from config import (
    AVAILABILITY_PATH,
    DATA_START_DATE,
    FORWARD_RETURNS_PATH,
    HISTORICAL_COMPONENTS_PATH,
    LIQUIDITY_PATH,
    MEMBERSHIP_PATH,
    PRICES_LONG_PATH,
    QUALITY_PATH,
    RAW_PRICES_PATH,
    RETURNS_PATH,
    UNIVERSE_PATH,
    VOLUME_PATH,
    VOLUME_QUALITY_PATH,
)
from data import build_and_save_dataset, load_saved_equity_data
from data_audit import run_data_audit
from get_tickers import get_sp500_history, get_sp500_tickers
from risk_free_rate import prepare_risk_free_rate


EQUITY_REQUIRED_PATHS = (
    RAW_PRICES_PATH,
    RETURNS_PATH,
    VOLUME_PATH,
    VOLUME_QUALITY_PATH,
    LIQUIDITY_PATH,
    PRICES_LONG_PATH,
    AVAILABILITY_PATH,
    FORWARD_RETURNS_PATH,
    MEMBERSHIP_PATH,
    QUALITY_PATH,
    UNIVERSE_PATH,
    HISTORICAL_COMPONENTS_PATH,
)


def prepare_equity_data():
    if all(os.path.exists(path) for path in EQUITY_REQUIRED_PATHS):
        print("Equity dataset found -> loading")
        return load_saved_equity_data()

    print("Equity dataset incomplete -> rebuilding")
    history = get_sp500_history()
    tickers = get_sp500_tickers(history)

    print(f"Historical source snapshots: {len(history)}")
    print(f"Historical ticker union since {DATA_START_DATE}: {len(tickers)}")

    return build_and_save_dataset(history, tickers)


def run_pipeline():
    print("Checking Data System files...")

    equity_data = prepare_equity_data()
    prices = equity_data[0]

    prepare_risk_free_rate(
        DATA_START_DATE,
        prices.index.max().date().isoformat(),
    )

    run_data_audit()

    print("Data System is ready")
    return equity_data


def print_summary(data):
    names = (
        "Prices",
        "Returns",
        "Volume",
        "Liquidity",
        "Long format",
        "Availability",
        "Forward Returns",
    )

    for name, frame in zip(names, data):
        print(f"\n{name}:")
        frame.info()


def main():
    data = run_pipeline()
    print_summary(data)


if __name__ == "__main__":
    main()
