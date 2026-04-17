from get_tickers import get_sp500_tickers
from data import (
    load_raw,
    load_returns,
    build_and_save_dataset
)


def main():
    print("Checking local data...")

    prices = load_raw()
    returns = load_returns()

    if prices is None or returns is None:
        print("No local data found -> downloading...")

        tickers = get_sp500_tickers()
        print(f"Loaded {len(tickers)} tickers")

        prices, returns = build_and_save_dataset(tickers)

    else:
        print("Loaded data from disk")

    print("\nPrices:")
    print(prices.head())

    print("\nReturns:")
    print(returns.head())


if __name__ == "__main__":
    main()