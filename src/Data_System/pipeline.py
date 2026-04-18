from get_tickers import get_sp500_tickers
from data import run_pipeline


def main():
    print("Checking local data...")

    data = run_pipeline()

    if data is None:
        print("No local data found -> building dataset")

        tickers = get_sp500_tickers()
        print(f"Loaded {len(tickers)} tickers")

        prices, returns, volume, liquidity, prices_long, availability,forward_returns = run_pipeline(tickers)

    else:
        print("Loaded data from disk")
        prices, returns, volume, liquidity, prices_long, availability,forward_returns = data

        expected = set(get_sp500_tickers())
        actual_with_data = set(prices.columns[prices.notna().any()])
        missing = expected - actual_with_data
        if missing:
            print(f"Warning: {len(missing)} tickers have been deleted: {list(missing)[:10]}...")

    print("\nPrices:")
    print(prices.info())

    print("\nReturns:")
    print(returns.info())

    print("\nVolume:")
    print(volume.info())

    print("\nLiquidity:")
    print(liquidity.info())

    print("\nLong format:")
    print(prices_long.info())

    print("\nAvailability:")
    print(availability.info())
    print(availability.shape)

    print("\nForward Returns:")
    print(forward_returns.info())


if __name__ == "__main__":
    main()

