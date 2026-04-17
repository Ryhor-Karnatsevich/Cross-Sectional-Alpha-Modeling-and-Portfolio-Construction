from get_tickers import get_sp500_tickers
from data import build_dataset


def main():
    print("Loading S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Loaded {len(tickers)} tickers")

    print("\nDownloading price data...")
    prices, returns = build_dataset(tickers)

    print("\nPrices (first 5 rows):")
    print(prices.head())

    print("\nReturns (first 5 rows):")
    print(returns.head())


if __name__ == "__main__":
    main()