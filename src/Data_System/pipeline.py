from data import run_pipeline


def main():
    print("Checking local data...")

    data = run_pipeline()

    print("Loaded data from disk or completed rebuild")
    prices, returns, volume, liquidity, prices_long, availability,forward_returns = data

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

