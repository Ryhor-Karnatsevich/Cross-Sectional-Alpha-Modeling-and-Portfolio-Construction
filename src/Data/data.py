import yfinance as yf
import pandas as pd

# visual configurations
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format

# Download
def download_data(tickers, start="2018-01-01", batch_size=50):
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        data = yf.download(
            tickers=batch,
            start=start,
            auto_adjust=True,
            progress=False
        )

        all_data.append(data)

    data = pd.concat(all_data, axis=1)
    return data


# Alignment
def get_price_matrix(data):
    close = data["Close"].copy()
    close = close.dropna(how="all")
    return close


# Returns
def compute_returns(prices):
    returns = prices.pct_change()
    return returns


# Missing Data
def clean_data(prices):
    prices = prices.sort_index()
    prices = prices.ffill(limit=5)

    min_assets = int(0.5 * prices.shape[1])
    prices = prices.dropna(thresh=min_assets)
    return prices


# Main Engine
def build_dataset(tickers):
    raw = download_data(tickers)

    prices = get_price_matrix(raw)
    prices = clean_data(prices)

    returns = compute_returns(prices)
    return prices, returns


# connection between scripts
if __name__ == "__main__":
    prices, returns = build_dataset(TICKERS)

    print(prices.head())
    print(returns.head())