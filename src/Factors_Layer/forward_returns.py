from factor_config import FORWARD_HORIZONS
from factor_storage import save_forward_return_matrix


# -------------------------
# FORWARD RETURN CALCULATION
def compute_forward_returns(prices, price_quality, horizon):
    future_prices = prices.shift(-horizon)
    future_quality = price_quality.shift(-horizon, fill_value=False)
    valid = (
        price_quality
        & future_quality
        & prices.notna()
        & future_prices.notna()
    )
    return (future_prices / prices - 1).where(valid)


# -------------------------
# COMPLETE FORWARD RETURN SET
def build_forward_return_matrices(inputs):
    prices = inputs["prices"]
    price_quality = inputs["price_quality"]

    for horizon in FORWARD_HORIZONS:
        print(f"Forward-return matrix: {horizon} trading days")
        forward_returns = compute_forward_returns(
            prices,
            price_quality,
            horizon,
        )
        save_forward_return_matrix(horizon, forward_returns)
