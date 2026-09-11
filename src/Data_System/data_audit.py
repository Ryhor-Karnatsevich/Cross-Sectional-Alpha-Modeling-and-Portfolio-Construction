"""Read-only audit of the complete Data System output bundle."""

from datetime import datetime
import hashlib
import os
import tempfile

import numpy as np
import pandas as pd

from config import (
    AUDIT_FILE_TIME_SPREAD_HOURS,
    AUDIT_NUMERIC_TOLERANCE,
    AUDIT_VOLUME_JUMP_RATIO,
    AVAILABILITY_PATH,
    CONFIRMED_REAL_RETURN_EVENTS,
    DATA_AUDIT_REPORT_PATH,
    DATA_START_DATE,
    FORWARD_RETURNS_PATH,
    HISTORICAL_COMPONENTS_PATH,
    LIQUIDITY_PATH,
    MEMBERSHIP_PATH,
    PRICES_LONG_PATH,
    QUALITY_PATH,
    RAW_PRICES_PATH,
    RETURNS_PATH,
    RISK_FREE_RATE_PATH,
    UNIVERSE_PATH,
    VOLUME_PATH,
    VOLUME_QUALITY_PATH,
)


DEFAULT_PATHS = {
    "prices": RAW_PRICES_PATH,
    "returns": RETURNS_PATH,
    "volume": VOLUME_PATH,
    "volume_quality": VOLUME_QUALITY_PATH,
    "liquidity": LIQUIDITY_PATH,
    "prices_long": PRICES_LONG_PATH,
    "availability": AVAILABILITY_PATH,
    "forward_returns": FORWARD_RETURNS_PATH,
    "membership": MEMBERSHIP_PATH,
    "quality": QUALITY_PATH,
    "universe": UNIVERSE_PATH,
    "historical_components": HISTORICAL_COMPONENTS_PATH,
    "risk_free_rate": RISK_FREE_RATE_PATH,
}

WIDE_MATRIX_NAMES = (
    "prices",
    "returns",
    "volume",
    "volume_quality",
    "liquidity",
    "availability",
    "forward_returns",
    "membership",
    "quality",
)


def add_check(checks, section, name, status, details):
    checks.append({
        "section": section,
        "name": name,
        "status": status,
        "details": str(details),
    })


def normalize_ticker(ticker):
    return str(ticker).strip().upper().replace(".", "-")


def normalize_history(history):
    history = history.loc[:, ["date", "tickers"]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="raise")
    history = history.sort_values("date").drop_duplicates("date", keep="last")
    history["tickers"] = history["tickers"].apply(
        lambda value: ",".join(
            normalize_ticker(ticker) for ticker in str(value).split(",")
        )
    )
    return history


def frame_is_aligned(frame, reference):
    return frame.index.equals(reference.index) and frame.columns.equals(reference.columns)


def numeric_mismatch_count(actual, expected, tolerance=AUDIT_NUMERIC_TOLERANCE):
    actual_values = actual.to_numpy(dtype=float)
    expected_values = expected.to_numpy(dtype=float)
    matches = np.isclose(
        actual_values,
        expected_values,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )
    return int((~matches).sum())


def build_bundle_fingerprint(paths):
    records = []
    for name, path in sorted(paths.items()):
        if os.path.exists(path):
            stat = os.stat(path)
            records.append(f"{name}|{stat.st_size}|{stat.st_mtime_ns}")
        else:
            records.append(f"{name}|MISSING")
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def load_files(paths, checks, inventory):
    loaded = {}

    for name, path in paths.items():
        if not os.path.exists(path):
            inventory.append((name, path, "MISSING", "-", "-"))
            add_check(checks, "Files", name, "FAIL", "Required file is missing")
            continue

        stat = os.stat(path)
        modified = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(
            timespec="seconds"
        )
        inventory.append(
            (name, path, "FOUND", f"{stat.st_size / 1024 / 1024:.2f} MB", modified)
        )

        try:
            if path.lower().endswith(".csv"):
                loaded[name] = pd.read_csv(path)
            else:
                loaded[name] = pd.read_parquet(path)
            add_check(checks, "Files", name, "PASS", "File opens successfully")
        except Exception as error:
            add_check(
                checks,
                "Files",
                name,
                "FAIL",
                f"Cannot read file: {type(error).__name__}: {error}",
            )

    return loaded


def check_file_generation_times(paths, checks):
    equity_names = [name for name in paths if name != "risk_free_rate"]
    existing_times = [
        os.path.getmtime(paths[name])
        for name in equity_names
        if os.path.exists(paths[name])
    ]
    if len(existing_times) != len(equity_names):
        add_check(
            checks,
            "Files",
            "Equity file generation times",
            "WARNING",
            "Cannot compare timestamps until every equity file exists",
        )
        return

    spread_hours = (max(existing_times) - min(existing_times)) / 3600
    status = "PASS" if spread_hours <= AUDIT_FILE_TIME_SPREAD_HOURS else "WARNING"
    add_check(
        checks,
        "Files",
        "Equity file generation times",
        status,
        f"Modification-time spread: {spread_hours:.2f} hours",
    )


def check_wide_matrices(loaded, checks):
    prices = loaded.get("prices")
    if prices is None:
        add_check(
            checks,
            "Matrices",
            "Wide matrix alignment",
            "WARNING",
            "Not checked because prices are unavailable",
        )
        return

    valid_index = (
        isinstance(prices.index, pd.DatetimeIndex)
        and prices.index.is_monotonic_increasing
        and not prices.index.duplicated().any()
    )
    add_check(
        checks,
        "Matrices",
        "Price date index",
        "PASS" if valid_index else "FAIL",
        f"Rows: {len(prices):,}; dates: {prices.index.min()} -> {prices.index.max()}",
    )

    unique_columns = prices.columns.is_unique
    add_check(
        checks,
        "Matrices",
        "Unique price tickers",
        "PASS" if unique_columns else "FAIL",
        f"Tickers: {prices.shape[1]:,}",
    )

    nonpositive = int(prices.notna().to_numpy().sum() - prices.gt(0).to_numpy().sum())
    infinite = int(np.isinf(prices.to_numpy(dtype=float)).sum())
    add_check(
        checks,
        "Matrices",
        "Valid observed prices",
        "PASS" if nonpositive == 0 and infinite == 0 else "FAIL",
        f"Non-positive: {nonpositive:,}; infinite: {infinite:,}",
    )

    for name in WIDE_MATRIX_NAMES[1:]:
        frame = loaded.get(name)
        if frame is None:
            continue
        aligned = frame_is_aligned(frame, prices)
        add_check(
            checks,
            "Matrices",
            f"{name} alignment",
            "PASS" if aligned else "FAIL",
            f"Shape: {frame.shape}; expected: {prices.shape}",
        )


def check_boolean_matrices(loaded, checks):
    prices = loaded.get("prices")
    for name in ("availability", "membership", "quality", "volume_quality"):
        frame = loaded.get(name)
        if frame is None:
            continue
        is_boolean = all(pd.api.types.is_bool_dtype(dtype) for dtype in frame.dtypes)
        add_check(
            checks,
            "Matrices",
            f"{name} boolean dtype",
            "PASS" if is_boolean else "FAIL",
            "All columns are boolean" if is_boolean else "Non-boolean columns found",
        )

    quality = loaded.get("quality")
    if prices is not None and quality is not None and frame_is_aligned(quality, prices):
        invalid_true = int((quality & ~(prices.notna() & prices.gt(0))).sum().sum())
        add_check(
            checks,
            "Quality",
            "Quality requires a positive observed price",
            "PASS" if invalid_true == 0 else "FAIL",
            f"Invalid True cells: {invalid_true:,}",
        )


def check_calculated_matrices(loaded, checks):
    prices = loaded.get("prices")
    quality = loaded.get("quality")
    returns = loaded.get("returns")
    forward_returns = loaded.get("forward_returns")
    availability = loaded.get("availability")
    membership = loaded.get("membership")
    volume = loaded.get("volume")
    volume_quality = loaded.get("volume_quality")
    liquidity = loaded.get("liquidity")

    if (
        prices is not None
        and quality is not None
        and returns is not None
        and frame_is_aligned(quality, prices)
        and frame_is_aligned(returns, prices)
    ):
        expected = prices.pct_change(fill_method=None)
        expected = expected.where(quality & quality.shift(1, fill_value=False))
        mismatches = numeric_mismatch_count(returns, expected)
        add_check(
            checks,
            "Calculations",
            "Daily returns reproduce from prices and quality",
            "PASS" if mismatches == 0 else "FAIL",
            f"Mismatched cells: {mismatches:,}",
        )

        suspicious = returns.abs().ge(0.5)
        extreme = returns.abs().ge(1.0)
        confirmed = pd.DataFrame(False, index=returns.index, columns=returns.columns)
        for ticker, date in CONFIRMED_REAL_RETURN_EVENTS:
            timestamp = pd.Timestamp(date)
            if ticker in confirmed.columns and timestamp in confirmed.index:
                confirmed.loc[timestamp, ticker] = True
        unconfirmed_extreme = int((extreme & ~confirmed).sum().sum())
        add_check(
            checks,
            "Quality",
            "Unclipped large returns",
            "PASS" if unconfirmed_extreme == 0 else "FAIL",
            (
                f"Absolute returns >=50%: {int(suspicious.sum().sum()):,}; "
                f"unconfirmed returns >=100%: {unconfirmed_extreme:,}"
            ),
        )

    if (
        prices is not None
        and quality is not None
        and forward_returns is not None
        and frame_is_aligned(quality, prices)
        and frame_is_aligned(forward_returns, prices)
    ):
        horizon = 21
        expected = prices.pct_change(horizon).shift(-horizon)
        expected = expected.where(quality & quality.shift(-horizon, fill_value=False))
        mismatches = numeric_mismatch_count(forward_returns, expected)
        add_check(
            checks,
            "Calculations",
            "21-day forward returns reproduce from prices and quality",
            "PASS" if mismatches == 0 else "FAIL",
            f"Mismatched cells: {mismatches:,}",
        )

    if all(frame is not None for frame in (prices, quality, membership, availability)):
        if all(
            frame_is_aligned(frame, prices)
            for frame in (quality, membership, availability)
        ):
            expected = prices.notna() & membership & quality
            mismatches = int((availability != expected).sum().sum())
            add_check(
                checks,
                "Calculations",
                "Availability formula",
                "PASS" if mismatches == 0 else "FAIL",
                f"Mismatched cells: {mismatches:,}",
            )

    if (
        prices is not None
        and volume is not None
        and volume_quality is not None
        and liquidity is not None
        and frame_is_aligned(volume, prices)
        and frame_is_aligned(volume_quality, prices)
        and frame_is_aligned(liquidity, prices)
    ):
        clean_volume = volume.where(volume_quality)
        expected = np.log1p((prices * clean_volume).rolling(20).mean())
        mismatches = numeric_mismatch_count(liquidity, expected)
        add_check(
            checks,
            "Calculations",
            "Liquidity formula",
            "PASS" if mismatches == 0 else "FAIL",
            f"Mismatched cells: {mismatches:,}",
        )


def longest_true_run(series):
    values = series.to_numpy(dtype=bool)
    if not values.any():
        return 0
    groups = np.cumsum(~values)
    return int(pd.Series(values).groupby(groups).sum().max())


def find_true_runs(series, minimum_length=5):
    runs = []
    start = None

    for position, (date, value) in enumerate(series.astype(bool).items()):
        if value and start is None:
            start = (position, date)
        elif not value and start is not None:
            start_position, start_date = start
            length = position - start_position
            if length >= minimum_length:
                runs.append((start_date, series.index[position - 1], length))
            start = None

    if start is not None:
        start_position, start_date = start
        length = len(series) - start_position
        if length >= minimum_length:
            runs.append((start_date, series.index[-1], length))

    return runs


def check_volume(loaded, checks):
    prices = loaded.get("prices")
    volume = loaded.get("volume")
    membership = loaded.get("membership")
    availability = loaded.get("availability")
    volume_quality = loaded.get("volume_quality")
    if prices is None or volume is None or not frame_is_aligned(volume, prices):
        add_check(
            checks,
            "Volume",
            "Volume diagnostics",
            "WARNING",
            "Not checked because aligned prices and volume are unavailable",
        )
        return

    volume_values = volume.to_numpy(dtype=float)
    negative = int((volume_values < 0).sum())
    infinite = int(np.isinf(volume_values).sum())
    add_check(
        checks,
        "Volume",
        "Valid volume values",
        "PASS" if negative == 0 and infinite == 0 else "FAIL",
        f"Negative: {negative:,}; infinite: {infinite:,}",
    )

    if volume_quality is not None and frame_is_aligned(volume_quality, volume):
        finite = pd.DataFrame(
            np.isfinite(volume_values),
            index=volume.index,
            columns=volume.columns,
        )
        expected_quality = volume.notna() & finite & volume.gt(0)
        mismatches = int((volume_quality != expected_quality).sum().sum())
        add_check(
            checks,
            "Volume",
            "Volume quality formula",
            "PASS" if mismatches == 0 else "FAIL",
            f"Mismatched cells: {mismatches:,}",
        )

    if membership is not None and frame_is_aligned(membership, prices):
        relevant = prices.notna() & membership
    else:
        relevant = prices.notna()

    relevant_count = int(relevant.sum().sum())
    missing = int((relevant & volume.isna()).sum().sum())
    zeros = int((relevant & volume.eq(0)).sum().sum())
    missing_ratio = missing / relevant_count if relevant_count else 0
    zero_ratio = zeros / relevant_count if relevant_count else 0
    status = "WARNING" if missing or zeros else "PASS"
    add_check(
        checks,
        "Volume",
        "Missing and zero volume during membership",
        status,
        (
            f"Relevant observations: {relevant_count:,}; missing: {missing:,} "
            f"({missing_ratio:.4%}); zero: {zeros:,} ({zero_ratio:.4%})"
        ),
    )

    if volume_quality is not None and frame_is_aligned(volume_quality, volume):
        raw_invalid = relevant & (
            volume.isna()
            | volume.le(0)
            | ~pd.DataFrame(
                np.isfinite(volume_values),
                index=volume.index,
                columns=volume.columns,
            )
        )
        invalid_count = int(raw_invalid.sum().sum())
        leaked = int((raw_invalid & volume_quality).sum().sum())
        add_check(
            checks,
            "Volume",
            "Raw invalid volume excluded by volume_quality",
            "PASS" if leaked == 0 else "FAIL",
            (
                f"Invalid raw observations: {invalid_count:,}; excluded: "
                f"{invalid_count - leaked:,}; still accepted: {leaked:,}"
            ),
        )

    positive_volume = volume.where(volume.gt(0))
    previous = positive_volume.shift(1)
    ratio = positive_volume.div(previous)
    valid_pair = volume.gt(0) & previous.gt(0)
    if volume_quality is not None and frame_is_aligned(volume_quality, volume):
        valid_pair &= volume_quality & volume_quality.shift(1, fill_value=False)
    large_jumps = relevant & valid_pair & (
        ratio.ge(AUDIT_VOLUME_JUMP_RATIO)
        | ratio.le(1 / AUDIT_VOLUME_JUMP_RATIO)
    )
    jump_count = int(large_jumps.sum().sum())
    affected_tickers = int(large_jumps.any().sum())
    jump_counts = large_jumps.sum().loc[lambda values: values.gt(0)].sort_values(
        ascending=False
    )
    top_jump_tickers = ", ".join(
        f"{ticker}: {int(count)}"
        for ticker, count in jump_counts.head(10).items()
    ) or "None"
    add_check(
        checks,
        "Volume",
        f"Positive volume jumps remaining after volume_quality >= {AUDIT_VOLUME_JUMP_RATIO}x",
        "WARNING" if jump_count else "PASS",
        (
            f"Remaining positive-to-positive events: {jump_count:,} across "
            f"{affected_tickers:,} tickers; most affected: {top_jump_tickers}. "
            "Not removed automatically because both observations are valid "
            "positive volumes."
        ),
    )

    problematic_runs = {}
    run_details = []
    invalid_volume = relevant & (volume.isna() | volume.eq(0))
    for ticker in invalid_volume.columns:
        ticker_runs = find_true_runs(invalid_volume[ticker], minimum_length=5)
        if ticker_runs:
            problematic_runs[ticker] = max(length for _, _, length in ticker_runs)
        for start, end, length in ticker_runs:
            dates = invalid_volume.loc[start:end].index
            zero_count = int(
                (relevant.loc[dates, ticker] & volume.loc[dates, ticker].eq(0)).sum()
            )
            missing_count = int(
                (relevant.loc[dates, ticker] & volume.loc[dates, ticker].isna()).sum()
            )
            if volume_quality is not None and frame_is_aligned(volume_quality, volume):
                still_accepted = int(volume_quality.loc[dates, ticker].sum())
            else:
                still_accepted = length
            run_details.append({
                "ticker": ticker,
                "start": pd.Timestamp(start),
                "end": pd.Timestamp(end),
                "length": length,
                "zeros": zero_count,
                "missing": missing_count,
                "still_accepted": still_accepted,
            })
    run_details.sort(key=lambda run: run["length"], reverse=True)
    run_counts = {}
    for run in run_details:
        run_counts[run["ticker"]] = run_counts.get(run["ticker"], 0) + 1
    run_summary = ", ".join(
        f"{ticker}: {run_counts[ticker]} runs, longest {longest} days"
        for ticker, longest in sorted(
            problematic_runs.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ) or "None"
    long_run_observations = sum(run["length"] for run in run_details)
    long_run_still_accepted = sum(run["still_accepted"] for run in run_details)
    add_check(
        checks,
        "Volume",
        "Runs of at least 5 missing/zero volume observations",
        "WARNING" if problematic_runs else "PASS",
        (
            f"Found {len(run_details):,} runs ({long_run_observations:,} "
            f"observations) across {len(problematic_runs):,} tickers: "
            f"{run_summary}."
        ),
    )

    add_check(
        checks,
        "Volume",
        "Long missing/zero runs excluded by volume_quality",
        "PASS" if long_run_still_accepted == 0 else "FAIL",
        (
            f"Observations inside long runs: {long_run_observations:,}; "
            f"excluded: {long_run_observations - long_run_still_accepted:,}; "
            f"still accepted: {long_run_still_accepted:,}"
        ),
    )

    if availability is not None and frame_is_aligned(availability, prices):
        if volume_quality is not None and frame_is_aligned(volume_quality, prices):
            usable_missing = int((availability & ~volume_quality).sum().sum())
        else:
            usable_missing = int((availability & volume.isna()).sum().sum())
        add_check(
            checks,
            "Volume",
            "Invalid volume excluded from liquidity inputs",
            "PASS" if volume_quality is not None else "WARNING",
            (
                f"Excluded from volume/liquidity analysis while retained for "
                f"price analysis: {usable_missing:,} observations"
            ),
        )


def check_prices_long(loaded, checks):
    prices = loaded.get("prices")
    long_prices = loaded.get("prices_long")
    if prices is None or long_prices is None:
        return

    column_lookup = {str(column).lower(): column for column in long_prices.columns}
    required = {"date", "ticker", "price"}
    if not required.issubset(column_lookup):
        add_check(
            checks,
            "Calculations",
            "Long prices schema",
            "FAIL",
            f"Columns: {list(long_prices.columns)}",
        )
        return

    normalized = long_prices.rename(columns={
        column_lookup["date"]: "date",
        column_lookup["ticker"]: "ticker",
        column_lookup["price"]: "price",
    }).loc[:, ["date", "ticker", "price"]]
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")

    duplicates = int(normalized.duplicated(["date", "ticker"]).sum())
    expected_rows = int(prices.notna().sum().sum())
    row_match = len(normalized) == expected_rows
    add_check(
        checks,
        "Calculations",
        "Long prices rows and uniqueness",
        "PASS" if row_match and duplicates == 0 else "FAIL",
        (
            f"Rows: {len(normalized):,}; expected: {expected_rows:,}; "
            f"duplicate date-ticker pairs: {duplicates:,}"
        ),
    )

    if row_match and duplicates == 0 and normalized["date"].notna().all():
        actual = normalized.set_index(["date", "ticker"])["price"].sort_index()
        expected = prices.stack().dropna().rename("price")
        expected.index = expected.index.set_names(["date", "ticker"])
        expected = expected.sort_index()
        same_index = actual.index.equals(expected.index)
        mismatches = (
            int((~np.isclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)).sum())
            if same_index
            else max(len(actual), len(expected))
        )
        add_check(
            checks,
            "Calculations",
            "Long prices reproduce wide prices",
            "PASS" if same_index and mismatches == 0 else "FAIL",
            f"Same keys: {same_index}; mismatched prices: {mismatches:,}",
        )


def historical_ticker_union(history):
    start = pd.Timestamp(DATA_START_DATE)
    position = history["date"].searchsorted(start, side="right") - 1
    if position < 0:
        raise ValueError("Historical components do not cover DATA_START_DATE")
    tickers = set()
    for value in history.iloc[position:]["tickers"]:
        tickers.update(value.split(","))
    return tickers


def build_expected_membership(history, dates, tickers):
    ticker_positions = {ticker: position for position, ticker in enumerate(tickers)}
    snapshots = np.zeros((len(history), len(tickers)), dtype=bool)
    for row_number, value in enumerate(history["tickers"]):
        positions = [
            ticker_positions[ticker]
            for ticker in value.split(",")
            if ticker in ticker_positions
        ]
        snapshots[row_number, positions] = True
    snapshots = pd.DataFrame(snapshots, index=history["date"], columns=tickers)
    membership = snapshots.reindex(pd.DatetimeIndex(dates), method="ffill")
    return membership.fillna(False).astype(bool)


def check_history_and_universe(loaded, checks):
    prices = loaded.get("prices")
    membership = loaded.get("membership")
    history = loaded.get("historical_components")
    universe = loaded.get("universe")

    normalized_history = None
    if history is not None:
        required = {"date", "tickers"}
        if not required.issubset(history.columns):
            add_check(
                checks,
                "Universe",
                "Historical components schema",
                "FAIL",
                f"Columns: {list(history.columns)}",
            )
        else:
            try:
                normalized_history = normalize_history(history)
                component_counts = normalized_history["tickers"].str.split(",").str.len()
                relevant = component_counts.loc[
                    normalized_history["date"] >= pd.Timestamp(DATA_START_DATE)
                ]
                valid = (
                    not relevant.empty
                    and relevant.between(450, 550).all()
                    and normalized_history["date"].is_unique
                )
                add_check(
                    checks,
                    "Universe",
                    "Historical component snapshots",
                    "PASS" if valid else "FAIL",
                    (
                        f"Snapshots: {len(normalized_history):,}; component range: "
                        f"{int(relevant.min()) if not relevant.empty else 0} -> "
                        f"{int(relevant.max()) if not relevant.empty else 0}"
                    ),
                )
            except Exception as error:
                add_check(
                    checks,
                    "Universe",
                    "Historical component snapshots",
                    "FAIL",
                    f"Cannot validate: {type(error).__name__}: {error}",
                )

    if (
        normalized_history is not None
        and prices is not None
        and membership is not None
        and frame_is_aligned(membership, prices)
    ):
        expected = build_expected_membership(
            normalized_history,
            prices.index,
            list(prices.columns),
        )
        mismatches = int((membership != expected).sum().sum())
        add_check(
            checks,
            "Universe",
            "Membership reproduces from historical snapshots",
            "PASS" if mismatches == 0 else "FAIL",
            f"Mismatched cells: {mismatches:,}",
        )
        latest_snapshot = normalized_history["date"].max()
        within_boundary = prices.index.max() <= latest_snapshot
        add_check(
            checks,
            "Universe",
            "Price data does not exceed membership source",
            "PASS" if within_boundary else "FAIL",
            f"Last price: {prices.index.max()}; last snapshot: {latest_snapshot}",
        )

    if universe is None:
        return

    required_columns = {
        "ticker",
        "yahoo_ticker",
        "download_method",
        "membership_observations",
        "price_coverage_during_membership",
        "has_price_during_membership",
        "has_any_price_data",
        "retained_in_dataset",
    }
    missing_columns = sorted(required_columns - set(universe.columns))
    add_check(
        checks,
        "Universe",
        "Universe report schema",
        "PASS" if not missing_columns else "FAIL",
        f"Missing columns: {missing_columns or 'none'}",
    )
    if missing_columns or prices is None:
        return

    universe_tickers = set(universe["ticker"].astype(str))
    price_tickers = set(map(str, prices.columns))
    ticker_match = universe_tickers == price_tickers
    add_check(
        checks,
        "Universe",
        "Universe report matches price columns",
        "PASS" if ticker_match else "FAIL",
        (
            f"Only in universe: {len(universe_tickers - price_tickers):,}; "
            f"only in prices: {len(price_tickers - universe_tickers):,}"
        ),
    )

    if normalized_history is not None:
        source_tickers = historical_ticker_union(normalized_history)
        source_match = source_tickers == universe_tickers
        add_check(
            checks,
            "Universe",
            "Universe contains the historical ticker union",
            "PASS" if source_match else "FAIL",
            (
                f"Source union: {len(source_tickers):,}; report: {len(universe_tickers):,}; "
                f"difference: {len(source_tickers ^ universe_tickers):,}"
            ),
        )

    coverage = pd.to_numeric(
        universe["price_coverage_during_membership"], errors="coerce"
    )
    invalid_coverage = int((coverage.isna() | ~coverage.between(0, 1)).sum())
    add_check(
        checks,
        "Universe",
        "Valid membership coverage values",
        "PASS" if invalid_coverage == 0 else "FAIL",
        f"Invalid rows: {invalid_coverage:,}",
    )

    has_price = universe["has_price_during_membership"].fillna(False).astype(bool)
    unavailable = int((~has_price).sum())
    low_coverage = int((coverage < 0.8).sum())
    methods = universe["download_method"].fillna("missing").astype(str)
    rejected = int(methods.eq("reused_symbol_rejected").sum())
    missing = int(methods.eq("missing").sum())
    add_check(
        checks,
        "Universe",
        "Historical data availability",
        "WARNING" if unavailable or low_coverage else "PASS",
        (
            f"No prices during membership: {unavailable:,}; coverage below 80%: "
            f"{low_coverage:,}; missing downloads: {missing:,}; rejected reused symbols: "
            f"{rejected:,}"
        ),
    )

    if membership is not None and frame_is_aligned(membership, prices):
        report = universe.set_index("ticker").reindex(prices.columns)
        expected_membership_counts = membership.sum().astype(float)
        reported_counts = pd.to_numeric(
            report["membership_observations"], errors="coerce"
        )
        mismatches = int(
            (~np.isclose(
                reported_counts.to_numpy(dtype=float),
                expected_membership_counts.to_numpy(dtype=float),
                equal_nan=False,
            )).sum()
        )
        add_check(
            checks,
            "Universe",
            "Universe membership counts",
            "PASS" if mismatches == 0 else "FAIL",
            f"Mismatched tickers: {mismatches:,}",
        )


def check_risk_free_rate(loaded, prices, checks):
    rates = loaded.get("risk_free_rate")
    if rates is None:
        return
    valid_schema = "annual_rate_pct" in rates.columns
    add_check(
        checks,
        "Risk-free rate",
        "DGS3MO schema",
        "PASS" if valid_schema else "FAIL",
        f"Columns: {list(rates.columns)}",
    )
    if not valid_schema:
        return

    valid_index = (
        isinstance(rates.index, pd.DatetimeIndex)
        and rates.index.is_monotonic_increasing
        and not rates.index.duplicated().any()
    )
    values = pd.to_numeric(rates["annual_rate_pct"], errors="coerce")
    negative = int((values.dropna() < 0).sum())
    infinite = int(np.isinf(values.to_numpy(dtype=float)).sum())
    add_check(
        checks,
        "Risk-free rate",
        "Valid DGS3MO observations",
        "PASS" if valid_index and negative == 0 and infinite == 0 else "FAIL",
        (
            f"Observed: {values.notna().sum():,}; missing: {values.isna().sum():,}; "
            f"negative: {negative:,}; infinite: {infinite:,}"
        ),
    )

    if prices is not None and values.notna().any():
        first_rate = values.dropna().index.min()
        last_rate = values.dropna().index.max()
        covers_start = first_rate <= prices.index.min() + pd.Timedelta(days=7)
        covers_end = last_rate >= prices.index.max() - pd.Timedelta(days=7)
        add_check(
            checks,
            "Risk-free rate",
            "DGS3MO covers the equity period",
            "PASS" if covers_start and covers_end else "WARNING",
            (
                f"Rates: {first_rate} -> {last_rate}; equities: "
                f"{prices.index.min()} -> {prices.index.max()}"
            ),
        )


def overall_status(checks):
    statuses = {check["status"] for check in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARNING" in statuses:
        return "WARNING"
    return "PASS"


def run_check_group(checks, group_name, function, *args):
    try:
        function(*args, checks)
    except Exception as error:
        add_check(
            checks,
            "Audit execution",
            group_name,
            "FAIL",
            f"Check could not finish: {type(error).__name__}: {error}",
        )


def markdown_escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(checks, inventory, fingerprint, checked_at):
    status = overall_status(checks)
    counts = {
        name: sum(check["status"] == name for check in checks)
        for name in ("PASS", "WARNING", "FAIL")
    }
    lines = [
        "# Data System Audit Report",
        "",
        f"- Checked at: `{checked_at}`",
        f"- Overall status: **{status}**",
        f"- Bundle fingerprint: `{fingerprint}`",
        (
            f"- Checks: PASS {counts['PASS']} / WARNING {counts['WARNING']} / "
            f"FAIL {counts['FAIL']}"
        ),
        "- Audit mode: read-only; no data were downloaded, deleted or corrected.",
        "",
        "## Checks",
        "",
        "| # | Section | Status | Check | Result |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for number, check in enumerate(checks, start=1):
        lines.append(
            "| {number} | {section} | **{status}** | {name} | {details} |".format(
                number=number,
                section=markdown_escape(check["section"]),
                status=check["status"],
                name=markdown_escape(check["name"]),
                details=markdown_escape(check["details"]),
            )
        )

    lines.extend([
        "",
        "## File inventory",
        "",
        "| Dataset | File | State | Size | Modified |",
        "| --- | --- | --- | --- | --- |",
    ])
    for name, path, state, size, modified in inventory:
        lines.append(
            f"| {markdown_escape(name)} | `{markdown_escape(path)}` | "
            f"{state} | {size} | {modified} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **FAIL** means that a file or a mathematical relationship is incorrect.",
        "- **WARNING** means that the dataset remains usable only with a documented limitation.",
        "- **PASS** means that the specific internal check succeeded.",
        "",
        "## Limits of this audit",
        "",
        "- Internal consistency does not prove that Yahoo or the historical membership source is correct.",
        "- Missing delisted securities and survivorship/data-availability bias cannot be repaired by this audit.",
        "- Modification times help detect mixed builds but do not make multi-file saving atomic.",
        "- Large price and volume moves are diagnostics; economic reality may require an external source.",
        "",
    ])
    return "\n".join(lines), status


def write_report(report, report_path):
    directory = os.path.dirname(report_path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix="data_audit_",
        suffix=".tmp",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(report)
        os.replace(temporary_path, report_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def run_data_audit(paths=None, report_path=DATA_AUDIT_REPORT_PATH):
    paths = dict(DEFAULT_PATHS if paths is None else paths)
    checks = []
    inventory = []
    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    fingerprint = build_bundle_fingerprint(paths)

    loaded = load_files(paths, checks, inventory)
    run_check_group(
        checks,
        "File generation times",
        check_file_generation_times,
        paths,
    )
    run_check_group(checks, "Wide matrices", check_wide_matrices, loaded)
    run_check_group(checks, "Boolean matrices", check_boolean_matrices, loaded)
    run_check_group(
        checks,
        "Calculated matrices",
        check_calculated_matrices,
        loaded,
    )
    run_check_group(checks, "Long prices", check_prices_long, loaded)
    run_check_group(
        checks,
        "History and universe",
        check_history_and_universe,
        loaded,
    )
    run_check_group(checks, "Volume", check_volume, loaded)
    run_check_group(
        checks,
        "Risk-free rate",
        check_risk_free_rate,
        loaded,
        loaded.get("prices"),
    )

    report, status = render_report(
        checks,
        inventory,
        fingerprint,
        checked_at,
    )
    write_report(report, report_path)
    print(f"Data audit: {status}")
    print(f"Audit report: {report_path}")
    return status, checks


if __name__ == "__main__":
    run_data_audit()
