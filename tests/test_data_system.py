import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "Data_System"))

import delete
import data as data_module
import risk_free_rate
from data import compute_availability, compute_liquidity, get_price_matrix, get_volume_matrix
from data_quality import build_point_in_time_quality_mask


pipeline_path = os.path.join(PROJECT_ROOT, "src", "Data_System", "pipeline.py")
pipeline_spec = importlib.util.spec_from_file_location("data_system_pipeline", pipeline_path)
data_system_pipeline = importlib.util.module_from_spec(pipeline_spec)
pipeline_spec.loader.exec_module(data_system_pipeline)


class DataSystemTests(unittest.TestCase):
    def test_pipeline_loads_complete_equity_bundle(self):
        expected = object()

        with (
            patch.object(data_system_pipeline.os.path, "exists", return_value=True),
            patch.object(
                data_system_pipeline,
                "load_saved_equity_data",
                return_value=expected,
            ) as load,
            patch.object(data_system_pipeline, "build_and_save_dataset") as build,
        ):
            result = data_system_pipeline.prepare_equity_data()

        self.assertIs(result, expected)
        load.assert_called_once_with()
        build.assert_not_called()

    def test_pipeline_rebuilds_incomplete_equity_bundle(self):
        history = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")]})
        tickers = ["A", "B"]
        expected = object()

        with (
            patch.object(data_system_pipeline.os.path, "exists", return_value=False),
            patch.object(
                data_system_pipeline,
                "get_sp500_history",
                return_value=history,
            ),
            patch.object(
                data_system_pipeline,
                "get_sp500_tickers",
                return_value=tickers,
            ),
            patch.object(
                data_system_pipeline,
                "build_and_save_dataset",
                return_value=expected,
            ) as build,
        ):
            result = data_system_pipeline.prepare_equity_data()

        self.assertIs(result, expected)
        build.assert_called_once_with(history, tickers)

    def test_pipeline_prepares_rate_after_equity(self):
        prices = pd.DataFrame(
            {"A": [10.0]},
            index=pd.DatetimeIndex(["2026-08-18"]),
        )
        equity_data = (prices,)

        with (
            patch.object(
                data_system_pipeline,
                "prepare_equity_data",
                return_value=equity_data,
            ),
            patch.object(
                data_system_pipeline,
                "prepare_risk_free_rate",
            ) as prepare_rate,
        ):
            result = data_system_pipeline.run_pipeline()

        self.assertIs(result, equity_data)
        prepare_rate.assert_called_once_with("2008-01-01", "2026-08-18")

    def test_quality_mask_never_changes_past_after_future_anomaly(self):
        index = pd.date_range("2020-01-01", periods=5)
        prices = pd.DataFrame(
            {"A": [10.0, 25.0, 24.0, 60.0, 61.0]},
            index=index,
        )
        membership = pd.DataFrame(True, index=index, columns=prices.columns)

        quality, extremes, quarantine = build_point_in_time_quality_mask(
            prices,
            membership,
            max_abs_daily_return=1.0,
            max_extreme_daily_returns=1,
        )

        self.assertEqual(extremes["A"].tolist(), [False, True, False, True, False])
        self.assertEqual(
            quarantine["A"].tolist(),
            [False, False, False, True, True],
        )
        self.assertEqual(
            quality["A"].tolist(),
            [True, True, True, False, False],
        )

    def test_availability_combines_membership_price_and_quality(self):
        index = pd.date_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"A": [10.0, np.nan, 11.0]}, index=index)
        membership = pd.DataFrame({"A": [True, True, False]}, index=index)
        quality = pd.DataFrame({"A": [True, True, True]}, index=index)

        result = compute_availability(prices, membership, quality)

        self.assertEqual(result["A"].tolist(), [True, False, False])

    def test_price_matrix_does_not_forward_fill_asset_gap(self):
        index = pd.date_range("2020-01-01", periods=3)
        columns = pd.MultiIndex.from_product([["Adj Close"], ["A", "B"]])
        raw = pd.DataFrame(
            [[5.0, 10.0], [5.5, np.nan], [6.0, 11.0]],
            index=index,
            columns=columns,
        )

        result = get_price_matrix(raw)

        self.assertTrue(pd.isna(result.loc[index[1], "B"]))

    def test_download_requests_raw_and_adjusted_price_fields(self):
        index = pd.DatetimeIndex(["2020-01-01"])
        columns = pd.MultiIndex.from_tuples(
            [("Close", "A"), ("Adj Close", "A"), ("Volume", "A")]
        )
        downloaded = pd.DataFrame([[100.0, 80.0, 1_000.0]], index=index, columns=columns)

        with (
            patch.object(data_module.yf, "set_tz_cache_location"),
            patch.object(data_module.yf, "download", return_value=downloaded) as download,
        ):
            result = data_module.download_data(["A"])

        self.assertFalse(download.call_args.kwargs["auto_adjust"])
        pd.testing.assert_frame_equal(result, downloaded)

    def test_adjusted_price_and_stored_volume_preserve_raw_dollar_volume(self):
        index = pd.date_range("2020-01-01", periods=20)
        raw_close = np.linspace(100.0, 119.0, len(index))
        adjusted_close = raw_close * 0.8
        raw_volume = np.linspace(1_000.0, 1_950.0, len(index))
        columns = pd.MultiIndex.from_tuples(
            [("Close", "A"), ("Adj Close", "A"), ("Volume", "A")]
        )
        raw = pd.DataFrame(
            np.column_stack([raw_close, adjusted_close, raw_volume]),
            index=index,
            columns=columns,
        )

        prices = get_price_matrix(raw)
        volume = get_volume_matrix(raw)

        expected_dollar_volume = raw["Close"] * raw["Volume"]
        actual_dollar_volume = prices * volume
        pd.testing.assert_frame_equal(actual_dollar_volume, expected_dollar_volume)

        expected_liquidity = np.log1p(expected_dollar_volume.rolling(20).mean())
        actual_liquidity = compute_liquidity(prices, volume)
        pd.testing.assert_frame_equal(actual_liquidity, expected_liquidity)

    def test_zero_adjusted_price_does_not_create_infinite_volume(self):
        index = pd.DatetimeIndex(["2020-01-01"])
        columns = pd.MultiIndex.from_tuples(
            [("Close", "A"), ("Adj Close", "A"), ("Volume", "A")]
        )
        raw = pd.DataFrame([[100.0, 0.0, 1_000.0]], index=index, columns=columns)

        volume = get_volume_matrix(raw)

        self.assertTrue(volume.empty)
        self.assertFalse(np.isinf(volume.to_numpy()).any())

    def test_existing_risk_free_file_is_not_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dgs3mo.parquet")
            with open(path, "w", encoding="utf-8") as file:
                file.write("existence is enough for the pipeline")

            with (
                patch.object(risk_free_rate, "RISK_FREE_RATE_PATH", path),
                patch.object(risk_free_rate, "download_dgs3mo") as download,
            ):
                risk_free_rate.prepare_risk_free_rate("2008-01-01", "2020-01-01")

            download.assert_not_called()

    def test_missing_risk_free_file_is_downloaded_and_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dgs3mo.parquet")
            expected = pd.DataFrame(
                {"annual_rate_pct": [2.0]},
                index=pd.DatetimeIndex(["2020-01-01"], name="date"),
            )

            with (
                patch.object(risk_free_rate, "RISK_FREE_RATE_PATH", path),
                patch.object(
                    risk_free_rate,
                    "download_dgs3mo",
                    return_value=expected,
                ) as download,
            ):
                risk_free_rate.prepare_risk_free_rate("2008-01-01", "2020-01-01")

            download.assert_called_once_with("2008-01-01", "2020-01-01")
            pd.testing.assert_frame_equal(pd.read_parquet(path), expected)

    def test_delete_removes_only_declared_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "first.parquet")
            second = os.path.join(directory, "second.csv")
            untouched = os.path.join(directory, "untouched.txt")

            for path in (first, second, untouched):
                with open(path, "w", encoding="utf-8") as file:
                    file.write("data")

            with patch.object(delete, "GENERATED_DATA_PATHS", (first, second)):
                delete.delete_generated_data()

            self.assertFalse(os.path.exists(first))
            self.assertFalse(os.path.exists(second))
            self.assertTrue(os.path.exists(untouched))


if __name__ == "__main__":
    unittest.main()
