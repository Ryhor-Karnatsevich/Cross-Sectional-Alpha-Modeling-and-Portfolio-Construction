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
import risk_free_rate
from data import compute_availability, get_price_matrix
from data_quality import build_point_in_time_quality_mask


class DataSystemTests(unittest.TestCase):
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
        columns = pd.MultiIndex.from_product([["Close"], ["A", "B"]])
        raw = pd.DataFrame(
            [[5.0, 10.0], [5.5, np.nan], [6.0, 11.0]],
            index=index,
            columns=columns,
        )

        result = get_price_matrix(raw)

        self.assertTrue(pd.isna(result.loc[index[1], "B"]))

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
