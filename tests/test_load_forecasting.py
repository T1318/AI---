import numpy as np
import pandas as pd
import unittest
try:
    import torch
except ModuleNotFoundError:
    torch = None

from load_forecasting.data import (
    build_latest_window,
    build_windows,
    clean_dataframe,
    resample_hourly,
    wavelet_feature_window,
)
if torch is not None:
    from load_forecasting.model import LoadTransformer
    from train import select_device


class LoadForecastingTests(unittest.TestCase):
    def test_clean_dataframe_converts_header_rows_and_invalid_values(self):
        raw = pd.DataFrame(
            [
                {"timeStamp": "时间", "TotalRealTimeLoad": "实时负荷"},
                {"timeStamp": "2025-04-01 00:00:00", "TotalRealTimeLoad": "-8888.7998"},
                {"timeStamp": "2025-04-01 01:00:00", "TotalRealTimeLoad": 20},
            ]
        )

        result = clean_dataframe(raw)

        self.assertEqual(len(result), 2)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["timeStamp"]))
        self.assertEqual(result["TotalRealTimeLoad"].isna().sum(), 0)
        self.assertEqual(result["TotalRealTimeLoad"].iloc[0], 20)

    def test_resample_hourly_averages_five_minute_values(self):
        frame = pd.DataFrame(
            {
                "timeStamp": pd.date_range("2025-04-01", periods=24, freq="5min"),
                "TotalRealTimeLoad": np.arange(24, dtype=float),
            }
        )

        result = resample_hourly(frame)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["TotalRealTimeLoad"].tolist(), [5.5, 17.5])

    def test_wavelet_feature_window_keeps_raw_and_four_components(self):
        frame = pd.DataFrame(
            {
                "timeStamp": pd.date_range("2025-04-01", periods=64, freq="h"),
                "TotalRealTimeLoad": np.arange(64, dtype=float),
            }
        )

        result, names = wavelet_feature_window(frame, ["TotalRealTimeLoad"])

        self.assertEqual(result.shape, (64, 11))
        self.assertEqual(names[:5], [
            "TotalRealTimeLoad",
            "TotalRealTimeLoad_cA3",
            "TotalRealTimeLoad_cD3",
            "TotalRealTimeLoad_cD2",
            "TotalRealTimeLoad_cD1",
        ])
        reconstructed = result[[
            "TotalRealTimeLoad_cA3",
            "TotalRealTimeLoad_cD3",
            "TotalRealTimeLoad_cD2",
            "TotalRealTimeLoad_cD1",
        ]].sum(axis=1)
        np.testing.assert_allclose(reconstructed, frame["TotalRealTimeLoad"], atol=1e-5)


    def test_build_windows_returns_24_step_targets(self):
        frame = pd.DataFrame(
            {
                "timeStamp": pd.date_range("2025-04-01", periods=80, freq="h"),
                "TotalRealTimeLoad": np.arange(80, dtype=float),
                "OutdoorTdbin": np.arange(80, dtype=float) + 10,
            }
        )

        x, y, feature_names = build_windows(
            frame,
            input_hours=56,
            horizon=24,
            wavelet_columns=["TotalRealTimeLoad", "OutdoorTdbin"],
        )

        self.assertEqual(x.shape, (1, 56, 16))
        self.assertEqual(y.shape, (1, 24))
        self.assertIn("hour_sin", feature_names)
        np.testing.assert_array_equal(y[0], np.arange(56, 80, dtype=float))

    def test_future_values_do_not_change_existing_input_window(self):
        frame = pd.DataFrame(
            {
                "timeStamp": pd.date_range("2025-04-01", periods=81, freq="h"),
                "TotalRealTimeLoad": np.arange(81, dtype=float),
                "OutdoorTdbin": np.arange(81, dtype=float) + 10,
            }
        )
        columns = ["TotalRealTimeLoad", "OutdoorTdbin"]
        original_x, _, _ = build_windows(frame, 56, 24, columns)
        changed = frame.copy()
        changed.loc[56:, columns] += 10000

        changed_x, _, _ = build_windows(changed, 56, 24, columns)

        np.testing.assert_allclose(original_x[0], changed_x[0])

    def test_build_latest_window_uses_last_input_hours(self):
        frame = pd.DataFrame(
            {
                "timeStamp": pd.date_range("2025-04-01", periods=60, freq="h"),
                "TotalRealTimeLoad": np.arange(60, dtype=float),
            }
        )

        x, names = build_latest_window(frame, 56, ["TotalRealTimeLoad"])

        self.assertEqual(x.shape, (1, 56, 11))
        self.assertEqual(x[0, 0, names.index("TotalRealTimeLoad")], 4)


    @unittest.skipIf(torch is None, "当前环境未安装 torch")
    def test_load_transformer_outputs_24_predictions(self):
        model = LoadTransformer(input_dim=8, d_model=16, nhead=4, num_layers=1, horizon=24)
        output = model(torch.randn(2, 5, 8))

        self.assertEqual(tuple(output.shape), (2, 24))

    @unittest.skipIf(torch is None, "当前环境未安装 torch")
    def test_auto_device_matches_cuda_availability(self):
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(str(select_device("auto")), expected)


if __name__ == "__main__":
    unittest.main()
