import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch
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
    from load_forecasting.LoadTransfromer import LoadTransformer
    from train import default_config, model_select, select_device, split_time_frames


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
        from data_processing.data_processing import denoise

        reconstructed = result[[
            "TotalRealTimeLoad_cA3",
            "TotalRealTimeLoad_cD3",
            "TotalRealTimeLoad_cD2",
            "TotalRealTimeLoad_cD1",
        ]].sum(axis=1)
        np.testing.assert_allclose(
            reconstructed,
            denoise(frame["TotalRealTimeLoad"].to_numpy()),
            atol=1e-5,
        )

    def test_wavelet_features_call_denoise_before_decompose(self):
        frame = pd.DataFrame(
            {
                "timeStamp": pd.date_range("2025-04-01", periods=64, freq="h"),
                "TotalRealTimeLoad": np.arange(64, dtype=float),
            }
        )
        calls = []

        def fake_denoise(signal):
            calls.append("denoise")
            return signal + 10

        def fake_decompose(signal):
            calls.append(("decompose", signal.copy()))
            components = {
                "cA3": signal,
                "cD3": np.zeros_like(signal),
                "cD2": np.zeros_like(signal),
                "cD1": np.zeros_like(signal),
            }
            return [], components

        with patch("load_forecasting.data.denoise", side_effect=fake_denoise):
            with patch("load_forecasting.data.decompose", side_effect=fake_decompose):
                result, _ = wavelet_feature_window(frame, ["TotalRealTimeLoad"])

        self.assertEqual(calls[0], "denoise")
        self.assertEqual(calls[1][0], "decompose")
        np.testing.assert_array_equal(calls[1][1], np.arange(64, dtype=float) + 10)
        np.testing.assert_array_equal(
            result["TotalRealTimeLoad"], np.arange(64, dtype=float) + 10
        )


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

    def test_build_windows_supports_hourly_and_daily_stride(self):
        frame = pd.DataFrame(
            {
                "timeStamp": pd.date_range("2025-04-01", periods=128, freq="h"),
                "TotalRealTimeLoad": np.arange(128, dtype=float),
            }
        )
        columns = ["TotalRealTimeLoad"]

        x_hourly, y_hourly, _ = build_windows(
            frame, 56, 24, columns, stride_hours=1
        )
        x_daily, y_daily, _ = build_windows(
            frame, 56, 24, columns, stride_hours=24
        )

        self.assertEqual(len(x_hourly), 49)
        self.assertEqual(len(x_daily), 3)
        np.testing.assert_array_equal(y_daily[:, 0], [56, 80, 104])

    @unittest.skipIf(torch is None, "当前环境未安装 torch")
    def test_time_split_keeps_targets_inside_each_period(self):
        frame = pd.DataFrame(
            {
                "timeStamp": pd.date_range("2025-04-01", periods=400, freq="h"),
                "TotalRealTimeLoad": np.arange(400, dtype=float),
            }
        )
        train_frame, valid_frame, test_frame = split_time_frames(frame, 56)
        columns = ["TotalRealTimeLoad"]

        _, y_train, _ = build_windows(train_frame, 56, 24, columns, 1)
        _, y_valid, _ = build_windows(valid_frame, 56, 24, columns, 24)
        _, y_test, _ = build_windows(test_frame, 56, 24, columns, 24)

        self.assertLess(y_train.max(), y_valid.min())
        self.assertLess(y_valid.max(), y_test.min())
        self.assertEqual(y_valid.min(), 280)
        self.assertEqual(y_test.min(), 340)

    @unittest.skipIf(torch is None, "当前环境未安装 torch")
    def test_default_config_uses_transformer_and_auto_device(self):
        config = default_config()

        self.assertEqual(config.model, "LoadTransformer")
        self.assertEqual(config.device, "auto")

    @unittest.skipIf(torch is None, "当前环境未安装 torch")
    def test_unknown_model_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知模型"):
            model_select(type("Args", (), {"model": "LSTM"})(), ["x"])

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
