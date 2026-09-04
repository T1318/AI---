from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from evaluate_test import (
    build_prediction_frame,
    build_test_windows,
    calculate_metrics,
    calculate_metrics_by_horizon,
    predict_test_windows,
    save_evaluation_outputs,
)
from load_forecasting.data import (
    TARGET_COLUMN,
    future_weather_frame,
    wavelet_feature_window,
)
from load_forecasting.model_registry import build_model, default_model_config


class EvaluateTestTests(unittest.TestCase):
    @staticmethod
    def _hourly_frame(periods=72):
        index = np.arange(periods, dtype=np.float32)
        return pd.DataFrame(
            {
                "timeStamp": pd.date_range("2026-09-01", periods=periods, freq="h"),
                "TotalRealTimeLoad": 100.0 + index,
                "OutdoorTdbin": 30.0 + index * 0.1,
                "OutdoorWetTemp": 24.0 + index * 0.05,
            }
        )

    def test_build_test_windows_uses_first_hours_as_context_and_stride_horizon(self):
        frame = self._hourly_frame()
        _, feature_names = wavelet_feature_window(
            frame.iloc[:64], [TARGET_COLUMN]
        )
        _, weather_names = future_weather_frame(frame.iloc[64:68])
        checkpoint = {
            "input_hours": 64,
            "horizon": 4,
            "wavelet_columns": [TARGET_COLUMN],
            "feature_names": feature_names,
            "future_weather_names": weather_names,
        }

        windows = build_test_windows(frame, checkpoint)

        self.assertEqual(windows["candidate_windows"], 2)
        self.assertEqual(windows["successful_windows"], 2)
        self.assertEqual(windows["skipped_windows"], 0)
        self.assertEqual(windows["history"].shape[0], 2)
        self.assertEqual(windows["timestamps"][0, 0], frame["timeStamp"].iloc[64])
        self.assertEqual(windows["timestamps"][1, 0], frame["timeStamp"].iloc[68])
        self.assertEqual(len(np.unique(windows["timestamps"].reshape(-1))), 8)

    def test_metrics_use_all_values_except_zero_mape_denominator(self):
        actual = np.array([0.0, 10.0], dtype=np.float32)
        predicted = np.array([5.0, 8.0], dtype=np.float32)

        result = calculate_metrics(predicted, actual)

        self.assertAlmostEqual(result["mae"], 3.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(14.5))
        self.assertAlmostEqual(result["mape"], 20.0)
        self.assertEqual(result["mape_hours_used"], 1)
        self.assertEqual(result["mape_hours_zero_excluded"], 1)

    def test_prediction_frame_and_horizon_metrics_keep_forecast_steps(self):
        timestamps = np.array(
            [pd.date_range("2026-09-01", periods=2, freq="h")], dtype=object
        )
        actual = np.array([[0.0, 10.0]], dtype=np.float32)
        predicted = np.array([[1.0, 8.0]], dtype=np.float32)

        detail = build_prediction_frame(timestamps, actual, predicted)
        by_horizon = calculate_metrics_by_horizon(predicted, actual)

        self.assertEqual(detail["forecast_step"].tolist(), [1, 2])
        self.assertTrue(np.isnan(detail.loc[0, "ape_percent"]))
        self.assertEqual(by_horizon["forecast_step"].tolist(), [1, 2])
        self.assertEqual(by_horizon.loc[0, "mape_hours_used"], 0)

    def test_three_transformers_can_predict_test_windows(self):
        history = np.random.default_rng(0).normal(size=(2, 8, 3)).astype("float32")
        weather = np.random.default_rng(1).normal(size=(2, 4, 8)).astype("float32")
        for model_type in (
            "LoadTransformer",
            "LoadTransformerHistoryOnly",
            "LoadTransformerEncoderDecoder",
        ):
            with self.subTest(model=model_type):
                config = default_model_config(model_type, horizon=4)
                config.update(
                    {"d_model": 16, "nhead": 4, "num_layers": 1, "dropout": 0.0}
                )
                model = build_model(model_type, 3, config)
                checkpoint = {
                    "model_type": model_type,
                    "model_config": config,
                    "input_dim": 3,
                    "model": model.state_dict(),
                    "x_mean": np.zeros(3),
                    "x_std": np.ones(3),
                    "future_weather_mean": np.zeros(8),
                    "future_weather_std": np.ones(8),
                    "y_mean": 10.0,
                    "y_std": 2.0,
                }

                prediction = predict_test_windows(
                    checkpoint, history, weather, torch.device("cpu"), batch_size=2
                )

                self.assertEqual(prediction.shape, (2, 4))

    def test_save_evaluation_outputs_creates_csv_json_and_png(self):
        timestamps = np.array(
            [pd.date_range("2026-09-01", periods=2, freq="h")], dtype=object
        )
        actual = np.array([[10.0, 12.0]], dtype=np.float32)
        predicted = np.array([[9.0, 13.0]], dtype=np.float32)
        detail = build_prediction_frame(timestamps, actual, predicted)
        overall = calculate_metrics(predicted, actual)
        by_horizon = calculate_metrics_by_horizon(predicted, actual)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_evaluation_outputs(detail, overall, by_horizon, Path(temp_dir))

            self.assertTrue((Path(temp_dir) / "predictions.csv").is_file())
            self.assertTrue((Path(temp_dir) / "metrics.json").is_file())
            self.assertTrue((Path(temp_dir) / "metrics_by_horizon.csv").is_file())
            self.assertTrue((Path(temp_dir) / "load_curve.png").is_file())


if __name__ == "__main__":
    unittest.main()
