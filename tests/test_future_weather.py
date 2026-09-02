from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from load_forecasting.data import (
    build_latest_history_window,
    build_windows_with_future_weather,
    load_weather_forecast_csv,
)
from load_forecasting.model_registry import (
    build_model,
    default_model_config,
    get_model_spec,
)
from load_forecasting.checkpoint import load_checkpoint
from train import add_weather_noise, default_config, train_model


class FutureWeatherTests(unittest.TestCase):
    def _frame(self, periods=80):
        return pd.DataFrame(
            {
                "timeStamp": pd.date_range("2026-01-01", periods=periods, freq="h"),
                "TotalRealTimeLoad": np.arange(periods, dtype=float),
                "OutdoorTdbin": np.arange(periods, dtype=float) + 10,
                "OutdoorWetTemp": np.arange(periods, dtype=float) + 5,
            }
        )

    def test_window_returns_history_future_weather_and_matching_target(self):
        history, weather, target, history_names, weather_names = (
            build_windows_with_future_weather(
                self._frame(),
                input_hours=56,
                horizon=24,
                wavelet_columns=["TotalRealTimeLoad", "OutdoorTdbin"],
            )
        )

        self.assertEqual(history.shape, (1, 56, 16))
        self.assertEqual(weather.shape, (1, 24, 8))
        self.assertEqual(target.shape, (1, 24))
        self.assertEqual(weather_names[:2], ["OutdoorTdbin", "OutdoorWetTemp"])
        np.testing.assert_array_equal(weather[0, :, 0], np.arange(56, 80) + 10)
        np.testing.assert_array_equal(target[0], np.arange(56, 80))
        self.assertEqual(len(history_names), 16)

    def test_latest_history_returns_last_valid_timestamp(self):
        history, names, last_time = build_latest_history_window(
            self._frame(60), 56, ["TotalRealTimeLoad"]
        )

        self.assertEqual(history.shape, (1, 56, 11))
        self.assertEqual(last_time, pd.Timestamp("2026-01-03 11:00:00"))
        self.assertIn("TotalRealTimeLoad", names)

    def test_weather_csv_requires_24_continuous_aligned_hours(self):
        expected = pd.Timestamp("2026-02-01 01:00:00")
        forecast = pd.DataFrame(
            {
                "timeStamp": pd.date_range(expected, periods=24, freq="h"),
                "OutdoorTdbin": np.linspace(20, 30, 24),
                "OutdoorWetTemp": np.linspace(15, 25, 24),
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weather.csv"
            forecast.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
            values, names, timestamps = load_weather_forecast_csv(path, expected, 24)

            self.assertEqual(values.shape, (1, 24, 8))
            self.assertEqual(names[:2], ["OutdoorTdbin", "OutdoorWetTemp"])
            self.assertEqual(timestamps[0], expected)

            forecast.iloc[:-1].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "24"):
                load_weather_forecast_csv(path, expected, 24)

    def test_transformer_uses_future_weather(self):
        config = default_model_config("LoadTransformer", 24)
        model = build_model("LoadTransformer", 8, config)
        model.eval()
        history = torch.randn(2, 32, 8)
        weather_a = torch.zeros(2, 24, 8)
        weather_b = torch.ones(2, 24, 8)

        with torch.no_grad():
            output_a = model(history, weather_a)
            output_b = model(history, weather_b)

        self.assertEqual(tuple(output_a.shape), (2, 24))
        self.assertFalse(torch.allclose(output_a, output_b))
        self.assertTrue(get_model_spec("LoadTransformer")["uses_future_weather"])
        self.assertFalse(get_model_spec("lstm")["uses_future_weather"])

    def test_weather_noise_changes_only_temperature_columns(self):
        torch.manual_seed(0)
        weather = torch.zeros(2, 24, 8)
        noise_scale = torch.tensor([1.0, 0.5, 0, 0, 0, 0, 0, 0])

        noisy = add_weather_noise(weather, noise_scale)

        self.assertFalse(torch.equal(noisy[..., :2], weather[..., :2]))
        self.assertTrue(torch.equal(noisy[..., 2:], weather[..., 2:]))

    def test_weather_model_train_save_load_round_trip(self):
        rng = np.random.default_rng(0)
        datasets = {
            "x_train": rng.normal(size=(4, 8, 3)).astype("float32"),
            "y_train": rng.normal(size=(4, 4)).astype("float32"),
            "x_valid": rng.normal(size=(2, 8, 3)).astype("float32"),
            "y_valid": rng.normal(size=(2, 4)).astype("float32"),
            "x_test": rng.normal(size=(2, 8, 3)).astype("float32"),
            "y_test": rng.normal(size=(2, 4)).astype("float32"),
            "future_weather_train": rng.normal(size=(4, 4, 8)).astype("float32"),
            "future_weather_valid": rng.normal(size=(2, 4, 8)).astype("float32"),
            "future_weather_test": rng.normal(size=(2, 4, 8)).astype("float32"),
            "y_valid_raw": np.full((2, 4), 10.0, dtype="float32"),
            "y_test_raw": np.full((2, 4), 10.0, dtype="float32"),
            "feature_names": ["f0", "f1", "f2"],
            "input_dim": 3,
            "x_mean": np.zeros(3),
            "x_std": np.ones(3),
            "y_mean": 0.0,
            "y_std": 1.0,
            "future_weather_names": [
                "OutdoorTdbin", "OutdoorWetTemp", "hour_sin", "hour_cos",
                "weekday_sin", "weekday_cos", "month_sin", "month_cos",
            ],
            "future_weather_mean": np.zeros(8),
            "future_weather_std": np.ones(8),
            "weather_noise_std": np.array([1.0, 0.5, 0, 0, 0, 0, 0, 0]),
            "weather_noise_scale": np.array([1.0, 0.5, 0, 0, 0, 0, 0, 0]),
        }
        config = default_config("LoadTransformer")
        config.model_config.update(
            {"d_model": 16, "nhead": 4, "num_layers": 1,
             "horizon": 4, "dropout": 0.0, "future_weather_dim": 8}
        )
        config.horizon = 4
        config.input_hours = 8
        config.epochs = 1
        config.device = "cpu"
        config.training_config["batch_size"] = 2

        with tempfile.TemporaryDirectory() as temp_dir:
            config.output = str(Path(temp_dir) / "weather.pt")
            model, result = train_model(config, datasets, save_checkpoint=True)
            checkpoint = load_checkpoint(config.output, map_location="cpu")

        self.assertEqual(checkpoint["future_weather_names"][:2], [
            "OutdoorTdbin", "OutdoorWetTemp"
        ])
        self.assertEqual(checkpoint["model_config"]["future_weather_dim"], 8)
        self.assertIn("mape", result)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
