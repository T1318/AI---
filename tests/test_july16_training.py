from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from load_forecasting.checkpoint import load_checkpoint
from train import (
    SAMPLING_JULY16_HOLDOUT,
    split_july16_holdout_frames,
    train_model,
)
from train_a_july16 import (
    JULY16_DATA,
    WAVELET_COLUMNS_JULY16,
    default_config_july16,
)


class July16TrainingTests(unittest.TestCase):
    @staticmethod
    def _three_month_frame():
        timestamps = pd.DatetimeIndex([])
        for start, end in (
            ("2025-04-01", "2025-04-30 23:00:00"),
            ("2025-07-01", "2025-07-31 23:00:00"),
            ("2025-09-01", "2025-09-30 23:00:00"),
        ):
            timestamps = timestamps.append(pd.date_range(start, end, freq="h"))
        return pd.DataFrame({"timeStamp": timestamps, "TotalRealTimeLoad": 1.0})

    def test_default_config_uses_encoder_decoder_and_156_features(self):
        config = default_config_july16()

        self.assertEqual(config.model, "LoadTransformerEncoderDecoder")
        self.assertEqual(config.data, JULY16_DATA)
        self.assertEqual(config.sampling_strategy, SAMPLING_JULY16_HOLDOUT)
        self.assertEqual(config.wavelet_columns, WAVELET_COLUMNS_JULY16)
        self.assertEqual(len(config.wavelet_columns), 30)
        self.assertEqual(
            config.output,
            "outputs/TestA_July16_LoadTransformerEncoderDecoder.pt",
        )
        self.assertEqual(
            config.validation_output,
            "outputs/TestA_July16_LoadTransformerEncoderDecoder_validation.csv",
        )

    def test_columns_use_all_instantaneous_equipment_power_without_totals(self):
        self.assertIn("ChPower04", WAVELET_COLUMNS_JULY16)
        self.assertIn("PriChWPPower06", WAVELET_COLUMNS_JULY16)
        self.assertIn("CWPPower06", WAVELET_COLUMNS_JULY16)
        self.assertIn("CTPower04", WAVELET_COLUMNS_JULY16)
        self.assertNotIn("CoolingCapacityTotalDaily", WAVELET_COLUMNS_JULY16)
        self.assertFalse(any("PowerTotal" in name for name in WAVELET_COLUMNS_JULY16))

    def test_july16_is_validation_and_removed_from_training(self):
        train_frame, valid_frame = split_july16_holdout_frames(
            self._three_month_frame(), input_hours=168, horizon=24
        )
        holdout_date = pd.Timestamp("2025-07-16").date()

        self.assertFalse(
            train_frame["timeStamp"].dt.date.eq(holdout_date).any()
        )
        self.assertEqual(
            valid_frame["timeStamp"].iloc[0], pd.Timestamp("2025-07-09")
        )
        self.assertEqual(
            valid_frame["timeStamp"].iloc[168], pd.Timestamp("2025-07-16")
        )
        self.assertEqual(
            valid_frame["timeStamp"].iloc[-1], pd.Timestamp("2025-07-16 23:00:00")
        )

    def test_best_checkpoint_writes_validation_prediction_csv(self):
        rng = np.random.default_rng(0)
        config = default_config_july16()
        config.input_hours = 8
        config.horizon = 4
        config.epochs = 1
        config.device = "cpu"
        config.model_config.update(
            {
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "horizon": 4,
                "dropout": 0.0,
            }
        )
        config.training_config["batch_size"] = 2
        datasets = {
            "x_train": rng.normal(size=(4, 8, 3)).astype("float32"),
            "future_weather_train": rng.normal(size=(4, 4, 8)).astype("float32"),
            "y_train": rng.normal(size=(4, 4)).astype("float32"),
            "x_valid": rng.normal(size=(1, 8, 3)).astype("float32"),
            "future_weather_valid": rng.normal(size=(1, 4, 8)).astype("float32"),
            "y_valid": rng.normal(size=(1, 4)).astype("float32"),
            "y_valid_raw": np.array([[0.0, 10.0, 20.0, 30.0]], dtype="float32"),
            "validation_timestamps": np.array(
                [pd.date_range("2025-07-16", periods=4, freq="h")]
            ),
            "feature_names": ["f0", "f1", "f2"],
            "x_mean": np.zeros(3),
            "x_std": np.ones(3),
            "y_mean": 0.0,
            "y_std": 1.0,
            "future_weather_names": [f"w{i}" for i in range(8)],
            "future_weather_mean": np.zeros(8),
            "future_weather_std": np.ones(8),
            "weather_noise_std": np.array([1.0, 0.5, 0, 0, 0, 0, 0, 0]),
            "weather_noise_scale": np.array([1.0, 0.5, 0, 0, 0, 0, 0, 0]),
            "has_validation": True,
            "has_test": False,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config.output = str(Path(temp_dir) / "model.pt")
            config.validation_output = str(Path(temp_dir) / "validation.csv")
            _, result = train_model(config, datasets, save_checkpoint=True)
            detail = pd.read_csv(config.validation_output)
            checkpoint = load_checkpoint(config.output, "cpu")

        self.assertEqual(len(detail), 4)
        self.assertEqual(
            detail.columns.tolist(),
            [
                "timeStamp",
                "actual_load",
                "predicted_load",
                "error",
                "absolute_error",
                "ape_percent",
            ],
        )
        self.assertTrue(np.isnan(detail.loc[0, "ape_percent"]))
        self.assertEqual(checkpoint["sampling_strategy"], SAMPLING_JULY16_HOLDOUT)
        self.assertIn("best_valid_mape", result)


if __name__ == "__main__":
    unittest.main()
