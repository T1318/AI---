from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from equipment_forecasting.data import load_control_plan_csv
from equipment_forecasting.predict import prediction_frame
from train_2 import (
    default_config,
    train_equipment_model,
)
from load_forecasting.checkpoint import load_checkpoint


class EquipmentTrainingTests(unittest.TestCase):
    def _datasets(self):
        rng = np.random.default_rng(0)
        names = [f"target_{i}" for i in range(28)]
        return {
            "x_train": rng.normal(size=(4, 8, 3)).astype("float32"),
            "plan_train": rng.normal(size=(4, 4, 24)).astype("float32"),
            "y_train": rng.normal(size=(4, 4, 28)).astype("float32"),
            "mask_train": np.ones((4, 4, 28), dtype="float32"),
            "x_valid": rng.normal(size=(2, 8, 3)).astype("float32"),
            "plan_valid": rng.normal(size=(2, 4, 24)).astype("float32"),
            "y_valid": rng.normal(size=(2, 4, 28)).astype("float32"),
            "y_valid_raw": np.full((2, 4, 28), 10.0, dtype="float32"),
            "mask_valid": np.ones((2, 4, 28), dtype="float32"),
            "x_test": rng.normal(size=(2, 8, 3)).astype("float32"),
            "plan_test": rng.normal(size=(2, 4, 24)).astype("float32"),
            "plan_test": rng.normal(size=(2, 4, 24)).astype("float32"),
            "y_test_raw": np.full((2, 4, 28), 10.0, dtype="float32"),
            "mask_test": np.ones((2, 4, 28), dtype="float32"),
            "history_names": ["h0", "h1", "h2"],
            "target_names": names,
            "plan_names": [f"plan_{i}" for i in range(24)],
            "x_mean": np.zeros(3), "x_std": np.ones(3),
            "plan_mean": np.zeros(24), "plan_std": np.ones(24),
            "target_mean": np.zeros(28), "target_std": np.ones(28),
        }

    def test_small_training_saves_equipment_checkpoint(self):
        config = default_config()
        config.model_config.update(
            {"d_model": 16, "nhead": 4, "num_layers": 1, "dropout": 0.0}
        )
        config.epochs = 1
        config.device = "cpu"
        config.training_config["batch_size"] = 2

        with tempfile.TemporaryDirectory() as temp_dir:
            config.output = str(Path(temp_dir) / "equipment.pt")
            model, result = train_equipment_model(config, self._datasets())
            checkpoint = load_checkpoint(config.output, "cpu")

        self.assertEqual(checkpoint["task"], "equipment_response")
        self.assertEqual(len(checkpoint["target_names"]), 28)
        self.assertIn("groups", result)
        self.assertIsNotNone(model)

    def test_control_plan_csv_and_prediction_frame(self):
        start = pd.Timestamp("2026-09-01 01:00:00")
        frame = pd.DataFrame({"timeStamp": pd.date_range(start, periods=24, freq="h")})
        frame["TotalRealTimeLoad"] = 500.0
        frame["OutdoorTdbin"] = 32.0
        frame["OutdoorWetTemp"] = 26.0
        for i in range(1, 4):
            frame[f"ChillerOn0{i}"] = 1
            frame[f"ChChWTempSupplySetPoint0{i}"] = 7.0
            frame[f"PriChWPVSDFreq0{i}"] = 40.0
            frame[f"CWPVSDFreq0{i}"] = 40.0
            frame[f"CTVSDFreq0{i}"] = 40.0

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.csv"
            frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
            plan, names, timestamps, masks = load_control_plan_csv(path, start, 24)

        self.assertEqual(plan.shape, (1, 24, 24))
        self.assertEqual(masks.shape, (1, 24, 28))
        output = prediction_frame(
            timestamps, np.ones((24, 28)), names_override=None
        )
        self.assertIn("COP01", output.columns)
        self.assertEqual(len(output), 24)


if __name__ == "__main__":
    unittest.main()
