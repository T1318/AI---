import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from data_processing.data_processing import denoise
from equipment_forecasting.hourly_data import EVALUATION_MODE, prepare_hourly_datasets
from equipment_forecasting.test_a_data import HISTORY_COLUMNS
from equipment_forecasting.test_a_model import TARGET_NAMES
from load_forecasting.checkpoint import load_checkpoint
from load_forecasting.data import wavelet_feature_window
from load_forecasting.predict import restore_model
from train import validation_summary
from train_a_july16_hourly import default_config_hourly, train_hourly_model
from train_equipment_a_july16 import equipment_metrics, restore_equipment_model, train_equipment_model
from train_equipment_a_july16_hourly import default_config_equipment_hourly


class HourlyObservedUpdateTests(unittest.TestCase):
    def frame(self):
        times = pd.date_range("2025-04-01", periods=60, freq="h").append(
            pd.date_range("2025-07-14", "2025-07-18 23:00", freq="h")
        ).append(pd.date_range("2025-09-01", periods=60, freq="h"))
        rng = np.random.default_rng(19)
        return pd.DataFrame({"timeStamp": times, **{
            name: rng.uniform(5, 20, len(times)).astype(np.float32) for name in HISTORY_COLUMNS
        }})

    def test_first_level_reconstructs_and_legacy_default_keeps_three_levels(self):
        frame = self.frame().iloc[:24]
        features, names = wavelet_feature_window(frame, HISTORY_COLUMNS, level=1)
        self.assertEqual(features.shape, (24, 96))
        self.assertEqual(names[:3], ["TotalRealTimeLoad", "TotalRealTimeLoad_cA1", "TotalRealTimeLoad_cD1"])
        np.testing.assert_allclose(features.iloc[:, 1] + features.iloc[:, 2],
                                   denoise(frame.TotalRealTimeLoad, level=1), atol=1e-5)
        self.assertEqual(wavelet_feature_window(self.frame().iloc[:60], HISTORY_COLUMNS)[0].shape[-1], 156)

    def test_isolation_alignment_and_observed_history_update(self):
        frame = self.frame()
        config = default_config_hourly()
        before = prepare_hourly_datasets(config, task="load", frame=frame)
        v = before["valid"]
        self.assertEqual(v["history"].shape, (24, 24, 96))
        self.assertEqual(v["weather"].shape, (24, 1, 8))
        np.testing.assert_array_equal(v["timestamps"].reshape(-1),
                                      pd.date_range("2025-07-16", periods=24, freq="h").to_numpy())
        self.assertTrue((v["history_timestamps"] < v["timestamps"]).all())
        self.assertEqual(v["history_timestamps"][1, -1], np.datetime64("2025-07-16T00:00"))
        for key in ("history_timestamps", "timestamps"):
            self.assertFalse((pd.DatetimeIndex(before["train"][key].reshape(-1)).normalize()
                              == pd.Timestamp("2025-07-16")).any())
        for part in (before["train"], v):
            joined = np.concatenate([part["history_timestamps"], part["timestamps"]], axis=1)
            self.assertTrue((np.diff(joined, axis=1) == np.timedelta64(1, "h")).all())
        # 改变10:00小时真实传感器，不能影响对10:00及此前的预测历史。
        changed = frame.copy()
        changed.loc[changed.timeStamp == pd.Timestamp("2025-07-16 10:00"), TARGET_NAMES] += 100
        after = prepare_hourly_datasets(config, task="load", frame=changed)
        np.testing.assert_array_equal(v["history"][:11], after["valid"]["history"][:11])
        self.assertFalse(np.allclose(v["history"][11], after["valid"]["history"][11]))
        np.testing.assert_array_equal(v["weather"], after["valid"]["weather"])
        for name, value in before["stats"].items():
            np.testing.assert_array_equal(value, after["stats"][name])

    def test_history_imputation_does_not_see_target_hour_and_labels_not_filled(self):
        frame = self.frame()
        frame.loc[frame.timeStamp == pd.Timestamp("2025-07-15 23:00"), "ChPower01"] = np.nan
        cfg = default_config_equipment_hourly()
        before = prepare_hourly_datasets(cfg, frame=frame)
        modified = frame.copy()
        modified.loc[modified.timeStamp == pd.Timestamp("2025-07-16"), "ChPower01"] = 10000
        after = prepare_hourly_datasets(cfg, frame=modified)
        np.testing.assert_array_equal(before["valid"]["history"][0], after["valid"]["history"][0])
        modified.loc[modified.timeStamp == pd.Timestamp("2025-07-16 03:00"), "ChPower01"] = np.nan
        with_missing = prepare_hourly_datasets(cfg, frame=modified)
        self.assertFalse(with_missing["valid"]["mask"][3, 0, 0])
        self.assertTrue(np.isnan(with_missing["valid"]["target_raw"][3, 0, 0]))

    def test_load_best_checkpoint_and_24_row_output(self):
        cfg = default_config_hourly()
        cfg.device, cfg.epochs = "cpu", 2
        cfg.model_config.update(d_model=16, nhead=4, num_layers=1, dropout=0)
        frame = self.frame()
        frame.loc[frame.timeStamp == pd.Timestamp("2025-07-16"), "TotalRealTimeLoad"] = 0
        data = prepare_hourly_datasets(cfg, task="load", frame=frame)
        calls = []
        def selection(*args):
            result = validation_summary(*args)
            calls.append(1)
            result["selection_value"] = len(calls)  # 强制第一轮最佳，第二轮更差。
            return result
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = tmp
            with patch("train.validation_summary", side_effect=selection):
                _, report = train_hourly_model(cfg, data)
            ckpt = load_checkpoint(Path(tmp) / "best.pt", "cpu")
            model = restore_model(ckpt)
            with torch.no_grad():
                expected = model(torch.from_numpy(data["valid"]["history"]),
                                 torch.from_numpy(data["valid"]["weather"])).numpy()
            expected = expected * ckpt["y_std"] + ckpt["y_mean"]
            csv = pd.read_csv(Path(tmp) / "july16_validation.csv")
            self.assertEqual(len(csv), 24)
            self.assertTrue(np.isnan(csv.ape_percent.iloc[0]))
            np.testing.assert_allclose(csv.predicted_load, expected.ravel(), atol=1e-5)
            self.assertEqual(ckpt["epoch"], 1)
            self.assertEqual(ckpt["wavelet_level"], 1)
            self.assertEqual(ckpt["input_dim"], 96)
            self.assertEqual(ckpt["horizon"], 1)
            self.assertEqual(report["evaluation_mode"], EVALUATION_MODE)
            self.assertEqual(len(pd.read_csv(Path(tmp) / "july16_metrics.csv")), 1)
            self.assertEqual(json.loads((Path(tmp) / "july16_summary.json").read_text())["validation_hours"], 24)

    def test_equipment_best_checkpoint_and_648_row_output(self):
        cfg = default_config_equipment_hourly()
        cfg.device, cfg.epochs = "cpu", 2
        cfg.model_config.update(d_model=16, nhead=4, num_layers=1, dropout=0)
        data = prepare_hourly_datasets(cfg, frame=self.frame())
        calls = []
        def selection(*args):
            table, report = equipment_metrics(*args)
            calls.append(1)
            if len(calls) <= 2:
                report["selection_value"] = len(calls)
            return table, report
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = tmp
            with patch("train_equipment_a_july16.equipment_metrics", side_effect=selection):
                _, report = train_equipment_model(cfg, data)
            ckpt = load_checkpoint(Path(tmp) / "best.pt", "cpu")
            model = restore_equipment_model(ckpt)
            self.assertEqual(ckpt["epoch"], 1)
            self.assertEqual(ckpt["wavelet_level"], 1)
            self.assertEqual(ckpt["model_config"]["history_dim"], 96)
            self.assertEqual(tuple(model(torch.from_numpy(data["valid"]["history"]),
                                        torch.from_numpy(data["valid"]["weather"])).shape), (24, 1, 27))
            self.assertEqual(len(pd.read_csv(Path(tmp) / "july16_validation.csv")), 648)
            self.assertEqual(len(pd.read_csv(Path(tmp) / "july16_metrics.csv")), 27)
            self.assertEqual(report["evaluation_mode"], EVALUATION_MODE)


if __name__ == "__main__":
    unittest.main()
