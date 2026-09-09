import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from equipment_forecasting.test_a_data import (
    HISTORY_COLUMNS, build_equipment_forecast_windows, fit_target_stats,
    split_equipment_holdout,
)
from equipment_forecasting.test_a_model import (
    EquipmentForecastTransformer, TARGET_NAMES, grouped_huber_loss,
)
from load_forecasting.checkpoint import load_checkpoint
from train_equipment_a_july16 import (
    default_config_equipment, equipment_metrics, predict_equipment,
    train_equipment_model,
)


class TestAEquipmentTests(unittest.TestCase):
    def frame(self, timestamps):
        rng = np.random.default_rng(20)
        values = {name: rng.uniform(1, 20, len(timestamps)) for name in HISTORY_COLUMNS}
        return pd.DataFrame({"timeStamp": timestamps, **values})

    def test_shapes_and_gradients_for_both_inputs(self):
        self.assertEqual(len(TARGET_NAMES), 27)
        self.assertEqual(len(set(TARGET_NAMES)), 27)
        model = EquipmentForecastTransformer(d_model=16, num_layers=1, dropout=0)
        h = torch.randn(2, 168, 156, requires_grad=True)
        w = torch.randn(2, 24, 8, requires_grad=True)
        y = model(h, w)
        self.assertEqual(tuple(y.shape), (2, 24, 27))
        y.square().mean().backward()
        self.assertGreater(h.grad.abs().sum().item(), 0)
        self.assertGreater(w.grad.abs().sum().item(), 0)

    def test_mask_excludes_missing_targets_but_includes_zero_power(self):
        p = torch.ones(1, 24, 27, requires_grad=True)
        y = torch.zeros_like(p)
        mask = torch.ones_like(p, dtype=torch.bool)
        mask[..., 2] = False
        y[..., 2] = float("nan")
        loss = grouped_huber_loss(p, y, mask)
        self.assertAlmostEqual(loss.item(), 0.5)
        loss.backward()
        self.assertEqual(p.grad[..., 2].abs().sum().item(), 0)
        self.assertGreater(p.grad[..., 0].abs().sum().item(), 0)

    def test_target_stats_are_per_column_and_ignore_missing(self):
        y = np.ones((1, 24, 27), dtype=np.float32)
        y[..., 1] = 100
        m = np.ones_like(y, dtype=bool)
        m[:, 0, :] = False
        y[:, 0, :] = np.nan
        mean, std = fit_target_stats(y, m)
        self.assertEqual(mean.shape, (27,))
        self.assertEqual(mean[0], 1)
        self.assertEqual(mean[1], 100)
        np.testing.assert_array_equal(std, np.ones(27))

    def test_holdout_and_month_gaps_never_enter_training_windows(self):
        times = pd.date_range("2025-04-01", periods=216, freq="h").append(
            pd.date_range("2025-07-01", "2025-07-31 23:00", freq="h")
        ).append(pd.date_range("2025-09-01", periods=216, freq="h"))
        frame = self.frame(times)
        tr, va = split_equipment_holdout(frame)
        train = build_equipment_forecast_windows(tr, stride=24)
        valid = build_equipment_forecast_windows(va, stride=24)
        for key in ("timestamps", "history_timestamps"):
            dates = pd.DatetimeIndex(train[key].reshape(-1)).normalize()
            self.assertNotIn(pd.Timestamp("2025-07-16"), dates)
        joined = np.concatenate([train["history_timestamps"], train["timestamps"]], axis=1)
        self.assertTrue((np.diff(joined, axis=1) == np.timedelta64(1, "h")).all())
        self.assertEqual(valid["target"].shape, (1, 24, 27))
        self.assertEqual(valid["history"].shape, (1, 168, 156))
        self.assertEqual(valid["weather"].shape, (1, 24, 8))
        self.assertEqual(valid["timestamps"][0, 0], np.datetime64("2025-07-16"))

    def test_future_targets_cannot_change_model_inputs(self):
        frame = self.frame(pd.date_range("2025-07-09", periods=192, freq="h"))
        before = build_equipment_forecast_windows(frame, stride=24)
        frame.loc[168:, TARGET_NAMES] += 1000
        frame.loc[170, "ChPower01"] = np.nan
        after = build_equipment_forecast_windows(frame, stride=24)
        np.testing.assert_array_equal(before["history"], after["history"])
        np.testing.assert_array_equal(before["weather"], after["weather"])
        self.assertFalse(after["mask"][0, 2, 0])

    def test_metrics_count_zero_and_missing_separately(self):
        y = np.full((1, 24, 27), 10.0)
        p = y.copy()
        m = np.ones_like(y, dtype=bool)
        y[0, 0, 0] = 0
        p[0, 0, 0] = 5
        y[0, 1, 0] = np.nan
        m[0, 1, 0] = False
        table, _ = equipment_metrics(p, y, m)
        self.assertEqual(table.iloc[0].mape_hours_used, 22)
        self.assertEqual(table.iloc[0].mape_hours_zero_excluded, 1)
        self.assertAlmostEqual(table.iloc[0].mae, 5 / 23)

    def test_train_restore_csv_uses_best_epoch_with_27_target_scales(self):
        cfg = default_config_equipment()
        cfg.input_hours = 8
        cfg.epochs = 2
        cfg.device = "cpu"
        cfg.model_config.update(d_model=16, num_layers=1, dropout=0)
        rng = np.random.default_rng(4)
        stats = {"history_mean": np.zeros(3, np.float32), "history_std": np.ones(3, np.float32),
                 "weather_mean": np.zeros(8, np.float32), "weather_std": np.ones(8, np.float32),
                 "target_mean": np.arange(27, dtype=np.float32) + 10,
                 "target_std": np.arange(27, dtype=np.float32) + 1}
        def part(n):
            return {"history": rng.normal(size=(n, 8, 3)).astype("float32"),
                    "weather": rng.normal(size=(n, 24, 8)).astype("float32"),
                    "target": rng.normal(size=(n, 24, 27)).astype("float32"),
                    "target_raw": np.full((n, 24, 27), 10, np.float32),
                    "mask": np.ones((n, 24, 27), bool),
                    "timestamps": np.array([pd.date_range("2025-07-16", periods=24, freq="h")] * n),
                    "feature_names": ["x", "y", "z"], "weather_names": [f"w{i}" for i in range(8)]}
        data = {"train": part(2), "valid": part(1), "stats": stats}
        data["valid"]["target_raw"][0, 0, 0] = 0
        calls = []
        def score(pred, true, mask):
            table, summary = equipment_metrics(pred, true, mask)
            calls.append(1)
            if len(calls) <= 2:
                summary["selection_value"] = float(len(calls))
            return table, summary
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = tmp
            with patch("train_equipment_a_july16.equipment_metrics", side_effect=score):
                _, summary = train_equipment_model(cfg, data)
            ckpt = load_checkpoint(Path(tmp) / "equipment_best.pt", "cpu")
            self.assertEqual(ckpt["epoch"], 1)
            self.assertEqual(ckpt["target_names"], TARGET_NAMES)
            result = predict_equipment(ckpt, data["valid"]["history"], data["valid"]["weather"])
            csv = pd.read_csv(Path(tmp) / "july16_validation.csv")
            self.assertEqual(len(csv), 24 * 27)
            self.assertTrue(np.isnan(csv.iloc[0].ape_percent))
            np.testing.assert_allclose(csv.predicted_value, result.reshape(-1), atol=1e-5)
            self.assertEqual(len(pd.read_csv(Path(tmp) / "july16_metrics.csv")), 27)
            self.assertEqual(json.loads((Path(tmp) / "july16_summary.json").read_text())["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
