import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from load_forecasting.LoadTransfromer import LoadTransformer
from load_forecasting.model_registry import build_model
from load_forecasting.predict import restore_model
from train import (
    build_checkpoint,
    default_config,
    metrics,
    model_config_from_args,
    save_best_checkpoint,
    validation_summary,
)


class TrainingCheckpointTests(unittest.TestCase):
    def test_validation_summary_selects_nonzero_mape_not_mse(self):
        datasets = {
            "y_mean": 0.0,
            "y_std": 1.0,
            "y_valid_raw": np.array([[0.0, 10.0]]),
        }

        summary = validation_summary(
            np.array([[5.0, 8.0]]),
            np.array([[0.0, 10.0]]),
            datasets,
        )

        self.assertAlmostEqual(summary["valid_mse"], 14.5)
        self.assertEqual(summary["valid_mape"], 20.0)
        self.assertEqual(summary["selection_metric"], "nonzero_mape")
        self.assertEqual(summary["selection_value"], 20.0)

    def test_metrics_exclude_zero_only_from_mape(self):
        result = metrics(np.array([5.0, 8.0]), np.array([0.0, 10.0]))

        self.assertEqual(result["mae"], 3.5)
        self.assertAlmostEqual(result["rmse"], math.sqrt(14.5))
        self.assertEqual(result["mape"], 20.0)
        self.assertEqual(result["mape_hours_used"], 1)
        self.assertEqual(result["mape_hours_zero_excluded"], 1)

    def test_metrics_return_nan_mape_when_all_targets_are_zero(self):
        result = metrics(np.array([1.0, 2.0]), np.array([0.0, 0.0]))

        self.assertTrue(math.isnan(result["mape"]))
        self.assertEqual(result["mape_hours_used"], 0)
        self.assertEqual(result["mape_hours_zero_excluded"], 2)

    def test_checkpoint_contains_model_type_config_and_best_epoch(self):
        args = default_config()
        feature_names = ["f0", "f1"]
        model_config = model_config_from_args(args)
        model = build_model(args.model, len(feature_names), model_config)

        checkpoint = build_checkpoint(
            model=model,
            args=args,
            epoch=3,
            validation={
                "selection_metric": "nonzero_mape",
                "selection_value": 12.5,
                "valid_mse": 0.25,
                "mape_hours_used": 10,
                "mape_hours_zero_excluded": 2,
            },
            feature_names=feature_names,
            x_mean=np.zeros(2),
            x_std=np.ones(2),
            y_mean=0.0,
            y_std=1.0,
        )

        self.assertEqual(checkpoint["checkpoint_version"], 1)
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["selection_metric"], "nonzero_mape")
        self.assertEqual(checkpoint["best_valid_mape"], 12.5)
        self.assertEqual(checkpoint["valid_mse"], 0.25)
        self.assertEqual(checkpoint["model_type"], "LoadTransformer")
        self.assertEqual(checkpoint["model_config"], model_config)
        self.assertEqual(checkpoint["input_dim"], 2)

    def test_every_validation_improvement_saves_checkpoint(self):
        with patch("train.torch.save") as save:
            best, saved = save_best_checkpoint("model.pt", 0.8, float("inf"), {"epoch": 1})
            best, saved_again = save_best_checkpoint("model.pt", 0.6, best, {"epoch": 2})
            final_best, not_saved = save_best_checkpoint("model.pt", 0.7, best, {"epoch": 3})

        self.assertTrue(saved)
        self.assertTrue(saved_again)
        self.assertFalse(not_saved)
        self.assertEqual(final_best, 0.6)
        self.assertEqual(save.call_count, 2)

    def test_prediction_restores_non_default_transformer_from_checkpoint(self):
        config = {
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "horizon": 12,
            "dropout": 0.0,
        }
        original = build_model("LoadTransformer", 7, config)
        checkpoint = {
            "model_type": "LoadTransformer",
            "model_config": config,
            "input_dim": 7,
            "model": original.state_dict(),
        }

        restored = restore_model(checkpoint)
        output = restored(torch.randn(2, 24, 7))

        self.assertIsInstance(restored, LoadTransformer)
        self.assertEqual(tuple(output.shape), (2, 12))

    def test_small_train_save_load_predict_round_trip(self):
        args = default_config()
        args.model_config["d_model"] = 16
        args.model_config["nhead"] = 4
        args.model_config["num_layers"] = 1
        args.horizon = 4
        args.model_config["horizon"] = 4
        args.model_config["dropout"] = 0.0
        args.input_hours = 8
        feature_names = ["f0", "f1", "f2"]
        model = build_model(args.model, len(feature_names), model_config_from_args(args))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(2, 8, 3)
        y = torch.randn(2, 4)

        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
        checkpoint = build_checkpoint(
            model, args, 1,
            {
                "selection_metric": "nonzero_mape",
                "selection_value": 10.0,
                "valid_mse": 0.5,
                "mape_hours_used": 8,
                "mape_hours_zero_excluded": 0,
            },
            feature_names,
            np.zeros(3), np.ones(3), 0.0, 1.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            _, saved = save_best_checkpoint(path, 0.5, float("inf"), checkpoint)
            loaded = torch.load(path, map_location="cpu")
            restored = restore_model(loaded)
            prediction = restored(torch.randn(1, 8, 3))

        self.assertTrue(saved)
        self.assertEqual(tuple(prediction.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
