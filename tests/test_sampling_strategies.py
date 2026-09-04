from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from load_forecasting.checkpoint import load_checkpoint
from search.tune_deep_models import default_tuning_config
from train import (
    SAMPLING_ALL_TRAIN,
    SAMPLING_ROLLING_MONTHLY_CV,
    cv_summary,
    default_config,
    prepare_datasets,
    run_cross_validation,
    split_final_evaluation_frames,
    split_rolling_monthly_folds,
    split_sampling_frames,
    train_model,
)


class SamplingStrategyTests(unittest.TestCase):
    @staticmethod
    def _frame():
        return pd.DataFrame(
            {
                "timeStamp": pd.date_range(
                    "2025-04-01", "2025-10-20 23:00:00", freq="h"
                ),
                "TotalRealTimeLoad": 1.0,
            }
        )

    def test_all_train_strategy_uses_every_row_without_holdouts(self):
        frame = self._frame()

        train_frame, valid_frame, test_frame = split_sampling_frames(
            frame, input_hours=168, strategy=SAMPLING_ALL_TRAIN
        )

        self.assertEqual(len(train_frame), len(frame))
        self.assertIsNone(valid_frame)
        self.assertIsNone(test_frame)

    def test_rolling_strategy_builds_three_expanding_month_folds(self):
        frame = self._frame()

        folds = split_rolling_monthly_folds(frame, input_hours=168)

        self.assertEqual([fold["name"] for fold in folds], ["fold_1", "fold_2", "fold_3"])
        expected = [
            ("2025-06-30 23:00:00", "2025-07-01", "2025-07-31 23:00:00"),
            ("2025-07-31 23:00:00", "2025-08-01", "2025-08-31 23:00:00"),
            ("2025-08-31 23:00:00", "2025-09-01", "2025-09-30 23:00:00"),
        ]
        for fold, (train_end, valid_start, valid_end) in zip(folds, expected):
            self.assertEqual(fold["train"]["timeStamp"].iloc[0], pd.Timestamp("2025-04-01"))
            self.assertEqual(fold["train"]["timeStamp"].iloc[-1], pd.Timestamp(train_end))
            self.assertEqual(fold["valid"]["timeStamp"].iloc[168], pd.Timestamp(valid_start))
            self.assertEqual(fold["valid"]["timeStamp"].iloc[-1], pd.Timestamp(valid_end))

    def test_final_evaluation_trains_through_september_and_tests_october(self):
        train_frame, test_frame = split_final_evaluation_frames(
            self._frame(), input_hours=168
        )

        self.assertEqual(train_frame["timeStamp"].iloc[0], pd.Timestamp("2025-04-01"))
        self.assertEqual(
            train_frame["timeStamp"].iloc[-1], pd.Timestamp("2025-09-30 23:00:00")
        )
        self.assertEqual(test_frame["timeStamp"].iloc[168], pd.Timestamp("2025-10-01"))

    def test_cv_summary_uses_mean_mape_and_median_best_epoch(self):
        summary = cv_summary(
            [
                {"best_valid_mape": 30.0, "best_epoch": 10},
                {"best_valid_mape": 20.0, "best_epoch": 30},
                {"best_valid_mape": 40.0, "best_epoch": 20},
            ]
        )

        self.assertEqual(summary["mean_valid_mape"], 30.0)
        self.assertEqual(summary["median_best_epoch"], 20)

    def test_default_config_selects_strategy_and_output_name(self):
        monthly = default_config(
            "LoadTransformerHistoryOnly", SAMPLING_ROLLING_MONTHLY_CV
        )
        all_train = default_config(
            "LoadTransformerHistoryOnly", SAMPLING_ALL_TRAIN
        )

        self.assertEqual(monthly.sampling_strategy, SAMPLING_ROLLING_MONTHLY_CV)
        self.assertEqual(all_train.sampling_strategy, SAMPLING_ALL_TRAIN)
        self.assertEqual(
            all_train.output, "outputs/LoadTransformerHistoryOnly_all_train.pt"
        )

    def test_tuning_always_uses_rolling_monthly_cv_strategy(self):
        config = default_tuning_config("LoadTransformerHistoryOnly")

        self.assertEqual(config.sampling_strategy, SAMPLING_ROLLING_MONTHLY_CV)
        self.assertIn("rolling_cv", config.study_name)
        self.assertIn("rolling_cv", config.study_path)
        self.assertIn("rolling_cv", config.result_path)

    def test_prepare_rolling_strategy_returns_three_folds_and_final_evaluation(self):
        config = default_config(
            "LoadTransformerHistoryOnly", SAMPLING_ROLLING_MONTHLY_CV
        )
        config.input_hours = 8
        config.horizon = 4

        def fake_windows(frame, input_hours, horizon, columns, stride_hours):
            count = 2 if stride_hours == 1 else 1
            return (
                np.zeros((count, input_hours, 3), dtype="float32"),
                np.zeros((count, horizon, 8), dtype="float32"),
                np.ones((count, horizon), dtype="float32"),
                ["f0", "f1", "f2"],
                [f"w{i}" for i in range(8)],
            )

        with patch("train.load_excel", return_value=self._frame()):
            with patch(
                "train.build_windows_with_future_weather",
                side_effect=fake_windows,
            ) as build:
                prepared = prepare_datasets(config)

        self.assertEqual(len(prepared["folds"]), 3)
        self.assertEqual(
            [fold["name"] for fold in prepared["folds"]],
            ["fold_1", "fold_2", "fold_3"],
        )
        self.assertFalse(prepared["final_evaluation"]["has_validation"])
        self.assertTrue(prepared["final_evaluation"]["has_test"])
        self.assertEqual(build.call_count, 8)

    def test_all_train_saves_final_epoch_without_validation_metrics(self):
        rng = np.random.default_rng(0)
        config = default_config(
            "LoadTransformerHistoryOnly", SAMPLING_ALL_TRAIN
        )
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
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config.output = str(Path(temp_dir) / "all_train.pt")
            _, result = train_model(config, datasets, save_checkpoint=True)
            checkpoint = load_checkpoint(config.output, "cpu")

        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["selection_metric"], "final_epoch")
        self.assertIsNone(checkpoint["best_valid_mape"])
        self.assertEqual(checkpoint["sampling_strategy"], SAMPLING_ALL_TRAIN)
        self.assertIn("train_mse", result)

    def test_cross_validation_runs_three_folds_then_one_final_evaluation(self):
        config = default_config(
            "LoadTransformerHistoryOnly", SAMPLING_ROLLING_MONTHLY_CV
        )
        prepared = {
            "folds": [{"name": f"fold_{i}"} for i in range(1, 4)],
            "final_evaluation": {"name": "october"},
        }
        fold_results = [
            (object(), {"best_valid_mape": 30.0, "best_epoch": 10}),
            (object(), {"best_valid_mape": 20.0, "best_epoch": 30}),
            (object(), {"best_valid_mape": 40.0, "best_epoch": 20}),
        ]
        final_result = (object(), {"mape": 25.0})

        with tempfile.TemporaryDirectory() as temp_dir:
            config.output = str(Path(temp_dir) / "rolling.pt")
            with patch(
                "train.train_model", side_effect=fold_results + [final_result]
            ) as train:
                model, summary = run_cross_validation(config, prepared)

            self.assertTrue((Path(temp_dir) / "rolling_cv.json").is_file())

        self.assertEqual(train.call_count, 4)
        self.assertFalse(train.call_args_list[0].kwargs["save_checkpoint"])
        self.assertTrue(train.call_args_list[3].kwargs["save_checkpoint"])
        self.assertEqual(train.call_args_list[3].args[0].epochs, 20)
        self.assertEqual(summary["mean_valid_mape"], 30.0)
        self.assertEqual(summary["october_test"]["mape"], 25.0)
        self.assertIs(model, final_result[0])


if __name__ == "__main__":
    unittest.main()
