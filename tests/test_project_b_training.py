import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from load_forecasting.data import WAVELET_COLUMNS
from train import (
    SAMPLING_ALL_TRAIN,
    SAMPLING_ROLLING_MONTHLY_CV,
    build_checkpoint,
    default_config,
    prepare_datasets,
)
from train_b import PROJECT_B_DATA, WAVELET_COLUMNS_B, default_config_b


class ProjectBTrainingTests(unittest.TestCase):
    def test_project_b_columns_remove_only_unavailable_inputs(self):
        self.assertEqual(len(WAVELET_COLUMNS), 22)
        self.assertEqual(len(WAVELET_COLUMNS_B), 20)
        self.assertEqual(
            WAVELET_COLUMNS_B,
            [
                column
                for column in WAVELET_COLUMNS
                if column not in {"PriChWDiffPress", "ChPower03"}
            ],
        )

    def test_project_b_config_changes_only_data_columns_and_output(self):
        config = default_config_b("LoadTransformerEncoderDecoder")
        all_train = default_config_b(
            "LoadTransformerEncoderDecoder", SAMPLING_ALL_TRAIN
        )

        self.assertEqual(config.data, PROJECT_B_DATA)
        self.assertEqual(config.wavelet_columns, WAVELET_COLUMNS_B)
        self.assertEqual(config.sampling_strategy, SAMPLING_ROLLING_MONTHLY_CV)
        self.assertEqual(config.output, "outputs/B_LoadTransformerEncoderDecoder.pt")
        self.assertEqual(
            all_train.output,
            "outputs/B_LoadTransformerEncoderDecoder_all_train.pt",
        )

    def test_project_a_default_still_uses_22_columns(self):
        config = default_config()

        self.assertEqual(config.wavelet_columns, WAVELET_COLUMNS)

    def test_prepare_datasets_passes_configured_columns_to_window_builder(self):
        config = default_config_b(
            "LoadTransformerHistoryOnly", SAMPLING_ALL_TRAIN
        )
        config.input_hours = 8
        config.horizon = 4
        frame = pd.DataFrame(
            {
                "timeStamp": pd.date_range("2023-04-01", periods=16, freq="h"),
                "TotalRealTimeLoad": 1.0,
            }
        )
        arrays = (
            np.zeros((2, 8, 106), dtype="float32"),
            np.zeros((2, 4, 8), dtype="float32"),
            np.ones((2, 4), dtype="float32"),
            [f"f{i}" for i in range(106)],
            [f"w{i}" for i in range(8)],
        )

        with patch("train.load_excel", return_value=frame):
            with patch(
                "train.build_windows_with_future_weather",
                return_value=arrays,
            ) as build:
                datasets = prepare_datasets(config)

        self.assertEqual(build.call_args.args[3], WAVELET_COLUMNS_B)
        self.assertEqual(datasets["input_dim"], 106)

    def test_checkpoint_records_project_b_columns(self):
        args = default_config_b("LoadTransformerHistoryOnly")
        checkpoint = build_checkpoint(
            model=torch.nn.Linear(1, 1),
            args=args,
            epoch=1,
            validation={
                "selection_metric": "nonzero_mape",
                "selection_value": 10.0,
                "valid_mse": 0.5,
                "mape_hours_used": 24,
                "mape_hours_zero_excluded": 0,
            },
            feature_names=[f"f{i}" for i in range(106)],
            x_mean=np.zeros(106),
            x_std=np.ones(106),
            y_mean=0.0,
            y_std=1.0,
        )

        self.assertEqual(checkpoint["wavelet_columns"], WAVELET_COLUMNS_B)
        self.assertEqual(checkpoint["input_dim"], 106)


if __name__ == "__main__":
    unittest.main()
