import unittest
from unittest.mock import patch

import numpy as np

from search import data_interface


class LoadXgboostDataTests(unittest.TestCase):
    @patch("search.data_interface.data_pipeline.build_tensors")
    @patch("search.data_interface.data_pipeline.load_project_hourly")
    def test_loads_flat_windows_from_project_pipeline(self, load_hourly, build_tensors):
        hourly = object()
        load_hourly.return_value = hourly
        build_tensors.return_value = {
            "x_train": np.zeros((2, 24)),
            "y_train": np.zeros(2),
            "x_val": np.zeros((1, 24)),
            "y_val": np.zeros(1),
        }

        result = data_interface.load_xgboost_data(
            project="B", data_dir="data", seq_len=24, train_ratio=0.75
        )

        load_hourly.assert_called_once_with("B", data_dir="data")
        build_tensors.assert_called_once_with(
            hourly, 24, "flat", train_ratio=0.75
        )
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
