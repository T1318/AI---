import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("xgboost"), "当前环境未安装 xgboost")
class XGBoostTrainingTests(unittest.TestCase):
    def test_flatten_train_predict_save_and_load(self):
        from train_xgboost import flatten_windows, train_xgboost_model
        from load_forecasting.xgboost_model import XGBoostModel

        datasets = {
            "x_train": np.random.randn(24, 4, 3).astype("float32"),
            "y_train": np.random.randn(24, 2).astype("float32"),
            "x_valid": np.random.randn(8, 4, 3).astype("float32"),
            "y_valid": np.random.randn(8, 2).astype("float32"),
            "y_valid_raw": np.random.randn(8, 2).astype("float32"),
            "x_test": np.random.randn(4, 4, 3).astype("float32"),
            "y_test_raw": np.random.randn(4, 2).astype("float32"),
            "y_mean": 0.0,
            "y_std": 1.0,
        }
        self.assertEqual(flatten_windows(datasets["x_train"]).shape, (24, 12))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "xgb.joblib"
            model, result = train_xgboost_model(
                datasets,
                {"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
                output_path=path,
            )
            restored = XGBoostModel.load(path)
            prediction = restored.predict(flatten_windows(datasets["x_test"]))

        self.assertEqual(prediction.shape, (4, 2))
        self.assertIn("mape", result)
        self.assertIsNotNone(model.feature_importances_)

    def test_xgboost_tuning_result_is_saved_as_json(self):
        from train_xgboost import save_xgboost_result
        from load_forecasting.model_registry import MODEL_SPECS

        search_space = MODEL_SPECS["xgboost"]["search_space"]
        best_params = {
            name: spec.get("choices", [spec.get("low")])[0]
            for name, spec in search_space.items()
        }

        study = type(
            "Study",
            (),
            {
                "best_value": 9.5,
                "best_params": best_params,
            },
        )()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best.json"
            result = save_xgboost_result(study, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(result, loaded)
        self.assertEqual(result["model_type"], "xgboost")
        self.assertEqual(result["objective"], "nonzero_mape")


if __name__ == "__main__":
    unittest.main()
