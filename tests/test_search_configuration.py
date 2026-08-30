from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from search import data_pipeline
from load_forecasting import model_registry


class SearchConfigurationTests(unittest.TestCase):
    def test_invalid_sensor_values_are_replaced_with_nan(self):
        values = pd.DataFrame(
            {
                "sensor": [1.0, -8888.7998, -9999.0, -99999.0, np.inf, -np.inf, 1.5e14]
            }
        )

        result, invalid_count = data_pipeline._mask_invalid_values(values)

        self.assertEqual(invalid_count, 6)
        self.assertEqual(result["sensor"].notna().sum(), 1)

    def test_cache_path_contains_cleaning_version(self):
        path = data_pipeline.get_cache_path("A")

        self.assertIn("v2", Path(path).stem)

    def test_default_project_paths_exist(self):
        self.assertTrue(Path(data_pipeline.DATA_DIR_DEFAULT).is_dir())

    def test_transformer_registry_loads_repository_model(self):
        model = model_registry.build_model(
            "LoadTransformer", 8,
            model_registry.default_model_config("LoadTransformer", 4),
        )

        self.assertEqual(type(model).__name__, "LoadTransformer")

    def test_requirements_include_search_dependencies(self):
        requirements = Path("data_processing/requirements.txt").read_text(encoding="utf-8").lower()

        for package in ("openpyxl", "optuna", "xgboost", "joblib"):
            self.assertIn(package, requirements)


if __name__ == "__main__":
    unittest.main()
