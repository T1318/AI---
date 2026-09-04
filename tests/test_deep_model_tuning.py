import unittest
from argparse import Namespace
from unittest.mock import patch

import torch

from search.tune_deep_models import create_objective

from load_forecasting.model_registry import (
    MODEL_SPECS,
    SUPPORTED_DEEP_MODEL_TYPES,
    build_model,
    get_model_spec,
    sample_model_configs,
)


class FirstChoiceTrial:
    def __init__(self):
        self.user_attrs = {}

    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low

    def report(self, value, step):
        pass

    def should_prune(self):
        return False

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class DeepModelTuningTests(unittest.TestCase):
    def test_objective_averages_three_independent_fold_results(self):
        trial = FirstChoiceTrial()
        args = Namespace(
            model="LoadTransformerHistoryOnly",
            horizon=4,
            epochs=1,
            device="cpu",
        )
        prepared = {"folds": [{"name": f"fold_{i}"} for i in range(1, 4)]}
        fold_results = [
            {"best_valid_mape": 30.0, "best_epoch": 10},
            {"best_valid_mape": 20.0, "best_epoch": 30},
            {"best_valid_mape": 40.0, "best_epoch": 20},
        ]

        with patch(
            "search.tune_deep_models.train_trial_fold",
            side_effect=fold_results,
        ) as train_fold:
            value = create_objective(prepared, args)(trial)

        self.assertEqual(value, 30.0)
        self.assertEqual(train_fold.call_count, 3)
        self.assertEqual(trial.user_attrs["best_epoch"], 20)
        self.assertEqual(trial.user_attrs["fold_best_epochs"], [10, 30, 20])

    def test_every_deep_model_has_search_space(self):
        self.assertEqual(
            {name for name, spec in MODEL_SPECS.items() if spec["kind"] == "deep"},
            set(SUPPORTED_DEEP_MODEL_TYPES),
        )

    def test_each_search_space_builds_valid_24_hour_model(self):
        trial = FirstChoiceTrial()
        x = torch.randn(2, 8, 8)

        for model_type in SUPPORTED_DEEP_MODEL_TYPES:
            with self.subTest(model=model_type):
                model_config, training_config = sample_model_configs(
                    trial, model_type, horizon=4
                )
                model = build_model(model_type, 8, model_config)
                output = (
                    model(x, torch.randn(2, 4, 8))
                    if get_model_spec(model_type)["uses_future_weather"]
                    else model(x)
                )
                self.assertEqual(tuple(output.shape), (2, 4))
                self.assertNotIn("input_hours", MODEL_SPECS[model_type]["search_space"])
                self.assertIn(training_config["optimizer"], ("Adam", "AdamW"))


if __name__ == "__main__":
    unittest.main()
