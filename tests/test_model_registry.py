import unittest

import torch

from load_forecasting.model_registry import (
    DEFAULT_TRAINING_CONFIG,
    MODEL_SPECS,
    SUPPORTED_DEEP_MODEL_TYPES,
    SUPPORTED_MODEL_TYPES,
    build_model,
    default_model_config,
    sample_model_configs,
)
from train import default_config


class FirstChoiceTrial:
    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low


class ModelRegistryTests(unittest.TestCase):
    def test_registry_contains_all_models_with_one_config_source(self):
        self.assertEqual(len(SUPPORTED_DEEP_MODEL_TYPES), 12)
        self.assertEqual(set(SUPPORTED_MODEL_TYPES), set(SUPPORTED_DEEP_MODEL_TYPES) | {"xgboost"})
        self.assertEqual(set(MODEL_SPECS), set(SUPPORTED_MODEL_TYPES))
        for spec in MODEL_SPECS.values():
            self.assertIn("defaults", spec)
            self.assertIn("search_space", spec)
            self.assertIn("kind", spec)

    def test_every_deep_model_builds_from_default_and_sampled_config(self):
        trial = FirstChoiceTrial()
        x = torch.randn(2, 8, 8)
        for model_type in SUPPORTED_DEEP_MODEL_TYPES:
            with self.subTest(model=model_type):
                default = default_model_config(model_type, 4)
                default_model = build_model(model_type, 8, default)
                default_output = (
                    default_model(x, torch.randn(2, 4, 8))
                    if model_type == "LoadTransformer"
                    else default_model(x)
                )
                self.assertEqual(tuple(default_output.shape), (2, 4))
                sampled, training = sample_model_configs(trial, model_type, 4)
                sampled_model = build_model(model_type, 8, sampled)
                sampled_output = (
                    sampled_model(x, torch.randn(2, 4, 8))
                    if model_type == "LoadTransformer"
                    else sampled_model(x)
                )
                self.assertEqual(tuple(sampled_output.shape), (2, 4))
                self.assertEqual(set(training), set(DEFAULT_TRAINING_CONFIG))

    def test_default_training_config_has_no_flat_transformer_fields(self):
        config = default_config("LoadTransformer")

        self.assertFalse(hasattr(config, "d_model"))
        self.assertFalse(hasattr(config, "nhead"))
        self.assertFalse(hasattr(config, "layers"))
        self.assertEqual(config.training_config, DEFAULT_TRAINING_CONFIG)
        self.assertEqual(config.output, "outputs/LoadTransformer_weather.pt")

    def test_xgboost_uses_registry_defaults_and_search_space(self):
        spec = MODEL_SPECS["xgboost"]

        self.assertEqual(spec["kind"], "xgboost")
        self.assertIn("n_estimators", spec["defaults"])
        self.assertIn("learning_rate", spec["search_space"])


if __name__ == "__main__":
    unittest.main()
