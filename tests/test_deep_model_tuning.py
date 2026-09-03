import unittest

import torch

from load_forecasting.model_registry import (
    MODEL_SPECS,
    SUPPORTED_DEEP_MODEL_TYPES,
    build_model,
    get_model_spec,
    sample_model_configs,
)


class FirstChoiceTrial:
    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low


class DeepModelTuningTests(unittest.TestCase):
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
