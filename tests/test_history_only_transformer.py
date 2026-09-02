import unittest

import torch

from load_forecasting.model_registry import (
    SUPPORTED_DEEP_MODEL_TYPES,
    build_model,
    default_model_config,
    get_model_spec,
)


class HistoryOnlyTransformerTests(unittest.TestCase):
    def test_history_only_transformer_is_registered_without_future_weather(self):
        self.assertIn("LoadTransformerHistoryOnly", SUPPORTED_DEEP_MODEL_TYPES)
        self.assertFalse(
            get_model_spec("LoadTransformerHistoryOnly")["uses_future_weather"]
        )
        self.assertTrue(get_model_spec("LoadTransformer")["uses_future_weather"])

    def test_history_only_transformer_predicts_24_hours_from_history(self):
        config = default_model_config("LoadTransformerHistoryOnly", horizon=24)
        model = build_model("LoadTransformerHistoryOnly", input_dim=116, model_config=config)

        output = model(torch.randn(2, 168, 116))

        self.assertEqual(tuple(output.shape), (2, 24))


if __name__ == "__main__":
    unittest.main()
