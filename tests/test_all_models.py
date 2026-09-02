import unittest

import torch

from load_forecasting.model_registry import (
    SUPPORTED_DEEP_MODEL_TYPES,
    build_model,
    default_model_config,
)
from load_forecasting.predict import restore_model


DEEP_MODELS = [
    "LoadTransformer",
    "LoadTransformerHistoryOnly",
    "lstm",
    "gru",
    "bilstm",
    "seq2seq",
    "kan",
    "kan_lstm",
    "cnn_lstm",
    "cnn_lstm_attention",
    "cnn_kan",
    "cnn_kan_attention",
]


class AllModelTests(unittest.TestCase):
    def test_supported_model_types_include_all_deep_models(self):
        self.assertEqual(set(SUPPORTED_DEEP_MODEL_TYPES), set(DEEP_MODELS))

    def test_all_models_forward_backward_and_restore(self):
        x = torch.randn(2, 8, 8)
        target = torch.randn(2, 4)

        for model_type in DEEP_MODELS:
            with self.subTest(model=model_type):
                config = default_model_config(model_type, horizon=4)
                model = build_model(model_type, 8, config)
                weather = torch.randn(2, 4, 8)
                output = (
                    model(x, weather)
                    if model_type == "LoadTransformer"
                    else model(x)
                )
                self.assertEqual(tuple(output.shape), (2, 4))
                torch.nn.functional.mse_loss(output, target).backward()

                checkpoint = {
                    "model_type": model_type,
                    "model_config": config,
                    "input_dim": 8,
                    "model": model.state_dict(),
                }
                restored = restore_model(checkpoint)
                restored_output = (
                    restored(x, weather)
                    if model_type == "LoadTransformer"
                    else restored(x)
                )
                self.assertEqual(tuple(restored_output.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
