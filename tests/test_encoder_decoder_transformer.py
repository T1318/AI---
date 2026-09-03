import unittest
from unittest.mock import patch

import torch
from torch import nn

from load_forecasting.model_registry import (
    MODEL_SPECS,
    build_model,
    default_model_config,
    get_model_spec,
)
from load_forecasting.predict import restore_model


class EncoderDecoderTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = default_model_config(
            "LoadTransformerEncoderDecoder", horizon=24
        )
        self.model = build_model(
            "LoadTransformerEncoderDecoder", input_dim=116, model_config=self.config
        ).eval()
        self.history = torch.randn(2, 168, 116)
        self.weather = torch.randn(2, 24, 8)

    def test_registry_marks_model_as_future_weather_model(self):
        spec = get_model_spec("LoadTransformerEncoderDecoder")

        self.assertEqual(spec["kind"], "deep")
        self.assertTrue(spec["uses_future_weather"])
        self.assertEqual(spec["defaults"]["future_weather_dim"], 8)
        self.assertEqual(spec["search_space"]["num_layers"]["high"], 3)

    def test_parallel_decoder_outputs_one_value_per_future_hour(self):
        output = self.model(self.history, self.weather)

        self.assertEqual(tuple(output.shape), (2, 24))

    def test_decoder_contains_self_attention_and_cross_attention(self):
        layer = self.model.decoder.layers[0]

        self.assertIsInstance(layer.self_attn, nn.MultiheadAttention)
        self.assertIsInstance(layer.multihead_attn, nn.MultiheadAttention)

    def test_parallel_decoder_does_not_receive_a_causal_mask(self):
        original_forward = self.model.decoder.forward
        with patch.object(
            self.model.decoder, "forward", wraps=original_forward
        ) as decoder_forward:
            self.model(self.history, self.weather)

        self.assertIsNone(decoder_forward.call_args.kwargs.get("tgt_mask"))

    def test_output_responds_to_history_and_future_weather(self):
        with torch.no_grad():
            baseline = self.model(self.history, self.weather)
            changed_history = self.model(self.history + 1.0, self.weather)
            changed_weather = self.model(self.history, self.weather + 1.0)

        self.assertFalse(torch.allclose(baseline, changed_history))
        self.assertFalse(torch.allclose(baseline, changed_weather))

    def test_checkpoint_restores_encoder_decoder_model(self):
        checkpoint = {
            "model_type": "LoadTransformerEncoderDecoder",
            "model_config": self.config,
            "input_dim": 116,
            "model": self.model.state_dict(),
        }

        restored = restore_model(checkpoint)

        self.assertEqual(
            tuple(restored(self.history, self.weather).shape),
            (2, 24),
        )


if __name__ == "__main__":
    unittest.main()
