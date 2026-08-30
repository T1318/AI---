import math
import torch
from torch import nn


class LoadTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        horizon=24,
        dropout=0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x):
        length = x.size(1)
        position = torch.arange(length, device=x.device, dtype=x.dtype)
        div = torch.exp(
            torch.arange(0, self.input_projection.out_features, 2, device=x.device)
            * (-math.log(10000.0) / self.input_projection.out_features)
        )
        encoding = torch.zeros(
            length,
            self.input_projection.out_features,
            device=x.device,
            dtype=x.dtype,
        )
        encoding[:, 0::2] = torch.sin(position[:, None] * div)
        encoding[:, 1::2] = torch.cos(
            position[:, None] * div[: encoding[:, 1::2].shape[1]]
        )
        hidden = self.input_projection(x) + encoding.unsqueeze(0)
        return self.output(self.encoder(hidden)[:, -1])
