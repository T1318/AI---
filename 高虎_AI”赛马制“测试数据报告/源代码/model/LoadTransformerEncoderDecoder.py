import math

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    """为连续时间步加入固定正弦位置编码。"""

    def __init__(self, d_model, max_length):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(
            position * div[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class LoadTransformerEncoderDecoder(nn.Module):
    """用历史状态编码、未来天气解码的并行负荷预测模型。"""

    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        horizon=24,
        dropout=0.1,
        future_weather_dim=8,
    ):
        super().__init__()
        self.horizon = horizon
        self.future_weather_dim = future_weather_dim

        self.history_projection = nn.Linear(input_dim, d_model)
        self.history_position = SinusoidalPositionEncoding(d_model, max_length=168)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.weather_projection = nn.Linear(future_weather_dim, d_model)
        self.weather_position = SinusoidalPositionEncoding(
            d_model, max_length=horizon
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, history_x, future_weather):
        if future_weather is None:
            raise ValueError("LoadTransformerEncoderDecoder 需要未来天气特征")
        if future_weather.size(1) != self.horizon:
            raise ValueError("未来天气长度必须与预测范围一致")
        if future_weather.size(2) != self.future_weather_dim:
            raise ValueError("未来天气特征维度与模型配置不一致")

        memory = self.encoder(
            self.history_position(self.history_projection(history_x))
        )
        weather_query = self.weather_position(
            self.weather_projection(future_weather)
        )
        decoded = self.decoder(weather_query, memory)
        return self.output(decoded).squeeze(-1)
