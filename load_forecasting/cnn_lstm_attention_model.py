"""CNN-LSTM-ATT 混合模型架构。

本模块只定义模型架构，不包含特征工程和超参数寻优。
卷积层、LSTM 隐藏层、注意力层和输出层均以构造参数形式暴露出来，方便后续调参或替换。
"""

import torch
from torch import nn


class CNNFeatureExtractor(nn.Module):
    """一维卷积特征提取器，在时间维度上提取局部特征。"""

    def __init__(
        self,
        input_size: int,
        conv_channels: int,
        kernel_size: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.conv_channels = int(conv_channels)
        self.kernel_size = int(kernel_size)
        self.num_layers = int(num_layers)

        layers = []
        in_channels = self.input_size
        for _ in range(self.num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=self.conv_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            in_channels = self.conv_channels
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class TemporalAttention(nn.Module):
    """自注意力残差块，用于增强时间序列中的关键信息。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class CnnLstmAttentionModel(nn.Module):
    """CNN 提取局部特征、LSTM 建模时间依赖、注意力增强关键信息的混合模型。"""

    def __init__(
        self,
        input_size: int,
        conv_channels: int,
        kernel_size: int,
        num_conv_layers: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        output_size: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        horizon: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.conv_channels = int(conv_channels)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.output_size = int(output_size)
        self.horizon = None if horizon is None else int(horizon)

        self.cnn = CNNFeatureExtractor(
            input_size=self.input_size,
            conv_channels=self.conv_channels,
            kernel_size=kernel_size,
            num_layers=num_conv_layers,
        )

        lstm_dropout = float(dropout) if int(lstm_num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.conv_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=int(lstm_num_layers),
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.attention = TemporalAttention(
            embed_dim=self.lstm_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        if self.horizon is not None:
            self.output_layer = nn.Linear(self.lstm_hidden_size, self.horizon)
        else:
            self.output_layer = nn.Linear(self.lstm_hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        out, _ = self.lstm(features)
        out = self.attention(out)
        if self.horizon is not None:
            return self.output_layer(out[:, -1, :])
        return self.output_layer(out)


if __name__ == "__main__":
    model = CnnLstmAttentionModel(
        input_size=8,
        conv_channels=32,
        kernel_size=3,
        num_conv_layers=2,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        output_size=1,
        num_heads=1,
    )
    x = torch.randn(4, 24, 8)
    out = model(x)
    print(out.shape)  # 期望输出：torch.Size([4, 24, 1])

