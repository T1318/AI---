"""CNN + LSTM 混合模型架构。

本模块只定义模型架构，不包含特征工程和超参数寻优。
卷积层、LSTM 隐藏层和输出层均以构造参数形式暴露出来，方便后续在模块外部进行调参或替换。
"""

import torch
from torch import nn


class CNNFeatureExtractor(nn.Module):
    """一维卷积特征提取器，在时间维度上提取局部特征。

    卷积层保存在 ``self.conv`` 中，通道数、卷积核大小和卷积层数均通过构造函数指定。
    使用同尺寸填充保持时间长度不变，便于后续直接接入 LSTM。
    """

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

        # 卷积层单独暴露，方便后续调参或替换。
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入序列进行卷积特征提取。

        参数:
            x: 形状为 ``(batch, seq_len, input_size)`` 的输入序列。

        返回:
            形状为 ``(batch, seq_len, conv_channels)`` 的特征序列。
        """
        # Conv1d 期望 (batch, channels, length)，先调整维度。
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        x = self.conv(x)       # (batch, conv_channels, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, conv_channels)
        return x


class CnnLstmModel(nn.Module):
    """CNN + LSTM 混合模型架构。

    CNN 用于提取局部特征，LSTM 用于建模时间依赖，最后经输出层得到预测。
    LSTM 的层数和宽度、卷积层数以及输出层宽度均通过构造参数指定。
    本模型不做特征工程，也不做超参数寻优。
    """

    def __init__(
        self,
        input_size: int,
        conv_channels: int,
        kernel_size: int,
        lstm_hidden_size: int,
        output_size: int,
        num_conv_layers: int = 1,
        lstm_num_layers: int = 1,
        dropout: float = 0.0,
        horizon: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.conv_channels = int(conv_channels)
        self.kernel_size = int(kernel_size)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.output_size = int(output_size)
        self.num_conv_layers = int(num_conv_layers)
        self.lstm_num_layers = int(lstm_num_layers)
        self.dropout = float(dropout)
        self.horizon = None if horizon is None else int(horizon)

        self.cnn = CNNFeatureExtractor(
            input_size=self.input_size,
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            num_layers=self.num_conv_layers,
        )

        # 当只有一层时，LSTM 的 dropout 参数不可用，需要置为 0。
        lstm_dropout = self.dropout if self.lstm_num_layers > 1 else 0.0
        # LSTM 隐藏层单独暴露，方便后续调参或替换。
        self.lstm = nn.LSTM(
            input_size=self.conv_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        # 输出层单独暴露，方便后续调参或替换。
        if self.horizon is not None:
            self.output_layer = nn.Linear(self.lstm_hidden_size, self.horizon)
        else:
            self.output_layer = nn.Linear(self.lstm_hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入序列进行预测。

        参数:
            x: 形状为 ``(batch, seq_len, input_size)`` 的输入序列。

        返回:
            形状为 ``(batch, seq_len, output_size)`` 的预测值。
        """
        features = self.cnn(x)              # (batch, seq_len, conv_channels)
        lstm_out, _ = self.lstm(features)   # (batch, seq_len, lstm_hidden_size)
        if self.horizon is not None:
            return self.output_layer(lstm_out[:, -1, :])
        predictions = self.output_layer(lstm_out)
        return predictions


if __name__ == "__main__":
    # 快速形状自检；需要先安装 PyTorch。
    model = CnnLstmModel(
        input_size=8,
        conv_channels=32,
        kernel_size=3,
        lstm_hidden_size=64,
        output_size=1,
        num_conv_layers=2,
        lstm_num_layers=2,
    )
    x = torch.randn(4, 24, 8)
    out = model(x)
    print(out.shape)  # 期望输出：torch.Size([4, 24, 1])

