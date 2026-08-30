"""双向 LSTM 负荷预测模型架构。

本模块只定义模型架构，不包含特征工程和超参数寻优。
双向 LSTM 隐藏层与输出层均以构造参数形式暴露出来，方便后续调参或替换。
"""

import torch
from torch import nn


class BiLSTMModel(nn.Module):
    """双向 LSTM 回归模型，可同时利用过去与未来时刻的信息。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        horizon: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.horizon = None if horizon is None else int(horizon)

        lstm_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        # 双向拼接后维度翻倍。
        if self.horizon is not None:
            self.output_layer = nn.Linear(self.hidden_size * 2, self.horizon)
        else:
            self.output_layer = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入序列进行预测。

        参数:
            x: 形状为 ``(batch, seq_len, input_size)`` 的输入序列。

        返回:
            形状为 ``(batch, seq_len, output_size)`` 的预测值。
        """
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        if self.horizon is not None:
            return self.output_layer(out[:, -1, :])
        return self.output_layer(out)


if __name__ == "__main__":
    # 快速形状自检；需要先安装 PyTorch。
    model = BiLSTMModel(input_size=8, hidden_size=64, output_size=1, num_layers=2)
    x = torch.randn(4, 24, 8)
    out = model(x)
    print(out.shape)  # 期望输出：torch.Size([4, 24, 1])

