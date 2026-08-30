"""LSTM 序列到序列（编码器-解码器）模型架构。

本模块只定义模型架构，不包含特征工程和超参数寻优。
隐藏层与输出层均以构造参数形式暴露出来，方便后续在模块外部进行调参或替换。
"""

from typing import Optional, Tuple

import torch
from torch import nn


class Encoder(nn.Module):
    """LSTM 编码器，将输入序列映射为上下文向量。

    循环隐藏层保存在 ``self.lstm`` 中，其层数和每层宽度均通过构造函数指定，
    因此可以在不修改架构的前提下自由调整。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        # 当只有一层时，LSTM 的 dropout 参数不可用，需要置为 0。
        lstm_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """对输入序列进行编码。

        参数:
            x: 形状为 ``(batch, seq_len, input_size)`` 的输入序列。

        返回:
            outputs: 形状为 ``(batch, seq_len, hidden_size)`` 的各时刻输出。
            state: ``(hidden, cell)`` 元组，两者形状均为
                ``(num_layers, batch, hidden_size)``。
        """
        outputs, state = self.lstm(x)
        return outputs, state


class Decoder(nn.Module):
    """LSTM 解码器，将带上下文条件的输入映射为预测值。

    循环隐藏层保存在 ``self.lstm``，输出层保存在 ``self.output_layer``，
    两者均单独暴露，方便后续独立调参或替换。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)

        # 当只有一层时，LSTM 的 dropout 参数不可用，需要置为 0。
        lstm_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        # 输出层单独暴露，方便后续调参或替换。
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """对带上下文条件的序列进行解码。

        参数:
            x: 形状为 ``(batch, seq_len, input_size)`` 的解码器输入。
            state: 可选的 ``(hidden, cell)`` 初始状态。

        返回:
            predictions: 形状为 ``(batch, seq_len, output_size)`` 的预测值。
            state: 更新后的 ``(hidden, cell)`` 状态。
        """
        outputs, state = self.lstm(x, state)
        predictions = self.output_layer(outputs)
        return predictions, state


class Seq2SeqLSTM(nn.Module):
    """编码器-解码器 LSTM 架构。

    编码器和解码器的隐藏层宽度、层数以及输出层宽度均通过构造参数指定。
    本模型不做特征工程，也不做超参数寻优。
    """

    def __init__(
        self,
        input_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        output_size: int,
        encoder_num_layers: int = 1,
        decoder_num_layers: int = 1,
        dropout: float = 0.0,
        horizon: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.encoder_hidden_size = int(encoder_hidden_size)
        self.decoder_hidden_size = int(decoder_hidden_size)
        self.output_size = int(output_size)
        self.encoder_num_layers = int(encoder_num_layers)
        self.decoder_num_layers = int(decoder_num_layers)
        self.dropout = float(dropout)
        self.horizon = None if horizon is None else int(horizon)

        self.encoder = Encoder(
            input_size=self.input_size,
            hidden_size=self.encoder_hidden_size,
            num_layers=self.encoder_num_layers,
            dropout=self.dropout,
        )

        # 解码器每个时刻的输入为“编码器上下文 + 上一步输出”，
        # 因此输入维度是编码器隐藏层宽度与输出层宽度之和。
        decoder_input_size = self.encoder_hidden_size + self.output_size
        self.decoder = Decoder(
            input_size=decoder_input_size,
            hidden_size=self.decoder_hidden_size,
            output_size=self.output_size,
            num_layers=self.decoder_num_layers,
            dropout=self.dropout,
        )

        if self.horizon is not None:
            self.horizon_head = nn.Linear(self.encoder_hidden_size, self.horizon)

    def forward(self, x: torch.Tensor, y_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成目标序列的预测结果。

        参数:
            x: 源序列，形状为 ``(batch, src_len, input_size)``。
            y_prev: 上一步目标值，形状为
                ``(batch, tgt_len, output_size)``。

        返回:
            预测值，形状为 ``(batch, tgt_len, output_size)``。
        """
        encoder_outputs, _ = self.encoder(x)
        context = encoder_outputs[:, -1, :]  # (batch, encoder_hidden_size)

        if y_prev is None:
            if self.horizon is None:
                raise ValueError("seq2seq 在非 horizon 模式下必须传入 y_prev")
            return self.horizon_head(context)

        tgt_len = y_prev.shape[1]
        context_seq = context.unsqueeze(1).repeat(1, tgt_len, 1)
        decoder_inputs = torch.cat((context_seq, y_prev), dim=-1)

        predictions, _ = self.decoder(decoder_inputs)
        return predictions


if __name__ == "__main__":
    # 快速形状自检；需要先安装 PyTorch。
    model = Seq2SeqLSTM(
        input_size=8,
        encoder_hidden_size=32,
        decoder_hidden_size=32,
        output_size=1,
        encoder_num_layers=2,
        decoder_num_layers=2,
    )
    x = torch.randn(4, 24, 8)
    y_prev = torch.randn(4, 12, 1)
    out = model(x, y_prev)
    print(out.shape)  # 期望输出：torch.Size([4, 12, 1])

