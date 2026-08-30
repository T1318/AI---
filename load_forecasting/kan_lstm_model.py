"""KAN-LSTM 混合模型架构。

本模块只定义模型架构，不包含特征工程和超参数寻优。
KAN 隐藏层、LSTM 隐藏层和输出层均以构造参数形式暴露出来，方便后续调参或替换。
"""

import torch
from torch import nn


class KANLinear(nn.Module):
    """基于 B 样条基函数的 KAN 线性层。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)

        step = (grid_range[1] - grid_range[0]) / self.grid_size
        self.num_knots = self.grid_size + 2 * self.spline_order + 1
        self.num_basis = self.grid_size + self.spline_order

        knots = torch.linspace(
            grid_range[0] - self.spline_order * step,
            grid_range[1] + self.spline_order * step,
            self.num_knots,
        )
        self.register_buffer("knots", knots)

        self.coef = nn.Parameter(
            torch.zeros(self.out_features, self.in_features, self.num_basis)
        )
        nn.init.normal_(self.coef, mean=0.0, std=0.1)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        knots = self.knots
        p = self.spline_order
        x = x.unsqueeze(-1)

        degree0 = (x >= knots[:-1]) & (x < knots[1:])
        degree0[..., -1] = degree0[..., -1] | (x[..., 0] >= knots[-1])
        bases = degree0.to(x.dtype)

        for k in range(1, p + 1):
            i = torch.arange(knots.shape[0] - k - 1, device=x.device)
            num1 = x - knots[i]
            den1 = knots[i + k] - knots[i]
            term1 = num1 / den1.clamp_min(1e-8) * bases[..., i]

            num2 = knots[i + k + 1] - x
            den2 = knots[i + k + 1] - knots[i + 1]
            term2 = num2 / den2.clamp_min(1e-8) * bases[..., i + 1]

            bases = term1 + term2

        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bases = self.b_splines(x)
        return torch.einsum("jik,bik->bj", self.coef, bases)


class KAN(nn.Module):
    """由多个 KANLinear 堆叠而成的前馈 KAN 网络。"""

    def __init__(
        self,
        layers_hidden,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers_hidden = [int(dim) for dim in layers_hidden]
        self.dropout = nn.Dropout(float(dropout))

        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(self.layers_hidden[:-1], self.layers_hidden[1:]):
            self.layers.append(KANLinear(in_dim, out_dim, grid_size, spline_order))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self.dropout(x)
        return x


class KanLstmModel(nn.Module):
    """先经 KAN 逐时间步变换，再经 LSTM 建模时间依赖的混合模型。"""

    def __init__(
        self,
        input_size: int,
        kan_hidden_sizes,
        kan_output_size: int,
        lstm_hidden_size: int,
        output_size: int,
        lstm_num_layers: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.0,
        horizon: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.kan_hidden_sizes = [int(dim) for dim in kan_hidden_sizes]
        self.kan_output_size = int(kan_output_size)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.output_size = int(output_size)
        self.lstm_num_layers = int(lstm_num_layers)
        self.horizon = None if horizon is None else int(horizon)

        layers_hidden = (
            [self.input_size] + self.kan_hidden_sizes + [self.kan_output_size]
        )
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=dropout,
        )

        lstm_dropout = float(dropout) if self.lstm_num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.kan_output_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        if self.horizon is not None:
            self.output_layer = nn.Linear(self.lstm_hidden_size, self.horizon)
        else:
            self.output_layer = nn.Linear(self.lstm_hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        flat = x.reshape(batch * seq_len, self.input_size)
        transformed = self.kan(flat).reshape(batch, seq_len, self.kan_output_size)
        out, _ = self.lstm(transformed)
        if self.horizon is not None:
            return self.output_layer(out[:, -1, :])
        return self.output_layer(out)


if __name__ == "__main__":
    model = KanLstmModel(
        input_size=8,
        kan_hidden_sizes=[16],
        kan_output_size=16,
        lstm_hidden_size=64,
        output_size=1,
        lstm_num_layers=2,
    )
    x = torch.randn(4, 24, 8)
    out = model(x)
    print(out.shape)  # 期望输出：torch.Size([4, 24, 1])

