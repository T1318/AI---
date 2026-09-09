"""测试项目A：历史Encoder、天气Decoder、四组设备参数输出。"""
import torch
from torch import nn
from torch.nn import functional as F

from load_forecasting.LoadTransformerEncoderDecoder import SinusoidalPositionEncoding


TARGET_GROUPS = {
    "power": (
        [f"ChPower0{i}" for i in range(1, 5)]
        + [f"PriChWPPower0{i}" for i in range(1, 7)]
        + [f"CWPPower0{i}" for i in range(1, 7)]
        + [f"CTPower0{i}" for i in range(1, 5)]
    ),
    "flow": ["PriChWFlow", "CWFlow01"],
    "cooling": ["TotalRealTimeLoad"],
    "temperature": [
        "PriChWTempSupply", "PriChWTempReturn", "CWTempSupply", "CWTempReturn",
    ],
}
TARGET_NAMES = [name for names in TARGET_GROUPS.values() for name in names]
GROUP_INDICES = {
    group: [TARGET_NAMES.index(name) for name in names]
    for group, names in TARGET_GROUPS.items()
}
NONNEGATIVE_INDICES = [
    i for i, name in enumerate(TARGET_NAMES) if name not in TARGET_GROUPS["temperature"]
]


class EquipmentForecastTransformer(nn.Module):
    """输出标准化的(batch, horizon, 27)，不输入未来负荷或设备功率。"""

    def __init__(self, history_dim=156, future_dim=8, input_hours=168,
                 horizon=24, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_hours = input_hours
        self.horizon = horizon
        self.history_projection = nn.Linear(history_dim, d_model)
        self.future_projection = nn.Linear(future_dim, d_model)
        self.history_position = SinusoidalPositionEncoding(d_model, input_hours)
        self.future_position = SinusoidalPositionEncoding(d_model, horizon)
        layer_args = dict(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                          dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_args), num_layers=num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(**layer_args), num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict({
            name: nn.Linear(d_model, len(columns))
            for name, columns in TARGET_GROUPS.items()
        })

    def forward(self, history, future_weather):
        if history.size(1) != self.input_hours or future_weather.size(1) != self.horizon:
            raise ValueError("历史长度或未来天气长度与模型配置不一致")
        memory = self.encoder(self.history_position(self.history_projection(history)))
        query = self.future_position(self.future_projection(future_weather))
        # 全部未来天气均已知，解码不使用因果掩码。
        decoded = self.norm(self.decoder(query, memory))
        return torch.cat([head(decoded) for head in self.heads.values()], dim=-1)


def grouped_huber_loss(prediction, target, valid_mask):
    """每组先按有效元素平均，四组等权；停机零功率仍参与损失。"""
    mask = valid_mask.bool()
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    errors = F.smooth_l1_loss(prediction, safe_target, reduction="none")
    losses = []
    for indices in GROUP_INDICES.values():
        group_mask = mask[..., indices]
        if group_mask.any():
            losses.append(errors[..., indices][group_mask].mean())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0
