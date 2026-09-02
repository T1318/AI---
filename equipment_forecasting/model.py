"""共享编码器、多设备输出头的设备响应 Transformer。"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .targets import GROUP_SLICES


class EquipmentResponseTransformer(nn.Module):
    def __init__(
        self, history_dim, plan_dim=24, d_model=64,
        nhead=4, num_layers=2, dropout=0.1,
    ):
        super().__init__()
        self.history_projection = nn.Linear(history_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.plan_projection = nn.Linear(plan_dim, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.heads = nn.ModuleDict({
            "chiller": nn.Linear(d_model, 12),
            "pump": nn.Linear(d_model, 8),
            "tower": nn.Linear(d_model, 6),
            "system": nn.Linear(d_model, 2),
        })

    def forward(self, history, future_plan):
        length = history.size(1)
        position = torch.arange(length, device=history.device, dtype=history.dtype)
        dim = self.history_projection.out_features
        div = torch.exp(
            torch.arange(0, dim, 2, device=history.device)
            * (-math.log(10000.0) / dim)
        )
        encoding = torch.zeros(length, dim, device=history.device, dtype=history.dtype)
        encoding[:, 0::2] = torch.sin(position[:, None] * div)
        encoding[:, 1::2] = torch.cos(
            position[:, None] * div[: encoding[:, 1::2].shape[1]]
        )
        hidden = self.history_projection(history) + encoding.unsqueeze(0)
        context = self.encoder(hidden)[:, -1].unsqueeze(1)
        context = context.expand(-1, future_plan.size(1), -1)
        fused = self.fusion(
            torch.cat([context, self.plan_projection(future_plan)], dim=-1)
        )
        return {name: head(fused) for name, head in self.heads.items()}


def concat_outputs(outputs):
    return torch.cat(
        [outputs["chiller"], outputs["pump"], outputs["tower"], outputs["system"]],
        dim=-1,
    )


def grouped_masked_huber_loss(prediction, target, mask):
    losses = []
    for group_slice in GROUP_SLICES.values():
        part_mask = mask[..., group_slice]
        part_loss = F.smooth_l1_loss(
            prediction[..., group_slice],
            target[..., group_slice],
            reduction="none",
        )
        denominator = part_mask.sum()
        if denominator > 0:
            losses.append((part_loss * part_mask).sum() / denominator)
    return torch.stack(losses).mean() if losses else prediction.sum() * 0


def apply_physical_constraints(prediction, mask=None):
    result = prediction.clone()
    nonnegative = list(range(0, 3)) + list(range(12, 23))
    result[..., nonnegative] = result[..., nonnegative].clamp_min(0)
    if mask is not None:
        result = result * mask
    return result
