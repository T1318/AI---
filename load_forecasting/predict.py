import argparse
from argparse import Namespace

import numpy as np
import torch
from torch import nn

from .data import build_latest_window, load_excel
from .model import LoadTransformer


def model_select(args: argparse.Namespace, feature_names: list[str]) -> nn.Module:
    if args.model == "LoadTransformer":
        return LoadTransformer(len(feature_names), args.d_model, args.nhead, args.layers, args.horizon)
    else:
        raise ValueError(f"未知模型: {args.model}")


def main() -> None:
    config = Namespace(
        data="附件3：训练数据集/训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx",
        output="outputs/load_transformer.pt",

        model = "LoadTransformer",
        input_hours=48,
        horizon=24,
        epochs=30,
        batch_size=64,
        d_model=64,
        nhead=4,
        layers=2,
        lr=1e-3,
        save_every=20,
        log_every=5,
        device="cuda",
        checkpoint="outputs/load_transformer.pt",
    )

    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    frame = load_excel(config.data)
    x, feature_names = build_latest_window(
        frame,
        checkpoint["input_hours"],
        checkpoint["wavelet_columns"],
    )
    if feature_names != checkpoint["feature_names"]:
        raise ValueError("预测特征顺序与训练检查点不一致")
    x = (x - checkpoint["x_mean"]) / checkpoint["x_std"]

    # 同样修改模型加载接口
    model = model_select(config, feature_names)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x[-1:])).numpy()[0] * checkpoint["y_std"] + checkpoint["y_mean"]
    np.savetxt(
        "prediction_24h.csv",
        prediction[:, None],
        delimiter=",",
        header="TotalRealTimeLoad",
        comments="",
    )
    print(prediction)


if __name__ == "__main__":
    main()
