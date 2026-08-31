from argparse import Namespace

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .data import build_latest_window, load_excel
from .model_registry import build_model


def restore_model(checkpoint):
    model = build_model(
        checkpoint["model_type"],
        checkpoint["input_dim"],
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    config = Namespace(
        data="附件3：训练数据集/训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx",
        checkpoint="outputs/load_transformer.pt",
        output="prediction_24h.csv",
    )

    checkpoint = load_checkpoint(config.checkpoint, map_location="cpu")
    frame = load_excel(config.data)
    x, feature_names = build_latest_window(
        frame,
        checkpoint["input_hours"],
        checkpoint["wavelet_columns"],
    )
    if feature_names != checkpoint["feature_names"]:
        raise ValueError("预测特征顺序与训练检查点不一致")
    x = (x - checkpoint["x_mean"]) / checkpoint["x_std"]

    model = restore_model(checkpoint)
    with torch.no_grad():
        prediction = model(torch.from_numpy(x[-1:])).numpy()[0] * checkpoint["y_std"] + checkpoint["y_mean"]
    np.savetxt(
        config.output,
        prediction[:, None],
        delimiter=",",
        header="TotalRealTimeLoad",
        comments="",
    )
    print(prediction)


if __name__ == "__main__":
    main()
