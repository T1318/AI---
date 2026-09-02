"""设备运行参数预测入口。"""
from argparse import Namespace

import numpy as np
import pandas as pd
import torch

from load_forecasting.checkpoint import load_checkpoint
from load_forecasting.data import build_latest_history_window, load_excel
from .data import load_control_plan_csv
from .model import (
    EquipmentResponseTransformer,
    apply_physical_constraints,
    concat_outputs,
)
from .targets import EQUIPMENT_TARGET_NAMES, efficiency_to_cop


def prediction_frame(timestamps, prediction, names_override=None):
    names = list(names_override or EQUIPMENT_TARGET_NAMES)
    result = pd.DataFrame(prediction, columns=names)
    result.insert(0, "timeStamp", timestamps)
    for i in range(1, 4):
        result[f"COP0{i}"] = efficiency_to_cop(
            result[f"ChRealtimeEfficiencyKW0{i}"]
        )
    return result


def main():
    config = Namespace(
        data="附件3：训练数据集/训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx",
        plan="future_control_plan_24h.csv",
        checkpoint="outputs/equipment_response.pt",
        output="equipment_prediction_24h.csv",
    )
    checkpoint = load_checkpoint(config.checkpoint, "cpu")
    frame = load_excel(config.data)
    history, history_names, last_time = build_latest_history_window(
        frame, checkpoint["input_hours"]
    )
    if history_names != checkpoint["history_names"]:
        raise ValueError("历史特征顺序与设备检查点不一致")
    plan, plan_names, timestamps, masks = load_control_plan_csv(
        config.plan, last_time + pd.Timedelta(hours=1), checkpoint["horizon"]
    )
    if plan_names != checkpoint["plan_names"]:
        raise ValueError("控制计划特征顺序与设备检查点不一致")
    history = (history - checkpoint["x_mean"]) / checkpoint["x_std"]
    plan = (plan - checkpoint["plan_mean"]) / checkpoint["plan_std"]
    model = EquipmentResponseTransformer(
        checkpoint["history_dim"], checkpoint["plan_dim"],
        **checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        pred_norm = concat_outputs(model(
            torch.from_numpy(history.astype(np.float32)),
            torch.from_numpy(plan.astype(np.float32)),
        ))
    prediction = (
        pred_norm.numpy()[0] * checkpoint["target_std"]
        + checkpoint["target_mean"]
    )
    prediction = apply_physical_constraints(
        torch.from_numpy(prediction), torch.from_numpy(masks[0])
    ).numpy()
    prediction_frame(
        timestamps, prediction, checkpoint["target_names"]
    ).to_csv(config.output, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(config.output)


if __name__ == "__main__":
    main()
