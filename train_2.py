"""设备运行参数多任务模型训练入口。"""
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from load_forecasting.checkpoint import load_checkpoint
from load_forecasting.data import WAVELET_COLUMNS, load_excel
from train import build_optimizer, split_time_frames
from .equipment_forecasting.data import build_equipment_windows
from .equipment_forecasting.metrics import grouped_metrics
from .equipment_forecasting.model import (
    EquipmentResponseTransformer,
    apply_physical_constraints,
    concat_outputs,
    grouped_masked_huber_loss,
)
from .equipment_forecasting.targets import EQUIPMENT_TARGET_NAMES, FUTURE_PLAN_NAMES


def default_config():
    return Namespace(
        data="附件3：训练数据集/训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx",
        output="outputs/equipment_response.pt",
        input_hours=168,
        horizon=24,
        epochs=100,
        model_config={
            "d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.1,
        },
        training_config={
            "lr": 1e-3, "weight_decay": 0.0,
            "batch_size": 64, "optimizer": "AdamW",
        },
        device="auto",
        log_every=5,
    )


def _masked_target_stats(target, mask):
    count = np.maximum(mask.sum(axis=(0, 1)), 1.0)
    mean = (target * mask).sum(axis=(0, 1)) / count
    variance = (((target - mean) ** 2) * mask).sum(axis=(0, 1)) / count
    return mean.astype(np.float32), (np.sqrt(variance) + 1e-6).astype(np.float32)


def prepare_equipment_datasets(config):
    frame = load_excel(config.data)
    train_frame, valid_frame, test_frame = split_time_frames(
        frame, config.input_hours
    )
    train = build_equipment_windows(
        train_frame, config.input_hours, config.horizon, WAVELET_COLUMNS, 1
    )
    valid = build_equipment_windows(
        valid_frame, config.input_hours, config.horizon, WAVELET_COLUMNS, 24
    )
    test = build_equipment_windows(
        test_frame, config.input_hours, config.horizon, WAVELET_COLUMNS, 24
    )
    x_train, plan_train, y_train, mask_train, history_names = train
    x_valid, plan_valid, y_valid, mask_valid, valid_names = valid
    x_test, plan_test, y_test, mask_test, test_names = test
    if history_names != valid_names or history_names != test_names:
        raise ValueError("设备模型历史特征顺序不一致")
    x_mean = x_train.mean(axis=(0, 1))
    x_std = x_train.std(axis=(0, 1)) + 1e-6
    plan_mean = plan_train.mean(axis=(0, 1))
    plan_std = plan_train.std(axis=(0, 1)) + 1e-6
    target_mean, target_std = _masked_target_stats(y_train, mask_train)
    scale_x = lambda value: ((value - x_mean) / x_std).astype(np.float32)
    scale_plan = lambda value: ((value - plan_mean) / plan_std).astype(np.float32)
    scale_y = lambda value: ((value - target_mean) / target_std).astype(np.float32)
    return {
        "x_train": scale_x(x_train), "plan_train": scale_plan(plan_train),
        "y_train": scale_y(y_train), "mask_train": mask_train,
        "x_valid": scale_x(x_valid), "plan_valid": scale_plan(plan_valid),
        "y_valid": scale_y(y_valid), "y_valid_raw": y_valid,
        "mask_valid": mask_valid,
        "x_test": scale_x(x_test), "plan_test": scale_plan(plan_test),
        "y_test_raw": y_test, "mask_test": mask_test,
        "history_names": history_names,
        "plan_names": list(FUTURE_PLAN_NAMES),
        "target_names": list(EQUIPMENT_TARGET_NAMES),
        "x_mean": x_mean, "x_std": x_std,
        "plan_mean": plan_mean, "plan_std": plan_std,
        "target_mean": target_mean, "target_std": target_std,
    }


def _selection_value(result):
    values = [
        item["mape"] for item in result["groups"].values()
        if np.isfinite(item["mape"])
    ]
    return float(np.mean(values)) if values else float("inf")


def train_equipment_model(config, datasets):
    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available()
        else config.device if config.device != "auto" else "cpu"
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(datasets["x_train"]),
            torch.from_numpy(datasets["plan_train"]),
            torch.from_numpy(datasets["y_train"]),
            torch.from_numpy(datasets["mask_train"]),
        ),
        batch_size=config.training_config["batch_size"],
        shuffle=True,
    )
    model = EquipmentResponseTransformer(
        history_dim=datasets["x_train"].shape[-1],
        plan_dim=datasets["plan_train"].shape[-1],
        **config.model_config,
    ).to(device)
    optimizer = build_optimizer(
        config.training_config["optimizer"], model.parameters(),
        config.training_config["lr"], config.training_config["weight_decay"],
    )
    best = float("inf")
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        for history, plan, target, mask in train_loader:
            optimizer.zero_grad()
            prediction = concat_outputs(model(history.to(device), plan.to(device)))
            loss = grouped_masked_huber_loss(
                prediction, target.to(device), mask.to(device)
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_norm = concat_outputs(model(
                torch.from_numpy(datasets["x_valid"]).to(device),
                torch.from_numpy(datasets["plan_valid"]).to(device),
            )).cpu().numpy()
        pred_raw = (
            pred_norm * datasets["target_std"] + datasets["target_mean"]
        )
        validation = grouped_metrics(
            pred_raw, datasets["y_valid_raw"], datasets["mask_valid"]
        )
        score = _selection_value(validation)
        if score < best:
            best = score
            torch.save({
                "task": "equipment_response",
                "epoch": epoch,
                "best_group_mape": score,
                "model": model.state_dict(),
                "model_config": dict(config.model_config),
                "history_dim": datasets["x_train"].shape[-1],
                "plan_dim": datasets["plan_train"].shape[-1],
                "input_hours": config.input_hours,
                "horizon": config.horizon,
                "history_names": datasets["history_names"],
                "plan_names": datasets["plan_names"],
                "target_names": datasets["target_names"],
                "x_mean": datasets["x_mean"], "x_std": datasets["x_std"],
                "plan_mean": datasets["plan_mean"],
                "plan_std": datasets["plan_std"],
                "target_mean": datasets["target_mean"],
                "target_std": datasets["target_std"],
            }, output)
        if epoch == 1 or epoch % config.log_every == 0:
            print(f"epoch={epoch} valid_group_mape={score:.6f}")
    checkpoint = load_checkpoint(output, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        pred_norm = concat_outputs(model(
            torch.from_numpy(datasets["x_test"]).to(device),
            torch.from_numpy(datasets["plan_test"]).to(device),
        )).cpu().numpy()
    pred_raw = pred_norm * datasets["target_std"] + datasets["target_mean"]
    pred_raw = apply_physical_constraints(
        torch.from_numpy(pred_raw), torch.from_numpy(datasets["mask_test"])
    ).numpy()
    result = grouped_metrics(
        pred_raw, datasets["y_test_raw"], datasets["mask_test"]
    )
    print(json.dumps(result["groups"], ensure_ascii=False))
    return model, result


def train_xgboost_baseline(datasets, independent=False, params=None):
    from load_forecasting.xgboost_model import XGBoostModel
    params = params or {"n_estimators": 100, "max_depth": 6, "n_jobs": -1}
    x = datasets["plan_train"].reshape(-1, datasets["plan_train"].shape[-1])
    y = datasets["y_train"].reshape(-1, 28)
    mask = datasets["mask_train"].reshape(-1, 28)
    if independent:
        models = []
        for index in range(28):
            valid = mask[:, index] > 0
            model = XGBoostModel(**params).fit(x[valid], y[valid, index])
            models.append(model)
        return models
    return XGBoostModel(**params).fit(x, y)


def main():
    config = default_config()
    datasets = prepare_equipment_datasets(config)
    train_equipment_model(config, datasets)


if __name__ == "__main__":
    main()
