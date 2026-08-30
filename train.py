import argparse
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from load_forecasting.data import WAVELET_COLUMNS, build_windows, load_excel
from load_forecasting.model_registry import (
    DEFAULT_TRAINING_CONFIG,
    build_model,
    default_model_config,
    get_model_spec,
)


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    error = pred - true
    nonzero = np.abs(true) > 1e-6
    used = int(nonzero.sum())
    excluded = int((~nonzero).sum())
    mape = (
        float(np.mean(np.abs(error[nonzero]) / np.abs(true[nonzero])) * 100)
        if used
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": mape,
        "mape_hours_used": used,
        "mape_hours_zero_excluded": excluded,
    }


def validation_summary(pred_normalized, true_normalized, datasets):
    """计算验证MSE，并以原始单位的非零MAPE作为检查点选择指标。"""
    valid_mse = float(np.mean((pred_normalized - true_normalized) ** 2))
    pred_raw = pred_normalized * datasets["y_std"] + datasets["y_mean"]
    result = metrics(pred_raw, datasets["y_valid_raw"])
    selection_value = result["mape"]
    if not np.isfinite(selection_value):
        selection_value = float("inf")
    return {
        "valid_mse": valid_mse,
        "valid_mape": result["mape"],
        "mape_hours_used": result["mape_hours_used"],
        "mape_hours_zero_excluded": result["mape_hours_zero_excluded"],
        "selection_metric": "nonzero_mape",
        "selection_value": float(selection_value),
    }


def model_config_from_args(args: argparse.Namespace) -> dict:
    return dict(args.model_config)


def model_select(args: argparse.Namespace, feature_names: list[str]) -> nn.Module:
    """兼容入口；模型构造统一交给 load_forecasting.model。"""
    get_model_spec(args.model)
    return build_model(args.model, len(feature_names), model_config_from_args(args))


def build_optimizer(name, parameters, lr, weight_decay=0.0):
    optimizers = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizers:
        raise ValueError(f"未知优化器: {name}")
    return optimizers[name](parameters, lr=lr, weight_decay=weight_decay)


def build_checkpoint(
    model,
    args,
    epoch,
    validation,
    feature_names,
    x_mean,
    x_std,
    y_mean,
    y_std,
):
    return {
        "checkpoint_version": 1,
        "epoch": epoch,
        "selection_metric": validation["selection_metric"],
        "best_valid_mape": float(validation["selection_value"]),
        "valid_mse": float(validation["valid_mse"]),
        "mape_hours_used": validation["mape_hours_used"],
        "mape_hours_zero_excluded": validation["mape_hours_zero_excluded"],
        "model_type": args.model,
        "model_config": model_config_from_args(args),
        "training_config": dict(args.training_config),
        "input_dim": len(feature_names),
        "model": model.state_dict(),
        "feature_names": list(feature_names),
        "wavelet_columns": list(WAVELET_COLUMNS),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "input_hours": args.input_hours,
        "horizon": args.horizon,
    }


def save_best_checkpoint(path, valid_loss, best_loss, checkpoint):
    if valid_loss >= best_loss:
        return best_loss, False
    torch.save(checkpoint, path)
    return float(valid_loss), True


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到可用 CUDA GPU")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_time_frames(
    frame, input_hours: int, train_ratio: float = 0.7, valid_ratio: float = 0.15
):
    train_end = int(len(frame) * train_ratio)
    valid_end = int(len(frame) * (train_ratio + valid_ratio))
    train_frame = frame.iloc[:train_end].reset_index(drop=True)
    valid_frame = frame.iloc[train_end - input_hours : valid_end].reset_index(drop=True)
    test_frame = frame.iloc[valid_end - input_hours :].reset_index(drop=True)
    return train_frame, valid_frame, test_frame


def prepare_datasets(args: argparse.Namespace) -> dict:
    frame = load_excel(args.data)
    train_frame, valid_frame, test_frame = split_time_frames(
        frame, args.input_hours
    )
    x_train, y_train, feature_names = build_windows(
        train_frame, args.input_hours, args.horizon, WAVELET_COLUMNS,
        stride_hours=1,
    )
    x_valid, y_valid, valid_feature_names = build_windows(
        valid_frame, args.input_hours, args.horizon, WAVELET_COLUMNS,
        stride_hours=24,
    )
    x_test, y_test, test_feature_names = build_windows(
        test_frame, args.input_hours, args.horizon, WAVELET_COLUMNS,
        stride_hours=24,
    )
    if feature_names != valid_feature_names or feature_names != test_feature_names:
        raise ValueError("训练、验证和测试特征顺序不一致")

    x_mean, x_std = x_train.mean(axis=(0, 1)), x_train.std(axis=(0, 1)) + 1e-6
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-6
    return {
        "x_train": ((x_train - x_mean) / x_std).astype(np.float32),
        "y_train": ((y_train - y_mean) / y_std).astype(np.float32),
        "x_valid": ((x_valid - x_mean) / x_std).astype(np.float32),
        "y_valid": ((y_valid - y_mean) / y_std).astype(np.float32),
        "x_test": ((x_test - x_mean) / x_std).astype(np.float32),
        "y_test": ((y_test - y_mean) / y_std).astype(np.float32),
        "y_valid_raw": y_valid.astype(np.float32),
        "y_test_raw": y_test.astype(np.float32),
        "feature_names": feature_names,
        "input_dim": len(feature_names),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }


def train_model(args: argparse.Namespace, datasets: dict, save_checkpoint=True):
    device = select_device(args.device)
    print(f"训练设备: {device}")
    feature_names = datasets["feature_names"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(datasets["x_train"]), torch.from_numpy(datasets["y_train"])),
        batch_size=args.training_config["batch_size"], shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(datasets["x_valid"]), torch.from_numpy(datasets["y_valid"])),
        batch_size=args.training_config["batch_size"],
    )

    model = model_select(args, feature_names).to(device)
    optimizer = build_optimizer(
        args.training_config["optimizer"],
        model.parameters(),
        args.training_config["lr"],
        args.training_config["weight_decay"],
    )
    loss_fn = nn.MSELoss()
    best = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x.to(device)), batch_y.to(device)).backward()
            optimizer.step()
        model.eval()

        predictions = []
        targets = []
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                predictions.append(model(batch_x.to(device)).cpu().numpy())
                targets.append(batch_y.numpy())
        validation = validation_summary(
            np.concatenate(predictions),
            np.concatenate(targets),
            datasets,
        )
        if not np.isfinite(validation["selection_value"]):
            raise ValueError("验证集没有非零负荷，无法按非零MAPE选择检查点")
        checkpoint = build_checkpoint(
            model, args, epoch, validation, feature_names,
            datasets["x_mean"], datasets["x_std"],
            datasets["y_mean"], datasets["y_std"],
        )
        if save_checkpoint:
            best, _ = save_best_checkpoint(
                output, validation["selection_value"], best, checkpoint
            )
        elif validation["selection_value"] < best:
            best = validation["selection_value"]
        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"epoch={epoch} valid_mse={validation['valid_mse']:.6f} "
                f"valid_mape={validation['valid_mape']:.6f}"
            )

    if save_checkpoint:
        checkpoint = torch.load(output, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(datasets["x_test"]).to(device)).cpu().numpy()
        pred = pred * datasets["y_std"] + datasets["y_mean"]
    result = metrics(pred, datasets["y_test_raw"])
    print(json.dumps(result, ensure_ascii=False))
    if save_checkpoint:
        np.savetxt(output.with_suffix(".csv"), np.column_stack([datasets["y_test_raw"], pred]), delimiter=",", header="真实值_24步,预测值_24步", comments="")
    return model, result


def apply_tuning_result(config: Namespace, path) -> Namespace:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    config.model = result.get("model_type", config.model)
    config.model_config = dict(result["model_config"])
    config.training_config = dict(result["training_config"])
    return config


def run(args: argparse.Namespace) -> None:
    datasets = prepare_datasets(args)
    train_model(args, datasets, save_checkpoint=True)


def default_config(model_type="LoadTransformer") -> Namespace:
    model_config = default_model_config(model_type, horizon=24)
    return Namespace(
        data="附件3：训练数据集/训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx",
        output=f"outputs/{model_type}.pt",

        model=model_type,
        model_config=model_config,
        input_hours=168,
        horizon=24,
        epochs=1000,
        training_config=dict(DEFAULT_TRAINING_CONFIG),
        log_every=5,
        device="auto",
    )


if __name__ == "__main__":
    run(default_config())
