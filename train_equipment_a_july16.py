"""测试项目A任务二：未来24小时功率、流量、瞬时冷量和总管温度。"""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from equipment_forecasting.test_a_data import HISTORY_COLUMNS, prepare_equipment_datasets
from equipment_forecasting.test_a_model import (
    EquipmentForecastTransformer, GROUP_INDICES, NONNEGATIVE_INDICES,
    TARGET_GROUPS, TARGET_NAMES, grouped_huber_loss,
)
from load_forecasting.checkpoint import load_checkpoint
from load_forecasting.model_registry import DEFAULT_TRAINING_CONFIG
from train import add_weather_noise, build_optimizer, select_device
from train_a_july16 import JULY16_DATA


def default_config_equipment():
    return Namespace(
        data=JULY16_DATA, input_hours=168, horizon=24, epochs=100,
        device="auto", log_every=5, seed=42,
        output_dir="outputs/TestA_July16_equipment",
        model_config=dict(d_model=64, nhead=4, num_layers=2, dropout=0.1),
        training_config=dict(DEFAULT_TRAINING_CONFIG),
    )


def equipment_metrics(prediction, actual, mask):
    """各列分别评价；MAPE先按参数、再按四组等权平均。"""
    if not np.isfinite(prediction).all():
        raise ValueError("模型输出存在NaN/Inf")
    rows = []
    for i, name in enumerate(TARGET_NAMES):
        valid = mask[..., i].astype(bool) & np.isfinite(actual[..., i])
        y = actual[..., i][valid].astype(float)
        error = prediction[..., i][valid].astype(float) - y
        nz = np.abs(y) > 1e-6
        rows.append({
            "target": name,
            "mae": float(np.abs(error).mean()) if len(y) else np.nan,
            "rmse": float(np.sqrt((error ** 2).mean())) if len(y) else np.nan,
            "mape": float((np.abs(error[nz]) / np.abs(y[nz])).mean() * 100) if nz.any() else np.nan,
            "hours_used": len(y), "mape_hours_used": int(nz.sum()),
            "mape_hours_zero_excluded": int((~nz).sum()),
        })
    table = pd.DataFrame(rows)
    groups = {}
    for name, indices in GROUP_INDICES.items():
        values = table.iloc[indices]["mape"].dropna()
        groups[name] = float(values.mean()) if len(values) else None
    defined = [v for v in groups.values() if v is not None]
    score = float(np.mean(defined)) if defined else float("inf")
    return table, {"group_mape": groups, "selection_value": score}


def validation_detail(timestamps, predicted, actual, mask):
    """长表：每个小时、每个参数一行；缺失标签和零分母APE留空。"""
    actual = np.where(mask, actual, np.nan).astype(float)
    error = predicted - actual
    ape = np.full(error.shape, np.nan)
    nz = np.isfinite(actual) & (np.abs(actual) > 1e-6)
    ape[nz] = np.abs(error[nz]) / np.abs(actual[nz]) * 100
    return pd.DataFrame({
        "timeStamp": np.repeat(np.asarray(timestamps).reshape(-1), len(TARGET_NAMES)),
        "target": np.tile(TARGET_NAMES, actual.shape[0] * actual.shape[1]),
        "actual_value": actual.reshape(-1), "predicted_value": predicted.reshape(-1),
        "error": error.reshape(-1), "absolute_error": np.abs(error).reshape(-1),
        "ape_percent": ape.reshape(-1), "valid_target": np.asarray(mask).reshape(-1),
    })


def _raw_prediction(model, history, weather, stats, device):
    model.eval()
    with torch.no_grad():
        normalized = model(torch.as_tensor(history, device=device),
                           torch.as_tensor(weather, device=device)).cpu().numpy()
    raw = normalized * stats["target_std"] + stats["target_mean"]
    # 只约束原始单位中的功率、流量、瞬时冷量；温度不强制归零。
    raw[..., NONNEGATIVE_INDICES] = np.maximum(raw[..., NONNEGATIVE_INDICES], 0)
    return raw


def restore_equipment_model(checkpoint, device="cpu"):
    if checkpoint.get("model_type") != "EquipmentForecastTransformer":
        raise ValueError("请使用测试项目A任务二检查点")
    if checkpoint["target_names"] != TARGET_NAMES:
        raise ValueError("检查点目标顺序与当前设备模型不一致")
    model = EquipmentForecastTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def predict_equipment(checkpoint, history, future_weather, device="cpu"):
    """接收未标准化的小波历史特征和天气8维特征，返回原始单位27项结果。"""
    stats = checkpoint["stats"]
    h = ((history - stats["history_mean"]) / stats["history_std"]).astype(np.float32)
    w = ((future_weather - stats["weather_mean"]) / stats["weather_std"]).astype(np.float32)
    model = restore_equipment_model(checkpoint, device)
    return _raw_prediction(model, h, w, stats, device)


def train_equipment_model(config, datasets):
    if config.epochs < 1:
        raise ValueError("epochs必须大于0")
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    train, valid, stats = datasets["train"], datasets["valid"], datasets["stats"]
    model_config = dict(config.model_config, history_dim=train["history"].shape[-1],
                        future_dim=train["weather"].shape[-1],
                        input_hours=config.input_hours, horizon=config.horizon)
    model = EquipmentForecastTransformer(**model_config).to(device)
    optimizer = build_optimizer(
        config.training_config["optimizer"], model.parameters(),
        config.training_config["lr"], config.training_config["weight_decay"],
    )
    loader = DataLoader(TensorDataset(*[
        torch.from_numpy(train[key]) for key in ("history", "weather", "target", "mask")
    ]), batch_size=config.training_config["batch_size"], shuffle=True)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / getattr(config, "checkpoint_name", "equipment_best.pt")
    noise = np.array([1.0, 0.5] + [0.0] * 6, dtype=np.float32)
    noise_scale = torch.as_tensor(noise / stats["weather_std"], device=device)
    best = float("inf")
    print(f"训练设备: {device}; 历史输入={train['history'].shape}; 目标={train['target'].shape}")
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        for h, w, y, mask in loader:
            optimizer.zero_grad()
            prediction = model(h.to(device), add_weather_noise(w.to(device), noise_scale))
            loss = grouped_huber_loss(prediction, y.to(device), mask.to(device))
            if not torch.isfinite(loss):
                raise ValueError("训练损失出现NaN/Inf")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(h)
        pred = _raw_prediction(model, valid["history"], valid["weather"], stats, device)
        _, summary = equipment_metrics(pred, valid["target_raw"], valid["mask"])
        score = summary["selection_value"]
        if not np.isfinite(score):
            raise ValueError("验证日所有目标均无有效非零观测，无法选择最佳检查点")
        if score < best:
            best = score
            torch.save({
                "task": "test_a_equipment_forecast", "checkpoint_version": 1,
                "model_type": "EquipmentForecastTransformer", "model": model.state_dict(),
                "model_config": model_config, "training_config": dict(config.training_config),
                "epoch": epoch, "best_valid_mape": best,
                "selection_metric": "group_mean_nonzero_mape",
                "target_names": list(TARGET_NAMES), "target_groups": TARGET_GROUPS,
                "history_columns": list(HISTORY_COLUMNS), "feature_names": train["feature_names"],
                "future_weather_names": train["weather_names"], "stats": stats,
                "input_hours": config.input_hours, "horizon": config.horizon,
                "wavelet": getattr(config, "wavelet", "db4"),
                "wavelet_level": getattr(config, "wavelet_level", 3),
                "evaluation_mode": getattr(config, "evaluation_mode", None),
                "validation_date": "2025-07-16", "future_inputs": "weather_and_time_only",
                "weather_noise_std": noise,
            }, checkpoint_path)
        if epoch == 1 or epoch % config.log_every == 0:
            print(f"epoch={epoch} train_huber={loss_sum / len(train['history']):.6f} valid_group_mape={score:.6f}")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = restore_equipment_model(checkpoint, device)
    predicted = _raw_prediction(model, valid["history"], valid["weather"], stats, device)
    table, summary = equipment_metrics(predicted, valid["target_raw"], valid["mask"])
    detail = validation_detail(valid["timestamps"], predicted, valid["target_raw"], valid["mask"])
    detail.to_csv(output / "july16_validation.csv", index=False, encoding="utf-8-sig",
                  date_format="%Y-%m-%d %H:%M:%S")
    table.to_csv(output / "july16_metrics.csv", index=False, encoding="utf-8-sig")
    summary.update(best_epoch=checkpoint["epoch"], selection_metric=checkpoint["selection_metric"])
    if getattr(config, "evaluation_mode", None):
        summary.update(evaluation_mode=config.evaluation_mode,
                       weather_source="historical_observations",
                       history_update="all_30_observed_columns", input_hours=config.input_hours,
                       horizon=config.horizon, validation_hours=len(valid["timestamps"]),
                       wavelet=checkpoint["wavelet"], wavelet_level=checkpoint["wavelet_level"])
    (output / "july16_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return model, summary


def run(config):
    return train_equipment_model(config, prepare_equipment_datasets(config))


if __name__ == "__main__":
    config = default_config_equipment()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=config.data)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=config.device)
    parser.add_argument("--batch-size", type=int, default=config.training_config["batch_size"])
    parser.add_argument("--output-dir", default=config.output_dir)
    args = parser.parse_args()
    for name in ("data", "epochs", "device", "output_dir"):
        setattr(config, name, getattr(args, name))
    config.training_config["batch_size"] = args.batch_size
    run(config)
