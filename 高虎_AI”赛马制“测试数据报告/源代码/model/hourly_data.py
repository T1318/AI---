"""两项任务共用的24h历史→1h预测，历史逐小时以真实观测更新。"""
import numpy as np
import pandas as pd

from .test_a_data import (
    HISTORY_COLUMNS, VALIDATION_START, build_equipment_forecast_windows,
    load_hourly_equipment, split_equipment_holdout,
)
from .test_a_model import TARGET_NAMES


EVALUATION_MODE = "24h_history_1h_ahead_observed_update"


def prepare_hourly_datasets(config, task="equipment", frame=None):
    """验证窗口可用当天较早观测；训练则完整隔离验证日，不更新验证期权重。"""
    if (config.input_hours, config.horizon, config.wavelet_level) != (24, 1, 1):
        raise ValueError("小时滚动入口要求input_hours=24、horizon=1、wavelet_level=1")
    if task not in ("load", "equipment"):
        raise ValueError(f"未知任务：{task}")
    if frame is None:
        frame = load_hourly_equipment(config.data)
    # split函数的24表示完整验证日，而非单个预测窗口的horizon。
    train_frame, valid_frame = split_equipment_holdout(frame, input_hours=24, horizon=24)
    targets = ["TotalRealTimeLoad"] if task == "load" else list(TARGET_NAMES)
    parts = [build_equipment_forecast_windows(
        part, input_hours=24, horizon=1, stride=1,
        wavelet_level=1, wavelet=config.wavelet, target_names=targets,
    ) for part in (train_frame, valid_frame)]
    train, valid = parts
    expected = pd.date_range(VALIDATION_START, periods=24, freq="h")
    if not pd.DatetimeIndex(valid["timestamps"].reshape(-1)).equals(expected):
        raise ValueError("验证日不能完整生成24次一步预测；检查历史、天气及目标缺失，不能伪造观测")
    stats = {}
    for key in ("history", "weather"):
        mean = train[key].mean(axis=(0, 1))
        std = train[key].std(axis=(0, 1))
        stats[key + "_mean"] = mean
        stats[key + "_std"] = np.where(std < 1e-6, 1, std).astype(np.float32)
    count = train["mask"].sum(axis=(0, 1))
    if (count == 0).any():
        raise ValueError(f"训练目标没有有效观测：{[n for n, c in zip(targets, count) if c == 0]}")
    values = np.where(train["mask"], train["target"], 0).astype(np.float64)
    mean = values.sum(axis=(0, 1)) / count
    variance = np.where(train["mask"], (values - mean) ** 2, 0).sum(axis=(0, 1)) / count
    std = np.sqrt(variance)
    stats["target_mean"] = mean.astype(np.float32)
    stats["target_std"] = np.where(std < 1e-6, 1, std).astype(np.float32)
    for part in parts:
        part["target_raw"] = np.where(part["mask"], part["target"], np.nan)
        for key in ("history", "weather", "target"):
            part[key] = ((part[key] - stats[key + "_mean"]) / stats[key + "_std"]).astype(np.float32)
    return {"train": train, "valid": valid, "stats": stats,
            "evaluation_mode": EVALUATION_MODE, "target_names": targets}


def load_training_arrays(prepared):
    """转换成公共负荷训练函数的接口；负荷输出为(batch,1)。"""
    stats = prepared["stats"]
    train, valid = prepared["train"], prepared["valid"]
    noise = np.array([1.0, 0.5] + [0.0] * 6, dtype=np.float32)
    result = {
        "has_validation": True, "has_test": False,
        "feature_names": train["feature_names"],
        "input_dim": train["history"].shape[-1],
        "x_mean": stats["history_mean"], "x_std": stats["history_std"],
        "y_mean": float(stats["target_mean"][0]), "y_std": float(stats["target_std"][0]),
        "future_weather_mean": stats["weather_mean"],
        "future_weather_std": stats["weather_std"],
        "future_weather_names": train["weather_names"],
        "weather_noise_std": noise, "weather_noise_scale": noise / stats["weather_std"],
        "validation_timestamps": valid["timestamps"],
        "y_valid_raw": valid["target_raw"].squeeze(-1),
    }
    for name, part in (("train", train), ("valid", valid)):
        result[f"x_{name}"] = part["history"]
        result[f"future_weather_{name}"] = part["weather"]
        result[f"y_{name}"] = part["target"].squeeze(-1)
    return result
