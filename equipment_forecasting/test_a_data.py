"""测试项目A设备预测数据；保留真实目标，隔离7月16日。"""
import numpy as np
import pandas as pd

from load_forecasting.data import (
    INVALID_VALUES, TIME_COLUMN, _interpolate_short_gaps,
    _is_continuous_hourly, future_weather_frame, wavelet_feature_window,
)
from train_a_july16 import WAVELET_COLUMNS_JULY16
from .test_a_model import NONNEGATIVE_INDICES, TARGET_NAMES


HISTORY_COLUMNS = list(WAVELET_COLUMNS_JULY16)
VALIDATION_START = pd.Timestamp("2025-07-16")
VALIDATION_END = pd.Timestamp("2025-07-17")


def load_hourly_equipment(path):
    """小时均值作为观测目标，不用插值值充当训练标签。"""
    raw = pd.read_excel(path, sheet_name="历史数据")
    required = list(dict.fromkeys([TIME_COLUMN] + HISTORY_COLUMNS + TARGET_NAMES))
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"设备数据缺少列：{missing}")
    raw = raw[required].copy()
    raw[TIME_COLUMN] = pd.to_datetime(raw[TIME_COLUMN], errors="coerce")
    raw = raw.dropna(subset=[TIME_COLUMN]).sort_values(TIME_COLUMN)
    if raw[TIME_COLUMN].duplicated().any():
        raise ValueError("数据存在重复时间戳，请先核对重复记录")
    for name in required[1:]:
        raw[name] = pd.to_numeric(raw[name], errors="coerce").replace(
            [*INVALID_VALUES, np.inf, -np.inf], np.nan,
        )
    for index in NONNEGATIVE_INDICES:
        name = TARGET_NAMES[index]
        raw.loc[raw[name] < 0, name] = np.nan
    return raw.set_index(TIME_COLUMN).resample("1h").mean().reset_index()


def split_equipment_holdout(frame, input_hours=168, horizon=24):
    """训练中删除验证日；跨验证日和跨月份缺口的窗口不可用。"""
    if horizon != 24:
        raise ValueError("单日验证策略的horizon必须为24")
    frame = frame.loc[
        (frame[TIME_COLUMN].dt.year == 2025)
        & frame[TIME_COLUMN].dt.month.isin([4, 7, 9])
    ].copy()
    times = frame[TIME_COLUMN]
    train = frame.loc[~((times >= VALIDATION_START) & (times < VALIDATION_END))]
    valid = frame.loc[
        (times >= VALIDATION_START - pd.Timedelta(hours=input_hours))
        & (times < VALIDATION_END)
    ]
    expected = pd.date_range(VALIDATION_START - pd.Timedelta(hours=input_hours),
                             periods=input_hours + horizon, freq="h")
    if not pd.DatetimeIndex(valid[TIME_COLUMN]).equals(expected):
        raise ValueError("7月16日及其历史上下文不完整")
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def build_equipment_forecast_windows(frame, input_hours=168, horizon=24, stride=1,
                                     wavelet_level=3, wavelet="db4", target_names=None):
    """历史156维、未来天气8维、目标27维；无效目标以掩码剔除。"""
    if stride < 1:
        raise ValueError("stride必须为正整数")
    target_names = TARGET_NAMES if target_names is None else list(target_names)
    histories, weather, targets, masks, timestamps, history_times = [], [], [], [], [], []
    feature_names = weather_names = None
    for start in range(0, len(frame) - input_hours - horizon + 1, stride):
        block = frame.iloc[start:start + input_hours + horizon]
        if not _is_continuous_hourly(block[TIME_COLUMN]):
            continue
        history = block.iloc[:input_hours].copy()
        # 只在当前历史窗口内填不超过3小时的缺口，不读取未来目标。
        history[HISTORY_COLUMNS] = _interpolate_short_gaps(history[HISTORY_COLUMNS], 3)
        if not np.isfinite(history[HISTORY_COLUMNS].to_numpy(dtype=float)).all():
            continue
        future = block.iloc[input_hours:]
        if not np.isfinite(future[["OutdoorTdbin", "OutdoorWetTemp"]].to_numpy(dtype=float)).all():
            continue
        y = future[target_names].to_numpy(dtype=np.float32)
        mask = np.isfinite(y)
        if not mask.any():
            continue
        x, feature_names = wavelet_feature_window(
            history, HISTORY_COLUMNS, level=wavelet_level, wavelet=wavelet,
        )
        w, weather_names = future_weather_frame(future)
        histories.append(x.to_numpy(dtype=np.float32))
        weather.append(w.to_numpy(dtype=np.float32))
        targets.append(np.where(mask, y, 0))
        masks.append(mask)
        timestamps.append(future[TIME_COLUMN].to_numpy())
        history_times.append(history[TIME_COLUMN].to_numpy())
    if not histories:
        raise ValueError("没有可用的设备预测窗口；请检查连续历史、天气和目标数据")
    return {
        "history": np.stack(histories), "weather": np.stack(weather),
        "target": np.stack(targets), "mask": np.stack(masks),
        "timestamps": np.stack(timestamps), "history_timestamps": np.stack(history_times),
        "feature_names": feature_names, "weather_names": weather_names,
    }


def fit_target_stats(target, mask):
    count = mask.sum(axis=(0, 1))
    if (count == 0).any():
        missing = [name for name, n in zip(TARGET_NAMES, count) if n == 0]
        raise ValueError(f"训练集目标没有有效观测：{missing}")
    values = np.where(mask, target, 0).astype(np.float64)
    mean = values.sum(axis=(0, 1)) / count
    variance = np.where(mask, (values - mean) ** 2, 0).sum(axis=(0, 1)) / count
    std = np.sqrt(variance)
    # 恒定目标保留单位尺度，避免1e-6尺度导致验证误差爆炸。
    return mean.astype(np.float32), np.where(std < 1e-6, 1, std).astype(np.float32)


def prepare_equipment_datasets(config):
    frame = load_hourly_equipment(config.data)
    train_frame, valid_frame = split_equipment_holdout(frame, config.input_hours, config.horizon)
    train = build_equipment_forecast_windows(train_frame, config.input_hours, config.horizon, 1)
    valid = build_equipment_forecast_windows(valid_frame, config.input_hours, config.horizon, 24)
    if len(valid["target"]) != 1:
        raise ValueError("7月16日必须生成恰好一个验证窗口")
    stats = {}
    for key in ("history", "weather"):
        mean = train[key].mean(axis=(0, 1))
        std = train[key].std(axis=(0, 1))
        stats[key + "_mean"] = mean
        stats[key + "_std"] = np.where(std < 1e-6, 1, std).astype(np.float32)
    stats["target_mean"], stats["target_std"] = fit_target_stats(train["target"], train["mask"])
    for part in (train, valid):
        part["target_raw"] = np.where(part["mask"], part["target"], np.nan)
        for key in ("history", "weather", "target"):
            part[key] = ((part[key] - stats[key + "_mean"]) / stats[key + "_std"]).astype(np.float32)
    return {"train": train, "valid": valid, "stats": stats}
