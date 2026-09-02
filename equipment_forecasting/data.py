"""设备预测窗口构造。"""
import numpy as np
import pandas as pd

from load_forecasting.data import (
    TARGET_COLUMN,
    TIME_COLUMN,
    WAVELET_COLUMNS,
    _is_continuous_hourly,
    wavelet_feature_window,
)
from .targets import (
    build_equipment_masks,
    build_masks_from_control_plan,
    build_future_control_plan,
    derive_equipment_targets,
)


def build_equipment_windows(
    df,
    input_hours=168,
    horizon=24,
    history_columns=None,
    stride_hours=1,
):
    columns = list(history_columns or WAVELET_COLUMNS)
    if len(df) < input_hours + horizon:
        raise ValueError("数据量不足以构造设备预测窗口")
    histories, plans, targets, masks = [], [], [], []
    history_names = None
    sample_count = len(df) - input_hours - horizon + 1
    for start in range(0, sample_count, stride_hours):
        block = df.iloc[start : start + input_hours + horizon]
        if not _is_continuous_hourly(block[TIME_COLUMN]):
            continue
        try:
            history, history_names = wavelet_feature_window(
                block.iloc[:input_hours], columns
            )
            future = block.iloc[input_hours : input_hours + horizon]
            plan = build_future_control_plan(future)
            target = derive_equipment_targets(future)
            mask = build_equipment_masks(future)
        except (KeyError, ValueError):
            continue
        histories.append(history.to_numpy(dtype=np.float32))
        plans.append(plan.to_numpy(dtype=np.float32))
        targets.append(target.to_numpy(dtype=np.float32))
        masks.append(mask.astype(np.float32))
    if not histories:
        raise ValueError("没有可用的设备预测窗口")
    return (
        np.stack(histories), np.stack(plans),
        np.stack(targets), np.stack(masks), history_names,
    )


def load_control_plan_csv(path, expected_start, horizon=24):
    raw = pd.read_csv(path)
    if len(raw) != horizon:
        raise ValueError(f"控制计划必须包含 {horizon} 条逐小时记录")
    if "timeStamp" not in raw.columns:
        raise ValueError("控制计划缺少 timeStamp")
    timestamps = pd.to_datetime(
        raw["timeStamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    expected = pd.date_range(pd.Timestamp(expected_start), periods=horizon, freq="h")
    if timestamps.isna().any() or not timestamps.reset_index(drop=True).equals(pd.Series(expected)):
        raise ValueError("控制计划时间必须从历史末端下一小时开始并逐小时连续")
    frame = raw.copy()
    frame["timeStamp"] = timestamps
    plan = build_future_control_plan(frame)
    masks = build_masks_from_control_plan(frame)
    return (
        plan.to_numpy(dtype=np.float32)[None, ...],
        list(plan.columns),
        expected,
        masks.astype(np.float32)[None, ...],
    )
