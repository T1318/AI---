"""设备目标、未来控制计划和运行掩码定义。"""
import numpy as np
import pandas as pd


CHILLER_TARGETS = (
    [f"ChPower0{i}" for i in range(1, 4)]
    + [f"ChRealtimeEfficiencyKW0{i}" for i in range(1, 4)]
    + [f"EvapDeltaT0{i}" for i in range(1, 4)]
    + [f"CondDeltaT0{i}" for i in range(1, 4)]
)
PUMP_TARGETS = (
    [f"PriChWPPower0{i}" for i in range(1, 4)]
    + [f"CWPPower0{i}" for i in range(1, 4)]
    + ["PriChWFlow01", "CWFlow01"]
)
TOWER_TARGETS = (
    [f"CTPower0{i}" for i in range(1, 4)]
    + [f"CTOutletTemp0{i}" for i in range(1, 4)]
)
SYSTEM_TARGETS = ["PriChWDeltaT", "CWDeltaT"]
EQUIPMENT_TARGET_NAMES = CHILLER_TARGETS + PUMP_TARGETS + TOWER_TARGETS + SYSTEM_TARGETS

GROUP_SLICES = {
    "chiller": slice(0, 12),
    "pump": slice(12, 20),
    "tower": slice(20, 26),
    "system": slice(26, 28),
}

TIME_FEATURE_NAMES = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
]
FUTURE_PLAN_NAMES = (
    ["TotalRealTimeLoad", "OutdoorTdbin", "OutdoorWetTemp"]
    + TIME_FEATURE_NAMES
    + [f"ChAMPS0{i}" for i in range(1, 4)]
    + [f"ChChWTempSupplySetPoint0{i}" for i in range(1, 4)]
    + [f"PriChWPVSDFreq0{i}" for i in range(1, 4)]
    + [f"CWPVSDFreq0{i}" for i in range(1, 4)]
    + [f"CTVSDFreq0{i}" for i in range(1, 4)]
)


def _time_features(timestamps):
    hour = timestamps.dt.hour.to_numpy()
    weekday = timestamps.dt.dayofweek.to_numpy()
    month = timestamps.dt.month.to_numpy()
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    })


def build_future_control_plan(df):
    result = pd.DataFrame(index=range(len(df)))
    for name in ["TotalRealTimeLoad", "OutdoorTdbin", "OutdoorWetTemp"]:
        result[name] = pd.to_numeric(df[name], errors="coerce").to_numpy()
    result[TIME_FEATURE_NAMES] = _time_features(
        pd.to_datetime(df["timeStamp"]).reset_index(drop=True)
    )
    for i in range(1, 4):
        planned_name = f"ChAMPS0{i}"
        values = pd.to_numeric(df[planned_name], errors="coerce").to_numpy()
        result[planned_name] = (values > 1e-6).astype(np.float32)
    for prefix in (
        "ChChWTempSupplySetPoint", "PriChWPVSDFreq",
        "CWPVSDFreq", "CTVSDFreq",
    ):
        for i in range(1, 4):
            name = f"{prefix}0{i}"
            result[name] = pd.to_numeric(df[name], errors="coerce").to_numpy()
    result = result[FUTURE_PLAN_NAMES].astype(np.float32)
    if result.isna().any().any():
        raise ValueError("未来控制计划中存在缺失值")
    return result


def derive_equipment_targets(df):
    result = pd.DataFrame(index=range(len(df)))
    direct = (
        [f"ChPower0{i}" for i in range(1, 4)]
        + [f"ChRealtimeEfficiencyKW0{i}" for i in range(1, 4)]
        + [f"PriChWPPower0{i}" for i in range(1, 4)]
        + [f"CWPPower0{i}" for i in range(1, 4)]
        + ["PriChWFlow01", "CWFlow01"]
        + [f"CTPower0{i}" for i in range(1, 4)]
        + [f"CTOutletTemp0{i}" for i in range(1, 4)]
    )
    for name in direct:
        result[name] = pd.to_numeric(df[name], errors="coerce").to_numpy()
    for i in range(1, 4):
        result[f"EvapDeltaT0{i}"] = (
            pd.to_numeric(df[f"ChEnterEvapTemp0{i}"], errors="coerce").to_numpy()
            - pd.to_numeric(df[f"ChLeaveEvapTemp0{i}"], errors="coerce").to_numpy()
        )
        result[f"CondDeltaT0{i}"] = (
            pd.to_numeric(df[f"ChLeaveCondTemp0{i}"], errors="coerce").to_numpy()
            - pd.to_numeric(df[f"ChEnterCondTemp0{i}"], errors="coerce").to_numpy()
        )
    result["PriChWDeltaT"] = (
        pd.to_numeric(df["PriChWTempReturn01"], errors="coerce").to_numpy()
        - pd.to_numeric(df["PriChWTempSupply01"], errors="coerce").to_numpy()
    )
    result["CWDeltaT"] = (
        pd.to_numeric(df["CWTempReturn01"], errors="coerce").to_numpy()
        - pd.to_numeric(df["CWTempSupply01"], errors="coerce").to_numpy()
    )
    result = result[EQUIPMENT_TARGET_NAMES].astype(np.float32)
    if result.isna().any().any():
        raise ValueError("设备目标中存在缺失值")
    return result


def build_equipment_masks(df):
    masks = np.ones((len(df), len(EQUIPMENT_TARGET_NAMES)), dtype=np.float32)
    for i in range(1, 4):
        on = (pd.to_numeric(df[f"ChAMPS0{i}"], errors="coerce").to_numpy() > 1e-6)
        masks[:, 3 + i - 1] = on
        masks[:, 6 + i - 1] = on
        masks[:, 9 + i - 1] = on
        tower_on = (
            pd.to_numeric(df[f"CTVSDFreq0{i}"], errors="coerce").to_numpy() > 1e-6
        )
        masks[:, 23 + i - 1] = tower_on
    pri_on = np.column_stack([
        pd.to_numeric(df[f"PriChWPVSDFreq0{i}"], errors="coerce").to_numpy()
        for i in range(1, 4)
    ]).max(axis=1) > 1e-6
    cw_on = np.column_stack([
        pd.to_numeric(df[f"CWPVSDFreq0{i}"], errors="coerce").to_numpy()
        for i in range(1, 4)
    ]).max(axis=1) > 1e-6
    masks[:, 18] = pri_on
    masks[:, 19] = cw_on
    masks[:, 26] = pri_on
    masks[:, 27] = cw_on
    return masks


def build_masks_from_control_plan(plan_df):
    raw = plan_df.copy()
    for i in range(1, 4):
        raw[f"ChAMPS0{i}"] = raw[f"ChAMPS0{i}"]
    return build_equipment_masks(raw)


def efficiency_to_cop(efficiency):
    values = np.asarray(efficiency, dtype=float)
    return np.where(values > 1e-6, 3.517 / values, np.nan)
