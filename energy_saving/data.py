"""原始观测构造一步响应样本，不插值目标、不使用评价日建模。"""
import numpy as np
import pandas as pd

from load_forecasting.data import INVALID_VALUES, _time_features


CONTROL_COLUMNS = [f"ChChWTempSupplySetting0{i}" for i in range(1, 5)]
POWER_GROUPS = {
    "chiller_power": [f"ChPower0{i}" for i in range(1, 5)],
    "pump_power": ([f"PriChWPPower0{i}" for i in range(1, 7)]
                   + [f"CWPPower0{i}" for i in range(1, 7)]),
    "tower_power": [f"CTPower0{i}" for i in range(1, 5)],
}
POWER_COLUMNS = [c for cols in POWER_GROUPS.values() for c in cols]
FLOW_COLUMN = "PriChWFlow"
SUPPLY_COLUMN = "PriChWTempSupply"
RETURN_COLUMN = "PriChWTempReturn"
LOAD_COLUMN = "TotalRealTimeLoad"
WEATHER_COLUMNS = ["OutdoorTdbin", "OutdoorWetTemp"]
RESPONSE_COLUMNS = list(POWER_GROUPS) + [FLOW_COLUMN, SUPPLY_COLUMN, RETURN_COLUMN]
STATE_COLUMNS = [LOAD_COLUMN, *WEATHER_COLUMNS, *RESPONSE_COLUMNS,
                 "CWFlow01", "CWTempSupply", "CWTempReturn"]
STATUS_COLUMNS = [f"on_{c}" for c in POWER_COLUMNS]


def read_hourly(path, min_samples=6):
    raw = pd.read_excel(path, sheet_name="历史数据")
    columns = list(dict.fromkeys([
        LOAD_COLUMN, *WEATHER_COLUMNS, *CONTROL_COLUMNS, *POWER_COLUMNS,
        FLOW_COLUMN, SUPPLY_COLUMN, RETURN_COLUMN, "CWFlow01", "CWTempSupply", "CWTempReturn",
    ]))
    missing = [c for c in ["timeStamp", *columns] if c not in raw]
    if missing:
        raise ValueError(f"测试项目A缺少字段：{missing}")
    raw = raw[["timeStamp", *columns]].copy()
    raw["timeStamp"] = pd.to_datetime(raw.timeStamp, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    raw = raw.dropna(subset=["timeStamp"]).sort_values("timeStamp")
    if raw.timeStamp.duplicated().any():
        raise ValueError("原始观测存在重复时间戳")
    for c in columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").replace(
            [*INVALID_VALUES, np.inf, -np.inf], np.nan,
        )
    nonnegative = [LOAD_COLUMN, FLOW_COLUMN, "CWFlow01", *POWER_COLUMNS]
    raw[nonnegative] = raw[nonnegative].where(raw[nonnegative] >= 0)
    raw[CONTROL_COLUMNS] = raw[CONTROL_COLUMNS].where(raw[CONTROL_COLUMNS] > 0)
    indexed = raw.set_index("timeStamp")
    hourly = indexed.resample("1h").mean().where(indexed.resample("1h").count() >= min_samples)
    for name, cols in POWER_GROUPS.items():
        hourly[name] = hourly[cols].sum(axis=1, min_count=len(cols))
    return hourly


def build_samples(hourly, eval_date, on_threshold_kw=1.0):
    """运行组合由基线实测功率推断；它是固定条件，不是优化变量。"""
    hourly = hourly.sort_index()
    if hourly.index.has_duplicates:
        raise ValueError("小时数据存在重复时间戳")
    start = pd.Timestamp(eval_date).normalize()
    end = start + pd.Timedelta(days=1)
    x = hourly[[LOAD_COLUMN, *WEATHER_COLUMNS]].copy()
    for c in STATE_COLUMNS + CONTROL_COLUMNS:
        x[f"previous_{c}"] = hourly[c].shift(1)
    time = _time_features(pd.Series(hourly.index))
    time.index = hourly.index
    x = pd.concat([x, time], axis=1)
    states = hourly[POWER_COLUMNS].gt(on_threshold_kw).astype(int)
    states.columns = STATUS_COLUMNS
    x = pd.concat([x, states, hourly[CONTROL_COLUMNS]], axis=1)
    y = hourly[RESPONSE_COLUMNS].copy()
    previous = pd.Series(hourly.index, index=hourly.index).shift(1)
    complete = (
        np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        & hourly[POWER_COLUMNS].notna().all(axis=1)
        & (pd.Series(hourly.index, index=hourly.index) - previous).eq(pd.Timedelta(hours=1))
    )
    current_eval = (hourly.index >= start) & (hourly.index < end)
    previous_eval = (previous >= start) & (previous < end)
    model_mask = complete & ~current_eval & ~previous_eval
    eval_mask = complete & current_eval
    return {"x": x.loc[model_mask], "y": y.loc[model_mask],
            "eval_x": x.loc[eval_mask], "eval_y": y.loc[eval_mask],
            "hourly": hourly, "eval_start": start}


def cooling_kw(flow, supply, return_temp, rho=1000.0, cp=4.186):
    """flow:m³/h，rho:kg/m³，cp:kJ/(kg·K)，输出kW。"""
    return rho * cp * np.asarray(flow) / 3600 * (np.asarray(return_temp) - np.asarray(supply))


def thermal_check(x, y, rho=1000.0, cp=4.186, max_mape=20.0):
    q = cooling_kw(y[FLOW_COLUMN], y[SUPPLY_COLUMN], y[RETURN_COLUMN], rho, cp)
    demand = x[LOAD_COLUMN].to_numpy()
    valid = (demand > 1e-6) & np.isfinite(q) & (q > 0)
    mape = float(np.mean(np.abs(q[valid] - demand[valid]) / demand[valid]) * 100) if valid.any() else None
    return {"hours": int(valid.sum()), "balance_mape_percent": mape,
            "passed": bool(valid.sum() >= 24 and mape is not None and mape <= max_mape),
            "max_mape_percent": max_mape}
