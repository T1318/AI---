"""项目A供水温度离线优化。结果为同一响应模型下的条件仿真，不是实测节能。"""
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from energy_saving.data import CONTROL_COLUMNS, RESPONSE_COLUMNS, build_samples, read_hourly, thermal_check
from energy_saving.optimization import energy_summary, optimize_day
from energy_saving.response import ResponseModel, response_metrics
from train_a_july16 import JULY16_DATA


@dataclass
class SimulationConfig:
    data: str = JULY16_DATA
    eval_date: str = "2025-07-16"
    output_dir: str = "outputs/energy_saving_A"
    max_candidates: int = 64
    n_estimators: int = 100
    min_samples_leaf: int = 5
    random_state: int = 42
    min_samples_per_hour: int = 6
    on_threshold_kw: float = 1.0
    min_neighbors: int = 5
    context_radius: float = 1.0
    joint_radius: float = 1.0
    min_temperature_change: float = 0.05
    rho: float = 1000.0
    cp: float = 4.186
    flow_unit: str = "m3/h"
    temperature_unit: str = "degC"
    power_unit: str = "kW"
    cooling_unit: str = "kW"
    max_balance_mape: float = 20.0
    max_power_mape: float = 20.0
    max_flow_mape: float = 20.0
    max_temperature_mae: float = 1.0


def default_config():
    return SimulationConfig()


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def run(config, hourly=None):
    if config.max_candidates < 2 or config.min_neighbors < 1:
        raise ValueError("max_candidates至少2，min_neighbors至少1")
    if config.rho <= 0 or config.cp <= 0:
        raise ValueError("水密度和比热必须为正")
    if hourly is None:
        print("读取原始Excel并构造完整小时观测……", flush=True)
        hourly = read_hourly(config.data, config.min_samples_per_hour)
    samples = build_samples(hourly, config.eval_date, config.on_threshold_kw)
    x, y = samples["x"], samples["y"]
    if len(x) < 60:
        raise ValueError("去掉评价日后完整建模样本少于60小时")
    cut = int(len(x) * 0.8)
    # 固定时间顺序80/20内部验证，无随机划分，也不在最终评价日选参数。
    internal_x, internal_y = x.iloc[:cut], y.iloc[:cut]
    validation_x, validation_y = x.iloc[cut:], y.iloc[cut:]
    def new_model():
        return ResponseModel(config.n_estimators, config.min_samples_leaf, config.random_state)
    internal_model = new_model().fit(internal_x, internal_y)
    internal_pred = internal_model.predict(validation_x)
    validation = response_metrics(validation_y, internal_pred, "internal_time_holdout")
    indexed = validation.set_index("target")
    model_checks = {
        "power_mape": bool(np.isfinite(indexed.loc["total_power", "mape_percent"]) and indexed.loc["total_power", "mape_percent"] <= config.max_power_mape),
        "flow_mape": bool(np.isfinite(indexed.loc["PriChWFlow", "mape_percent"]) and indexed.loc["PriChWFlow", "mape_percent"] <= config.max_flow_mape),
        "temperature_mae": bool(indexed.loc[["PriChWTempSupply", "PriChWTempReturn"], "mae"].max() <= config.max_temperature_mae),
    }
    error_margin = float(np.quantile(np.abs(
        internal_pred.iloc[:, :3].sum(axis=1) - validation_y.iloc[:, :3].sum(axis=1)
    ), 0.9))
    units_match = (config.flow_unit, config.temperature_unit, config.power_unit, config.cooling_unit) == ("m3/h", "degC", "kW", "kW")
    balance = thermal_check(x, y, config.rho, config.cp, config.max_balance_mape)
    reliable = bool(units_match and balance["passed"] and all(model_checks.values()))
    print(f"完整建模样本={len(x)}，评价样本={len(samples['eval_x'])}，全局可信度检查={reliable}", flush=True)
    model = new_model().fit(x, y)
    if len(samples["eval_x"]):
        validation = pd.concat([
            validation, response_metrics(samples["eval_y"], model.predict(samples["eval_x"]), "evaluation_day_baseline"),
        ], ignore_index=True)
    detail = optimize_day(model, samples, config, reliable, error_margin)
    summary = energy_summary(detail)
    bounds = {c: {"min": float(lo), "max": float(hi)}
              for c, lo, hi in zip(CONTROL_COLUMNS, model.control_min, model.control_max)}
    metadata = {
        "project": "测试项目A", "configuration": asdict(config),
        "evaluation_mode": "one_step_conditional_offline_simulation",
        "demand_source": "evaluation_hour_observed_TotalRealTimeLoad",
        "weather_source": "evaluation_hour_observations",
        "baseline": "historical_setpoints_evaluated_by_same_response_model",
        "unit_basis": "user_competition_template_units_plus_historical_thermal_balance_check",
        "units_match": units_match, "thermal_balance": balance,
        "internal_model_checks": model_checks, "global_checks_passed": reliable,
        "power_error_margin_kw": error_margin,
        "power_margin_method": "internal_holdout_90th_percentile_absolute_total_power_error_not_confidence_interval",
        "control_bounds": bounds,
        "model_period": [str(x.index.min()), str(x.index.max())],
        "internal_fit_end": str(internal_x.index[-1]),
        "internal_validation_start": str(validation_x.index[0]),
        "excluded_date_and_previous_state_date": str(samples["eval_start"].date()),
        "fit_hours": len(x), "internal_validation_hours": len(validation_x),
        "status_definition": "each_baseline_observed_device_power_above_on_threshold_kw; fixed_for_all_candidates",
        "formula": "Q[kW]=rho[kg/m3]*cp[kJ/(kg*K)]*Flow[m3/h]/3600*(Treturn-Tsupply); E[kWh]=sum(P[kW]*dt[h])",
        "reason_counts": detail.reason.value_counts().to_dict(),
        "limitations": [
            "Historical temperature extrema are not certified site safety limits.",
            "Observed conditional associations do not prove causal control effects.",
            "Each hour restarts from observed history, not a continuous counterfactual trajectory.",
            "State schedule is inferred from observed power and held fixed; no start/stop optimization.",
            "Unsupported or infeasible hours are excluded, not extrapolated to a whole-day saving.",
            "Whole-day report requires full_day_reportable=true; all results remain simulated potential.",
        ],
    }
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata,
                 "feature_names": model.features, "response_names": RESPONSE_COLUMNS}, output / "response_model.joblib")
    validation.to_csv(output / "model_validation.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(output / "hourly_optimization.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    pd.DataFrame([summary]).to_csv(output / "energy_summary.csv", index=False, encoding="utf-8-sig")
    (output / "analysis_metadata.json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return detail, summary, metadata


if __name__ == "__main__":
    config = default_config()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=config.data)
    parser.add_argument("--eval-date", default=config.eval_date)
    parser.add_argument("--output-dir", default=config.output_dir)
    parser.add_argument("--max-candidates", type=int, default=config.max_candidates)
    parser.add_argument("--n-estimators", type=int, default=config.n_estimators)
    options = parser.parse_args()
    for key, value in vars(options).items():
        setattr(config, key, value)
    run(config)
