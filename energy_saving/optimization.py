"""受供冷和训练支持范围约束的一步温度优化；能耗按相同小时集合比较。"""
import numpy as np
import pandas as pd

from .data import CONTROL_COLUMNS, FLOW_COLUMN, LOAD_COLUMN, RETURN_COLUMN, SUPPLY_COLUMN, cooling_kw


def choose_candidate(predictions, demand, rho=1000.0, cp=4.186, error_margin_kw=0.0):
    """第0行必须是基线；基线不满足供冷时不声称优化节能。"""
    power = predictions.iloc[:, :3].sum(axis=1).to_numpy()
    flow = predictions[FLOW_COLUMN].to_numpy()
    supply = predictions[SUPPLY_COLUMN].to_numpy()
    returned = predictions[RETURN_COLUMN].to_numpy()
    q = cooling_kw(flow, supply, returned, rho, cp)
    finite = np.isfinite(predictions.to_numpy()).all(axis=1)
    feasible = finite & (predictions.iloc[:, :3].to_numpy() >= 0).all(axis=1)
    feasible &= (flow > 0) & (returned > supply) & (q >= demand) & (power > 0)
    if not np.isfinite(demand) or demand <= 0:
        return 0, "no_positive_demand", False, power, q
    if not feasible[0]:
        return 0, "baseline_cooling_infeasible", False, power, q
    improving = feasible & (power < power[0] - error_margin_kw)
    improving[0] = False
    if not improving.any():
        reason = "no_feasible_alternative" if not feasible[1:].any() else "improvement_below_model_error"
        return 0, reason, True, power, q
    indices = np.flatnonzero(improving)
    best = int(indices[np.argmin(power[indices])])
    return best, "optimized", True, power, q


def optimize_day(model, samples, config, reliable, error_margin_kw):
    eval_x = samples["eval_x"]
    start = samples["eval_start"]
    rows = []
    for timestamp in pd.date_range(start, periods=24, freq="h"):
        row = {"timeStamp": timestamp, "duration_hours": 1.0, "eligible": False,
               "optimized": False, "reason": "missing_observations", "baseline_power_kw": np.nan,
               "optimized_power_kw": np.nan, "observed_power_kw": np.nan,
               "required_cooling_kw": np.nan, "baseline_cooling_kw": np.nan,
               "optimized_cooling_kw": np.nan, "candidate_count": 0, "support_count": 0,
               "support_reason": "missing_observations", "baseline_feasible": False,
               "candidate_selection_reason": "not_evaluated"}
        if timestamp in samples["hourly"].index:
            obs = samples["hourly"].loc[timestamp]
            row["observed_power_kw"] = obs[["chiller_power", "pump_power", "tower_power"]].sum(min_count=3)
            row["required_cooling_kw"] = obs[LOAD_COLUMN]
        if timestamp not in eval_x.index:
            rows.append(row)
            continue
        baseline = eval_x.loc[timestamp]
        candidates, support_reason, support_count = model.supported_candidates(
            baseline, max_candidates=config.max_candidates, min_neighbors=config.min_neighbors,
            context_radius=config.context_radius, joint_radius=config.joint_radius,
            min_change=config.min_temperature_change,
        )
        predictions = model.predict(candidates)
        best, reason, feasible, powers, cooling = choose_candidate(
            predictions, float(baseline[LOAD_COLUMN]), config.rho, config.cp, error_margin_kw,
        )
        row.update(support_reason=support_reason, baseline_feasible=bool(feasible),
                   candidate_selection_reason=reason)
        eligible = reliable and support_reason == "supported" and feasible
        if not eligible:
            best = 0
            reason = "global_model_or_unit_check_failed" if not reliable else (
                support_reason if support_reason != "supported" else reason
            )
        row.update(eligible=bool(eligible), optimized=bool(eligible and best != 0), reason=reason,
                   baseline_power_kw=float(powers[0]), optimized_power_kw=float(powers[best]),
                   baseline_cooling_kw=float(cooling[0]), optimized_cooling_kw=float(cooling[best]),
                   candidate_count=len(candidates), support_count=support_count,
                   model_error_margin_kw=error_margin_kw)
        for c in CONTROL_COLUMNS:
            row[f"baseline_{c}"] = float(baseline[c])
            row[f"optimized_{c}"] = float(candidates.iloc[best][c])
        for c in predictions.columns:
            row[f"baseline_predicted_{c}"] = float(predictions.iloc[0][c])
            row[f"optimized_predicted_{c}"] = float(predictions.iloc[best][c])
        rows.append(row)
    return pd.DataFrame(rows)


def energy_summary(detail):
    valid = detail.eligible.astype(bool)
    hours = float(detail.loc[valid, "duration_hours"].sum())
    observed = detail.observed_power_kw.notna()
    baseline = optimized = saving = rate = None
    if valid.any():
        baseline = float((detail.loc[valid, "baseline_power_kw"] * detail.loc[valid, "duration_hours"]).sum())
        optimized = float((detail.loc[valid, "optimized_power_kw"] * detail.loc[valid, "duration_hours"]).sum())
        saving = baseline - optimized
        rate = saving / baseline * 100 if baseline > 0 else None
    return {
        "project": "测试项目A", "evaluation_hours": len(detail), "eligible_hours": hours,
        "coverage_percent": hours / len(detail) * 100 if len(detail) else 0,
        "full_day_reportable": bool(len(detail) == 24 and valid.all()),
        "simulation_baseline_kwh": baseline, "simulation_optimized_kwh": optimized,
        "saving_kwh": saving, "saving_percent": rate,
        "observed_baseline_kwh": float((detail.loc[observed, "observed_power_kw"] * detail.loc[observed, "duration_hours"]).sum()) if observed.any() else None,
        "observed_hours": int(observed.sum()),
        "scope": "eligible_hours_only_no_extrapolation",
        "method": "one_step_conditional_offline_simulation",
    }
