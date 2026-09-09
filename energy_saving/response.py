"""固定超参数随机森林响应模型及训练数据支持范围。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .data import CONTROL_COLUMNS, RESPONSE_COLUMNS, STATUS_COLUMNS


class ResponseModel:
    def __init__(self, n_estimators=100, min_samples_leaf=5, random_state=42):
        self.forest = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=-1,
        )

    def fit(self, x, y):
        if len(x) < 20:
            raise ValueError("设备响应模型有效建模样本不足20小时")
        self.features = list(x.columns)
        self.context_columns = [c for c in x if c not in CONTROL_COLUMNS + STATUS_COLUMNS]
        self.reference = x.copy()
        self.control_min = x[CONTROL_COLUMNS].min().to_numpy()
        self.control_max = x[CONTROL_COLUMNS].max().to_numpy()
        self.context_scale = x[self.context_columns].std(ddof=0).to_numpy()
        self.context_scale = np.where(self.context_scale < 1e-6, 1, self.context_scale)
        self.control_scale = np.maximum(self.control_max - self.control_min, 0.1)
        self.y_mean = y[RESPONSE_COLUMNS].mean().to_numpy()
        self.y_std = y[RESPONSE_COLUMNS].std(ddof=0).to_numpy()
        self.y_std = np.where(self.y_std < 1e-6, 1, self.y_std)
        self.forest.fit(x[self.features].to_numpy(), (y[RESPONSE_COLUMNS].to_numpy() - self.y_mean) / self.y_std)
        return self

    def predict(self, x):
        pred = self.forest.predict(x[self.features].to_numpy()) * self.y_std + self.y_mean
        return pd.DataFrame(pred, index=x.index, columns=RESPONSE_COLUMNS)

    def supported_candidates(self, baseline, max_candidates=64, min_neighbors=5,
                             context_radius=1.0, joint_radius=1.0, min_change=0.05):
        """同组合、相似工况的实测联合设定；停用冷机设定固定为基线。"""
        controls = baseline[CONTROL_COLUMNS].to_numpy(dtype=float)
        status = baseline[STATUS_COLUMNS].to_numpy(dtype=int)
        active = status[:4].astype(bool)
        pool = self.reference.loc[
            (self.reference[STATUS_COLUMNS].to_numpy() == status).all(axis=1)
        ]
        baseline_frame = pd.DataFrame([baseline], columns=self.features)
        if ((controls < self.control_min) | (controls > self.control_max)).any():
            return baseline_frame, "baseline_temperature_out_of_range", 0
        if len(pool) < min_neighbors or not active.any():
            return baseline_frame, "no_same_equipment_support", 0
        ctx = baseline[self.context_columns].to_numpy(dtype=float)
        distance = np.sqrt(np.mean(((pool[self.context_columns].to_numpy() - ctx) / self.context_scale) ** 2, axis=1))
        pool = pool.iloc[np.flatnonzero(distance <= context_radius)]
        if len(pool) < min_neighbors:
            return baseline_frame, "insufficient_context_support", len(pool)
        ctrl = pool[CONTROL_COLUMNS].to_numpy(dtype=float)
        ctx_dist2 = np.mean(((pool[self.context_columns].to_numpy() - ctx) / self.context_scale) ** 2, axis=1)
        def support(candidate):
            ctrl_dist2 = np.mean(((ctrl - candidate) / self.control_scale) ** 2, axis=1)
            return int((np.sqrt(ctx_dist2 + ctrl_dist2) <= joint_radius).sum())
        baseline_count = support(controls)
        if baseline_count < min_neighbors:
            return baseline_frame, "baseline_out_of_support", baseline_count
        if np.all(np.ptp(ctrl[:, active], axis=0) < min_change):
            return baseline_frame, "insufficient_control_variation", baseline_count
        # 在近邻中取完整控制组合，不独立拼接各列最高/最低值。
        ordered = pool.iloc[np.argsort(ctx_dist2)][CONTROL_COLUMNS].drop_duplicates()
        rows = [baseline.copy()]
        for candidate in ordered.to_numpy():
            candidate = candidate.copy()
            candidate[~active] = controls[~active]
            if ((candidate < self.control_min) | (candidate > self.control_max)).any():
                continue
            if np.max(np.abs(candidate - controls)) < min_change:
                continue
            if support(candidate) < min_neighbors:
                continue
            row = baseline.copy()
            row[CONTROL_COLUMNS] = candidate
            rows.append(row)
            if len(rows) >= max_candidates:
                break
        return pd.DataFrame(rows, columns=self.features), (
            "supported" if len(rows) > 1 else "no_supported_alternative"
        ), baseline_count


def response_metrics(actual, predicted, stage):
    records = []
    for name in [*RESPONSE_COLUMNS, "total_power"]:
        a = actual.iloc[:, :3].sum(axis=1).to_numpy() if name == "total_power" else actual[name].to_numpy()
        p = predicted.iloc[:, :3].sum(axis=1).to_numpy() if name == "total_power" else predicted[name].to_numpy()
        nz = np.abs(a) > 1e-6
        records.append({"stage": stage, "target": name, "hours": len(a),
                        "mae": float(np.mean(np.abs(p - a))),
                        "rmse": float(np.sqrt(np.mean((p - a) ** 2))),
                        "mape_percent": float(np.mean(np.abs(p[nz] - a[nz]) / np.abs(a[nz])) * 100) if nz.any() else np.nan})
    return pd.DataFrame(records)
