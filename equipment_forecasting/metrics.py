"""设备参数逐目标和分组指标。"""
import numpy as np

from .targets import EQUIPMENT_TARGET_NAMES, GROUP_SLICES


def _metrics(pred, true, mask):
    valid = mask.astype(bool)
    error = pred - true
    mae = float(np.mean(np.abs(error[valid]))) if valid.any() else float("nan")
    rmse = (
        float(np.sqrt(np.mean(error[valid] ** 2)))
        if valid.any() else float("nan")
    )
    mape_mask = valid & (np.abs(true) > 1e-6)
    mape = (
        float(np.mean(np.abs(error[mape_mask]) / np.abs(true[mape_mask])) * 100)
        if mape_mask.any() else float("nan")
    )
    return {"mae": mae, "rmse": rmse, "mape": mape}


def grouped_metrics(prediction, target, mask):
    pred = np.asarray(prediction)
    true = np.asarray(target)
    valid = np.asarray(mask)
    targets = {
        name: _metrics(pred[..., i], true[..., i], valid[..., i])
        for i, name in enumerate(EQUIPMENT_TARGET_NAMES)
    }
    groups = {
        name: _metrics(pred[..., part], true[..., part], valid[..., part])
        for name, part in GROUP_SLICES.items()
    }
    return {"targets": targets, "groups": groups}
