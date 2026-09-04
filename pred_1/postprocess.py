# -*- coding: utf-8 -*-
"""
postprocess.py
==============
LoadTransformer 训练后推理脚本（后处理 / 推理入口）。

流程严格按视频中（聊天记录 2026-08-30 03:05 韩兄→涛哥）确认的 4 步：
    1. 读取训练好的模型参数    -- LoadTransformer.pt
    2. 读取待预测数据           -- 项目 A 历史数据 xlsx（5min 粒度）
    3. 进行滚动预测             -- 默认：历史末段 14 天逐日 24h 回测
    4. 绘制预测结果和真实结果曲线图 -- PNG + CSV 一并落盘

用法
----
    # 默认回测 14 天（用项目 A 训练数据，得到 pred vs actual 对比曲线）
    python postprocess.py

    # 指定 checkpoint 与数据
    python postprocess.py --ckpt LoadTransformer.pt --data 项目A.xlsx

    # 单步预测未来 24h（无真实值，仅预测曲线 + CSV）
    python postprocess.py --mode forecast --days 1

    # 回测天数改为 7
    python postprocess.py --backtest 7

输出
----
    postprocess_out/
        backtest_detail.csv      # time, actual, pred（仅回测模式）
        pred_vs_actual.png       # 回测对比曲线
        forecast_detail.csv      # time, predicted_load（仅 forecast 模式）
        forecast_curve.png       # 未来预测曲线

依赖
----
    numpy, pandas, torch, PyWavelets, matplotlib, openpyxl
"""

from __future__ import annotations

import argparse
import os
import sys

# 让 Windows GBK 控制台也能正常打印中文与特殊符号
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境也能稳定写 PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import torch

# 让脚本能找到同目录的 models_registry（用于 LoadTransformer 的真实结构 Nanwang_Transformer）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_registry  # noqa: E402

# ---------------------------------------------------------------------------
# 常量（与训练侧口径严格对齐）
# ---------------------------------------------------------------------------
WAVELET = "db4"
WAVE_LEVEL = 3
TARGET_COLUMN = "TotalRealTimeLoad"

# 默认路径（脚本放在 C:/Users/Administrator/Desktop/赛马/ 下时可直接使用）
DEFAULT_CKPT = r"C:/Users/Administrator/Desktop/LoadTransformer.pt"
DEFAULT_DATA = r"C:/Users/Administrator/Desktop/赛马/数据/训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx"
DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "赛马ai"
)
DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "postprocess_out"
)


# =====================================================================
# Step 1: 读取训练好的模型参数
# =====================================================================
def load_checkpoint(ckpt_path: str, models_dir: str):
    """读取 LoadTransformer.pt，返回 (model, meta)。

    meta 包含：model_type / input_dim / feature_names / wavelet_columns /
              x_mean / x_std / y_mean / y_std / input_hours / horizon / output_form
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    required = ("model_type", "model", "model_config", "input_dim",
                "feature_names", "wavelet_columns", "x_mean", "x_std",
                "y_mean", "y_std", "input_hours", "horizon")
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError("checkpoint 缺少字段: " + ", ".join(missing))

    meta = {
        "model_type": ckpt["model_type"],
        "model_config": ckpt["model_config"],
        "input_dim": int(ckpt["input_dim"]),
        "feature_names": list(ckpt["feature_names"]),
        "wavelet_columns": list(ckpt["wavelet_columns"]),
        "x_mean": np.asarray(ckpt["x_mean"], dtype=np.float64),
        "x_std": np.asarray(ckpt["x_std"], dtype=np.float64),
        "y_mean": float(ckpt["y_mean"]),
        "y_std": float(ckpt["y_std"]),
        "input_hours": int(ckpt["input_hours"]),
        "horizon": int(ckpt["horizon"]),
        "output_form": models_registry.get_output_form(ckpt["model_type"]),
    }

    # 用队友的模型类动态重建并载入权重（不拷贝代码，队友更新模型文件后自动跟随）
    model = models_registry.build_model(
        meta["model_type"], meta["model_config"],
        meta["input_dim"], meta["horizon"], models_dir,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, meta


# =====================================================================
# Step 2: 读取待预测数据
# =====================================================================
def load_hourly(data_path: str) -> pd.DataFrame:
    """读 xlsx → 跳过中文说明行 → 5min→1h 均值重采样。"""
    raw = pd.read_excel(data_path)
    if "timeStamp" not in raw.columns:
        first = raw.columns[0]
        raw = raw.rename(columns={first: "timeStamp"})
    # 第 0 行通常是中文参数名说明
    raw = raw.iloc[1:].copy()
    raw["timeStamp"] = pd.to_datetime(raw["timeStamp"])
    raw = raw.set_index("timeStamp")
    raw = raw.apply(pd.to_numeric, errors="coerce")
    hourly = raw.resample("1h").mean()
    if hourly.empty:
        raise ValueError(f"数据文件没有可用的小时级数据: {data_path}")
    return hourly


def _modwt_decompose(x: np.ndarray):
    """3 层平稳小波分解 + MODWT 归一化（cA3/√8、cD3/√8、cD2/2、cD1/√2）。

    口径依据：与 checkpoint x_mean/x_std 统计精确吻合（std 比值 = (√2)^layer）。
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    n_pad = n + ((-(n)) % (2 ** WAVE_LEVEL))
    pad = (np.concatenate([x, np.repeat(x[-1], n_pad - n)])
           if n_pad > n else x)
    # trim_approx=True: [cA3, cD3, cD2, cD1]
    coefs = pywt.swt(pad, WAVELET, level=WAVE_LEVEL, trim_approx=True)
    cA3, cD3, cD2, cD1 = [np.asarray(c[:n], dtype=np.float64) for c in coefs]
    g = 2.0 ** (WAVE_LEVEL / 2.0)
    return cA3 / g, cD3 / g, cD2 / 2.0, cD1 / np.sqrt(2.0)


def build_features(hourly: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """按 checkpoint 的 feature_names 构造特征矩阵（116 维）。"""
    wave_cols = meta["wavelet_columns"]
    missing = [c for c in wave_cols if c not in hourly.columns]
    if missing:
        raise ValueError(
            "数据缺少 checkpoint 所需的原始特征列: " + ", ".join(missing)
        )

    out = {}
    for c in wave_cols:
        s = hourly[c].astype(float).copy()
        s = s.interpolate(limit_direction="both")  # 小波分解不允许 NaN
        cA3, cD3, cD2, cD1 = _modwt_decompose(s.values)
        out[c] = s.values
        out[f"{c}_cA3"] = cA3
        out[f"{c}_cD3"] = cD3
        out[f"{c}_cD2"] = cD2
        out[f"{c}_cD1"] = cD1
    feat = pd.DataFrame(out, index=hourly.index)

    # 时间编码（hour/weekday/month 的 sin/cos，weekday 取 dayofweek，0=周一）
    idx = hourly.index
    h = np.asarray(idx.hour, dtype=float)
    d = np.asarray(idx.dayofweek, dtype=float)
    m = np.asarray(idx.month, dtype=float)
    feat["hour_sin"] = np.sin(2 * np.pi * h / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * h / 24)
    feat["weekday_sin"] = np.sin(2 * np.pi * d / 7)
    feat["weekday_cos"] = np.cos(2 * np.pi * d / 7)
    feat["month_sin"] = np.sin(2 * np.pi * m / 12)
    feat["month_cos"] = np.cos(2 * np.pi * m / 12)

    # 严格按训练时的 feature_names 顺序重排
    feat = feat[meta["feature_names"]]

    if not np.isfinite(feat.values).all():
        bad = feat.columns[~np.isfinite(feat.values).all(axis=0)].tolist()
        raise ValueError("特征中存在无法填补的 NaN/Inf: " + ", ".join(bad))
    return feat


def standardize(x: pd.DataFrame, meta: dict) -> np.ndarray:
    """用 checkpoint 中的均值/方差对特征做 z-score 标准化。"""
    arr = (x.values - meta["x_mean"]) / meta["x_std"]
    return arr.astype(np.float64)


# =====================================================================
# Step 3: 滚动预测
# =====================================================================
def predict_one_block(model, meta: dict, x_hist: np.ndarray) -> np.ndarray:
    """单步预测 24h，返回反标准化后的原始量纲。"""
    window = x_hist[-meta["input_hours"]:]
    t = torch.from_numpy(window).float().unsqueeze(0)  # (1, 168, 116)
    with torch.no_grad():
        out = model(t)
    pred = models_registry.adapt_output(out, meta["horizon"], meta["model_type"])
    if hasattr(pred, "numpy"):
        pred = pred.numpy()
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    return pred * meta["y_std"] + meta["y_mean"]


def rolling_backtest(model, meta: dict, hourly: pd.DataFrame,
                     days: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """对历史最后 days 天逐日做 24h 预测，并与真实值对比算 MAE/RMSE/MAPE。

    关键：每个预测起点都只用此前历史重新构造小波特征，避免 SWT 边界系数
    泄露未来真值造成回测虚高。
    """
    if days <= 0:
        raise ValueError("backtest 天数必须为正整数")
    if TARGET_COLUMN not in hourly.columns:
        raise ValueError(f"回测需要真实值列 {TARGET_COLUMN}")

    forecast_hours = meta["horizon"]
    first_end = len(hourly) - days * forecast_hours
    if first_end < meta["input_hours"]:
        max_days = max(0, (len(hourly) - meta["input_hours"]) // forecast_hours)
        raise ValueError(
            f"数据不足以回测 {days} 天：需要至少 "
            f"{meta['input_hours'] + days * forecast_hours} 小时，"
            f"当前只有 {len(hourly)} 小时（最多可回测 {max_days} 天）"
        )

    rows = []
    for d in range(days):
        end = first_end + d * forecast_hours
        history = hourly.iloc[:end]
        feat = build_features(history, meta)
        x_hist = standardize(feat, meta)
        pred = predict_one_block(model, meta, x_hist)

        t_idx = hourly.index[end:end + forecast_hours]
        actual = hourly[TARGET_COLUMN].reindex(t_idx).values
        for tt, a, p in zip(t_idx, actual, pred):
            rows.append({"time": tt, "actual": a, "pred": p})

    bt = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    mask = np.isfinite(bt["actual"]) & (np.abs(bt["actual"]) > 1e-6)  # 剔除零负荷
    used = int(mask.sum())
    excluded = int((~mask & np.isfinite(bt["actual"])).sum())
    if used == 0:
        raise ValueError("回测区间内无有效非零负荷小时，无法计算指标")
    mae = float(np.mean(np.abs(bt.loc[mask, "pred"] - bt.loc[mask, "actual"])))
    rmse = float(np.sqrt(np.mean((bt.loc[mask, "pred"] - bt.loc[mask, "actual"]) ** 2)))
    mape = float(np.mean(np.abs(bt.loc[mask, "pred"] - bt.loc[mask, "actual"])
                         / np.abs(bt.loc[mask, "actual"])) * 100)
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE_%": mape,
               "hours_used": used, "hours_zero_excluded": excluded}
    return bt, metrics


def rolling_forecast(model, meta: dict, hourly: pd.DataFrame,
                     days: int) -> pd.DataFrame:
    """对未来 days 天做滚动预测；未知特征列用上周同时刻值外推。"""
    cur = hourly.copy()
    all_pred = []
    remaining = days * meta["horizon"]
    while remaining > 0:
        feat = build_features(cur, meta)
        x_hist = standardize(feat, meta)
        if len(x_hist) < meta["input_hours"]:
            raise SystemExit(
                f"数据不足 {meta['input_hours']}h（现有 {len(x_hist)}h），无法预测"
            )
        pred = predict_one_block(model, meta, x_hist)
        pred = pred[:remaining]
        times = pd.date_range(cur.index[-1] + pd.Timedelta(hours=1),
                              periods=len(pred), freq="h")
        all_pred.append(pd.DataFrame({"time": times, "predicted_load": pred}))
        remaining -= len(pred)
        if remaining > 0:
            cur = _extend_with_pred(cur, pred)
    return pd.concat(all_pred, ignore_index=True)


def _extend_with_pred(hourly: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """把预测写回历史（目标列写预测值，其他列用上周同时刻值外推）。"""
    last = hourly.index[-1]
    new_idx = pd.date_range(last + pd.Timedelta(hours=1),
                            periods=len(pred), freq="h")
    ext = pd.DataFrame(index=new_idx, columns=hourly.columns, dtype=float)
    for c in hourly.columns:
        if c == TARGET_COLUMN:
            ext[c] = pred
        else:
            shifted = hourly[c].shift(24 * 7).reindex(new_idx)
            ext[c] = shifted.fillna(hourly[c].iloc[-1])
    return pd.concat([hourly, ext])


# =====================================================================
# Step 4: 绘制预测 vs 真实曲线
# =====================================================================
def plot_backtest(bt: pd.DataFrame, out_png: str, days: int,
                  metrics: dict) -> None:
    """绘制回测对比曲线：真实值（粗蓝）+ 预测值（粗橙），标题带指标。"""
    if bt.empty:
        raise ValueError("没有可绘制的回测数据")
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(bt["time"], bt["actual"], label="Actual load",
            linewidth=1.8, color="#1f77b4")
    ax.plot(bt["time"], bt["pred"], label="Predicted load",
            linewidth=1.6, color="#ff7f0e")
    title = (f"Backtest: last {days} day(s)  |  "
             f"MAPE={metrics['MAPE_%']:.2f}%  "
             f"MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
             f"(n={metrics['hours_used']}, zero-excluded={metrics['hours_zero_excluded']})")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cooling load")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_forecast(fc: pd.DataFrame, out_png: str, days: int) -> None:
    """绘制未来预测曲线（无真实值，仅预测线）。"""
    if fc.empty:
        raise ValueError("没有可绘制的预测数据")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(fc["time"], fc["predicted_load"], label="Predicted load",
            linewidth=1.8, color="#ff7f0e")
    ax.set_title(f"Forecast: next {days} day(s)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cooling load")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# =====================================================================
# 入口
# =====================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="LoadTransformer 后处理 / 推理入口：读模型→读数据→滚动预测→画图"
    )
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint .pt 路径")
    ap.add_argument("--data", default=DEFAULT_DATA, help="待预测数据 xlsx 路径")
    ap.add_argument("--mode", choices=["backtest", "forecast"], default="backtest",
                    help="运行模式：backtest=历史回测有真值对比 / forecast=未来预测")
    ap.add_argument("--days", type=int, default=1,
                    help="forecast 模式下的滚动预测天数（默认 1 = 未来 24h）")
    ap.add_argument("--backtest", dest="backtest_days", type=int, default=14,
                    help="backtest 模式下的回测天数（默认 14）")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                    help="队友模型代码目录（默认 ./赛马ai）")
    ap.add_argument("--out", default=DEFAULT_OUTDIR, help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"输出目录: {args.out}")

    # Step 1: 读模型
    print(f"[1/4] 读取 checkpoint: {args.ckpt}")
    model, meta = load_checkpoint(args.ckpt, args.models_dir)
    print(f"      模型 {meta['model_type']}（{meta['output_form']} 形态），"
          f"输入 {meta['input_dim']} 维 × {meta['input_hours']}h → "
          f"输出 {meta['horizon']}h")

    # Step 2: 读数据
    print(f"[2/4] 读取数据: {args.data}")
    hourly = load_hourly(args.data)
    print(f"      小时级数据 {len(hourly)} 行，"
          f"范围 {hourly.index[0]} ~ {hourly.index[-1]}")

    if args.mode == "backtest":
        # Step 3: 滚动回测
        print(f"[3/4] 滚动回测最近 {args.backtest_days} 天 ...")
        bt, metrics = rolling_backtest(model, meta, hourly, days=args.backtest_days)
        print(f"      MAPE={metrics['MAPE_%']:.2f}%  "
              f"MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
              f"(n={metrics['hours_used']}, zero-excluded={metrics['hours_zero_excluded']})")

        # Step 4: 画图
        print(f"[4/4] 绘制预测 vs 真实曲线 ...")
        csv_path = os.path.join(args.out, "backtest_detail.csv")
        png_path = os.path.join(args.out, "pred_vs_actual.png")
        bt.to_csv(csv_path, index=False, encoding="utf-8-sig")
        plot_backtest(bt, png_path, args.backtest_days, metrics)
        print(f"      明细 → {csv_path}")
        print(f"      对比图 → {png_path}")
    else:  # forecast
        print(f"[3/4] 滚动预测未来 {args.days} 天 ...")
        fc = rolling_forecast(model, meta, hourly, days=args.days)
        print(f"[4/4] 绘制预测曲线 ...")
        csv_path = os.path.join(args.out, "forecast_detail.csv")
        png_path = os.path.join(args.out, "forecast_curve.png")
        fc.to_csv(csv_path, index=False, encoding="utf-8-sig")
        plot_forecast(fc, png_path, args.days)
        print(f"      预测明细 → {csv_path}")
        print(f"      预测曲线 → {png_path}")

    print("[OK] 全部完成。")


if __name__ == "__main__":
    main()