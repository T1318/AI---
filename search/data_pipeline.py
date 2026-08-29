# -*- coding: utf-8 -*-
"""
data_pipeline.py —— 临时数据管线（寻优专用最短可用版）

职责边界（2026-08-28 与用户确认）：
    本文件只做到"小时级干净数据 + 按模型输出形态开窗"为止。
    特征选择、滞后特征、归一化策略等高级特征工程【不做】——那是特征工程队友的活，
    队友的管线到位后整体替换本文件即可，寻优引擎不用动。

处理内容（每个项目的 xlsx → 模型可用的窗口张量）：
    1. 丢掉中文参数名行，剔除累计量列（电度值/当日累计）——累计量做小时均值无意义
    2. 数值化 + 剔除传感器坏点（|值|>1e6，项目B实测有 1.5e14 的坏点）+ 负负荷置NaN
    3. 5分钟级短缺口线性插值（最多30分钟），重采样为小时均值
    4. 补最小日历特征（小时sin/cos、星期几）
    5. 按时间顺序 80/20 切分（标准化只用训练段统计量，防泄漏）
    6. 按模型输出形态开窗：
         form="seq"      → x:(N,L,F)  y:(N,L,1)   逐时刻预测下一步（10个序列模型）
         form="horizon"  → x:(N,L,F)  y:(N,24)    直接预测未来24小时（Transformer）
         form="seq2seq"  → x:(N,L,F)  y_prev:(N,24,1)  y:(N,24,1)  （Seq2SeqLSTM）
         form="flat"     → x:(N,L*F)  y:(N,)      展平窗口预测下一小时（XGBoost）

已知数据质量问题（写死在代码里的处理，均在运行时打印）：
    - 项目A实际只到 2025-10-20（文件名说到10-31）
    - 项目B负荷列有 1.5e14 坏点
    - 项目C 5分钟级 77% 零负荷（停机时段）；小时级零负荷比例单独打印
    - 三个文件"系统参数"表全为空
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 配置
# =============================================================================
DATA_DIR_DEFAULT = r"C:/Users/23263/Desktop/赛马/数据"
PROJECT_FILES = {
    "A": "训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx",
    "B": "训练数据项目B历史数据_2023-04-01_2023-10-31.xlsx",
    "C": "训练数据项目C历史数据_2026-01-01_2026-07-31.xlsx",
}
TARGET_COL = "TotalRealTimeLoad"          # 瞬时冷量：负荷预测的目标列
_CUMULATIVE_KEYWORDS = ("电度值", "累计")   # 中文参数名含这些字样 → 累计量列，剔除
_ABS_OUTLIER = 1e6                        # |值|超过它 → 传感器坏点
_INTERP_5MIN_LIMIT = 6                    # 5分钟级最多插值6步（30分钟）
_HORIZON = 24                             # 比赛口径：预测未来24小时（固定，不寻优）


# =============================================================================
# 第一步：项目 xlsx → 小时级干净 DataFrame
# =============================================================================
def load_project_hourly(project, data_dir=DATA_DIR_DEFAULT, use_cache=True, verbose=True):
    """读取某项目的 xlsx，返回小时级 DataFrame（索引=时间，列=特征+目标）。

    首次读取较慢（xlsx 几万行），结果缓存到 cache/{项目}_hourly.csv，
    之后直接读缓存（改了清洗规则请删缓存重跑）。
    """
    assert project in PROJECT_FILES, "项目必须是 A/B/C，收到 %r" % project

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    cache_path = os.path.join(cache_dir, "%s_hourly.csv" % project)
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["timeStamp"], index_col="timeStamp")
        if verbose:
            print("[数据] %s 项目使用缓存: %d 小时 × %d 列" % (project, len(df), df.shape[1]))
        return df

    path = os.path.join(data_dir, PROJECT_FILES[project])
    if not os.path.exists(path):
        raise FileNotFoundError("找不到数据文件: %s（可用 --data_dir 改路径）" % path)

    raw = pd.read_excel(path, sheet_name="历史数据")
    chinese = raw.iloc[0]          # 第0行数据是中文参数名行
    df = raw.iloc[1:].copy()

    # --- 剔除累计量列与时间列，其余数值化 ---
    keep = []
    dropped = []
    for col in df.columns:
        if col == "timeStamp":
            continue
        cn = str(chinese.get(col, ""))
        if any(k in cn for k in _CUMULATIVE_KEYWORDS):
            dropped.append("%s(%s)" % (col, cn))
            continue
        keep.append(col)
    if verbose:
        print("[数据] 剔除累计量列 %d 个: %s" % (len(dropped), "、".join(dropped[:6]) + ("..." if len(dropped) > 6 else "")))

    ts = pd.to_datetime(df["timeStamp"])
    data = df[keep].apply(pd.to_numeric, errors="coerce")
    data.index = ts
    data = data[~data.index.duplicated(keep="first")].sort_index()

    # --- 坏点与非法值 ---
    n_outlier = int((data.abs() > _ABS_OUTLIER).sum().sum())
    data = data.mask(data.abs() > _ABS_OUTLIER)
    if TARGET_COL in data.columns:
        data.loc[data[TARGET_COL] < 0, TARGET_COL] = np.nan
    if verbose:
        print("[数据] 剔除坏点读数 %d 个（|值|>%.0e）" % (n_outlier, _ABS_OUTLIER))

    # --- 5分钟级插值 + 小时重采样 ---
    data = data.interpolate(method="time", limit=_INTERP_5MIN_LIMIT)
    hourly = data.resample("1h").mean()
    n_before = len(hourly)
    hourly = hourly[hourly[TARGET_COL].notna()]
    hourly = hourly.interpolate(limit=3).dropna(axis=0)

    # --- 零方差列剔除（对模型无信息，还可能让标准化除零）---
    nunique = hourly.nunique()
    zero_var = [c for c in hourly.columns if nunique[c] <= 1]
    if zero_var:
        hourly = hourly.drop(columns=zero_var)
        if verbose:
            print("[数据] 剔除零方差列 %d 个: %s" % (len(zero_var), ", ".join(zero_var[:8])))

    # --- 最小日历特征 ---
    hourly["feat_hour_sin"] = np.sin(2 * np.pi * hourly.index.hour / 24.0)
    hourly["feat_hour_cos"] = np.cos(2 * np.pi * hourly.index.hour / 24.0)
    hourly["feat_dow"] = hourly.index.dayofweek.astype("float32")

    hourly = hourly.rename_axis("timeStamp")
    if verbose:
        zero_ratio = float((hourly[TARGET_COL] == 0).mean())
        print("[数据] %s: 5分钟 %d 行 → 小时 %d 行（目标缺失丢弃 %d 小时），"
              "小时级零负荷占比 %.1f%%，特征列 %d 个"
              % (project, len(df), len(hourly), n_before - len(hourly), zero_ratio * 100,
                 hourly.shape[1] - 1))

    os.makedirs(cache_dir, exist_ok=True)
    hourly.to_csv(cache_path)
    if verbose:
        print("[数据] 已缓存至 %s（改清洗规则后请删掉重生成）" % cache_path)
    return hourly


# =============================================================================
# 第二步：小时 DataFrame → 各形态的窗口张量
# =============================================================================
def _make_windows(feats, targ, seq_len, form, horizon=_HORIZON):
    """在一段连续序列内部开滑动窗口（不跨段，防泄漏）。返回 numpy 数组。"""
    T, F = feats.shape
    # 各形态对窗口起点 i 的约束：
    #   seq/flat     : i+seq_len+1 <= T
    #   horizon      : i+seq_len+horizon <= T
    #   seq2seq      : 同 horizon，且 y_prev 要取到 i+seq_len-horizon >= 0（seq_len<horizon 时起点右移）
    if form in ("seq", "flat"):
        i_min, i_max = 0, T - seq_len - 1
    elif form in ("horizon", "seq2seq"):
        i_min, i_max = max(0, horizon - seq_len), T - seq_len - horizon
    else:
        raise ValueError("未知 form %r" % form)
    if i_max < i_min:
        raise ValueError("数据太短，开不出窗口：T=%d, seq_len=%d, form=%s" % (T, seq_len, form))

    xs, ys, y_prevs = [], [], []
    for i in range(i_min, i_max + 1):
        x = feats[i:i + seq_len]
        if form == "seq":
            # 逐时刻预测下一步：窗口每个位置预测 t+1 的负荷
            ys.append(targ[i + 1:i + seq_len + 1])
        elif form == "horizon":
            # 直接预测未来 horizon 小时
            ys.append(targ[i + seq_len:i + seq_len + horizon])
        elif form == "seq2seq":
            # 解码器吃"窗口末尾往前 horizon 步的真实负荷"（teacher forcing），
            # 预测窗口之后的 horizon 步 —— 这是对 y_prev 语义的工程选择，见 README 争议点
            y_prevs.append(targ[i + seq_len - horizon:i + seq_len])
            ys.append(targ[i + seq_len:i + seq_len + horizon])
        elif form == "flat":
            xs.append(x.reshape(-1))
            ys.append(targ[i + seq_len])
            continue
        else:
            raise ValueError("未知 form %r" % form)
        xs.append(x)
    if not xs:
        raise ValueError("数据太短，开不出窗口：T=%d, seq_len=%d" % (T, seq_len))
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    if form == "seq":
        ys = ys.reshape(ys.shape[0], ys.shape[1], 1)
    elif form == "seq2seq":
        ys = ys.reshape(ys.shape[0], ys.shape[1], 1)
        y_prevs = np.asarray(y_prevs, dtype=np.float32).reshape(len(y_prevs), horizon, 1)
        return xs, ys, y_prevs
    return xs, ys, None


def build_tensors(hourly, seq_len, form, train_ratio=0.8, horizon=_HORIZON):
    """标准化（只用训练段统计量）+ 分段开窗。返回 dict（numpy，喂给 DataLoader）。"""
    feats = hourly.drop(columns=[TARGET_COL]).values.astype(np.float64)
    targ = hourly[TARGET_COL].values.astype(np.float64)

    split = int(len(feats) * train_ratio)
    f_mu, f_sd = feats[:split].mean(0), feats[:split].std(0)
    t_mu, t_sd = targ[:split].mean(), targ[:split].std()
    f_sd = np.where(f_sd < 1e-8, 1.0, f_sd)   # 防除零
    if t_sd < 1e-8:
        raise ValueError("训练段目标列几乎恒定，无法寻优，请检查数据")

    feats_n = (feats - f_mu) / f_sd
    targ_n = (targ - t_mu) / t_sd

    x_tr, y_tr, yp_tr = _make_windows(feats_n[:split], targ_n[:split], seq_len, form, horizon)
    x_va, y_va, yp_va = _make_windows(feats_n[split:], targ_n[split:], seq_len, form, horizon)

    out = {
        "x_train": x_tr, "y_train": y_tr, "x_val": x_va, "y_val": y_va,
        "input_dim": feats.shape[1], "target_mean": t_mu, "target_std": t_sd,
    }
    if form == "seq2seq":
        out["y_prev_train"], out["y_prev_val"] = yp_tr, yp_va
    return out


def make_loader(arrays, batch_size, shuffle):
    """把若干同长度 numpy 数组打包成 DataLoader。"""
    tensors = [torch.as_tensor(a) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def val_mape_raw(tensors, y_pred_n):
    """把归一化预测值还原成原始单位算 MAPE（剔除真实值为0的小时——MAPE分母为零无定义）。

    y_pred_n: numpy，形状与 tensors["y_val"] 相同（归一化值）。
    返回 (mape, 参与计算的小时数, 被剔除的零负荷小时数)。
    """
    y_true_n = tensors["y_val"]
    y_true = y_true_n * tensors["target_std"] + tensors["target_mean"]
    y_pred = np.asarray(y_pred_n) * tensors["target_std"] + tensors["target_mean"]
    mask = y_true.flatten() != 0
    if mask.sum() == 0:
        return float("nan"), 0, int((~mask).sum())
    t = y_true.flatten()[mask]
    p = y_pred.flatten()[mask]
    m = np.mean(np.abs((t - p) / np.abs(t))) * 100.0
    return float(m), int(mask.sum()), int((~mask).sum())


# =============================================================================
# demo 合成数据（冒烟测试用，不进正式流程）
# =============================================================================
def make_demo_hourly(n_hours=1440, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-06-01", periods=n_hours, freq="h")
    hour = idx.hour.values
    base = 500.0 + 300.0 * np.sin(2 * np.pi * (hour - 6) / 24.0)      # 日周期
    week = 100.0 * np.sin(2 * np.pi * idx.dayofweek.values / 7.0)     # 周周期
    load = np.clip(base + week + rng.normal(0, 40, n_hours), 0, None)
    df = pd.DataFrame({
        "OutdoorTdbin": 28 + 6 * np.sin(2 * np.pi * (hour - 14) / 24.0) + rng.normal(0, 1, n_hours),
        "OutdoorWetTemp": 24 + 4 * np.sin(2 * np.pi * (hour - 13) / 24.0) + rng.normal(0, 1, n_hours),
        "PriChWTempSupply01": 7 + rng.normal(0, 0.5, n_hours),
        "CWTempReturn": 32 + rng.normal(0, 1.5, n_hours),
        "feat_hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "feat_hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "feat_dow": idx.dayofweek.astype("float32"),
        TARGET_COL: load,
    }, index=idx).rename_axis("timeStamp")
    return df
