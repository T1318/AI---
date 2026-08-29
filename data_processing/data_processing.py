# -*- coding: utf-8 -*-
"""冷站数据预处理汇总模块：CSV 读取 → 极端条件筛选 → 小波分解/去噪。
接口、参数与算法说明见同目录 README_data_processing.md。"""
import numpy as np
import pandas as pd
import pywt
# ============ 配置区============
# CSV 文件路径
DATA_PATH = 'data.csv'
# 时间列候选名，按优先级匹配（取第一个命中的列）
TIME_COLUMNS = ['Timestamp', 'timestamp', 'time', 'Time', 'datetime', 'DateTime', 'date']
# 极端工况判定阈值（来自赛题说明；低温仅北方项目）
EXTREME_THRESHOLDS = {
    'outdoor_temp_high': 35.0,   # 干球高温 ℃
    'wet_bulb_high':    28.0,    # 湿球高温 ℃
    'humidity_high':    85.0,    # 高湿 %
    'outdoor_temp_low': -10.0,   # 低温 ℃（仅北方项目）
    'load_change_rate': 0.20,    # 相邻小时负荷变化率
}
# sigmoid 过渡带尺度：必须远小于正常值与阈值的距离，否则正常段留底噪
SOFT_SCALES = {
    'outdoor_temp_high': 1.0,   # ℃
    'humidity_high':    5.0,    # %
    'outdoor_temp_low': 2.0,    # ℃
    'load_change_rate': 0.03,
}
# ============ 1. CSV 读取 ============
def load_csv(path):
    """读取 CSV，解析时间列并按时间排序。"""
    df = pd.read_csv(path)
    time_col = None
    for cand in TIME_COLUMNS:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(time_col).reset_index(drop=True)
    return df

def summarize(df):
    """打印行列数、时间范围、各列缺失数，返回 df 原样。"""
    print(f"行数: {df.shape[0]}   列数: {df.shape[1]}")
    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if time_cols:
        c = time_cols[0]
        print(f"时间范围: {df[c].min()}  ~  {df[c].max()}")
    print("各列缺失数:")
    print(df.isna().sum().to_string())
    return df
# ============ 2. 极端条件筛选 ============
def detect_extreme_conditions(df, config, thresholds=None):
    """阈值法检测极端工况，返回布尔掩码。config 含 'outdoor_temp'/'humidity'，
    'cooling_load' 可选（启用负荷骤变检测）。"""
    thr = thresholds or EXTREME_THRESHOLDS
    n = len(df)
    mask = np.zeros(n, dtype=bool)

    T = df[config['outdoor_temp']].values
    RH = df[config['humidity']].values
    mask |= (T >= thr['outdoor_temp_high'])
    mask |= (RH >= thr['humidity_high'])
    mask |= (T <= thr['outdoor_temp_low'])

    if 'cooling_load' in config:
        Q = df[config['cooling_load']].values
        Q_prev = np.roll(Q, 1)
        Q_prev[0] = Q[0]
        change_rate = np.abs(Q - Q_prev) / np.maximum(np.abs(Q_prev), 1e-6)
        mask |= (change_rate >= thr['load_change_rate'])
    return mask

def extreme_soft_mask(df, config, thresholds=None, scales=None):
    """极端程度 sigmoid 软掩码 [0,1]（→ denoise 的 protect_mask）。
    g→1 越极端保护越强，多条件取逐点最大；NaN 按非极端处理。"""
    thr = thresholds or EXTREME_THRESHOLDS
    sc = scales or SOFT_SCALES
    T = df[config['outdoor_temp']].values
    RH = df[config['humidity']].values

    g = _safe_sigmoid(T, thr['outdoor_temp_high'], sc['outdoor_temp_high'])
    g = np.maximum(g, _safe_sigmoid(RH, thr['humidity_high'], sc['humidity_high']))
    g = np.maximum(g, _safe_sigmoid(thr['outdoor_temp_low'] - T, 0.0,
                                    sc['outdoor_temp_low']))

    if 'cooling_load' in config:
        Q = df[config['cooling_load']].values
        Q_prev = np.roll(Q, 1)
        Q_prev[0] = Q[0]
        rate = np.abs(Q - Q_prev) / np.maximum(np.abs(Q_prev), 1e-6)
        g = np.maximum(g, _safe_sigmoid(rate, thr['load_change_rate'],
                                        sc['load_change_rate']))
    return g

def _sigmoid(z):
    """数值稳定 sigmoid。"""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def _safe_sigmoid(metric, threshold, scale):
    """(指标-阈值)/尺度 经 sigmoid；NaN → 0。"""
    z = (np.asarray(metric, dtype=float) - threshold) / scale
    z = np.where(np.isfinite(z), z, -np.inf)
    return _sigmoid(z)

def filter_extreme_data(df, extreme_mask):
    """极端条件数据子集（保留全部原始列，不剔除任何数据）。"""
    return df[extreme_mask]

def extract_continuous_segments(extreme_mask, min_length=48):
    """提取长度 >= min_length 的连续极端段 [(start, end), ...]。"""
    mask = np.asarray(extreme_mask, dtype=bool)
    if mask.sum() == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_length]

# ============ 3. 小波分解 / 去噪 ============
def decompose(x, wavelet='db4', level=3):
    """多级小波分解 + 单支重构。返回 (coeffs, components)：
    components 含 1 条 cA + level 条 cD，均与原信号等长且相加 == 原信号。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    coeffs = pywt.wavedec(x, wavelet, level=level, mode='periodization')

    names = [f'cA{level}'] + [f'cD{i}' for i in range(level, 0, -1)]
    components = {}
    for i, name in enumerate(names):
        single = [c if j == i else np.zeros_like(c)
                  for j, c in enumerate(coeffs)]
        components[name] = pywt.waverec(
            single, wavelet, mode='periodization')[:n]

    return coeffs, components

def denoise(x, protect_mask=None, wavelet='db4', level=3, protect_factor=0.0):
    """BayesShrink 逐层自适应阈值去噪（soft 收缩）。
    protect_mask 接受 bool 掩码或 [0,1] 软掩码；protect_factor=0 且 g=1
    的点严格输出原值。返回去噪后的一维数组（与 x 等长）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if protect_mask is not None:
        if len(protect_mask) != n:
            raise ValueError(
                f"protect_mask 长度 {len(protect_mask)} 与信号长度 {n} 不一致")
        protect_mask = np.asarray(protect_mask)

    coeffs = pywt.wavedec(x, wavelet, level=level, mode='periodization')

    noise_std = np.median(np.abs(coeffs[-1])) / 0.6745
    if noise_std < 1e-12:
        return pywt.waverec(coeffs, wavelet, mode='periodization')[:n]
    noise_var = noise_std ** 2
    arr, slices = pywt.coeffs_to_array(coeffs)
    thresh = np.zeros_like(arr)  # cA 层不阈值（保留趋势骨架）

    for i in range(1, len(slices)):
        detail_level = level + 1 - i
        scale = 2 ** detail_level
        sl = _layer_slice(slices, i)
        layer = arr[sl]

        signal_std = np.sqrt(max(np.mean(layer ** 2) - noise_var, 0.0))
        lam_j = (noise_var / signal_std) if signal_std > 1e-12 else np.inf

        if protect_mask is not None:
            # 软掩码嵌入：λ_eff = λ × (1 − g·(1 − protect_factor))，g∈[0,1]
            g_coeff = _mask_to_coeffs(protect_mask, len(layer), scale, wavelet)
            mix = 1.0 - g_coeff * (1.0 - protect_factor)
            # 纯噪声层(λ=∞)以层内最大幅值替代，使软插值保持有界
            lam_eff = lam_j if np.isfinite(lam_j) else np.max(np.abs(layer))
            thresh[sl] = lam_eff * mix
        else:
            thresh[sl] = lam_j

    arr = np.sign(arr) * np.maximum(np.abs(arr) - thresh, 0.0)
    coeffs_denoised = pywt.array_to_coeffs(arr, slices, 'wavedec')
    return pywt.waverec(coeffs_denoised, wavelet, mode='periodization')[:n]

def _layer_slice(coeff_slices, i):
    """第 i 层系数在 coeffs_to_array 扁平数组中的切片。"""
    entry = coeff_slices[i]
    return entry['d'][0] if isinstance(entry, dict) else entry[0]

def _mask_to_coeffs(mask, n_coeffs, scale, wavelet):
    """时间域掩码 → 系数域，取覆盖时段最大值，再按滤波器宽度向外膨胀。"""
    total = n_coeffs * scale
    m = np.clip(np.asarray(mask, dtype=float), 0.0, 1.0)
    padded = np.concatenate([
        m, np.zeros(max(0, total - len(m)))])[:total]
    prot = padded.reshape(n_coeffs, scale).max(axis=1)
    # dbN 滤波器长度 > 2：系数重构波形超出名义支撑，不膨胀会边界泄漏（勿删）
    support = pywt.Wavelet(wavelet).dec_len - 1
    if support > 0 and n_coeffs > 1:
        acc = np.zeros(n_coeffs)
        for s in range(-support, support + 1):
            acc = np.maximum(acc, np.roll(prot, s))
        prot = acc
    return prot

if __name__ == '__main__':
    #此处为初步加载文件显示
    df = load_csv(DATA_PATH)
    summarize(df)
