from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.data_processing import decompose


TIME_COLUMN = "timeStamp"
TARGET_COLUMN = "TotalRealTimeLoad"
INVALID_VALUES = (-8888.7998, -9999.0, -99999.0)
FIVE_MINUTE_GAP_LIMIT = 6
HOURLY_GAP_LIMIT = 3

WAVELET_COLUMNS = [
    "TotalRealTimeLoad",
    "OutdoorTdbin",
    "OutdoorWetTemp",
    "PriChWFlow01",
    "PriChWDiffPress",
    "PriChWTempSupply01",
    "PriChWTempReturn01",
    "CWFlow01",
    "CWTempSupply01",
    "CWTempReturn01",
    "ChPower01",
    "ChPower02",
    "ChPower03",
    "PriChWPVSDFreq01",
    "PriChWPVSDFreq02",
    "PriChWPVSDFreq03",
    "CWPVSDFreq01",
    "CWPVSDFreq02",
    "CWPVSDFreq03",
    "CTVSDFreq01",
    "CTVSDFreq02",
    "CTVSDFreq03",
]


def load_excel(path: str | Path, sheet_name: str = "历史数据") -> pd.DataFrame:
    """读取赛题数据，清洗5分钟记录并聚合为小时数据。"""
    raw = pd.read_excel(path, sheet_name=sheet_name)
    return resample_hourly(clean_dataframe(raw))


def _interpolate_short_gaps(values: pd.DataFrame, limit: int) -> pd.DataFrame:
    """只填补长度不超过 limit 的连续缺失段。"""
    result = values.copy()
    for column in result.columns:
        series = result[column]
        missing = series.isna()
        if not missing.any():
            continue
        groups = missing.ne(missing.shift()).cumsum()
        gap_sizes = missing.groupby(groups).transform("sum")
        interpolated = series.interpolate(limit_direction="both")
        fillable = missing & (gap_sizes <= limit)
        result.loc[fillable, column] = interpolated.loc[fillable]
    return result


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """解析时间和数值列，将异常占位值转为空值并填补短缺口。"""
    result = df.copy()
    result[TIME_COLUMN] = pd.to_datetime(result[TIME_COLUMN], errors="coerce")
    result = result.dropna(subset=[TIME_COLUMN]).sort_values(TIME_COLUMN).reset_index(drop=True)

    for column in result.columns:
        if column == TIME_COLUMN:
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] = result[column].replace(list(INVALID_VALUES), np.nan)

    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = _interpolate_short_gaps(
        result[numeric_columns], FIVE_MINUTE_GAP_LIMIT
    )
    return result


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """将5分钟数据按小时求均值，并只填补不超过3小时的缺口。"""
    numeric = df.set_index(TIME_COLUMN).select_dtypes(include=[np.number])
    hourly = numeric.resample("1h").mean()
    hourly = _interpolate_short_gaps(hourly, HOURLY_GAP_LIMIT)
    return hourly.reset_index()


def _time_features(timestamps: pd.Series) -> pd.DataFrame:
    """把小时、星期和月份转换为周期正余弦编码。"""
    hour = timestamps.dt.hour.to_numpy()
    weekday = timestamps.dt.dayofweek.to_numpy()
    month = timestamps.dt.month.to_numpy()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        }
    )


def wavelet_feature_window(
    window: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """在单个历史窗口内生成原始值、四路小波分量和时间特征。"""
    missing_columns = [column for column in columns if column not in window.columns]
    if missing_columns:
        raise ValueError(f"数据缺少小波输入列: {missing_columns}")
    if window[columns].isna().any().any():
        raise ValueError("小波输入窗口中存在缺失值")

    arrays = []
    feature_names = []
    for column in columns:
        signal = window[column].to_numpy(dtype=float)
        _, components = decompose(signal)
        arrays.append(signal)
        feature_names.append(column)
        for component_name in ("cA3", "cD3", "cD2", "cD1"):
            arrays.append(components[component_name])
            feature_names.append(f"{column}_{component_name}")

    time_features = _time_features(window[TIME_COLUMN].reset_index(drop=True))
    arrays.extend(time_features[column].to_numpy() for column in time_features.columns)
    feature_names.extend(time_features.columns.tolist())
    values = np.column_stack(arrays).astype(np.float32)
    return pd.DataFrame(values, columns=feature_names), feature_names


def feature_frame(
    df: pd.DataFrame, wavelet_columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """为一个连续历史窗口构造模型输入特征。"""
    return wavelet_feature_window(df, wavelet_columns or WAVELET_COLUMNS)


def _is_continuous_hourly(timestamps: pd.Series) -> bool:
    if len(timestamps) <= 1:
        return True
    return bool(timestamps.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all())


def _valid_sample(block: pd.DataFrame, columns: list[str]) -> bool:
    required = list(dict.fromkeys(columns + [TARGET_COLUMN]))
    return (
        _is_continuous_hourly(block[TIME_COLUMN])
        and not block[required].isna().any().any()
    )


def build_windows(
    df: pd.DataFrame,
    input_hours: int,
    horizon: int,
    wavelet_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """用过去 input_hours 小时预测未来 horizon 小时。"""
    columns = list(wavelet_columns or WAVELET_COLUMNS)
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少小波输入列: {missing_columns}")
    if len(df) < input_hours + horizon:
        raise ValueError("数据量不足以构造一个完整窗口")

    xs = []
    ys = []
    feature_names = None
    sample_count = len(df) - input_hours - horizon + 1
    for start in range(sample_count):
        block = df.iloc[start : start + input_hours + horizon]
        if not _valid_sample(block, columns):
            continue
        input_window = block.iloc[:input_hours]
        features, names = wavelet_feature_window(input_window, columns)
        xs.append(features.to_numpy(dtype=np.float32))
        ys.append(
            block[TARGET_COLUMN]
            .iloc[input_hours : input_hours + horizon]
            .to_numpy(dtype=np.float32)
        )
        feature_names = names

    if not xs:
        raise ValueError("清洗后没有可用的连续小时窗口")
    return np.stack(xs), np.stack(ys), feature_names


def build_latest_window(
    df: pd.DataFrame,
    input_hours: int,
    wavelet_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """构造数据末端最近一个完整历史窗口，用于预测未知未来。"""
    columns = list(wavelet_columns or WAVELET_COLUMNS)
    if len(df) < input_hours:
        raise ValueError("数据量不足以构造预测窗口")

    for end in range(len(df), input_hours - 1, -1):
        window = df.iloc[end - input_hours : end]
        if _valid_sample(window, columns):
            features, names = wavelet_feature_window(window, columns)
            return features.to_numpy(dtype=np.float32)[None, ...], names
    raise ValueError("找不到连续且无缺失的最新小时窗口")
