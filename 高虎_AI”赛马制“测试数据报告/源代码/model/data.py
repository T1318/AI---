from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.data_processing import decompose, denoise


TIME_COLUMN = "timeStamp"
TARGET_COLUMN = "TotalRealTimeLoad"
INVALID_VALUES = (-8888.7998, -9999.0, -99999.0)
FIVE_MINUTE_GAP_LIMIT = 6
HOURLY_GAP_LIMIT = 3
FUTURE_WEATHER_COLUMNS = ["OutdoorTdbin", "OutdoorWetTemp"]

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
    result[TIME_COLUMN] = pd.to_datetime(
        result[TIME_COLUMN],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
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
    window: pd.DataFrame, columns: list[str], level: int = 3, wavelet: str = "db4"
) -> tuple[pd.DataFrame, list[str]]:
    """在单个历史窗口内生成降噪值、四路小波分量和时间特征。"""
    missing_columns = [column for column in columns if column not in window.columns]
    if missing_columns:
        raise ValueError(f"数据缺少小波输入列: {missing_columns}")
    if window[columns].isna().any().any():
        raise ValueError("小波输入窗口中存在缺失值")

    arrays = []
    feature_names = []
    for column in columns:
        signal = window[column].to_numpy(dtype=float)
        denoised_signal = denoise(signal, wavelet=wavelet, level=level)
        _, components = decompose(denoised_signal, wavelet=wavelet, level=level)
        arrays.append(denoised_signal)
        feature_names.append(column)
        for component_name in [f"cA{level}"] + [f"cD{i}" for i in range(level, 0, -1)]:
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


def future_weather_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = [column for column in FUTURE_WEATHER_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"数据缺少未来天气列: {missing}")
    weather = df[FUTURE_WEATHER_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if weather.isna().any().any():
        raise ValueError("未来天气中存在缺失值")
    time_features = _time_features(df[TIME_COLUMN].reset_index(drop=True))
    result = pd.concat(
        [weather.reset_index(drop=True), time_features], axis=1
    ).astype(np.float32)
    return result, list(result.columns)


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
    stride_hours: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """用过去 input_hours 小时预测未来 horizon 小时。"""
    if stride_hours <= 0:
        raise ValueError("stride_hours 必须大于0")
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
    for start in range(0, sample_count, stride_hours):
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


def build_windows_with_future_weather(
    df: pd.DataFrame,
    input_hours: int,
    horizon: int,
    wavelet_columns: list[str] | None = None,
    stride_hours: int = 1,
):
    """构造历史特征、同一目标区间的未来天气和负荷目标。"""
    if stride_hours <= 0:
        raise ValueError("stride_hours 必须大于0")
    columns = list(wavelet_columns or WAVELET_COLUMNS)
    required = list(dict.fromkeys(columns + FUTURE_WEATHER_COLUMNS + [TARGET_COLUMN]))
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"数据缺少窗口列: {missing}")
    if len(df) < input_hours + horizon:
        raise ValueError("数据量不足以构造一个完整窗口")

    histories, future_weather, targets = [], [], []
    history_names = weather_names = None
    sample_count = len(df) - input_hours - horizon + 1
    for start in range(0, sample_count, stride_hours):
        block = df.iloc[start : start + input_hours + horizon]
        if not _is_continuous_hourly(block[TIME_COLUMN]):
            continue
        if block[required].isna().any().any():
            continue
        history_frame, history_names = wavelet_feature_window(
            block.iloc[:input_hours], columns
        )
        weather_frame, weather_names = future_weather_frame(
            block.iloc[input_hours : input_hours + horizon]
        )
        histories.append(history_frame.to_numpy(dtype=np.float32))
        future_weather.append(weather_frame.to_numpy(dtype=np.float32))
        targets.append(
            block[TARGET_COLUMN]
            .iloc[input_hours : input_hours + horizon]
            .to_numpy(dtype=np.float32)
        )
    if not histories:
        raise ValueError("清洗后没有可用的连续小时天气窗口")
    return (
        np.stack(histories),
        np.stack(future_weather),
        np.stack(targets),
        history_names,
        weather_names,
    )


def build_latest_window(
    df: pd.DataFrame,
    input_hours: int,
    wavelet_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """构造数据末端最近一个完整历史窗口，用于预测未知未来。"""
    values, names, _ = build_latest_history_window(
        df, input_hours, wavelet_columns
    )
    return values, names


def build_latest_history_window(
    df: pd.DataFrame,
    input_hours: int,
    wavelet_columns: list[str] | None = None,
):
    """返回最近完整历史窗口、特征名和窗口末端时间。"""
    columns = list(wavelet_columns or WAVELET_COLUMNS)
    if len(df) < input_hours:
        raise ValueError("数据量不足以构造预测窗口")
    for end in range(len(df), input_hours - 1, -1):
        window = df.iloc[end - input_hours : end]
        if _valid_sample(window, columns):
            features, names = wavelet_feature_window(window, columns)
            return (
                features.to_numpy(dtype=np.float32)[None, ...],
                names,
                pd.Timestamp(window[TIME_COLUMN].iloc[-1]),
            )
    raise ValueError("找不到连续且无缺失的最新小时窗口")


def load_weather_forecast_csv(path, expected_start, horizon=24):
    """读取并校验未来逐小时干球、湿球温度预报。"""
    raw = pd.read_csv(path)
    required = [TIME_COLUMN] + FUTURE_WEATHER_COLUMNS
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"天气预报缺少列: {missing}")
    if len(raw) != horizon:
        raise ValueError(f"天气预报必须包含 {horizon} 条逐小时记录")
    timestamps = pd.to_datetime(
        raw[TIME_COLUMN],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    if timestamps.isna().any():
        raise ValueError("天气预报时间格式无效")
    expected = pd.date_range(pd.Timestamp(expected_start), periods=horizon, freq="h")
    if not timestamps.reset_index(drop=True).equals(pd.Series(expected)):
        raise ValueError("天气预报必须从历史末端下一小时开始并逐小时连续")
    frame = raw.copy()
    frame[TIME_COLUMN] = timestamps
    weather, names = future_weather_frame(frame)
    return weather.to_numpy(dtype=np.float32)[None, ...], names, expected
