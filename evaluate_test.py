"""使用独立测试集评价三个负荷 Transformer。"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from load_forecasting.checkpoint import load_checkpoint
from load_forecasting.data import (
    TARGET_COLUMN,
    TIME_COLUMN,
    build_windows_with_future_weather,
    load_excel,
)
from load_forecasting.model_registry import get_model_spec
from load_forecasting.predict import restore_model
from train import forward_model, select_device


DEFAULT_TEST_DATA = (
    "附件3：训练数据集/训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx"
)
SUPPORTED_TRANSFORMERS = {
    "LoadTransformer",
    "LoadTransformerHistoryOnly",
    "LoadTransformerEncoderDecoder",
}


def build_test_windows(frame, checkpoint):
    """按预测范围作为步长，构造互不重叠的测试窗口和目标时间戳。"""
    input_hours = int(checkpoint["input_hours"])
    horizon = int(checkpoint["horizon"])
    sample_count = len(frame) - input_hours - horizon + 1
    if sample_count <= 0:
        raise ValueError(
            f"测试集至少需要 {input_hours + horizon} 小时，当前只有 {len(frame)} 小时"
        )

    starts = list(range(0, sample_count, horizon))
    histories, weather_windows, targets, timestamps = [], [], [], []
    skipped = 0
    last_error = None
    for start in starts:
        block = frame.iloc[start : start + input_hours + horizon]
        try:
            history, weather, target, feature_names, weather_names = (
                build_windows_with_future_weather(
                    block,
                    input_hours,
                    horizon,
                    checkpoint["wavelet_columns"],
                    stride_hours=horizon,
                )
            )
        except ValueError as exc:
            skipped += 1
            last_error = exc
            continue

        if feature_names != checkpoint["feature_names"]:
            raise ValueError("测试集历史特征顺序与检查点不一致")
        expected_weather = checkpoint.get("future_weather_names", weather_names)
        if expected_weather and weather_names != expected_weather:
            raise ValueError("测试集天气特征顺序与检查点不一致")

        histories.append(history[0])
        weather_windows.append(weather[0])
        targets.append(target[0])
        timestamps.append(
            block[TIME_COLUMN]
            .iloc[input_hours : input_hours + horizon]
            .to_numpy()
        )

    if not histories:
        detail = f"，最后一个错误：{last_error}" if last_error else ""
        raise ValueError(f"测试集中没有可用的连续预测窗口{detail}")

    return {
        "history": np.stack(histories).astype(np.float32),
        "future_weather": np.stack(weather_windows).astype(np.float32),
        "target": np.stack(targets).astype(np.float32),
        "timestamps": np.stack(timestamps),
        "candidate_windows": len(starts),
        "successful_windows": len(histories),
        "skipped_windows": skipped,
    }


def predict_test_windows(checkpoint, history, future_weather, device, batch_size=64):
    """按检查点标准化测试输入，批量预测并恢复负荷原始量纲。"""
    model_type = checkpoint["model_type"]
    if model_type not in SUPPORTED_TRANSFORMERS:
        raise ValueError(
            f"测试脚本仅支持三个Transformer，当前检查点模型为 {model_type}"
        )

    history = (
        (history - np.asarray(checkpoint["x_mean"]))
        / np.asarray(checkpoint["x_std"])
    ).astype(np.float32)
    uses_weather = get_model_spec(model_type)["uses_future_weather"]
    if uses_weather:
        future_weather = (
            (future_weather - np.asarray(checkpoint["future_weather_mean"]))
            / np.asarray(checkpoint["future_weather_std"])
        ).astype(np.float32)
    else:
        future_weather = future_weather.astype(np.float32)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(history),
            torch.from_numpy(future_weather),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model = restore_model(checkpoint).to(device)
    predictions = []
    with torch.no_grad():
        for batch_history, batch_weather in loader:
            predictions.append(
                forward_model(
                    model_type,
                    model,
                    batch_history.to(device),
                    batch_weather.to(device),
                ).cpu().numpy()
            )
    normalized = np.concatenate(predictions)
    return normalized * float(checkpoint["y_std"]) + float(checkpoint["y_mean"])


def calculate_metrics(predicted, actual):
    """MAE/RMSE使用全部有效值，MAPE只使用非零真实负荷。"""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    valid = np.isfinite(predicted) & np.isfinite(actual)
    if not valid.any():
        raise ValueError("没有可用于评价的有限预测值和真实值")

    error = predicted - actual
    nonzero = valid & (np.abs(actual) > 1e-6)
    used = int(nonzero.sum())
    excluded = int((valid & ~nonzero).sum())
    mape = (
        float(np.mean(np.abs(error[nonzero]) / np.abs(actual[nonzero])) * 100)
        if used
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(error[valid]))),
        "rmse": float(np.sqrt(np.mean(error[valid] ** 2))),
        "mape": mape,
        "mape_hours_used": used,
        "mape_hours_zero_excluded": excluded,
    }


def calculate_metrics_by_horizon(predicted, actual):
    """分别计算h+1至h+24各预测步长的指标。"""
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    rows = []
    for index in range(predicted.shape[1]):
        row = {"forecast_step": index + 1}
        row.update(calculate_metrics(predicted[:, index], actual[:, index]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_prediction_frame(timestamps, actual, predicted):
    """将二维预测窗口展开成逐小时时间序列。"""
    actual_flat = np.asarray(actual).reshape(-1)
    predicted_flat = np.asarray(predicted).reshape(-1)
    error = predicted_flat - actual_flat
    nonzero = np.abs(actual_flat) > 1e-6
    ape = np.full(actual_flat.shape, np.nan, dtype=float)
    ape[nonzero] = np.abs(error[nonzero]) / np.abs(actual_flat[nonzero]) * 100
    horizon = np.asarray(actual).shape[1]
    return pd.DataFrame(
        {
            "timeStamp": pd.to_datetime(np.asarray(timestamps).reshape(-1)),
            "forecast_step": np.tile(np.arange(1, horizon + 1), len(actual)),
            "actual_load": actual_flat,
            "predicted_load": predicted_flat,
            "error": error,
            "absolute_error": np.abs(error),
            "ape_percent": ape,
        }
    )


def _json_safe_metrics(metrics):
    return {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in metrics.items()
    }


def save_evaluation_outputs(detail, overall, by_horizon, output_dir):
    """保存逐时结果、指标和真实/预测负荷对比图。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        output_dir / "predictions.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    by_horizon.to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe_metrics(overall), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(16, 6))
    axis.plot(detail["timeStamp"], detail["actual_load"], label="Actual load")
    axis.plot(
        detail["timeStamp"], detail["predicted_load"], label="Predicted load"
    )
    axis.set_title(
        f"Test load forecast | MAE={overall['mae']:.3f}, "
        f"RMSE={overall['rmse']:.3f}, MAPE={overall['mape']:.3f}%"
    )
    axis.set_xlabel("Time")
    axis.set_ylabel("Load")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_dir / "load_curve.png", dpi=160)
    plt.close(figure)


def evaluate(checkpoint_path, data_path, output_dir, device="auto", batch_size=64):
    """执行完整的独立测试集预测、评价和结果保存。"""
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    frame = load_excel(data_path)
    windows = build_test_windows(frame, checkpoint)
    selected_device = select_device(device)
    predicted = predict_test_windows(
        checkpoint,
        windows["history"],
        windows["future_weather"],
        selected_device,
        batch_size,
    )
    actual = windows["target"]
    overall = calculate_metrics(predicted, actual)
    by_horizon = calculate_metrics_by_horizon(predicted, actual)
    detail = build_prediction_frame(windows["timestamps"], actual, predicted)
    save_evaluation_outputs(detail, overall, by_horizon, output_dir)
    return overall, windows


def parse_args():
    parser = argparse.ArgumentParser(description="独立测试集负荷预测与评价")
    parser.add_argument("--checkpoint", required=True, help="训练检查点 .pt 路径")
    parser.add_argument("--data", default=DEFAULT_TEST_DATA, help="测试集Excel路径")
    parser.add_argument(
        "--output-dir", default="outputs/test_evaluation", help="结果输出目录"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    overall, windows = evaluate(
        args.checkpoint,
        args.data,
        args.output_dir,
        args.device,
        args.batch_size,
    )
    print(f"预测设备: {select_device(args.device)}")
    print(
        f"候选窗口={windows['candidate_windows']}，"
        f"成功={windows['successful_windows']}，"
        f"跳过={windows['skipped_windows']}"
    )
    print(json.dumps(_json_safe_metrics(overall), ensure_ascii=False, indent=2))
    print(f"结果目录: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
