"""任务一：24h历史、1h天气预测1h负荷，验证日逐小时加入真实观测。"""
from argparse import ArgumentParser
import json
from pathlib import Path

import pandas as pd
import torch

from equipment_forecasting.hourly_data import (
    EVALUATION_MODE, load_training_arrays, prepare_hourly_datasets,
)
from train import metrics, train_model
from train_a_july16 import default_config_july16


def default_config_hourly():
    config = default_config_july16("LoadTransformerEncoderDecoder")
    config.input_hours = 24
    config.horizon = 1
    config.model_config["horizon"] = 1
    config.epochs = 100
    config.seed = 42
    config.wavelet = "db4"
    config.wavelet_level = 1
    config.evaluation_mode = EVALUATION_MODE
    config.output_dir = "outputs/TestA_July16_load_hourly"
    config.output = str(Path(config.output_dir) / "best.pt")
    config.validation_output = str(Path(config.output_dir) / "july16_validation.csv")
    return config


def train_hourly_model(config, prepared):
    """复用MSE训练及最优权重导出，另保存逐时更新评价口径和指标。"""
    if config.epochs < 1:
        raise ValueError("epochs必须大于0")
    torch.manual_seed(config.seed)
    output = Path(config.output_dir)
    config.output = str(output / "best.pt")
    config.validation_output = str(output / "july16_validation.csv")
    model, result = train_model(config, load_training_arrays(prepared), save_checkpoint=True)
    # CSV来自train_model恢复后的最佳权重；从该文件计算，保证报告和文件一致。
    detail = pd.read_csv(config.validation_output)
    scores = metrics(detail["predicted_load"].to_numpy(), detail["actual_load"].to_numpy())
    pd.DataFrame([dict(target="TotalRealTimeLoad", **scores)]).to_csv(
        output / "july16_metrics.csv", index=False,
    )
    summary = dict(scores, best_epoch=result["best_epoch"], selection_metric="nonzero_mape",
                   evaluation_mode=EVALUATION_MODE, weather_source="historical_observations",
                   history_update="all_30_observed_columns", input_hours=24, horizon=1,
                   validation_hours=len(detail), wavelet=config.wavelet,
                   wavelet_level=config.wavelet_level)
    (output / "july16_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return model, summary


def run(config):
    return train_hourly_model(config, prepare_hourly_datasets(config, task="load"))


def parse_config():
    config = default_config_hourly()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=config.data)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=config.device)
    parser.add_argument("--batch-size", type=int, default=config.training_config["batch_size"])
    parser.add_argument("--output-dir", default=config.output_dir)
    args = parser.parse_args()
    for key in ("data", "epochs", "device", "output_dir"):
        setattr(config, key, getattr(args, key))
    config.training_config["batch_size"] = args.batch_size
    return config


if __name__ == "__main__":
    run(parse_config())
