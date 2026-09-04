"""项目C深度模型训练入口，复用 train.py 的 README 流程。

项目C与项目A的列差异：冷机和冷却塔各2台（无 *03），一次侧流量/压差列改名
（PriChWFlow / PriChWDP01），因此小波列为 20 个，特征维度 = 20 × 5 + 6 = 106。

用法:
    python train_project_c.py                              # 默认 LoadTransformer
    python train_project_c.py --model lstm --epochs 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import train as train_module
from load_forecasting import data as data_module
from load_forecasting.model_registry import SUPPORTED_DEEP_MODEL_TYPES

PROJECT_C_DATA = "训练数据项目C历史数据_2026-01-01_2026-07-31(1).xlsx"

PROJECT_C_WAVELET_COLUMNS = [
    "TotalRealTimeLoad",
    "OutdoorTdbin",
    "OutdoorWetTemp",
    "PriChWFlow",
    "PriChWDP01",
    "PriChWTempSupply01",
    "PriChWTempReturn01",
    "CWFlow01",
    "CWTempSupply01",
    "CWTempReturn01",
    "ChPower01",
    "ChPower02",
    "PriChWPVSDFreq01",
    "PriChWPVSDFreq02",
    "PriChWPVSDFreq03",
    "CWPVSDFreq01",
    "CWPVSDFreq02",
    "CWPVSDFreq03",
    "CTVSDFreq01",
    "CTVSDFreq02",
]

# prepare_datasets 与 build_checkpoint 调用时读取模块级 WAVELET_COLUMNS，
# 替换后项目C检查点会记录这20列，预测入口用项目C数据可直接恢复特征。
data_module.WAVELET_COLUMNS = PROJECT_C_WAVELET_COLUMNS
train_module.WAVELET_COLUMNS = PROJECT_C_WAVELET_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="项目C冷负荷预测训练（168小时输入预测24小时）")
    parser.add_argument("--model", default="LoadTransformer", choices=SUPPORTED_DEEP_MODEL_TYPES)
    parser.add_argument("--data", default=PROJECT_C_DATA, help="项目C训练数据Excel路径")
    parser.add_argument("--output", default=None, help="默认 outputs/<模型名>_C.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = train_module.default_config(args.model)
    config.data = args.data
    config.output = args.output or f"outputs/{args.model}_C.pt"
    config.epochs = args.epochs
    config.device = args.device
    config.training_config["batch_size"] = args.batch_size
    config.training_config["lr"] = args.lr
    if not Path(config.data).exists():
        available = sorted(p.name for p in Path.cwd().glob("*.xlsx"))
        raise FileNotFoundError(
            f"找不到项目C数据文件: {config.data}\n当前目录下的Excel文件: {available}"
        )
    print(f"项目C训练: 模型={config.model} 数据={config.data} 输出={config.output} "
          f"epochs={config.epochs} device={config.device}")
    datasets = train_module.prepare_datasets(config)
    print(f"窗口样本: 训练={datasets['x_train'].shape[0]} 验证={datasets['x_valid'].shape[0]} "
          f"测试={datasets['x_test'].shape[0]} 特征维度={datasets['input_dim']}")
    train_module.train_model(config, datasets, save_checkpoint=True)


if __name__ == "__main__":
    main()
