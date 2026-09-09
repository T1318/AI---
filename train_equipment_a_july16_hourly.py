"""任务二：24h历史、1h天气预测下一小时27项设备参数。"""
from argparse import ArgumentParser

from equipment_forecasting.hourly_data import EVALUATION_MODE, prepare_hourly_datasets
from train_equipment_a_july16 import default_config_equipment, train_equipment_model


def default_config_equipment_hourly():
    config = default_config_equipment()
    config.input_hours = 24
    config.horizon = 1
    config.epochs = 100
    config.wavelet = "db4"
    config.wavelet_level = 1
    config.evaluation_mode = EVALUATION_MODE
    config.checkpoint_name = "best.pt"
    config.output_dir = "outputs/TestA_July16_equipment_hourly"
    return config


def run(config):
    return train_equipment_model(config, prepare_hourly_datasets(config, task="equipment"))


def parse_config():
    config = default_config_equipment_hourly()
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
