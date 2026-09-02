from argparse import Namespace

import numpy as np
import pandas as pd
import torch

from .checkpoint import load_checkpoint
from .data import (
    build_latest_history_window,
    load_excel,
    load_weather_forecast_csv,
)
from .model_registry import build_model, get_model_spec


def restore_model(checkpoint):
    model = build_model(
        checkpoint["model_type"],
        checkpoint["input_dim"],
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    config = Namespace(
        data="附件3：训练数据集/训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx",
        weather="future_weather_24h.csv",
        checkpoint="outputs/LoadTransformer_weather.pt",
        output="prediction_24h.csv",
    )

    checkpoint = load_checkpoint(config.checkpoint, map_location="cpu")
    frame = load_excel(config.data)
    x, feature_names, last_time = build_latest_history_window(
        frame,
        checkpoint["input_hours"],
        checkpoint["wavelet_columns"],
    )
    if feature_names != checkpoint["feature_names"]:
        raise ValueError("预测特征顺序与训练检查点不一致")
    x = (x - checkpoint["x_mean"]) / checkpoint["x_std"]

    model = restore_model(checkpoint)
    uses_weather = get_model_spec(checkpoint["model_type"])["uses_future_weather"]
    with torch.no_grad():
        if uses_weather:
            weather, weather_names, timestamps = load_weather_forecast_csv(
                config.weather,
                last_time + pd.Timedelta(hours=1),
                checkpoint["horizon"],
            )
            if weather_names != checkpoint["future_weather_names"]:
                raise ValueError("天气特征顺序与训练检查点不一致")
            weather = (
                weather - checkpoint["future_weather_mean"]
            ) / checkpoint["future_weather_std"]
            prediction = model(
                torch.from_numpy(x), torch.from_numpy(weather.astype(np.float32))
            ).numpy()[0]
        else:
            timestamps = pd.date_range(
                last_time + pd.Timedelta(hours=1),
                periods=checkpoint["horizon"],
                freq="h",
            )
            prediction = model(torch.from_numpy(x)).numpy()[0]
    prediction = prediction * checkpoint["y_std"] + checkpoint["y_mean"]
    pd.DataFrame(
        {"timeStamp": timestamps, "TotalRealTimeLoad": prediction}
    ).to_csv(
        config.output,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    print(prediction)


if __name__ == "__main__":
    main()
