import argparse
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from load_forecasting.data import (
    TIME_COLUMN,
    WAVELET_COLUMNS,
    build_windows_with_future_weather,
    load_excel,
)
from load_forecasting.checkpoint import load_checkpoint
from load_forecasting.model_registry import (
    DEFAULT_TRAINING_CONFIG,
    build_model,
    default_model_config,
    get_model_spec,
)


SAMPLING_ALL_TRAIN = "all_train"
SAMPLING_ROLLING_MONTHLY_CV = "rolling_monthly_cv"
SAMPLING_JULY16_HOLDOUT = "july16_holdout"
SUPPORTED_SAMPLING_STRATEGIES = (
    SAMPLING_ALL_TRAIN,
    SAMPLING_ROLLING_MONTHLY_CV,
    SAMPLING_JULY16_HOLDOUT,
)


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    error = pred - true
    nonzero = np.abs(true) > 1e-6
    used = int(nonzero.sum())
    excluded = int((~nonzero).sum())
    mape = (
        float(np.mean(np.abs(error[nonzero]) / np.abs(true[nonzero])) * 100)
        if used
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": mape,
        "mape_hours_used": used,
        "mape_hours_zero_excluded": excluded,
    }


def validation_summary(pred_normalized, true_normalized, datasets):
    """计算验证MSE，并以原始单位的非零MAPE作为检查点选择指标。"""
    valid_mse = float(np.mean((pred_normalized - true_normalized) ** 2))
    pred_raw = pred_normalized * datasets["y_std"] + datasets["y_mean"]
    result = metrics(pred_raw, datasets["y_valid_raw"])
    selection_value = result["mape"]
    if not np.isfinite(selection_value):
        selection_value = float("inf")
    return {
        "valid_mse": valid_mse,
        "valid_mape": result["mape"],
        "mape_hours_used": result["mape_hours_used"],
        "mape_hours_zero_excluded": result["mape_hours_zero_excluded"],
        "selection_metric": "nonzero_mape",
        "selection_value": float(selection_value),
    }


def model_config_from_args(args: argparse.Namespace) -> dict:
    return dict(args.model_config)


def model_select(args: argparse.Namespace, feature_names: list[str]) -> nn.Module:
    """兼容入口；模型构造统一交给 load_forecasting.model。"""
    get_model_spec(args.model)
    return build_model(args.model, len(feature_names), model_config_from_args(args))


def build_optimizer(name, parameters, lr, weight_decay=0.0):
    optimizers = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizers:
        raise ValueError(f"未知优化器: {name}")
    return optimizers[name](parameters, lr=lr, weight_decay=weight_decay)


def build_checkpoint(
    model,
    args,
    epoch,
    validation,
    feature_names,
    x_mean,
    x_std,
    y_mean,
    y_std,
    future_weather_names=None,
    future_weather_mean=None,
    future_weather_std=None,
    weather_noise_std=None,
):
    if validation is None:
        selection_metric = "final_epoch"
        best_valid_mape = None
        valid_mse = None
        mape_hours_used = 0
        mape_hours_zero_excluded = 0
    else:
        selection_metric = validation["selection_metric"]
        best_valid_mape = float(validation["selection_value"])
        valid_mse = float(validation["valid_mse"])
        mape_hours_used = validation["mape_hours_used"]
        mape_hours_zero_excluded = validation["mape_hours_zero_excluded"]
    return {
        "checkpoint_version": 1,
        "epoch": epoch,
        "selection_metric": selection_metric,
        "best_valid_mape": best_valid_mape,
        "valid_mse": valid_mse,
        "mape_hours_used": mape_hours_used,
        "mape_hours_zero_excluded": mape_hours_zero_excluded,
        "sampling_strategy": getattr(
            args, "sampling_strategy", SAMPLING_ROLLING_MONTHLY_CV
        ),
        "model_type": args.model,
        "model_config": model_config_from_args(args),
        "training_config": dict(args.training_config),
        "input_dim": len(feature_names),
        "model": model.state_dict(),
        "feature_names": list(feature_names),
        "wavelet_columns": list(
            getattr(args, "wavelet_columns", WAVELET_COLUMNS)
        ),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "input_hours": args.input_hours,
        "horizon": args.horizon,
        "wavelet": getattr(args, "wavelet", "db4"),
        "wavelet_level": getattr(args, "wavelet_level", 3),
        "evaluation_mode": getattr(args, "evaluation_mode", None),
        "future_weather_names": list(future_weather_names or []),
        "future_weather_mean": future_weather_mean,
        "future_weather_std": future_weather_std,
        "weather_noise_std": weather_noise_std,
    }


def save_best_checkpoint(path, valid_loss, best_loss, checkpoint):
    if valid_loss >= best_loss:
        return best_loss, False
    torch.save(checkpoint, path)
    return float(valid_loss), True


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到可用 CUDA GPU")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_time_frames(
    frame, input_hours: int, train_ratio: float = 0.7, valid_ratio: float = 0.15
):
    train_end = int(len(frame) * train_ratio)
    valid_end = int(len(frame) * (train_ratio + valid_ratio))
    train_frame = frame.iloc[:train_end].reset_index(drop=True)
    valid_frame = frame.iloc[train_end - input_hours : valid_end].reset_index(drop=True)
    test_frame = frame.iloc[valid_end - input_hours :].reset_index(drop=True)
    return train_frame, valid_frame, test_frame


def split_sampling_frames(frame, input_hours, strategy):
    """为全量训练返回数据；滚动交叉验证使用专用切分函数。"""
    if strategy == SAMPLING_ALL_TRAIN:
        return frame.reset_index(drop=True), None, None
    if strategy == SAMPLING_ROLLING_MONTHLY_CV:
        raise ValueError("滚动月度交叉验证包含三折，请使用专用切分函数")
    if strategy == SAMPLING_JULY16_HOLDOUT:
        raise ValueError("7月16日隔离验证需要同时提供预测长度")
    raise ValueError(f"未知采样策略: {strategy}")


def _period_frame(frame, start, end):
    timestamps = frame["timeStamp"]
    positions = np.flatnonzero(
        ((timestamps >= start) & (timestamps < end)).to_numpy()
    )
    if not len(positions):
        raise ValueError(f"数据缺少时间段 {start} ～ {end}")
    return frame.iloc[positions[0] : positions[-1] + 1].reset_index(drop=True)


def _period_frame_with_context(frame, start, end, input_hours):
    timestamps = frame["timeStamp"]
    positions = np.flatnonzero(
        ((timestamps >= start) & (timestamps < end)).to_numpy()
    )
    if not len(positions):
        raise ValueError(f"数据缺少时间段 {start} ～ {end}")
    if positions[0] < input_hours:
        raise ValueError(f"{start} 之前不足 {input_hours} 小时历史上下文")
    return frame.iloc[
        positions[0] - input_hours : positions[-1] + 1
    ].reset_index(drop=True)


def split_rolling_monthly_folds(frame, input_hours):
    """构造4～6→7月、4～7→8月、4～8→9月三折。"""
    year = int(frame["timeStamp"].iloc[0].year)
    april_start = pd.Timestamp(year=year, month=4, day=1)
    folds = []
    for index, valid_month in enumerate((7, 8, 9), start=1):
        valid_start = pd.Timestamp(year=year, month=valid_month, day=1)
        valid_end = valid_start + pd.offsets.MonthBegin(1)
        folds.append({
            "name": f"fold_{index}",
            "train": _period_frame(frame, april_start, valid_start),
            "valid": _period_frame_with_context(
                frame, valid_start, valid_end, input_hours
            ),
        })
    return folds


def split_final_evaluation_frames(frame, input_hours):
    """返回4～9月训练段和带历史上下文的10月测试段。"""
    year = int(frame["timeStamp"].iloc[0].year)
    april_start = pd.Timestamp(year=year, month=4, day=1)
    october_start = pd.Timestamp(year=year, month=10, day=1)
    november_start = pd.Timestamp(year=year, month=11, day=1)
    return (
        _period_frame(frame, april_start, october_start),
        _period_frame_with_context(
            frame, october_start, november_start, input_hours
        ),
    )


def split_july16_holdout_frames(frame, input_hours, horizon=24):
    """隔离7月16日作为验证目标，并从训练数据删除该日。"""
    year = int(frame[TIME_COLUMN].iloc[0].year)
    holdout_start = pd.Timestamp(year=year, month=7, day=16)
    holdout_end = holdout_start + pd.Timedelta(hours=horizon)
    context_start = holdout_start - pd.Timedelta(hours=input_hours)
    valid_frame = _period_frame(frame, context_start, holdout_end)
    if len(valid_frame) != input_hours + horizon:
        raise ValueError("7月16日验证窗口时间不连续或数据不完整")
    timestamps = frame[TIME_COLUMN]
    train_frame = frame.loc[
        ~((timestamps >= holdout_start) & (timestamps < holdout_end))
    ].reset_index(drop=True)
    return train_frame, valid_frame


def cv_summary(fold_results):
    """汇总三折平均验证MAPE和最佳轮数中位数。"""
    return {
        "mean_valid_mape": float(
            np.mean([result["best_valid_mape"] for result in fold_results])
        ),
        "median_best_epoch": int(
            np.median([result["best_epoch"] for result in fold_results])
        ),
        "folds": list(fold_results),
    }


def add_weather_noise(weather, noise_scale):
    """只给未来干球、湿球温度加入训练扰动。"""
    scale = noise_scale.to(device=weather.device, dtype=weather.dtype)
    return weather + torch.randn_like(weather) * scale.view(1, 1, -1)


def forward_model(model_type, model, history, future_weather):
    if get_model_spec(model_type)["uses_future_weather"]:
        return model(history, future_weather)
    return model(history)


def _prepare_dataset_from_frames(
    args, train_frame, valid_frame=None, test_frame=None
):
    """用训练段拟合标准化参数，并转换可选的验证段和测试段。"""
    wavelet_columns = getattr(args, "wavelet_columns", WAVELET_COLUMNS)
    x_train, weather_train, y_train, feature_names, weather_names = build_windows_with_future_weather(
        train_frame, args.input_hours, args.horizon, wavelet_columns,
        stride_hours=1,
    )
    x_mean, x_std = x_train.mean(axis=(0, 1)), x_train.std(axis=(0, 1)) + 1e-6
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-6
    weather_mean = weather_train.mean(axis=(0, 1))
    weather_std = weather_train.std(axis=(0, 1)) + 1e-6
    raw_noise = np.array([1.0, 0.5] + [0.0] * 6, dtype=np.float32)
    datasets = {
        "x_train": ((x_train - x_mean) / x_std).astype(np.float32),
        "y_train": ((y_train - y_mean) / y_std).astype(np.float32),
        "future_weather_train": ((weather_train - weather_mean) / weather_std).astype(np.float32),
        "feature_names": feature_names,
        "input_dim": len(feature_names),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "future_weather_names": weather_names,
        "future_weather_mean": weather_mean,
        "future_weather_std": weather_std,
        "weather_noise_std": raw_noise,
        "weather_noise_scale": raw_noise / weather_std,
        "has_validation": valid_frame is not None,
        "has_test": test_frame is not None,
    }
    if valid_frame is not None:
        x_valid, weather_valid, y_valid, valid_feature_names, valid_weather_names = build_windows_with_future_weather(
            valid_frame, args.input_hours, args.horizon, wavelet_columns,
            stride_hours=24,
        )
        if feature_names != valid_feature_names:
            raise ValueError("训练和验证特征顺序不一致")
        if weather_names != valid_weather_names:
            raise ValueError("训练和验证天气特征顺序不一致")
        datasets.update({
            "x_valid": ((x_valid - x_mean) / x_std).astype(np.float32),
            "y_valid": ((y_valid - y_mean) / y_std).astype(np.float32),
            "future_weather_valid": ((weather_valid - weather_mean) / weather_std).astype(np.float32),
            "y_valid_raw": y_valid.astype(np.float32),
        })
    if test_frame is not None:
        x_test, weather_test, y_test, test_feature_names, test_weather_names = build_windows_with_future_weather(
            test_frame, args.input_hours, args.horizon, wavelet_columns,
            stride_hours=24,
        )
        if feature_names != test_feature_names:
            raise ValueError("训练和测试特征顺序不一致")
        if weather_names != test_weather_names:
            raise ValueError("训练和测试天气特征顺序不一致")
        datasets.update({
            "x_test": ((x_test - x_mean) / x_std).astype(np.float32),
            "y_test": ((y_test - y_mean) / y_std).astype(np.float32),
            "future_weather_test": ((weather_test - weather_mean) / weather_std).astype(np.float32),
            "y_test_raw": y_test.astype(np.float32),
        })
    return datasets


def prepare_datasets(args: argparse.Namespace) -> dict:
    frame = load_excel(args.data)
    if args.sampling_strategy == SAMPLING_ALL_TRAIN:
        train_frame, _, _ = split_sampling_frames(
            frame, args.input_hours, args.sampling_strategy
        )
        return _prepare_dataset_from_frames(args, train_frame)
    if args.sampling_strategy == SAMPLING_JULY16_HOLDOUT:
        train_frame, valid_frame = split_july16_holdout_frames(
            frame, args.input_hours, args.horizon
        )
        datasets = _prepare_dataset_from_frames(
            args, train_frame, valid_frame=valid_frame
        )
        datasets["validation_timestamps"] = (
            valid_frame[TIME_COLUMN]
            .iloc[args.input_hours : args.input_hours + args.horizon]
            .to_numpy()[None, ...]
        )
        return datasets
    if args.sampling_strategy != SAMPLING_ROLLING_MONTHLY_CV:
        raise ValueError(f"未知采样策略: {args.sampling_strategy}")

    folds = []
    for fold in split_rolling_monthly_folds(frame, args.input_hours):
        dataset = _prepare_dataset_from_frames(
            args, fold["train"], valid_frame=fold["valid"]
        )
        dataset["name"] = fold["name"]
        folds.append(dataset)
    final_train, october_test = split_final_evaluation_frames(
        frame, args.input_hours
    )
    final_evaluation = _prepare_dataset_from_frames(
        args, final_train, test_frame=october_test
    )
    return {
        "sampling_strategy": SAMPLING_ROLLING_MONTHLY_CV,
        "folds": folds,
        "final_evaluation": final_evaluation,
    }


def train_model(args: argparse.Namespace, datasets: dict, save_checkpoint=True):
    device = select_device(args.device)
    print(f"训练设备: {device}")
    feature_names = datasets["feature_names"]

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(datasets["x_train"]),
            torch.from_numpy(datasets["future_weather_train"]),
            torch.from_numpy(datasets["y_train"]),
        ),
        batch_size=args.training_config["batch_size"], shuffle=True,
    )
    has_validation = datasets.get(
        "has_validation",
        args.sampling_strategy != SAMPLING_ALL_TRAIN,
    )
    has_test = datasets.get("has_test", "x_test" in datasets)
    valid_loader = None
    if has_validation:
        valid_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(datasets["x_valid"]),
                torch.from_numpy(datasets["future_weather_valid"]),
                torch.from_numpy(datasets["y_valid"]),
            ),
            batch_size=args.training_config["batch_size"],
        )

    model = model_select(args, feature_names).to(device)
    optimizer = build_optimizer(
        args.training_config["optimizer"],
        model.parameters(),
        args.training_config["lr"],
        args.training_config["weight_decay"],
    )
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_epoch = 0
    best_validation = None
    best_state = None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    last_train_mse = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for batch_x, batch_weather, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_weather = batch_weather.to(device)
            if get_model_spec(args.model)["uses_future_weather"]:
                batch_weather = add_weather_noise(
                    batch_weather,
                    torch.as_tensor(datasets["weather_noise_scale"]),
                )
            prediction = forward_model(
                args.model, model, batch_x, batch_weather
            )
            loss = loss_fn(prediction, batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(batch_x)
            train_items += len(batch_x)
        last_train_mse = train_loss_sum / train_items

        if not has_validation:
            if getattr(args, "verbose", True) and (
                epoch == 1 or epoch % args.log_every == 0
            ):
                print(f"epoch={epoch} train_mse={last_train_mse:.6f}")
            continue
        model.eval()

        predictions = []
        targets = []
        with torch.no_grad():
            for batch_x, batch_weather, batch_y in valid_loader:
                predictions.append(
                    forward_model(
                        args.model,
                        model,
                        batch_x.to(device),
                        batch_weather.to(device),
                    ).cpu().numpy()
                )
                targets.append(batch_y.numpy())
        validation = validation_summary(
            np.concatenate(predictions),
            np.concatenate(targets),
            datasets,
        )
        if not np.isfinite(validation["selection_value"]):
            raise ValueError("验证集没有非零负荷，无法按非零MAPE选择检查点")
        if validation["selection_value"] < best:
            best = validation["selection_value"]
            best_epoch = epoch
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            checkpoint = build_checkpoint(
                model, args, epoch, validation, feature_names,
                datasets["x_mean"], datasets["x_std"],
                datasets["y_mean"], datasets["y_std"],
                datasets["future_weather_names"],
                datasets["future_weather_mean"],
                datasets["future_weather_std"],
                datasets["weather_noise_std"],
            )
            if save_checkpoint:
                torch.save(checkpoint, output)
        if getattr(args, "verbose", True) and (
            epoch == 1 or epoch % args.log_every == 0
        ):
            print(
                f"epoch={epoch} valid_mse={validation['valid_mse']:.6f} "
                f"valid_mape={validation['valid_mape']:.6f}"
            )

    if not has_validation:
        checkpoint = build_checkpoint(
            model, args, args.epochs, None, feature_names,
            datasets["x_mean"], datasets["x_std"],
            datasets["y_mean"], datasets["y_std"],
            datasets["future_weather_names"],
            datasets["future_weather_mean"],
            datasets["future_weather_std"],
            datasets["weather_noise_std"],
        )
        if save_checkpoint:
            torch.save(checkpoint, output)
        model.eval()
        result = {
            "train_mse": float(last_train_mse),
            "selection_metric": "final_epoch",
        }
        if not has_test:
            return model, result
        with torch.no_grad():
            pred = forward_model(
                args.model,
                model,
                torch.from_numpy(datasets["x_test"]).to(device),
                torch.from_numpy(datasets["future_weather_test"]).to(device),
            ).cpu().numpy()
            pred = pred * datasets["y_std"] + datasets["y_mean"]
        result.update(metrics(pred, datasets["y_test_raw"]))
        if getattr(args, "verbose", True):
            print(json.dumps(result, ensure_ascii=False))
        if save_checkpoint:
            np.savetxt(
                output.with_suffix(".csv"),
                np.column_stack([datasets["y_test_raw"], pred]),
                delimiter=",",
                header="真实值_24步,预测值_24步",
                comments="",
            )
        return model, result

    if save_checkpoint:
        checkpoint = load_checkpoint(output, map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(best_state)
    model.eval()
    validation_result = {
        "best_valid_mape": float(best),
        "best_epoch": int(best_epoch),
        "valid_mse": float(best_validation["valid_mse"]),
        "mape_hours_used": best_validation["mape_hours_used"],
        "mape_hours_zero_excluded": best_validation[
            "mape_hours_zero_excluded"
        ],
    }
    if not has_test:
        validation_output = getattr(args, "validation_output", None)
        if save_checkpoint and validation_output:
            with torch.no_grad():
                validation_prediction = forward_model(
                    args.model,
                    model,
                    torch.from_numpy(datasets["x_valid"]).to(device),
                    torch.from_numpy(
                        datasets["future_weather_valid"]
                    ).to(device),
                ).cpu().numpy()
            validation_prediction = (
                validation_prediction * datasets["y_std"]
                + datasets["y_mean"]
            )
            actual = datasets["y_valid_raw"]
            predicted_flat = validation_prediction.reshape(-1)
            actual_flat = actual.reshape(-1)
            error = predicted_flat - actual_flat
            nonzero = np.abs(actual_flat) > 1e-6
            ape = np.full(actual_flat.shape, np.nan, dtype=float)
            ape[nonzero] = (
                np.abs(error[nonzero]) / np.abs(actual_flat[nonzero]) * 100
            )
            detail = pd.DataFrame({
                "timeStamp": pd.to_datetime(
                    datasets["validation_timestamps"].reshape(-1)
                ),
                "actual_load": actual_flat,
                "predicted_load": predicted_flat,
                "error": error,
                "absolute_error": np.abs(error),
                "ape_percent": ape,
            })
            validation_path = Path(validation_output)
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            detail.to_csv(
                validation_path,
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
            )
        return model, validation_result
    with torch.no_grad():
        pred = forward_model(
            args.model,
            model,
            torch.from_numpy(datasets["x_test"]).to(device),
            torch.from_numpy(datasets["future_weather_test"]).to(device),
        ).cpu().numpy()
        pred = pred * datasets["y_std"] + datasets["y_mean"]
    result = metrics(pred, datasets["y_test_raw"])
    result.update(validation_result)
    if getattr(args, "verbose", True):
        print(json.dumps(result, ensure_ascii=False))
    if save_checkpoint:
        np.savetxt(output.with_suffix(".csv"), np.column_stack([datasets["y_test_raw"], pred]), delimiter=",", header="真实值_24步,预测值_24步", comments="")
    return model, result


def apply_tuning_result(config: Namespace, path) -> Namespace:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    config.model = result.get("model_type", config.model)
    config.model_config = dict(result["model_config"])
    config.training_config = dict(result["training_config"])
    return config


def run_cross_validation(args, prepared, save_checkpoint=True):
    """训练三折，随后按最佳epoch中位数完成4～9月训练和10月测试。"""
    fold_results = []
    for fold in prepared["folds"]:
        print(f"开始 {fold['name']}")
        _, result = train_model(args, fold, save_checkpoint=False)
        fold_results.append({
            "name": fold["name"],
            "best_valid_mape": result["best_valid_mape"],
            "best_epoch": result["best_epoch"],
        })
    summary = cv_summary(fold_results)
    print(json.dumps(summary, ensure_ascii=False))

    final_args = Namespace(**vars(args))
    final_args.epochs = summary["median_best_epoch"]
    print(
        f"使用三折最佳epoch中位数 {final_args.epochs}，"
        "训练4～9月并测试10月"
    )
    model, test_result = train_model(
        final_args,
        prepared["final_evaluation"],
        save_checkpoint=save_checkpoint,
    )
    summary["october_test"] = test_result
    if save_checkpoint:
        summary_path = Path(args.output).with_name(
            Path(args.output).stem + "_cv.json"
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return model, summary


def run(args: argparse.Namespace):
    datasets = prepare_datasets(args)
    if args.sampling_strategy == SAMPLING_ROLLING_MONTHLY_CV:
        return run_cross_validation(args, datasets, save_checkpoint=True)
    return train_model(args, datasets, save_checkpoint=True)


def default_config(
    model_type="LoadTransformer",
    sampling_strategy=SAMPLING_ROLLING_MONTHLY_CV,
) -> Namespace:
    model_config = default_model_config(model_type, horizon=24)
    if sampling_strategy not in SUPPORTED_SAMPLING_STRATEGIES:
        raise ValueError(f"未知采样策略: {sampling_strategy}")
    return Namespace(
        data="附件3：训练数据集/训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx",
        output=(
            f"outputs/{model_type}_all_train.pt"
            if sampling_strategy == SAMPLING_ALL_TRAIN
            else "outputs/LoadTransformer_weather.pt"
            if model_type == "LoadTransformer"
            else f"outputs/{model_type}.pt"
        ),

        model=model_type,
        model_config=model_config,
        wavelet_columns=list(WAVELET_COLUMNS),
        input_hours=168,
        horizon=24,
        epochs=1000,
        sampling_strategy=sampling_strategy,
        training_config=dict(DEFAULT_TRAINING_CONFIG),
        log_every=5,
        verbose=True,
        device="auto",
    )


if __name__ == "__main__":
    run(default_config())
