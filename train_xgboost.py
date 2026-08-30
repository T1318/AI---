"""使用统一116维窗口特征训练或寻优多输出 XGBoost。"""
import argparse
from argparse import Namespace
import json
from pathlib import Path

import numpy as np

from load_forecasting.model_registry import (
    build_model,
    default_model_config,
    sample_model_configs,
)
from train import default_config, metrics, prepare_datasets


def flatten_windows(x):
    return x.reshape(len(x), -1)


def _build_xgboost(params):
    try:
        return build_model("xgboost", 0, params)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("运行 XGBoost 前请先安装 xgboost 和 joblib") from exc


def train_xgboost_model(datasets, params, output_path=None):
    model = _build_xgboost(params)
    model.fit(flatten_windows(datasets["x_train"]), datasets["y_train"])
    pred = model.predict(flatten_windows(datasets["x_test"]))
    pred_raw = pred * datasets["y_std"] + datasets["y_mean"]
    result = metrics(pred_raw, datasets["y_test_raw"])
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
    return model, result


def create_xgboost_objective(datasets):
    def objective(trial):
        params, _ = sample_model_configs(trial, "xgboost")
        model = _build_xgboost(params)
        model.fit(flatten_windows(datasets["x_train"]), datasets["y_train"])
        pred = model.predict(flatten_windows(datasets["x_valid"]))
        pred_raw = pred * datasets["y_std"] + datasets["y_mean"]
        return metrics(pred_raw, datasets["y_valid_raw"])["mape"]
    return objective


def save_xgboost_result(study, path):
    result = {
        "model_type": "xgboost",
        "objective": "nonzero_mape",
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def tune_xgboost(config, datasets):
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("运行 XGBoost 寻优前请先安装 optuna") from exc
    db = Path(config.study_path).resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="xgboost_A",
        storage="sqlite:///" + db.as_posix(),
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(
        create_xgboost_objective(datasets),
        n_trials=config.n_trials,
        timeout=config.timeout,
    )
    save_xgboost_result(study, config.result_path)
    return train_xgboost_model(datasets, study.best_params, config.output)


def default_xgboost_config():
    config = default_config()
    config.output = "outputs/xgboost_model.joblib"
    config.params = default_model_config("xgboost")
    config.n_trials = 20
    config.timeout = None
    config.study_path = "optuna_results/xgboost_A.db"
    config.result_path = "optuna_results/xgboost_A_best.json"
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    config = default_xgboost_config()
    datasets = prepare_datasets(config)
    if args.tune:
        _, result = tune_xgboost(config, datasets)
    else:
        _, result = train_xgboost_model(datasets, config.params, config.output)
    print(result)


if __name__ == "__main__":
    main()
