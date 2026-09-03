"""超参数寻优重写，入口在此"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from load_forecasting.model_registry import (
    SUPPORTED_DEEP_MODEL_TYPES,
    build_model,
    get_model_spec,
    sample_model_configs,
)
from train import (
    apply_tuning_result,
    add_weather_noise,
    build_optimizer,
    default_config,
    metrics,
    forward_model,
    prepare_datasets,
    train_model,
)


def _require_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("运行寻优前请先安装 optuna") from exc
    return optuna


def create_objective(datasets, args):
    def objective(trial):
        model_config, training = sample_model_configs(
            trial, args.model, args.horizon
        )
        device = torch.device(args.device)
        model = build_model(args.model, datasets["input_dim"], model_config).to(device)
        optimizer = build_optimizer(
            training["optimizer"], model.parameters(),
            training["lr"], training["weight_decay"],
        )
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(datasets["x_train"]),
                torch.from_numpy(datasets["future_weather_train"]),
                torch.from_numpy(datasets["y_train"]),
            ),
            batch_size=training["batch_size"], shuffle=True,
        )
        valid_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(datasets["x_valid"]),
                torch.from_numpy(datasets["future_weather_valid"]),
                torch.from_numpy(datasets["y_valid"]),
            ),
            batch_size=training["batch_size"],
        )
        best_value, best_epoch, best_metrics = float("inf"), 0, None
        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, weather, y in train_loader:
                x, weather, y = x.to(device), weather.to(device), y.to(device)
                if get_model_spec(args.model)["uses_future_weather"]:
                    weather = add_weather_noise(
                        weather,
                        torch.as_tensor(datasets["weather_noise_scale"]),
                    )
                optimizer.zero_grad()
                loss_fn(forward_model(args.model, model, x, weather), y).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                pred = np.concatenate(
                    [
                        forward_model(
                            args.model,
                            model,
                            x.to(device),
                            weather.to(device),
                        ).cpu().numpy()
                        for x, weather, _ in valid_loader
                    ]
                )
            current = metrics(
                pred * datasets["y_std"] + datasets["y_mean"],
                datasets["y_valid_raw"],
            )
            value = current["mape"] if np.isfinite(current["mape"]) else float("inf")
            if best_metrics is None or value < best_value:
                best_value, best_epoch, best_metrics = float(value), epoch, current
            trial.report(value, epoch)
            if trial.should_prune():
                raise _require_optuna().TrialPruned()
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("model_config", model_config)
        trial.set_user_attr("training_config", training)
        trial.set_user_attr("mape_hours_used", best_metrics["mape_hours_used"])
        trial.set_user_attr(
            "mape_hours_zero_excluded", best_metrics["mape_hours_zero_excluded"]
        )
        return best_value
    return objective


def default_tuning_config(model_type):
    config = default_config(model_type)
    config.n_trials = 20
    config.timeout = None
    config.epochs = 30
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.study_name = f"{model_type}_A"
    config.study_path = f"optuna_results/{model_type}_A.db"
    config.result_path = f"optuna_results/{model_type}_A_best.json"
    config.output = f"outputs/{model_type}_tuned.pt"
    return config


def save_result(study, args):
    result = {
        "study_name": args.study_name,
        "model_type": args.model,
        "objective": "nonzero_mape",
        "best_value": float(study.best_value),
        "best_epoch": int(study.best_trial.user_attrs["best_epoch"]),
        "model_config": study.best_trial.user_attrs["model_config"],
        "training_config": study.best_trial.user_attrs["training_config"],
    }
    path = Path(args.result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_DEEP_MODEL_TYPES)
    parsed = parser.parse_args()
    args = default_tuning_config(parsed.model)
    optuna = _require_optuna()
    db = Path(args.study_path).resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage="sqlite:///" + db.as_posix(),
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    datasets = prepare_datasets(args)
    study.optimize(
        create_objective(datasets, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    save_result(study, args)
    apply_tuning_result(args, args.result_path)
    train_model(args, datasets, save_checkpoint=True)


if __name__ == "__main__":
    main()
