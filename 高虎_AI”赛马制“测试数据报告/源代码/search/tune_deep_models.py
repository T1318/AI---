"""超参数寻优重写，入口在此"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from load_forecasting.model_registry import (
    SUPPORTED_DEEP_MODEL_TYPES,
    sample_model_configs,
)
from train import (
    SAMPLING_ROLLING_MONTHLY_CV,
    apply_tuning_result,
    cv_summary,
    default_config,
    prepare_datasets,
    train_model,
)


def _require_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("运行寻优前请先安装 optuna") from exc
    return optuna


def train_trial_fold(dataset, args, model_config, training_config):
    """使用一组超参数从头训练一个Fold并返回最佳验证结果。"""
    fold_args = argparse.Namespace(**vars(args))
    fold_args.model_config = dict(model_config)
    fold_args.training_config = dict(training_config)
    fold_args.verbose = False
    _, result = train_model(fold_args, dataset, save_checkpoint=False)
    return result


def create_objective(prepared, args):
    def objective(trial):
        model_config, training = sample_model_configs(
            trial, args.model, args.horizon
        )
        fold_results = []
        for fold_index, dataset in enumerate(prepared["folds"], start=1):
            result = train_trial_fold(
                dataset, args, model_config, training
            )
            fold_results.append({
                "name": dataset["name"],
                "best_valid_mape": result["best_valid_mape"],
                "best_epoch": result["best_epoch"],
            })
            trial.report(
                float(np.mean([
                    item["best_valid_mape"] for item in fold_results
                ])),
                fold_index,
            )
            if trial.should_prune():
                raise _require_optuna().TrialPruned()
        summary = cv_summary(fold_results)
        trial.set_user_attr("best_epoch", summary["median_best_epoch"])
        trial.set_user_attr(
            "fold_best_epochs",
            [item["best_epoch"] for item in fold_results],
        )
        trial.set_user_attr(
            "fold_valid_mapes",
            [item["best_valid_mape"] for item in fold_results],
        )
        trial.set_user_attr("model_config", model_config)
        trial.set_user_attr("training_config", training)
        return summary["mean_valid_mape"]
    return objective


def default_tuning_config(model_type):
    config = default_config(model_type, SAMPLING_ROLLING_MONTHLY_CV)
    config.n_trials = 20
    config.timeout = None
    config.epochs = 30
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.study_name = f"{model_type}_rolling_cv_A"
    config.study_path = f"optuna_results/{model_type}_rolling_cv_A.db"
    config.result_path = f"optuna_results/{model_type}_rolling_cv_A_best.json"
    config.output = f"outputs/{model_type}_tuned.pt"
    return config


def save_result(study, args):
    result = {
        "study_name": args.study_name,
        "model_type": args.model,
        "objective": "three_fold_mean_nonzero_mape",
        "best_value": float(study.best_value),
        "best_epoch": int(study.best_trial.user_attrs["best_epoch"]),
        "fold_best_epochs": study.best_trial.user_attrs["fold_best_epochs"],
        "fold_valid_mapes": study.best_trial.user_attrs["fold_valid_mapes"],
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
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    datasets = prepare_datasets(args)
    study.optimize(
        create_objective(datasets, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    result = save_result(study, args)
    apply_tuning_result(args, args.result_path)
    args.epochs = result["best_epoch"]
    train_model(
        args,
        datasets["final_evaluation"],
        save_checkpoint=True,
    )


if __name__ == "__main__":
    main()
