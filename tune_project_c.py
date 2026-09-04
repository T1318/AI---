"""项目C超参寻优入口。

复用 search/tune_deep_models 的寻优流程，数据窗口限定 2026-04-13 ~ 07-31
（泵持续运转期，剔除隆冬全停段）；先 Optuna 寻优（默认 20 trial × 30 轮），
再用最优超参完整重训（默认 300 轮）。

用法:
    python tune_project_c.py --model lstm gru
    python tune_project_c.py --all
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train as train_module
from load_forecasting import data as data_module
from train_project_c import PROJECT_C_DATA, PROJECT_C_WAVELET_COLUMNS

DATE_START = "2026-04-13"
DATE_END = "2026-07-31 23:59:59"

TUNE_MODELS = [
    "lstm", "gru", "bilstm", "seq2seq",
    "cnn_lstm", "cnn_lstm_attention",
    "kan_lstm", "cnn_kan", "cnn_kan_attention",
]

data_module.WAVELET_COLUMNS = PROJECT_C_WAVELET_COLUMNS
train_module.WAVELET_COLUMNS = PROJECT_C_WAVELET_COLUMNS

_orig_load_excel = train_module.load_excel


def load_excel_date_filtered(path):
    frame = _orig_load_excel(path)
    t = pd.to_datetime(frame[data_module.TIME_COLUMN])
    out = frame.loc[(t >= pd.Timestamp(DATE_START)) & (t <= pd.Timestamp(DATE_END))].reset_index(drop=True)
    print(f"[数据窗口] {DATE_START}~{DATE_END}: {len(frame)} -> {len(out)} 小时 "
          f"({out[data_module.TIME_COLUMN].iloc[0]} ~ {out[data_module.TIME_COLUMN].iloc[-1]})")
    return out


train_module.load_excel = load_excel_date_filtered


def tune_model(model: str, datasets: dict, n_trials: int, tune_epochs: int,
               final_epochs: int, device: str, tag: str) -> dict:
    from search.tune_deep_models import _require_optuna, create_objective, save_result

    optuna = _require_optuna()
    args = train_module.default_config(model)
    args.data = PROJECT_C_DATA
    args.device = device
    args.output = Path(f"outputs/{model}_{tag}_tuned.pt")
    args.n_trials = n_trials
    args.epochs = tune_epochs
    args.study_name = f"{model}_{tag}"
    args.study_path = f"optuna_results/{model}_{tag}.db"
    args.result_path = f"optuna_results/{model}_{tag}_best.json"

    db = Path(args.study_path).resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage="sqlite:///" + db.as_posix(),
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    print(f"\n===== [{model}] Optuna 寻优: {n_trials} trials x {tune_epochs} epochs =====")
    t0 = time.time()
    study.optimize(create_objective(datasets, args), n_trials=n_trials)
    print(f"[{model}] 寻优耗时 {time.time() - t0:.0f}s, 完成/剪枝 trial: "
          f"{sum(1 for s in study.trials if s.state.name == 'COMPLETE')}"
          f"/{sum(1 for s in study.trials if s.state.name == 'PRUNED')}")

    if not study.trials or not np.isfinite(study.best_value):
        print(f"[{model}] 无有效 trial（目标值均为 inf），跳过最终重训")
        return {"model": model, "status": "no_valid_trial"}

    result = save_result(study, args)
    print(f"[{model}] 最优验证MAPE={result['best_value']:.4f}% "
          f"(trial内best_epoch={result['best_epoch']})\n"
          f"    model_config={result['model_config']}\n"
          f"    training_config={result['training_config']}")

    train_module.apply_tuning_result(args, args.result_path)
    args.epochs = final_epochs
    print(f"===== [{model}] 最优超参完整重训: {final_epochs} epochs -> {args.output} =====")
    t1 = time.time()
    _, test_metrics = train_module.train_model(args, datasets, save_checkpoint=True)
    ck = torch.load(args.output, weights_only=False, map_location="cpu")
    print(f"[{model}] 重训耗时 {time.time() - t1:.0f}s, "
          f"best epoch={ck['epoch']}, 验证MAPE={ck['best_valid_mape']:.4f}%")
    return {
        "model": model,
        "status": "ok",
        "valid_mape": float(ck["best_valid_mape"]),
        "best_epoch": int(ck["epoch"]),
        "test": {k: round(float(v), 4) for k, v in test_metrics.items() if isinstance(v, (int, float, np.floating))},
        "model_config": result["model_config"],
        "training_config": result["training_config"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="项目C超参寻优（数据窗口 2026-04-13~07-31）")
    parser.add_argument("--model", nargs="+", choices=TUNE_MODELS)
    parser.add_argument("--all", action="store_true", help="全部9个模型（除kan/LoadTransformer）")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trial 数，默认20")
    parser.add_argument("--tune-epochs", type=int, default=30, help="每trial训练轮数，默认30")
    parser.add_argument("--final-epochs", type=int, default=300, help="寻优后完整重训轮数，默认300")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--tag", default="C2", help="输出文件后缀标签，默认C2")
    parsed = parser.parse_args()
    # create_objective 内部直接 torch.device(args.device)，不接受 auto，须先解析
    parsed.device = train_module.select_device(parsed.device).type

    models = TUNE_MODELS if parsed.all else (parsed.model or [])
    if not models:
        parser.error("请用 --model 指定模型或使用 --all")
    if not Path(PROJECT_C_DATA).exists():
        raise FileNotFoundError(f"找不到项目C数据文件: {PROJECT_C_DATA}")

    config = train_module.default_config(models[0])
    config.data = PROJECT_C_DATA
    config.device = parsed.device
    print(f"项目C寻优: 模型={models} 数据窗口={DATE_START}~{DATE_END} "
          f"trials={parsed.trials} tune_epochs={parsed.tune_epochs} "
          f"final_epochs={parsed.final_epochs} device={parsed.device} tag={parsed.tag}")

    datasets = train_module.prepare_datasets(config)
    print(f"窗口样本: 训练={datasets['x_train'].shape} 验证={datasets['x_valid'].shape} "
          f"测试={datasets['x_test'].shape} 特征维度={datasets['input_dim']}")

    results = []
    for m in models:
        try:
            results.append(tune_model(
                m, datasets, parsed.trials, parsed.tune_epochs,
                parsed.final_epochs, parsed.device, parsed.tag,
            ))
        except Exception as exc:  # 单模型失败不阻断后续模型
            print(f"[{m}] 训练失败: {exc!r}")
            results.append({"model": m, "status": f"failed: {exc!r}"})

    print("\n===== 全部模型汇总 =====")
    for r in results:
        if r.get("status") == "ok":
            print(f"{r['model']:22s} 验证MAPE={r['valid_mape']:7.2f}%  "
                  f"best_epoch={r['best_epoch']:4d}  "
                  f"测试MAE={r['test'].get('mae', float('nan')):.2f} "
                  f"RMSE={r['test'].get('rmse', float('nan')):.2f} "
                  f"MAPE={r['test'].get('mape', float('nan')):.2f}%")
        else:
            print(f"{r['model']:22s} {r['status']}")


if __name__ == "__main__":
    main()
