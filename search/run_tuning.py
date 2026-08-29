# -*- coding: utf-8 -*-
"""
run_tuning.py —— 12 个模型统一的超参数寻优入口（唯一入口）

用法：
    python run_tuning.py --model lstm --project A --n_trials 50
    python run_tuning.py --model cnn_kan_attention --project B --timeout 3600
    python run_tuning.py --model transformer --project C --epochs 40
    python run_tuning.py --model xgboost --project A
    python run_tuning.py --model lstm --demo --n_trials 3        # 合成数据快速自检
    python run_tuning.py --list                                   # 列出全部可用模型

结构（2026-08-28 定稿）：
    search_spaces.py   调哪些参数、范围多大（12 个字典）
    model_registry.py  模型类在外部哪里、参数怎么翻译（队友代码不拷贝进来）
    search_engine.py   共用搜索逻辑：采样翻译 / 建study / 剪枝 / 记录 / 汇报
    data_pipeline.py   临时数据管线：清洗→小时重采样→按形态开窗（队友特征工程到位后整体替换）
    本文件             训练循环（全项目唯一一份）+ 三种输出形态的分支

三种输出形态（对应队友模型的真实接口，2026-08-28 逐一核实）：
    form="seq"      10 个模型：预测窗口内每个时刻的下一步 → 逐点 MSE
    form="horizon"  Transformer：一次输出未来 24 小时 → 24h MSE
    form="seq2seq"  Seq2SeqLSTM：forward(x, y_prev)，训练/验证都喂真实 y_prev（teacher forcing）
    form="flat"     XGBoost：展平窗口预测下一小时（无 epoch 循环，不参与逐轮剪枝）

如实声明（重要）：
    ★ 逐轮汇报和最终返回的都是【验证集】损失（修复 hongze.py 用 train_loss 的问题）
    ★ 不同形态的损失语义不同（逐点 vs 24h），模型间横向比较最优值时要注意口径
    ★ 各试验另记录原始单位 MAPE（剔除真实值为0的小时）到 user_attrs，供答辩引用
"""

import argparse

import numpy as np
import optuna
import torch
import torch.nn as nn

from search_spaces import SEARCH_SPACES
from search_engine import suggest_params, fix_constraints, build_optimizer, run_search, mape  # noqa: F401
import model_registry
import data_pipeline

CONFIG = argparse.Namespace(
    epochs=30,
    device="cuda" if torch.cuda.is_available() else "cpu",
    loss="MSELoss",
    train_ratio=0.8,
    horizon=24,
)


# =============================================================================
# 训练 / 验证（三种形态共用一份循环，只在 batch 解包和 forward 处分支）
# =============================================================================
def _forward(model, batch, form):
    """按形态解包 batch 并前向传播，返回 (pred, y)。"""
    if form == "seq2seq":
        x, y_prev, y = batch
        return model(x, y_prev), y
    x, y = batch
    return model(x), y


def train_one_epoch(model, loader, optimizer, loss_fn, form, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = [t.to(device) for t in batch]
        pred, y = _forward(model, batch, form)
        optimizer.zero_grad()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(pred)
        n += len(pred)
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, form, device, collect=False):
    """验证。collect=True 时额外收集全部预测（算原始单位 MAPE 用）。"""
    model.eval()
    total, n = 0.0, 0
    preds = []
    for batch in loader:
        batch = [t.to(device) for t in batch]
        pred, y = _forward(model, batch, form)
        total += loss_fn(pred, y).item() * len(pred)
        n += len(pred)
        if collect:
            preds.append(pred.cpu().numpy())
    val_loss = total / max(n, 1)
    if collect:
        return val_loss, np.concatenate(preds, axis=0)
    return val_loss


# =============================================================================
# objective：一组超参数 → 验证集损失
# =============================================================================
def make_objective(model_name, entry, model_cls, hourly, args):
    """闭包工厂：把模型/数据/配置固定进 objective。"""
    space = SEARCH_SPACES[model_name]
    form = entry["form"]

    def objective(trial):
        torch.manual_seed(1000 + trial.number)   # 固定随机性，参数对比才公平
        params = suggest_params(trial, space)
        params = fix_constraints(params, entry["constraints"])

        # ---- XGBoost 分支：无 epoch 循环 ----
        if form == "flat":
            tensors = data_pipeline.build_tensors(
                hourly, args.seq_len, "flat",
                train_ratio=args.train_ratio, horizon=args.horizon)
            model = model_cls(**entry["build_params"](params, {}))
            model.fit(tensors["x_train"], tensors["y_train"])
            pred = model.predict(tensors["x_val"])
            val_loss = float(np.mean((pred - tensors["y_val"]) ** 2))
            m, n_used, n_zero = data_pipeline.val_mape_raw(tensors, pred)
            trial.set_user_attr("val_mape", m)
            trial.set_user_attr("mape_hours_used", n_used)
            trial.set_user_attr("mape_hours_zero_excluded", n_zero)
            return val_loss

        # ---- torch 三形态共用循环 ----
        seq_len = params["seq_len"]
        tensors = data_pipeline.build_tensors(
            hourly, seq_len, form,
            train_ratio=args.train_ratio, horizon=args.horizon)
        arrays_tr = [tensors["x_train"], tensors["y_train"]]
        arrays_va = [tensors["x_val"], tensors["y_val"]]
        if form == "seq2seq":
            arrays_tr = [tensors["x_train"], tensors["y_prev_train"], tensors["y_train"]]
            arrays_va = [tensors["x_val"], tensors["y_prev_val"], tensors["y_val"]]
        train_loader = data_pipeline.make_loader(arrays_tr, params["batch_size"], shuffle=True)
        val_loader = data_pipeline.make_loader(arrays_va, params["batch_size"], shuffle=False)

        dims = {"input_dim": tensors["input_dim"], "horizon": args.horizon}
        model = model_cls(**entry["build_params"](params, dims)).to(args.device)
        optimizer = build_optimizer(params["optimizer"], model.parameters(), params["lr"])
        loss_fn = getattr(nn, CONFIG.loss)()

        val_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, loss_fn, form, args.device)
            val_loss = validate(model, val_loader, loss_fn, form, args.device)
            trial.report(val_loss, epoch)             # ★ 汇报验证损失
            if trial.should_prune():
                raise optuna.TrialPruned()

        # 最后一轮收集预测，算原始单位 MAPE 存档（答辩用）
        _, pred = validate(model, val_loader, loss_fn, form, args.device, collect=True)
        m, n_used, n_zero = data_pipeline.val_mape_raw(tensors, pred)
        trial.set_user_attr("val_mape", m)
        trial.set_user_attr("mape_hours_used", n_used)
        trial.set_user_attr("mape_hours_zero_excluded", n_zero)
        return val_loss

    return objective


# =============================================================================
# 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="统一超参数寻优入口")
    parser.add_argument("--model", help="模型名（--list 查看全部12个）")
    parser.add_argument("--project", default="A", choices=["A", "B", "C"],
                        help="冷站项目（数据文件三选一，默认A）")
    parser.add_argument("--demo", action="store_true", help="用合成小时数据代替真实xlsx（自检用）")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--timeout", type=int, default=None, help="最长运行秒数")
    parser.add_argument("--seq_len", type=int, default=24,
                        help="XGBoost 展平窗口长度（其余模型 seq_len 在参数空间里寻优）")
    parser.add_argument("--train_ratio", type=float, default=CONFIG.train_ratio)
    parser.add_argument("--horizon", type=int, default=CONFIG.horizon, help="预测时距（比赛口径24h，勿改）")
    parser.add_argument("--models_dir", default=model_registry.MODELS_DIR_DEFAULT,
                        help="队友模型代码所在目录")
    parser.add_argument("--data_dir", default=data_pipeline.DATA_DIR_DEFAULT,
                        help="真实数据xlsx所在目录")
    parser.add_argument("--list", action="store_true", help="列出全部可用模型后退出")
    args = parser.parse_args()

    if args.list:
        print("可用模型（%d 个）:" % len(model_registry.REGISTRY))
        for name in sorted(model_registry.REGISTRY):
            e = model_registry.REGISTRY[name]
            print("  %-20s form=%-8s %s::%s" % (name, e["form"], e["file"], e["class_name"]))
        return
    if not args.model:
        parser.error("必须指定 --model（或用 --list 查看全部）")

    entry = model_registry.get_entry(args.model)
    model_cls = model_registry.load_model_class(args.model, args.models_dir)
    print("模型: %s（%s::%s, form=%s）" % (args.model, entry["file"], entry["class_name"], entry["form"]))

    # ---- 数据 ----
    if args.demo:
        hourly = data_pipeline.make_demo_hourly()
        print("[数据] demo 合成小时数据: %d 小时 × %d 列" % (len(hourly), hourly.shape[1]))
        tag = "demo"
    else:
        hourly = data_pipeline.load_project_hourly(args.project, data_dir=args.data_dir)
        tag = args.project
    args.device = CONFIG.device
    print("设备=%s, 训练比例=%.2f, horizon=%d" % (args.device, args.train_ratio, args.horizon))

    # ---- 寻优 ----
    run_search(
        objective=make_objective(args.model, entry, model_cls, hourly, args),
        study_name="%s_%s" % (args.model, tag),
        n_trials=args.n_trials,
        timeout=args.timeout,
        storage_dir="optuna_results",
        pruner_warmup=max(3, args.epochs // 5),   # 前几轮不掐，给慢热模型机会
    )


if __name__ == "__main__":
    main()
