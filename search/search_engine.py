# -*- coding: utf-8 -*-
"""
search_engine.py —— 三个模型共用的搜索逻辑（会议确认的"同一套"）

会议原话（2026-08-24，师兄）：
    "创建三个字典……但是他们的搜索，我应该是用同一套（逻辑）。"

本文件实现这套共用逻辑，包括：
    1. suggest_params(): 把 search_spaces.py 里的参数字典翻译成 Optuna 建议
    2. run_search():     创建 study、跑寻优、输出最优参数（含 sqlite 断点续跑）
    3. build_optimizer(): 按名字字符串构造 PyTorch 优化器（同 hongze.py 的写法）

入口脚本（run_xgboost.py / run_transformer.py / 将来 run_lstm.py）
只需要提供各自的 objective 函数（"一组参数 → 一个验证分数"），
其余全部复用这里的逻辑——这就是"三个字典 + 一套搜索"的落地方式。

重要设计决策（来自 hongze.py 的教训，会议前已与用户确认）：
    ★ objective 返回的必须是【验证集】指标，不是训练集指标，
      否则 Optuna 会挑出最容易过拟合的参数组。
"""

import optuna


# -----------------------------------------------------------------------------
# 1. 参数翻译：字典 → Optuna 建议
# -----------------------------------------------------------------------------
def suggest_params(trial, space):
    """
    遍历一个模型的超参数空间字典，对每个参数调用 Optuna 对应的 suggest 方法。

    参数:
        trial : optuna.trial.Trial，Optuna 传进来的"本次试验"对象
        space : dict，形如 search_spaces.XGBOOST_SPACE

    返回:
        params : dict {参数名: 本试验采用的值}，直接可传给模型构建函数
    """
    params = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"],
                step=spec.get("step", 1), log=spec.get("log", False))
        elif t == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False))
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError("未知参数类型 %r（参数 %r），请检查 search_spaces.py" % (t, name))
    return params


def fix_constraints(params, rules):
    """
    通用版约束修正：采样后修正可能违反结构约束的参数组合。

    rules 由 model_registry.py 按模型声明，目前支持一种规则：
        {"type": "divisible", "big": "d_model", "small": "nhead"}
        含义：params[big] 必须能被 params[small] 整除。
        修正方式：把 small 降到能整除的最大值（保持结构合法，代价是采样分布略偏，
        追求严格分布的话应改用条件采样，这里选择简单可靠的降头修正）。
    """
    for rule in rules or []:
        if rule.get("type") != "divisible":
            raise ValueError("未知约束类型 %r" % rule)
        big, small = rule["big"], rule["small"]
        if big in params and small in params:
            b, s = params[big], params[small]
            while s > 1 and b % s != 0:
                s -= 1
            params[small] = s
    return params


def fix_transformer_constraints(params):
    """旧入口兼容：Transformer 的 d_model % nhead 整除修正（内部走通用版）。"""
    return fix_constraints(params, [{"type": "divisible", "big": "d_model", "small": "nhead"},
                                    {"type": "divisible", "big": "d_model", "small": "num_heads"}])


# -----------------------------------------------------------------------------
# 2. PyTorch 优化器构造（与 hongze.py 相同的花活，按字符串名字取类）
# -----------------------------------------------------------------------------
def build_optimizer(name, model_params, lr):
    """
    name : "Adam" / "AdamW" / "RMSprop" / "SGD" 等字符串
    返回构造好的 optimizer 实例。
    """
    import torch.optim
    try:
        cls = torch.optim.__dict__[name]
    except KeyError:
        raise ValueError("未知优化器 %r，请检查 search_spaces.py 的 choices" % name)
    return cls(params=model_params, lr=lr)


# -----------------------------------------------------------------------------
# 3. 主搜索流程
# -----------------------------------------------------------------------------
def run_search(objective,                  # 函数: trial -> 验证指标（越小越好）
              study_name,                  # 本次寻优任务名（sqlite 里的标签，断点续跑靠它）
              n_trials=20,                 # 最多试多少组参数
              timeout=None,                # 最长运行秒数（与 n_trials 先到为准）
              storage_dir=".",             # sqlite 数据库所在目录
              pruner_warmup=10,            # 剪枝宽限：前 N 个 epoch 不掐（深度模型用）
              direction="minimize",        # 指标方向：损失/MAPE 用 minimize
              verbose=True):
    """
    共用搜索主入口。返回 (best_params, best_value, study)。

    剪枝器 MedianPruner 的含义：
        某试验进行到第 N 步时，指标比其他试验在第 N 步的中位数还差 → 提前掐掉，
        省下的时间用于下一组试验。n_warmup_steps=10 表示前10步不掐（刚起步谁都说不好）。
    """
    import os
    os.makedirs(storage_dir, exist_ok=True)
    storage = "sqlite:///" + os.path.join(storage_dir, "optuna_%s.db" % study_name)

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=pruner_warmup),
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,   # 数据库里已有同名任务 → 接着跑，不重头再来
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   callbacks=[_print_progress] if verbose else None)

    if verbose:
        print("\n" + "=" * 60)
        print("寻优完成。共 %d 次试验（其中被剪枝 %d 次）"
              % (len(study.trials),
                 sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)))
        print("最优验证指标: %.6f" % study.best_value)
        print("最优超参数:")
        for k, v in study.best_params.items():
            print("    %-20s = %s" % (k, v))
        print("试验记录已存至: %s" % storage)
        print("=" * 60)

    return study.best_params, study.best_value, study


def _print_progress(study, trial):
    """每次试验结束时打印一行进度（回调）。"""
    state = trial.state.name
    val = "%.6f" % trial.value if trial.value is not None else "  (剪枝/失败)"
    best = "%.6f" % study.best_value if study.best_value is not None else "-"
    print("[trial %3d] state=%-8s value=%s  (当前最优=%s)"
          % (trial.number, state, val, best))


# -----------------------------------------------------------------------------
# 通用指标（竞赛评分口径就是 MAPE，模型无关，三个入口共用）
# -----------------------------------------------------------------------------
def mape(y_true, y_pred, eps=1e-6):
    """平均绝对百分比误差（%）。注意：真实值接近 0 时 MAPE 会爆炸，加 eps 保护。"""
    n = len(y_true)
    s = 0.0
    for i in range(n):
        t = float(y_true[i])
        p = float(y_pred[i])
        s += abs((t - p) / (abs(t) + eps))
    return 100.0 * s / n
