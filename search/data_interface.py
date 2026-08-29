# -*- coding: utf-8 -*-
"""
data_interface.py —— 数据接口（接线点①）

会议原话（2026-08-24，师兄，解释 hongze.py 里缺失的两行）：
    "creat_dataloader 那两行你可以认为就是创建一个数据集的接口，
     将来接到我那部分。训练脚本每次写得不太一样……所以没发给你。"

本文件就是给师兄的数据加载代码留的"插座"。约定统一返回格式：

    返回 dict，包含：
        "xgboost":     (X_train, y_train, X_val, y_val)   —— 二维特征表 + 一维标签
        "transformer": (train_loader, val_loader, input_dim, output_dim)
                       —— PyTorch DataLoader + 特征维数（供模型占位接口建模型用）
        "lstm":        同 transformer（暂未接线）

当前状态：真实数据未发放、师兄的加载代码未到 → 默认抛 NotImplementedError。
冒烟/演示时传 demo=True，走 demo/synthetic_demo.py 的合成冷站数据。
合成数据只用于验证管线，不进入正式交付流程。
"""

import os
import sys

_DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
if _DEMO_DIR not in sys.path:          # 让 demo/ 下的合成数据模块可以被找到
    sys.path.insert(0, _DEMO_DIR)


def load_xgboost_data(demo=False):
    """
    XGBoost 用：返回 (X_train, y_train, X_val, y_val)，均为 numpy 数组。
    X 形状 (样本数, 特征数)，y 形状 (样本数,)。
    """
    if demo:
        from synthetic_demo import make_xgboost_dataset
        return make_xgboost_dataset()
    # ------------------------- 接线点①-XGBoost -------------------------
    # TODO: 师兄/队友的数据代码到位后，把下面的 raise 替换为真实加载，
    #       并按时间顺序切分训练/验证集（注意：不能随机打散时序数据！）
    #       示例（示意，不要照抄）：
    #       df = pd.read_csv(数据路径)
    #       ... 特征工程（滞后特征、气象特征、时间特征）...
    #       return X_train, y_train, X_val, y_val
    # --------------------------------------------------------------------
    raise NotImplementedError(
        "[接线点①] 真实数据未接入。请替换本函数体，或在演示模式下运行（demo=True）")


def load_sequence_data(model_type, demo=False, **demo_kwargs):
    """
    LSTM / Transformer 用：返回 (train_loader, val_loader, input_dim, output_dim)。
    train_loader 每个 batch 形如 (x, y)：
        x: (batch_size, seq_len, input_dim)  过去 seq_len 步的输入序列
        y: (batch_size, output_dim)          要预测的目标（可改为 (batch, horizon, output_dim)）
    """
    if demo:
        from synthetic_demo import make_sequence_dataset
        # demo 固定窗口长度 12；正式接线时 seq_len 应由寻优试验动态传入（见下方 TODO）
        return make_sequence_dataset(model_type, seq_len=demo_kwargs.get("seq_len", 12),
                                     batch_size=demo_kwargs.get("batch_size", 32))
    # ------------------------- 接线点①-序列模型 -------------------------
    # TODO: 师兄的 creat_dataloader(...) 到位后在此接入。
    #       注意 seq_len 是被寻优的参数，理想情况下 loader 应支持按 seq_len 重建窗口
    #       （接线时需要和师兄确认：窗口长度是寻优时变，还是固定后只寻优模型结构）。
    # --------------------------------------------------------------------
    raise NotImplementedError(
        "[接线点①] 真实数据未接入。请替换本函数体，或在演示模式下运行（demo=True）")
