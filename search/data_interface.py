# -*- coding: utf-8 -*-
"""统一的数据加载接口。"""

import os
import sys

from . import data_pipeline

_DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)


def load_xgboost_data(project="A", data_dir=None, seq_len=24, train_ratio=0.8,
                      demo=False):
    """加载 XGBoost 所需的扁平窗口数据。"""
    if demo:
        from synthetic_demo import make_xgboost_dataset
        return make_xgboost_dataset()

    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "附件3：训练数据集",
        )
    hourly = data_pipeline.load_project_hourly(project, data_dir=data_dir)
    tensors = data_pipeline.build_tensors(
        hourly, seq_len, "flat", train_ratio=train_ratio
    )
    return (tensors["x_train"], tensors["y_train"],
            tensors["x_val"], tensors["y_val"])


def load_sequence_data(model_type, demo=False, **demo_kwargs):
    """加载 LSTM / Transformer 所需的序列数据。"""
    if demo:
        from synthetic_demo import make_sequence_dataset
        return make_sequence_dataset(
            model_type,
            seq_len=demo_kwargs.get("seq_len", 12),
            batch_size=demo_kwargs.get("batch_size", 32),
        )
    raise NotImplementedError(
        "真实序列数据尚未接入，请在演示模式下运行（demo=True）"
    )
