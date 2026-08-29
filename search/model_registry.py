# -*- coding: utf-8 -*-
"""
model_registry.py —— 12 个模型的注册表（模型名 → 外部模型代码 + 参数翻译）

职责边界（2026-08-28 与用户确认）：
    队友的模型代码【不拷贝】进本目录——本文件只记录"去哪里找、叫什么类、
    参数怎么翻译"，运行时从外部路径动态加载（默认指向桌面 赛马ai/，可用
    --models_dir 改）。队友更新模型代码后你零同步成本，交付物只含你自己的活。

每条注册项的字段：
    file          : 模型所在文件名（相对 models_dir）
    class_name    : 模型类名
    form          : 输出形态（决定训练目标怎么构造，见 data_pipeline.py）
                       "seq"      输出 (batch, seq_len, output)  逐时刻预测下一步
                       "horizon"  输出 (batch, 24)                直接预测未来24小时
                       "seq2seq"  forward(x, y_prev) 双输入
                       "flat"     非时序展平特征（XGBoost）
    constraints   : 结构整除约束，交给 search_engine.fix_constraints 修正
    build_params  : 采样字典 + 数据维度 → 模型构造函数的命名参数（翻译层）

用法：
    from model_registry import load_model_class, get_entry
    cls = load_model_class("cnn_lstm")            # 动态加载外部模型类
    entry = get_entry("cnn_lstm")
    kwargs = entry["build_params"](params, dims)  # dims 需含 input_dim（和 horizon）
    model = cls(**kwargs)
"""

import importlib.util
import os

MODELS_DIR_DEFAULT = r"C:/Users/23263/Desktop/赛马/赛马ai"

_DIV = lambda big, small: {"type": "divisible", "big": big, "small": small}


# =============================================================================
# 参数翻译函数：采样字典 → 模型构造参数
#   约定：d 是 run_tuning.py 传入的维度信息 {"input_dim": F, "horizon": 24}
# =============================================================================
def _rnn_params(p, d):
    """LSTM / GRU / BiLSTM：构造参数与采样键同名，直接映射。"""
    return {
        "input_size": d["input_dim"],
        "hidden_size": p["hidden_size"],
        "output_size": 1,
        "num_layers": p["num_layers"],
        "dropout": p["dropout"],
    }


def _seq2seq_params(p, d):
    return {
        "input_size": d["input_dim"],
        "encoder_hidden_size": p["encoder_hidden_size"],
        "decoder_hidden_size": p["decoder_hidden_size"],
        "output_size": 1,
        "encoder_num_layers": p["encoder_num_layers"],
        "decoder_num_layers": p["decoder_num_layers"],
        "dropout": p["dropout"],
    }


def _kan_params(p, d):
    """KAN：把 kan_depth/kan_width 两个结构设计参数折算成 hidden_sizes 列表。"""
    return {
        "input_size": d["input_dim"],
        "hidden_sizes": [p["kan_width"]] * p["kan_depth"],
        "output_size": 1,
        "grid_size": p["grid_size"],
        "spline_order": p["spline_order"],
        "dropout": p["dropout"],
    }


def _kan_lstm_params(p, d):
    return {
        "input_size": d["input_dim"],
        "kan_hidden_sizes": [p["kan_width"]] * p["kan_depth"],
        "kan_output_size": p["kan_output_size"],
        "lstm_hidden_size": p["lstm_hidden_size"],
        "output_size": 1,
        "lstm_num_layers": p["lstm_num_layers"],
        "grid_size": p["grid_size"],
        "spline_order": p["spline_order"],
        "dropout": p["dropout"],
    }


def _cnn_lstm_params(p, d):
    return {
        "input_size": d["input_dim"],
        "conv_channels": p["conv_channels"],
        "kernel_size": p["kernel_size"],
        "lstm_hidden_size": p["lstm_hidden_size"],
        "output_size": 1,
        "num_conv_layers": p["num_conv_layers"],
        "lstm_num_layers": p["lstm_num_layers"],
        "dropout": p["dropout"],
    }


def _cnn_lstm_att_params(p, d):
    return {
        "input_size": d["input_dim"],
        "conv_channels": p["conv_channels"],
        "kernel_size": p["kernel_size"],
        "num_conv_layers": p["num_conv_layers"],
        "lstm_hidden_size": p["lstm_hidden_size"],
        "lstm_num_layers": p["lstm_num_layers"],
        "output_size": 1,
        "num_heads": p["num_heads"],
        "dropout": p["dropout"],
    }


def _cnn_kan_params(p, d):
    return {
        "input_size": d["input_dim"],
        "conv_channels": p["conv_channels"],
        "kernel_size": p["kernel_size"],
        "num_conv_layers": p["num_conv_layers"],
        "kan_hidden_sizes": [p["kan_width"]] * p["kan_depth"],
        "output_size": 1,
        "grid_size": p["grid_size"],
        "spline_order": p["spline_order"],
        "dropout": p["dropout"],
    }


def _cnn_kan_att_params(p, d):
    return {
        "input_size": d["input_dim"],
        "conv_channels": p["conv_channels"],
        "kernel_size": p["kernel_size"],
        "num_conv_layers": p["num_conv_layers"],
        "kan_hidden_sizes": [p["kan_width"]] * p["kan_depth"],
        "output_size": 1,
        "num_heads": p["num_heads"],
        "grid_size": p["grid_size"],
        "spline_order": p["spline_order"],
        "dropout": p["dropout"],
    }


def _transformer_params(p, d):
    """队友 Nanwang_Transformer：构造参数名是 nhead；horizon 固定 24（比赛口径）。"""
    return {
        "input_dim": d["input_dim"],
        "d_model": p["d_model"],
        "nhead": p["nhead"],
        "num_layers": p["num_layers"],
        "horizon": d["horizon"],
        "dropout": p["dropout"],
    }


def _xgboost_params(p, d):
    """XGBoostModel(**params) 直接透传 XGBRegressor 原生参数。"""
    return dict(p)


# =============================================================================
# 注册表本体
# =============================================================================
REGISTRY = {
    "xgboost": {
        "file": "xgboost_model.py", "class_name": "XGBoostModel",
        "form": "flat", "constraints": [], "build_params": _xgboost_params,
    },
    "lstm": {
        "file": "lstm_model.py", "class_name": "LSTMModel",
        "form": "seq", "constraints": [], "build_params": _rnn_params,
    },
    "gru": {
        "file": "gru_model.py", "class_name": "GRUModel",
        "form": "seq", "constraints": [], "build_params": _rnn_params,
    },
    "bilstm": {
        "file": "bilstm_model.py", "class_name": "BiLSTMModel",
        "form": "seq", "constraints": [], "build_params": _rnn_params,
    },
    "seq2seq": {
        "file": "lstm_seq2seq_model.py", "class_name": "Seq2SeqLSTM",
        "form": "seq2seq", "constraints": [], "build_params": _seq2seq_params,
    },
    "kan": {
        "file": "kan_model.py", "class_name": "KANModel",
        "form": "seq", "constraints": [], "build_params": _kan_params,
    },
    "kan_lstm": {
        "file": "kan_lstm_model.py", "class_name": "KanLstmModel",
        "form": "seq", "constraints": [], "build_params": _kan_lstm_params,
    },
    "cnn_lstm": {
        "file": "cnn_lstm_model.py", "class_name": "CnnLstmModel",
        "form": "seq", "constraints": [], "build_params": _cnn_lstm_params,
    },
    "cnn_lstm_attention": {
        "file": "cnn_lstm_attention_model.py", "class_name": "CnnLstmAttentionModel",
        "form": "seq", "constraints": [_DIV("lstm_hidden_size", "num_heads")],
        "build_params": _cnn_lstm_att_params,
    },
    "cnn_kan": {
        "file": "cnn_kan_model.py", "class_name": "CnnKanModel",
        "form": "seq", "constraints": [], "build_params": _cnn_kan_params,
    },
    "cnn_kan_attention": {
        "file": "cnn_kan_attention_model.py", "class_name": "CnnKanAttentionModel",
        "form": "seq", "constraints": [_DIV("conv_channels", "num_heads")],
        "build_params": _cnn_kan_att_params,
    },
    "transformer": {
        "file": "model.py", "class_name": "Nanwang_Transformer",
        "form": "horizon", "constraints": [_DIV("d_model", "nhead")],
        "build_params": _transformer_params,
    },
}


# =============================================================================
# 动态加载（带缓存；路径含中文也能加载，importlib 按文件路径读源码）
# =============================================================================
_LOADED = {}


def get_entry(model_name):
    if model_name not in REGISTRY:
        raise KeyError("未注册的模型 %r。可用: %s" % (model_name, ", ".join(sorted(REGISTRY))))
    return REGISTRY[model_name]


def load_model_class(model_name, models_dir=MODELS_DIR_DEFAULT):
    """从外部目录动态加载模型类。返回类对象（不实例化）。"""
    entry = get_entry(model_name)
    path = os.path.join(models_dir, entry["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            "找不到模型文件 %s\n（外部模型目录 models_dir=%s，可用 --models_dir 修改；\n"
            "队友的模型代码不拷贝进本工程，注册表只做引用）" % (path, models_dir))
    key = os.path.abspath(path) + "::" + entry["class_name"]
    if key in _LOADED:
        return _LOADED[key]
    spec = importlib.util.spec_from_file_location(
        "_ext_model_%s" % model_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, entry["class_name"])
    _LOADED[key] = cls
    return cls


if __name__ == "__main__":
    # 自检：逐个加载外部模型类并打印（需要外部模型目录存在）
    for name in sorted(REGISTRY):
        entry = REGISTRY[name]
        try:
            cls = load_model_class(name)
            print("OK   %-20s -> %s.%s" % (name, entry["file"], entry["class_name"]))
            _ = cls.__name__  # noqa
        except Exception as e:  # noqa
            print("FAIL %-20s -> %s" % (name, e))
