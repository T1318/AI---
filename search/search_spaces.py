# -*- coding: utf-8 -*-
"""
search_spaces.py —— 12 个模型的超参数空间配置库（超参数库）

设计沿用 2026-08-24 会议约定："每一个超参数对应字典的一个键，搜索逻辑用同一套"。

字典格式（由 search_engine.suggest_params 统一翻译成 Optuna 调用）：
    {"type": "int",         "low": 100, "high": 1000}              整数，闭区间
    {"type": "float",       "low": 1e-3, "high": 0.3, "log": True} 连续小数，log=按对数轴采样
    {"type": "categorical", "choices": ["Adam", "AdamW"]}          只能从给定选项中选

键名规则（重要）：
    A. 键名 = 队友模型类的构造参数名  → 原样传给模型（如 hidden_size、num_layers）
    B. 键名 = 训练参数（lr / batch_size / optimizer / seq_len）→ 引擎和训练循环消费
    C. 键名 = 结构设计参数（kan_depth / kan_width）→ 不是构造参数名，
       由 model_registry.py 的翻译函数折算成构造参数（如 hidden_sizes=[width]*depth）

修改原则：只改这一个文件就能调整任何模型的寻参范围，不用动引擎和入口。
整除约束（多头注意力）由 model_registry.py 声明、search_engine.fix_constraints 修正。
"""

# =============================================================================
# 训练参数（各深度模型共用，写在一起方便统一改）
# =============================================================================
_TRAINING = {
    "lr":        {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "optimizer": {"type": "categorical", "choices": ["Adam", "AdamW", "RMSprop"]},
    "seq_len":   {"type": "int", "low": 6, "high": 24},   # 回看多少个小时
}
_RNN_STRUCT = {   # LSTM/GRU/BiLSTM 共用的结构段
    "hidden_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
    "num_layers":  {"type": "int", "low": 1, "high": 3},
    "dropout":     {"type": "float", "low": 0.0, "high": 0.5},
}
_KAN_STRUCT = {   # KAN 系列共用的结构段
    "kan_depth":    {"type": "int", "low": 1, "high": 3},                 # KAN 层数
    "kan_width":    {"type": "categorical", "choices": [8, 16, 32, 64]},  # 每层宽度
    "grid_size":    {"type": "categorical", "choices": [3, 5, 7]},        # B样条网格数
    "spline_order": {"type": "categorical", "choices": [2, 3]},           # 样条阶数
    "dropout":      {"type": "float", "low": 0.0, "high": 0.5},
}
_CNN_STRUCT = {   # CNN 系列共用的结构段（kernel 只取奇数，保证卷积后长度不变）
    "conv_channels":  {"type": "categorical", "choices": [16, 32, 64]},
    "kernel_size":    {"type": "categorical", "choices": [3, 5, 7]},
    "num_conv_layers": {"type": "int", "low": 1, "high": 3},
}


def _merge(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out


# =============================================================================
# 1. XGBoost（构造参数与 XGBRegressor 原生参数一致，原样透传）
# =============================================================================
XGBOOST_SPACE = {
    "n_estimators":     {"type": "int",   "low": 100, "high": 1000},
    "max_depth":        {"type": "int",   "low": 3,   "high": 10},
    "learning_rate":    {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
    "subsample":        {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    "min_child_weight": {"type": "int",   "low": 1,   "high": 10},
    "reg_lambda":       {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
}

# =============================================================================
# 2~4. 纯 RNN 系：LSTM / GRU / BiLSTM（结构同构，各自一份字典方便单独改范围）
# =============================================================================
LSTM_SPACE   = _merge(_RNN_STRUCT, _TRAINING)
GRU_SPACE    = _merge(_RNN_STRUCT, _TRAINING)
BILSTM_SPACE = _merge(_RNN_STRUCT, _TRAINING)   # hidden_size 为单侧宽度，双向拼接后翻倍

# =============================================================================
# 5. Seq2Seq LSTM（编码器/解码器分开调）
# =============================================================================
SEQ2SEQ_SPACE = _merge({
    "encoder_hidden_size":  {"type": "categorical", "choices": [32, 64, 128, 256]},
    "decoder_hidden_size":  {"type": "categorical", "choices": [32, 64, 128, 256]},
    "encoder_num_layers":   {"type": "int", "low": 1, "high": 3},
    "decoder_num_layers":   {"type": "int", "low": 1, "high": 3},
    "dropout":              {"type": "float", "low": 0.0, "high": 0.5},
}, _TRAINING)

# =============================================================================
# 6. 纯 KAN
# =============================================================================
KAN_SPACE = _merge(_KAN_STRUCT, _TRAINING)

# =============================================================================
# 7. KAN + LSTM
# =============================================================================
KAN_LSTM_SPACE = _merge(_KAN_STRUCT, {
    "kan_output_size":   {"type": "categorical", "choices": [8, 16, 32, 64]},  # KAN 输出宽度=LSTM输入
    "lstm_hidden_size":  {"type": "categorical", "choices": [32, 64, 128, 256]},
    "lstm_num_layers":   {"type": "int", "low": 1, "high": 3},
}, _TRAINING)

# =============================================================================
# 8. CNN + LSTM
# =============================================================================
CNN_LSTM_SPACE = _merge(_CNN_STRUCT, {
    "lstm_hidden_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
    "lstm_num_layers":  {"type": "int", "low": 1, "high": 3},
    "dropout":          {"type": "float", "low": 0.0, "high": 0.5},
}, _TRAINING)

# =============================================================================
# 9. CNN + LSTM + Attention（约束：lstm_hidden_size % num_heads == 0）
# =============================================================================
CNN_LSTM_ATT_SPACE = _merge(CNN_LSTM_SPACE, {
    "num_heads": {"type": "categorical", "choices": [1, 2, 4, 8]},
})

# =============================================================================
# 10. CNN + KAN
# =============================================================================
CNN_KAN_SPACE = _merge(_CNN_STRUCT, _KAN_STRUCT, _TRAINING)

# =============================================================================
# 11. CNN + KAN + Attention（约束：conv_channels % num_heads == 0）
# =============================================================================
CNN_KAN_ATT_SPACE = _merge(CNN_KAN_SPACE, {
    "num_heads": {"type": "categorical", "choices": [1, 2, 4, 8]},
})

# =============================================================================
# 12. Transformer（队友 Nanwang_Transformer；注意：
#     - 构造参数名是 nhead 不是 num_heads；dim_feedforward 在模型内固定为 d_model*4，不可调
#     - 约束：d_model % nhead == 0）
# =============================================================================
TRANSFORMER_SPACE = _merge({
    "d_model":    {"type": "categorical", "choices": [32, 64, 128]},
    "nhead":      {"type": "categorical", "choices": [2, 4, 8]},
    "num_layers": {"type": "int", "low": 1, "high": 4},
    "dropout":    {"type": "float", "low": 0.05, "high": 0.5},
}, _TRAINING)

# =============================================================================
# 汇总表：run_tuning.py 按模型名取字典（键名与 model_registry.REGISTRY 一致）
# =============================================================================
SEARCH_SPACES = {
    "xgboost":           XGBOOST_SPACE,
    "lstm":              LSTM_SPACE,
    "gru":               GRU_SPACE,
    "bilstm":            BILSTM_SPACE,
    "seq2seq":           SEQ2SEQ_SPACE,
    "kan":               KAN_SPACE,
    "kan_lstm":          KAN_LSTM_SPACE,
    "cnn_lstm":          CNN_LSTM_SPACE,
    "cnn_lstm_attention": CNN_LSTM_ATT_SPACE,
    "cnn_kan":           CNN_KAN_SPACE,
    "cnn_kan_attention": CNN_KAN_ATT_SPACE,
    "transformer":       TRANSFORMER_SPACE,
}


def check_space_consistency(space):
    """兼容旧入口的自检：检查整除类约束在 choices 组合下是否都成立。"""
    constraints = []
    if "d_model" in space and "nhead" in space:
        bad = [(dm, nh) for dm in space["d_model"]["choices"]
               for nh in space["nhead"]["choices"] if dm % nh != 0]
        if bad:
            constraints.append("警告: d_model/nhead 存在不能整除的组合: %s" % bad)
    return constraints


if __name__ == "__main__":
    # 单独运行本文件时打印全部空间，方便快速检查改动
    for name, space in SEARCH_SPACES.items():
        print("=" * 60)
        print("模型: %-20s (共 %d 个待寻优超参数)" % (name, len(space)))
        for k, v in space.items():
            print("  %-20s %s" % (k, v))
