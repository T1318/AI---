from copy import deepcopy
from importlib import import_module


DEFAULT_TRAINING_CONFIG = {
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 64,
    "optimizer": "AdamW",
}

TRAINING_SEARCH_SPACE = {
    "lr": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "optimizer": {"type": "categorical", "choices": ["Adam", "AdamW"]},
}

_RNN_SEARCH = {
    "hidden_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
    "num_layers": {"type": "int", "low": 1, "high": 3},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
}
_CNN_SEARCH = {
    "conv_channels": {"type": "categorical", "choices": [16, 32, 64]},
    "kernel_size": {"type": "categorical", "choices": [3, 5, 7]},
    "num_conv_layers": {"type": "int", "low": 1, "high": 3},
}
_KAN_SEARCH = {
    "kan_depth": {"type": "int", "low": 1, "high": 3},
    "kan_width": {"type": "categorical", "choices": [8, 16, 32, 64]},
    "grid_size": {"type": "categorical", "choices": [3, 5, 7]},
    "spline_order": {"type": "categorical", "choices": [2, 3]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
}


def _merge(*parts):
    result = {}
    for part in parts:
        result.update(deepcopy(part))
    return result


def _factory(module_name, class_name):
    def build(input_dim, config):
        module = import_module(f".{module_name}", __package__)
        cls = getattr(module, class_name)
        return cls(input_size=input_dim, output_size=1, **config)
    return build


def _transformer_factory(input_dim, config):
    from .LoadTransfromer import LoadTransformer
    return LoadTransformer(input_dim=input_dim, **config)


def _xgboost_factory(input_dim, config):
    from .xgboost_model import XGBoostModel
    return XGBoostModel(**config)


def _identity(params):
    return dict(params)


def _kan_adapter(params):
    return {
        "hidden_sizes": [params["kan_width"]] * params["kan_depth"],
        "grid_size": params["grid_size"],
        "spline_order": params["spline_order"],
        "dropout": params["dropout"],
    }


def _kan_lstm_adapter(params):
    return {
        "kan_hidden_sizes": [params["kan_width"]] * params["kan_depth"],
        "kan_output_size": params["kan_output_size"],
        "lstm_hidden_size": params["lstm_hidden_size"],
        "lstm_num_layers": params["lstm_num_layers"],
        "grid_size": params["grid_size"],
        "spline_order": params["spline_order"],
        "dropout": params["dropout"],
    }


def _cnn_kan_adapter(params):
    result = {
        "conv_channels": params["conv_channels"],
        "kernel_size": params["kernel_size"],
        "num_conv_layers": params["num_conv_layers"],
        "kan_hidden_sizes": [params["kan_width"]] * params["kan_depth"],
        "grid_size": params["grid_size"],
        "spline_order": params["spline_order"],
        "dropout": params["dropout"],
    }
    if "num_heads" in params:
        result["num_heads"] = params["num_heads"]
    return result


MODEL_SPECS = {
    "LoadTransformer": {
        "kind": "deep", 
        "factory": _transformer_factory,
        "defaults": {
            "d_model": 64, 
            "nhead": 4, 
            "num_layers": 2, 
            "dropout": 0.1
            },
        "search_space": {
            "d_model": {"type": "categorical", "choices": [32, 64, 128]},
            "nhead": {"type": "categorical", "choices": [2, 4, 8]},
            "num_layers": {"type": "int", "low": 1, "high": 4},
            "dropout": {"type": "float", "low": 0.05, "high": 0.5},
            }, 
        "config_adapter": _identity,
    },

    "lstm": {
        "kind": "deep", 
        "factory": _factory("lstm_model", "LSTMModel"), 
        "defaults": {
            "hidden_size": 64, 
            "num_layers": 2, 
            "dropout": 0.1
        }, 
        "search_space": deepcopy(_RNN_SEARCH), 
        "config_adapter": _identity
        },

    "gru": {
        "kind": "deep", 
        "factory": _factory("gru_model", "GRUModel"), 
        "defaults": {
            "hidden_size": 64, 
            "num_layers": 2, 
            "dropout": 0.1}, 
        "search_space": deepcopy(_RNN_SEARCH), 
        "config_adapter": _identity
    },

    "bilstm": {
        "kind": "deep", 
        "factory": _factory("bilstm_model", "BiLSTMModel"), 
        "defaults": {
            "hidden_size": 64, 
            "num_layers": 2, 
            "dropout": 0.1
        }, 
        "search_space": deepcopy(_RNN_SEARCH), 
        "config_adapter": _identity
    },

    "seq2seq": {
        "kind": "deep", 
        "factory": _factory("lstm_seq2seq_model", "Seq2SeqLSTM"),
        "defaults": {
            "encoder_hidden_size": 64, 
            "decoder_hidden_size": 64, 
            "encoder_num_layers": 2, 
            "decoder_num_layers": 2, 
            "dropout": 0.1},
        "search_space": {
            "encoder_hidden_size": {"type": "categorical", "choices": [32, 64, 128, 256]}, 
            "decoder_hidden_size": {"type": "categorical", "choices": [32, 64, 128, 256]}, 
            "encoder_num_layers": {"type": "int", "low": 1, "high": 3}, 
            "decoder_num_layers": {"type": "int", "low": 1, "high": 3}, 
            "dropout": {"type": "float", "low": 0.0, "high": 0.5}},
        "config_adapter": _identity,
    },

    "kan": {
        "kind": "deep", 
        "factory": _factory("kan_model", "KANModel"), 
        "defaults": {
            "hidden_sizes": [64, 64], 
            "grid_size": 5, 
            "spline_order": 3, 
            "dropout": 0.1}, 
            "search_space": deepcopy(_KAN_SEARCH), 
            "config_adapter": _kan_adapter
            },

    "kan_lstm": {
        "kind": "deep", 
        "factory": _factory("kan_lstm_model", "KanLstmModel"),
        "defaults": {
            "kan_hidden_sizes": [64], 
            "kan_output_size": 64, 
            "lstm_hidden_size": 64, 
            "lstm_num_layers": 2, 
            "grid_size": 5, 
            "spline_order": 3, 
            "dropout": 0.1
            },
        "search_space": _merge(_KAN_SEARCH, {
            "kan_output_size": {"type": "categorical", "choices": [8, 16, 32, 64]}, 
            "lstm_hidden_size": {"type": "categorical", "choices": [32, 64, 128, 256]}, 
            "lstm_num_layers": {"type": "int", "low": 1, "high": 3}}),
        "config_adapter": _kan_lstm_adapter,
    },

    "cnn_lstm": {
        "kind": "deep", 
        "factory": _factory("cnn_lstm_model", "CnnLstmModel"),
        "defaults": {
            "conv_channels": 32, 
            "kernel_size": 3, 
            "lstm_hidden_size": 64, 
            "num_conv_layers": 2, 
            "lstm_num_layers": 2, 
            "dropout": 0.1
            },
        "search_space": _merge(_CNN_SEARCH, {
            "lstm_hidden_size": {"type": "categorical", "choices": [32, 64, 128, 256]}, 
            "lstm_num_layers": {"type": "int", "low": 1, "high": 3}, 
            "dropout": {"type": "float", "low": 0.0, "high": 0.5}}),
        "config_adapter": _identity,
    },

    "cnn_lstm_attention": {
        "kind": "deep", 
        "factory": _factory("cnn_lstm_attention_model", "CnnLstmAttentionModel"),
        "defaults": {
            "conv_channels": 32, 
            "kernel_size": 3, 
            "num_conv_layers": 2, 
            "lstm_hidden_size": 64, 
            "lstm_num_layers": 2, 
            "num_heads": 4, 
            "dropout": 0.1
            },
        "search_space": _merge(_CNN_SEARCH, {
            "lstm_hidden_size": {"type": "categorical", "choices": [32, 64, 128, 256]}, 
            "lstm_num_layers": {"type": "int", "low": 1, "high": 3}, 
            "num_heads": {"type": "categorical", "choices": [1, 2, 4, 8]}, 
            "dropout": {"type": "float", "low": 0.0, "high": 0.5}}),
        "config_adapter": _identity,
    },
    "cnn_kan": {
        "kind": "deep", 
        "factory": _factory("cnn_kan_model", "CnnKanModel"), 
        "defaults": {
            "conv_channels": 32, 
            "kernel_size": 3, 
            "num_conv_layers": 2, 
            "kan_hidden_sizes": [64], 
            "grid_size": 5, 
            "spline_order": 3, 
            "dropout": 0.1
            }, 
        "search_space": _merge(_CNN_SEARCH, _KAN_SEARCH), 
        "config_adapter": _cnn_kan_adapter
        },

    "cnn_kan_attention": {
        "kind": "deep", 
        "factory": _factory("cnn_kan_attention_model", "CnnKanAttentionModel"), 
        "defaults": {
            "conv_channels": 32, 
            "kernel_size": 3, 
            "num_conv_layers": 2, 
            "kan_hidden_sizes": [64], 
            "num_heads": 4, 
            "grid_size": 5, 
            "spline_order": 3, 
            "dropout": 0.1
            }, 
        "search_space": _merge(_CNN_SEARCH, _KAN_SEARCH, {
            "num_heads": {"type": "categorical", "choices": [1, 2, 4, 8]}}), 
            "config_adapter": _cnn_kan_adapter},

    "xgboost": {
        "kind": "xgboost", "factory": _xgboost_factory,
        "defaults": {
            "n_estimators": 300, 
            "max_depth": 6, 
            "n_jobs": -1},
        "search_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 1000}, 
            "max_depth": {"type": "int", "low": 3, "high": 10}, 
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True}, 
            "subsample": {"type": "float", "low": 0.6, "high": 1.0}, 
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0}, 
            "min_child_weight": {"type": "int", "low": 1, "high": 10}, 
            "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True}},
        "config_adapter": _identity,
    },
}

SUPPORTED_MODEL_TYPES = tuple(MODEL_SPECS)
SUPPORTED_DEEP_MODEL_TYPES = tuple(name for name, spec in MODEL_SPECS.items() if spec["kind"] == "deep")


def get_model_spec(model_type):
    if model_type not in MODEL_SPECS:
        raise ValueError(f"未知模型: {model_type}")
    return MODEL_SPECS[model_type]


def default_model_config(model_type, horizon=24):
    config = deepcopy(get_model_spec(model_type)["defaults"])
    if get_model_spec(model_type)["kind"] == "deep":
        config["horizon"] = horizon
    return config


def build_model(model_type, input_dim, model_config):
    return get_model_spec(model_type)["factory"](input_dim, dict(model_config))


def _suggest(trial, name, spec):
    if spec["type"] == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if spec["type"] == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))


def sample_model_configs(trial, model_type, horizon=24):
    spec = get_model_spec(model_type)
    sampled = {name: _suggest(trial, name, value) for name, value in spec["search_space"].items()}
    model_config = spec["config_adapter"](sampled)
    if spec["kind"] == "deep":
        model_config["horizon"] = horizon
        training_config = {name: _suggest(trial, name, value) for name, value in TRAINING_SEARCH_SPACE.items()}
    else:
        training_config = {}
    return model_config, training_config
