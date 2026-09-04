from train import default_config, apply_tuning_result, run

model_type = "LoadTransformerEncoderDecoder"

config = default_config(model_type)

apply_tuning_result(
    config,
    f"optuna_results/{model_type}_A_best.json",
)

config.epochs = 100
config.device = "auto"
config.output = f"outputs/{model_type}_best_params.pt"

run(config)