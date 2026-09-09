"""项目B负荷预测训练入口，仅覆盖数据和历史输入列。"""
from load_forecasting.data import WAVELET_COLUMNS
from train import (
    SAMPLING_ALL_TRAIN,
    SAMPLING_ROLLING_MONTHLY_CV,
    default_config,
    run,
)


PROJECT_B_DATA = (
    "附件3：训练数据集/训练数据项目B历史数据_2023-04-01_2023-10-31.xlsx"
)
WAVELET_COLUMNS_B = [
    column
    for column in WAVELET_COLUMNS
    if column not in {"PriChWDiffPress", "ChPower03"}
]


def default_config_b(
    model_type="LoadTransformer",
    sampling_strategy=SAMPLING_ROLLING_MONTHLY_CV,
):
    """创建与A版训练逻辑一致的项目B配置。"""
    config = default_config(model_type, sampling_strategy)
    config.data = PROJECT_B_DATA
    config.wavelet_columns = list(WAVELET_COLUMNS_B)
    suffix = "_all_train" if sampling_strategy == SAMPLING_ALL_TRAIN else ""
    config.output = f"outputs/B_{model_type}{suffix}.pt"
    return config


if __name__ == "__main__":
    run(default_config_b())
