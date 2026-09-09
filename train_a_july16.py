"""4月、7月、9月数据训练入口，7月16日作为隔离验证日。"""
from train import SAMPLING_JULY16_HOLDOUT, default_config, run


JULY16_DATA = (
    "附件4：测试数据集/测试数据项目A历史数据_2025年4月、7月、9月.xlsx"
)
WAVELET_COLUMNS_JULY16 = [
    "TotalRealTimeLoad",
    "OutdoorTdbin",
    "OutdoorWetTemp",
    "PriChWFlow",
    "PriChWDP01",
    "PriChWTempSupply",
    "PriChWTempReturn",
    "CWFlow01",
    "CWTempSupply",
    "CWTempReturn",
    *[f"ChPower0{i}" for i in range(1, 5)],
    *[f"PriChWPPower0{i}" for i in range(1, 7)],
    *[f"CWPPower0{i}" for i in range(1, 7)],
    *[f"CTPower0{i}" for i in range(1, 5)],
]


def default_config_july16(model_type="LoadTransformerEncoderDecoder"):
    """创建7月16日隔离验证训练配置。"""
    config = default_config(model_type, SAMPLING_JULY16_HOLDOUT)
    prefix = f"outputs/TestA_July16_{model_type}"
    config.data = JULY16_DATA
    config.wavelet_columns = list(WAVELET_COLUMNS_JULY16)
    config.output = prefix + ".pt"
    config.validation_output = prefix + "_validation.csv"
    return config


if __name__ == "__main__":
    run(default_config_july16())
