# AI 冷站负荷预测

本项目面向南网能源“AI+冷站优化控制”赛题，使用冷机、水泵、冷却塔、环境气象和历史负荷等冷站运行数据，预测未来 24 小时逐时冷负荷 `TotalRealTimeLoad`。

当前主要任务是时序回归预测，不直接生成设备控制策略。

## 数据与模型流程

```text
项目 Excel（5 分钟数据）
→ 删除说明行、解析时间、处理异常值
→ 有限插值并聚合为小时数据
→ 按月份执行扩展窗口交叉验证
→ 在每个历史窗口内逐列 denoise
→ 对降噪结果执行三级 db4 小波分解
→ 生成降噪值、cA3、cD3、cD2、cD1 和时间特征
→ Transformer 同时接收未来 24 小时天气预报
→ 使用过去 168 小时预测未来 24 小时
```

默认输入维度为：

```text
22 个关键变量 × 5 个特征 + 6 个时间特征 = 116
```

`LoadTransformer` 和 `LoadTransformerEncoderDecoder` 额外接收未来天气：

```text
历史输入：(batch, 168, 116)
未来天气：(batch, 24, 8)
输出负荷：(batch, 24)
```

未来天气的 8 个特征为干球温度、湿球温度和 6 个时间正余弦编码。`LoadTransformerHistoryOnly` 和其他深度模型继续使用原 116 维单输入。

训练窗口每次滑动 1 小时；验证和测试窗口每次滑动 24 小时。默认依次执行“4～6月训练→7月验证、4～7月训练→8月验证、4～8月训练→9月验证”，再用 4～9 月训练并在 10 月测试。所有数据均在构造窗口前按时间切分。

## 项目结构

```text
AI 节能/
├── train.py                       深度模型统一训练入口
├── train_xgboost.py               XGBoost 独立训练和寻优入口
├── evaluate_test.py               独立测试集预测、指标和曲线输出
├── evaluate_test.ipynb            测试集评价 Notebook
├── AGENTS.md                      仓库协作与开发规范
├── README.md                      项目说明和运行指南
├── data_processing/
│   ├── data_processing.py         极端工况、小波分解和小波去噪
│   ├── requirements.txt           Python 依赖
│   ├── README_data_processing.md  数据处理接口说明
│   └── README_wavelet_core.md     小波算法说明
├── load_forecasting/
│   ├── data.py                    Excel 清洗、小时聚合、小波特征和窗口构造
│   ├── model_registry.py          模型、默认参数和搜索空间的唯一配置源
│   ├── LoadTransfromer.py         Transformer 模型（保持当前文件拼写）
│   ├── LoadTransformerHistoryOnly.py 仅历史输入 Transformer
│   ├── LoadTransformerEncoderDecoder.py Encoder-Decoder Transformer
│   ├── checkpoint.py              跨 PyTorch 版本加载可信检查点
│   ├── predict.py                 深度模型预测入口
│   ├── lstm_model.py              LSTM
│   ├── gru_model.py               GRU
│   ├── bilstm_model.py            BiLSTM
│   ├── lstm_seq2seq_model.py      Seq2Seq LSTM
│   ├── kan_model.py               KAN
│   ├── kan_lstm_model.py          KAN + LSTM
│   ├── cnn_lstm_model.py          CNN + LSTM
│   ├── cnn_lstm_attention_model.py CNN + LSTM + Attention
│   ├── cnn_kan_model.py           CNN + KAN
│   ├── cnn_kan_attention_model.py CNN + KAN + Attention
│   └── xgboost_model.py           XGBoost 多输出包装器
├── equipment_forecasting/
│   ├── targets.py                 28个设备目标、24维控制计划和运行掩码
│   ├── data.py                    设备预测窗口与控制计划CSV接口
│   ├── model.py                   共享编码器、多设备输出头Transformer
│   ├── metrics.py                 逐参数和设备分组指标
│   └── predict.py                 设备参数预测入口
├── train_2.py                     设备模型训练与XGBoost基线
├── search/
│   ├── tune_deep_models.py        13 个深度模型统一 Optuna 寻优入口
│   ├── data_pipeline.py           旧模型对比实验的数据管线
│   └── data_interface.py          XGBoost/序列数据接口
├── tests/                         单元测试和 CPU 冒烟测试
├── 附件3：训练数据集/             A/B/C 三个项目训练数据
└── AI----master/                  历史参考代码，主项目运行时不依赖
```

## 运行环境

推荐环境：

```text
Python >= 3.10
PyTorch >= 2.0
```

安装依赖：

```bash
python -m pip install -r data_processing/requirements.txt
```

检查环境：

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

在 AutoDL 上建议明确使用 Python 3.10 环境：

```bash
/root/miniconda3/envs/dl/bin/python train.py
```

## 支持的模型

模型名称、默认参数和搜索空间统一定义在 `load_forecasting/model_registry.py`。

```text
LoadTransformer
LoadTransformerHistoryOnly
LoadTransformerEncoderDecoder
lstm
gru
bilstm
seq2seq
kan
kan_lstm
cnn_lstm
cnn_lstm_attention
cnn_kan
cnn_kan_attention
xgboost
```

查看模型列表：

```bash
python -c "from load_forecasting.model_registry import SUPPORTED_MODEL_TYPES; print(SUPPORTED_MODEL_TYPES)"
```

增加或调整模型时，只修改 `MODEL_SPECS`，不要在训练或寻优文件中复制参数。

## 深度模型训练

### 默认滚动三折训练

```bash
python train.py
```

默认使用 `LoadTransformer` 和 `rolling_monthly_cv` 策略：

```text
Fold 1：4～6月训练 → 7月验证
Fold 2：4～7月训练 → 8月验证
Fold 3：4～8月训练 → 9月验证
→ 汇总三折平均非零MAPE
→ 取三折最佳epoch的中位数
→ 4～9月重新训练
→ 10月独立测试
```

训练窗口每次滑动 1 小时，验证和测试窗口每次滑动 24 小时。每个 Fold 都重新初始化模型，不能接续上一折权重。

指定模型训练：

```python
from train import SAMPLING_ROLLING_MONTHLY_CV, default_config, run

config = default_config(
    "LoadTransformerEncoderDecoder",
    SAMPLING_ROLLING_MONTHLY_CV,
)
config.epochs = 100
config.device = "auto"
run(config)
```

输出包括最终模型、10月测试预测和三折汇总：

```text
outputs/LoadTransformerEncoderDecoder.pt
outputs/LoadTransformerEncoderDecoder.csv
outputs/LoadTransformerEncoderDecoder_cv.json
```

### 使用全部数据训练最终模型

全量策略将 4～10 月全部窗口用于训练，没有验证集和测试集。模型训练固定轮数并保存最后一轮，适合超参数和训练轮数已经确定后的最终预测模型。

```python
import json
from train import (
    SAMPLING_ALL_TRAIN,
    apply_tuning_result,
    default_config,
    run,
)

model_type = "LoadTransformerEncoderDecoder"
result_path = f"optuna_results/{model_type}_rolling_cv_A_best.json"

config = default_config(model_type, SAMPLING_ALL_TRAIN)
apply_tuning_result(config, result_path)

with open(result_path, encoding="utf-8") as file:
    tuning_result = json.load(file)

config.epochs = tuning_result["best_epoch"]
config.device = "auto"
run(config)
```

默认输出：

```text
outputs/LoadTransformerEncoderDecoder_all_train.pt
```

该检查点使用 `selection_metric="final_epoch"`，不包含验证或测试指标。暂时没有寻优结果时，也可以手动设置模型参数、学习率和固定训练轮数。

```python
from train import SAMPLING_ALL_TRAIN, default_config, run

config = default_config("lstm", SAMPLING_ALL_TRAIN)
config.epochs = 100
config.model_config["hidden_size"] = 128
config.training_config["batch_size"] = 64
config.training_config["lr"] = 5e-4
run(config)
```

## 深度模型超参数搜索

统一入口：

```bash
python -m search.tune_deep_models --model 模型名
```

示例：

```bash
python -m search.tune_deep_models --model LoadTransformer
python -m search.tune_deep_models --model LoadTransformerHistoryOnly
python -m search.tune_deep_models --model LoadTransformerEncoderDecoder
python -m search.tune_deep_models --model lstm
```

默认设置：

```text
trial数量：20
每个Fold最多训练：30轮
每组参数训练：3个独立Fold模型
优化目标：三折平均非零MAPE最小
训练损失：MSE
剪枝器：MedianPruner
```

每个 Trial 的流程：

```text
同一组模型和训练参数
→ Fold 1训练并记录7月最佳MAPE和epoch
→ Fold 2重新初始化并记录8月结果
→ Fold 3重新初始化并记录9月结果
→ 返回三折平均MAPE
```

寻优结束后，程序取三折最佳epoch的中位数，使用最佳参数训练4～9月，并在10月测试一次。不会自动执行4～10月全量训练。

以 Encoder-Decoder Transformer 为例，输出为：

```text
optuna_results/LoadTransformerEncoderDecoder_rolling_cv_A.db
optuna_results/LoadTransformerEncoderDecoder_rolling_cv_A_best.json
outputs/LoadTransformerEncoderDecoder_tuned.pt
outputs/LoadTransformerEncoderDecoder_tuned.csv
```

最优参数JSON包含：

```text
model_config
training_config
fold_valid_mapes
fold_best_epochs
best_epoch（中位数）
best_value（三折平均非零MAPE）
```

SQLite数据库支持断点续跑。新的三折Study名称包含 `rolling_cv`，不会与旧单折数据库混用。由于每个 Trial 训练3次，耗时大约是原单折寻优的3倍。

## XGBoost 训练与寻优

XGBoost 使用与深度模型相同的 116 维小时窗口特征，然后展平：

```text
(N, 168, 116) → (N, 19488)
```

目标为 `(N, 24)`，一次预测未来 24 小时。

普通训练：

```bash
python train_xgboost.py
```

超参数寻优：

```bash
python train_xgboost.py --tune
```

输出：

```text
outputs/xgboost_model.joblib
optuna_results/xgboost_A.db
optuna_results/xgboost_A_best.json
```

## 预测

预测入口：

```bash
python -m load_forecasting.predict
```

运行前修改 `load_forecasting/predict.py` 中的配置：

```python
config = Namespace(
    data="最新数据.xlsx",
    weather="future_weather_24h.csv",
    checkpoint="outputs/LoadTransformer_weather.pt",
    output="prediction_24h.csv",
)
```

Transformer 的天气预报 CSV 必须正好包含连续 24 小时：

```csv
timeStamp,OutdoorTdbin,OutdoorWetTemp
2026-09-01 01:00:00,31.2,25.1
2026-09-01 02:00:00,30.8,24.9
```

第 1 条时间必须是历史运行数据最后有效时刻之后 1 小时。

输入 Excel 必须包含训练使用的关键列，并至少提供最近连续 168 小时数据。程序会自动：

```text
读取最新数据
→ 小时聚合
→ 查找最近连续 168 小时
→ denoise 和 decompose
→ 读取并校验未来 24 小时干球、湿球温度预报
→ 使用检查点中的训练统计量标准化
→ 根据检查点恢复模型
→ 输出未来 24 小时负荷
```

预测结果：

```text
prediction_24h.csv
```

共 24 行并包含 `timeStamp` 和 `TotalRealTimeLoad`，第 1 行表示最后数据时刻之后第 1 小时，第 24 行表示之后第 24 小时。

历史版及其他单输入模型不要求天气 CSV；预测入口会根据检查点中的模型类型自动判断，只需修改检查点路径：

```python
checkpoint="outputs/lstm.pt"
```

或使用寻优模型：

```python
checkpoint="outputs/lstm_tuned.pt"
```

## 独立测试集评价

测试集结构与训练集一致时，可以使用命令行脚本：

```bash
python evaluate_test.py \
  --checkpoint outputs/LoadTransformerEncoderDecoder_tuned.pt \
  --data "你的测试集.xlsx" \
  --output-dir outputs/test_evaluation \
  --device auto
```

也可以逐单元运行 `evaluate_test.ipynb`。测试窗口每次移动24小时，输出总体指标、h+1～h+24分步指标和真实/预测负荷曲线。

```text
outputs/test_evaluation/predictions.csv
outputs/test_evaluation/metrics.json
outputs/test_evaluation/metrics_by_horizon.csv
outputs/test_evaluation/load_curve.png
```

暂时用训练集代替测试集只能验证代码流程，不能代表模型对新数据的泛化性能。

## 设备运行参数预测

任务二使用：

```text
过去168小时历史状态
+ 未来24小时负荷、天气和已知控制计划
→ 未来24小时、28个设备运行参数
```

未来控制计划包含负荷、干球/湿球温度、时间编码、3台冷机启停和供水温度设定、冷冻水泵/冷却水泵/冷却塔频率。

训练：

```bash
python train_2.py
```

默认检查点：

```text
outputs/equipment_response.pt
```

预测前准备 `future_control_plan_24h.csv`，包含24条连续小时记录和以下基础列：

```text
timeStamp
TotalRealTimeLoad
OutdoorTdbin
OutdoorWetTemp
ChillerOn01/02/03
ChChWTempSupplySetPoint01/02/03
PriChWPVSDFreq01/02/03
CWPVSDFreq01/02/03
CTVSDFreq01/02/03
```

预测：

```bash
python -m equipment_forecasting.predict
```

输出：

```text
equipment_prediction_24h.csv
```

模型输出28个设备参数，并根据 `ChRealtimeEfficiencyKW` 派生 `COP01/02/03`。COP公式只有在效率字段单位确认为 `kW/RT` 时才具有物理意义。

设备模型按冷机、水泵、冷却塔和系统温差四组计算掩码 Huber Loss；关闭设备的效率和无意义温差不参与损失。模块同时提供逐目标 XGBoost 和 MultiOutput XGBoost 基线函数。

## 测试

运行全部测试：

```bash
python -m unittest discover -s tests -v
```

编译检查：

```bash
python -m compileall -q load_forecasting search train.py train_2.py train_xgboost.py evaluate_test.py tests
```

测试覆盖：

```text
数据清洗和小时聚合
denoise → decompose 顺序
小波重构
滑动窗口和时间切分
13 个深度模型前向、反向和检查点恢复
三折扩展窗口切分、平均MAPE和最佳epoch中位数
单一模型注册表和搜索空间
最佳非零 MAPE 检查点
PyTorch 新旧版本检查点加载
XGBoost 多输出训练、保存和恢复
```

## 输出文件

```text
outputs/*.pt              深度模型最佳检查点
outputs/*.csv             测试集真实值与预测值
outputs/*.joblib          XGBoost 模型
outputs/*_cv.json         三折验证和10月测试汇总
optuna_results/*.db       Optuna SQLite 试验记录
optuna_results/*_best.json 最佳超参数
prediction_24h.csv        未来 24 小时预测
```

这些生成文件默认不应提交到源码仓库。

## 常见问题

### Python 版本过低

出现：

```text
TypeError: unsupported operand type(s) for |: 'type' and 'type'
```

表示 Python 低于 3.10。请使用 Python 3.10 或更高版本。

### CUDA 可用但训练使用 CPU

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

显卡驱动支持的 CUDA 版本必须与 PyTorch wheel 兼容。

### PyTorch 2.6 检查点加载失败

项目通过 `load_forecasting/checkpoint.py` 显式加载自己生成的可信检查点：

```python
torch.load(..., weights_only=False)
```

并兼容不支持 `weights_only` 参数的旧版 PyTorch。

### 时间格式警告

如果 Pandas 提示无法推断时间格式，通常不会中断运行，但会降低解析速度。数据时间列应保持统一的：

```text
YYYY-MM-DD HH:MM:SS
```

## 注意事项

- `TotalRealTimeLoad` 的历史值会作为输入特征，未来 24 小时原始负荷作为监督目标。
- 累计量列不应直接作为预测输入。
- 降噪和小波分解必须限制在当前历史窗口内部，禁止对全量数据预处理后再切分。
- MAE 和 RMSE 使用全部时段；MAPE 只统计真实负荷非零时段。
- 测试集不得参与超参数选择，只在最终模型评估时使用。
- Transformer 天气融合层改变了模型结构，旧的 `LoadTransformer.pt` 权重不能直接用于新模型，必须重新训练。
