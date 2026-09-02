# AI 冷站负荷预测

本项目面向南网能源“AI+冷站优化控制”赛题，使用冷机、水泵、冷却塔、环境气象和历史负荷等冷站运行数据，预测未来 24 小时逐时冷负荷 `TotalRealTimeLoad`。

当前主要任务是时序回归预测，不直接生成设备控制策略。

## 数据与模型流程

```text
项目 Excel（5 分钟数据）
→ 删除说明行、解析时间、处理异常值
→ 有限插值并聚合为小时数据
→ 按时间切分训练/验证/测试集
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

`LoadTransformer` 额外接收未来天气：

```text
历史输入：(batch, 168, 116)
未来天气：(batch, 24, 8)
输出负荷：(batch, 24)
```

未来天气的 8 个特征为干球温度、湿球温度和 6 个时间正余弦编码。其他深度模型与 XGBoost 继续使用原 116 维单输入。

训练窗口每次滑动 1 小时；验证和测试窗口每次滑动 24 小时。数据在构造窗口前按时间切分，避免训练、验证和测试目标重叠。

## 项目结构

```text
AI 节能/
├── train.py                       深度模型统一训练入口
├── train_xgboost.py               XGBoost 独立训练和寻优入口
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
│   ├── train.py                   设备模型训练与XGBoost基线
│   └── predict.py                 设备参数预测入口
├── search/
│   ├── tune_deep_models.py        11 个深度模型统一 Optuna 寻优入口
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

### 训练默认 Transformer

```bash
python train.py
```

默认配置：

```text
模型：LoadTransformer
输入窗口：168 小时
预测范围：24 小时
训练轮数：1000
设备：auto（CUDA 可用时使用 GPU，否则使用 CPU）
优化器：AdamW
学习率：1e-3
批次：64
```

默认输出：

```text
outputs/LoadTransformer_weather.pt
outputs/LoadTransformer_weather.csv
```

### 训练其他深度模型

```bash
python -c "from train import default_config, run; run(default_config('lstm'))"
python -c "from train import default_config, run; run(default_config('gru'))"
python -c "from train import default_config, run; run(default_config('bilstm'))"
python -c "from train import default_config, run; run(default_config('seq2seq'))"
python -c "from train import default_config, run; run(default_config('kan'))"
python -c "from train import default_config, run; run(default_config('kan_lstm'))"
python -c "from train import default_config, run; run(default_config('cnn_lstm'))"
python -c "from train import default_config, run; run(default_config('cnn_lstm_attention'))"
python -c "from train import default_config, run; run(default_config('cnn_kan'))"
python -c "from train import default_config, run; run(default_config('cnn_kan_attention'))"
```

### 自定义训练参数

```python
from train import default_config, run

config = default_config("lstm")
config.epochs = 100
config.device = "auto"
config.training_config["batch_size"] = 32
config.training_config["lr"] = 5e-4
config.training_config["optimizer"] = "AdamW"
config.training_config["weight_decay"] = 1e-4

run(config)
```

训练损失使用 MSE。每轮同时计算验证 MSE 和原始单位的非零 MAPE，只有验证集非零 MAPE 改善时才覆盖保存最佳检查点。

检查点记录：

```text
模型类型和模型参数
训练参数
最佳 epoch
最佳验证 MAPE
验证 MSE
标准化均值和标准差
特征名称和小波列
输入窗口与预测长度
```

## 深度模型超参数搜索

统一入口：

```bash
python -m search.tune_deep_models --model 模型名
```

示例：

```bash
python -m search.tune_deep_models --model LoadTransformer
python -m search.tune_deep_models --model lstm
python -m search.tune_deep_models --model cnn_lstm_attention
python -m search.tune_deep_models --model cnn_kan_attention
```

默认寻优设置：

```text
trial 数量：20
每个 trial 最多：30 轮
输入窗口：固定 168 小时
目标：验证集非零 MAPE 最小
剪枝器：MedianPruner
```

寻优时数据处理和小波特征只计算一次，所有 trial 复用相同训练/验证数组。测试集只在寻优结束后的最终训练中使用。

输出示例：

```text
optuna_results/lstm_A.db
optuna_results/lstm_A_best.json
outputs/lstm_tuned.pt
outputs/lstm_tuned.csv
```

SQLite 数据库支持断点续跑。再次运行相同模型命令会继续使用原 study，并新增 trial。

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

其他深度模型仍使用单输入，不要求天气 CSV；预测入口会根据检查点中的模型类型自动判断，只需修改检查点路径：

```python
checkpoint="outputs/lstm.pt"
```

或使用寻优模型：

```python
checkpoint="outputs/lstm_tuned.pt"
```

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
python -m equipment_forecasting.train
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
python -m compileall -q load_forecasting search train.py train_xgboost.py tests
```

测试覆盖：

```text
数据清洗和小时聚合
denoise → decompose 顺序
小波重构
滑动窗口和时间切分
11 个深度模型前向、反向和检查点恢复
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
