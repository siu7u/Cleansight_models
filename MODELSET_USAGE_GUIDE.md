# CleanSight 模型集使用指南

## 1. 项目定位

`Cleansight_models` 是 CleanSight 项目的模型资产与评测仓库，用于集中管理：

- YOLO 目标检测模型
- GRU / Causal TCN / Transformer 等时序模型
- 模型 checkpoint 与 registry
- 单模型 benchmark
- 端到端 3 分钟流程 benchmark
- 数据版本、YOLO 版本、特征映射版本、时序模型版本的对齐信息

后端 `CleanSightBackend` 负责在线推理、接口服务和端到端运行；模型集仓库负责训练、模型登记、离线评测和交付物管理。

## 2. 仓库结构

```text
Cleansight_models/
├── yolo-detection/                 # YOLO 检测模型训练与评估
├── temporal-gru/                   # GRU 时序模型
├── temporal-causal-tcn/            # Causal TCN 时序模型
├── temporal-transformer/           # Transformer 时序模型
├── benchmark/
│   ├── single_model/               # 单模型效果验证
│   ├── e2e_3min/                   # 3 分钟端到端流程验证
│   └── temporal_feed_mode/         # 整段喂 vs 流式喂评测
├── tools/                          # 通用评测/导出工具
├── registry/                       # 汇总后的模型权重交付目录
├── references/                     # 数据源、标注导出、引用材料
└── README.md
```

## 3. 环境准备

建议使用后端项目中的虚拟环境运行模型集脚本：

```bash
source ../CleanSightBackend/.venv/bin/activate

../CleanSightBackend/.venv/bin/python --version
```

检查 YOLO 依赖：

```bash
../CleanSightBackend/.venv/bin/python -c "import ultralytics; print('ultralytics ok')"
```

检查 TCN 依赖：

```bash
../CleanSightBackend/.venv/bin/python -c "import pytorch_tcn; print('pytorch_tcn ok')"
```

如果服务器没有图形界面，运行训练或可视化脚本时建议加：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
```

## 4. YOLO 检测模型使用

YOLO 模型位于：

```text
yolo-detection/
```

典型流程：

```bash
cd Cleansight_models/yolo-detection

python 00_status.py
python 02_build_dataset.py
python 03_train.py
python 04_validate.py
```

训练完成后，权重通常位于：

```text
runs/detect/.../weights/best.pt
runs/detect/.../weights/last.pt
```

可交付版本需要复制到统一的 registry 或 ModelScope 上传目录，并用子目录区分模型来源、训练组别和版本。

## 5. 时序模型训练

当前已有三个时序模型仓库：

```text
temporal-gru/
temporal-causal-tcn/
temporal-transformer/
```

### GRU

```bash
cd ../Cleansight_models/temporal-gru

PYTHONDONTWRITEBYTECODE=1 ../../CleanSightBackend/.venv/bin/python main.py \
  --mode full \
  --model gru \
  --epochs 10 \
  --window 64 \
  --verbose \
  --auto_save \
  --save_dir checkpoints/gru \
  --export_dir registry/gru-v1 \
  --visualize \
  --output_dir experiments/gru
```

### Causal TCN

```bash
cd ../Cleansight_models/temporal-causal-tcn

PYTHONDONTWRITEBYTECODE=1 ../../CleanSightBackend/.venv/bin/python main.py \
  --mode full \
  --model tcn \
  --epochs 10 \
  --window 64 \
  --verbose \
  --auto_save \
  --save_dir checkpoints/tcn \
  --export_dir registry/tcn-v1 \
  --visualize \
  --output_dir experiments/tcn
```

### Transformer

```bash
cd ../Cleansight_models/temporal-transformer

PYTHONDONTWRITEBYTECODE=1 ../../CleanSightBackend/.venv/bin/python main.py \
  --mode full \
  --model transformer \
  --epochs 10 \
  --window 64 \
  --verbose \
  --auto_save \
  --save_dir checkpoints/transformer \
  --export_dir registry/transformer-v1 \
  --visualize \
  --output_dir experiments/transformer
```

## 6. 单模型评测

时序模型详细评测示例：

```bash
cd ../Cleansight_models

MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 \
../CleanSightBackend/.venv/bin/python tools/eval_temporal_detailed.py \
  --repo temporal-gru \
  --model gru \
  --checkpoint registry/gru-v1/gru-final-**-**.pt
```

TCN 和 Transformer 只需要替换对应仓库、模型名和 checkpoint：

```text
temporal-causal-tcn / tcn / registry/tcn-v1/...
temporal-transformer / transformer / registry/transformer-v1/...
```

## 7. 整段喂 vs 流式喂 Benchmark

该 benchmark 用于确认同一时序模型在两种输入方式下的差异：

- 整段喂：一次输入完整序列 `[1, T, F]`
- 流式喂：每 tick 输入最近 `window` 帧 `[1, window, F]`，只取最后一帧预测

快速验收脚本链路：

```bash
cd ../Cleansight_models

../CleanSightBackend/.venv/bin/python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py \
  --device cpu \
  --max-videos 1 \
  --max-frames 256
```

正式全量评测：

```bash
../CleanSightBackend/.venv/bin/python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py --device auto
```

只跑单个模型：

```bash
../CleanSightBackend/.venv/bin/python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py --model gru --device auto
../CleanSightBackend/.venv/bin/python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py --model tcn --device auto
../CleanSightBackend/.venv/bin/python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py --model transformer --device auto
```

输出：

```text
benchmark/temporal_feed_mode/feed_mode_summary.md
benchmark/temporal_feed_mode/feed_mode_summary.json
```

`--max-videos` 和 `--max-frames` 只用于 smoke test。正式汇报或打榜时不要传这两个参数。

## 8. 端到端 3 分钟流程 Benchmark

端到端 benchmark 用于验证完整洗消流程，不只验证单个模型效果。

仅生成 case 报告：

```bash
cd ../Cleansight_models

../CleanSightBackend/.venv/bin/python benchmark/e2e_3min/run_e2e_benchmark.py \
  --case benchmark/e2e_3min/cases/clean_001.yaml
```

如果已有后端推理输出：

```bash
../CleanSightBackend/.venv/bin/python benchmark/e2e_3min/run_e2e_benchmark.py \
  --case benchmark/e2e_3min/cases/clean_001.yaml \
  --prediction benchmark/e2e_3min/outputs/clean_001.prediction.json
```

端到端在线推理结果需要由 `CleanSightBackend` 生成，模型集仓库本身只负责评测和报告。

## 9. 模型版本管理规范

每个可交付模型版本应至少包含：

```text
checkpoint.pt
CARD.md
pin.yaml
eval_report.md
```

字段含义：

- `checkpoint.pt`：模型权重
- `CARD.md`：模型说明、输入输出、门禁结果
- `pin.yaml`：钉定 dataset / yolo / feature_mapping / temporal model 版本
- `eval_report.md`：评测指标报告

推荐命名：

```text
gru-v1
tcn-v1
transformer-v1
yolo-v1
```

不得在同一个版本号背后静默替换数据集、特征映射或 checkpoint。

## 10. 接入 Backend 的边界

模型集仓库负责：

- 训练模型
- 管理 checkpoint
- 输出评估报告
- 维护模型版本说明
- 生成 benchmark 结果

`CleanSightBackend` 负责：

- 加载 YOLO detector
- 加载 temporal analyzer
- 在线视频流处理
- MediaMTX / RTSP 接入
- WebSocket / HTTP 推理接口
- 端到端 CLEAN 阶段验证

接入时需要确认：

```text
YOLO checkpoint 路径
时序 checkpoint 路径
feature_mapping 版本
input_dim
window size
类别 mapping
```

## 11. 当前状态与后续工作

当前模型集已经具备：

- YOLO 检测模型训练链路
- GRU / Causal TCN / Transformer 三类时序模型模板
- 单模型评测脚本
- 端到端 3 分钟流程 benchmark
- 整段喂与流式喂一致性 benchmark
- registry / CARD / pin 的版本管理雏形

后续重点：

- 使用新 YOLO 模型生成 `clean-v1` 的64维特征
- 统一离线训练特征与在线推理特征的 `step()` 实现
- 用新特征重训三个时序模型
- 在 Backend 中完成端到端在线推理验证
- 将最终 checkpoint、报告和 pin 信息上传到 ModelScope 并打 tag
