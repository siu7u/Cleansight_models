# CleanSight 模型集使用指南

本文位于 `docs/`，给出当前统一训练评估链路的实际操作；命令均从 `Cleansight_models/`
仓库根目录执行。

## 1. 选择正确入口

| 需求 | 推荐入口 |
|---|---|
| 训练一个 YOLO/GRU/MS-TCN/Transformer | `python -m framework.cleansight_eval.cli.train` |
| 评估单个 checkpoint | `python -m benchmark.cli.eval` |
| 汇总多个模型 | `python -m benchmark.cli.matrix` |
| 校验固定测试集 | `python tools/validate_testsets.py` |
| 全序列与流式一致性 | `benchmark/temporal_feed_mode/` |
| 3 分钟业务场景 | `benchmark/e2e_3min/` |

各 `temporal-*` 和 `yolo-detection/pipeline/` 独立脚本仅用于历史资产复现，不再经过集中 manager；
它们不是新实验的默认入口。

## 2. 环境准备

```bash
source ../CleanSightBackend/.venv/bin/activate
pip install -r framework/requirements.txt
python -c "import torch, yaml; print(torch.__version__)"
```

运行 YOLO 前再确认：

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

服务器无显示环境时：

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib
```

## 3. 数据与配置

实验配置位于 `framework/experiments/*.yaml`。已登记数据集通过稳定引用接入：

```yaml
data:
  dataset_ref: temporal.actionmixed-v2
  split_train: train
  split_val: val
  split_eval: test
```

根目录、版本、类别、feature mapping 和 manifest 由 `benchmark/testsets.yaml` 解析。只有未登记的
临时/合成数据才直接使用 `data.root`；相对路径仍以 YAML 所在目录为基准。

评估前执行：

```bash
python tools/validate_testsets.py --catalog benchmark/testsets.yaml --json
```

`ok: false` 表示数据完整性或 split 泄漏校验未通过。数据组修复前可以进行 exploratory 调试，但不能
把该结果作为 locked holdout benchmark。

## 4. 配置选择

| 模型 | 配置 | 流水线 |
|---|---|---|
| YOLO large | `framework/experiments/yolo-clean-large.yaml` | `detection` |
| YOLO small | `framework/experiments/yolo-clean-small.yaml` | `detection` |
| GRU | `framework/experiments/gru-actionmixed.yaml` | `sliding_window_temporal` |
| MS-TCN | `framework/experiments/mstcn-actionmixed.yaml` | `full_sequence_temporal` |
| MS-TCN++ | `framework/experiments/mstcn2-actionmixed.yaml` | `full_sequence_temporal` |
| Transformer | `framework/experiments/transformer-actionmixed.yaml` | `full_sequence_temporal` |

复制最接近的配置建立新实验。`schema_version: 1` 必填；未知顶层字段、未知 section 字段和未知
`-S` 路径会直接报错，避免拼写错误被静默忽略。

## 5. 训练

### 5.1 YOLO

```bash
python -m framework.cleansight_eval.cli.train \
  --config framework/experiments/yolo-clean-large.yaml
```

YOLO 使用 `train.batch`：

```bash
python -m framework.cleansight_eval.cli.train \
  --config framework/experiments/yolo-clean-large.yaml \
  -S train.epochs=50 -S train.batch=8
```

### 5.2 GRU 因果滑窗

```bash
python -m framework.cleansight_eval.cli.train \
  --config framework/experiments/gru-actionmixed.yaml
```

滑窗输入为 `[B, window, F]`，末帧监督；ActionMixed 当前 `F=40`，配置中的 `window` 必须不大于
最短视频可用帧数。

### 5.3 MS-TCN / MS-TCN++ / Transformer 全序列

```bash
python -m framework.cleansight_eval.cli.train \
  --config framework/experiments/mstcn-actionmixed.yaml

python -m framework.cleansight_eval.cli.train \
  --config framework/experiments/mstcn2-actionmixed.yaml

python -m framework.cleansight_eval.cli.train \
  --config framework/experiments/transformer-actionmixed.yaml
```

全序列输入为 `[1, T, F]`，逐帧监督。当前 MS-TCN 和 Transformer 是非因果模型，只能用于
`full_sequence_temporal`，不能直接作为在线滑窗模型。

### 5.4 Resume

时序训练支持完整恢复 optimizer、epoch 和 best metric：

```bash
python -m framework.cleansight_eval.cli.train \
  --config framework/experiments/gru-actionmixed.yaml \
  --resume runs/<run>/checkpoints/last.pt \
  -S train.epochs=50
```

`train.epochs` 表示最终目标 epoch，不是“额外再训练多少轮”。当前 framework 的 `--resume` 主要
用于时序训练；已登记数据集会校验 checkpoint 中的 dataset version/revision、feature mapping、
labels 和 train split fingerprint，任一关键身份漂移都会拒绝继续训练。YOLO 是否恢复应使用
Ultralytics 对应训练语义，不要假设两者完全相同。

## 6. 训练产物

```text
runs/<model>-<timestamp>/
├── config.resolved.json
├── env.json
├── status.json
├── history.csv                 # 时序；逐 epoch
├── training_curves.png         # 时序；自动生成
└── checkpoints/
    ├── best.pt
    ├── best.pt.meta.json
    ├── last.pt
    └── last.pt.meta.json
```

YOLO 的 best/last 位于 `checkpoints/<data.name>/weights/`，训练曲线使用 Ultralytics 生成的
`results.png`。`env.json` 仅用于训练排障，不写入精简评估 JSON。

## 7. 评估

### 7.1 YOLO

```bash
python -m benchmark.cli.eval \
  --config framework/experiments/yolo-clean-large.yaml \
  --ckpt runs/<run>/checkpoints/group1_large/weights/best.pt
```

评估参数来自 YAML：`conf`、`iou`、`imgsz`、`eval_split`、`max_det` 和 `agnostic_nms`，实际生效值
会写入 `metrics.details.effective_parameters`。

### 7.2 时序模型

```bash
python -m benchmark.cli.eval \
  --config framework/experiments/gru-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/best.pt

python -m benchmark.cli.eval \
  --config framework/experiments/transformer-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/best.pt
```

评估始终保持逐视频边界：Accuracy 按帧 micro，Edit 按视频 macro mean，分段 F1/P/R 使用各视频
独立匹配后的跨视频 TP/FP/FN micro 聚合。

## 8. Formal 与外部 checkpoint

### Formal

适用于正式归档，要求：

- `data.dataset_ref + split_eval` 能唯一推导已登记 testset，且数据门禁通过；
- checkpoint sidecar 使用当前 metadata schema，并与权重 SHA-256 绑定；
- 配置与 checkpoint 的模型类型、维度、类别数兼容；
- 保存 prediction artifact。

### Exploratory

适用于组员提供的外部 `.pt` 或临时数据。外部 YOLO 使用：

```yaml
model:
  type: yolo
  allow_missing_meta: true
evaluation:
  mode: exploratory
```

这不会伪造 metadata；报告会保留外部导入/降级事实。正式发布前仍需补齐可信 sidecar 和固定 testset。

外部时序权重也使用同一个开关，但 YAML 必须完整声明模型重建结构和特征语义：

```yaml
model:
  type: gru
  input_dim: 40
  num_classes: 6
  hidden: 128
  num_layers: 3
  allow_missing_meta: true
feature_schema:
  dim: 40
  version: actionmixed-bbox-8cls-v1
evaluation:
  mode: exploratory
```

当前接受裸 state dict、`model_state` 或 `state_dict` 包装。仅在 sidecar 缺失时使用 YAML fallback；
如果 sidecar 已存在但摘要损坏，不会绕过校验。模型参数键和张量形状仍以 `strict=True` 完整匹配，
结果标记 `checkpoint_metadata_bound=false`，不能作为 formal benchmark。

## 9. 评估输出与报告

一次 eval 会生成：

| 产物 | 作用 |
|---|---|
| `evals/*.evaluation.json` | 机器可读 EvaluationResult v2 |
| `artifacts/*.predictions.json` | YOLO 逐图或时序逐视频预测 |
| `checkpoints/<ckpt>.eval.md` | 当前 checkpoint 的人读报告 |
| `checkpoints/EVALUATION_REPORT.md` | 按时序/YOLO 分类追加的版本报告 |
| `evals/*.delivery.manifest.json` | 交付文件路径、大小、SHA-256、Schema |
| `viz/segmentation-<split>-pNN.png` | 滑窗与全序列时序的测试 GT/Prediction timeline；路径和 SHA-256 写入评估 artifact |

报告不包含完整环境和 Git 信息，也不自动填写发布结论。人工维护区用于评估后记录主要问题、是否进入
版本以及下一步动作。

## 10. 汇总矩阵

```bash
python -m benchmark.cli.matrix --runs runs
python -m benchmark.cli.matrix --runs runs --pipeline full_sequence_temporal
```

输出 `matrix.json` 和 `matrix.md`。不同任务指标取并集，`N/A`、`MISSING` 和未产出保持不同语义。

## 11. 专项 Benchmark

### 11.1 全序列与流式一致性

```bash
python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py \
  --device cpu --max-videos 1 --max-frames 256
```

带 `--max-videos/--max-frames` 的结果只是 smoke test。正式运行时移除限制，并记录 checkpoint、
feature mapping、device 和推理模式。

### 11.2 3 分钟端到端

```bash
python benchmark/e2e_3min/run_e2e_benchmark.py \
  --case benchmark/e2e_3min/cases/clean_001.yaml \
  --prediction benchmark/e2e_3min/outputs/clean_001.prediction.json
```

prediction 应由 `CleanSightBackend` 导出。本仓库只评分，不负责在线推理和生产动作。

端到端时间线与单时序模型评估共用 `benchmark/core/metrics.py`：按动作名和 Temporal IoU 做全局
贪心一对一匹配，在 0.1/0.25/0.5 阈值下输出 TP/FP/FN、Precision、Recall、F1、匹配段平均 IoU
和边界 MAE。端到端仍额外使用 `allowed_time_error_sec`、流程结果和必需动作存在性生成业务 PASS/FAIL。

## 12. Schema 与交付

- `schemas/evaluation-result-v2.schema.json`：评估结果结构；
- `schemas/prediction-artifact-v1.schema.json`：预测 artifact 结构；
- `schemas/delivery-manifest-v1.schema.json`：稳定交付文件清单。

Schema 是外部契约，不是指标实现。指标真源在 `benchmark/core/metrics.py` 和
`benchmark/evaluators/`，运行时校验在对应 Python `validate_*` 函数。

## 13. 常见问题

### `FileNotFoundError: /abs/path/to/.../data.yaml`

配置仍使用占位路径。把 `data.data_yaml` 改成相对当前 YAML 的路径或真实绝对路径。

### `git add framework` 找不到目录

如果当前目录已经是 `Cleansight_models/framework`，使用 `git add .`；若位于仓库根目录，使用
`git add framework/`。提交前先运行 `git status --short`，避免把 `runs/` 和权重加入 Git。

### formal 评估因 testset validation 失败

先运行 `tools/validate_testsets.py` 查看具体泄漏或缺失项。不要关闭校验伪装正式结果；需要临时调试时，
复制一份实验 YAML 并显式设为 `evaluation.mode: exploratory`。

### Transformer 的 nested tensor warning

`norm_first=True` 会让 PyTorch 不采用 nested-tensor 快速路径。这是性能提示，不影响当前普通 padded/
定长张量前向的数值正确性；是否调整结构应通过同一 checkpoint 契约下的基准验证，而不是只为消除 warning。

## 14. Git 与发布边界

不要提交权重、`runs/`、视频、原始数据或本地密钥。ModelScope 上传、模型注册和上线判断由外部模型
管理流程或人工负责；本仓库提供 checkpoint、CARD、pin、评估事实和交付 manifest。
