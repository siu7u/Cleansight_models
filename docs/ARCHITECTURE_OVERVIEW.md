# 仓库架构简述

本仓库负责模型训练、checkpoint 评测、固定 benchmark 和交付产物；线上视频流、业务告警和真实
端到端推理由相邻的 CleanSightBackend 负责。

## 1. 整体结构

```text
Cleansight_models/
├── framework/              # 统一训练、预测和评测入口
├── benchmark/              # 数据集身份、指标和结果格式真源
├── external_checkpoints/   # 外部 checkpoint 的配套 YAML
├── yolo-detection/         # YOLO 数据构建、旧流水线和 registry
├── temporal-*/             # 历史时序模型资产与复现脚本
├── schemas/                # 对外 JSON Schema
├── tools/                  # 校验和历史兼容工具
├── usage/                  # YAML 与命令行教程
└── docs/                   # 设计、接入和评测文档
```

新开发的主路径是：

```text
framework + benchmark + schemas
```

`temporal-*` 和 `yolo-detection/pipeline` 主要用于保存与复现历史模型，不再通过单独的
model manager 转发。

## 2. Framework：负责运行模型

[`framework/cleansight_eval`](../framework/cleansight_eval/) 提供统一 CLI，并分成三层：

```text
cli/        读取 YAML，选择 Pipeline，组织完整命令
core/       配置、checkpoint、run、报告、完整性和矩阵
temporal/   时序数据、特征、模型、训练、预测和 timeline
detection/  YOLO 训练、预测和检测产物
```

当前有三条 Pipeline：

| Pipeline | 输入与用途 |
|---|---|
| `detection` | 单帧图像输入，运行 YOLO 检测。 |
| `sliding_window_temporal` | `[B, window, F]` 因果窗口输入，面向实时/流式时序模型。 |
| `full_sequence_temporal` | `[B, T, F]` 完整序列输入，面向离线时序模型。 |

Pipeline 只负责训练和预测，输出统一的 `PredictionOutput`，不在内部定义正式指标。

## 3. Benchmark：负责数据身份与评判

[`benchmark`](../benchmark/) 负责：

- 在 [`testsets.yaml`](../benchmark/testsets.yaml) 登记数据集版本、split、manifest 和重叠策略；
- 计算 Accuracy、Edit、F1、Precision、Recall、Temporal IoU 和 YOLO 指标；
- 生成统一的 `EvaluationResult`、prediction artifact 和 delivery manifest；
- 区分单模型、feed-mode 与端到端三分钟评测。

正式评测优先通过 `dataset_ref` 引用已登记数据。直接填写本地 `data.root` 的配置只能作为
`exploratory` 使用，不能冒充锁定 testset 的正式结果。

## 4. YAML：负责组合，不负责实现

实验 YAML 选择：

- 使用哪条 Pipeline；
- 使用哪个已注册模型及其网络参数；
- 数据集与 train/val/test split；
- feature mapping、输入维度、类别顺序和 normalization；
- 训练超参数与评测模式。

YAML 不能实现一个未知网络，也不能创造不存在的特征提取方式。新增时序架构通常需要：

1. 在 `framework/cleansight_eval/temporal/models/` 增加模型类；
2. 在模型注册表中登记；
3. 增加一份实验 YAML。

所有 YAML 的位置和作用统一记录在 [`usage/YAML_CONFIG.md`](../usage/YAML_CONFIG.md)。

## 5. 外部 checkpoint

[`external_checkpoints`](../external_checkpoints/) 采用下面的配套结构：

```text
external_checkpoints/<model-id>/
├── <model-id>.pt      # 本地权重，Git 忽略
└── <model-id>.yaml    # 模型与输入契约，Git 跟踪
```

评测外部权重时，YAML 中的模型类型、tensor shape、feature version、类别顺序、normalization 和
窗口必须与训练时一致。没有可信 metadata 的权重只能设置为 `exploratory`，加载时仍会严格检查
参数键和 tensor shape。

组员可以从 [`external-temporal-template.yaml`](../external_checkpoints/external-temporal-template.yaml)
复制配置。

## 6. 训练和评测流程

训练流程：

```text
实验 YAML
  → CLI 选择 Pipeline
  → 加载 train/val 数据与特征
  → 构造并训练模型
  → 保存 checkpoint + metadata + history
```

评测流程：

```text
实验 YAML + checkpoint
  → Pipeline 加载并预测
  → PredictionOutput
  → benchmark 按统一口径计算指标
  → evaluation.json + predictions.json + report + timeline
```

常用入口：

```bash
python -m framework.cleansight_eval.cli.train --config <experiment.yaml>

python -m framework.cleansight_eval.cli.eval \
  --config <experiment.yaml> \
  --ckpt <checkpoint.pt>

python -m framework.cleansight_eval.cli.matrix --runs runs
```

## 7. 修改内容时应该去哪里

| 需求 | 主要修改位置 |
|---|---|
| 调整实验超参数 | `framework/experiments/*.yaml` |
| 接入外部 checkpoint | `external_checkpoints/<model-id>/*.yaml` |
| 新增时序网络 | `framework/cleansight_eval/temporal/models/` |
| 修改时序特征 | `framework/cleansight_eval/temporal/features/` 和 feature version |
| 修改训练/推理方式 | 对应 temporal 或 detection Pipeline |
| 修改指标定义 | `benchmark/core/metrics.py` 或 evaluator |
| 修改固定数据集/split | `benchmark/testsets.yaml` 和 manifest |
| 修改 YOLO 数据构建 | `yolo-detection/pipeline/` |
| 修改结果对外格式 | `schemas/` 和对应 Python 校验器 |

更详细的抽象原则见 [`DESIGN.md`](DESIGN.md)，实际命令见
[`usage/TEST_COMMANDS.md`](../usage/TEST_COMMANDS.md)。
