# cleansight_eval —— 三条流水线的训练与评估框架

本目录是对 `docs/TRAIN_EVAL_REQUIREMENTS.md` 的一次落地。**架构是三条完整流水线 + 一条薄
公共层**。三条 pipeline 负责训练和预测；正式指标与 artifact 统一由 benchmark evaluator 生成：

| 流水线 | `pipeline` | 训练/预测输入 | 主要指标 | 模型前向基准 |
|---|---|---|---|---|
| 单帧检测 | `detection` | 单帧图像 + 检测标注 | mAP / P / R | N/A |
| 全序列时序 | `full_sequence_temporal` | 完整特征序列（逐帧监督） | acc / edit / F1 | N/A |
| 历史滑窗时序 | `sliding_window_temporal` | 历史特征窗口（末帧监督） | acc / edit / F1 | 单窗模型前向 |

> **核心约束**：一个模型在训练和评估时必须属于**同一条流水线**，采用一致的输入构造与
> 输出语义。不做"训练用窗口、评估用全量"、"一个 checkpoint 同时支持全量和滑窗"这类组合。
>
> **关键简化**：监督/loss 语义属于**流水线**而非模型——全序列一律逐帧 CE，滑窗一律末帧
> CE + 因果平滑。于是模型退化为可替换的纯 `nn.Module` 组件（只提供网络结构），由 `model.type`
> 选取；不再有 `family`/`feeding`/`task` 三层交叉抽象。

## 目录职责

| 层 | 目录 | 职责 |
|---|---|---|
| **公共层** | `cleansight_eval/core/` | run 组织、配置、环境、checkpoint 守卫、模型执行事实 `PredictionOutput`、结果兼容导出、矩阵、报告与完整性检查 |
| **评估真源** | `../benchmark/core/`、`../benchmark/evaluators/` | 指标口径、固定 testset、评估器、`EvaluationResult v2`、prediction artifact 与交付 manifest |
| **时序域** | `cleansight_eval/temporal/` | 两条时序流水线（`full_sequence_pipeline` / `sliding_window_pipeline`）+ 共享的 `data`（loader + meta）/ `metrics`（指标 + 延迟）/ `util`；模型在 `models/`（`gru`/`mstcn`/`mstcn2`/`transformer` + 注册表） |
| **检测域** | `cleansight_eval/detection/` | 单帧检测流水线（`pipeline`）+ 薄 ultralytics 适配器（`yolo`）+ 指标 |
| CLI | `cleansight_eval/cli/` | `train`/`eval` 按 `pipeline` 分派；`matrix` 汇总三类正式结果成单一矩阵 |
| 实验配置层 | `experiments/` | 流水线 + 模型类型/规模 + 数据 + 特征 + 训练参数 |

> 时序共享的 `data`/`metrics`/`artifacts`/`util` **只在两条时序流水线间复用**，绝不跨到 detection——
> 检测输入是图像、由 ultralytics 从 `data.yaml` 自持读入，与时序的 40 维特征序列是两套不相交
> 的数据格式。`feature_schema` 是上游检测/特征提取与下游时序之间唯一的显式接口。

## 核心不变量

- **结果三态**（`benchmark/core/result.py`）：`NOT_APPLICABLE` / `MISSING` / `COMPUTED` 严格区分。
  禁止用 0 冒充 N/A、禁止缺失伪装成 N/A。
- **checkpoint 自带绑定元信息**（`core/checkpoint.py`）：schema v1 sidecar 保存重建配置并绑定
  checkpoint SHA-256/大小；内容替换或配置错配都拒绝加载。
- **推理语义显式**（写入正式结果 `inference`）：滑窗记录窗口/推进/冷启动/reset/平滑；
  全序列与检测不伪造模型前向基准，生产延迟始终交给后端测量。
- **执行与判分分层**（`core/execution.py`）：pipeline 的 `predict()` 只产事实；CLI 把它交给
  `benchmark/evaluators`，pipeline 不再拥有 `evaluate()`。
- **异构评估矩阵**（`core/matrix.py`）：允许不同模型不同指标列，不生成综合分数。
- **不含业务门槛/自动晋升判断**：只产出评估事实（晋升决定由人负责）。

> 抽象/复用/过度设计的取舍准则见仓库级 [`docs/DESIGN.md`](../docs/DESIGN.md)。

## 环境准备

依赖清单在 [`requirements.txt`](requirements.txt)（当前测试环境 Python 3.10）。**核心**（numpy/torch/
PyYAML）跑时序即够；**ultralytics** 只有跑 YOLO 检测才需要（体积大）；tqdm 可选、pytest 仅测试用。

```bash
# A) 复用团队已有项目 venv（推荐，依赖已装齐）
source <venv>/bin/activate
# B) 新建独立 venv
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r framework/requirements.txt
```

## 用法

三类模型**共用同一套 CLI**（`train` / `eval` / `matrix`），由配置里的 `pipeline` 字段分派到
对应流水线；换模型只换 `--config`。本文统一从仓库根目录执行，模块名使用
`framework.cleansight_eval`，避免在根目录和 `framework/` 之间切换后混淆相对路径。

- **训练**读配置、跑训练、落 checkpoint（+ 重建元信息 sidecar），打印 `run_dir` 与
  `checkpoint` 路径。`-S/--set 点路径=值`（可多次）临时覆盖配置、不改文件；核心 CLI **不预设
  任何纵的调参名**，各纵按自己超参词汇寻址，如 `-S train.epochs=5`（两纵通用）、`-S train.batch=8`
  （检测/ultralytics）、`-S train.window=32`（时序滑窗）。时序训练结束后从 ``history.csv``
  自动生成 ``training_curves.png``；检测训练复用 Ultralytics 的 ``results.png``。
- **评估**加载 checkpoint 时校验绑定元信息，调用统一 `predict()` 后交给 benchmark evaluator；产出
  定义的 schema v2 `*.evaluation.json` 写入同 run 的 `evals/`，逐视频/逐图预测写入 `artifacts/`。结果只记录模型评估事实：testset
  fingerprint、checkpoint SHA-256、指标口径和 artifact SHA-256；环境、命令和 Git 信息不进入评估结果，训练期 `env.json` 仅供独立排障；同时写
  `*.delivery.manifest.json` 供外部仓库按文件契约消费。滑窗和全序列时序默认直接从本次
  `PredictionOutput` 生成 `viz/segmentation-<split>-pNN.png`，不重复加载模型或执行推理。
- **矩阵**把 `runs/` 下所有新旧评估结果汇成一张异构矩阵（`matrix.json` 机读 + `matrix.md` 人读）；
  `--pipeline <名>` 只汇总某一类流水线做同类对比（输出带 `.<名>` 后缀，不覆盖全量矩阵）。

> **目录全自动**：训练每跑一次开一个 run 目录 `runs/<type>-<时间戳>/`（下挂 `checkpoints/`、
> `evals/`、`config.resolved.json`、`env.json`）；评估输出目录从 `--ckpt` 向上自动定位到同 run
> 的 `evals/`。你唯一要手填的是 `--ckpt`——训练结束打印的那行 `[train] checkpoint=...`。

### 1. 单帧检测（`pipeline: detection`，YOLO）

由 ultralytics 自持训练/验证；输入是图像，无 `feature_schema`。checkpoint 是嵌套的
`<name>/weights/best.pt`（`name` 取 `data.name`）。

```bash
python -m framework.cleansight_eval.cli.train --config framework/experiments/yolo-group1.yaml
python -m framework.cleansight_eval.cli.eval --config framework/experiments/yolo-group1.yaml \
  --ckpt runs/<run>/checkpoints/group1_large/weights/best.pt
```

### 2. 历史滑窗时序（`pipeline: sliding_window_temporal`，GRU 参照）

有界因果窗逐帧推理，训练造"窗口+末帧"样本、评估逐窗前推；测单窗模型前向耗时（非生产延迟）。模型必须**因果**
（`gru`），非因果模型（`mstcn` / `transformer`）会被拒绝。可靠训练统一保存 `best.pt` 和
`last.pt`。

```bash
python -m framework.cleansight_eval.cli.train --config framework/experiments/gru-actionmixed.yaml
python -m framework.cleansight_eval.cli.eval --config framework/experiments/gru-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/best.pt
```

### 3. 全序列时序（`pipeline: full_sequence_temporal`，MS-TCN 参照）

一次看到完整序列、逐帧监督、逐帧 argmax；以 `batch_size=1` 逐条喂入；**延迟标 N/A**。
可选模型包括 MS-TCN、MS-TCN++ 和 Transformer；可靠训练统一保存 `best.pt` 和 `last.pt`。

```bash
python -m framework.cleansight_eval.cli.train --config framework/experiments/mstcn-actionmixed.yaml
python -m framework.cleansight_eval.cli.eval --config framework/experiments/mstcn-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/best.pt
```

### 汇总评估矩阵（三类结果汇入同一张表）

```bash
python -m framework.cleansight_eval.cli.matrix --runs runs
```

## 配置字段速查

每个 `experiments/*.yaml` 都是**带行内注释的模板**——配新实验请复制最接近的那个改。校验分两层：
框架层通用字段（`pipeline`/`model`/`data`）在 [core/config.py](cleansight_eval/core/config.py)，
流水线专属必填字段在各 `validate_config`。

### 顶层（三类共用）

| 字段 | 必填 | 取值 | 说明 |
|---|---|---|---|
| `schema_version` | ✅ | `1` | 配置契约版本；未知字段和未知 override 路径会被拒绝 |
| `pipeline` | ✅ | `detection` / `full_sequence_temporal` / `sliding_window_temporal` | 分派到哪条流水线 |
| `model` | ✅ | 映射，含 `type` | 见下（各流水线字段不同） |
| `data` | ✅ | 映射 | 见下 |
| `train` | 时序✅ / 检测可选 | 映射 | 训练超参 |
| `feature_schema` | **时序✅ / 检测无** | 映射 | 上游特征格式契约，训练前校验维度 |
| `evaluation` | 正式评估建议✅ | 映射 | 固定 testset、artifact 和 smoke 限制 |

### 时序（两条时序流水线）

```yaml
model:
  type: gru            # ✅ gru（因果，两条都可）/ mstcn·mstcn2·transformer（非因果，仅全序列）
  input_dim: 40        # ✅ 特征维；须等于 loader 产出（8 检测类 × 5）
  num_classes: 6       # ✅ 动作类数；须等于 labels/data.yaml 的 names 数
  hidden: 128          # 模型超参（gru/mstcn 均用 hidden）
  num_layers: 3        # gru 专属
  # allow_missing_meta: true  # 仅 exploratory 外部裸 .pt；按本段结构 strict 加载
data:
  dataset_ref: temporal.actionmixed-v2  # ✅ 从 benchmark catalog 解析根目录、类别和 manifest
  split_train: train   # ✅ 训练用子目录
  split_val: val       # ✅ 训练期模型选择
  split_eval: test     # ✅ 评估用子目录
feature_schema:
  dim: 40              # ✅ 须与 input_dim / loader 一致，否则训练前报错
  version: actionmixed-bbox-8cls-v1
  # 可选：按 frames/data.yaml 的目标名或类别 ID，将对应 [presence,cx,cy,w,h] 清零。
  # mask_targets: [syringe, air_gun]
evaluation:
  mode: formal  # testset 由 dataset_ref + split_eval 唯一推导
  limits:
    is_smoke: false
train:
  epochs: 20           # 可选，默认 20
  lr: 0.001            # 可选，默认 1e-3
  batch_size: 32       # 滑窗用；全序列固定 batch_size=1
  window: 16           # 滑窗必填/默认 64，须 ≤ 最短视频采样帧数；全序列不需要
  weight_decay: 0.0    # 可选
  grad_clip: 5.0       # 可选，缺省则不裁剪
```

`feature_schema.mask_targets` 是 ActionMixed 特征层参数，滑窗与全序列时序流水线共用，
因此不需要在 GRU、Transformer、MS-TCN 等模型内分别实现。未配置或配置为空列表时行为与原来一致；
未知目标名、越界 ID、错误参数类型会在流水线启动前报错。单个目标也可在训练命令中临时覆盖：

```bash
python -m framework.cleansight_eval.cli.train \
  --config framework/experiments/gru-actionmixed.yaml \
  -S feature_schema.mask_targets=syringe
```

训练期随机目标遮罩使用独立的 `augmentation.target_mask`，不改变 feature schema。当前
`frame_dropout` 的 `probability` 表示每个指定目标在每个采样帧独立清零 5 维特征的概率；
它只作用于 train，val/test 保持干净输入。配置会随 resolved config 和 checkpoint metadata 保存。

组员提供的外部裸时序 `.pt` 可在 `evaluation.mode: exploratory` 下显式设置
`model.allow_missing_meta: true`。此时模型由 YAML 重建，接受裸 state dict、`model_state` 或
`state_dict` 包装，并以 `strict=True` 校验全部参数；结果保留 `missing_meta_fallback` 和未绑定事实。
formal 模式仍要求与权重 SHA-256 绑定的同名 `.meta.json`。

### 检测（YOLO）

```yaml
model:
  type: yolo           # ✅ 检测适配器 key
  weights: yolo11n.pt  # 起始权重
  imgsz: 640           # 可选，默认 640
data:
  data_yaml: ../.../datasets/group1_large/data.yaml  # ✅ 标准 YOLO 数据集清单
  eval_split: val      # 可选，默认 val
  name: group1_large   # 可选，展示名 + 权重子目录名
evaluation:
  mode: exploratory     # 外部权重缺 metadata 时使用；正式模型改为 formal
  testset_id: yolo.group1_large.val
  save_predictions: true
  conf: 0.001
  iou: 0.7
  max_det: 300
  agnostic_nms: false
train:                 # 整段可选，透传给 ultralytics
  epochs: 100
  batch: 16            # 注意是 batch，不是 batch_size
  patience: 20
```

历史 envelope 不原地覆盖，使用转换命令生成旁路 v2 文件；矩阵可同时读取新旧文件：

```bash
python -m framework.cleansight_eval.cli.upgrade_envelope \
  --input runs/<run>/evals/<old>.envelope.json
```

## 评估输出

一次 `eval` 会写入四组互相链接的产物：

| 位置 | 内容 |
|---|---|
| `evals/*.evaluation.json` | 统一 `EvaluationResult v2`，只保存评估事实 |
| `artifacts/*.predictions.json` | YOLO 逐图预测或时序逐视频预测/真值 |
| `checkpoints/<ckpt>.eval.md` | 当前 checkpoint 专属人读报告 |
| `checkpoints/EVALUATION_REPORT.md` | 按时序/YOLO 分类的追加式版本报告 |
| `evals/*.delivery.manifest.json` | checkpoint、metadata、配置、报告和 artifact 的文件摘要清单 |

外部结构契约位于仓库根目录 [`../schemas/`](../schemas/)；运行时仍由
`benchmark/core/result.py`、`benchmark/core/artifacts.py` 和 `benchmark/core/delivery.py` 中的
Python 校验器负责。Schema 不执行模型、指标、复制或上传。

### Formal 与 Exploratory

- `formal`：testset 必须在 `benchmark/testsets.yaml` 登记并通过校验；metadata 必须与 checkpoint
  SHA-256 绑定；prediction artifact 必须保存。
- `exploratory`：允许外部 YOLO 权重缺少 metadata，或临时 testset 尚未满足正式条件。结果可用于
  调试，但不能称为正式 benchmark。

评估 CLI 当前不提供 `-S` 覆盖；需要切换 profile 时请复制/编辑实验 YAML，确保模式变化能被保留。

## 常见问题

- `FileNotFoundError: /abs/path/to/...`：配置仍是占位路径；`data.root` 和 `data.data_yaml` 的相对路径
  以实验 YAML 所在目录解析。
- `formal testset validation failed`：先运行
  `python tools/validate_testsets.py --catalog benchmark/testsets.yaml --json`。数据充足时应修复切分；小数据
  开发阶段可在对应 testset 设置 `split_overlap_policy: frame`，允许同源视频分段但禁止相同帧ID
  重复；只有特殊排查才使用 `allow`。非严格策略必须保留开发期 purpose，不能把结果描述为独立同源
  隔离 benchmark。
- 外部 `.pt` 没有 `.meta.json`：仅在 `evaluation.mode: exploratory` 且
  `model.allow_missing_meta: true` 时允许 YOLO 探索性评估。
- Transformer nested-tensor warning：是 `norm_first=True` 未使用 nested-tensor 快速路径的性能提示，
  不等同于评估失败。

## 测试

```bash
pytest   # 从仓库根目录运行，需已激活项目 venv
```

- `test_temporal_metrics.py` / `test_detection_metrics.py`：指标口径独立可测（免 ultralytics）。
- `test_checkpoint_compat.py`：错配 checkpoint 拒绝加载。
- `test_envelope_matrix.py` / `test_cross_vertical_matrix.py`：三态、以及**三类结果汇入单一异构
  矩阵**（最关键的不变量守卫）。
- `test_pipeline_smoke.py` / `test_mstcn_smoke.py`：时序合成数据端到端 train→eval→matrix。
- `test_detection_smoke.py`：检测流水线端到端（注入假 adapter，免 ultralytics）。

> 冒烟用合成数据机械验证链路；**数值对齐验收**需在有真实数据的机器上执行。

## 扩展点

- **新增同架构变体**：只改 `experiments/*.yaml` 的 `model` 段（hidden/num_layers…）。
- **新增时序模型**（causal-TCN / LSTM…）：在 `temporal/models/` 加一个纯 `nn.Module`
  文件（输入 `[B,T,F]`→输出 `[B,T,C]`），并在 `temporal/models/__init__.py` 的注册表登记一行
  `{"build": ..., "causal": <bool>}`。`causal=True` 才允许进滑窗流水线。**两条时序流水线零改动**
  即可复用——监督口径与推理由流水线拥有，模型只管网络结构。**可选 duck-type 钩子**（有则调、
  无则退化，不写基类）让个别模型携带自身训练细节而不污染脊柱：`fit_normalization(features)`
  训练前按训练集统计写归一化 buffer（如 MS-TCN）；`compute_loss(x, y, criterion)` 让模型自持
  训练配方（如 MS-TCN++ 的多 stage 深监督 + T-MSE，仅全序列流水线调用）。
- **新增检测器**（DETR…）：在 `detection/yolo.py` 旁加适配器并在 `get_adapter` 登记，暴露
  `train`/`val` 即可，不需实现任何时序接口。
