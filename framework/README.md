# cleansight_eval —— 三条流水线的训练与评估框架

本目录是对 `docs/TRAIN_EVAL_REQUIREMENTS.md` 的一次落地。**架构是三条完整流水线 + 一条薄
公共层**。真实需求只有三种固定组合，框架就直接按这三种定义完整流水线，每条内部同时负责
训练与评估，并保证训练与评估使用同一种数据组织：

| 流水线 | `pipeline` | 训练/评估输入 | 主要指标 | 延迟 |
|---|---|---|---|---|
| 单帧检测 | `detection` | 单帧图像 + 检测标注 | mAP / P / R | N/A |
| 全序列时序 | `full_sequence_temporal` | 完整特征序列（逐帧监督） | acc / edit / F1 | N/A |
| 历史滑窗时序 | `sliding_window_temporal` | 历史特征窗口（末帧监督） | acc / edit / F1 | 单 tick |

> **核心约束**：一个模型在训练和评估时必须属于**同一条流水线**，采用一致的输入构造与
> 输出语义。不做"训练用窗口、评估用全量"、"一个 checkpoint 同时支持全量和滑窗"这类组合。
>
> **关键简化**：监督/loss 语义属于**流水线**而非模型——全序列一律逐帧 CE，滑窗一律末帧
> CE + 因果平滑。于是模型退化为可替换的纯 `nn.Module` 组件（只提供网络结构），由 `model.type`
> 选取；不再有 `family`/`feeding`/`task` 三层交叉抽象。

## 目录职责

| 层 | 目录 | 职责 |
|---|---|---|
| **公共层** | `cleansight_eval/core/` | run 组织、配置（格式中立）、环境、checkpoint 重建元信息 + 守卫、结果三态信封、异构矩阵、完整性检查（含特征维度契约校验） |
| **时序域** | `cleansight_eval/temporal/` | 两条时序流水线（`full_sequence_pipeline` / `sliding_window_pipeline`）+ 共享的 `data`（loader + meta）/ `metrics`（指标 + 延迟）/ `util`；模型在 `models/`（`gru`/`mstcn`/`mstcn2` + 注册表） |
| **检测域** | `cleansight_eval/detection/` | 单帧检测流水线（`pipeline`）+ 薄 ultralytics 适配器（`yolo`）+ 指标 |
| CLI | `cleansight_eval/cli/` | `train`/`eval` 按 `pipeline` 分派（`_registry.py`）；`matrix` 汇总三类信封成单一矩阵 |
| 实验配置层 | `experiments/` | 流水线 + 模型类型/规模 + 数据 + 特征 + 训练参数 |

> 时序共享的 `data`/`metrics`/`util` **只在两条时序流水线间复用**，绝不跨到 detection——
> 检测输入是图像、由 ultralytics 从 `data.yaml` 自持读入，与时序的 40 维特征序列是两套不相交
> 的数据格式。`feature_schema` 是上游检测/特征提取与下游时序之间唯一的显式接口。

## 核心不变量

- **结果三态**（`core/envelope.py`）：`NOT_APPLICABLE` / `MISSING` / `COMPUTED` 严格区分。
  禁止用 0 冒充 N/A、禁止缺失伪装成 N/A。
- **checkpoint 自带重建元信息**（`core/checkpoint.py`）：保存 `type` + 模型配置 +
  feature schema；加载时校验，错配立即抛 `CompatibilityError`，不静默加载。
- **推理语义显式**（挂进信封 `inference_semantics`）：滑窗记录窗口/推进/冷启动/reset/平滑；
  全序列与检测绝不产生虚假实时延迟——延迟标记为 `N/A`。
- **异构评估矩阵**（`core/matrix.py`）：允许不同模型不同指标列，不生成综合分数。
- **不含业务门槛/自动晋升判断**：只产出评估事实（晋升决定由人负责）。

> 抽象/复用/过度设计的取舍准则见仓库级 [`docs/DESIGN.md`](../docs/DESIGN.md)。

## 环境准备

依赖清单在 [`requirements.txt`](requirements.txt)（Python 3.12 验证）。**核心**（numpy/torch/
PyYAML）跑时序即够；**ultralytics** 只有跑 YOLO 检测才需要（体积大）；tqdm 可选、pytest 仅测试用。

```bash
# A) 复用团队已有项目 venv（推荐，依赖已装齐）
source <venv>/bin/activate
# B) 新建独立 venv
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r framework/requirements.txt
```

## 用法

三类模型**共用同一套 CLI**（`train` / `eval` / `matrix`），由配置里的 `pipeline` 字段分派到
对应流水线；换模型只换 `--config`。先激活 venv、进入 `framework/`，之后直接用 `python`：

```bash
cd framework
```

- **训练**读配置、跑训练、落 checkpoint（+ 重建元信息 sidecar），打印 `run_dir` 与
  `checkpoint` 路径。`-S/--set 点路径=值`（可多次）临时覆盖配置、不改文件；核心 CLI **不预设
  任何纵的调参名**，各纵按自己超参词汇寻址，如 `-S train.epochs=5`（两纵通用）、`-S train.batch=8`
  （检测/ultralytics）、`-S train.window=32`（时序滑窗）。
- **评估**加载 checkpoint 时校验重建元信息，错配即抛 `CompatibilityError`；产出一份三态信封
  写入同 run 的 `evals/`。训练与评估同属一条流水线，输入构造与输出语义一致。
- **矩阵**把 `runs/` 下所有信封汇成一张异构矩阵（`matrix.json` 机读 + `matrix.md` 人读）；
  `--pipeline <名>` 只汇总某一类流水线做同类对比（输出带 `.<名>` 后缀，不覆盖全量矩阵）。

> **目录全自动**：训练每跑一次开一个 run 目录 `runs/<type>-<时间戳>/`（下挂 `checkpoints/`、
> `evals/`、`config.resolved.json`、`env.json`）；评估输出目录从 `--ckpt` 向上自动定位到同 run
> 的 `evals/`。你唯一要手填的是 `--ckpt`——训练结束打印的那行 `[train] checkpoint=...`。

### 1. 单帧检测（`pipeline: detection`，YOLO）

由 ultralytics 自持训练/验证；输入是图像，无 `feature_schema`。checkpoint 是嵌套的
`<name>/weights/best.pt`（`name` 取 `data.name`）。

```bash
python -m cleansight_eval.cli.train --config experiments/yolo-group1.yaml
python -m cleansight_eval.cli.eval --config experiments/yolo-group1.yaml \
  --ckpt runs/<run>/checkpoints/group1_large/weights/best.pt
```

### 2. 历史滑窗时序（`pipeline: sliding_window_temporal`，GRU 参照）

有界因果窗逐帧推理，训练造"窗口+末帧"样本、评估逐窗前推；测单 tick 实时延迟。模型必须**因果**
（`gru`），非因果模型（`mstcn`）会被拒绝。checkpoint 形如 `gru-final-<stamp>.pt`。

```bash
python -m cleansight_eval.cli.train --config experiments/gru-actionmixed.yaml
python -m cleansight_eval.cli.eval --config experiments/gru-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/gru-final-<stamp>.pt
```

### 3. 全序列时序（`pipeline: full_sequence_temporal`，MS-TCN 参照）

一次看到完整序列、逐帧监督、逐帧 argmax；以 `batch_size=1` 逐条喂入；**延迟标 N/A**。
checkpoint 形如 `mstcn-final-<stamp>.pt`。

```bash
python -m cleansight_eval.cli.train --config experiments/mstcn-actionmixed.yaml
python -m cleansight_eval.cli.eval --config experiments/mstcn-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/mstcn-final-<stamp>.pt
```

### 汇总评估矩阵（三类信封汇入同一张表）

```bash
python -m cleansight_eval.cli.matrix --runs runs
```

## 配置字段速查

每个 `experiments/*.yaml` 都是**带行内注释的模板**——配新实验请复制最接近的那个改。校验分两层：
框架层通用字段（`pipeline`/`model`/`data`）在 [core/config.py](cleansight_eval/core/config.py)，
流水线专属必填字段在各 `validate_config`。

### 顶层（三类共用）

| 字段 | 必填 | 取值 | 说明 |
|---|---|---|---|
| `pipeline` | ✅ | `detection` / `full_sequence_temporal` / `sliding_window_temporal` | 分派到哪条流水线 |
| `model` | ✅ | 映射，含 `type` | 见下（各流水线字段不同） |
| `data` | ✅ | 映射 | 见下 |
| `train` | 时序✅ / 检测可选 | 映射 | 训练超参 |
| `feature_schema` | **时序✅ / 检测无** | 映射 | 上游特征格式契约，训练前校验维度 |

### 时序（两条时序流水线）

```yaml
model:
  type: gru            # ✅ 模型注册表键：gru（因果，两条都可）/ mstcn·mstcn2（非因果，仅全序列）
  input_dim: 40        # ✅ 特征维；须等于 loader 产出（8 检测类 × 5）
  num_classes: 6       # ✅ 动作类数；须等于 labels/data.yaml 的 names 数
  hidden: 128          # 模型超参（gru/mstcn 均用 hidden）
  num_layers: 3        # gru 专属
data:
  root: /abs/path/to/cleansight-ActionMixed   # ✅ 数据集根
  split_train: train   # ✅ 训练用子目录
  split_eval: test     # ✅ 评估用子目录
  name: cleansight-ActionMixed                # 可选，展示名
  action_mapping: labels/data.yaml            # 可选，默认 labels/data.yaml
  labels_dir: labels   # 可选，默认 labels
  frames_dir: frames   # 可选，默认 frames
feature_schema:
  dim: 40              # ✅ 须与 input_dim / loader 一致，否则训练前报错
  version: actionmixed-bbox-8cls-v1
train:
  epochs: 20           # 可选，默认 20
  lr: 0.001            # 可选，默认 1e-3
  batch_size: 32       # 滑窗用；全序列固定 batch_size=1
  window: 16           # 滑窗必填/默认 64，须 ≤ 最短视频采样帧数；全序列不需要
  weight_decay: 0.0    # 可选
  grad_clip: 5.0       # 可选，缺省则不裁剪
```

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
train:                 # 整段可选，透传给 ultralytics
  epochs: 100
  batch: 16            # 注意是 batch，不是 batch_size
  patience: 20
```

## 测试

```bash
cd framework && python -m pytest tests -q   # 需已激活项目 venv
```

- `test_temporal_metrics.py` / `test_detection_metrics.py`：指标口径独立可测（免 ultralytics）。
- `test_checkpoint_compat.py`：错配 checkpoint 拒绝加载。
- `test_envelope_matrix.py` / `test_cross_vertical_matrix.py`：三态、以及**三类信封汇入单一异构
  矩阵**（最关键的不变量守卫）。
- `test_pipeline_smoke.py` / `test_mstcn_smoke.py`：时序合成数据端到端 train→eval→matrix。
- `test_detection_smoke.py`：检测流水线端到端（注入假 adapter，免 ultralytics）。

> 冒烟用合成数据机械验证链路；**数值对齐验收**需在有真实数据的机器上执行。

## 扩展点

- **新增同架构变体**：只改 `experiments/*.yaml` 的 `model` 段（hidden/num_layers…）。
- **新增时序模型**（Transformer / causal-TCN…）：在 `temporal/models/` 加一个纯 `nn.Module`
  文件（输入 `[B,T,F]`→输出 `[B,T,C]`），并在 `temporal/models/__init__.py` 的注册表登记一行
  `{"build": ..., "causal": <bool>}`。`causal=True` 才允许进滑窗流水线。**两条时序流水线零改动**
  即可复用——监督口径与推理由流水线拥有，模型只管网络结构。**可选 duck-type 钩子**（有则调、
  无则退化，不写基类）让个别模型携带自身训练细节而不污染脊柱：`fit_normalization(features)`
  训练前按训练集统计写归一化 buffer（如 MS-TCN）；`compute_loss(x, y, criterion)` 让模型自持
  训练配方（如 MS-TCN++ 的多 stage 深监督 + T-MSE，仅全序列流水线调用）。
- **新增检测器**（DETR…）：在 `detection/yolo.py` 旁加适配器并在 `get_adapter` 登记，暴露
  `train`/`val` 即可，不需实现任何时序接口。
