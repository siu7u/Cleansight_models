# cleansight_eval —— 分层训练与评估框架（骨架 + GRU 参照实现）

本目录是对 `docs/TRAIN_EVAL_REQUIREMENTS.md` 的一次落地。**架构为"两纵一脊"**：
检测（单帧无状态）与时序（滑窗/因果）是两个**互不 import 的独立纵**，各自拥有自己的
模型、喂入、指标与编排；两者**只共享一条薄脊柱** `core/`（信封 + 矩阵 + run/config/
checkpoint/integrity/environment）并汇入**同一份异构矩阵**。

> 设计取舍：检测由 ultralytics 自持训练/验证，时序是手写因果循环——二者在代码层几乎
> 零共享。此前用 `task/family/feeding` 四个"对等注册表"强行统一，检测在每个抽象上都退化
> （family 无视 Protocol、`single_frame.evaluate` 直接 raise）。现已**删除这些跨域假抽象**，
> 不再强行把两类模型抽象成一个。CLI 靠一个 `task→纵` 小映射分派，是唯一同时 import 两纵的地方。

## 目录职责

| 层 | 目录 | 职责 | 归属 |
|---|---|---|---|
| **共享脊柱** | `cleansight_eval/core/` | run 组织、配置（格式中立）、环境、checkpoint 重建元信息 + 守卫、结果三态信封、异构矩阵、完整性检查、`feature_schema` 上→下游契约 | 两纵共享，**不 import 任何纵** |
| **时序纵** | `cleansight_eval/temporal/` | 自持编排(orchestration) + 时序 family(gru) + 喂入(full_sequence/windowed_causal/stateful) + 指标/类型/loader/perf；纵内自带 `get_family`/`get_feeding` 注册表 | 时序专属 |
| **检测纵** | `cleansight_eval/detection/` | 薄 ultralytics 适配器(adapter, train/val) + 指标 + 编排；单帧语义为纵内常量，无 family/feeding Protocol | 检测专属 |
| CLI | `cleansight_eval/cli/` | `train`/`eval` 按 `task→纵` 分派（`_registry.py`）；`matrix` 汇总两纵信封成单一矩阵 | 组合根 |
| 实验配置层 | `experiments/` | 族+规模+任务+喂入模式+数据+格式+训练/评估参数 | 配置 |

> 两纵**故意不共享** family/feeding/task 抽象。时序纵的 family 是"网络+forward+loss+因果契约"，
> 检测纵的 adapter 是"ultralytics train/val 封装"——两套不相交的契约，各自演化。
> `feature_schema` 是两纵之间唯一的显式接口：上游检测/特征提取声明产出格式，下游时序声明消费格式并校验维度。

## 输入与喂入模式（修正后的认知）

一个模型的"输入"由两条**正交**的轴描述，谁都不吞并谁：

- **格式（feature schema）**：每个输入单元长什么样 —— `dim` + `layout`（各通道语义）
  + `version`（哪版格式）。**只讲格式，不讲来源**。时序是特征向量 schema，检测是图像
  （`modality: image, imgsz`）。
- **喂入模式（feeding mode）**：单元怎么按时间打包给模型 —— 窗口长度、因果性、
  状态/reset、读/监督哪一帧。`offline`（窗口→∞）、`realtime`（有界因果窗）、
  `single_frame`（窗口=1 无状态）都是这条轴上的**取值**。

`输入 = 格式 × 喂入模式`。换特征提取器只动"格式"，换推理协议只动"喂入模式"。

**喂入模式是 train/eval 中立的共享轴（关键，别再当成评估专属）**：

- **一个实验只有一个喂入模式，训练与评估共用它**：训练怎么喂，评估就怎么喂。不做
  "同一 checkpoint 用多种喂入分别评估"的扩展——那是多余设计。
- 训练与评估**唯一真正不同的**，是选定喂入模式之后**外面那圈机器**：训练是
  loss+反向传播，评估是算指标+出信封。这圈机器与喂入模式**正交**。
- 编排 = **Task**；建模型/前向 = **Family**；喂入模式 = **feeding**；地基（配置/设备/
  run 目录/checkpoint/信封/矩阵）= **core**。启动任一流程都需要这几者，光靠"两个协议"不够。

**已落地**：喂入模式是 `cleansight_eval/temporal/feeding/` 下的**纵内**注册表
（`get_feeding`，时序专属），由**顶层单个 `feeding:` 字段**表达，训练与评估共用。以 `windowed_causal`
为例：训练侧 `build_training_dataset` 造"窗口+末帧"样本，评估侧 `evaluate` 逐窗推理——同
一喂入规格的单一真源。信封字段亦从 `execution` 改名为 `feeding`。backprop 与打分外壳各自
保留（正交于喂入模式）。

## 核心不变量

- **结果三态**（`core/envelope.py`，§10）：`NOT_APPLICABLE` / `MISSING` / `COMPUTED`
  严格区分。禁止用 0 冒充 N/A、禁止缺失伪装成 N/A。
- **checkpoint 自带重建元信息**（`core/checkpoint.py`，§7.2/§8.1）：保存 family +
  模型配置 + feature schema；加载时校验，错配立即抛 `CompatibilityError`，不静默加载。
- **喂入语义显式**（§8.3）：windowed_causal 信封记录窗口、推进、冷启动、reset、平滑；
  full_sequence 绝不产生虚假实时延迟——延迟标记为 `N/A`（§8.4/§13.6）。
- **异构评估矩阵**（`core/matrix.py`，§9）：允许不同模型不同指标列，不生成综合分数。
- **不含业务门槛/自动晋升判断**（§4.5）：只产出评估事实。

## 环境准备

依赖清单在 [`requirements.txt`](requirements.txt)（Python 3.12 验证）。**核心**（numpy/torch/
PyYAML）跑时序即够；**ultralytics** 只有跑 YOLO 检测纵才需要（体积大）；tqdm 可选、pytest 仅测试用。

两种方式，任选其一：

```bash
# A) 复用团队已有的项目 venv（推荐，依赖已装齐）——本机路径见团队约定
source <venv>/bin/activate

# B) 新建独立 venv
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r framework/requirements.txt        # 纯时序可删掉 ultralytics 那行再装
```

## 用法

三类模型**共用同一套 CLI**（`train` / `eval` / `matrix`），由配置里的 `task` 字段分派到
对应的纵；换模型只换 `--config`。**先激活虚拟环境（见上）、进入 `framework/`，之后所有命令直接
用 `python`**（用 `python -m` 从 `framework/` 运行时当前目录已自动进 `sys.path`，无需再设
`PYTHONPATH`）：

```bash
cd framework
```

通用规则：

- **训练**读配置、跑训练、落 checkpoint（+ 重建元信息 sidecar），打印 `run_dir` 与
  `checkpoint` 路径。`--epochs/--lr/--batch_size/--window` 可临时覆盖配置，不改文件。
- **评估**用**训练同一喂入模式**（`cfg["feeding"]`，不做多模式扫描），加载 checkpoint
  时校验重建元信息，错配即抛 `CompatibilityError`；产出一份三态信封写入同 run 的 `evals/`。
- **矩阵**把 `runs/` 下所有信封汇成一张异构矩阵（`matrix.json` 机读 + `matrix.md` 人读），
  三类模型的信封**汇入同一张表**，指标列可不同、保留 N/A/MISSING/已计算三态。

> **目录全自动，无需手建**：训练每跑一次自动开一个 run 目录 `runs/<family>-<时间戳>/`
> （下挂 `checkpoints/`、`evals/`、`config.resolved.json`、`env.json`；根目录 `runs/` 可用
> `--runs-dir` 改）；评估的输出目录也从 `--ckpt` 向上自动定位到同 run 的 `evals/`，无需
> `--out-dir`。你唯一要手动填的是 `--ckpt`——就是训练结束时打印的那行 `[train] checkpoint=...`，
> 直接复制。下文 `<stamp>` 即该时间戳，路径给的是**形状**。

### 1. YOLO —— 单帧检测（`task: detection`, `feeding: single_frame`）

由 ultralytics 自持训练/验证；输入是图像，无 `feature_schema`。checkpoint 是嵌套的
`<name>/weights/best.pt`（`name` 取 `data.name`，如 `group1_large`）。

```bash
# 训练：产出 runs/<run>/checkpoints/<name>/weights/best.pt（+ best.pt.meta.json）
python -m cleansight_eval.cli.train --config experiments/yolo-group1.yaml

# 评估：mAP/P/R 逐类三态；实时延迟标 N/A（离线检测不测）
python -m cleansight_eval.cli.eval \
  --config experiments/yolo-group1.yaml \
  --ckpt runs/<run>/checkpoints/group1_large/weights/best.pt
```

### 2. windowed_causal —— 实时因果滑窗（`task: temporal`, `feeding: windowed_causal`, GRU 参照）

有界因果窗口逐帧推理，训练造"窗口+末帧"样本、评估逐窗前推同一喂入规格；测单 tick 实时延迟。
checkpoint 直接落在 `checkpoints/` 下，形如 `gru-final-<时间戳>.pt`。

```bash
# 训练：产出 runs/<run>/checkpoints/gru-final-<stamp>.pt（+ 同名 .meta.json）
python -m cleansight_eval.cli.train --config experiments/gru-actionmixed.yaml

# 评估：时序逐帧指标 + 实时延迟（single_tick_ms）
python -m cleansight_eval.cli.eval \
  --config experiments/gru-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/gru-final-<stamp>.pt
```

### 3. full_sequence —— 离线全序列（`task: temporal`, `feeding: full_sequence`, MS-TCN 参照）

一次看到完整序列、逐帧监督、非因果。以 `train_batch_size=1` 逐条喂入；**实时延迟标 N/A**
（离线不产生虚假实时延迟）。与 windowed_causal **共用同一时序编排器**，差异只在 feeding 与
family 两条多态轴。checkpoint 形如 `mstcn-final-<时间戳>.pt`。

```bash
# 训练：产出 runs/<run>/checkpoints/mstcn-final-<stamp>.pt（+ 同名 .meta.json）
python -m cleansight_eval.cli.train --config experiments/mstcn-actionmixed.yaml

# 评估：时序逐帧指标；延迟标 N/A
python -m cleansight_eval.cli.eval \
  --config experiments/mstcn-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/mstcn-final-<stamp>.pt
```

### 汇总评估矩阵（三类信封汇入同一张表）

```bash
python -m cleansight_eval.cli.matrix --runs runs
```

## 配置字段速查

每个 `experiments/*.yaml` 都是**带行内注释的模板**——配新实验请复制最接近的那个改。下表列
字段/是否必填/默认值/在哪消费。校验分两层：框架层通用字段（`family`/`model`/`task`/`feeding`/`data`）
在 [core/config.py](cleansight_eval/core/config.py)，纵专属必填字段在各 `validate_config`
（[时序](cleansight_eval/temporal/orchestration.py) / [检测](cleansight_eval/detection/orchestration.py)）。

### 顶层（三类共用）

| 字段 | 必填 | 取值 | 说明 |
|---|---|---|---|
| `family` | ✅ | `gru` / `mstcn` / `yolo` | 选族/适配器（纵内注册表键） |
| `task` | ✅ | `temporal` / `detection` | 分派到哪个纵 |
| `feeding` | ✅ | `windowed_causal` / `full_sequence` / `single_frame` | 喂入模式，训练评估共用 |
| `model` | ✅ | 映射 | 见下（各纵字段不同） |
| `data` | ✅ | 映射 | 见下（各纵字段不同） |
| `train` | 时序✅ / 检测可选 | 映射 | 训练超参 |
| `feature_schema` | **时序✅ / 检测无** | 映射 | 上游特征格式契约，训练前校验维度 |

### 时序（windowed_causal / full_sequence）

```yaml
model:
  input_dim: 40        # ✅ 特征维；须等于 loader 产出（8 检测类 × 5）
  num_classes: 6       # ✅ 动作类数；须等于 labels/data.yaml 的 names 数
  hidden: 128          # 族专属超参（gru）；mstcn 亦用 hidden
  num_layers: 3        # gru 专属；mstcn 用 arch: ms_tcn
data:
  root: /abs/path/to/cleansight-ActionMixed   # ✅ 数据集根
  split_train: train   # ✅ 训练用的子目录
  split_eval: test     # ✅ 评估用的子目录
  name: cleansight-ActionMixed                # 可选，进信封/矩阵的展示名
  action_mapping: labels/data.yaml            # 可选，默认 labels/data.yaml
  labels_dir: labels   # 可选，默认 labels
  frames_dir: frames   # 可选，默认 frames
feature_schema:
  dim: 40              # ✅ 须与 input_dim / loader 一致，否则训练前报错
  version: actionmixed-bbox-8cls-v1           # 格式版本号（人读溯源）
train:
  epochs: 20           # 可选，默认 20
  lr: 0.001            # 可选，默认 1e-3
  batch_size: 32       # 可选，默认 32；full_sequence 会被 train_batch_size=1 覆盖
  window: 16           # 可选，默认 64；须 ≤ 最短视频采样帧数；full_sequence 忽略
  weight_decay: 0.0    # 可选，默认 0
  grad_clip: 5.0       # 可选，缺省则不裁剪
```

### 检测（YOLO）

```yaml
model:
  weights: yolo11n.pt  # 起始权重；nano 快、要更准换 yolo11s.pt 等
  imgsz: 640           # 可选，默认 640
data:
  data_yaml: ../.../datasets/group1_large/data.yaml  # ✅ 指向标准 YOLO 数据集清单
  eval_split: val      # 可选，默认 val（train/val/test）
  name: group1_large   # 可选，进信封/矩阵的展示名 + 权重子目录名
train:                 # 整段可选，透传给 ultralytics
  epochs: 100          # 默认 100
  batch: 16            # 默认 16（注意是 batch，不是 batch_size）
  patience: 20         # 默认 20
# 检测无 feature_schema：输入是图像；feeding 固定 single_frame，写别的会报错。
```

> **CLI 临时覆盖**（不改文件）：`--epochs/--lr/--batch_size/--window` 会写进 `train` 段
> （[core/config.py](cleansight_eval/core/config.py) `apply_overrides`）。注意覆盖用的是
> `--batch_size`，检测配置里的键却是 `batch`——覆盖只对时序的 `batch_size` 生效。

## 测试

```bash
cd framework && python -m pytest tests -q   # 需已激活项目 venv
```

- `test_temporal_metrics.py`：时序指标口径可独立测试（§12.3）。
- `test_detection_metrics.py`：检测指标三态组装（免 ultralytics）。
- `test_checkpoint_compat.py`：错配 checkpoint 拒绝加载。
- `test_envelope_matrix.py`：三态与矩阵机读/人读。
- `test_pipeline_smoke.py`：时序合成数据端到端 train→eval→matrix。
- `test_cross_vertical_matrix.py`：**两纵信封汇入单一异构矩阵**（拆分后的关键不变量守卫）。
- `test_detection_smoke.py`：检测纵 orchestration 端到端（注入假 adapter，免 ultralytics）。

> 说明：真实 `Endo_Project` 数据在本机为指向 Linux 路径的软链接、不可用，冒烟测试
> 用合成数据机械验证链路。**数值对齐验收**（新 realtime 指标 == 旧
> `benchmark/temporal_feed_mode` streaming）需在有真实数据的机器上执行。

## 扩展点

- 新增同架构变体：只改 `experiments/*.yaml` 的 `model` 段（§13.12）。
- 新增时序模型族（Transformer/causal-TCN）：加 `temporal/family/<name>.py`（网络+族契约同处
  一个自足文件）并在 `temporal/family/__init__.py` 登记；复用时序喂入与指标（§13.1/§13.2）。
  **族契约是纵内约定（build_network/prepare/forward/compute_loss/predict_frame_logits/
  checkpoint_meta），不是跨域 Protocol。** `prepare` 是训练前钩子（如离线分割 fit 输入归一化），
  无需时空操作即可（GRU 即空实现）。
- **离线双向分割**（MS-TCN 等，见 `temporal/family/mstcn.py`）：`feeding: full_sequence`——一次看到
  完整序列、逐帧监督、延迟标 N/A。与因果滑窗（GRU）**共用同一编排器**，差异只落在 feeding
  （`build_training_dataset`/`train_batch_size`/`requires_performance`）与 family（`prepare`/
  `forward`/`compute_loss`）两条多态轴上——编排器无 `if 模型类型` 分支，故不硬拆子纵。
- 新增检测器（DETR…）：在 `detection/adapter.py` 加适配器并在 `get_adapter` 登记，暴露
  `train`/`val` 即可，不需实现任何时序接口（§13.4）。
- 新增喂入模式：加 `temporal/feeding/<name>.py` 并在纵内 `get_feeding` 登记；喂入契约含
  `evaluate`/`requires_performance`/`train_batch_size`，可训练模式再实现 `build_training_dataset`；
  `stateful.py` 留占位（§11.4）。
