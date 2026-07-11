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

## 用法（用仓库 venv）

```bash
PY=../CleanSightBackend/.venv/bin/python

# 训练（配置驱动；可用 --epochs/--lr 等临时覆盖）
cd framework && PYTHONPATH=. $PY -m cleansight_eval.cli.train --config experiments/gru-actionmixed.yaml

# 评估（与训练同一喂入模式，出一份三态信封）
PYTHONPATH=. $PY -m cleansight_eval.cli.eval --config experiments/gru-actionmixed.yaml --ckpt <runs/.../checkpoints/xxx.pt>

# 汇总评估矩阵（matrix.json 机读 + matrix.md 人读）
PYTHONPATH=. $PY -m cleansight_eval.cli.matrix --runs runs
```

## 测试

```bash
cd framework && ../CleanSightBackend/.venv/bin/python -m pytest tests -q
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
- 新增时序模型族（Transformer/causal-TCN）：加 `temporal/family/<name>.py` 并在
  `temporal/family/__init__.py` 登记；复用时序喂入与指标（§13.1/§13.2）。**族契约是纵内约定
  （build_network/forward/loss/predict/checkpoint_meta），不是跨域 Protocol。**
- 新增检测器（DETR…）：在 `detection/adapter.py` 加适配器并在 `get_adapter` 登记，暴露
  `train`/`val` 即可，不需实现任何时序接口（§13.4）。
- 新增喂入模式：加 `temporal/feeding/<name>.py` 并在纵内 `get_feeding` 登记；`stateful.py`
  留占位（§11.4）。
