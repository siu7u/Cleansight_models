# 图像特征训练接入流程说明

> **状态标记**：本文档是**流程说明与规划框架**，不是已实现能力的清单。截至 2026-09，
> 仓库内与"图像（像素级）"相关的训练只有独立的 ROI 图像分类流水线（`roi_classification`，
> 尚未正式训练）；**像素级特征作为时序模型输入尚无实现**。文中每节都标注了现状与缺口。

## 1. 术语界定：仓库中"图像特征"的两种形态

| 形态 | 定义 | 现状 |
|---|---|---|
| **A. 独立图像分类** | 对 ROI 图像块做多标签分类（`roi_classification` / `feature_fusion`），输出独立预测，**不进入时序输入** | 代码齐备（pipeline/配置/数据裁剪/评估器），**从未正式训练**，exploratory 占位 |
| **B. 像素特征进时序** | 每帧 CNN embedding（或 ROI embedding）作为时序模型输入的一部分（`[B,T,F]` 中的像素派生通道） | **无实现、无数据支持、无文档流程**——本文档主体 |

区分关键：A 的"图像分类"与 B 的"图像**特征**"是两条不同的训练链路，不要混用术语。

## 2. 共同前提：图像数据源（当前最大的阻塞点）

**v3 auto 数据集没有图片**（实测目录结构仅 `labels/` + `frames/` + `task_ids.yaml`）——
而我们现有的全部 bbox 系特征实验（40/80/144 维）都基于 v3。接图像特征前必须先回答"图从哪来"：

| 候选图像源 | 状态 | 对齐要求 |
|---|---|---|
| 原始 project-16 视频抽帧 | 视频在录制方/队友处，仓库无 | v3 `frames/` 帧号 = 原始视频 **stride-4 抽样帧号** → 抽帧后逐帧对齐即可挂到现有 manifest |
| 手动通道 `cleansight-ActionMixed/images/`（606MB） | 仓库本地有 | 与 v3 **不同源不同视频**——只能做手动通道（actionmixed-v2）实验，**禁止与 v3 混合** |
| YOLO 数据集 `cleansight-yolo/images/` | 仓库本地有 | 检测帧与 v3 时序帧号不对齐，仅可做检测侧实验 |

**决策记录要求**：选定图像源后，需要登记新的数据集条目（data_root/images 目录）、帧对齐校验
（每帧文件名 ↔ manifest 帧号一一对应）与 revision——沿用 catalog 契约体系，不能绕过。

## 3. 形态 A：ROI 图像分类（roi_classification）正式化流程

现状：`roi-fusion.yaml`（exploratory）+ 数据自动裁剪（`classification/data.py` 从 YOLO GT 框
裁 ROI + 随机背景负样本）+ 评估器已注册；**从未跑过正式训练**。

正式化步骤：

1. 确定淘汰类清单：`python -m benchmark.cli.analyze --config yolo-clean-small.yaml --ckpt <best.pt>`
   （YOLO 逐类 P/R < 0.3 的类，参考 `docs/YOLO_OPTIMIZATION.md` §4）
2. 填 `roi-fusion.yaml` 的 `data.classes`（实际淘汰类）与 `data.group_dir`
3. 冒烟训练：`cli.train --config framework/experiments/roi-fusion.yaml -S train.epochs=1`；
   数据裁剪自动构建并缓存到 `runs/feature_fusion/datasets/`
4. 探索性评估：`benchmark.cli.eval --config roi-fusion.yaml --ckpt <best.pt>`
5. 淘汰类与口径稳定后**登记正式 testset**（以裁剪数据集的固定划分为 manifest）→
   `evaluation.mode: formal` 重训重评
6. 按 modelset-quality 规则登记 registry（CARD.md + pin.yaml + 评测报告）

## 4. 形态 B：像素特征进时序训练（规划流程）

在 [`README.md`](./README.md) 第 3 节"新增方案检查清单"（9 步，针对 bbox 文本派生特征）基础上，
图像特征多出以下步骤与决策点：

### 4.1 图像源与帧对齐（新增前置步骤，阻塞一切）

- 选定图像源（见第 2 节）→ 抽帧/对齐工具 → 产出与 train/val/test manifest **一一对应**的帧文件
- 校验：每个 manifest 视频的每个标签帧号都必须存在对应帧文件（镜像 `validate_testsets.py` 的
  bbox 逐帧检查逻辑）

### 4.2 特征提取策略决策（形态 B 的核心分叉）

| 策略 | 做法 | 优点 | 代价 |
|---|---|---|---|
| **离线预计算 embedding**（推荐） | 训练/评估前批量跑 CNN，逐帧 embedding 存为特征文件（如 npy/npz），随数据集登记 | 训练快、可复用、与 v3"无图分发"哲学一致；后续特征契约沿用现有 catalog 体系 | 数据集变大；embedding 契约变化要升版本重算 |
| **在线提取** | 训练时读帧 → CNN → embedding | 可做图像级数据增强 | 图像需随数据分发（体积大）；训练慢；**部署端必须复刻同一图像管线** |

### 4.3 recipe 实现

- 新建 `framework/cleansight_eval/temporal/features/image_*.py`：帧加载 → 预处理 →
  backbone 前向（可复用 `classification/model.py` 的 `BACKBONE_CONFIGS` 与构建逻辑）→ embedding
- 因果性红线：帧级 CNN 只允许看当前帧（单帧无状态 → 因果 ✓）；**禁止用未来帧或跨帧聚合**
- 归一化与预处理（如 ImageNet mean/std）必须写进 feature mapping 契约与版本说明
- 确定性：推理路径禁随机（随机增强只允许出现在训练期且要可复现）

### 4.4 与 bbox 特征的关系（决策点，先对照后融合）

建议顺序：① 先单独训练"纯图像特征"时序模型，与现有 40 维 bbox 基线对照（证明图像通道本身
有效）→ ② 再拼接到 bbox 特征做多模态输入（`feature_blocks` 机制已支持多块声明）→ ③ 根据结果
决定是否保留。不要一步到位做双流模型。

### 4.5 登记、配置、测试、门禁、对照

沿用现有 9 步清单的第 3~9 步（testsets 登记 / catalog 校验 / 配置 / 单测 / validate 门禁 /
正式对照 / 文档同步），差异点：
- 图像数据条目的 manifest 换为帧文件清单或沿用原 manifest + 帧对齐校验
- 单元测试需覆盖：帧缺失、图像解码失败、embedding 维度、确定性（同帧两次提取同值）

### 4.6 部署影响（重大架构决策，需提前知会）

CleanSightBackend 目前消费的是**检测框文本派生特征**（推理时只需 YOLO 检测结果，无像素管线）。
形态 B 一旦上线，后端需要新增"像素 → CNN embedding"推理管线（读帧/解码/预处理/backbone），
并保证与训练期提取口径一致。**这一条不成立则形态 B 只能停留在离线评测。**

## 5. 收益验证纪律（来自既有实验的教训）

- 队友 ROI 实测：通用预训练 ROI embedding 对下游提升很小（tracker IDF1 0.4755→0.4806），
  结论"若用于动作模型，必须按动作指标重新评估"——**图像特征接入后必须用 Frame-F1 /
  Segment-F1 对照，禁止只报 acc**（acc 在 65%+ idle 数据上具有欺骗性，见坍缩分析）
- 训练配方必须健康（短训/正则/早停），否则对照结论被过拟合污染（详见
  [`FEATURE_STRATEGY_COMPARE.md`](../FEATURE_STRATEGY_COMPARE.md) 的坍缩分析）

## 6. 执行顺序建议

1. 决策图像源（需要原始视频或确认用手动通道）——**卡点，先解决**
2. 写抽帧/对齐工具 + 校验脚本
3. 按 4.2 选离线预计算 → 4.3 recipe → 4.5 登记链路
4. 按 4.4 先做"纯图像 vs bbox 基线"单路对照
5. 有效再谈拼接与部署（4.6）
