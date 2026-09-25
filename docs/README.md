# docs 文档分类索引

> 本文件是 `docs/` 的**唯一分类导航**：新增文档后请在此登记（见文末「维护约定」）。
> 仓库根 [`README.md`](../README.md) 只保留最常用入口，完整清单以本文件为准。
> 文档按**主题**分组；实验报告与周报按**时间倒序**排列。

## 1. 先看这里（入口）

| 文档 | 内容 |
|---|---|
| [`MODELSET_OVERVIEW.md`](MODELSET_OVERVIEW.md) | 模型集现状、使用入口与汇报要点（合并原 STATUS / PRESENTATION / USAGE 三份） |
| [`PROJECT_FLOW.md`](PROJECT_FLOW.md) | 数据 → 训练 → 评测 → 交付的流程关系 |
| [`TEAM_GUIDE.md`](TEAM_GUIDE.md) | 组员上手：clone 后 5 分钟跑通第一个训练 |
| [`../README.md`](../README.md) | 仓库总入口：职责划分、完整架构、文档索引 |

## 2. 架构与设计

| 文档 | 内容 |
|---|---|
| [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) | 用目录和数据流快速说明当前仓库结构 |
| [`DESIGN.md`](DESIGN.md) | 设计准则：`framework` / `benchmark` 的职责与抽象边界 |
| [`TRAIN_EVAL_REQUIREMENTS.md`](TRAIN_EVAL_REQUIREMENTS.md) | 训练与评测的需求定义 |
| [`TRAIN_EVAL_IMPLEMENTATION_STATUS.md`](TRAIN_EVAL_IMPLEMENTATION_STATUS.md) | 实现状态：当前能力与剩余事项 |

## 3. 数据、标注与训练手册

| 文档 | 内容 |
|---|---|
| [`DATASET_BUILDING_GUIDE.md`](DATASET_BUILDING_GUIDE.md) | Label Studio 建数据的硬性契约与操作步骤（只标动作、框由 YOLO 自动标注） |
| [`AUTO_ANNOTATION.md`](AUTO_ANNOTATION.md) | 自动标注完整指南：视频 → 标注 JSON → 时序训练数据（run / run-dataset / convert / 优化参数 / 可视化 / 质量报告） |
| [`AUTO_ANNOTATION_QUICKSTART.md`](AUTO_ANNOTATION_QUICKSTART.md) | 自动标注最小命令集，5 分钟跑通与常见报错排查 |
| [`MODEL_ONBOARDING.md`](MODEL_ONBOARDING.md) | 新模型接入手册：新增时序网络或检测器要改哪几处 |
| [`YOLO_OPTIMIZATION.md`](YOLO_OPTIMIZATION.md) | YOLO 优化工作流：sweep / analyze / 特征融合 |
| [`YOLO_REVIEW_FLOW.md`](YOLO_REVIEW_FLOW.md) | YOLO 结果人工审核闭环：预标注 → 人工改框+标动作 → 导出 → convert |
| [`TEMPORAL_DATASET_TRANSFORMATION_PLAN.md`](TEMPORAL_DATASET_TRANSFORMATION_PLAN.md) | 时序数据集改造计划（分阶段，含 Phase 0 验收项） |
| [`../usage/YAML_CONFIG.md`](../usage/YAML_CONFIG.md) | **所有受跟踪 YAML 的文档索引**（内容 / 读取方 / 快速定位） |
| [`../usage/TEST_COMMANDS.md`](../usage/TEST_COMMANDS.md) | 评测、timeline、矩阵与 pytest 的常用命令行 |
| [`../usage/FEATURE_EVAL_RUNBOOK.md`](../usage/FEATURE_EVAL_RUNBOOK.md) | 特征方案评测实操顺序（数据更新 → 登记 → 口径 → 矩阵 → 门禁）与常见陷阱 |

## 4. 评测口径与性能

| 文档 | 内容 |
|---|---|
| [`EVAL.md`](EVAL.md) | 指标定义、聚合口径、完整性检查与预测可视化（含 §3.4 指标注册表） |
| [`INFERENCE_CHAIN_PERF.md`](INFERENCE_CHAIN_PERF.md) | 推理链路实测时延，支撑「预计算入数据集 vs 现场推理」决策 |

## 5. 特征方案（feature mapping）

| 文档 | 内容 |
|---|---|
| [`features/README.md`](features/README.md) | **特征契约唯一索引**：各方案的语义、维度、代码/登记/配置定位与实测结论；新增方案照检查清单执行 |
| [`features/INPUT_DESIGN_PROPOSAL.md`](features/INPUT_DESIGN_PROPOSAL.md) | 输入设计提案：体检探针 + 推荐方案 P1/P2 + 有证据否掉的方向 |
| [`features/INPUT_DESIGN_V3_REVIEW_20260924.md`](features/INPUT_DESIGN_V3_REVIEW_20260924.md) | 评审意见：《输入特征设计提案与 v3 验证计划》—— 状态核实、与既有实测证据的冲突、成功标准可判读性、集成风险与推进顺序 |
| [`features/IMAGE_FEATURE_TRAINING.md`](features/IMAGE_FEATURE_TRAINING.md) | 图像（像素级）特征训练流程与正式方案（ROI 网格 144 + 健康配方） |

## 6. 实验报告与周报（时间倒序）

| 文档 | 一句话结论 |
|---|---|
| [`WEEKLY_REPORT_20260924.md`](WEEKLY_REPORT_20260924.md) | 周报（09-19 ~ 09-24）：容量轴 / 特征杠杆轴 / TimesFM 探针三条线收官 |
| [`EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md`](EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md) | TimesFM 时序基础模型可行性探针：**预测式预警路线证伪，不引入主线** |
| [`EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`](EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md) | 13 个精度杠杆 × 多 seed：只有 4 条有效（`mstcn2` / h128 / roi-144 / `best_metric=val_edit`） |
| [`EXPERIMENT_REPORT_MSTCN_CAPACITY_20260922.md`](EXPERIMENT_REPORT_MSTCN_CAPACITY_20260922.md) | MS-TCN 容量研究：加参数默认配方无用，换结构比加参数值 |
| [`mstcn-capacity/MSTCN_CAPACITY_STUDY.md`](mstcn-capacity/MSTCN_CAPACITY_STUDY.md) | 容量研究主体文档（合并原 5 份），含参数量、容量曲线与学习曲线（[目录说明](mstcn-capacity/README.md)） |
| [`EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md`](EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md) | 图像 embedding E0/E1 机制床：图像通道**不是免费增益**（帧级指标显著退化） |
| [`EXPERIMENT_REPORT_PLANB_V3_20260905.md`](EXPERIMENT_REPORT_PLANB_V3_20260905.md) | 方案 b v3 正式轮实验报告 |
| [`EXPERIMENT_REPORT_ROI144_GPU_20260904.md`](EXPERIMENT_REPORT_ROI144_GPU_20260904.md) | ROI 144 维 GPU 复跑实验报告 |
| [`FEATURE_STRATEGY_COMPARE.md`](FEATURE_STRATEGY_COMPARE.md) | 特征策略逐轮机制记录（第十三~二十七轮）与全部原始对照数字 |
| [`YOLO_WORK_SUMMARY.md`](YOLO_WORK_SUMMARY.md) | YOLO 线工作总结 |

## 7. 协作约定

| 文档 | 内容 |
|---|---|
| [`BRANCH_CONVENTION.md`](BRANCH_CONVENTION.md) | 分支体量、提交纪律与合并流程 |

## 8. 图表库

| 位置 | 内容 |
|---|---|
| [`figures/README.md`](figures/README.md) | **跨报告复用的图表库**（周报/汇报引用）：容量曲线 · 架构对照 · 特征契约缺口 · 选点口径 · TimesFM 探针 · 在线代价 · 图像 E1，共 7 张；每张配同名 JSON 旁证（所画数字 + 来源标注），生成脚本 [`tools/plot_report_figures.py`](../tools/plot_report_figures.py) |

> 报告专属插图可留在各自报告目录（先例 [`mstcn-capacity/figures/`](mstcn-capacity/figures/)）；
> **跨报告复用**或**被周报引用**的图统一放 `figures/`。

---

## 维护约定

- **实验报告**命名：`EXPERIMENT_REPORT_<主题>_<YYYYMMDD>.md`；**周报**命名：`WEEKLY_REPORT_<YYYYMMDD>.md`。
- 新增报告后：在本文件 §6 登记（含一句话结论），并在相关索引回写——特征类同步
  [`features/README.md`](features/README.md)，指标口径类同步 [`EVAL.md`](EVAL.md)，
  涉及 YAML 的同步 [`usage/YAML_CONFIG.md`](../usage/YAML_CONFIG.md)。
- 多文件报告（含图/数据）自建子目录（先例：[`mstcn-capacity/`](mstcn-capacity/)），并在 §6 登记其主文档。
- 新增/修改图表：放 [`figures/`](figures/README.md) 并在 §8 登记，图内数字必须能在报告里溯源
  （纪律见 [`figures/README.md`](figures/README.md) §3）。

## 归类待定（暂按当前判断归入，确认后可调整）

以下文档的归属存在两种合理读法，先放在当前分类，不影响使用：

- [`FEATURE_STRATEGY_COMPARE.md`](FEATURE_STRATEGY_COMPARE.md) —— 既是**逐轮实验记录**（§6 报告），
  也被当作**指标/特征口径参考**（可归 §4 评测口径）。
- [`TEMPORAL_DATASET_TRANSFORMATION_PLAN.md`](TEMPORAL_DATASET_TRANSFORMATION_PLAN.md) ——
  是**计划**（可归 §2 设计）也是**数据侧操作依据**（现归 §3 手册）。
- [`YOLO_WORK_SUMMARY.md`](YOLO_WORK_SUMMARY.md) —— 是**阶段总结报告**（§6）也可视为
  YOLO 线的入口说明（§3 手册）。
