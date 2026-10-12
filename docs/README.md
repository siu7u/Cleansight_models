# docs 文档分类索引

> 本文件是 `docs/` 的**唯一分类导航**：新增文档后请在此登记（见文末「维护约定」）。
> 仓库根 [`README.md`](../README.md) 只保留最常用入口，完整清单以本文件为准。
> 分类原则：**入口层（顶层手册）→ 主题层（设计 / 数据 / 评测 / 特征）→ 报告层（系列 / 单次 / 周报 / 调研）**；
> 报告层内部按**时间倒序**，系列报告收进子目录。**状态标记**见文末约定，读到 `⚠️`/`❌` 先看更正声明。

## 0. 当前状态速览（2026-09-29）

| 问题 | 答案在 |
|---|---|
| 帧准确率现在到底是多少？ | [`frame-acc/`](experiments/frame-acc/README.md) 的 32-seed 定标：**53.90 → 58.31（+4.41pp，32/0）** |
| 改了哪些特征契约？为什么有效？ | [`features/README.md`](features/README.md)、[`weeks/2026-09-26_2026-10-02/TOPIC_03_04_DATA_AND_MECHANISM_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_03_04_DATA_AND_MECHANISM_20260929.md) |
| 哪些方向已经排除、不要重试？ | [`frame-acc/EXPERIMENT_REPORT_FRAME_ACC_SUMMARY_20260929.md`](experiments/frame-acc/EXPERIMENT_REPORT_FRAME_ACC_SUMMARY_20260929.md) §4（16 条轴） |
| 指标口径与评测纪律？ | [`EVAL.md`](EVAL.md) + [`weeks/…/TOPIC_09_11_METRIC_AND_OFFLINE_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_09_11_METRIC_AND_OFFLINE_20260929.md)（五条纪律） |
| 有哪些实验报告、结论各自是什么？ | [`experiments/README.md`](experiments/README.md)（类别索引，含状态标记） |
| 这周做了什么、哪项课题到哪一步？ | [`weeks/2026-09-26_2026-10-02/README.md`](weeks/2026-09-26_2026-10-02/README.md) + [`TOPIC_COVERAGE_11_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_COVERAGE_11_20260929.md) |
| 要给人讲 20 分钟？ | [`weeks/…/STAGE_REPORT_20260929.md`](weeks/2026-09-26_2026-10-02/STAGE_REPORT_20260929.md) |

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
| [`weeks/2026-09-26_2026-10-02/TOPIC_09_11_METRIC_AND_OFFLINE_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_09_11_METRIC_AND_OFFLINE_20260929.md) | **评测纪律与指标可信度（现行）**：`acc` 的口径与三条局限、噪声地板与最小可检测效应、`best_metric` 选择泄漏、**指标分歧（`acc` +4.59pp 而 `edit` −7.93pp）**、五条评测纪律、离线/滑窗适用边界 |

## 5. 特征方案（feature mapping）

| 文档 | 内容 |
|---|---|
| [`features/README.md`](features/README.md) | **特征契约唯一索引**：各方案的语义、维度、代码/登记/配置定位与实测结论；新增方案照检查清单执行 |
| [`features/INPUT_DESIGN_PROPOSAL.md`](features/INPUT_DESIGN_PROPOSAL.md) | 输入设计提案：体检探针 + 推荐方案 P1/P2 + 有证据否掉的方向 |
| [`features/INPUT_DESIGN_V3_REVIEW_20260924.md`](features/INPUT_DESIGN_V3_REVIEW_20260924.md) | 评审意见：《输入特征设计提案与 v3 验证计划》—— 状态核实、与既有实测证据的冲突、成功标准可判读性、集成风险与推进顺序 |
| [`features/IMAGE_FEATURE_TRAINING.md`](features/IMAGE_FEATURE_TRAINING.md) | 图像（像素级）特征训练流程与正式方案（ROI 网格 144 + 健康配方） |

## 6. 实验报告（`docs/experiments/`）

> **实验报告已归为一类**：全部 `EXPERIMENT_REPORT_*` 收进 [`experiments/`](experiments/README.md)，
> 攻关系列再收进 [`experiments/frame-acc/`](experiments/frame-acc/README.md)。
> 分类原则与状态标记见该目录的 [`README.md`](experiments/README.md)（含「什么算实验报告」的边界）。

| 位置 | 内容 |
|---|---|
| [`experiments/README.md`](experiments/README.md) | **实验报告类别索引**：系列 + 10 份专项报告（逐份一句话结论与状态）、与口径/周度/调研类文档的边界、维护约定 |
| [`experiments/frame-acc/README.md`](experiments/frame-acc/README.md) | 攻关系列导航与逐轮状态表（**R2 / R9 含更正声明**） |
| [`experiments/frame-acc/EXPERIMENT_REPORT_FRAME_ACC_SUMMARY_20260929.md`](experiments/frame-acc/EXPERIMENT_REPORT_FRAME_ACC_SUMMARY_20260929.md) | **12 轮浓缩总报告**（唯一有效杠杆 / 16 条已排除轴 / 4 处自我更正） |
| `experiments/*.md`（10 份） | 专项报告：段级 S-NCM 与段覆盖、架构调研、ROI 评估口径、TimesFM 探针、13 个精度杠杆、MS-TCN 容量、图像 E1、PlanB v3、ROI144 GPU |

## 7. 调研与可行性（文献 / 基础模型）

| 文档 | 一句话结论 |
|---|---|
| [`LITERATURE_SURVEY_OFFLINE_TAS_20260927.md`](LITERATURE_SURVEY_OFFLINE_TAS_20260927.md) | 离线（整段）TAS 架构综述：面向 144 维手工特征、小数据、**"整段认错"天花板**的可选方案与可行性判断 |
| [`literature_survey_streaming_tas_identity_20260927.md`](literature_survey_streaming_tas_identity_20260927.md) | 在线/流式 TAS 架构 + **段级标签同一性（segment identity）** 目标调研；约束与结论同上（同批、英文撰写） |
| [`EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md`](experiments/EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md) | TimesFM 时序基础模型可行性探针：**预测式预警路线证伪，不引入主线** |

## 8. 周工作目录（`docs/weeks/`）

| 位置 | 内容 |
|---|---|
| [`weeks/README.md`](weeks/README.md) | **周目录约定**：一周 = 周六~次周五；文件夹名 `YYYY-MM-DD_YYYY-MM-DD`；每份过程文档放当周目录 |
| [`weeks/2026-09-26_2026-10-02/README.md`](weeks/2026-09-26_2026-10-02/README.md) | **本周简报**：状态总览、按主题产出、本周更正过的结论、详细报告索引、遗留与下周计划、逐日工作日志（795 行） |
| [`weeks/2026-09-26_2026-10-02/STAGE_REPORT_20260929.md`](weeks/2026-09-26_2026-10-02/STAGE_REPORT_20260929.md) | **阶段性报告（可讲稿）**：技术同事向 20~30 分钟，含讲法、限定条件与 Q&A 预案 |
| [`weeks/2026-09-26_2026-10-02/TOPIC_COVERAGE_11_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_COVERAGE_11_20260929.md) | **11 项课题覆盖表**：逐项状态 / 一句话结论 / 证据、未做项与阻塞原因、跨课题六条判断、下一步 |
| [`weeks/2026-09-26_2026-10-02/TOPIC_01_ROI_SHIFT_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_01_ROI_SHIFT_20260929.md) | 课题①：镜像/取景偏移**双重否定**；附带 x 承重 / y 宽容 |
| [`weeks/2026-09-26_2026-10-02/TOPIC_02_AUGMENTATION_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_02_AUGMENTATION_20260929.md) | 课题②：序列级增强实测（jitter 中性 / tscale 有害 / 叠加更差） |
| [`weeks/2026-09-26_2026-10-02/TOPIC_03_04_DATA_AND_MECHANISM_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_03_04_DATA_AND_MECHANISM_20260929.md) | 课题③④：数据侧（误差预算、先验、增益来源）与原理侧机制（829 行，12 条【推断】） |
| [`weeks/2026-09-26_2026-10-02/TOPIC_05_08_ARCHITECTURE_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_05_08_ARCHITECTURE_20260929.md) | 课题⑤⑧：五架构同协议对照 → **架构不是杠杆**，契约收益补偿不了架构差距 |
| [`weeks/2026-09-26_2026-10-02/TOPIC_06_07_DETECTION_AND_ROI_THEORY_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_06_07_DETECTION_AND_ROI_THEORY_20260929.md) | 课题⑥⑦：检测不是瓶颈；ROI"方向成立、按格承载不成立"；`count` 净负、`max_area` 承重（571 行） |
| [`weeks/2026-09-26_2026-10-02/TOPIC_09_11_METRIC_AND_OFFLINE_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_09_11_METRIC_AND_OFFLINE_20260929.md) | 课题⑨⑪：指标可信度 + 五条评测纪律 + 离线/流式适用边界 |
| [`weeks/2026-09-26_2026-10-02/TOPIC_10_FOLD_20260929.md`](weeks/2026-09-26_2026-10-02/TOPIC_10_FOLD_20260929.md) | 课题⑩：划分稳健性（源域 30/30 折、目标批次 LOO 区间），增益非划分产物但分布不均 |

## 9. 周报归档

| 文档 | 内容 |
|---|---|
| [`WEEKLY_REPORT_20260924.md`](WEEKLY_REPORT_20260924.md) | 周报（09-19 ~ 09-24）：容量轴 / 特征杠杆轴 / TimesFM 探针三条线收官 |

## 10. 协作约定

| 文档 | 内容 |
|---|---|
| [`BRANCH_CONVENTION.md`](BRANCH_CONVENTION.md) | 分支体量、提交纪律与合并流程 |

## 11. 图表库

| 位置 | 内容 |
|---|---|
| [`figures/README.md`](figures/README.md) | **跨报告复用的图表库**（周报/汇报引用）：容量曲线 · 架构对照 · 特征契约缺口 · 选点口径 · TimesFM 探针 · 在线代价 · 图像 E1，共 7 张；每张配同名 JSON 旁证（所画数字 + 来源标注），生成脚本 [`tools/plot_report_figures.py`](../tools/plot_report_figures.py) |

> 报告专属插图可留在各自报告目录（先例 [`mstcn-capacity/figures/`](mstcn-capacity/figures/)、
> 系列报告可新建 `experiments/frame-acc/figures/`）；
> **跨报告复用**或**被周报引用**的图统一放 `figures/`。

---

## 维护约定

- **实验报告**命名：`EXPERIMENT_REPORT_<主题>_<YYYYMMDD>.md`；**周报**命名：`WEEKLY_REPORT_<YYYYMMDD>.md`。
- **同一主题的多轮报告**收进 `docs/<主题>/` 子目录（先例：[`frame-acc/`](experiments/frame-acc/README.md)、
  [`mstcn-capacity/`](mstcn-capacity/)），并在子目录内建 `README.md` 做**逐轮状态表**；
  移动报告时必须全量更新引用（`grep -rn <旧路径> docs usage README.md`）。
- 新增报告后：在本文件对应小节登记（含一句话结论），并回写相关索引——特征类同步
  [`features/README.md`](features/README.md)，指标口径类同步 [`EVAL.md`](EVAL.md)，
  涉及 YAML 的同步 [`usage/YAML_CONFIG.md`](../usage/YAML_CONFIG.md)，
  当周产生的过程文档同步当周 `weeks/<周>/README.md`。
- **报告状态标记**（写进报告顶部或本索引）：
  ✅ 现行/真源 · ⚠️ **含更正声明**（部分结论已被后续轮次推翻，须读声明） ·
  ❌ 已作废（仅存档） · 📌 仅历史对照。
- 新增/修改图表：放 [`figures/`](figures/README.md) 并在 §11 登记，图内数字必须能在报告里溯源
  （纪律见 [`figures/README.md`](figures/README.md) §3）。

## 归类待定（暂按当前判断归入，确认后可调整）

以下文档的归属存在两种合理读法，先放在当前分类，不影响使用：

- [`FEATURE_STRATEGY_COMPARE.md`](FEATURE_STRATEGY_COMPARE.md) —— 既是**逐轮实验记录**（属实验报告类，见 §6），
  也被当作**指标/特征口径参考**（可归 §4 评测口径）。
- [`TEMPORAL_DATASET_TRANSFORMATION_PLAN.md`](TEMPORAL_DATASET_TRANSFORMATION_PLAN.md) ——
  是**计划**（可归 §2 设计）也是**数据侧操作依据**（现归 §3 手册）。
- [`YOLO_WORK_SUMMARY.md`](YOLO_WORK_SUMMARY.md) —— 是**阶段总结报告**（§6）也可视为
  YOLO 线的入口说明（§3 手册）。
- [`literature_survey_streaming_tas_identity_20260927.md`](literature_survey_streaming_tas_identity_20260927.md) ——
  文件名小写与另一篇调研（大写）**不一致**，现按原名保留、同归 §7；如需统一命名请一次性改并登记。
