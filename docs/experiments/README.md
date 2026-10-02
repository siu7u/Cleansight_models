# 实验报告（`docs/experiments/`）

> **这一类放什么**：一切以「跑实验 → 得读数 → 下结论」为主体的报告——
> 轮次报告、专项实验、可行性探针。**不放**：口径/契约类规范（留在 `docs/` 顶层）、
> 手册与流程（`docs/` 与 `usage/`）、周度过程记录（`docs/weeks/`）、文献调研（`docs/*SURVEY*`）。
>
> 命名沿用仓库约定 `EXPERIMENT_REPORT_<主题>_<YYYYMMDD>.md`。
> **状态标记**：✅ 现行/真源 · ⚠️ 含更正声明（部分结论已被推翻，先读声明） · ❌ 已作废 · 📌 仅历史对照。

## 1. 攻关系列（子目录，12 轮 + 总报告）

| 位置 | 内容 |
|---|---|
| [`frame-acc/README.md`](frame-acc/README.md) | **系列导航与逐轮状态表**：R1~R12 逐轮一句话 + 状态、三条纵向结论、目录约定 |
| [`frame-acc/EXPERIMENT_REPORT_FRAME_ACC_SUMMARY_20260929.md`](frame-acc/EXPERIMENT_REPORT_FRAME_ACC_SUMMARY_20260929.md) | ✅ **系列入口（12 轮浓缩）**：目标与口径 / 最终定标 / 唯一有效杠杆 / 已排除 16 条轴 / 工程改动 / 4 处自我更正 / 复现 |
| `frame-acc/EXPERIMENT_REPORT_FRAME_ACC_{20260928,R2…R12_20260929}.md` | 逐轮报告；**R2 与 R9 顶部有更正声明**（被推翻的结论以 R3 / R10 为准） |

**这一系列的净结论**：帧级 `acc` like-for-like **53.90 → 58.31（+4.41pp，32 seed 配对 32/0）**；
唯一有效杠杆是**特征契约的维度预算 + 通道语义**；其余 16 条轴已排除
（架构 13 族、解码、校准、集成、正则、容量、轮数、调度、选点口径、更多数据、学习式表示、
原始框拼接、格内 dx/dy、布局×通道联合、镜像/偏移补偿、数据增强）。

## 2. 专项与单次实验报告（时间倒序）

| 文档 | 一句话结论 | 状态 |
|---|---|---|
| [`EXPERIMENT_REPORT_SEGMENT_NCM_20260927.md`](EXPERIMENT_REPORT_SEGMENT_NCM_20260927.md) | S-NCM 段级重标注（训练无关、零参数）在 4/4 个 `mstcn2` 臂上显著提高 `F1@0.1`（+3.4~+8.4，p≤0.0132）；并更正第 2 轮误判（`asformer cw0.03` 的"段召回 +33.3pp"是覆盖率口径被刷满的**退化解**） | ✅ 现行 |
| [`EXPERIMENT_REPORT_SEGMENT_COVERAGE_20260927.md`](EXPERIMENT_REPORT_SEGMENT_COVERAGE_20260927.md) | 换「动作段真的被检出」验收口径后配方重排：找到 `mstcn2 s2l5 h32 + cw0.03`（6.75 万参数 = 旗舰 1/49，段召回 +25.0pp，p=0.0070）；代价是 `edit` 50.17→41.84，**是取舍不是免费增益** | ✅ 现行 |
| [`EXPERIMENT_REPORT_ARCH_SURVEY_20260927.md`](EXPERIMENT_REPORT_ARCH_SURVEY_20260927.md) | 架构调研：旗舰 acc 低于平凡基线；模型在真值帧上给正确类的概率低至 0.005 → **瓶颈在"证据形成"而非编码器/解码器**；`asformer` 显著更差、硬时长 Viterbi 崩溃 | ✅ 现行 |
| [`EXPERIMENT_REPORT_ROI_EVAL_PROTOCOL_20260927.md`](EXPERIMENT_REPORT_ROI_EVAL_PROTOCOL_20260927.md) | ROI 分类评估口径体检：in-sample 读数不可引用（micro F1 −4.7pp、稀有类 −30.6pp） | ✅ 现行 |
| [`EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md`](EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md) | TimesFM 时序基础模型可行性探针：**预测式预警路线证伪，不引入主线** | ✅ 现行 |
| [`EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`](EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md) | 13 个精度杠杆 × 多 seed：只有 4 条有效（`mstcn2` / h128 / roi-144 / `best_metric=val_edit`） | ✅ 现行 |
| [`EXPERIMENT_REPORT_MSTCN_CAPACITY_20260922.md`](EXPERIMENT_REPORT_MSTCN_CAPACITY_20260922.md) | MS-TCN 容量研究：加参数默认配方无用，换结构比加参数值 | ✅ 现行 |
| [`EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md`](EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md) | 图像 embedding E0/E1 机制床：图像通道**不是免费增益**（帧级指标显著退化） | ✅ 现行 |
| [`EXPERIMENT_REPORT_PLANB_V3_20260905.md`](EXPERIMENT_REPORT_PLANB_V3_20260905.md) | 方案 b v3 正式轮实验报告 | 📌 历史对照 |
| [`EXPERIMENT_REPORT_ROI144_GPU_20260904.md`](EXPERIMENT_REPORT_ROI144_GPU_20260904.md) | ROI 144 维 GPU 复跑实验报告 | 📌 历史对照 |

> 容量研究的**主体合并文档**在 [`../mstcn-capacity/MSTCN_CAPACITY_STUDY.md`](../mstcn-capacity/MSTCN_CAPACITY_STUDY.md)
> （含图），本目录的 `MSTCN_CAPACITY_20260922` 是其当轮报告。

## 3. 相关但不在本目录

| 位置 | 为什么不算「实验报告」 |
|---|---|
| [`../FEATURE_STRATEGY_COMPARE.md`](../FEATURE_STRATEGY_COMPARE.md) | 特征策略**逐轮机制记录 + 全部原始对照数字**，被当口径参考反复引用（跨度大，跨报告性质） |
| [`../../usage/YAML_CONFIG.md`](../../usage/YAML_CONFIG.md) | 受跟踪 YAML 的**配置索引**（规范类） |
| [`../EVAL.md`](../EVAL.md) | 指标**口径定义**（规范类） |
| [`../WEEKLY_REPORT_20260924.md`](../WEEKLY_REPORT_20260924.md) | 周报（`docs/weeks/` 约定生效前的旧周报，按周度记录归此） |
| [`../LITERATURE_SURVEY_OFFLINE_TAS_20260927.md`](../LITERATURE_SURVEY_OFFLINE_TAS_20260927.md) · [`../literature_survey_streaming_tas_identity_20260927.md`](../literature_survey_streaming_tas_identity_20260927.md) | 文献调研（不产新实验读数） |

## 4. 维护约定

- 新增实验报告 → 放本目录，命名 `EXPERIMENT_REPORT_<主题>_<YYYYMMDD>.md`，
  并在 [`../README.md`](../README.md) §6 与当周 `docs/weeks/<周>/README.md` 登记（含一句话结论与状态标记）。
- **同一主题多轮** → 在本目录下建子目录 + 子目录内 `README.md` 做逐轮状态表（先例 [`frame-acc/`](frame-acc/README.md)）。
- 移动/重命名报告 → 必须全量更新引用：`grep -rn "<旧文件名>" docs usage tools framework tmp README.md`，
  并跑一次相对链接检查（见 [`../README.md`](../README.md) 维护约定）。
