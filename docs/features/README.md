# 可用特征提取方案索引（时序模型输入）

> 本目录是**特征提取方案（feature mapping）的唯一索引**：记录每个已实现/已登记的契约的
> 语义、维度、代码位置、数据登记、训练配置与实测结论，供后续方案设计与横向对比参考。
> **新增或修改方案时必须同步更新本索引**（见文末「新增方案检查清单」）。
> **要动手跑一轮评测**（数据更新 → 登记 → 口径 → 矩阵 → 门禁）见
> [`../usage/FEATURE_EVAL_RUNBOOK.md`](../usage/FEATURE_EVAL_RUNBOOK.md)（实操手册，含常见陷阱）。
>
> 图像（像素级）特征训练流程见 [`IMAGE_FEATURE_TRAINING.md`](./IMAGE_FEATURE_TRAINING.md)：
> 正式训练方案已定稿（ROI 网格 144 + 健康配方，§3.4）；像素特征进时序（形态 B）的**提取工具
> 与训练侧接入均已落地**（§3.5），机制床 E0/E1 对照见本目录 §2.1；独立 ROI 图像分类已移出
> 训练主线（D7）。
>
> 输入设计方向的**提案**（未跑训练对照、不含新模型指标）见
> [`INPUT_DESIGN_PROPOSAL.md`](./INPUT_DESIGN_PROPOSAL.md)：体检探针
> `tools/probe_input_features.py` + 本周推荐方案 P1/P2 + 有证据否掉的方向。

## 0. 总览

| 契约版本（feature_mapping） | 维度 | 提取范围 | 语义要点 | 代码 | 数据集登记 | 训练配置 |
|---|---|---|---|---|---|---|
| `actionmixed-bbox-8cls-v1` | 40 | 整个画面 | 每类取面积最大框 `[presence,cx,cy,w,h]` | `temporal/data.py` `featurize_frame_bbox` | `temporal.actionmixed-auto-v3` | `gru/mstcn/transformer-actionmixed-auto.yaml` |
| `actionmixed-roi-grid-v1` | 144 | 整个画面（2×3 网格） | 每 (类,区域) 统计 `[presence,count,max_area]` | `features/roi_bbox.py` | `temporal.actionmixed-auto-roi-v1` | `*-actionmixed-auto-roi.yaml` |
| `actionmixed-roi-grid-v2` | 96 | 整个画面（**可见性重排**：高频 3 类 3×3、低频 5 类全局 1 区） | 每类块宽不等（27/3），通道语义同 v1 | `features/roi_bbox_v2.py` | `temporal.actionmixed-auto-roi-v2` | `gru-actionmixed-auto-roi-v2.yaml` |
| `actionmixed-roi-grid-presence-v1` | 48 | 整个画面（2×3 网格，**只留 presence**） | roi-v1 的通道 0 切片：8 类 × 6 区域的存在性平面；丢弃 `count` / `max_area` | `features/roi_bbox.py` `build_roi_presence_frame_features` | `temporal.actionmixed-auto-roi-presence-v1` | 复用 `mstcn-actionmixed-auto-roi.yaml` + `-S` 覆盖（见 FEATURE_STRATEGY_COMPARE 第十八轮） |
| `actionmixed-bbox-hand-8cls-v1` | 40 | 仅手部周围 | 只编码 hand 框扩张 1.5 倍区域内的框，坐标相对区域归一化；无 hand 全零 | `features/hand_bbox.py` | `temporal.actionmixed-auto-hand-v1` | `gru-actionmixed-auto-hand.yaml` |
| `actionmixed-bbox-global-hand-8cls-v1` | 80 | 全局 + 手部 | 全局 40 维与手部 40 维拼接（每类 10 维块） | `features/hand_bbox.py` + `data.py` 拼接 | `temporal.actionmixed-auto-global-hand-v1` | `gru-actionmixed-auto-global-hand.yaml` |
| `actionmixed-bbox-embed-mbv3s-v1` | 616 | 整个画面（bbox）+ 逐帧整图 | 左 40 维 bbox 块（同 1.1）+ 右 576 维冻结 mobilenet_v3_small 整图 embedding；进 GRU 前经 64 维线性投影头；`mask_targets` 只作用 bbox 块 | `features/image_embed.py` + `models/gru.py`（投影头） | `temporal.actionmixed-v2-embed-mbv3s-v1` | `gru-actionmixed-embed.yaml` |
| `legacy-20d-v1` | 20 | —（历史预存特征） | Endo Project npy 特征，仅评测兼容 | `temporal/data.py` `_load_legacy_endo_split` | `temporal.endo-project-v1` | `legacy-*.yaml` |
| `clean_bbox_v2_*` 族（113/121/249） | 113/121/249 | 整个画面 | CLEAN 离线模型特征（含速度/业务先验），exploratory 评测用 | `features/clean_bbox_v2.py` | 未登记（外部 checkpoint 配套） | `external_checkpoints/*.yaml` |

公共语义（所有 bbox 系契约）：因果、无状态、逐帧独立计算；空 bbox 文件 → 全零；
`feature_schema.mask_targets` 按检测类整块清零（块宽随契约不同：bbox 5 / ROI-v1 18 / ROI-presence 6 / ROI-v2 27 或 3 / 全局+手部 10）；
目标遮罩增强的块宽默认由特征维 ÷ 检测类数推导，**块宽不等的契约（roi-grid-v2）必须走
`features.block_dims_for_version` 的显式块宽**（否则静默遮错列，见 INPUT_DESIGN_PROPOSAL §4）。

## 1. 各方案详述

### 1.1 `actionmixed-bbox-8cls-v1`（40 维，全局基线）

- **布局**：8 类 × 5 维，类顺序同 `frames/data.yaml`（hand, scope_control_body, scope_mid_section,
  scope_distal_end, syringe, air_gun, short_brush, brush_tip_out）。
- **编码**：每类取该帧**面积最大**的一个框 → `[presence, cx, cy, w, h]`（YOLO 归一化坐标）；缺席全零。
- **定位**：作为所有空间策略的对照基线；`FEATURE_STRATEGY_COMPARE.md` 中实测段级 edit/F1 最好。
- **代价**：每类每帧只保留一个框，丢失多目标数量与空间分布。

### 1.2 `actionmixed-roi-grid-v1`（144 维，ROI 空间分区）

- **布局**：8 类 × 6 区域（2×3 网格，行优先）× 3 通道 = 144 维，class-major（每类 18 维）。
- **通道**：`[presence, count, max_area]`；坐标按框中心落入网格区域归类，越界钳制。
- **动机**：显式编码空间分布、支持多目标（count）、抗检测框抖动（位置量化到区域粒度）。
- **注意**：区域划分是契约的一部分，改网格/通道 = 新版本必须重训；稀有类若检测不到，
  对应通道恒零，特征改造不解决检测召回问题。
- **状态**：正式训练 3 seed 已跑通——CPU 轮（2026-09-03，`runs/formal_roi/`）与 GPU 轮
  （2026-09-04，`runs/formal_roi_20260905/`，数据根 `-lhh`）均三 seed 零坍缩；中位指标与
  设备口径见 `IMAGE_FEATURE_TRAINING.md` §3.3/§3.4。

### 1.2.1 `actionmixed-roi-grid-v2`（96 维，可见性重排 ROI）

- **动机**：v1 按类别表整齐分配（每类 18 维），但实测 54/144 维恒零（38%）；高频 3 类
  （hand / scope_control_body / scope_mid_section，出现率 95.2/91.4/69.8%）的空间信息被
  低频类稀释，而低频类散在 6 个区域后每格几乎恒零（依据见 `INPUT_DESIGN_PROPOSAL.md`
  §1.1 覆盖、§1.3 通道体检、§2 P2b）。
- **布局**：高频 3 类用 3×3 网格（每类 27 维），低频 5 类（scope_distal_end / syringe /
  air_gun / short_brush / brush_tip_out）合并为**全局 1 区域**（每类 3 维）→ 8 类共 96 维；
  class-major，通道语义与区域编号规则同 v1。
- **实测特征密度**（新数据：train 9,575 / val 3,384 / test 2,639 帧）：恒零通道
  train 12/96（12%）、val 27/96（28%）、test 15/96（16%）；v1 同口径为 33%/48%/46%。
  低频 5 类在 train 上由 v1 的稀疏格子变为 3/3 全活跃，非零密度 0.103 vs v1 0.065。
- **实测指标**（2026-09-18，CPU，新 test = project-18 八视频，3 seed 中位数）：acc 53.09 /
  edit 17.54 / F1@0.1 25.88 / F1@0.25 14.29；v1(144) 同口径为 52.97 / 19.89 / 25.58 / 13.64
  —— **维度减到 2/3 不损指标，也没有换来泛化增益**；完整矩阵见
  [`../FEATURE_STRATEGY_COMPARE.md`](../FEATURE_STRATEGY_COMPARE.md) 第五轮。
- **实现要点**：块宽不等（高频 27 / 低频 3），catalog 用 `feature_layout.groups` 校验总维度，
  遮罩与随机遮罩必须按 `features.roi_grid_v2_block_dims` 的显式块宽切片；单元测试
  `framework/tests/test_roi_features_v2.py`。
- **参数**：分组与网格为 v2 固定值；改分组/网格/通道数 = 新 feature mapping 版本。

### 1.3 `actionmixed-bbox-hand-8cls-v1`（40 维，仅手部周围）

- **锚点**：本帧面积最大的 hand 框（检测类 ID 0），绕中心扩张 `HAND_REGION_EXPAND=1.5` 倍并
  钳制到画面内，得手部区域。
- **编码**：只取**中心落入手部区域**的框，每类最大框 → `[presence, cx_rel, cy_rel, w_rel, h_rel]`，
  坐标/尺寸相对区域归一化；**无 hand 帧全零**（val 实测占 4.9%）。
- **实测结论**（`FEATURE_STRATEGY_COMPARE.md`）：整体最弱——scope 类在手部区域内 presence
  仅 14%（全局 85%），动作判别依赖区域外上下文。
- **参数**：扩张倍数 1.5 为 v1 固定值；多 hand 时取最大框，其余手不参与锚定。

### 1.4 `actionmixed-bbox-global-hand-8cls-v1`（80 维，全局+手部双通道）

- **布局**：左 40 维 = 全局编码（1.1），右 40 维 = 手部编码（1.3），每类 10 维块
  （catalog 侧用 `feature_blocks: 2` 校验维度）。
- **实测结论**：帧级 acc 明显领先基线（val 56.4 vs 22.5 / test 59.8 vs 50.3），段级指标待多
  seed 复跑确认；推理延迟与 40 维无实质差异。
- **实现**：`data.py` 中按帧拼接两个 recipe，无独立特征代码。

### 1.5 `legacy-20d-v1`（历史 20 维）

- Endo Project 时代预存 `features/*.npy` + `groundTruth/` + `mapping.txt`；仅框架兼容加载，
  不再有训练入口。三类（Idle/Long_Brushing/Short_Brushing）。

### 1.6 CLEAN 离线特征族（113/121/249 维）

- 迁自 CleanSightBackend 的 CLEAN offline segmenter 数学口径（`features/clean_bbox_v2.py`），
  含速度特征、业务先验（pair features）与居中窗口统计；只供外部裸 checkpoint 的 exploratory
  评测，不冒充真实检测置信度下的正式结果。

### 1.7 `actionmixed-bbox-embed-mbv3s-v1`（616 维，bbox + 图像 embedding，形态 B）

- **布局**：左 40 维 bbox 块（口径同 1.1）× 8 类 × `[presence, cx, cy, w, h]`；右 576 维
  逐帧整图 mobilenet_v3_small embedding（`extract_embeddings.py` 离线预计算，backbone 冻结）。
- **对齐与因果**：按标签行**位置**对齐（第 i 个标签帧 ↔ npy 第 i 行）；行数/列数不符立即报错
  （陈旧产物错位比训练失败更贵）；每帧只依赖当前帧图像与当前帧检测框，因果、无状态。
- **产物根目录可换**：`data.embedding_root` 由 catalog 的 `feature_embed.root` 注入
  （`{root, feat_dim, backbone}`）；换 backbone / 换数据源 = 换登记条目，不改代码；
  embedding 维度变化 = 新 feature mapping 版本。
- **投影头**：进 GRU 前经 `model.image_proj_dim`（E1 = 64）线性投影再拼接，是形态 B 唯一
  新增的可训练参数（`models/gru.py`；非 GRU 架构声明 `image_dim` 直接报错）。
- **遮罩语义**：`mask_targets` 与 train 期目标随机遮罩只作用于 bbox 块（图像块无检测类结构，
  按总维度 ÷ 类数推导会错切）。
- **状态**：机制床（人工标注 `cleansight-ActionMixed`，9,532 帧，576 维 embedding 已预计算）
  E0 vs E1 多 seed 对照见 §2.1。

## 2. 横向对比结论速查（截至 2026-08-31，GRU 20 epoch 单 seed）

> ⚠️ 本表来自第一轮（20 epoch 无正则）诊断数据，该轮存在**全 idle 坍缩污染**，仅作历史
> 对照、不作策略结论；正式结论以第三轮多 seed 中位数（`FEATURE_STRATEGY_COMPARE.md`）为准。

| 策略 | dim | test acc | test edit | test F1@0.1 | 结论 |
|---|---:|---:|---:|---:|---|
| 全局+手部 | 80 | **59.76** | 16.02 | 11.11 | 帧级 acc 领先，段级待验证 |
| 整个画面（基线） | 40 | 50.34 | **25.97** | **23.81** | 段级最好 |
| 仅手部 | 40 | 48.02 | 11.47 | 11.11 | 整体最弱 |

完整过程与数据侧解释见 [`FEATURE_STRATEGY_COMPARE.md`](../FEATURE_STRATEGY_COMPARE.md)。

### 2.1 形态 B（bbox + 图像 embedding）E0/E1 机制床对照（2026-09-11，CPU，GRU 健康配方，3 seed）

数据为人工标注机制床 `cleansight-ActionMixed`（9,532 帧；E0 `temporal.actionmixed-v2` 40 维、
E1 `temporal.actionmixed-v2-embed-mbv3s-v1` 616 维），运行目录 `runs/embed_ablation/`：

| 指标（3 seed 中位数） | E0 bbox-40 | E1 bbox+embed-616 |
|---|---:|---:|
| 段级 edit | 42.49 | **45.22** |
| 段级 F1@0.25 | 28.00 | **34.62** |
| 段级 F1@0.5 | **15.69** | 7.69 |
| 帧级 macro_f1 | **46.63** | 35.66 |
| recall air_injection | **86.96** | 0.00 |
| recall long_brush_insert | **55.94** | 5.59 |

- 段级 edit / F1@0.25 在 3/3 seed 不劣于基线，但帧级宏指标与 `air_injection`、
  `long_brush_insert` recall 明显下降 → **图像通道不是免费增益，机制床上未定论**。
- 两臂 6/6 run 零坍缩；E1 早停更晚（8~9 epoch vs 5）说明投影头确实在学。
- **限制**：机制床与正式 v3 auto 不同源、CPU 口径、3 seed、E2/E3 未跑——
  不能并入方案b 正式数字。完整证据与复现命令见
  [`EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md`](../EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md)。

## 3. 新增方案检查清单（后续追加时照此执行）

1. **实现 recipe**：`framework/cleansight_eval/temporal/features/<name>.py`，定义契约常量
   （版本号/维度/布局常量）与逐帧构建函数；因果、无状态、中文 docstring 注明形状与语义。
2. **接入分发**：`temporal/data.py` 的 `load_split` 按 `feature_schema.version` 增加分发分支；
   拼接型契约在分发处组合，不重复实现。
3. **登记**：`framework/testsets.yaml` 新增 dataset 条目（同一原始数据可复用 revision，
   `dataset_version` 独立）与 train/val/test 三个 testset 条目（复用 manifest）；
   ROI 型声明 `feature_layout`，多块拼接型声明 `feature_blocks`。
4. **catalog 校验**：确认 `core/catalog.py` 的维度断言覆盖新契约（`检测类数 × 每类块宽 × 块数`）。
5. **配置**：新增 `framework/experiments/*.yaml`，与对照基线**同模型同超参**，仅特征契约不同。
6. **测试**：`framework/tests/test_<name>.py` 覆盖逐帧编码、空帧、遮罩、load_split 分发与维度。
7. **门禁**：`python tools/validate_testsets.py --catalog framework/testsets.yaml --json` 全绿。
8. **对照**：正式训练 + `benchmark.cli.eval`，结果与结论追加到本文档第 2 节与
   `FEATURE_STRATEGY_COMPARE.md`。
9. **文档**：更新本索引总览表（第 0 节）与新增方案详述（第 1 节），并同步
   `usage/YAML_CONFIG.md`、`docs/MODELSET_OVERVIEW.md`、`usage/TEST_COMMANDS.md`。
