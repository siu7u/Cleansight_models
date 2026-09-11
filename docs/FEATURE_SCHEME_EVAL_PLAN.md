# 特征提取方案评估计划（feature-scheme eval plan）

> 立项：2026-09 · 对应实验区 `docs/FEATURE_LAB.md` · 评估对象：时序动作识别（长短毛刷刷洗 insert/withdraw 为重点对比对）
> 本文是**方案文档**：定义四个方案的编码口径、统一实验矩阵与评测口径；实验结果另行回写到
> `docs/FEATURE_LAB.md` 与实验报告，不混入本文。

## 1. 目标与总原则

用**同一份数据、同一套模型、同一套训练配方、同一套指标**，只更换帧级特征的提取方式，
回答一个问题：**哪一种视觉特征让时序模型识别动作段最准、边界最稳。**

硬约束（沿用 modelset-quality 与 FEATURE_LAB 纪律）：

- 特征集一旦改变 → 新 feature_mapping 版本，模型必须重训，不与旧版本混评。
- 在线可部署优先：特征尽量**因果、逐帧独立**（只依赖当前帧），全帧 CNN embedding 允许
  预计算离线口径，但需在报告中显式标注 offline 属性。
- 评测分 `formal` / `exploratory`；无置信度标注路径一律 exploratory。
- 单 seed 不下结论，正式排序需 ≥3 seed。

## 2. 现有资产盘点（写方案前先看清楚已有什么）

| 资产 | 位置 | 状态 |
|---|---|---|
| 基线特征 clean_bbox_v2（113/121/249 维，含插补/速度/pair 统计） | `framework/cleansight_eval/temporal/features/clean_bbox_v2.py`（本分支） | ✅ 现行 |
| 手部区域 bbox 编码 `actionmixed-bbox-hand-8cls-v1`（40 维） | `temporal/features/hand_bbox.py`（feat/roi-training 分支） | ✅ 已实现+测试 |
| 全局+手部拼接 `actionmixed-bbox-global-hand-8cls-v1`（80 维） | 同上（data.py 拼接） | ✅ 已实现 |
| ROI 网格 `actionmixed-roi-grid-v1`（2×3 网格 × 8 类 × 3 通道 = 144 维） | `temporal/features/roi_bbox.py`（feat/roi-training 分支） | ✅ 已实现，先行实验**段级指标领先基线 ~15 点** |
| 全帧 CNN embedding 离线预计算（resnet18/34/50、mobilenet、efficientnet） | `temporal/features/extract_embeddings.py`（feat/roi-training 分支） | ✅ 已实现+测试 |
| 策略横向对比先例（含 3-seed 矩阵工具 `tools/run_strategy_matrix.py`） | feat/roi-training 分支 `docs/FEATURE_STRATEGY_COMPARE.md` | ✅ 可复用流程 |
| 数据 `temporal.actionmixed-auto-v3`（train 14 / val 4，test 已并入取消） | datasets + testsets.yaml | ✅ 就绪 |

> 结论：四个方案**基本不需要从零造轮子**，主要工作是把 feat/roi-training 分支的特征实现
> 迁移/对齐到本分支，并补齐"全局视觉特征"与"手部特征"的**组合融合**路径。

## 3. 四个方案定义

### 基线 B0：仅 bbox —— bbox 怎么变成向量矩阵

逐帧 YOLO 文本 `class cx cy w h` → 每帧一个定长向量 → 视频即 `[T, D]` 矩阵。三个候选编码：

| 编码 | 维度 | 思路 | 定位 |
|---|---:|---|---|
| B0a 8cls-v1 max-box | 40 | 每类取最大框，编码 `[presence, cx, cy, w, h]` | **推荐主基线**：最简单、与分支已有对比实验可比 |
| B0b ROI 网格 | 144 | 2×3 空间网格 × 8 类 × `[presence, count, max_area]`，丢绝对坐标保空间分布 | 强基线（先行实验已领先），建议作为"必须打赢的对手" |
| B0c clean_bbox_v2 | 113 | 每目标 8 通道统计 + 短缺失插补 + pair 关系 | 现行正式特征，作为工程现状参照 |

基线实验至少含 B0a + B0c；B0b 结果已有（第三轮矩阵），可直接引用或复跑对齐数据版本。

### 方案 S1：bbox + 手部特征

- 特征：`actionmixed-bbox-global-hand-8cls-v1`，全局 40 维 + 手部区域 40 维拼接 = **80 维**。
- 手部区域定义：面积最大 hand 框绕中心扩张 1.5 倍（v1 固定），区域内每类取最大框，
  坐标相对区域归一化；无 hand 帧全零（v3 val 中占比 4.9%）。
- 已知风险（来自数据侧事实）：scope 类在手部区域 presence 仅 14~16%，手部通道会稀释
  scope 信号——这正是要靠拼接而非替换来缓解的。
- 升级方向（二期，不进本轮）：接入手部关键点模型（如 MediaPipe Hands 21 关键点）替换
  hand box 锚定，形成 `actionmixed-hand-kp-v1`，需重走数据管线与版本。

### 方案 S2：bbox + 整帧全局视觉特征

- 特征：B0（40 维）⊕ 全帧 CNN embedding（resnet18，512 维，ImageNet 预训练、去分类头，
  eval+no_grad 逐帧因果计算，缺图帧补零）。
- 维度控制：512 维原始 embedding 与 40 维 bbox 量纲失衡，先做**冻结投影**（PCA 到 64~128 维
  或 1×1 线性投影层随模型训练），投影方式写入 feature_mapping 版本。
- 属性标注：embedding 依赖帧图，为 **offline 预计算**口径；在线部署需补 backbone 推理，
  延迟另测（可参考 `tools/benchmark_gpu_roi_track_latency.py` 的做法），本轮只评质量。

### 方案 S3：手部特征 + 全局视觉特征（全组合）

- 特征：手部 40 维 ⊕ 全局 bbox 40 维 ⊕ 全局 CNN embedding（投影后）= 三路拼接（或同维投影后
  concat），即 S1+S2 的超集。
- 消融由本矩阵天然覆盖：S3 − S2 ≈ 手部贡献，S3 − S1 ≈ 全局视觉贡献；若实现支持
  dropout 特征遮罩（FEATURE_LAB 迁移清单项），可在单模型内做特征消融复核。

## 4. 统一实验矩阵

| 变量 | 取值（本轮固定） |
|---|---|
| 数据 | `temporal.actionmixed-auto-v3`，train 14 / val 4；评测主指标看 **val**（test 已并入取消） |
| 模型 | GRU（hidden=128, 3 层）先行；最优特征复跑 mstcn / transformer 各一次验证泛化 |
| 配方 | weight_decay=1e-4 / dropout=0.2 / patience=4 / best_metric=val_f1_0.5（对齐第三轮健康配方） |
| seed | 42 / 0 / 1，报中位数与最差值 |
| 输入 | 同一 labels/frames，仅特征契约不同 |

| 方案 | feature_mapping | 维度 | 状态 |
|---|---|---:|---|
| B0a | `actionmixed-bbox-8cls-v1` | 40 | 基线 |
| B0b | `actionmixed-roi-grid-v1` | 144 | **参照行**：强基线（先行实验段级领先 ~15 点），本轮随矩阵复跑对齐当前数据版本 |
| B0c | `clean_bbox_v2_top1_impute` | 113 | 基线（工程现状） |
| S1 | `actionmixed-bbox-global-hand-8cls-v1` | 80 | 待跑 |
| S2 | `actionmixed-bbox-cnn-resnet18-v1`（新建，需登记） | 40+投影 | 待跑 |
| S3 | `actionmixed-bbox-hand-cnn-v1`（新建，需登记） | 80+投影 | 待跑 |

> B0b（ROI 144 维）先行结果来自 feat/roi-training 分支旧数据版本（train 13/val 3/test 2 +
> task#204 修正前后混合），与本矩阵不可直接比数字；故随本轮统一复跑（3 seed、当前 v3
> train 14/val 4），仅作参照行，不重复计入方案排序判据——排序判据仍是 §6 第 5 步的
> 「新方案须同时优于 B0a 与 B0b」。

## 5. 评测口径

- **段级主线**：edit score、F1@IoU 0.1 / 0.25 / 0.5（帧级 acc 仅作参考——val 中 idle 占比高，
  acc 会被"永远猜 idle"抬高，见第一轮坍缩教训）。
- **重点混淆对**：long_brush insert vs withdraw 混淆矩阵（FEATURE_LAB §4 P0 对比）。
- **边界定位**：边界帧误差分布（模糊渐变段单独统计）。
- 结果落盘走统一 `EvaluationResult`（schema v2），标注 data split、feature_schema、device、
  seed 与 offline/online 属性。

## 6. 执行步骤

1. **迁移**：从 `feat/roi-training` 合入 `hand_bbox.py` / `roi_bbox.py` / `extract_embeddings.py`
   及其测试（或 cherry-pick），保持 feature 语义与该分支一致，升版本需显式声明。
2. **补齐 S2/S3**：新增 embedding 投影与三路拼接的 feature_mapping 注册、data.py 装配、
   实验 YAML（`gru-actionmixed-auto-{s2,s3}.yaml`）与单测。
3. **预计算 embedding**：`python -m framework.cleansight_eval.temporal.features.extract_embeddings
   --root datasets/... --splits train val --backbone resnet18 --out-dir runs/image_embeddings/...`。
4. **跑矩阵**：复用/改写 `tools/run_strategy_matrix.py`（3 seed × 6 特征集，含 B0b 参照行），产物进
   `runs/strategy_compare/`，汇总回写 `docs/FEATURE_LAB.md`。
5. **结论判据**：以 val edit/F1@0.25 的 3-seed 中位数排序；新方案须同时优于 B0a 与 B0b
   才算"特征增益成立"，否则 ROI 网格（更便宜）胜出。

## 7. 已知限制

- val 仅 4 个视频，段级指标方差大；结论表述必须带 seed 区间，不夸大排序。
- 标注无检测置信度（5 列格式），B0c 置信度通道为默认值填充 → 相关对比仅 exploratory。
- S2/S3 的 CNN 为 ImageNet 预训练未微调；若三方案差距不显著，二期再做冻结微调
  （linear probe / 末层微调）后再评。
- 全帧 embedding 的因果性成立但计算代价高，质量结论不等于可部署结论。
