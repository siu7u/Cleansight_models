# 训练数据类别分布与逐类准确率（feature-lab）

> 数据：`temporal.actionmixed-auto-v3`（train 14 / val 4）· 模型：GRU 健康配方 · 3 seed（42/7/2026）·
> CPU/WSL exploratory 口径 · 生成脚本逻辑见提交说明；上游结果见 [`FEATURE_LAB.md`](FEATURE_LAB.md)。

## 1. 训练数据类别帧分布（含 idle 占比）

| 动作类别 | train 帧数 | train 占比 | val 帧数 | val 占比 |
|---|---:|---:|---:|---:|
| idle | 6274 | 65.52% | 2308 | 68.20% |
| water_injection | 194 | 2.03% | 17 | 0.50% |
| flush | 893 | 9.33% | 183 | 5.41% |
| long_brush_insert | 1289 | 13.46% | 495 | 14.63% |
| long_brush_withdraw | 437 | 4.56% | 206 | 6.09% |
| short_brush_cleaning | 488 | 5.10% | 175 | 5.17% |
| **合计** | **9575** | 100% | **3384** | 100% |

- **idle 占比：train 65.52%（6274/9575 帧），val 68.20%（2308/3384 帧）**。
- 稀有类警告：water_injection 在 train 仅 2.03%、val 仅 0.50%——该类指标方差极大，不可单独用于选型。

## 2. 逐类帧级准确率（定版候选 roi-144，3-seed 中位数）

整体帧准确率 acc = **55.47%**，macro F1 = **0.2029**（帧级 acc 受 idle 主导抬高，参考价值有限）。

| 动作类别 | precision | recall | F1 | IoU | val 支持帧数 |
|---|---:|---:|---:|---:|---:|
| idle | 0.6549 | 0.8042 | 0.7219 | 0.5648 | 2308 |
| water_injection | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17 |
| flush | 0.4179 | 0.0000 | 0.3533 | 0.0000 | 183 |
| long_brush_insert | 0.0413 | 0.0000 | 0.0428 | 0.0000 | 495 |
| long_brush_withdraw | 0.0833 | 0.1019 | 0.1033 | 0.0545 | 206 |
| short_brush_cleaning | 0.1739 | 0.0571 | 0.0976 | 0.0513 | 175 |

### 2.1 混淆矩阵（roi-144，3 seed 合计，行 = 真值，列 = 预测）

| 真值\预测 | idle | water_injection | flush | long_brush_insert | long_brush_withdraw | short_brush_cleaning |
|---|---:|---:|---:|---:|---:|---:|
| **idle** | 5329 | 51 | 78 | 638 | 657 | 171 |
| **water_injection** | 51 | 0 | 0 | 0 | 0 | 0 |
| **flush** | 351 | 114 | 56 | 15 | 13 | 0 |
| **long_brush_insert** | 1303 | 0 | 0 | 44 | 102 | 36 |
| **long_brush_withdraw** | 524 | 0 | 0 | 24 | 70 | 0 |
| **short_brush_cleaning** | 487 | 0 | 0 | 0 | 0 | 38 |

## 3. 六特征集逐类 F1 对比（3-seed 中位数）

> ⚠️ **配方混杂提示（2026-09-12 审计）**：本表 bbox-40 / hand-40 / global-hand-80 三列
> 产生于未含健康配方的 YAML（无 dropout/weight_decay/patience，best.pt 按 val_acc 选型），
> 与 roi-144 / S2 / S3 口径不一致；配置已修复但**重跑已暂停（待后续执行）**，
> 替换前这三列仅作诊断参考。详见 [`FEATURE_LAB.md`](FEATURE_LAB.md) §6.2。

| 动作类别 | bbox-40 (B0a) | roi-144 (B0b) | hand-40 | global-hand-80 (S1) | bbox+cnn-120 (S2) | hand+bbox+cnn-160 (S3) |
|---|---:|---:|---:|---:|---:|---:|
| idle | 0.7851 | 0.7219 | 0.8154 | 0.7799 | 0.7113 | 0.7422 |
| water_injection | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a |
| flush | 0.4422 | 0.3533 | n/a | 0.2157 | 0.4008 | 0.3313 |
| long_brush_insert | 0.0000 | 0.0428 | n/a | n/a | 0.0279 | 0.0000 |
| long_brush_withdraw | n/a | 0.1033 | n/a | n/a | 0.0000 | 0.0000 |
| short_brush_cleaning | 0.5212 | 0.0976 | 0.5390 | 0.4308 | 0.2338 | 0.2932 |

### 3.1 关键观察

- **idle 占 train 65.5% / val 68.2%**，模型帧级 acc 约 55~69% 基本由 idle 主导；
- 非 idle 类逐类召回普遍极低（多数在 0~10%），说明模型识别动作段的能力仍弱——
  这正是段级指标（edit / F1@IoU）才是主线、帧级 acc 不可用作选型的原因；
- 六个特征集对 water_injection 全为 0 命中：val 仅 17 帧（0.50%），属数据不足而非特征问题；
- roi-144 相对 bbox-40 的逐类差异：长毛刷 insert 有非零 F1（0.0428 vs 0），
  而 short_brush_cleaning 反而更弱（0.0976 vs 0.5212）——两方案的段级优劣由段匹配而非
  单帧判对决定；
- 多类出现 `n/a`：该 seed 下该类无有效预测（metric 未定义），属预期现象。

## 4. 口径说明与限制

- 逐类 precision/recall/F1/IoU 取自 `EvaluationResult.metrics.details.temporal.frame.per_class`，
  帧级口径为 micro（跨视频汇总帧）；3 seed 取中位数。
- 混淆矩阵为 3 个 seed 的逐像素计数合计（每 seed 独立评估 3384 帧）。
- 帧级 acc 在 idle 占比 68.2% 的 val 上会被"永远猜 idle"抬到 68% 以上，
  **选型应以段级 edit / F1@IoU 为准**（见 FEATURE_LAB §6）。
- val 仅 4 视频，稀有类支持帧过少，逐类数字仅供诊断，不能作为单类结论。
