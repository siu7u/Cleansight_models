# MS-TCN 容量与规模研究

> **正式报告：[`../EXPERIMENT_REPORT_MSTCN_CAPACITY_20260922.md`](../EXPERIMENT_REPORT_MSTCN_CAPACITY_20260922.md)**（按仓库实验报告惯例的 6 节骨架，含汇总图）。
> **详细版：[`MSTCN_CAPACITY_STUDY.md`](./MSTCN_CAPACITY_STUDY.md)**（结论速览 + 六个部分详述 + 复跑命令 + 代码改动清单）。原 `PARAMS_INVENTORY.md` / `PARAMS_VS_INPUT_DIM.md` / `CAPACITY_EXPERIMENT.md` /
> `CAPACITY_LR_FOLLOWUP.md` / `GAP_FILLING.md` 已并入该文件，并于 2026-09-22 删除。

## 结论速览

| 问题 | 结论 | 详见 |
|---|---|---|
| 现在多大？ | `mstcn` h32 = **3.8 万参**（0.038 M），四种时序架构里最小；87% 参数在 8 个残差块 | 第 1 部分 |
| 输入变大 → 参数同步变大？ | **不同步**：每维只 +`hidden`（32）；40→616 维只 +53% | 第 2 部分 |
| 默认配方加参数有用吗？ | **没用**：lr=0.002 下 h256 的 acc 55.10 ≈ 全 idle 基线，配对 edit 8 胜 14 负 | 第 3 部分 |
| 改配方后呢？ | **有用但 `mstcn` 到 h128 为止**：lr=0.0005 下 h128 vs h32 配对 28/10、p=0.0026；h256 ≈ h128（p=0.80） | 第 4、5 部分 |
| 换结构还有空间吗？ | **有**：`mstcn2` h128/s4l10（331 万参）60 轮 edit **51.47**、逐 seed 摆幅 **0.64**；h128/s2l5（100 万参）拿约 90% | 第 6.2 部分 |
| 加特征值不值？ | **值**：40 → 144 维在 h32 上 edit +14.29（配对 p=0.020），参数量只 +9%，坍缩明显减少 | 第 6.1 部分 |
| 容量极限在哪？ | `mstcn` 有效区间 h128–h256；h8（3.4 千参）太小、h512（847 万参）回落 | 第 5 部分 |

## 批次速查（逐 run 指标在各批次的 `CAPACITY_SUMMARY.md`）

| 批次 | 内容 | run |
|---|---|---:|
| `runs/capacity-mstcn` | 默认配方 5 容量点 × 3 seed | 15 |
| `runs/capacity-lr002-e60` / `-lr0005-e60` / `-lr0005-e150` | lr × 预算三臂（含容量边界 h8/h512、seed 补到 5） | 9 / 13 / 15 |
| `runs/capacity-lr001-e60` | lr 网格补 0.001 档 | 9 |
| `runs/curve-frac025` / `-frac050` | 数据规模轴（25% / 50% 训练视频） | 4 + 4 |
| `runs/mstcn2-cap-*` | mstcn2 六个组合 × 3 seed | 15 |
| `runs/capacity-recipe20` | 既有配方臂（≤20 轮 + patience=4） | 15 |
| `runs/mstcn-40d-cap` | 40 维契约的容量梯度 | 9 |
| `runs/roi-real-airgun-brushtip` | 分类真实数据（20,394 个 ROI 裁剪） | 1 |

复跑命令见合并文档的**附 A**；跨批次汇总用 `python tmp/gap_summary.py`。

## 维护约定

- 新增实验直接往 `MSTCN_CAPACITY_STUDY.md` 加"第 N 部分"或补充小节，并把结论回填到本文速览表。
- 结论必须带判读等级与噪声地板（跨 seed 摆幅 / 配对检验）；单 seed 结果不进速览表。
- 数字必须能由文档内的复跑命令或 `runs/<批次>/CAPACITY_SUMMARY.md` 复现；口径变化（数据 revision、
  seed 集、epochs/patience、lr、设备）必须写明。
