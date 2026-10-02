# 帧准确率攻关系列（`docs/experiments/frame-acc/`）

> **系列主题**：把时序模型的帧级准确率 `acc`
> （`accuracy/frame-wise-micro-across-items/percent/v3`）从旗舰基线 **53.90** 往上推，
> like-for-like 口径下最终定标 **58.31（+4.41pp，32 seed 配对 32/0）**。
> **读法**：时间紧只看 [`EXPERIMENT_REPORT_FRAME_ACC_SUMMARY_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_SUMMARY_20260929.md)
> （12 轮浓缩，157 行）；要看过程按 R1→R12 顺序读，**但必须先看下表的「状态」列**——
> 有两轮含更正声明、其结论已被后续轮次推翻。

## 1. 逐轮索引与状态

| 轮 | 文档 | 一句话 | 状态 |
|---|---|---|---|
| — | [`…_SUMMARY_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_SUMMARY_20260929.md) | **12 轮浓缩总报告**：目标与口径 / 最终定标 / 有效杠杆 / 已排除 16 条轴 / 工程改动 / 方法学教训 / 复现 | ✅ **入口，现行** |
| R1 | [`…_20260928.md`](EXPERIMENT_REPORT_FRAME_ACC_20260928.md) | 5 个轴、20 余次尝试，最好 56.69（+2.77pp）；**天花板定位** | 现行 |
| R2 | [`…_R2_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R2_20260929.md) | 候选根因"自动标注的检测框" | ⚠️ **top 有更正声明**：§3/§5 因果结论已被 R3 推翻 |
| R3 | [`…_R3_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R3_20260929.md) | 上游修复方案实测校验 + **对本系列前两轮的自检更正** | 现行（R2 的更正依据） |
| R4 | [`…_R4_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R4_20260929.md) | **首个显著正收益**：维度预算分给有证据的类（v1→v2，+2.56pp，8/8，p=0.0078） | 现行 |
| R5 | [`…_R5_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R5_20260929.md) | 系统扫轴 + "饱和"结论；源域发现 `presence` 完全冗余 | 现行 |
| R6 | [`…_R6_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R6_20260929.md) | **通道语义替换 v4**（全栈 59.15 / 单模型 57.59）——**本系列最关键一轮** | 现行，注意口径混用风险见 SUMMARY §6 |
| R7 | [`…_R7_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R7_20260929.md) | 解码轴复核（含一次自我纠错）+ v5（dx,dy）否定 | 现行 |
| R8 | [`…_R8_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R8_20260929.md) | "学习式表示"路线整体否定（集合编码器 −7.84pp） | 现行 |
| R9 | [`…_R9_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R9_20260929.md) | 补学习率调度 + **量化测量精度**（8-seed 噪声 ±0.7~1.3pp） | ⚠️ **top 有更正声明**：§2.1 两条结论被 R10 的 32-seed 复核推翻 |
| R10 | [`…_R10_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R10_20260929.md) | **32-seed 定标**：+4.41pp（配对 +4.47，32/0，CI [+3.33,+5.11]） | ✅ **最终读数来源** |
| R11 | [`…_R11_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R11_20260929.md) | 联合空间饱和 + 标签平滑 / 跨 seed 权重平均否定 + **两种口径同时定标** | ✅ **终标口径来源** |
| R12 | [`…_R12_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R12_20260929.md) | 通道集合高功效筛选（否定）+ **局部空间收敛判定** | ✅ 收口轮 |

## 2. 系列的三条纵向结论（跨轮不变）

1. **唯一有效杠杆 = 特征契约的"维度预算 + 通道语义"**（R4 → R6），其余 16 条轴全部排除（见 SUMMARY §4）。
2. **口径必须先立**：R9 才发现 8-seed 噪声幅度，R10 才给出可信点估计与 CI，R11 才统一两种口径。
   早期几轮的"提升"读数按后期标准多数不可引用。
3. **自我更正留档**：R2→R3、R9→R10 两处推翻，加上 val 拟合偏置跨 seed 假象与权重平均坍缩，
   共 4 处已登记在 SUMMARY §6。

## 3. 相关入口

- 方法学与评测纪律（现行）：[`../weeks/2026-09-26_2026-10-02/TOPIC_09_11_METRIC_AND_OFFLINE_20260929.md`](../../weeks/2026-09-26_2026-10-02/TOPIC_09_11_METRIC_AND_OFFLINE_20260929.md)
- 机制与数据侧解释：[`../weeks/2026-09-26_2026-10-02/TOPIC_03_04_DATA_AND_MECHANISM_20260929.md`](../../weeks/2026-09-26_2026-10-02/TOPIC_03_04_DATA_AND_MECHANISM_20260929.md)
- 特征契约本身：[`../features/README.md`](README.md)
- 汇报用稿：[`../weeks/2026-09-26_2026-10-02/STAGE_REPORT_20260929.md`](../../weeks/2026-09-26_2026-10-02/STAGE_REPORT_20260929.md)

## 4. 目录约定

本目录只放**同一攻关系列**的轮次报告与总报告；一次性实验报告仍留在 `docs/` 顶层
（与 [`../mstcn-capacity/`](../../mstcn-capacity/) 的容量研究同属"多文件报告自建子目录"惯例）。
新增轮次后：在此登记（含状态列），并同步 [`../README.md`](../../../README.md) §6 与当周 `docs/weeks/<周>/README.md`。
