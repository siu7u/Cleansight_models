# CleanSight YOLO 增强实验总览（EXPERIMENTS）

> 本目录是实验的**唯一事实来源**：结论、日志索引、权重位置、续接状态。
> 新会话先读 `STATE.json` 和 `LOG.md`，再决定下一步。

## 实验矩阵（mAP50，yolo11s / yolo11n × 4 增强预设，group1_large）

| 增强预设 | yolo11s | yolo11n | 说明 |
|---|---|---|---|
| **default** | **0.6431** 🏆 | **0.6395** 🏆 | 官方默认：mosaic 开 + 轻度 HSV + fliplr 0.5 |
| mosaic_off | 0.5726 | 0.6122 | 关 mosaic（手类下降明显） |
| mild | 0.5727 | 0.5899 | 仅翻转+微缩放 |
| strong | 0.5204 | 0.5933 | mixup 0.15 + 重 HSV（大模型伤害大） |

**group2_small（5 类，yolo11s，4 预设全部完成）**：

| 预设 | mAP50 | P | R |
|---|---|---|---|
| strong | 0.1878 | 0.5801 | 0.1730 |
| default | 0.1840 | 0.6150 | 0.1754 |
| mosaic_off | 0.1720 | 0.6770 | 0.1791 |
| mild | 0.1083 | 0.2069 | 0.1152 |

逐类（strong 为例）：syringe 0.483 / air_gun 0.345 / scope_distal_end 0.108 /
short_brush 0.002 / **brush_tip_out 0.000** → 后两类达淘汰标准（<0.3），转 ROI 特征融合。

## 核心结论

1. **正式训练用 default 增强**（mosaic 开、轻度 HSV），P≥0.71 达目标线。
2. **禁用 strong**（mixup+重 HSV）：yolo11s 的 `scope_control_body` 被腰斩（0.617→0.285）。
3. **mosaic 应保留**：关闭后 hand 类 0.860→0.803。
4. **yolo11n 性价比高**：mAP50 仅差 0.004、训练快一倍、`scope_mid_section` 反而更好。
5. **group2_small 走 ROI 融合**，别再调 YOLO 增强。

## 权重（-best 标注，供分享）

`EXPERIMENTS/best_weights/` 下每个完成实验的最优权重命名为
`<model>-<preset>-best.pt`（另有 `<group>` 区分时加组名）。权重不提交 git，
分享时直接拷贝该目录。

## 续接（新会话如何无缝继续）

```bash
# 1. 同步状态（扫描 runs/ 日志与结果，刷新 STATE.json / LOG.md）
python tools/update_experiment_state.py

# 2. 读状态
cat EXPERIMENTS/STATE.json

# 3. 查看正在跑/待跑的实验，按 STATE.json 里的命令继续
```

## 目录结构

```text
EXPERIMENTS/
├── README.md        # 本文件：总览 + 结论
├── LOG.md           # 追加式训练日志（时间线）
├── STATE.json       # 机器可读状态（新会话续接入口）
└── best_weights/    # -best 权重副本（git 忽略）
```
