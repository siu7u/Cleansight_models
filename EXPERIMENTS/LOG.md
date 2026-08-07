# CleanSight YOLO 增强实验日志

> 追加式日志。新条目追加在末尾。同步方式：`python tools/update_experiment_state.py`。

## 2026-08-06 · 环境搭建

- 数据集：ModelScope `lhh010/cleansight-yolo` git clone 全量
  （group1_large 21,526 train / 5,169 val；group2_small 14,916 / 4,746）
- 实验子集：`datasets/cleansight-yolo-sub/`，2,000 train / 400 val，等间隔抽样
- 环境：Windows venv `E:\cleansight-venv-win`（torch 2.8.0+cpu / ultralytics 8.4.115）
- **框架修复**：`YoloAdapter.train` 不再丢弃增强参数（原白名单只转发
  epochs/batch/patience，sweep 的 hsv/mixup/cos_lr 等全部静默失效）；
  `core/config.py` 注册 train 段增强字段。提交 `30bc6f7`。

## 2026-08-06 · yolo11s × group1_large × 4 预设（8 epoch @ 480）

日志 `runs/aug_g1_yolo11s_win.log`，结果 `runs/aug_compare_group1_large_yolo11s_20260806-192102.json`

| 预设 | mAP50 | mAP50-95 | P | R | 耗时 |
|---|---|---|---|---|---|
| default | **0.6431** | **0.2358** | **0.7134** | 0.6020 | 74.7m |
| strong | 0.5204 | 0.1893 | 0.6196 | 0.5840 | 76.8m |
| mosaic_off | 0.5726 | 0.2050 | 0.5489 | 0.6046 | 74.9m |
| mild | 0.5727 | 0.2169 | 0.6994 | 0.5549 | 74.5m |

**结论**：default 最优；strong 的 scope_control_body 崩到 0.285；mosaic 应保留。

## 2026-08-06 · yolo11n × group1_large × 4 预设（8 epoch @ 480）

日志 `runs/aug_g1_yolo11n.log`，结果 `runs/aug_compare_group1_large_yolo11n_20260806-215642.json`

| 预设 | mAP50 | mAP50-95 | P | R | 耗时 |
|---|---|---|---|---|---|
| default | **0.6395** | **0.2411** | 0.7083 | **0.6326** | 37.2m |
| strong | 0.5933 | 0.2064 | 0.6158 | 0.6250 | 37.4m |
| mosaic_off | 0.6122 | 0.2274 | 0.6934 | 0.5963 | 36.5m |
| mild | 0.5899 | 0.2318 | **0.7180** | 0.5547 | 37.0m |

**结论**：与 yolo11s 几乎持平、耗时减半、scope_mid_section 反而更好 → 部署优先 yolo11n。

## 2026-08-06 · group2_small × yolo11s default（8 epoch @ 480）

日志 `runs/aug_g2_yolo11s.log`

| 预设 | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| default | 0.1840 | 0.0491 | 0.6150 | 0.1754 |

**结论**：R 仅 0.175，纯 YOLO 检不动小目标/稀有类（brush_tip_out 116 实例等），
应转 ROI 特征融合。提交 `063aa7b`。

## 2026-08-06 · group2_small × yolo11s 剩余预设（strong/mosaic_off/mild）

日志 `runs/aug_g2_yolo11s_rest.log`（进行中）

## 2026-08-07 13:37:08 状态同步
- 实验日志: 6 条
- 见 STATE.json 详情

## 2026-08-07 13:37:51 状态同步
- 实验日志: 6 条
- 见 STATE.json 详情

## 2026-08-07 17:39:10 状态同步
- 实验日志: 7 条
- 见 STATE.json 详情

## 2026-08-07 · group2_small × yolo11s 全部 4 预设完成

日志：runs/aug_g2_yolo11s.log（default）+ runs/aug_g2_yolo11s_rest.log（strong/mosaic_off/mild）
结果：runs/aug_compare_group2_small_yolo11s_20260807-173516.json

| 预设 | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| strong | 0.1878 | 0.0551 | 0.5801 | 0.1730 |
| default | 0.1840 | 0.0491 | 0.6150 | 0.1754 |
| mosaic_off | 0.1720 | 0.0475 | 0.6770 | 0.1791 |
| mild | 0.1083 | 0.0380 | 0.2069 | 0.1152 |

逐类 mAP50（strong 为例）：syringe 0.483 / air_gun 0.345 / scope_distal_end 0.108 /
short_brush 0.002 / **brush_tip_out 0.000**。

**结论**：
1. 4 预设 mAP50 全部 0.11~0.19、R ~0.17 —— 纯 YOLO @480 检不动小目标，与增强无关。
2. **brush_tip_out（0.000）与 short_brush（0.002）达到淘汰标准（<0.3）**，应转 ROI 特征融合。
3. 与 g1 相反，g2 上 mild 最差（syringe 0.053）——小目标需更强增强，但救不了根本瓶颈。
4. 建议：用 benchmark.cli.analyze 出淘汰决策 → roi-fusion.yaml 训练分类器，或试 1280+P2。

## 2026-08-07 17:40:46 状态同步
- 实验日志: 7 条
- 见 STATE.json 详情
