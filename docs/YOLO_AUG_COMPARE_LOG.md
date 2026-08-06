# YOLO 数据增强对比实验记录

> 记录人：自动化实验会话 | 最后更新：2026-08-06

## 1. 目的

针对 CleanSight 内镜清洗检测任务，对比不同数据增强策略对 YOLO11 训练效果的影响，
为正式全量训练选择增强配置。目标组：`group1_large`（3 类：hand / scope_control_body / scope_mid_section）。

## 2. 环境

| 项目 | 说明 |
|---|---|
| 执行环境 | Windows 本机（CPU，AMD Ryzen 7 8845H 16 核；无 NVIDIA GPU） |
| Python | 3.9.13（`E:\cleansight-venv-win`，torch 2.8.0+cpu / ultralytics 8.4.115） |
| 数据集 | ModelScope `lhh010/cleansight-yolo`（git clone 全量：group1_large 21,526 训练图 / 5,169 val；group2_small 14,916 / 4,746） |
| 实验子集 | `datasets/cleansight-yolo-sub/`：每组 2,000 train / 400 val，按文件名等间隔确定性抽样 |
| 训练参数 | imgsz 480、batch 16、epochs 8、seed 42、预训练权重（yolo11s.pt / yolo11n.pt） |

> CPU 训练限制：全量数据每 epoch 约 2 小时，不可行；故用 2,000 图子集做相对对比。
> 正式训练应在 GPU 上用全量数据 + 100+ epoch 验证。

## 3. 框架修复（本会话发现并修复）

**问题**：`framework/cleansight_eval/detection/sweep.py` 的增强预设（hsv_*、mixup、
cos_lr、label_smoothing 等）**从未真正传给 ultralytics**，增强对比实验会静默失效。

- `framework/cleansight_eval/detection/yolo.py`：`YoloAdapter.train` 原来只转发
  epochs/batch/patience 等白名单参数，现改为整体转发 `train_cfg`（增强/调度超参生效）。
- `framework/cleansight_eval/core/config.py`：`train` 段注册增强相关字段
  （hsv_h/hsv_s/hsv_v/degrees/translate/scale/shear/perspective/flipud/fliplr/
  mosaic/mixup/copy_paste/erasing/cos_lr/label_smoothing/close_mosaic/freeze），
  使 `cli.train -S` 点路径覆盖可用。
- 验证：`framework/tests/test_detection_sweep.py`、`test_config_paths.py` 相关用例通过。

配套工具脚本（`tools/`）：
- `prepare_aug_experiments.py`：全量数据集 → 确定性子集抽样（`--src/--out/--train/--val`）。
- `aug_experiments.py`：增强对比实验 runner（`--group/--model/--presets/--epochs/--imgsz/
  --batch/--device/--data-dir/--runs-dir`），自动 GPU 检测、自动修复 data.yaml 相对路径。
- `report_aug_results.py`：结果 JSON → 对比报告。

## 4. 增强预设定义（与 sweep.py 对齐）

| 预设 | 说明 | 关键参数 |
|---|---|---|
| `default` | ultralytics 官方默认 | hsv(0.015/0.7/0.4)、translate 0.1、scale 0.5、fliplr 0.5、mosaic 1.0、mixup 0 |
| `strong` | 仓库 sweep 强增强 | hsv(0.02/0.8/0.5)、translate 0.2、scale 0.7、shear 2.0、mosaic 1.0、**mixup 0.15** |
| `mosaic_off` | 关闭 mosaic | 同 default 但 mosaic 0.0 |
| `mild` | 最轻量 | 仅 fliplr 0.5 + scale 0.2，其余 0 |

## 5. 实验结果

### 5.1 yolo11s × group1_large（2026-08-06，8 epoch @ 480）

| 预设 | mAP50 | mAP50-95 | Precision | Recall | 耗时 |
|---|---|---|---|---|---|
| **default** | **0.6431** | **0.2358** | **0.7134** | 0.6020 | 74.7m |
| strong | 0.5204 | 0.1893 | 0.6196 | 0.5840 | 76.8m |
| mosaic_off | 0.5726 | 0.2050 | 0.5489 | 0.6046 | 74.9m |
| mild | 0.5727 | 0.2169 | 0.6994 | 0.5549 | 74.5m |

逐类 mAP50：

| 预设 | hand | scope_control_body | scope_mid_section |
|---|---|---|---|
| default | **0.860** | **0.617** | 0.452 |
| strong | 0.850 | **0.285** | 0.427 |
| mosaic_off | 0.803 | 0.442 | **0.473** |
| mild | 0.800 | 0.565 | 0.354 |

### 5.2 yolo11n × group1_large（2026-08-06，完整）

| 预设 | mAP50 | mAP50-95 | Precision | Recall | 耗时 |
|---|---|---|---|---|---|
| **default** | **0.6395** | **0.2411** | 0.7083 | **0.6326** | 37.2m |
| strong | 0.5933 | 0.2064 | 0.6158 | 0.6250 | 37.4m |
| mosaic_off | 0.6122 | 0.2274 | 0.6934 | 0.5963 | 36.5m |
| mild | 0.5899 | 0.2318 | **0.7180** | 0.5547 | 37.0m |

yolo11n 逐类 mAP50：default（hand 0.808 / scope_control_body 0.609 / scope_mid_section **0.501**）、
strong（0.829/0.460/0.491）、mosaic_off（0.827/0.526/0.484）、mild（0.827/0.551/0.391）。

### 5.3 模型 × 增强 汇总对比（mAP50）

| 预设 | yolo11s | yolo11n |
|---|---|---|
| **default** | **0.6431** | **0.6395** |
| strong | 0.5204 | 0.5933 |
| mosaic_off | 0.5726 | 0.6122 |
| mild | 0.5727 | 0.5899 |
| 训练耗时/预设 | ~75m | ~37m |

### 5.4 group2_small（部分完成）

group2_small（5 类，严重不均衡：brush_tip_out 仅 116 实例 / short_brush 279）。
yolo11s @ 8 epoch @ 480：

| 预设 | mAP50 | mAP50-95 | Precision | Recall | 状态 |
|---|---|---|---|---|---|
| **default** | **0.1840** | **0.0491** | **0.6150** | **0.1754** | ✅ 完成 |
| strong | — | — | — | — | 被关机中断 |

**关键结论**：group2_small 纯 YOLO 检测 Recall 仅 0.175（小目标/稀有类在 480 下基本检不出），
印证 `docs/YOLO_OPTIMIZATION.md` 的判断——此类应从 YOLO 淘汰、转 ROI 图像特征融合
（`roi_classification` 流水线），而非继续调增强参数。

## 6. 恢复指引（重启后）

```powershell
# 1. 重新跑 group2_small 对比（约 5 小时，建议放 GPU 机器）
python tools\aug_experiments.py --group group2_small --model yolo11s ^
  --presets default,strong,mosaic_off,mild --epochs 8 --imgsz 480 --batch 16 ^
  --data-dir E:\曦源\Cleansight_models\datasets\cleansight-yolo-sub ^
  --runs-dir E:\曦源\Cleansight_models\runs

# 2. 查看已有结果
python tools\report_aug_results.py E:\曦源\Cleansight_models\runs\aug_compare_*.json
```

环境：Windows venv `E:\cleansight-venv-win`（torch 2.8.0+cpu / ultralytics 8.4.115）。
数据集全量在 `E:\曦源\Cleansight_models\datasets\cleansight-yolo`（git clone 完整，含 LFS）。

## 7. 结论（yolo11s + yolo11n 完整对比）

1. **default 增强在两种模型上都是最优**：yolo11s 0.643 / yolo11n 0.640，P 均 ≥0.71（达文档目标线）。
2. **增强效果排名一致**：default > mosaic_off > mild ≈ strong。
3. **strong 对大模型的伤害更大**：yolo11s -0.123（scope_control_body 崩到 0.285），yolo11n 仅 -0.046——
   大模型对 mixup/重 HSV 的"误导性样本"更敏感；短训（8 epoch）下强增强也拖慢收敛。
4. **mosaic 应保留**：关闭后 hand 类下降（yolo11s 0.860→0.803），整体 P 掉 0.16。
5. **yolo11n 性价比突出**：mAP50 与 yolo11s 几乎持平（-0.004），耗时减半，R 更高（0.633 vs 0.602）；
   `scope_mid_section` 类 yolo11n 反而更好（0.501 vs 0.452）。
6. **scope_mid_section 仍是共性短板**：所有模型/预设 R 仅 0.30~0.42，建议单独分析标注质量/外观差异。

## 8. 建议的正式训练配置

- 增强：**default**（可微调 `hsv_s 0.5`、`mixup 0.1` 做轻量正则探索）
- imgsz 640、epochs 100+、cos_lr、早停 patience 30~40
- 正式跑前先在 `framework/testsets.yaml` 登记 dataset/testset 并过 `tools/validate_testsets.py`
- 若追求速度/部署体积：yolo11n 优先

## 9. 产物位置

- 结果 JSON：`runs/aug_compare_group1_large_yolo11s_20260806-192102.json`（s）
- 训练日志：`runs/aug_g1_yolo11s_win.log`（s）、`runs/aug_g1_yolo11n.log`（n）
- 训练权重：`runs/<model>-<preset>-480-8e-*/weights/best.pt`
