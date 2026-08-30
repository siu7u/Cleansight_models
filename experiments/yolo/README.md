# YOLO 优化工作时间线

## 当前结论

- 检测权重推荐：`runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt`。
- 单帧检测：`imgsz=640 + conf=0.30 + Gaussian Soft-NMS@0.55`。
- 视频检测+追踪：`conf=0.25 + NMS IoU=0.55 + BoT-SORT high-clean`。
- 后者在 41 个 Label Studio task / 4290 帧上达到 `F1=0.7009`、`IDF1=0.5416`、
  `ID Switches=199`，优于旧的 conf 0.30 / IoU 0.70。
- 通用预训练 ROI embedding 对 tracker 的 IDF1 提升很小；若用于动作模型，必须按动作指标重新评估。

## 组织结构

```text
experiments/yolo/
├── training/
│   ├── 01_strong_cos/
│   ├── 02_stable_adamw/
│   └── 03_ft768_lowlr/
├── postprocess/
│   ├── 01_single_frame_methods/
│   ├── 02_tracking_gt_evaluation/
│   ├── 03_botsort_tuning/
│   └── 04_validation_and_tracker_input/
└── performance/
    ├── 01_gpu_latency/
    └── 02_roi_backbone_compare/
```

## 时间顺序

| 日期 | 实验/修改 | 核心结果 | 详细文档 |
|---|---|---|---|
| 2026-08-06 | YOLO11s strong-cos 训练 | test mAP50 `0.7305`，当前推荐权重 | [training/01](training/01_strong_cos/README.md) |
| 2026-08-07 | AdamW 稳定训练 | Precision 略升，mAP/Recall 下降 | [training/02](training/02_stable_adamw/README.md) |
| 2026-08-07 | 768 低学习率微调 | val mAP50 约 `0.6054`，未超过基线 | [training/03](training/03_ft768_lowlr/README.md) |
| 2026-08-10~11 | 单帧后处理全量比较 | top-k F1 高但不适合动态人数场景 | [postprocess/01](postprocess/01_single_frame_methods/README.md) |
| 2026-08-13~14 | 可视化与 Label Studio 轨迹 GT | 默认 BoT-SORT IDF1 `0.4791` | [postprocess/02](postprocess/02_tracking_gt_evaluation/README.md) |
| 2026-08-18 | BoT-SORT 控制变量调参 | high-clean IDF1 `0.5204` | [postprocess/03](postprocess/03_botsort_tuning/README.md) |
| 2026-08-21 | val 选后处理 + tracker 输入复验 | conf .25/IoU .55，IDF1 `0.5416` | [postprocess/04](postprocess/04_validation_and_tracker_input/README.md) |
| 2026-08-24 | GPU ROI/track 延迟 | tracker `45.72 FPS` | [performance/01](performance/01_gpu_latency/README.md) |
| 2026-08-25 | ROI backbone 对比 | IDF1 提升小，MobileNet 显存最低 | [performance/02](performance/02_roi_backbone_compare/README.md) |

## 代码位置

- 训练参数透传：`framework/cleansight_eval/detection/yolo.py`、`framework/cleansight_eval/core/config.py`。
- 显式 run id：`framework/cleansight_eval/cli/train.py`、`framework/cleansight_eval/core/run.py` 及各 pipeline。
- 单帧后处理：`benchmark/cli/postprocess_detection.py`。
- 轨迹评测：`tools/evaluate_labelstudio_trackers.py`。
- 对比视频：`tools/render_tracking_comparison.py`。
- val 选后处理：`tools/explore_yolo_postprocess_next.py`。
- GPU 延迟：`tools/benchmark_gpu_roi_track_latency.py`。
- ROI backbone：`tools/compare_roi_backbones_for_tracking.py`。

## 指标口径

- 检测 mAP/P/R 来自固定 test split；后处理脚本对已保存的高召回 prediction artifact 复算。
- tracking GT 来自 Label Studio 原生 `videorectangle.sequence` 的 track id，不来自逐帧 YOLO txt。
- IDF1、ID Switches、Fragments 与框检测 F1 描述不同问题，不能只用一个指标排序所有用途。
- GPU 脚本把 ROI forward、tracker update 和 YOLO 预计算分开计时，不把 microbenchmark 称作生产端到端延迟。
