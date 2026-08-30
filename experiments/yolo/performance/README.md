# YOLO / Tracker / ROI 性能实验

本目录回答两个工程问题：ROI 特征提取和 tracker 分别需要多少时间/显存；哪些通用 ROI backbone 适合作为
BoT-SORT ReID 外观特征。两项实验都不是生产端到端延迟，也不代表动作模型最终收益。

| 顺序 | 实验 | 结论 |
|---:|---|---|
| 1 | [GPU ROI 与 tracker 延迟](01_gpu_latency/README.md) | tracker update 约 45.72 FPS，CPU/GMC/读图是重要开销 |
| 2 | [ROI backbone 对比](02_roi_backbone_compare/README.md) | tracker IDF1 增益很小，需按动作任务复验 |

脚本：`tools/benchmark_gpu_roi_track_latency.py`、`tools/compare_roi_backbones_for_tracking.py`。
