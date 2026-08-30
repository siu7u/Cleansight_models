# YOLO 后处理与追踪实验

本目录按所属关系记录“单帧框过滤 → 轨迹 GT 评测 → BoT-SORT 调参 → val 选参与 tracker 输入复验”。

| 顺序 | 实验 | 结论 |
|---:|---|---|
| 1 | [单帧后处理方法](01_single_frame_methods/README.md) | top-k 离线 F1 高，但动态目标数场景不采用 |
| 2 | [Label Studio 轨迹 GT](02_tracking_gt_evaluation/README.md) | 默认 BoT-SORT 综合最好 |
| 3 | [BoT-SORT 控制变量调参](03_botsort_tuning/README.md) | high-clean 将 IDF1 提至 0.5204 |
| 4 | [val 选参与 tracker 输入复验](04_validation_and_tracker_input/README.md) | 推荐 conf .25 / IoU .55 / high-clean |

共同检测权重：`runs/yolo-20260806-135408/.../best.pt`。共同 group1 数据：
`datasets/cleansight-yolo/group1_large/`。轨迹实验只覆盖 group1 的三类大目标。

相关脚本保持在原可运行路径：

- `benchmark/cli/postprocess_detection.py`
- `tools/evaluate_labelstudio_trackers.py`
- `tools/render_tracking_comparison.py`
- `tools/explore_yolo_postprocess_next.py`
- `tools/visualize_detections.py`
