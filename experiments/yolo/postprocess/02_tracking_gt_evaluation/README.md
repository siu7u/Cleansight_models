# 2026-08-13~14：检测可视化与 Label Studio 轨迹 GT 评测

## 简单总结

逐帧 YOLO txt 没有 track id，早期 tracker 比较无法可靠计算 IDF1。本次从 Label Studio 原生
`videorectangle.sequence` 恢复每个对象的 GT 轨迹，并与 `t{task_id}_*.jpg` 对齐，共匹配 41 个 task、4290 帧。
默认 BoT-SORT 在 F1/IDF1 的综合表现最好：`F1=0.6881`、`IDF1=0.4791`；TrackTrack 的 ID switch 少，
但 recall 降至 `0.6060`，不适合作为动作时序前端。

## 输入与脚本

- Label Studio：`datasets/labelstudio-yolo-test/project-15-at-2026-08-13-05-16-2be8e556.json`。
- 图片：`datasets/cleansight-yolo/group1_large/images/test`。
- 评测：`tools/evaluate_labelstudio_trackers.py`。
- 对比视频：`tools/render_tracking_comparison.py`。
- 普通框预览：`tools/visualize_detections.py`（本分支新增 `max_frames`）。

## 评测命令

```bash
python tools/evaluate_labelstudio_trackers.py \
  --labelstudio datasets/labelstudio-yolo-test/project-15-at-2026-08-13-05-16-2be8e556.json \
  --image-dir datasets/cleansight-yolo/group1_large/images/test \
  --ckpt runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt \
  --out-dir runs/labelstudio_track_eval \
  --imgsz 640 --max-det 20 --device 0 --half \
  --iou-match 0.5 \
  --trackers botsort.yaml,tracktrack.yaml,ocsort.yaml,deepocsort.yaml,bytetrack.yaml \
  --confs 0.15,0.25,0.30 \
  --nms-ious 0.70
```

`--iou-match` 是预测框与 GT 的评测匹配阈值；`--confs/--nms-ious` 是 detector/tracker 输入网格；
`--max-tasks` 仅用于调试，正式结果使用 0（全量）。

## 对比视频命令

```bash
python tools/render_tracking_comparison.py \
  --task-id 97 \
  --tracker botsort.yaml \
  --conf 0.30 --iou 0.70 --imgsz 640 --max-det 20 --fps 6 \
  --output runs/labelstudio_track_eval/task97_side_by_side_before_after_botsort.mp4
```

## 结果

| 方法 | P | R | F1 | IDF1 | ID switches | Fragments |
|---|---:|---:|---:|---:|---:|---:|
| BoT-SORT | 0.6352 | 0.7504 | 0.6881 | 0.4791 | 289 | 295 |
| TrackTrack | 0.6658 | 0.6060 | 0.6345 | 0.4699 | 82 | 237 |
| OC-SORT | 0.5984 | 0.7089 | 0.6490 | 0.4669 | 240 | 328 |

下一步固定检测器，只调 BoT-SORT 的轨迹启动、低分框、buffer、匹配、GMC 和 fuse score，避免跨 tracker/检测器
同时变化。报告：`runs/labelstudio_track_eval/labelstudio_gt_track_eval.{md,json}`。
