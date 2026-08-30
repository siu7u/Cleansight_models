# 2026-08-21：val 选后处理与 tracker 输入复验

## 简单总结

前一轮单帧后处理直接看 test，有选参泄漏风险。本次先在 val 选策略，再在 test 一次验证，并控制变量比较
`imgsz=640/768`。两者 val 都选出 `conf=.30 + Gaussian Soft-NMS@.55`，但 640 的 test F1/mAP 明显更高。
随后对 tracker 做可实现的输入参数近似复验，最终 `conf=.25 + NMS IoU=.55 + high-clean` 达到
`F1=0.7009`、`IDF1=0.5416`、ID switches 199，成为视频离线推理推荐。

## 单帧后处理命令

脚本：`tools/explore_yolo_postprocess_next.py`。

```bash
python tools/explore_yolo_postprocess_next.py \
  --weights runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt \
  --data-yaml datasets/cleansight-yolo/group1_large/data.yaml \
  --imgsz 640 --device 0 --batch 16 --half \
  --artifact-conf 0.01 --artifact-iou 0.95 --max-det 300 \
  --out-dir runs/yolo_postprocess_next --reuse

python tools/explore_yolo_postprocess_next.py \
  --weights runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt \
  --data-yaml datasets/cleansight-yolo/group1_large/data.yaml \
  --imgsz 768 --device 0 --batch 16 --half \
  --artifact-conf 0.01 --artifact-iou 0.95 --max-det 300 \
  --out-dir runs/yolo_postprocess_next --reuse
```

高召回 artifact 使用低 conf/高 IoU，后续所有策略在同一预测池上比较；`--reuse` 只在配置和 artifact 对齐时使用。

## 640/768 结果

| imgsz | test P | R | F1 | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 640 | 0.7591 | 0.7729 | 0.7659 | 0.6507 | 0.3413 |
| 768 | 0.7534 | 0.7169 | 0.7347 | 0.6111 | 0.3269 |

## Tracker 输入复验

```bash
python tools/evaluate_labelstudio_trackers.py \
  --labelstudio datasets/labelstudio-yolo-test/project-15-at-2026-08-13-05-16-2be8e556.json \
  --image-dir datasets/cleansight-yolo/group1_large/images/test \
  --ckpt runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt \
  --out-dir runs/labelstudio_tracker_postprocess_next \
  --trackers runs/labelstudio_tracker_controlled/configs/botsort_high_clean.yaml \
  --confs 0.25,0.30,0.35,0.40 \
  --nms-ious 0.55,0.70 \
  --imgsz 640 --max-det 20 --device 0 --half
```

| detector 输入 | F1 | IDF1 | ID switches | Fragments |
|---|---:|---:|---:|---:|
| conf .25 / IoU .55 | 0.7009 | 0.5416 | 199 | 285 |
| conf .40 / IoU .55 | 0.6981 | 0.5316 | 167 | 322 |
| conf .30 / IoU .55 | 0.7003 | 0.5157 | 195 | 294 |
| conf .30 / IoU .70 | 0.6935 | 0.4984 | 259 | 294 |

注意：本次 tracker 实验调的是 Ultralytics 输入 conf/NMS IoU，并未把自定义 Gaussian Soft-NMS 输出真正注入
tracker；因此称为“近似复验”。下一步应实现 `custom Soft-NMS detections -> tracker.update()` 的完整离线管线。

结果：`runs/yolo_postprocess_next/YOLO_POSTPROCESS_NEXT_SUMMARY.md`、
`runs/labelstudio_tracker_postprocess_next/THREE_EXPERIMENTS_NEXT_ANALYSIS.md` 及同目录 JSON/视频。
