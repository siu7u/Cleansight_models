# 2026-08-18：BoT-SORT 控制变量调参

## 简单总结

默认 BoT-SORT 已是最佳 tracker，但 IDF1 仅 0.4791。本次固定 YOLO best、conf .30、NMS IoU .70，逐项比较
track buffer、match threshold、轨迹启动/维持阈值、GMC 和 score fusion。`botsort_high_clean` 通过更保守的
轨迹启动/维持阈值，将 `IDF1` 提至 `0.5204`、F1 提至 `0.6930`，ID switches 从 289 降至 255。

## 配置

推荐文件：`runs/labelstudio_tracker_controlled/configs/botsort_high_clean.yaml`。

```yaml
tracker_type: botsort
track_high_thresh: 0.35
track_low_thresh: 0.15
new_track_thresh: 0.35
track_buffer: 30
match_thresh: 0.8
fuse_score: true
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: false
model: auto
```

## 命令

```bash
python tools/evaluate_labelstudio_trackers.py \
  --labelstudio datasets/labelstudio-yolo-test/project-15-at-2026-08-13-05-16-2be8e556.json \
  --image-dir datasets/cleansight-yolo/group1_large/images/test \
  --ckpt runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt \
  --out-dir runs/labelstudio_tracker_controlled \
  --device 0 --half --imgsz 640 --max-det 20 \
  --trackers runs/labelstudio_tracker_controlled/configs/botsort_default.yaml,runs/labelstudio_tracker_controlled/configs/botsort_high_clean.yaml,runs/labelstudio_tracker_controlled/configs/botsort_match090.yaml \
  --confs 0.30 --nms-ious 0.70
```

正式报告包含更多配置，目录为 `runs/labelstudio_tracker_controlled/configs/`。

## 结果与分析

| 方法 | P | R | F1 | IDF1 | ID switches | Fragments |
|---|---:|---:|---:|---:|---:|---:|
| high-clean | 0.6461 | 0.7473 | 0.6930 | 0.5204 | 255 | 294 |
| match=.90 | 0.6243 | 0.7593 | 0.6852 | 0.5084 | 262 | 311 |
| default | 0.6352 | 0.7504 | 0.6881 | 0.4791 | 289 | 295 |

更保守启动最有效；单纯增大 buffer 没改善；放宽匹配会错关联。tracker 不能补长期漏检或拆分错误合并框。
下一步用 val 选 detector 后处理，再复验传入 tracker 的 conf/IoU。

报告：`runs/labelstudio_tracker_controlled/CONTROLLED_TRACKER_REPORT.md`、
`runs/labelstudio_tracker_controlled/labelstudio_gt_track_eval.{md,json}`；可视化：
`runs/labelstudio_tracker_controlled/task95_predict_default_optimized_botsort_trails.mp4`。
