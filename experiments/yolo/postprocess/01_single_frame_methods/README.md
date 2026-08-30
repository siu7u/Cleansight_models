# 2026-08-10~11：YOLO 单帧后处理方法测试

## 简单总结

strong-cos 权重已固定，但需要在不重训的前提下平衡 precision/recall。本次对保存的高召回 prediction artifact
比较全局/逐类阈值、class-aware/agnostic NMS、Soft-NMS、WBF、top-k、面积过滤和 max-det。逐类阈值+top-k
离线 F1 最高 `0.7776`，但真实视频可能出现额外人员/4 只手，固定目标数会错误删除有效框，因此主线转向
不限制动态目标数的阈值 + NMS/Soft-NMS。

## 脚本与命令

脚本：`benchmark/cli/postprocess_detection.py`。它只读取已保存 artifact，不运行 YOLO，也不修改 checkpoint。

```bash
python -m benchmark.cli.postprocess_detection \
  --predictions runs/yolo-20260806-135408/artifacts/detection-yolo-20260806-153902.predictions.json \
  --labels-dir datasets/cleansight-yolo/group1_large/labels/test \
  --official-eval runs/yolo-20260806-135408/evals/detection-yolo-20260806-153902.evaluation.json \
  --out-json runs/postprocess_reports/yolo11s-strong-cos-best-postprocess.json \
  --out-md runs/postprocess_reports/yolo11s-strong-cos-best-postprocess.md \
  --out-splits-csv runs/postprocess_reports/yolo11s-strong-cos-best-postprocess-splits.csv
```

参数：`--predictions` 是逐图预测 artifact；`--labels-dir` 是同 split YOLO 真值；`--official-eval` 可选，用于对照
Ultralytics 正式结果；三个 `--out-*` 分别写结构化结果、人读报告和逐 split/图片统计。

## 结果

| 策略 | mAP50 | mAP50-95 | P | R | F1 | 框数 |
|---|---:|---:|---:|---:|---:|---:|
| 逐类阈值 + top-k | 0.6169 | 0.3296 | 0.8167 | 0.7421 | 0.7776 | 18797 |
| 逐类阈值 + NMS@0.55 | 0.6193 | 0.3307 | 0.8050 | 0.7454 | 0.7741 | 19153 |
| global conf=.30 + NMS@.55 | 0.6508 | 0.3416 | 0.7524 | 0.7745 | 0.7633 | 21294 |

分析：top-k 是对当前数据分布的强先验，在线/真实视频鲁棒性差；后续使用 val 而非 test 选参数，并加入
Gaussian Soft-NMS。轨迹质量不能从单帧 F1 推断，需要恢复真实 track id 单独评估。

## 结果文件

- `runs/postprocess_reports/yolo11s-strong-cos-best-postprocess.{md,json}`。
- `runs/postprocess_reports/yolo11s-strong-cos-best-postprocess-full.{md,json}`。
- `runs/postprocess_reports/yolo11s-strong-cos-best-postprocess-splits.csv`。
- `runs/postprocess_reports/yolo11s-strong-cos-best-trackers.{md,json}`（早期无原生 track GT，仅作探索）。
