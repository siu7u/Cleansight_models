# 2026-08-24：GPU ROI 特征与 tracker update 延迟

## 简单总结

为评估 ROI 特征和 tracking 能否用于离线视频，本次把两段计时拆开：一段读取图片/裁 ROI/ResNet50 forward，
另一段先预计算 YOLO，再只计 BoT-SORT update。RTX 4060 Laptop 上 ROI GPU forward 吞吐 `784.91 ROI/s`，
tracker update `21.87 ms/frame`、`45.72 FPS`。端到端瓶颈不只在 GPU，还包括读图、OpenCV、GMC 和 YOLO。

## 环境与固定参数

- GPU：RTX 4060 Laptop 8188 MiB。
- PyTorch `2.11.0+cu128`、TorchVision `0.26.0+cu128`、Ultralytics `8.4.115`。
- ROI：ResNet50、224×224、batch 64。
- tracker：BoT-SORT high-clean；检测 conf .25 / IoU .55 / imgsz 640 / max_det 20。

## 命令

```bash
python tools/benchmark_gpu_roi_track_latency.py \
  --image-dir datasets/cleansight-yolo/group1_large/images/test \
  --labels-dir datasets/cleansight-yolo/group1_large/labels/test \
  --data-yaml datasets/cleansight-yolo/group1_large/data.yaml \
  --ckpt runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt \
  --tracker-config runs/labelstudio_tracker_controlled/configs/botsort_high_clean.yaml \
  --out-dir runs/gpu_latency_bench \
  --device 0 \
  --roi-size 224 --roi-padding 0.2 --roi-batch 64 --roi-fp16 \
  --track-batch 16 --track-imgsz 640 --track-conf 0.25 --track-iou 0.55 \
  --track-max-det 20 --track-fp16
```

`--limit-frames` 仅供调试；正式报告不限制。脚本会记录 GPU/软件版本和显存峰值。

## 结果

| 项目 | 数值 |
|---|---:|
| 帧数 / ROI 数 | 6786 / 20686 |
| ROI 总耗时 | 159.0011 s |
| 平均每 ROI 总耗时 | 7.6864 ms |
| GPU forward 吞吐 | 784.9140 ROI/s |
| ROI peak allocated/reserved | 404.24 / 592.00 MiB |
| tracker update 总耗时 | 148.4261 s |
| tracker update 平均每帧 | 21.8724 ms |
| tracker update 吞吐 | 45.7197 FPS |
| YOLO 预计算 | 151.9700 s |

分析：45.7 FPS 足以支持离线处理，但不能当作整条生产链路 FPS。下一步在动作模型真实批处理管线中测端到端，
并分别优化图片解码、GMC/CPU 关联和 YOLO batch。

报告：`runs/gpu_latency_bench/ROI_FEATURE_LATENCY_REPORT.md`、
`TRACK_POSTPROCESS_LATENCY_REPORT.md`、`gpu_roi_track_latency_report.{md,json}`。
