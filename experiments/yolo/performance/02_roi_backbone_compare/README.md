# 2026-08-25：ROI backbone 作为 BoT-SORT ReID 特征的对比

## 简单总结

为动作模型接入 ROI embedding 前做工程筛选，本次固定 YOLO 和 high-clean tracker，只替换 ResNet18、
MobileNetV3-small、EfficientNet-B0、ConvNeXt-Tiny、DINOv2 ViT-S/14 的 per-detection embedding，并与 no-ReID
比较。所有 backbone 的 IDF1 只从 `0.4755` 提升到最高 `0.4806`，提升很小；ConvNeXt 最慢/显存最高且
ID switches 增加，MobileNet 显存最低，ResNet18 最快。

## 命令

```bash
python tools/compare_roi_backbones_for_tracking.py \
  --labelstudio datasets/labelstudio-yolo-test/project-15-at-2026-08-13-05-16-2be8e556.json \
  --image-dir datasets/cleansight-yolo/group1_large/images/test \
  --ckpt runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt \
  --tracker-config runs/labelstudio_tracker_controlled/configs/botsort_high_clean.yaml \
  --out-dir runs/roi_backbone_tracker_compare \
  --backbones resnet18,mobilenet_v3_small,efficientnet_b0,convnext_tiny,dinov2_vits14 \
  --device 0 --imgsz 640 --conf 0.25 --nms-iou 0.55 --max-det 20 \
  --iou-match 0.5 --yolo-batch 16 --roi-batch 64 --roi-padding 0.2 --fp16
```

`--max-tasks` 只用于调试。DINOv2 通过 Torch Hub 加载，离线环境需要预先准备缓存；报告会记录加载状态。

## 结果

| Backbone | ROI/s | ms/ROI | Peak MiB | F1 | IDF1 | ID switches |
|---|---:|---:|---:|---:|---:|---:|
| no-ReID | - | - | - | 0.6167 | 0.4755 | 363 |
| ResNet18 | 1092.08 | 0.9157 | 311.08 | 0.6168 | 0.4759 | 373 |
| MobileNetV3-small | 1052.60 | 0.9500 | 133.06 | 0.6172 | 0.4772 | 371 |
| EfficientNet-B0 | 712.20 | 1.4041 | 378.69 | 0.6164 | 0.4804 | 364 |
| ConvNeXt-Tiny | 492.89 | 2.0289 | 477.83 | 0.6171 | 0.4806 | 393 |
| DINOv2 ViT-S/14 | 689.95 | 1.4494 | 228.98 | 0.6167 | 0.4783 | 384 |

结论：tracker 指标不足以选择动作模型 backbone。若只看工程成本，MobileNet 显存最低、ResNet18 吞吐最高；
若用于动作识别，应固定检测/tracking 后比较动作 Frame-F1、Segment F1 和边界误差，并考虑在本领域数据上微调。

报告：`runs/roi_backbone_tracker_compare/ROI_BACKBONE_TRACKER_COMPARE_SUMMARY.md`、
`roi_backbone_tracker_compare.json`、`*_report.md`。
