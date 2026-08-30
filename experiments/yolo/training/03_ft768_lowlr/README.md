# 2026-08-07：strong-cos best 的 768 低学习率微调

## 简单总结

为改善小目标和贴近目标，从 strong-cos best 继续训练，输入增至 768，使用低学习率、弱增强和 rect batch。
训练成功，但最佳 val mAP50 约 `0.6054`，低于 strong-cos，说明仅提高分辨率没有带来收益，后续评测/后处理
继续固定 `imgsz=640`。

## 配置与命令

配置：`runs/optimized_configs/yolo11s-group1-large-ft768-lowlr-lowaug.yaml`；run：
`runs/yolo11s-ft768-lowlr-lowaug/`。

```bash
python -m framework.cleansight_eval.cli.train \
  --config runs/optimized_configs/yolo11s-group1-large-ft768-lowlr-lowaug.yaml \
  --run-id yolo11s-ft768-lowlr-lowaug
```

核心参数：weights 指向 strong-cos best，`imgsz=768`、`epochs=24`、`batch=8`、`optimizer=SGD`、
`lr0/lrf=0.001/0.2`、`warmup_epochs=2`、`rect=true`、`mosaic=mixup=0`。

## 结果与后续

最佳权重位于
`runs/yolo11s-ft768-lowlr-lowaug/checkpoints/group1_large_yolo11s_ft768_lowlr_lowaug_frombest/weights/best.pt`，
训练曲线在同目录 `results.png`。val 未超过 baseline，未继续做正式发布候选。

下一步转向不重训模型的后处理：在保存的高召回预测上比较阈值、NMS、Soft-NMS、WBF、top-k 和面积过滤；
同时在后续 val/test 实验再次控制变量比较 640 与 768。
