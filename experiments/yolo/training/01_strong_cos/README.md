# 2026-08-06：YOLO11s strong-cos 完整训练

## 简单总结

当时新 YOLO 数据集和统一训练/评测框架已就绪，但缺少正式 GPU baseline。本次从 `yolo11s.pt` 训练 30 epoch，
使用 cosine LR、强 mosaic/mixup 和较大的几何增强。test 得到 `mAP50=0.7305`、`mAP50-95=0.3897`、
`Precision=0.7195`、`Recall=0.7061`，成为后续后处理/tracker 的固定检测器。

## 配置与命令

配置：`runs/optimized_configs/yolo11s-group1-large-strong-cos.yaml`；resolved：
`runs/yolo-20260806-135408/config.resolved.json`。

```bash
python -m framework.cleansight_eval.cli.train \
  --config runs/optimized_configs/yolo11s-group1-large-strong-cos.yaml \
  --run-id yolo-20260806-135408
```

```bash
python -m benchmark.cli.eval \
  --config runs/optimized_configs/yolo11s-group1-large-strong-cos.yaml \
  --ckpt runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt
```

`--run-id` 由本分支新增，必须是字母、数字、点、下划线或短横线，不能包含路径分隔符。

核心参数：`imgsz=640`、`epochs=30`、`batch=16`、`cos_lr=true`、`close_mosaic=10`、
`mosaic=1.0`、`mixup=0.15`、`scale=0.7`、`translate=0.2`、`shear=2.0`。

## 结果

| 指标 | 值 |
|---|---:|
| mAP@0.5 | 0.7305 |
| mAP@0.5:0.95 | 0.3897 |
| Precision | 0.7195 |
| Recall | 0.7061 |

逐类 P/R：hand `0.9013/0.7786`，scope_control_body `0.7022/0.6706`，
scope_mid_section `0.5549/0.6692`。

训练曲线在第 1 epoch 出现最高 val mAP50，后续未提升，可能与小数据、强增强、预训练权重已较强和分布差异有关。
下一步用更保守 AdamW 和从 best 低学习率微调做对照，而不覆盖当前 best。

## 产物

- 权重/曲线：`runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/`。
- 评估：`runs/yolo-20260806-135408/evals/detection-yolo-20260806-153902.evaluation.json`。
- 预测：`runs/yolo-20260806-135408/artifacts/detection-yolo-20260806-153902.predictions.json`。
