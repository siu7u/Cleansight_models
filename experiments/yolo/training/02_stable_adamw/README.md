# 2026-08-07：AdamW 稳定训练

## 简单总结

strong-cos 的 val 峰值过早，本次降低初始学习率，改用 AdamW、5 epoch warmup、保守增强和明确 loss 权重，
观察能否稳定超过 baseline。test `Precision=0.7317` 略高，但 `mAP50=0.6854`、`Recall=0.6808` 均下降，
因此不作为最佳模型。

## 配置与命令

配置：`runs/optimized_configs/yolo11s-group1-large-stable-adamw.yaml`；run：`runs/yolo-20260806-154457/`。

```bash
python -m framework.cleansight_eval.cli.train \
  --config runs/optimized_configs/yolo11s-group1-large-stable-adamw.yaml \
  --run-id yolo-20260806-154457

python -m benchmark.cli.eval \
  --config runs/optimized_configs/yolo11s-group1-large-stable-adamw.yaml \
  --ckpt runs/yolo-20260806-154457/checkpoints/group1_large_yolo11s_stable_adamw/weights/best.pt
```

核心参数：`epochs=40`、`batch=16`、`optimizer=AdamW`、`lr0=0.0015`、`lrf=0.05`、
`weight_decay=0.01`、`warmup_epochs=5`、`box/cls/dfl=7.5/0.7/1.5`、`mosaic/mixup=0.7/0.05`。

## 结果与分析

| 指标 | strong-cos | AdamW |
|---|---:|---:|
| mAP50 | 0.7305 | 0.6854 |
| mAP50-95 | 0.3897 | 0.3707 |
| Precision | 0.7195 | 0.7317 |
| Recall | 0.7061 | 0.6808 |

更保守的优化提高了 precision，但牺牲 recall 和整体 AP，说明主要问题不是单纯训练不稳定。该 run 的
`status.json` 遗留为 `running`，但已有完整 evaluation JSON；交接时以评估产物和权重完整性为准。下一步从
strong-cos best 继续低学习率 768 微调，专门验证分辨率收益。

产物：`runs/yolo-20260806-154457/evals/detection-yolo-20260807-155210.evaluation.json`、
`runs/yolo-20260806-154457/artifacts/detection-yolo-20260807-155210.predictions.json`。
