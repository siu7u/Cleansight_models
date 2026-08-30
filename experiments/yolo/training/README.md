# YOLO 检测训练实验

本目录比较同一 group1_large 数据上的三种训练路线。固定数据来源为 `datasets/cleansight-yolo/group1_large/`，
模型基于 `yolo11s.pt` 或 strong-cos best 继续训练。

| 顺序 | 实验 | 结论 |
|---:|---|---|
| 1 | [strong-cos](01_strong_cos/README.md) | test mAP50 `0.7305`，保留为最佳权重 |
| 2 | [stable AdamW](02_stable_adamw/README.md) | Precision 略高但 mAP/Recall 下降 |
| 3 | [768 低学习率微调](03_ft768_lowlr/README.md) | val 未超过 strong-cos，不采用 |

训练统一入口：`python -m framework.cleansight_eval.cli.train --config <yaml> [--run-id <id>]`；正式评估入口：
`python -m benchmark.cli.eval --config <yaml> --ckpt <best.pt>`。

实验 YAML 当前在 `runs/optimized_configs/`，该目录被忽略；交付时必须单独同步这三份 YAML，或复制到受跟踪的
配置目录并重新校验路径后再提交。
