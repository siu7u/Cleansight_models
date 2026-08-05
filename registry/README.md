# 模型资产 Registry

本目录保存已登记模型的轻量交付信息，例如 `CARD.md`、`pin.yaml`、类别表、训练配置和评测报告。
训练与推理代码不放在这里，分别由 `framework` 和 `benchmark` 提供。

目录约定：

```text
registry/
├── temporal/<model-version>/
└── detection/<model-version>/
```

历史已受 Git 跟踪的时序 `.pt` 在本次布局迁移中保留，以保证旧结果可复核；新增版本不得继续
提交权重，应在 `pin.yaml` 中记录 ModelScope/对象存储位置、revision、文件名和 SHA-256。
