# CleanSight Train-Eval 实现状态

更新日期：2026-07-17。本文按当前 `feat/eval-frame` 代码记录实现事实；需求边界见
[`TRAIN_EVAL_REQUIREMENTS.md`](TRAIN_EVAL_REQUIREMENTS.md)。

## 职责边界

- framework：配置、run、训练、checkpoint、模型预测、落盘和报告。
- benchmark：testset、指标口径、PredictionOutput 评估器、结果/artifact/delivery schema。
- 外部模型管理系统或人工：版本注册、上传、发布和上线决策。
- 后端：真实 pipeline/端到端延迟与生产验收。

## 四阶段完成情况

| 阶段 | 状态 | 实现 |
|---|---|---|
| 评估职责分离 | ✅ | pipeline 仅 `predict()`；`benchmark/evaluators/` 生成正式结果 |
| 正确性契约 | ✅ | 配置 schema v1、未知字段拒绝、formal/exploratory、显式 micro/macro、检测有效参数 |
| checkpoint 与溯源 | ✅ | metadata schema v1 + SHA-256 绑定；命令、Git、依赖、CUDA/cuDNN、数据 fingerprint |
| 稳定交付 | ✅ | `delivery.manifest.json` + `schemas/*.schema.json`，不耦合复制、上传或发布 |

## 仍保留的非阻塞事项

- 检测 artifact 只保存预测，复算指标仍需固定 testset 真值，因此 `recomputable` 不是纯单文件能力。
- 旧无 schema metadata 只能用于 `exploratory`；重新训练或补写绑定 metadata 后才能进入 `formal`。
- 混合精度和梯度累积仍按 P3 延后；训练曲线已由统一 ``history.csv`` 自动生成 PNG。
