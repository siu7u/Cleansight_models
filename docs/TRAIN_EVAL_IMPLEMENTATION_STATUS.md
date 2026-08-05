# CleanSight Train-Eval 实现状态

更新日期：2026-08-05。本文按当前 `feat/eval-frame` 代码记录实现事实；需求边界见
[`TRAIN_EVAL_REQUIREMENTS.md`](TRAIN_EVAL_REQUIREMENTS.md)。

## 职责边界

- framework：配置、run、训练、checkpoint、模型预测、落盘、报告、**数据契约 catalog、指标原语 metrics**。
- benchmark：testset 口径消费、指标三态翻译、PredictionOutput 评估器、结果/artifact/delivery schema。
- 外部模型管理系统或人工：版本注册、上传、发布和上线决策。
- 后端：真实 pipeline/端到端延迟与生产验收。

## 四阶段完成情况

| 阶段 | 状态 | 实现 |
|---|---|---|
| 评估职责分离 | ✅ | pipeline 仅 `predict()`；`benchmark/evaluators/` 生成正式结果 |
| 依赖方向单向 | ✅ | `benchmark → framework`；catalog/metrics 下沉到 framework core，framework 生产代码不再 import benchmark |
| 正确性契约 | ✅ | 配置 schema v1、未知字段拒绝、formal/exploratory、显式 micro/macro、检测有效参数 |
| checkpoint 与溯源 | ✅ | metadata schema v1 + SHA-256 绑定；命令、Git、依赖、CUDA/cuDNN、数据 fingerprint |
| 稳定交付 | ✅ | `delivery.manifest.json` + `schemas/*.schema.json`，不耦合复制、上传或发布 |

## 后续新增能力

- YOLO 优化实验编排：`framework/cleansight_eval/cli/sweep.py`（预设/grid，复用 YoloAdapter）。
- 小目标逐类分析与淘汰决策：`benchmark/cli/analyze.py` + `benchmark/core/analysis.py`。
- ROI 特征融合：`framework/cleansight_eval/classification/`（`roi_classification` pipeline）
  + `benchmark/evaluators/classification.py`。

## 仍保留的非阻塞事项

- 检测 artifact 只保存预测，复算指标仍需固定 testset 真值，因此 `recomputable` 不是纯单文件能力。
- 旧无 schema metadata 只能用于 `exploratory`；重新训练或补写绑定 metadata 后才能进入 `formal`。
- 混合精度和梯度累积仍按 P3 延后；训练曲线已由统一 ``history.csv`` 自动生成 PNG。
- ROI 特征融合的正式 testset 登记暂缓：先以 `exploratory` 使用，淘汰类确定后再钉定。
