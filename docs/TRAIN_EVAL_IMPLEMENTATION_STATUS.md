# CleanSight Train-Eval 实现状态

更新日期：**2026-09-24**（原 2026-08-05）。本文按当前 `feat/roi-training` 分支代码记录实现事实；
需求边界见 [`TRAIN_EVAL_REQUIREMENTS.md`](TRAIN_EVAL_REQUIREMENTS.md)。

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

### 2026-09-24 追记（本周新增）

- **指标口径唯一注册表**：`framework/cleansight_eval/core/metrics.py` 的 `TEMPORAL_METRIC_SPECS` /
  `CLASSIFICATION_METRIC_SPECS` 成为全仓库唯一指标定义处（`spec / unit / training_key`）；
  训练侧 `val_*` 与评测侧同名指标**同输入下数值相等**，由 `tests/test_metric_consistency.py`
  （15 个测试 / 53 条断言）锁定；**选点词表由注册表派生**（3 → 5 个，不再手写枚举）。
- **训练/数据旋钮**：`data.train_video_fraction`（数据规模轴）、`model.sequence_normalization`
  （`none`/`demean`）、`train.class_weight_clip`。
- **探针工具链**：`tools/probe_{channel_subsets,boundary_error,selection_transfer,seed_ensemble,
  segment_visibility,offline_postprocess}.py`（各配单测）与 `tools/compare_runs.py`
  —— 后者把"逐 (seed, 视频) 配对 Wilcoxon + 段数比 + 非 idle 帧"固化成一条命令。
- **报告图表库**：`docs/figures/`（7 张图 + JSON 旁证）+ `tools/plot_report_figures.py`，
  数据一致性纪律见 [`figures/README.md`](figures/README.md) §3。

## 仍保留的非阻塞事项

- 检测 artifact 只保存预测，复算指标仍需固定 testset 真值，因此 `recomputable` 不是纯单文件能力。
- 旧无 schema metadata 只能用于 `exploratory`；重新训练或补写绑定 metadata 后才能进入 `formal`。
- 混合精度和梯度累积仍按 P3 延后；训练曲线已由统一 ``history.csv`` 自动生成 PNG。
- ROI 特征融合的正式 testset 登记暂缓：先以 `exploratory` 使用，淘汰类确定后再钉定。
- **`--resume` 语义错位（已知未修，2026-09-22 记录）**：会导致"以为续训、实际未续训"，
  属**会污染实验结论**的隐患，不只是易用性问题。
- **架构门禁红灯（2026-09-24）**：`tests/test_architecture_boundaries.py` 2 条失败，12 个违规文件中
  10 个来自**已提交旧文件**（HEAD 上即红）；需一次决策：放宽 `tools/` 的运行规则，还是把执行模型的
  工具下沉到允许层。
- **测试基线 7 红 / 399 绿**（2026-09-24，28s）：除上述 2 条门禁外，`test_config_paths` /
  `test_temporal_masking` / `test_pipeline_smoke` 为**预先存在**（2026-09-11 已记录），
  `test_predict_timeline` 属**数据集版本漂移**（引用旧视频名），`test_checkpoint_compat`
  为 torchscript 归档相关（环境/权重）。
- **学习曲线协议待改**：`data.train_video_fraction` 的曲线**不单调**，结论暂不可用，
  协议需改为随机子集（记录见 [`mstcn-capacity/MSTCN_CAPACITY_STUDY.md`](mstcn-capacity/MSTCN_CAPACITY_STUDY.md) §0.2）。
