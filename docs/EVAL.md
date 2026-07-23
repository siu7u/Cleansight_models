# CleanSight 评估能力总览（EVAL）

> 面向：需要了解「当前能对模型评估到什么程度」的人。
> 本文只描述**已落地、可运行**的评估能力，不含规划中的指标。
> 设计准则见 [`DESIGN.md`](DESIGN.md)，需求边界见 [`TRAIN_EVAL_REQUIREMENTS.md`](TRAIN_EVAL_REQUIREMENTS.md)，
> 框架用法见 [`../framework/README.md`](../framework/README.md)。

## 0. 一句话

用「三态度量 + 口径版本(spec) + 推理语义标注」把**单帧检测**、**实时滑窗时序**、**离线全序列时序**
三类异构模型放进同一张评估矩阵横向比较，同时保留「哪些数字天然不可比」的信息，不折算成单一分数、
不做 PASS/FAIL 判断。

## 快速运行

以下命令从仓库根目录执行：

```bash
# 固定 testset 消费前校验
python tools/validate_testsets.py --catalog benchmark/testsets.yaml --json

# 单 checkpoint 评估
python -m framework.cleansight_eval.cli.eval \
  --config framework/experiments/gru-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/best.pt

# 跨模型汇总
python -m framework.cleansight_eval.cli.matrix --runs runs
```

`formal` 结果必须使用校验通过的固定 testset；外部权重或临时数据使用 `exploratory`，并在报告中
保留降级事实。外部裸时序权重需同时设置 `model.allow_missing_meta: true`，由 YAML 声明模型结构；
该 fallback 不写可信 sidecar。裸 state dict、常见包装、仅含受限 NumPy normalizer 的可信包装和
TorchScript 归档最终都转换为 state dict，仍严格校验全部参数键和张量形状；未知 pickle 全局对象
不会通过 `weights_only=False` 降级加载。

## 1. 评估产物：统一 EvaluationResult

每次评估产出一个 `EvaluationResult`，正式定义位于
[`benchmark/core/result.py`](../benchmark/core/result.py)，写为
`<run>/evals/{pipeline}-{model_type}-{timestamp}.evaluation.json`。结果里的每个指标不是裸数字，
而是 `MetricValue`，区分三种状态：

| 状态 | 含义 | 典型场景 |
|---|---|---|
| `computed` | 真算出来了，附口径 `spec` | 全序列模型的帧准确率 |
| `not_applicable` | 该指标对此模型/流水线天然不适用，附 `reason` | 离线全序列模型不测单窗前向耗时 |
| `missing` | 应有但拿不到/失败，附 `reason` | 验证集无该类样本，逐类精度无法评估 |

配套 `spec`（口径版本号，任何影响数值的口径变化都递增版本）和 `reason`（为何 N/A 或 missing）。
这样矩阵里 `N/A`、`MISSING`、空白三者语义分明，**不会用 0 冒充「没测」**。

当前落盘格式为 schema v2，记录 `run / model / pipeline / testset / feature_schema /
metrics.summary / metrics.details / performance / inference / artifacts / limits / integrity`。其中
checkpoint 与 sidecar、testset manifest、prediction artifact 都记录 SHA-256，便于归档后核验。
`framework/core/envelope.py` 只保留历史 `EvalEnvelope` import 别名，不再定义第二套 schema；旧
`*.envelope.json` 仍可读取并参与矩阵汇总。

对应外部契约为 [`evaluation-result-v2.schema.json`](../schemas/evaluation-result-v2.schema.json)。
另外两份 Schema 分别约束 prediction artifact 和 delivery manifest。Schema 不参与指标计算；仓库
运行时仍由 `benchmark/core/result.py`、`artifacts.py`、`delivery.py` 的 Python 校验器执行检查。

## 2. 三条评估流水线

评估入口 [cli/eval.py](../framework/cleansight_eval/cli/eval.py) 按配置中的 `pipeline` 字段
分派（组合根 `cli/_registry.py`）：

| pipeline | 任务 | 推理语义 | 适用模型 |
|---|---|---|---|
| `detection` | 单帧目标检测 | `single_frame`，无状态逐图独立推理 | YOLO |
| `sliding_window_temporal` | 实时行为分割 | `windowed_causal`，滑窗逐帧前进、取窗口末帧决策 | 因果模型（GRU） |
| `full_sequence_temporal` | 离线行为分割 | `full_sequence`，整段一次前向 | 非因果模型（MS-TCN / MS-TCN++ / Transformer / CLEAN ASFormer、BiGRU、BiLSTM+MS-TCN），也可跑因果模型作离线上界 |

**滑窗 vs 全序列的意义**：同一种因果网络结构可以分别在两条流水线中建立独立实验——全序列用于
观察完整上下文下的离线表现，滑窗用于评估只能看到历史窗口的在线语义。一个 checkpoint 的训练和
评估仍必须属于同一条流水线，不能把固定窗口训练的权重直接当成任意长全序列模型。滑窗只接受因果
模型，非因果模型在配置校验阶段被拒。

### 2.1 模型执行与指标判分边界

三条 pipeline 都实现同一个 duck-type 方法 `predict(cfg, ckpt, device)`，返回
[`PredictionOutput`](../framework/cleansight_eval/core/execution.py)：

```text
checkpoint + dataset
        │
        ▼
pipeline.predict() ──► predictions / targets / labels / native_metrics / raw timing
        │
        ▼
benchmark evaluator ─► metrics + EvaluationResult + prediction artifact
        │
        ▼
framework CLI ───────► evaluation/report/delivery manifest 落盘
```

`PredictionOutput` 不包含 `MetricValue`、指标 spec、PASS/FAIL 或报告字段，因此 framework 的模型
运行能力可以被固定 benchmark 直接复用。pipeline 不再暴露正式 `evaluate()`；CLI 是唯一组合根。
framework 负责运行模型和 run 内落盘，benchmark 负责指标、结果 schema 和 artifact。

## 3. 当前覆盖的指标

### 3.1 时序（真源：[benchmark/evaluators/temporal.py](../benchmark/evaluators/temporal.py)）

| 指标 | spec | 粒度 | 定义 |
|---|---|---|---|
| 帧准确率 `acc` | `accuracy/frame-wise-micro-across-items/percent/v3` | 帧级 | 合并所有视频帧做 micro accuracy |
| 编辑分 `edit` | `edit/levenshtein-item-macro-mean/percent/v3` | 逐视频段级 | 各视频独立计算后做 macro mean |
| 分段 F1 `f1@0.1/0.25/0.5` | `segmental_f1/...one-to-one-global-greedy-iou/percent/v4` | 段级 | 每视频独立匹配，再汇总 TP/FP/FN 做 micro F1 |
| `tp/fp/fn@0.5` | `segmental_counts/...one-to-one-global-greedy-iou/v4` | 段级 | 主阈值 0.5 的跨视频 micro 计数 |
| `precision/recall@0.5` | `segmental_precision/recall/...global-greedy.../percent/v4` | 段级 | 由跨视频汇总的 TP/FP/FN 得出 |
| `temporal_iou@0.5` | `temporal_iou/matched-segment-global-greedy.../percent/v4` | 段级 | 所有已匹配片段合并后的平均 IoU |
| `frame.macro_f1/macro_iou/micro_f1` | `classification/frame-micro-pool-per-class/percent/v3` | 帧级 | 帧池化混淆矩阵派生的分类指标 |

- 数值真源是 [`benchmark/core/metrics.py`](../benchmark/core/metrics.py)，framework 只做 0..1 到
  0..100 的三态适配。
- `metrics.summary` 保留主指标；所有 IoU 阈值详情、逐类 P/R/F1/IoU 和混淆矩阵放在
  `metrics.details.temporal`，避免矩阵横向无限膨胀。
- 所有视频保持独立边界，禁止把不同视频先拼成一条序列再算 Edit/F1。
- 3 分钟端到端动作时间线复用同一个区间匹配核心和 0.1/0.25/0.5 阈值；区别仅在于时序模型以帧
  为区间单位，端到端以秒为区间单位，并额外保留业务结果与边界误差 PASS/FAIL 门禁。

### 3.2 检测（真源：[benchmark/evaluators/detection.py](../benchmark/evaluators/detection.py)）

| 指标 | spec | 粒度 |
|---|---|---|
| `mAP@0.5` | `map/coco-0.5/v1` | 整体 |
| `mAP@0.5:0.95` | `map/coco-0.5:0.95/v1` | 整体 |
| `precision` / `recall` | `precision/detection-iou0.5/v1`、`recall/detection-iou0.5/v1` | 整体（IoU 0.5） |
| `metrics.details.per_class.<类名>.precision/recall` | 同上 | 逐类 |

- 逐类指标遍历 `data.yaml` 声明的全部类别：验证集**有样本** → `computed`；**无样本** → `missing`
  （标 `验证集无该类样本，无法评估`，而非 0）。
- `metrics.summary` 只保留整体 mAP/P/R；逐类结果及共享口径放进 `metrics.details`，避免主结果和矩阵按类别无限展开。
- 底层复用 ultralytics `val()`，本模块只把结果翻译成三态结果，不含任何业务门槛或 PASS/FAIL。

### 3.3 模型前向耗时（仅滑窗流水线）

| 指标 | spec | 说明 |
|---|---|---|
| `model_forward_mean_ms` / `model_forward_median_ms` / `model_forward_p95_ms` | `latency/model-forward-single-window/ms/v2` | 单窗口 `[1, window, input_dim]` 模型前向 |

- 执行层保存 warmup 20 次 + 正式 200 次的逐次原始样本，CUDA 会同步后计时；评估层再汇总
  mean/median/p95，并在 `spec` 内记录 `device/window/warmup/runs`。采样 scope 明确为
  `model_forward_single_window`，不含数据加载、特征提取和报告写盘。
- **全序列流水线**对这三项标 `not_applicable`（离线一次性推理不代表实时行为），**不造假数字**。

## 4. 指标 × 流水线 覆盖矩阵

| 指标 | 全序列时序 | 滑窗时序 | 检测 |
|---|---|---|---|
| 帧准确率 acc | ✓ | ✓ | — |
| 编辑分 edit | ✓ | ✓ | — |
| 分段 F1@0.1/0.25/0.5 | ✓ | ✓ | — |
| TP/FP/FN、P/R、Temporal IoU@0.5 | ✓ | ✓ | — |
| 帧级 macro/micro、逐类 P/R/F1/IoU | ✓ | ✓ | — |
| mAP@0.5 / @0.5:0.95 | — | — | ✓ |
| precision / recall（整体） | — | — | ✓ |
| precision/recall（逐类） | — | — | ✓ / MISSING |
| 延迟 mean/median/p95 | N/A | ✓ | N/A |

`✓`=computed，`N/A`=not_applicable，`—`=该指标在此流水线下不产出（矩阵中留空）。

## 5. 输入契约与因果处理

- **时序特征契约** `actionmixed-bbox-8cls-v1`（[temporal/data.py](../framework/cleansight_eval/temporal/data.py)）：
  每帧 8 类检测框 × 5 维（`presence, cx, cy, w, h`，每类取最大面积框）= **40 维**。
  目录约定 `labels/<split>/*.txt`（逐帧动作 id）+ `frames/<split>/*.txt`（YOLO 框）。
  可选的 `feature_schema.mask_targets` 用于固定消融，接受 `frames/data.yaml` 中的目标名或类别 ID，只把对应
  类别的 5 维清零，不改变 40 维输入形状；训练时写入 resolved config 与 checkpoint metadata，
  评估时写入结果的 feature schema，保证遮罩实验可追溯。
- **滑窗因果推理**：冷启动前 `window−1` 帧填 idle；每视频重置状态；`causal_decision` 做因果平滑
  （`MIN_DURATION=25` 帧最小持续时长；仅在 3 类 Idle/Long/Short 时叠加类别转移先验，其他类别数退化为
  仅最小持续时长平滑）。

## 6. 完整性检查

每个结果落盘前经 [core/integrity.py](../framework/cleansight_eval/core/integrity.py) 校验，结果写入
`integrity: {ok, checks, issues}`：

- **checkpoint 兼容**：metadata schema v1 绑定 checkpoint SHA-256/大小；同时检查改变张量形状的字段（`type / input_dim / num_classes`），`window` 等可在
  eval 时覆盖不算冲突；时序 checkpoint 还记录 dataset version/revision、train/val/eval split
  fingerprint 和动作/检测映射摘要，resume 时 train fingerprint 漂移会被拒绝。
- **特征维度**：实际特征维度须与期望一致（时序为 40）。
- **结果完备**：必填字段齐全，且每个 `computed` 指标都带非空 `spec`。
- **testset 固定**：时序正式配置通过 `data.dataset_ref + split_eval` 唯一推导 testset；检测配置
  可继续显式使用 `evaluation.testset_id`。结果记录
  manifest hash 和复合 fingerprint，并按 testset 的 `split_overlap_policy` 执行或显式放宽
  train/val/test 泄漏检查；默认 `error` 按源视频隔离，`frame` 允许同源分段但禁止具体帧重复，
  `allow` 完全放宽跨 split 门禁。清单 v2 的 `datasets` 保存公共数据契约，`testsets` 只保存 split
  级样本清单和用途。ActionMixed loader 只读取 manifest 项，validator 同时要求 manifest 与
  `labels/<split>` 严格一致，并将动作标签和逐帧 bbox 内容纳入 fingerprint。
- **artifact 可追溯**：要求逐视频/逐图 prediction artifact 存在并带 SHA-256；时序 artifact
  还会实际调用 benchmark 复算，确认 `recomputable=true`。
- **评估 profile**：`formal` 必须使用已登记且校验通过的 testset 和绑定 metadata；
  `exploratory` 允许外部权重或 ad-hoc 数据；显式开启 `model.allow_missing_meta` 后，时序 Pipeline
  可按 YAML 重建裸 state dict，但结果会记录 metadata 来源和未绑定状态。

## 7. 矩阵聚合与可视化

- **聚合**（[core/matrix.py](../framework/cleansight_eval/core/matrix.py)）：递归扫描新的
  `evals/*.evaluation.json` 和历史 `evals/*.envelope.json`，
  可按 pipeline 过滤；固定 ID 列 + 所有模型指标列的并集，逐格保留三态；渲染 Markdown 表并带 `N/A / MISSING /
  空白` 图例。
- **时序分割可视化**（[temporal/viz.py](../framework/cleansight_eval/temporal/viz.py)）：GT / Pred 双色带状图，
  滑窗和全序列评估都直接消费本次 `PredictionOutput`，逐视频对照、标注帧数与帧准确率，分页输出
  `viz/segmentation-<split>-pNN.png`（默认每页 6 个视频），不会为出图重复执行模型推理；图片路径和
  SHA-256 进入 `artifacts.visualization`。可用 `evaluation.visualize: false` 关闭。
- **checkpoint 报告**（[core/report.py](../framework/cleansight_eval/core/report.py)）：每个 `.pt` 旁写
  `<checkpoint>.eval.md`，并向同目录唯一的 `EVALUATION_REPORT.md` 追加版本记录。
- **稳定交付清单**：每次评估写 `*.delivery.manifest.json`，列出 checkpoint、metadata、evaluation、
  artifact 和报告的相对路径、大小、SHA-256 与内容 schema。独立 JSON Schema 位于 `schemas/`。
- **prediction artifact**：schema 与校验统一位于
  [`benchmark/core/artifacts.py`](../benchmark/core/artifacts.py)。时序保存逐视频预测与真值并支持
  独立复算；检测保存逐图类别、置信度与归一化框，结合固定 testset 真值后复算指标。

## 8. 边界（当前不做）

- 不做任何 PASS/FAIL、业务门槛或加权总分——只报原始指标，判定留给使用方。
- 检测侧逐类只暴露 precision/recall（不含逐类 mAP）。
- 本仓库只测时序滑窗单窗模型前向，不覆盖检测推理延迟、pipeline 或端到端吞吐。
- framework CLI 仍负责把“模型运行 → benchmark 结果 → run 内文件”串起来；benchmark 已成为
  指标、testset、`EvaluationResult v2` 及两类 prediction artifact 的唯一真源。
- `decision` 只允许用于明确的 benchmark 协议判定，不代表发布/上线结论。旧 `release_gate.py`
  已退出 model manager，只生成需人工审阅的事实清单。
