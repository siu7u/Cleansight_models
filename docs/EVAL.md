# CleanSight 评估能力总览（EVAL）

> 面向：需要了解「当前能对模型评估到什么程度」的人。
> 本文只描述**已落地、可运行**的评估能力，不含规划中的指标。
> 设计准则见 [`DESIGN.md`](DESIGN.md)，需求边界见 [`TRAIN_EVAL_REQUIREMENTS.md`](TRAIN_EVAL_REQUIREMENTS.md)，
> 框架用法见 [`../framework/README.md`](../framework/README.md)。

## 0. 一句话

用「三态度量 + 口径版本(spec) + 推理语义标注」把**单帧检测**、**实时滑窗时序**、**离线全序列时序**
三类异构模型放进同一张评估矩阵横向比较，同时保留「哪些数字天然不可比」的信息，不折算成单一分数、
不做 PASS/FAIL 判断。

## 1. 评估产物：三态信封

每次评估产出一个 `EvalEnvelope`（[core/envelope.py](../framework/cleansight_eval/core/envelope.py)），
写为 `<run>/evals/{pipeline}-{model_type}-{timestamp}.envelope.json`。信封里的每个指标不是裸数字，
而是 `MetricValue`，区分三种状态：

| 状态 | 含义 | 典型场景 |
|---|---|---|
| `computed` | 真算出来了，附口径 `spec` | 全序列模型的帧准确率 |
| `not_applicable` | 该指标对此模型/流水线天然不适用，附 `reason` | 离线全序列模型没有单 tick 延迟 |
| `missing` | 应有但拿不到/失败，附 `reason` | 验证集无该类样本，逐类精度无法评估 |

配套 `spec`（口径版本号，任何影响数值的口径变化都递增版本）和 `reason`（为何 N/A 或 missing）。
这样矩阵里 `N/A`、`MISSING`、空白三者语义分明，**不会用 0 冒充「没测」**。

当前落盘格式为 schema v2，记录 `run / model / pipeline / testset / feature_schema /
metrics.summary / metrics.details / performance / inference / artifacts / limits / integrity`。其中
checkpoint 与 sidecar、testset manifest、prediction artifact 都记录 SHA-256，便于归档后核验。

## 2. 三条评估流水线

评估入口 [cli/eval.py](../framework/cleansight_eval/cli/eval.py) 按配置中的 `pipeline` 字段
分派（组合根 `cli/_registry.py`）：

| pipeline | 任务 | 推理语义 | 适用模型 |
|---|---|---|---|
| `detection` | 单帧目标检测 | `single_frame`，无状态逐图独立推理 | YOLO |
| `sliding_window_temporal` | 实时行为分割 | `windowed_causal`，滑窗逐帧前进、取窗口末帧决策 | 因果模型（GRU） |
| `full_sequence_temporal` | 离线行为分割 | `full_sequence`，整段一次前向 | 非因果模型（MS-TCN / MS-TCN++ / Transformer），也可跑因果模型作离线上界 |

**滑窗 vs 全序列的意义**：同一时序模型可分别跑两条线——全序列是「看得到全部上下文」的**离线上界**，
滑窗是「只能看到历史窗口」的**实时代价**。两者数字之差即为实时化的性能损失。滑窗只接受因果模型，
非因果模型在配置校验阶段被拒。

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
pipeline.evaluate() ─► metrics + EvalEnvelope + prediction artifact
```

`PredictionOutput` 不包含 `MetricValue`、指标 spec、PASS/FAIL 或报告字段，因此 framework 的模型
运行能力可以被固定 benchmark 直接复用。现有 CLI 仍调用 `evaluate()`；它内部只消费 `predict()`
输出进行兼容判分，避免一次迁移同时破坏已有 JSON、报告和矩阵。

## 3. 当前覆盖的指标

### 3.1 时序（两条时序流水线共用，[temporal/metrics.py](../framework/cleansight_eval/temporal/metrics.py)）

| 指标 | spec | 粒度 | 定义 |
|---|---|---|---|
| 帧准确率 `acc` | `accuracy/frame-wise/percent/v2` | 帧级 | `100 × 正确帧 / 总帧` |
| 编辑分 `edit` | `edit/levenshtein-item-mean/percent/v2` | 逐视频段级 | 各视频分别算段序列 Levenshtein，再按视频平均 |
| 分段 F1 `f1@0.1/0.25/0.5` | `segmental_f1/label-aware-one-to-one-iou/percent/v2` | 段级 | 同类别片段按 IoU 一对一匹配 |
| `tp/fp/fn@0.5` | `segmental_counts/label-aware-one-to-one-iou/v2` | 段级 | 主阈值 0.5 的匹配计数 |
| `precision/recall@0.5` | `segmental_precision/recall/.../percent/v2` | 段级 | 由主阈值 TP/FP/FN 得出 |
| `temporal_iou@0.5` | `temporal_iou/matched-segment-mean/percent/v2` | 段级 | 已匹配片段的平均 IoU |
| `frame.macro_f1/macro_iou/micro_f1` | `classification/per-class/percent/v2` | 帧级 | 混淆矩阵派生的总体分类指标 |

- 数值真源是 [`benchmark/core/metrics.py`](../benchmark/core/metrics.py)，framework 只做 0..1 到
  0..100 的三态适配。
- `metrics.summary` 保留主指标；所有 IoU 阈值详情、逐类 P/R/F1/IoU 和混淆矩阵放在
  `metrics.details.temporal`，避免矩阵横向无限膨胀。
- 所有视频保持独立边界，禁止把不同视频先拼成一条序列再算 Edit/F1。

### 3.2 检测（[detection/metrics.py](../framework/cleansight_eval/detection/metrics.py)）

| 指标 | spec | 粒度 |
|---|---|---|
| `mAP@0.5` | `map/coco-0.5/v1` | 整体 |
| `mAP@0.5:0.95` | `map/coco-0.5:0.95/v1` | 整体 |
| `precision` / `recall` | `precision/detection-iou0.5/v1`、`recall/detection-iou0.5/v1` | 整体（IoU 0.5） |
| `precision:<类名>` / `recall:<类名>` | 同上 | 逐类 |

- 逐类指标遍历 `data.yaml` 声明的全部类别：验证集**有样本** → `computed`；**无样本** → `missing`
  （标 `验证集无该类样本，无法评估`，而非 0）。
- 底层复用 ultralytics `val()`，本模块只把结果翻译成三态信封，不含任何业务门槛或 PASS/FAIL。

### 3.3 实时延迟（仅滑窗流水线）

| 指标 | spec | 说明 |
|---|---|---|
| `latency_mean_ms` / `latency_median_ms` / `latency_p95_ms` | `latency/single_tick_ms/v1` | 单窗口 `[1, window, input_dim]` 前向、取末帧的一 tick 耗时 |

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
- **滑窗因果推理**：冷启动前 `window−1` 帧填 idle；每视频重置状态；`causal_decision` 做因果平滑
  （`MIN_DURATION=25` 帧最小持续时长；仅在 3 类 Idle/Long/Short 时叠加类别转移先验，其他类别数退化为
  仅最小持续时长平滑）。

## 6. 完整性检查

每个信封落盘前经 [core/integrity.py](../framework/cleansight_eval/core/integrity.py) 校验，结果写入
`integrity: {ok, checks, issues}`：

- **checkpoint 兼容**：只卡改变张量形状的字段（`type / input_dim / num_classes`），`window` 等可在
  eval 时覆盖不算冲突；不兼容立即报错。
- **特征维度**：实际特征维度须与期望一致（时序为 40）。
- **信封完备**：必填字段齐全，且每个 `computed` 指标都带非空 `spec`。
- **testset 固定**：正式配置通过 `evaluation.testset_id` 关联 `benchmark/testsets.yaml`，记录
  manifest hash 和复合 fingerprint，并执行 train/val/test 源视频泄漏检查。
- **artifact 可追溯**：要求逐视频/逐图 prediction artifact 存在并带 SHA-256；时序 artifact
  还会实际调用 benchmark 复算，确认 `recomputable=true`。

## 7. 矩阵聚合与可视化

- **聚合**（[core/matrix.py](../framework/cleansight_eval/core/matrix.py)）：递归扫描 `evals/*.envelope.json`，
  可按 pipeline 过滤；固定 ID 列 + 所有模型指标列的并集，逐格保留三态；渲染 Markdown 表并带 `N/A / MISSING /
  空白` 图例。
- **时序分割可视化**（[temporal/viz.py](../framework/cleansight_eval/temporal/viz.py)）：GT / Pred 双色带状图，
  逐视频对照、标注帧数与帧准确率，分页输出 PNG（默认每页 6 个视频）。
- **checkpoint 报告**（[core/report.py](../framework/cleansight_eval/core/report.py)）：每个 `.pt` 旁写
  `<checkpoint>.eval.md`，并向同目录唯一的 `EVALUATION_REPORT.md` 追加版本记录。
- **prediction artifact**：时序保存逐视频预测与真值并支持复算；检测保存逐图类别、置信度与归一化框。

## 8. 边界（当前不做）

- 不做任何 PASS/FAIL、业务门槛或加权总分——只报原始指标，判定留给使用方。
- 检测侧逐类只暴露 precision/recall（不含逐类 mAP）。
- 延迟只测时序滑窗单 tick，不覆盖检测推理延迟、端到端吞吐。
- framework 与 benchmark 的最终职责拆分尚未完成；当前正式评估仍由 framework CLI 编排，
  benchmark 已作为时序指标、testset 和时序 artifact 的公共真源。
