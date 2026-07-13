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

信封同时记录：`model_type / model_id / pipeline / checkpoint / dataset / feature_schema /
num_params / inference_semantics / integrity / timestamp`。

## 2. 三条评估流水线

评估入口 [cli/eval.py](../framework/cleansight_eval/cli/eval.py) 按配置中的 `pipeline` 字段
分派（组合根 `cli/_registry.py`）：

| pipeline | 任务 | 推理语义 | 适用模型 |
|---|---|---|---|
| `detection` | 单帧目标检测 | `single_frame`，无状态逐图独立推理 | YOLO |
| `sliding_window_temporal` | 实时行为分割 | `windowed_causal`，滑窗逐帧前进、取窗口末帧决策 | 因果模型（GRU） |
| `full_sequence_temporal` | 离线行为分割 | `full_sequence`，整段一次前向 | 非因果模型（MS-TCN / MS-TCN++），也可跑因果模型作离线上界 |

**滑窗 vs 全序列的意义**：同一时序模型可分别跑两条线——全序列是「看得到全部上下文」的**离线上界**，
滑窗是「只能看到历史窗口」的**实时代价**。两者数字之差即为实时化的性能损失。滑窗只接受因果模型，
非因果模型在配置校验阶段被拒。

## 3. 当前覆盖的指标

### 3.1 时序（两条时序流水线共用，[temporal/metrics.py](../framework/cleansight_eval/temporal/metrics.py)）

| 指标 | spec | 粒度 | 定义 |
|---|---|---|---|
| 帧准确率 `acc` | `acc/frame-wise/v1` | 帧级 | `100 × 正确帧 / 总帧` |
| 编辑分 `edit` | `edit/levenstein-norm/v1` | 段级 | 段序列的归一化 Levenshtein：`(1 − D/max(m,n))×100` |
| 分段 F1 `f1@0.1` / `f1@0.25` / `f1@0.5` | `segmental_f1/iou/v1` | 段级 | 按 IoU 阈值匹配段，三个重叠阈值各出一个 F1 |

- **段的定义**：`get_labels_start_end_time` 把连续同类、非 `background` 的帧归成一段。
- **三件套的分工**：`acc` 看逐帧对错，`edit` 看段序列的顺序，`f1@IoU` 看段的时间重叠质量。
- **注意**：时序侧**不**输出逐类 precision/recall（逐类 P/R 仅检测侧有）。

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

- 测量口径：warmup 20 次 + 正式 200 次，CUDA 会同步后计时；`spec` 内记录 `device/window/warmup/runs`。
- **全序列流水线**对这三项标 `not_applicable`（离线一次性推理不代表实时行为），**不造假数字**。

## 4. 指标 × 流水线 覆盖矩阵

| 指标 | 全序列时序 | 滑窗时序 | 检测 |
|---|---|---|---|
| 帧准确率 acc | ✓ | ✓ | — |
| 编辑分 edit | ✓ | ✓ | — |
| 分段 F1@0.1/0.25/0.5 | ✓ | ✓ | — |
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
`integrity: {ok, issues}`：

- **checkpoint 兼容**：只卡改变张量形状的字段（`type / input_dim / num_classes`），`window` 等可在
  eval 时覆盖不算冲突；不兼容立即报错。
- **特征维度**：实际特征维度须与期望一致（时序为 40）。
- **信封完备**：必填字段齐全，且每个 `computed` 指标都带非空 `spec`。

## 7. 矩阵聚合与可视化

- **聚合**（[core/matrix.py](../framework/cleansight_eval/core/matrix.py)）：递归扫描 `evals/*.envelope.json`，
  可按 pipeline 过滤；固定 ID 列 + 所有模型指标列的并集，逐格保留三态；渲染 Markdown 表并带 `N/A / MISSING /
  空白` 图例。
- **时序分割可视化**（[temporal/viz.py](../framework/cleansight_eval/temporal/viz.py)）：GT / Pred 双色带状图，
  逐视频对照、标注帧数与帧准确率，分页输出 PNG（默认每页 6 个视频）。

## 8. 边界（当前不做）

- 不做任何 PASS/FAIL、业务门槛或加权总分——只报原始指标，判定留给使用方。
- 时序侧不含逐类 precision/recall、不含混淆矩阵。
- 检测侧逐类只暴露 precision/recall（不含逐类 mAP）。
- 延迟只测时序滑窗单 tick，不覆盖检测推理延迟、端到端吞吐。
