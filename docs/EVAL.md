# CleanSight 评估能力总览（EVAL）

> 面向：需要了解「当前能对模型评估到什么程度」的人。
> 本文只描述**已落地、可运行**的评估能力，不含规划中的指标。
> 设计准则见 [`DESIGN.md`](DESIGN.md)，需求边界见 [`TRAIN_EVAL_REQUIREMENTS.md`](TRAIN_EVAL_REQUIREMENTS.md)，
> 框架用法见 [`../framework/README.md`](../framework/README.md)。

## 0. 一句话

用「三态度量 + 口径版本(spec) + 推理语义标注」把**单帧检测**、**实时滑窗时序**、**离线全序列时序**
和 **ROI 图像分类**放进同一张评估矩阵横向比较，同时保留「哪些数字天然不可比」的信息，不折算成单一分数、
不做 PASS/FAIL 判断。

## 快速运行

以下命令从仓库根目录执行：

```bash
# 固定 testset 消费前校验
python tools/validate_testsets.py --catalog framework/testsets.yaml --json

# 单 checkpoint 评估
python -m benchmark.cli.eval \
  --config framework/experiments/gru-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/best.pt

# 跨模型汇总
python -m benchmark.cli.matrix --runs runs
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
旧 `*.envelope.json` 由 `benchmark.core.result.EvaluationResult` 的兼容读取路径处理，framework
不再保留第二个结果类型或 import 转发模块。

对应外部契约为 [`evaluation-result-v2.schema.json`](../schemas/evaluation-result-v2.schema.json)。
另外两份 Schema 分别约束 prediction artifact 和 delivery manifest。Schema 不参与指标计算；仓库
运行时仍由 `benchmark/core/result.py`、`artifacts.py`、`delivery.py` 的 Python 校验器执行检查。

## 2. 评估流水线

评估入口 [benchmark/cli/eval.py](../benchmark/cli/eval.py) 按配置中的 `pipeline` 字段，
通过 framework 的 [Pipeline 注册表](../framework/cleansight_eval/core/registry.py) 获取唯一
模型推理实现：

| pipeline | 任务 | 推理语义 | 适用模型 |
|---|---|---|---|
| `detection` | 单帧目标检测 | `single_frame`，无状态逐图独立推理 | YOLO |
| `sliding_window_temporal` | 实时行为分割 | `windowed_causal`，滑窗逐帧前进、取窗口末帧决策 | 因果模型（GRU） |
| `full_sequence_temporal` | 离线行为分割 | `full_sequence`，整段一次前向 | 非因果模型（MS-TCN / MS-TCN++ / Transformer / CLEAN ASFormer、BiGRU、BiLSTM+MS-TCN），也可跑因果模型作离线上界 |
| `roi_classification` | ROI 多标签分类 | `single_roi`，无状态逐 ROI 独立推理 | feature_fusion（CNN backbone + MLP 头），指标见 §3.5 |

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
benchmark CLI ───────► evaluation/report/delivery manifest 落盘
```

`PredictionOutput` 不包含 `MetricValue`、指标 spec、PASS/FAIL 或报告字段，因此 framework 的模型
运行能力可以被固定 benchmark 直接复用。pipeline 不再暴露正式 `evaluate()`；benchmark CLI
是唯一评测组合根。framework 负责运行模型，benchmark 负责指标、结果 schema、artifact 和落盘。

## 3. 当前覆盖的指标

### 3.1 时序（口径注册表：[framework/cleansight_eval/core/metrics.py](../framework/cleansight_eval/core/metrics.py)；三态适配：[benchmark/evaluators/temporal.py](../benchmark/evaluators/temporal.py)）

| 指标 | spec | 粒度 | 定义 |
|---|---|---|---|
| 帧准确率 `acc` | `accuracy/frame-wise-micro-across-items/percent/v3` | 帧级 | 合并所有视频帧做 micro accuracy |
| 编辑分 `edit` | `edit/levenshtein-item-macro-mean/percent/v3` | 逐视频段级 | 各视频独立计算后做 macro mean。**实现是"段标签序列"上的归一化 Levenshtein 相似度**（`edit_score`：先把逐帧标签折叠成段、**只取每段的标签、丢弃时长**，再算 `1 − 距离/max(段数)`）→ 它对**边界/时长完全不敏感**，只度量"段标签的出现顺序"，并对**多出/漏掉一段**重罚。实测：真值 `idle(20) flush(10) idle(20)` 下，预测 `idle(38) flush(1) idle(11)`（时长全错、顺序对）edit=**100.0** 而 acc 仅 78；预测多出一个 `insert` 段时 acc 98 而 edit=**60.0**。**读 edit 必须同时看段数比与非 idle 帧**，不要把它当作"边界质量"（2026-09-23 第二十四轮固化该语义） |
| 分段 F1 `f1@0.1/0.25/0.5` | `segmental_f1/...one-to-one-global-greedy-iou/percent/v4` | 段级 | 每视频独立匹配，再汇总 TP/FP/FN 做 micro F1 |
| `tp/fp/fn@0.5` | `segmental_counts/...one-to-one-global-greedy-iou/v4` | 段级 | 主阈值 0.5 的跨视频 micro 计数 |
| `precision/recall@0.5` | `segmental_precision/recall/...global-greedy.../percent/v4` | 段级 | 由跨视频汇总的 TP/FP/FN 得出 |
| `temporal_iou@0.5` | `temporal_iou/matched-segment-global-greedy.../percent/v4` | 段级 | 所有已匹配片段合并后的平均 IoU |
| `frame.macro_f1/macro_iou/micro_f1` | `classification/frame-micro-pool-per-class/percent/v3` | 帧级 | 帧池化混淆矩阵派生的分类指标 |

- 数值真源是 [`framework/cleansight_eval/core/metrics.py`](../framework/cleansight_eval/core/metrics.py)：该模块的
  `TEMPORAL_METRIC_SPECS` 是**全仓库唯一的指标定义处**（名字、spec、单位、details 路径、训练侧别名），
  benchmark 只做 0..1 到 0..100 的三态适配，训练侧与工具读同一张表（见 §3.4）。
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

### 3.4 训练期 validation 与选点口径（与 §3.1 同一注册表）

训练期的验证指标不是"另一套口径"，而是 §3.1 同一批指标的**训练侧别名**：名字、聚合方式、单位、
spec 版本全部来自唯一注册表
[`framework/cleansight_eval/core/metrics.py`](../framework/cleansight_eval/core/metrics.py) 的
`TEMPORAL_METRIC_SPECS`（声明了 `training_key` 的项即可用于选点）。

| 训练侧（`history.csv` 列 / `train.best_metric`） | 评测侧（`metrics.summary` 键） | 单位 | spec |
|---|---|---|---|
| `val_acc` | `acc` | 百分数（0..100，2 位小数） | `accuracy/frame-wise-micro-across-items/percent/v3` |
| `val_edit` | `edit` | 百分数 | `edit/levenshtein-item-macro-mean/percent/v3` |
| `val_f1_0.1` / `val_f1_0.25` / `val_f1_0.5` | `f1@0.1` / `f1@0.25` / `f1@0.5` | 百分数 | `segmental_f1/...global-greedy-iou/percent/v4` |

- **单位约定**：注册表 `unit=percent` 的指标对外恒为 0..100（四舍五入 2 位）；
  `metrics.details.temporal` 里的原始值恒为 0..1 比率（计数类为整数），需要详情时按注册表的
  `detail_path` 取值，不要各自 `*100`。
- **一致性由测试保证**：[`tests/test_metric_consistency.py`](../tests/test_metric_consistency.py) 断言
  ①评测器落盘的 spec 与注册表逐一相同、②同输入下训练侧 `val_*` 与评测侧同名指标数值完全相等、
  ③选点词表 = 注册表派生的 `training_key` 集合（不存在第二份枚举）、④两条流水线的
  `history.csv` 列覆盖全部可选指标。新增指标只在注册表登记一次即可全链路生效。
- 实测（2026-09-20）：同一 checkpoint 在 val split 上，`history.csv` 第 1 行的
  `val_acc/val_edit/val_f1_0.1/val_f1_0.25/val_f1_0.5` 与 `benchmark.cli.eval` 产出的
  `acc/edit/f1@0.1/f1@0.25/f1@0.5` **逐项相等**；统一口径前后重跑同一 checkpoint，评测
  `metrics.summary` 与 `metrics.details` **完全一致**（纯定义收敛，不改数值）。
- 范围说明：**时序**（§3.1/§3.4，注册表 `core/metrics.py`）与 **ROI 分类**（§3.5，注册表
  `core/metrics.py` 里的 `CLASSIFICATION_METRIC_SPECS`）各自有一个口径注册表，两者结构相同（`spec / unit / training_key`），
  都保证"训练侧与评测侧同名同实现"；检测（§3.2）沿用 ultralytics 原生指标口径，没有训练侧
  选点耦合。三个任务都遵循"三态 + spec + 不折算综合分"的约定。
- **统一范围的边界**："统一"指的是**进入 `evaluation.json` 的指标口径**（以及训练侧同名选点/
  history 键）。诊断工具里为自身用途现算的临场统计不在其列——例如
  `tools/visualize_predictions.py` 打印的逐序列 `frame-acc`（渲染进度）、
  `tools/quality_report.py` 的标注质量 IoU、`tools/per_tag_eval.py` 的 PR 曲线；
  它们不写进评测结果，也不参与模型间比较。

### 3.5 ROI 分类（口径注册表：[framework/cleansight_eval/core/metrics.py](../framework/cleansight_eval/core/metrics.py) 的 `CLASSIFICATION_METRIC_SPECS`）

| 指标 | spec | 粒度 | 训练侧键（history.json） |
|---|---|---|---|
| `precision` | `precision/multi-label-micro/v1` | 多标签 micro | `val_precision` |
| `recall` | `recall/multi-label-micro/v1` | 多标签 micro | `val_recall` |
| `f1` | `f1/multi-label-micro/v1` | 多标签 micro | `val_f1` |
| `exact_match` | `exact-match/multi-label/v1` | 样本级（全标签一致） | `val_exact_match`（旧别名 `val_acc` 保留，同值） |
| 逐类 P/R/F1/support | 同上 | 逐类 | — |

- **单位**：0..1 比率，保留 4 位小数（与时序侧的百分数不同，注册表里显式声明 `unit`，不靠约定俗成）。
- **判正阈值**：`DECISION_THRESHOLD = 0.5`，训练期验证与正式评测都经 `decide()` 生成 0/1 预测——
  阈值只有一处，改它不会出现"训练期 val_acc 与评测 exact_match 口径不同"。
- **同一实现**：micro/逐类 P/R/F1 与 exact_match 由 `confusion_counts → merge_counts →
  metrics_from_counts` 一条链算出；训练循环逐 batch 累加计数，评测路径一次性计算，两者数值等价
  （`tests/test_metric_consistency.py::test_classification_batch_merge_equals_oneshot`）。
- 实测（2026-09-20）：共享实现与统一前的内联公式在随机 200×3 多标签样本上**逐项相同**；
  评测器 4 个 spec 字符串与改前**逐字节一致**。
- **选点口径**（2026-09-20 起与 §3.4 对齐）：`train.best_metric` 支持
  `val_loss / val_precision / val_recall / val_f1 / val_exact_match`，词表由注册表派生
  （`CLASSIFICATION_BEST_METRICS = ("val_loss", *classification_training_keys())`），方向由
  `best_metric_mode` 给出（`val_loss` 越小越好，其余越大越好）。缺省仍是 `val_loss`，与历史行为一致；
  未注册的值在 `validate_config` 阶段直接报错。时序侧不含 `val_loss`（其时序指标是选点口径，
  早停另按 val_loss），分类侧保留它是为了兼容既有配方——这是两处唯一的有意差异。
- **端到端实测（2026-09-20）**：合成 ROI 数据集（33 个裁剪、2 类）跑真实 CLI——
  `train.best_metric=val_f1` 时 `history.json` 记录 `val_precision/val_recall/val_f1/val_exact_match`
  与旧别名 `val_acc`（同值），`status.json`/checkpoint meta 记录
  `{"name": "val_f1", "mode": "max", "value": 0.7826, "epoch": 3}`；随后 `benchmark.cli.eval`
  在同一 checkpoint 上产出 4 个注册表指标（precision 0.6852 / recall 1.0 / f1 0.8132 /
  exact_match 0.4848），spec 与注册表逐字符一致。缺省配置（不写 `best_metric`）仍按 `val_loss` 选。
- **跑通这条链路时修掉的三个既有 bug**（都在 `classification/pipeline.py`，与指标统一无关但挡住了验证）：
  ① `_fit` 用逐张量 `.detach()` 快照 `model.state_dict()`，而 `FeatureFusionModel` 不是 `nn.Module`、
  state_dict 是两层嵌套 dict → 抛 `AttributeError`（改为整体 `copy.deepcopy`）；
  ② `predict()` 重建模型时不传 `hidden_dim`/`freeze_backbone`/`dropout` → 非默认 `hidden_dim` 的
  checkpoint 直接 shape mismatch（改为优先取 checkpoint meta 的 `model` 段）；
  ③ 训练末批大小为 1 时 resnet 的 BatchNorm 报错（数据规模问题，验证时用 batch_size=5 规避，未改代码）。
- **仍未解决的环境阻塞**：`_fit` 依赖 scikit-learn，当前后端 venv 未安装；上面那次端到端验证是
  在 `PYTHONPATH=tmp/sklearn_shim`（`train_test_split` 的 numpy 等价实现）下跑的，验证的是**指标链路**，
  不是 sklearn 的精确随机流。要让分类训练在正式环境可跑，需在后端 venv 安装 scikit-learn；
  另外 `datasets/cleansight-yolo/group2_small/data.yaml` 的 `names` 是 dict 形式而
  `build_roi_dataset` 只认 list 形式，用真实数据训练前需先修其一。

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

训练侧另有选点口径（`train.best_metric`）：两条时序流水线都可用 `acc/edit/f1@0.1/f1@0.25/f1@0.5`
这五个指标选 best checkpoint（见 §3.4），检测流水线用 ultralytics 内部口径（`best.pt` = 验证集 mAP）。

ROI 分类（§3.5）不在本矩阵内：它按**多标签 micro P/R/F1 + 样本级 exact_match**（0..1 比率）评估，
与"逐帧/分段时序"不可横向折算；边界与单位见 §3.5。

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
- **离线全序列（`full_sequence_temporal`：mstcn / mstcn2 / transformer）没有任何后处理平滑**：
  `evaluation.smoothing_min_duration` **只实现在 `sliding_window_pipeline`**，离线流水线是
  **逐帧 argmax、零平滑、无冷启动填充**。两条流水线因此不可直接比大小——把因果侧（带 md 平滑）的数字
  当成离线侧的参照会得出反向结论（2026-09-22 实盘踩过：`docs/FEATURE_STRATEGY_COMPARE.md` 第十六轮
  §16.5 修正了容量报告与配图里的这处跨协议比较）。要做"同等后处理"的对照，用
  `tools/probe_offline_postprocess.py`（对已存盘 predictions 补平滑，参数在 val 上选、再报 test）。
- **段级指标的读数陷阱**：`docs/FEATURE_STRATEGY_COMPARE.md` 第十九轮给出无偏分解——离线模型匹配段里
  92% 是 idle；`boundary_mae` 只在已匹配段上统计（有偏），不能当"典型边界误差"外推上限。
  逐类帧级 P/R/F1 用 `tools/compare_runs.py --per-class` 直接产出。

## 6. 完整性检查

每个结果落盘前经 [benchmark/core/integrity.py](../benchmark/core/integrity.py) 校验，结果写入
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

- **聚合**（[benchmark/core/matrix.py](../benchmark/core/matrix.py)）：递归扫描新的
  `evals/*.evaluation.json` 和历史 `evals/*.envelope.json`，
  可按 pipeline 过滤；固定 ID 列 + 所有模型指标列的并集，逐格保留三态；渲染 Markdown 表并带 `N/A / MISSING /
  空白` 图例。
- **时序分割可视化**（[benchmark/visualizers/temporal.py](../benchmark/visualizers/temporal.py)）：GT / Pred 双色带状图，
  滑窗和全序列评估都直接消费本次 `PredictionOutput`，逐视频对照、标注帧数与帧准确率，分页输出
  `viz/segmentation-<split>-pNN.png`（默认每页 6 个视频），不会为出图重复执行模型推理；图片路径和
  SHA-256 进入 `artifacts.visualization`。可用 `evaluation.visualize: false` 关闭。
- **时序预测视频**（[tools/visualize_predictions.py](../tools/visualize_predictions.py)）：把评测时持久化的
  预测产物（`runs/<run>/artifacts/*.predictions.json`）叠加到视频帧上，顶部横幅按动作色显示**当前帧预测
  动作阶段**（Pred），下方小字显示人工真值（GT，不符时标注 MISMATCH），可选叠加 YOLO 检测框；同样不重复
  执行模型推理。用法：

  ```bash
  python tools/visualize_predictions.py \
      --artifact runs/mstcn-20260817-165320/artifacts/full_sequence_temporal-mstcn-20260817-165622.predictions.json \
      --dataset datasets/cleansight-ActionMixed \
      --images datasets/cleansight-ActionMixed/images/test \
      --out-dir outputs/visualizations
  ```
- **单命令可用性检测**（[tools/predict_timeline.py](../tools/predict_timeline.py)）：不需要先跑评测，
  直接 `.pt + 数据集 → 动作段时间线 + 带状图 + 叠加预测视频` 三样产物。模型配置从
  `<ckpt>.meta.json` sidecar 自动读取，推理走 framework 流水线公开接口；时间线把连续同类
  预测帧合并成段（stdout + `timeline_<序列>.json`，段起止为真实帧号），用于快速判断某个
  checkpoint 对某段视频的结果是否可用：

  ```bash
  python tools/predict_timeline.py \
      --ckpt runs/mstcn-20260817-165320/checkpoints/best.pt \
      --dataset datasets/cleansight-ActionMixed \
      --sequence a2ade960-clip_....mp4 \
      --images datasets/cleansight-ActionMixed/images/test \
      --out-dir outputs/visualizations/pred_check
  ```
- **checkpoint 报告**（[benchmark/core/report.py](../benchmark/core/report.py)）：每个 `.pt` 旁写
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
- benchmark CLI 负责把“调用 framework 模型运行 → benchmark 判分 → run 内文件”串起来；
  framework 不再提供第二个 eval/report/matrix 实现。
- `decision` 只允许用于明确的 benchmark 协议判定，不代表发布/上线结论。旧 `release_gate.py`
  已退出 model manager，只生成需人工审阅的事实清单。
