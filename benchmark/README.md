# CleanSight Benchmark

`benchmark/` 是评估定义层，不负责训练模型。它统一维护评测口径消费、指标三态翻译、
PredictionOutput 评估器、`EvaluationResult v2`、prediction artifact、报告、矩阵和
delivery manifest。

单模型评测入口是 `benchmark.cli.eval`。benchmark 负责组织评测、判分和落盘；需要加载
checkpoint 或执行模型时，调用 framework 唯一的 Pipeline 推理能力，不重新实现模型。
另有 `benchmark.cli.analyze` 做小目标逐类阈值分析与淘汰决策。

> **职责边界**：数据契约（`framework/testsets.yaml` → `framework/cleansight_eval/core/catalog.py`）
> 与指标原语（`framework/cleansight_eval/core/metrics.py`）归 framework；benchmark 消费它们，
> 不重新解析 catalog 或复制指标算法。依赖方向单向 `benchmark → framework`。

## Benchmark 层级

| 层级 | 目录/入口 | 回答的问题 |
|---|---|---|
| 单模型 | `benchmark.cli.eval`、`benchmark/single_model/` | 单个 YOLO 或时序模型本身表现如何 |
| 逐类分析 | `benchmark.cli.analyze` | 小目标各类 P/R 是否达标、哪些类应淘汰转特征融合 |
| Feed mode | `benchmark/temporal_feed_mode/` | 全序列与流式/滑窗推理是否一致、损失多大 |
| 端到端场景 | `benchmark/e2e_3min/` | 完整业务流程的动作、阶段和告警是否正确 |

三类结论不能混用。单模型指标不能替代后端端到端验收，smoke test 也不能作为正式 benchmark。

## 职责边界

```text
benchmark cli.eval
        ↓ 调用
framework pipeline.predict()
        ↓ PredictionOutput
benchmark/evaluators/{detection,temporal,classification}.py
        ↓ EvaluationResult + pending prediction artifact
benchmark cli.eval
        ↓ evaluation/artifact/report/delivery 落盘
```

- `framework/cleansight_eval/core/metrics.py`：时序指标数值真源（framework 提供，benchmark 消费）；
- `benchmark/evaluators/`：把检测/时序预测事实变成统一结果；
- `benchmark/core/result.py`：结果 schema 和三态指标；
- `benchmark/core/artifacts.py`：逐图/逐视频预测 artifact；
- `benchmark/core/artifact_io.py`：artifact 确定性落盘、哈希与可复算标记；
- `framework/cleansight_eval/core/catalog.py`：固定数据集指针、fingerprint 和泄漏检查（数据契约归 framework）；
- `benchmark/core/report.py` / `matrix.py`：人读报告和跨模型结果矩阵；
- `benchmark/core/delivery.py`：稳定交付清单；
- `schemas/`：上述 JSON 的跨语言外部契约。

三类核心产物各司其职：

| 产物 | 默认位置 | 保存内容 | 主要用途 |
|---|---|---|---|
| `*.evaluation.json` | `<run>/evals/` | 模型/checkpoint 哈希、testset/fingerprint、feature schema、三态指标与 spec、推理语义、完整性及 artifact 引用 | 一次评测的结构化结论，也是矩阵和报告的输入 |
| `*.predictions.json` | `<run>/artifacts/` | 时序逐视频预测/真值与对齐信息；或检测逐图类别、confidence、归一化 bbox | 保存可审计预测证据；时序可直接复算指标，检测需结合固定 GT |
| `*.delivery.manifest.json` | `<run>/evals/` | checkpoint、metadata、evaluation、artifact、报告和配置等文件的 role/path/required/size/SHA-256 | 交付前核对文件集合；不复制、不上传、不决定发布 |

评测还可能生成 `*.eval.md`、`EVALUATION_REPORT.md` 和 timeline PNG；这些是人读呈现，
会被 EvaluationResult 引用或列入 delivery manifest，但不代替上面三种机器可读契约。

benchmark 不上传模型、不注册版本，也不自动决定发布或上线。

## 固定 Testset

测试集登记在 `framework/testsets.yaml`，数据本体仍由数据组维护。评估前执行：

```bash
python tools/validate_testsets.py --catalog framework/testsets.yaml --json
```

校验内容包括 manifest 可读性、预期样本、特征维度/类别信息以及 train/val/test 源视频泄漏。
`formal` 评估要求所选 testset 已登记且校验通过；`exploratory` 可以保留降级事实，但不能作为锁定
测试集结果发布。

每个 testset 可设置 `split_overlap_policy: error | frame | allow`，默认 `error`：`error` 要求源视频
跨 split 隔离；`frame` 允许同源视频分段，但阻断相同帧ID进入多个 split；`allow` 不做跨 split
重叠门禁。缺失文件、单个清单内重复样本、类别和特征维度等检查始终执行。策略会进入 testset
fingerprint 和评测元数据；时序 `frame` 模式还会把实际帧ID分配纳入 fingerprint。`frame` / `allow`
的结果只能标为开发期 benchmark。

`testsets.yaml` schema v2 将重复信息分成两层：`datasets` 保存数据集级公共契约（family、版本、
类别、特征映射、维度、数据根目录和重叠策略），`testsets` 通过 `dataset` 引用公共定义，只声明
`split`、样本 `manifest`、`purpose` 和必要的 `expected_items`。加载器仍兼容旧版 schema v1；v2
若在 split 中重复声明公共字段会直接报错，防止两份配置逐渐不一致。

## 单模型评估（推荐入口）

### YOLO

```bash
python -m benchmark.cli.eval \
  --config framework/experiments/yolo-clean-large.yaml \
  --ckpt runs/<run>/checkpoints/group1_large/weights/best.pt
```

指标包括 mAP@0.5、mAP@0.5:0.95、整体 Precision/Recall 和逐类 Precision/Recall。检测指标来自
Ultralytics `val()`，benchmark evaluator 负责统一三态、spec、有效推理参数和结果结构；逐图预测
另存 artifact，复算时仍需固定 testset 真值。

### 时序

```bash
python -m benchmark.cli.eval \
  --config framework/experiments/gru-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/best.pt
```

时序 evaluator 从逐视频预测与真值重算 Accuracy、Edit、F1@0.1/0.25/0.5、TP/FP/FN、P/R、
Temporal IoU 和帧级分类指标。不同视频保持独立边界：Accuracy 是帧 micro，Edit 是逐视频 macro
mean，分段计数在各视频独立匹配后做跨视频 micro 聚合。

时序片段与 3 分钟端到端动作时间线共用 `framework/cleansight_eval/core/metrics.py` 的区间比较算法：同类别候选
按 Temporal IoU 从高到低做全局贪心一对一匹配，默认阈值为 0.1/0.25/0.5，再统一由 TP/FP/FN
计算 Precision、Recall 和 F1。时序模型的区间单位是帧，端到端时间线的区间单位是秒；IoU、
P/R/F1 口径相同，边界 MAE 的单位随输入时间轴变化。

### 历史批量脚本

`benchmark/single_model/run_yolo_benchmark.py` 和 `run_temporal_benchmark.py` 是批量包装器：前者要求
显式传入分组 YOLO 权重，后者读取顶层历史时序 registry；两者都逐个调用统一 eval 并汇总
`EvaluationResult`，不再解析旧 acceptance report 或直接执行模型。新 framework checkpoint
优先走上面的统一 eval，再用 matrix 汇总：

```bash
python -m benchmark.cli.matrix --runs runs
```

不要把旧 summary JSON 与新的 `*.evaluation.json` 当成相同文件；历史结果由兼容读取/升级路径处理，
新工具应以 `EvaluationResult v2` 为准。

## 全序列与流式 Feed-mode

快速 smoke：

```bash
python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py \
  --device cpu --max-videos 1 --max-frames 256
```

正式运行时移除 `--max-videos` 和 `--max-frames`，并记录 checkpoint、输入形状、`input_dim`、
`window`、feature mapping、类别 mapping、device、推理模式和 latency scope。

固定窗口训练出的 checkpoint 不能默认视为支持任意长全序列；必须先验证结构语义和 feed-mode 一致性。

## 3 分钟端到端

先复制并填写 case：

```bash
cp benchmark/e2e_3min/cases/example.yaml benchmark/e2e_3min/cases/clean_001.yaml
```

已有后端 prediction 时：

```bash
python benchmark/e2e_3min/run_e2e_benchmark.py \
  --case benchmark/e2e_3min/cases/clean_001.yaml \
  --prediction benchmark/e2e_3min/outputs/clean_001.prediction.json
```

没有 prediction 时可以只传 `--case` 生成待接入报告。prediction 应由 `CleanSightBackend` 运行真实
视频后导出；模型仓库不代替在线推理。

端到端约定：

```text
视频 → YOLO → feature mapping → 时序模型/analyzer
     → 流程结论、动作时间线、阶段时间和告警
```

端到端报告保留两层结论：

- 通用时间线指标：与时序模型评估共用一对一匹配、TP/FP/FN、Precision、Recall、F1 和 Temporal IoU；
- 业务 PASS/FAIL：在共享匹配结果上继续检查流程结果、必需动作是否出现，以及阶段起止误差是否位于
  `allowed_time_error_sec` 内。

“关键动作召回”表只是必需动作存在性门禁，不等同于基于 TP/FP/FN 的统计 Recall。

## 人工发布审阅（遗留兼容）

`benchmark/release_gate.py` 文件名为历史兼容，目前只整理 benchmark、延迟、因果性和参数量等证据，
不做上线 PASS/FAIL。需要读取旧 summary 时可直接运行：

```bash
python benchmark/release_gate.py \
  --summary benchmark/single_model/latest/yolo_summary.json \
  --version yolo-large-v2 \
  --latency-ms 12.3 \
  --causality by-construction-causal \
  --num-params 256131
```

模型发布、ModelScope 上传和上线门槛属于外部模型管理流程或人工决策。

## 输出契约

| 文件 | Schema | 说明 |
|---|---|---|
| `*.evaluation.json` | `evaluation-result-v2` | 指标、推理语义、testset、完整性和 artifact 引用 |
| `*.predictions.json` | `prediction-artifact-v1` | 检测逐图或时序逐视频预测 |
| `*.delivery.manifest.json` | `delivery-manifest-v1` | 交付文件路径、required、大小和 SHA-256 |

JSON Schema 位于仓库根目录 `schemas/`。运行时结构校验仍由 `benchmark/core/*` 的 Python 校验函数
执行，Schema 主要服务 CI、外部仓库和跨语言读取。

## 测试

```bash
pytest tests/test_metrics.py \
  tests/test_result_schema.py \
  tests/test_temporal_eval_artifacts.py \
  tests/test_delivery_manifest.py \
  tests/test_evaluator_boundaries.py
```

真实数据全量 benchmark 与合成数据 smoke test 应在报告中明确区分。
