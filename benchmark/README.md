# CleanSight Benchmark

`benchmark/` 是评估定义层，不负责训练模型。它统一维护固定 testset、指标口径、PredictionOutput
评估器、`EvaluationResult v2`、prediction artifact 和 delivery manifest。

单模型的默认运行入口仍是 framework CLI；benchmark 负责“如何判分”，framework 负责“如何运行模型
并把结果落到 run”。

## Benchmark 层级

| 层级 | 目录/入口 | 回答的问题 |
|---|---|---|
| 单模型 | framework `cli.eval`、`benchmark/single_model/` | 单个 YOLO 或时序模型本身表现如何 |
| Feed mode | `benchmark/temporal_feed_mode/` | 全序列与流式/滑窗推理是否一致、损失多大 |
| 端到端场景 | `benchmark/e2e_3min/` | 完整业务流程的动作、阶段和告警是否正确 |

三类结论不能混用。单模型指标不能替代后端端到端验收，smoke test 也不能作为正式 benchmark。

## 职责边界

```text
framework pipeline.predict()
        ↓ PredictionOutput
benchmark/evaluators/{detection,temporal}.py
        ↓ EvaluationResult + pending prediction artifact
framework cli.eval
        ↓ evaluation/report/delivery 落盘
```

- `benchmark/core/metrics.py`：时序指标数值真源；
- `benchmark/evaluators/`：把检测/时序预测事实变成统一结果；
- `benchmark/core/result.py`：结果 schema 和三态指标；
- `benchmark/core/artifacts.py`：逐图/逐视频预测 artifact；
- `benchmark/core/testsets.py`：固定数据集指针、fingerprint 和泄漏检查；
- `benchmark/core/delivery.py`：稳定交付清单；
- `schemas/`：上述 JSON 的跨语言外部契约。

benchmark 不上传模型、不注册版本，也不自动决定发布或上线。

## 固定 Testset

测试集登记在 `benchmark/testsets.yaml`，数据本体仍由数据组维护。评估前执行：

```bash
python tools/validate_testsets.py --catalog benchmark/testsets.yaml --json
```

校验内容包括 manifest 可读性、预期样本、特征维度/类别信息以及 train/val/test 源视频泄漏。
`formal` 评估要求所选 testset 已登记且校验通过；`exploratory` 可以保留降级事实，但不能作为锁定
测试集结果发布。

## 单模型评估（推荐入口）

### YOLO

```bash
python -m framework.cleansight_eval.cli.eval \
  --config framework/experiments/yolo-clean-large.yaml \
  --ckpt runs/<run>/checkpoints/group1_large/weights/best.pt
```

指标包括 mAP@0.5、mAP@0.5:0.95、整体 Precision/Recall 和逐类 Precision/Recall。检测指标来自
Ultralytics `val()`，benchmark evaluator 负责统一三态、spec、有效推理参数和结果结构；逐图预测
另存 artifact，复算时仍需固定 testset 真值。

### 时序

```bash
python -m framework.cleansight_eval.cli.eval \
  --config framework/experiments/gru-actionmixed.yaml \
  --ckpt runs/<run>/checkpoints/best.pt
```

时序 evaluator 从逐视频预测与真值重算 Accuracy、Edit、F1@0.1/0.25/0.5、TP/FP/FN、P/R、
Temporal IoU 和帧级分类指标。不同视频保持独立边界：Accuracy 是帧 micro，Edit 是逐视频 macro
mean，分段计数在各视频独立匹配后做跨视频 micro 聚合。

### 历史批量脚本

`benchmark/single_model/run_yolo_benchmark.py` 和 `run_temporal_benchmark.py` 继续用于汇总旧 registry/
acceptance report。新 framework checkpoint 优先走上面的统一 eval，再用 matrix 汇总：

```bash
python -m framework.cleansight_eval.cli.matrix --runs runs
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
