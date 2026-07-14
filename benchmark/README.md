# CleanSight Benchmark

本目录把 benchmark 分成两类：

- `single_model/`：验证单个模型效果。用于 YOLO 检测模型、GRU/TCN/Transformer 等时序模型的离线指标与延迟。
- `e2e_3min/`：验证 3 分钟洗消流程端到端效果。用于检查完整链路是否能给出正确流程结论、关键动作召回和告警。

两类 benchmark 的结论不能混用：

- 单模型 benchmark 回答“模型本身准不准”。
- 端到端 benchmark 回答“完整业务流程能不能判断对”。

## 当前状态

- YOLO 单模型 benchmark 已能汇总训练验证结果，当前两个分组均为 FAIL。
- 3 分钟端到端 benchmark 评分器已可运行，能根据 prediction JSON 输出 PASS / FAIL、动作召回和阶段时间误差。
- `benchmark/e2e_3min/reports/clean_001.md` 当前只代表评分器在给定 prediction JSON 时跑通；真实后端在线推理导出的 `clean_001.prediction.json` 仍待接入。
- 时序单模型 benchmark 需要继续补齐延迟结果，并将结果写回各模型 `CARD.md`。

## 单模型 YOLO

从模型集根目录执行：

```bash
../CleanSightBackend/.venv/bin/python benchmark/single_model/run_yolo_benchmark.py
```

默认会调用：

```text
yolo-detection/pipeline/04_validate.py
```

并把各组报告汇总到：

```text
benchmark/single_model/latest/yolo_summary.md
benchmark/single_model/latest/yolo_summary.json
benchmark/single_model/reports/yolo_summary_<version-or-timestamp>.md
benchmark/single_model/reports/yolo_summary_<version-or-timestamp>.json
```

只验证指定组：

```bash
../CleanSightBackend/.venv/bin/python benchmark/single_model/run_yolo_benchmark.py group1_large
../CleanSightBackend/.venv/bin/python benchmark/single_model/run_yolo_benchmark.py group2_small
```

如果只想汇总已有 `acceptance_report.md`，不重新跑验证：

```bash
../CleanSightBackend/.venv/bin/python benchmark/single_model/run_yolo_benchmark.py --skip-run
```

为本次汇总指定版本名：

```bash
../CleanSightBackend/.venv/bin/python benchmark/single_model/run_yolo_benchmark.py --skip-run --version yolo-large-v2
```

推荐在评估版本化权重时显式绑定模型 id、权重和 split：

```bash
../CleanSightBackend/.venv/bin/python model_manager/manager.py benchmark single_model_yolo \
  --model yolo.group1_large \
  --weights yolo-detection/pipeline/versioned_weights/yolo-large-v2/best.pt \
  --split val \
  --version yolo-large-v2 \
  --run
```

此时 summary JSON 会记录 `schema_version`、`model_id`、`checkpoint`、`dataset.split`、`metrics` 和 `gates`，后续 release gate 可以直接读取 JSON，不需要解析 Markdown。

## 单模型时序

从模型集根目录执行：

```bash
../CleanSightBackend/.venv/bin/python benchmark/single_model/run_temporal_benchmark.py
```

输出：

```text
benchmark/single_model/latest/temporal_summary.md
benchmark/single_model/latest/temporal_summary.json
benchmark/single_model/reports/temporal_summary_<version-or-timestamp>.md
benchmark/single_model/reports/temporal_summary_<version-or-timestamp>.json
```

为本次汇总指定版本名：

```bash
../CleanSightBackend/.venv/bin/python benchmark/single_model/run_temporal_benchmark.py --version temporal-v2
```

该脚本会依次调用已有的：

```text
tools/eval_temporal_detailed.py
tools/measure_temporal_latency.py
```

其中 `eval_temporal_detailed.py` 会按视频分别推理并保存逐视频预测 artifact，
再基于该 artifact 计算 frame accuracy、edit 和 segmental F1，避免把多个
独立视频拼成一条序列。`measure_temporal_latency.py` 只测单个随机
`[1, window, input_dim]` 张量的模型 forward microbenchmark，输出字段为
`model_forward_mean_ms`、`model_forward_median_ms`、`model_forward_p95_ms`。
该结果不包含特征读取、窗口维护、后处理、YOLO 特征提取或端到端 IO，不能
直接当作部署端到端延迟。

## Release Gate

上线门禁会读取一个或多个 benchmark summary JSON,并检查上线前三项必填:

- 运行延迟:部署机实测,用 `--latency-ms` 或 CARD 里的延迟记录提供。
- 感受域/因果性:在线模型必须是因果或 by-construction causal,用 `--causality` 或 CARD 记录提供。
- 模型参数量:用 `--num-params` 或 CARD 记录提供。

输出:

```text
benchmark/release_gate/latest/release_gate.md
benchmark/release_gate/latest/release_gate.json
benchmark/release_gate/reports/release_gate_<version-or-timestamp>.md
benchmark/release_gate/reports/release_gate_<version-or-timestamp>.json
```

示例:

```bash
../CleanSightBackend/.venv/bin/python model_manager/manager.py benchmark release_gate \
  --summary benchmark/single_model/latest/yolo_summary.json \
  --version yolo-large-v2 \
  --latency-ms 12.3 \
  --causality by-construction-causal \
  --num-params 256131 \
  --run
```

## 3 分钟端到端

先复制并填写 case：

```bash
cp benchmark/e2e_3min/cases/example.yaml benchmark/e2e_3min/cases/clean_001.yaml
```

如果已经有端到端预测时间线 JSON：

```bash
../CleanSightBackend/.venv/bin/python benchmark/e2e_3min/run_e2e_benchmark.py \
  --case benchmark/e2e_3min/cases/clean_001.yaml \
  --prediction benchmark/e2e_3min/outputs/clean_001.prediction.json
```

如果还没有预测文件，也可以先生成待接入报告：

```bash
../CleanSightBackend/.venv/bin/python benchmark/e2e_3min/run_e2e_benchmark.py \
  --case benchmark/e2e_3min/cases/clean_001.yaml
```

端到端输入输出约定：

```text
3 分钟视频
  -> YOLO 检测
  -> feature_mapping
  -> 时序模型 / analyzer
  -> 流程结论、关键动作、阶段时间、告警
```
