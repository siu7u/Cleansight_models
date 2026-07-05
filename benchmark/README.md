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
cleansight-yolo-pipeline-main/04_validate.py
```

并把各组报告汇总到：

```text
benchmark/single_model/yolo_summary.md
benchmark/single_model/yolo_summary.json
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

## 单模型时序

从模型集根目录执行：

```bash
../CleanSightBackend/.venv/bin/python benchmark/single_model/run_temporal_benchmark.py
```

输出：

```text
benchmark/single_model/temporal_summary.md
benchmark/single_model/temporal_summary.json
```

该脚本会依次调用已有的：

```text
tools/eval_temporal_detailed.py
tools/measure_temporal_latency.py
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
