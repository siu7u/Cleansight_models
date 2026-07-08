# 时序喂法 Benchmark

本目录用于比较同一时序模型在两种输入方式下的表现：

- 整段喂：一次输入完整特征序列 `[1, T, F]`。
- 流式喂：逐 tick 维护最近 `window` 帧，只输入 `[1, window, F]` 并取最后一帧预测。

当前默认数据源为旧版 `legacy-20d-v1`：

```text
../CleanSightBackend/MS-TCN2/data/Endo_Project
```

该 benchmark 用于先验证评测框架。等 `clean-v1` 的新 YOLO 特征生成并重训时序模型后，可以继续复用同一脚本，只需要调整模型配置、`input_dim` 和数据目录。

## 运行

从模型集根目录执行：

```bash
../CleanSightBackend/.venv/bin/python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py
```

只跑一个模型：

```bash
../CleanSightBackend/.venv/bin/python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py --model tcn
```

快速验收脚本链路：

```bash
../CleanSightBackend/.venv/bin/python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py \
  --device cpu \
  --max-videos 1 \
  --max-frames 256
```

`--max-videos` 和 `--max-frames` 只用于 smoke test。正式汇报或打榜时不要传这两个参数，或改用 GPU 后台全量运行。

## 输出

```text
benchmark/temporal_feed_mode/feed_mode_summary.md
benchmark/temporal_feed_mode/feed_mode_summary.json
```

## 指标说明

报告会输出：

- full_sequence 与 streaming 的 Acc / Edit / F1。
- full_sequence 与 streaming 的逐类召回。
- 两种模式的逐帧预测一致率。
- streaming 单 tick 前向延迟。

评估时会裁掉前 `window - 1` 帧，使整段喂和流式喂在相同帧范围上比较。
