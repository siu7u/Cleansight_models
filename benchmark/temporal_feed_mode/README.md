# 时序喂法 Benchmark

本目录用于比较同一时序模型架构/checkpoint 在两种合法输入方式下的表现：

- 整段喂：一次输入完整特征序列 `[1, T, F]`。
- 流式喂：逐 tick 维护最近 `window` 帧，只输入 `[1, window, F]` 并取最后一帧预测。

当前默认数据源为旧版 `legacy-20d-v1`：

```text
../CleanSightBackend/MS-TCN2/data/Endo_Project
```

该脚本当前面向旧 `legacy-20d-v1` 资产，不会自动发现 framework 新 run。等新特征和 checkpoint
满足两种 feed mode 的结构契约后，可以继续复用，但必须同步调整模型配置、`input_dim`、类别映射和
数据目录。固定窗口训练出的 checkpoint 不能默认宣称支持任意长全序列。

## 运行

从模型集根目录执行：

```bash
python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py
```

只跑一个模型：

```bash
python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py --model tcn
```

快速验收脚本链路：

```bash
python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py \
  --device cpu \
  --max-videos 1 \
  --max-frames 256
```

`--max-videos` 和 `--max-frames` 只用于 smoke test。正式汇报或打榜时不要传这两个参数，或改用 GPU 后台全量运行。

## 输出

```text
benchmark/temporal_feed_mode/latest/feed_mode_summary.md
benchmark/temporal_feed_mode/latest/feed_mode_summary.json
benchmark/temporal_feed_mode/reports/feed_mode_summary_<version-or-timestamp>.md
benchmark/temporal_feed_mode/reports/feed_mode_summary_<version-or-timestamp>.json
```

可以用 `--version` 指定本次 summary 的版本名：

```bash
python benchmark/temporal_feed_mode/run_feed_mode_benchmark.py --version temporal-v2
```

## 指标说明

报告会输出：

- full_sequence 与 streaming 的 Acc / Edit / F1。
- full_sequence 与 streaming 的逐类召回。
- 两种模式的逐帧预测一致率。
- streaming 单窗模型 forward 的 mean/median/p95；不含数据、特征提取、后处理、I/O 和生产链路。

评估时会裁掉前 `window - 1` 帧，使整段喂和流式喂在相同帧范围上比较。

正式报告必须同时记录 checkpoint、feature mapping、输入形状、`input_dim`、`window`、类别映射、
device 和推理模式；带 `--max-videos`/`--max-frames` 的结果只能标为 smoke。
