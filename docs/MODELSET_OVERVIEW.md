# CleanSight 模型集总览（现状 + 用法 + 汇报要点）

> 本文合并原 `MODELSET_STATUS_SUMMARY.md` / `MODELSET_PRESENTATION_SUMMARY.md` /
> `MODELSET_USAGE_GUIDE.md` 三份文档，作为模型集的**唯一现状与使用入口**。
> 最新工作汇报见 [`YOLO_WORK_SUMMARY.md`](YOLO_WORK_SUMMARY.md)；架构原则见
> [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) 与 [`DESIGN.md`](DESIGN.md)。

## 1. 总体结论

`Cleansight_models` 已从零散模型整理成**可训练、可评估、可登记、可复现**的模型资产仓库：

- **三条半流水线**：单帧检测（`detection` / YOLO）、全序列时序（`full_sequence_temporal`）、
  历史滑窗时序（`sliding_window_temporal`）、ROI 图像分类（`roi_classification` / 特征融合）。
- **训练/推理在 `framework/`，评测在 `benchmark/`**，依赖方向单向 `benchmark → framework`。
- **数据契约**（`framework/testsets.yaml`）由 framework 的 catalog 层统一维护，训练与评测共用。
- 当前处于**研发验证阶段**：YOLO 与时序模型指标均未达到生产晋升要求。

一句话汇报版：仓库已完成基础架构、YOLO 分组训练、三类时序模型、模型卡、版本钉定、单模型
benchmark、流式一致性 benchmark 和端到端评分器框架；当前重点是 YOLO 优化（large 组 P/R≥0.7、
small 组淘汰 <0.3 的类走特征融合）并用新 YOLO 特征重训时序模型。

## 2. 选择正确入口

| 需求 | 推荐入口 |
|---|---|
| **组员快速上手（训练/数据/环境）** | **`docs/TEAM_GUIDE.md`**（一条命令一个模型） |
| 训练一个 YOLO/GRU/MS-TCN/Transformer/特征融合 | `python -m framework.cleansight_eval.cli.train --model <别名>`（`--list-models` 查看；进阶用 `--config <yaml>`） |
| 一键下载训练数据集 / 校验就绪 | `python -m framework.cleansight_eval.cli.dataset --preset all` / `--check` |
| 环境检查与安装 | `python tools/team_env.py` / `--setup` / `--setup-venv` |
| YOLO 多方法优化实验（预设/grid） | `python -m framework.cleansight_eval.cli.sweep` |
| 小目标逐类阈值分析与淘汰决策 | `python -m benchmark.cli.analyze` |
| 评估单个 checkpoint | `python -m benchmark.cli.eval` |
| 汇总多个模型 | `python -m benchmark.cli.matrix` |
| 校验固定测试集 | `python tools/validate_testsets.py` |
| 全序列与流式一致性 | `benchmark/temporal_feed_mode/` |
| 3 分钟业务场景 | `benchmark/e2e_3min/` |

迁移前的独立脚本已冻结到 `legacy/`，不再作为受支持入口。

## 3. 环境准备

```bash
source ../CleanSightBackend/.venv/bin/activate
pip install -r framework/requirements.txt
python -c "import torch, yaml; print(torch.__version__)"
```

运行 YOLO 前再确认：

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

服务器无显示环境时：

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib
```

## 4. 数据与配置

- 实验配置位于 `framework/experiments/*.yaml`，带行内注释，配新实验复制最接近的改。
- 已登记数据集通过 `data.dataset_ref` 稳定引用；根目录、版本、类别、feature mapping 和
  manifest 由 `framework/testsets.yaml`（catalog）解析。只有未登记的临时/合成数据才直接
  使用 `data.root`；相对路径以 YAML 所在目录为基准。
- YOLO 检测数据从 ModelScope `lhh010/cleansight-yolo` 下载到 `datasets/cleansight-yolo/`
  （`python download_modelscope_dataset.py --preset yolo`）。
- 评估前执行 `python tools/validate_testsets.py --catalog framework/testsets.yaml --json`；
  `ok: false` 表示数据完整性或 split 泄漏校验未通过，修复前只能 exploratory 调试。

### 配置选择

| 模型 | 配置 | 流水线 |
|---|---|---|
| YOLO large | `framework/experiments/yolo-clean-large.yaml` | `detection` |
| YOLO small | `framework/experiments/yolo-clean-small.yaml` | `detection` |
| ROI 分类（特征融合） | `framework/experiments/roi-fusion.yaml` | `roi_classification` |
| GRU | `framework/experiments/gru-actionmixed.yaml` | `sliding_window_temporal` |
| MS-TCN | `framework/experiments/mstcn-actionmixed.yaml` | `full_sequence_temporal` |
| MS-TCN++ | `framework/experiments/mstcn2-actionmixed.yaml` | `full_sequence_temporal` |
| Transformer | `framework/experiments/transformer-actionmixed.yaml` | `full_sequence_temporal` |
| 历史 GRU v1 | `framework/experiments/legacy-gru-v1.yaml` | `sliding_window_temporal` |
| 历史 Causal TCN v1 | `framework/experiments/legacy-causal-tcn-v1.yaml` | `sliding_window_temporal` |
| 历史 Causal Transformer v1 | `framework/experiments/legacy-causal-transformer-v1.yaml` | `sliding_window_temporal` |

## 5. 训练

```bash
# YOLO 检测
python -m framework.cleansight_eval.cli.train --config framework/experiments/yolo-clean-large.yaml
# ROI 特征融合
python -m framework.cleansight_eval.cli.train --config framework/experiments/roi-fusion.yaml
# 时序
python -m framework.cleansight_eval.cli.train --config framework/experiments/gru-actionmixed.yaml
```

临时调参用 `-S` 覆盖（不改文件）：`-S train.epochs=5`、`-S train.batch=8`、
`-S train.window=32`。训练输出 run 目录自动创建于 `runs/<type>-<时间戳>/`，含
`checkpoints/`、`evals/`、`config.resolved.json`、`env.json`；时序另存 `history.csv` 与
`training_curves.png`，检测复用 Ultralytics 的 `results.csv`/`results.png`。

## 6. YOLO 优化工作流

完整指南见 [`YOLO_OPTIMIZATION.md`](YOLO_OPTIMIZATION.md)。三步闭环：

```bash
# 1) 多方法实验（预设或 grid）
python -m framework.cleansight_eval.cli.sweep --group group1_large --preset large_baseline large_s
# 2) 小目标逐类阈值分析 → 淘汰 <0.3 的类
python -m benchmark.cli.analyze --config framework/experiments/yolo-clean-small.yaml --ckpt <best.pt>
# 3) 淘汰类走 ROI 特征融合
python -m framework.cleansight_eval.cli.train --config framework/experiments/roi-fusion.yaml
```

## 7. 评估

```bash
python -m benchmark.cli.eval --config <yaml> --ckpt <checkpoint路径>
python -m benchmark.cli.matrix --runs runs
```

- `formal`：testset 必须在 catalog 登记并通过校验；metadata 必须与 checkpoint 绑定。
- `exploratory`：允许外部裸 checkpoint（`allow_missing_meta: true`），结果标记
  `missing_meta_fallback`。
- 检测单帧无状态；滑窗时序记录窗口/推进/延迟；全序列延迟标 N/A。
- 评估产物：`*.evaluation.json`（三态指标 + fingerprint + artifact 引用）、
  `*.predictions.json`（逐图/逐视频事实）、`*.delivery.manifest.json`（交付文件清单）、
  人读报告与 timeline PNG。benchmark 只出评估事实，不自动判定晋升。

## 8. 当前模型结果（历史基线，2026-07）

### YOLO（旧数据集基线，均已 FAIL）

| 分组 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| `group1_large` | 0.522 | 0.181 | 0.594 | 0.501 |
| `group2_small` | 0.343 | 0.200 | 0.351 | 0.394 |

### 时序（旧 20 维特征基线，不晋升）

| 模型 | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU | 68.54 | 70.77 | 48.74 | 40.34 | 25.21 |
| Causal TCN | 69.23 | 44.62 | 46.81 | 40.43 | 27.66 |
| Transformer | 69.70 | 66.15 | 46.43 | 41.07 | 33.93 |

> 以上为新数据建设前的历史基线。新数据集（`datasets/cleansight-yolo`，5.6 万图 / 8 类）的
> 正式基线正在重建中，见 [`YOLO_WORK_SUMMARY.md`](YOLO_WORK_SUMMARY.md)。

## 9. 当前最大缺口

1. **YOLO 指标未达标**：large 组 P/R 需 ≥0.7；small 组 <0.3 的类转 ROI 特征融合。
2. **新 YOLO 特征尚未闭环到时序**：`feature_mapping.py` 新版为 64 维
   （8 类 × [present, cx, cy, w, h, conf, dcx, dcy]），现有时序 v1 checkpoint 仍是旧 20 维，
   需要用新 YOLO 输出重新生成 64 维特征并重训时序模型。
3. **端到端真实验收未完成**：`benchmark/e2e_3min` 评分器已存在，但真实
   `clean_001.prediction.json` 需要 `CleanSightBackend` 在线推理导出。
4. **ModelScope 与复刻链路未完全落地**：本地上传目录已整理，仍需上传、回填地址/revision、
   完善 `pin.yaml` schema 与一键复刻脚本。

## 10. 流式 Benchmark 结论（历史）

`temporal_feed_mode` 全量评测（旧特征基线）：

| 模型 | Stream Acc | 一致率 | Stream p95 延迟 |
| --- | ---: | ---: | ---: |
| GRU | 90.99% | 91.53% | 0.6350 ms |
| TCN | 85.18% | 99.98% | 2.6282 ms |
| Transformer | 90.59% | 70.35% | 2.4183 ms |

建议：GRU 作在线效果优先基线，TCN 作在线/离线一致性对照，Transformer 暂不作首选在线模型。

## 11. 质量规范与边界

- 结果三态（computed / not_applicable / missing）严格区分，禁止用 0 冒充 N/A。
- checkpoint 自带绑定元信息（sha256/size），配置错配拒绝加载。
- 依赖方向：`benchmark → framework`（单向），framework 不反向 import benchmark。
- 仓库 Git 只保存源码、配置、registry 元数据与报告；权重/数据/训练输出不入 Git
  （权重由 ModelScope / pin.yaml 引用）。
- benchmark 不上传模型、不注册版本、不自动决定发布。
