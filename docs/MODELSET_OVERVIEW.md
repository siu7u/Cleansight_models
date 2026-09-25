# CleanSight 模型集总览（现状 + 用法 + 汇报要点）

> 本文合并原 `MODELSET_STATUS_SUMMARY.md` / `MODELSET_PRESENTATION_SUMMARY.md` /
> `MODELSET_USAGE_GUIDE.md` 三份文档，作为模型集的**唯一现状与使用入口**。
> 最新工作汇报见 [`YOLO_WORK_SUMMARY.md`](YOLO_WORK_SUMMARY.md)；架构原则见
> [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) 与 [`DESIGN.md`](DESIGN.md)。

> **最新状态（2026-09-24 更新）**
>
> - **当前最佳时序配方**：`mstcn2` `s4l10 h128`（331 万参）+ `roi-grid-144`(z-score)，
>   lr 5e-4 / 60 轮 / `dropout 0.3` / T-MSE 0.15 / clip 0.1，
>   **`best_metric = val_edit`** → edit 中位 **51.08**、F1@0.1 40.01、acc 53.98（8 seed）；
>   换回默认选点 `val_f1_0.5` 时 3~8 seed 读数 51.47、逐 seed 摆幅 0.64。
>   依据：[`EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`](EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md) §2.1/§2.3/§2.4。
> - **两条轴已探尽**：容量轴（加参数默认配方无用，`mstcn` h128 到顶）与特征轴
>   （**无任何替代契约超过 `roi-grid-144`**，其唯一稳健优势是 insert 召回）。
> - **本周新增契约**：`actionmixed-roi-grid-presence-v1`（48 维，presence 平面）。
> - **选点口径是一等参数**：现行默认 `val_f1_0.5` 与 test 的 Spearman ρ 仅 **0.199**（326 run 体检）。
> - **已否掉的外部方案**：TimesFM 时序基础模型（预测式预警路线证伪，见
>   [`EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md`](EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md)）。
> - 本周汇总与风险见 [`WEEKLY_REPORT_20260924.md`](WEEKLY_REPORT_20260924.md)；
>   可复用图表见 [`figures/README.md`](figures/README.md)。

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
| GRU（ROI 空间特征） | `framework/experiments/gru-actionmixed-auto-roi.yaml` | `sliding_window_temporal` |
| GRU（ROI 可见性重排 96 维） | `framework/experiments/gru-actionmixed-auto-roi-v2.yaml` | `sliding_window_temporal` |
| GRU（手部区域特征） | `framework/experiments/gru-actionmixed-auto-hand.yaml` | `sliding_window_temporal` |
| GRU（全局+手部） | `framework/experiments/gru-actionmixed-auto-global-hand.yaml` | `sliding_window_temporal` |
| GRU（bbox + 图像 embedding，形态 B） | `framework/experiments/gru-actionmixed-embed.yaml` | `sliding_window_temporal` |
| MS-TCN | `framework/experiments/mstcn-actionmixed.yaml` | `full_sequence_temporal` |
| MS-TCN（ROI 空间特征） | `framework/experiments/mstcn-actionmixed-auto-roi.yaml` | `full_sequence_temporal` |
| MS-TCN++ | `framework/experiments/mstcn2-actionmixed.yaml` | `full_sequence_temporal` |
| Transformer | `framework/experiments/transformer-actionmixed.yaml` | `full_sequence_temporal` |
| Transformer（ROI 空间特征） | `framework/experiments/transformer-actionmixed-auto-roi.yaml` | `full_sequence_temporal` |
| 历史 GRU v1 | `framework/experiments/legacy-gru-v1.yaml` | `sliding_window_temporal` |
| 历史 Causal TCN v1 | `framework/experiments/legacy-causal-tcn-v1.yaml` | `sliding_window_temporal` |
| 历史 Causal Transformer v1 | `framework/experiments/legacy-causal-transformer-v1.yaml` | `sliding_window_temporal` |

ROI 空间特征变体（`-roi` 后缀）与对应 40 维 bbox 基线同模型同超参，仅特征契约不同：
`actionmixed-roi-grid-v1` 把画面按 2×3 网格分区，每 (检测类, 区域) 统计
[presence, count, max_area] 共 144 维（recipe 见 `framework/cleansight_eval/temporal/features/roi_bbox.py`），
用于对照"空间分区信息"对动作识别的影响；`actionmixed-roi-grid-v2` 按实测可见性重排维度预算
（高频 3 类 3×3 网格 27 维/类、低频 5 类全局 1 区域 3 维/类，共 96 维，
recipe 见 `features/roi_bbox_v2.py`，依据 `docs/features/INPUT_DESIGN_PROPOSAL.md` §2 P2b），
用于对照"按可见性分配 vs 按类别表整齐分配"。

形态 B（像素特征进时序，E 系列）把冻结 backbone 预计算的逐帧整图 embedding 拼进时序输入：
`actionmixed-bbox-embed-mbv3s-v1` = 40 维 bbox 块 + 576 维 mobilenet_v3_small embedding
（recipe 见 `framework/cleansight_eval/temporal/features/image_embed.py`），进 GRU 前经 64 维
线性投影头（`model.image_dim` / `model.image_proj_dim`）；embedding 产物由
`features/extract_embeddings.py` 离线预计算，训练侧不依赖图像与 GPU。

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
>
> **补注（2026-09-24）**：上表是**旧 20 维特征**口径，与当前主线（`roi-grid-144`）**不可直接比较**。
> 当前 mainline 数据为 `temporal.actionmixed-auto-roi-v1`（144 维，revision `6375eba9…`，
> test 8 视频 / 2,639 帧），最佳配方与指标见本文顶部「最新状态」及
> [`EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`](EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md)。

## 9. 当前最大缺口

1. **YOLO 指标未达标**：large 组 P/R 需 ≥0.7；small 组 <0.3 的类转 ROI 特征融合。
2. ~~**新 YOLO 特征尚未闭环到时序**（旧 20 维 / 64 维叙事）~~ —— **已过期（2026-09-24 更新）**：
   时序输入早已切到 `actionmixed-roi-grid-v1`（144 维），特征轴也已探尽。**当前真实缺口是精度天花板**：
   最强模型上 **51.6% 的真值段连标签都认错**（flush 20/21、withdraw 25/27、sbc 18/18），
   错误的大头**不在标签边界附近**（距最近切换 ≤3 帧的帧占 16.1%、只承载 21.9% 的错误）
   —— 即瓶颈不在模型/特征/损失，见 [`EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`](EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md) §4.1。
3. **端到端真实验收未完成**：`benchmark/e2e_3min` 评分器已存在，但真实
   `clean_001.prediction.json` 需要 `CleanSightBackend` 在线推理导出。
4. **架构门禁红灯（2026-09-24 新增识别）**：`tests/test_architecture_boundaries.py` 2 条失败，
   12 个违规文件中 **10 个来自已提交旧文件**（HEAD 上即红），需一次决策（放宽 `tools/` 运行规则
   vs 下沉执行模型的工具）。
5. **`--resume` 语义错位（已知未修）**：会导致"以为续训、实际未续训"，属会污染实验结论的隐患
   （记录见 [`mstcn-capacity/MSTCN_CAPACITY_STUDY.md`](mstcn-capacity/MSTCN_CAPACITY_STUDY.md) §0.1）。
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
