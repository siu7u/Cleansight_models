# CleanSight 模型集总览（现状 + 用法 + 汇报要点）

> 本文合并原 `MODELSET_STATUS_SUMMARY.md` / `MODELSET_PRESENTATION_SUMMARY.md` /
> `MODELSET_USAGE_GUIDE.md` 三份文档，作为模型集的**唯一现状与使用入口**。
> 最新工作汇报见 [`YOLO_WORK_SUMMARY.md`](YOLO_WORK_SUMMARY.md)；架构原则见
> [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) 与 [`DESIGN.md`](DESIGN.md)。

> **最新状态（2026-09-24 更新）**
>
> - **2026-10-10~11 v4.1 准确率攻关（4 轮，离线与在线两条轨都拿到验证过的增益）**：
>   离线单模型 **test acc 64.91 → 79.15**（`actionmixed-bbox-global-hand-8cls-v2`，80 维）、
>   **edit 62.64 → 70.41**（`actionmixed-combo-v3`，232 维 = `roi-grid-v5` ‖ `bbox-hand-8cls-v2`）；
>   在线（滑窗因果）**acc 58.82 → 64.06、edit 49.35 → 55.20**（换契约 + 固定预算取末轮）。
>   三条杠杆：① 手部区域锚点由"面积最大的 hand 框"改为"**全部 hand 框并集**"（单变量；
>   test 段级 f1@0.1/0.25/0.5 配对 p=0.017/0.011/0.0006）；② 两段拼接把 `sbc` 召回从 0 补到 50.4；
>   ③ 多成员投票（`combo-v3` 16 成员投票 test 80.10/73.40、val 79.93/78.78）。
>   逐项读数、边界与复现见 [`weeks/2026-10-10_2026-10-16/README.md`](weeks/2026-10-10_2026-10-16/README.md)。
>   **待定**：v4.1 旗舰契约尚未切换（只要 acc 选 80 维 `global-hand-v2`，要 `sbc`/edit 选 232 维
>   `combo-v3`），故 `mstcn2-auto-roi-v4-v41.yaml` 仍是仓库默认。
> - **2026-10-12 时序侧两条例杆（均在既有 v4.1 盘面上，不换数据、不换契约）**：
>   ① **离线软时长惩罚解码**（`β·max(0, D_c−d)` 的 Viterbi，**训练无关、零参数**）——
>   F1 三个 IoU 门槛 × 两个 split = **18/18 显著为正**（16 seed 配对）；逐契约采纳 β =
>   `combo-v3` **0.5** / `hand-v2` **0.1** / `global-hand-v2` **0.05**；已在**真实评测链路**
>   端到端复现（`combo-v3` β=0.5 ⇒ val edit **+0.73**；`hand-v2` β=0.1 ⇒ val edit +0.51、acc −0.01），
>   且 **β=0 时与已存盘评测逐位一致**（不写配置 ⇒ 行为完全不变）。用法：实验 YAML 的
>   `evaluation` 段写 `duration_soft_strength` 与 `duration_min_frames`（长度 = 类别数，
>   依据 train 真值 q0.10 估得）。**余**：`D_c` 需人工填入，下一步做「训练时估好、随 checkpoint
>   meta 落盘、推理只读」。实现与判别性用例见 `framework/cleansight_eval/temporal/util.py`
>   与 `framework/tests/test_soft_duration_decoding.py`。
>   ② **在线轨门槛 `smoothing_min_duration`**——该值是**硬召回上限**（输出里不可能出现短于它的段），
>   而 **6 个在线臂全部走默认 25、从未被扫过**。8 seed 配对（`online-gru-hand-v2`）：
>   门槛 25 → 10 使 **acc val +3.55 / test +3.59（两 split 一致，p=0.0078 / 0.0234）**；
>   `edit` 在该臂的 test 上无净收益（−0.22、p=1.0）⇒ **该臂不报 edit**，但第二个臂
>   `online-gru-global-hand-v2`（80 维、单 seed）acc 与 edit 皆为正 ⇒ 「edit 不迁移」是**臂特定**、
>   不是通则。**适合按 acc 目标调低该门槛；逐臂各自选值。**
>   两条的机制、边界与复现见
>   [`weeks/2026-10-10_2026-10-16/TEMPORAL_SOFTDUR_20261012.md`](weeks/2026-10-10_2026-10-16/TEMPORAL_SOFTDUR_20261012.md)
>   与 [`…/ONLINE_MIN_DURATION_20261012.md`](weeks/2026-10-10_2026-10-16/ONLINE_MIN_DURATION_20261012.md)。
> - **检测侧（2026-10-12）**：首次 `fraction=1.0` 的 group1 全量训练完成中（此前 **11 个历史 run
>   全是 `fraction=0.2` 探针**，故旧读数不得当作模型质量）；诊断为**目标「P/R ≥ 0.7」的联合门槛
>   在现有模型上不可达**（逐类与整体两种口径皆然）、**阈值无解**，硬件侧起点是最小的 `yolo11n`。
>   详见 [`weeks/2026-10-10_2026-10-16/YOLO_GROUP1_FULL_20261012.md`](weeks/2026-10-10_2026-10-16/YOLO_GROUP1_FULL_20261012.md)。
> - **2026-10-08 数据换代（v4.1）**：auto 数据推进到 v4.1（15fps 重采样 + **标签时间轴 ×1.25
>   修正** + train 28 / val 7 / test 8），本地把新数据目录改名为
>   `datasets/cleansight-ActionMixed-auto-lhh`（路径同 v3、内容已换代）。因此**下文与各处引用的
>   时序读数全部是旧 v3 口径，与 v4.1 不可比**，auto 链 14 个配置均需重训；检测侧逐坐标未变，
>   YOLO 不受影响。新契约 `temporal.actionmixed-auto-roi-v4-v41` 与旗舰基线配置
>   `framework/experiments/mstcn2-auto-roi-v4-v41.yaml` 已就绪，详见
>   [`weeks/2026-10-03_2026-10-09/V41_DATA_UPDATE_20261008.md`](weeks/2026-10-03_2026-10-09/V41_DATA_UPDATE_20261008.md)。
> - **2026-10-09 v4.1 全量重训完成（20 臂 × 8 seed，正式评测）**：特征轴 acc 64.91~**76.27**、
>   架构轴 33.75~66.37、在线轨 50.45~55.45。两个必须知道的结论：① **特征契约排名不随数据口径
>   迁移**（新旧尺子 Spearman ρ=−0.149，v3 冠军 roi-v4 在 v4.1 上垫底）；② **val_edit 选点在部分
>   臂上反噬**（transformer 的 best 比 last 低 17.12pp）。逐臂表/逐类召回/YOLO 复评见
>   [`weeks/2026-10-03_2026-10-09/TOPIC §7`](weeks/2026-10-03_2026-10-09/V41_DATA_UPDATE_20261008.md)
>   与 `runs/v41/SUMMARY_ALL_ARMS.md`。**下文 §4/§8 的配置与指标均为 v3 口径历史记录。**
> - **当前最佳时序配方**：`mstcn2` `s4l10 h128`（331 万参）+ `roi-grid-144`(z-score)，
>   lr 5e-4 / 60 轮 / `dropout 0.3` / T-MSE 0.15 / clip 0.1，
>   **`best_metric = val_edit`** → edit 中位 **51.08**、F1@0.1 40.01、acc 53.98（8 seed）；
>   换回默认选点 `val_f1_0.5` 时 3~8 seed 读数 51.47、逐 seed 摆幅 0.64。
>   依据：[`EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`](experiments/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md) §2.1/§2.3/§2.4。
> - **两条轴已探尽**：容量轴（加参数默认配方无用，`mstcn` h128 到顶）与特征轴
>   （**无任何替代契约超过 `roi-grid-144`**，其唯一稳健优势是 insert 召回）。
> - **本周新增契约**：`actionmixed-roi-grid-presence-v1`（48 维，presence 平面）。
> - **选点口径是一等参数**：现行默认 `val_f1_0.5` 与 test 的 Spearman ρ 仅 **0.199**（326 run 体检）。
> - **已否掉的外部方案**：TimesFM 时序基础模型（预测式预警路线证伪，见
>   [`EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md`](experiments/EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md)）。
> - 本周汇总与风险见 [`WEEKLY_REPORT_20260924.md`](WEEKLY_REPORT_20260924.md)；
>   可复用图表见 [`figures/README.md`](figures/README.md)。

## 1. 总体结论

`Cleansight_models` 已从零散模型整理成**可训练、可评估、可登记、可复现**的模型资产仓库：

- **三条半流水线**：单帧检测（`detection` / YOLO）、全序列时序（`full_sequence_temporal`）、
  历史滑窗时序（`sliding_window_temporal`）、ROI 图像分类（`roi_classification` / 特征融合，**已评估不采用**）。
- **训练/推理在 `framework/`，评测在 `benchmark/`**，依赖方向单向 `benchmark → framework`。
- **数据契约**（`framework/testsets.yaml`）由 framework 的 catalog 层统一维护，训练与评测共用。
- 当前处于**研发验证阶段**：YOLO 与时序模型指标均未达到生产晋升要求。

一句话汇报版：仓库已完成基础架构、YOLO 分组训练、三类时序模型、模型卡、版本钉定、单模型
benchmark、流式一致性 benchmark 和端到端评分器框架；当前重点是 YOLO 优化（large 组 P/R≥0.7、
small 组淘汰 <0.3 的类**暂无替代方案**——ROI 特征融合方案已评估不采用，见 §9.1）并用新 YOLO 特征重训时序模型。

## 2. 选择正确入口

| 需求 | 推荐入口 |
|---|---|
| **组员快速上手（训练/数据/环境）** | **`docs/TEAM_GUIDE.md`**（一条命令一个模型） |
| 训练一个 YOLO/GRU/MS-TCN/Transformer | `python -m framework.cleansight_eval.cli.train --model <别名>`（`--list-models` 查看；进阶用 `--config <yaml>`） |
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
| GRU | `framework/experiments/gru-actionmixed.yaml` | `sliding_window_temporal` |
| GRU（ROI 空间特征） | `framework/experiments/gru-actionmixed-auto-roi.yaml` | `sliding_window_temporal` |
| GRU（ROI 可见性重排 96 维） | `framework/experiments/gru-actionmixed-auto-roi-v2.yaml` | `sliding_window_temporal` |
| GRU（手部区域特征） | `framework/experiments/gru-actionmixed-auto-hand.yaml` | `sliding_window_temporal` |
| GRU（全局+手部） | `framework/experiments/gru-actionmixed-auto-global-hand.yaml` | `sliding_window_temporal` |
| GRU（bbox + 图像 embedding，形态 B） | `framework/experiments/gru-actionmixed-embed.yaml` | `sliding_window_temporal` |
| MS-TCN | `framework/experiments/mstcn-actionmixed.yaml` | `full_sequence_temporal` |
| MS-TCN（ROI 空间特征） | `framework/experiments/mstcn-actionmixed-auto-roi.yaml` | `full_sequence_temporal` |
| MS-TCN++ | `framework/experiments/mstcn2-actionmixed.yaml` | `full_sequence_temporal` |
| MS-TCN++（**v4.1 旗舰基线**，128 维 roi-grid-v4） | `framework/experiments/mstcn2-auto-roi-v4-v41.yaml` | `full_sequence_temporal` |
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
# 3) 淘汰类：暂无自动替代方案（ROI 特征融合方案已评估不采用，见 §9.1）
```

> 第 3 步原先指向 `roi-fusion.yaml`（ROI 特征融合）。该方案已于 2026-10-02 评估为**不采用**：
> 配置与模型别名已移除、实验产物已回收，理由见 §9.1。

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
> [`EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`](experiments/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md)。

## 9. 当前最大缺口

1. **YOLO 指标未达标**：large 组 P/R 需 ≥0.7；small 组 <0.3 的类原计划转 ROI 特征融合，该方案**已评估不采用**（§9.1）。
2. ~~**新 YOLO 特征尚未闭环到时序**（旧 20 维 / 64 维叙事）~~ —— **已过期（2026-09-24 更新）**：
   时序输入早已切到 `actionmixed-roi-grid-v1`（144 维），特征轴也已探尽。**当前真实缺口是精度天花板**：
   最强模型上 **51.6% 的真值段连标签都认错**（flush 20/21、withdraw 25/27、sbc 18/18），
   错误的大头**不在标签边界附近**（距最近切换 ≤3 帧的帧占 16.1%、只承载 21.9% 的错误）
   —— 即瓶颈不在模型/特征/损失，见 [`EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`](experiments/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md) §4.1。
3. **端到端真实验收未完成**：`benchmark/e2e_3min` 评分器已存在，但真实
   `clean_001.prediction.json` 需要 `CleanSightBackend` 在线推理导出。
4. **架构门禁红灯（2026-09-24 新增识别）**：`tests/test_architecture_boundaries.py` 2 条失败，
   12 个违规文件中 **10 个来自已提交旧文件**（HEAD 上即红），需一次决策（放宽 `tools/` 运行规则
   vs 下沉执行模型的工具）。
5. **`--resume` 语义错位（已知未修）**：会导致"以为续训、实际未续训"，属会污染实验结论的隐患
   （记录见 [`mstcn-capacity/MSTCN_CAPACITY_STUDY.md`](mstcn-capacity/MSTCN_CAPACITY_STUDY.md) §0.1）。
4. **ModelScope 与复刻链路未完全落地**：本地上传目录已整理，仍需上传、回填地址/revision、
   完善 `pin.yaml` schema 与一键复刻脚本。

### 9.1 已停用：ROI 特征融合（`roi_classification`）——2026-10-02 评估为**不采用**

**背景**：`roi_classification`（模型别名 `feature_fusion`，配置文件原为 `roi-fusion.yaml`）是"**检测不达标的
稀有类替补方案**"：`benchmark.cli.analyze` 逐类扫描后把 P/R < 0.3 的类从 YOLO 淘汰，改为在 GT 框上裁
224×224 ROI 图、用预训练 CNN（torchvision resnet/efficientnet/mobilenet）+ MLP 多标签头做二次分类。
**它从来不是时序模型的输入**——时序链消费的是 YOLO 检测框算出的统计特征，两条链互不相干。

**不采用的三条理由**（均为实测）：

1. **无法验收**：它要救的类恰恰没有足够样本验收。`brush_tip_out` 在留出集只有 **21 个正样本**，
   precision 的 95% CI 宽达 **[0.38, 0.70]**——撑不起任何性能声明；`air_gun`（293 正样本，F1 0.975）站得住，
   但一个类撑不起一条链。
2. **接不上真实推理（循环依赖）**：训练与评估都用 **GT 框**裁剪，而它要救的类正是**检不出框**的类。
   要落地必须先"去框化"（改成密集/滑窗分类器），那是**另一个形态**，本轮未试、也无证据。
3. **它服务的决策从未执行**：`datasets/cleansight-yolo/` 下没有 `group2_small_kept`，当前时序特征仍使用
   全部 8 个检测类（含 `air_gun` / `brush_tip_out`）——即这条链是给一个**没发生过的淘汰决策**准备的备胎。

**本次处置（选项 B：停用保留）**：

| 项 | 处置 |
|---|---|
| 实验配置 | **已删除** `framework/experiments/roi-fusion{,‑heldout}.yaml`（`usage/YAML_CONFIG.md` 同步移除） |
| 模型别名 | **已移除** `feature_fusion`（`core/model_aliases.py`）→ `cli.train --model` / `cli.sweep` 不再能选到 |
| 实现与测试 | **保留**（`framework/cleansight_eval/classification/`、`benchmark/evaluators/classification.py`、`framework/tests/test_classification_pipeline.py`），可回滚、可供"去框化"改造复用 |
| 实验产物 | **已回收** `runs/feature_fusion/`（2.9 GB 裁剪缓存）与 `runs/roi-real-airgun-brushtip/`（44 MB） |
| 决策依据 | 保留 [`experiments/EXPERIMENT_REPORT_ROI_EVAL_PROTOCOL_20260927.md`](experiments/EXPERIMENT_REPORT_ROI_EVAL_PROTOCOL_20260927.md)（in-sample vs 留出集的量化对照） |

**若日后重启**：先做"去框化"（每帧对候选区域出概率）＋ **登记正式 testset** ＋ 把目标类的正样本补到
几百量级，再谈验收；沿用现有的 `feature_fusion` 实现即可。

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
