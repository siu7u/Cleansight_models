# 特征提取范围三策略横向对比（bbox 编码固定）

> 实验目的：bbox 特征编码固定为每类最大框 `[presence, cx, cy, w, h]`，只改变**提取范围**，
> 横向对比三种策略对动作识别的表现。模型为 GRU（hidden=128, 3 层）、seed 42、
> 同一 v3 数据三 split（train 13 / val 3 / test 2），仅特征契约不同。
>
> **两轮结果**：第一轮 20 epoch 无正则（被过拟合坍缩污染，仅作诊断证据）；第二轮
> 5 epoch + weight_decay=1e-4（配方健康，作为策略对比的有效基线，见文末坍缩分析）。
>
> **数据版本说明（2026-09-03）**：本报告所有 run 训练于 task#204 sbc 修正**之前**的 v3
> 标签；该修正只改 train 一个视频（sbc 39→48 帧），**val/test 标签未变**——test 指标结论
> 不受影响，train 侧学习的 sbc 差异在最终结论前由"多 seed 重跑（修正后数据）"覆盖。

## 策略与特征契约

| 策略 | feature_mapping | 维度 | 说明 |
|---|---|---|---|
| A 整个画面 | `actionmixed-bbox-8cls-v1` | 40 | 基线，全局坐标 |
| B 仅手部周围 | `actionmixed-bbox-hand-8cls-v1` | 40 | 只编码面积最大 hand 框扩张 1.5 倍区域内的框，坐标相对区域归一化；无 hand 全零 |
| C 全局+手部 | `actionmixed-bbox-global-hand-8cls-v1` | 80 | A 与 B 拼接 |

实现：`framework/cleansight_eval/temporal/features/hand_bbox.py`；登记：
`temporal.actionmixed-auto-hand-v1` / `temporal.actionmixed-auto-global-hand-v1`
（revision 与 v3 相同）；配置：`framework/experiments/gru-actionmixed-auto{,-hand,-global-hand}.yaml`。

## 数据侧事实（v3 val，1926 帧）

- 无 hand 帧占比 4.9%（手部特征全零）；hand 类 presence 95.1%。
- 关键差异：`scope_control_body`/`scope_mid_section` 全局 presence 84.7%/72.2%，
  但**手部区域内只有 14.2%/15.9%**——手部策略会丢掉大部分 scope 类信号。
- 稀有类（syringe/air_gun/brush_tip_out）手部区域内 presence ≤ 1%。

## 结果（正式 test，锚定 task#195/#199；test 仅含 idle/long_brush_insert/long_brush_withdraw）

### 第二轮：5 epoch + weight_decay=1e-4（配方健康，可信对比）

| 策略 | dim | acc | edit | F1@0.1 | F1@0.25 | F1@0.5 | 非 idle 预测帧 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ROI 网格** | 144 | 36.33 | **48.48** | **53.06** | **36.73** | — | 1109 |
| **A 整个画面** | 40 | 25.97 | 35.28 | 37.50 | 16.67 | 8.33 | 1158 |
| C 全局+手部 | 80 | **27.87** | 32.25 | 27.27 | 22.73 | 4.55 | 1356 |
| B 仅手部 | 40 | 26.07 | 30.09 | 29.79 | 17.02 | 12.77 | 1733 |

- **ROI 网格（actionmixed-roi-grid-v1, 144 维）在健康配方下段级指标全面领先**：
  edit 48.48 / F1@0.1 53.06，比 40 维基线高出 ~15 个点——空间分区信息确实带来增益
  （早期 20 epoch 轮因坍缩从未跑过 ROI 网格，矩阵补齐后才暴露）。
- 三策略均恢复非 idle 预测（第一轮全 idle 坍缩已解除）→ 配方修复验证通过。
- 帧级 acc 全面下降（25~36%）是**健康信号**：不再"永远猜 idle"（acc 在 65%+ idle 数据上
  是误导指标，段级 edit/F1 才是主线）。
- 单 seed（42）结论，需多 seed 复跑确认排序。

运行目录（未入库）：`tmp/try_fix/gru-20260903-{135238,135327,135413,161938}/`。

### 第一轮：20 epoch 无正则（坍缩诊断证据，不作策略结论）

| 策略 | dim | acc | edit | F1@0.1 | 状态 |
|---|---:|---:|---:|---:|---|
| C 全局+手部 | 80 | 59.76 | 16.02 | 11.11 | 全 idle 坍缩（GPU 版 edit 6.93） |
| A 整个画面 | 40 | 50.34 | 25.97 | 23.81 | idle 主导 |
| B 仅手部 | 40 | 48.02 | 11.47 | 11.11 | 全 idle 坍缩 |

运行目录：`tmp/compare_strategies/gru-20260831-{192341,192623,192907}/`（CPU）、
`runs/compare_strategies/gru-20260903-122653/`（GPU，C 策略）。

### 训练期最佳 val（第二轮 history.csv）

| 策略 | best val acc | best val edit | best val F1@0.5 |
|---|---:|---:|---:|
| A 整个画面 | 38.01 (ep5) | 26.49 (ep2) | 3.85 (ep4) |
| B 仅手部 | 30.89 (ep1) | 24.15 (ep2) | 6.25 (ep2) |
| C 全局+手部 | 42.69 (ep5) | 22.25 (ep1) | 4.59 (ep3) |

## 第三轮：四策略 × 3 seed 一键矩阵（2026-09-03，健康配方 + 配方代码修复）

工具：`python tools/run_strategy_matrix.py`（提交 `43f15ef`）；配方 weight_decay=1e-4 /
dropout=0.2 / patience=4 / best_metric=val_f1_0.5（选择指标），epochs 上限 20（实际 5~9
早停）；数据为含 task#204 修正的 v3。完整表见 `runs/strategy_compare/STRATEGY_SUMMARY.md`。

| 策略 | seed | edit | F1@0.1 | F1@0.25 | 非idle帧 | | 中位数 edit / F1@0.1 / F1@0.25 |
|---|---:|---:|---:|---:|---:|---|---:|
| ROI 网格 144 | 7/42/2026 | 23.6/53.3/28.4 | 28.6/33.3/31.8 | 23.8/20.8/22.7 | 813/369/267 | → **28.4 / 31.8 / 22.7** |
| 全局+手部 80 | 7/42/2026 | 18.4/18.4/6.9 | 15.8/20.0/11.8 | 10.5/15.0/5.9 | 120/288/0 | → 18.4 / 15.8 / 10.5 |
| 手部 40 | 7/42/2026 | 25.1/6.9/6.9 | 15.8/11.8/11.8 | 10.5/5.9/5.9 | 82/0/0 | → 6.9 / 11.8 / 5.9 |
| 全局 40 | 7/42/2026 | 36.6/9.3/6.9 | 28.6/11.4/11.8 | 14.3/5.7/5.9 | 179/8/0 | → 9.3 / 11.8 / 5.9 |

> **设备注**：第三轮全部 run 为 **CPU**（2026-09-03，自动化会话无 GPU），本表中位数仅在
> CPU 环境内可复现。GPU 全策略矩阵见下方第四轮（`runs/strategy_compare_gpu/`）。

## 第四轮：GPU 全策略矩阵（2026-09-04，RTX 4060 Laptop，数据根 `-lhh`）

工具与配方同第三轮（`python tools/run_strategy_matrix.py --runs-dir runs/strategy_compare_gpu`），
四策略 × seed 42/7/2026，test 仍锚定 task#195/#199；完整逐 seed 表见
`runs/strategy_compare_gpu/STRATEGY_SUMMARY.md`（坍缩 = 非 idle 预测帧为 0）。

| 策略 | seed(7/42/2026) edit | F1@0.1 | F1@0.25 | 非idle帧 | 中位 edit / F1@0.1 / F1@0.25 |
|---|---:|---:|---:|---:|---:|
| ROI 网格 144 | 21.2/55.6/23.6 | 19.1/44.9/19.5 | 14.3/36.7/14.6 | 769/477/152 | → **23.6 / 19.5 / 14.6** |
| 全局 40 | 9.3/18.4/41.3 | 16.7/20.0/27.3 | 11.1/15.0/18.2 | 63/421/236 | → 18.4 / 20.0 / 15.0 |
| 全局+手部 80 | 6.9/18.4/16.0 | 11.8/15.8/11.1 | 5.9/10.5/5.6 | 0/121/31 | → 16.0 / 11.8 / 5.9 |
| 手部 40 | 6.9/6.9/6.9 | 11.8/11.8/11.8 | 5.9/5.9/5.9 | 0/0/0 | → 6.9 / 11.8 / 5.9 |

- **跨设备稳健性**：ROI 网格 144 仍是唯一 **CPU/GPU 均三 seed 零坍缩** 的策略；roi-grid GPU
  中位（23.59/19.51/14.63）与 `runs/formal_roi_20260905/` 正式轮完全一致。
- **手部 ROI 通道无增益**：仅手部 40 在 GPU 三 seed 全坍缩（非 idle 预测 0 帧）；全局+手部
  80 仍有 1/3 坍缩 seed、中位 F1@0.1 11.8 与仅手部持平——与 CPU 轮一致（scope 类在手部
  区域内 presence 仅 ~15%，手部策略丢掉大部分判别上下文）。
- **GPU 口径排序与 CPU 不同**：全局 40 在 GPU 上零坍缩且中位 F1@0.1（20.0）≈ roi-grid
  （19.5），roi-grid 仅中位 edit 领先（23.6 vs 18.4）；CPU 轮 roi-grid 全面领先（31.8 vs
  ≤15.8）。**设备差异大于单轮噪声、逐 seed 漂移大于策略间距：跨设备只能定性比较**，
  正式数字必须锚定设备口径、多 seed 取中位数。

> **val 中间口径（2026-09-05 补充，仅作类别可见性参考，非独立测试）**：v3 test 不覆盖
> flush / short_brush_cleaning（帧级 support=0），无法给出这两类的正式 test 指标；作为
> 中间可见性证据，对 GPU 矩阵 12 个 best.pt 在 val（flush 147 帧 / sb_cleaning 110 帧 /
> lb_insert 180 帧 / lb_withdraw 90 帧）上补跑帧级逐类评测（运行目录未入库：
> `tmp/planb_val_evals/`，配置为同实验 YAML 的 `split_eval: val` 变体）。注意 best.pt 由
> 同一 val 按 val_f1_0.5 选出，存在选择偏差，且 seed 方差大，只读趋势不读绝对值：
> flush 帧级 recall 跨策略/seed 为 0~86%（global-40/global+hand-80 部分 seed 可达 70%+，
> 其余 seed 为 0）；sb_cleaning recall 0~52%；long_brush_insert recall 几乎全为 0、
> lb_withdraw 仅 global+hand-80/roi-grid 各一个 seed 达 53%——**长短毛刷与 flush 的可靠
> 逐类结论仍需 action-test 新数据补 test 覆盖后重测**（见 IMAGE_FEATURE_TRAINING.md §5.2）。

## 第五轮：**新 test（project-18）** 五方案矩阵（2026-09-18，CPU，数据 revision `6375eba9…`）

> **口径变了**：前四轮的 test 是 project-16 的 #195/#199 两个视频（只覆盖 idle/insert/withdraw）；
> 2026-09-17 起 ModelScope HEAD 把 **project-18（action-test）8 视频 / 2,639 帧**并入 test
> （train 14 / val 4 未动）。本轮是该 test 上的**首轮多方案多 seed 对照**，并顺带补齐了
> 逐类帧级 recall（此前文档明确缺 flush / sb_cleaning 的 test 覆盖）。
> 工具：`python tools/run_strategy_matrix.py --runs-dir runs/strategy_cmp_p18`
> （本轮起汇总表头自证 testset 口径、附逐类 recall、同策略同 seed 自动去重）；
> 完整逐 seed 表见 `runs/strategy_cmp_p18/STRATEGY_SUMMARY.md`。
> **设备口径**：全部 CPU（自动化会话无 GPU）；跨设备只能定性比较（第四轮已证）。

| 策略 | dim | acc | edit | F1@0.1 | F1@0.25 | 非idle预测帧 |
|---|---:|---:|---:|---:|---:|---:|
| roi-grid-96-v2（可见性重排，本轮新增） | 96 | **53.09** | 17.54 | 25.88 | 14.29 | 149~446 |
| roi-grid-144 | 144 | 52.97 | **19.89** | 25.58 | 13.64 | 172~338 |
| bbox-80-global-hand | 80 | 52.07 | 17.97 | 25.00 | 13.95 | 224~265 |
| bbox-40-hand | 40 | 51.08 | 15.62 | 23.53 | 11.90 | 43~347 |
| bbox-40-global | 40 | 48.09 | 23.10 | **30.43** | **17.39** | 241~319 |

逐类帧级 recall（各 seed 中位数，support = test 真值帧数）：

| 策略 | idle 1458 | water 0 | flush 164 | lb_insert 707 | lb_withdraw 259 | sbc 51 |
|---|---:|---:|---:|---:|---:|---:|
| bbox-40-global | 84.5 | n/a | 0.0 | 0.0 | 8.9 | 0.0 |
| bbox-40-hand | 88.3 | n/a | 0.0 | 0.0 | 23.2 | 0.0 |
| bbox-80-global-hand | 91.5 | n/a | 0.0 | 5.7 | 0.0 | 0.0 |
| roi-grid-144 | 91.3 | n/a | 0.0 | 5.7 | 0.0 | 0.0 |
| roi-grid-96-v2 | 92.5 | n/a | 0.0 | 0.0 | 14.3 | 0.0 |

**结论（本轮新增证据）**：

1. **换 test 口径后所有方案都退化成"近 idle 预测"**：flush（164 帧）、sb_cleaning（51 帧）
   五个方案三 seed **全为 0 recall**，lb_insert（707 帧）≤5.7%，只有 lb_withdraw 出现
   非零（hand 23.2 / v2 14.3 / global 8.9）。同时 idle recall 高达 84.5~92.5 —— 即
   **模型基本只说 idle**（acc 48~53% 正是 idle 占比附近的产物）。
2. **旧 test 上的排序不能外推**：第四轮（旧 test）roi-grid 中位 F1@0.1 19.5、手部策略坍缩；
   本轮 bbox-40-global 反而段级最好（edit 23.1 / F1@0.1 30.4），roi-grid 只在 acc 与
   稳定性上略优。**结论必须锚定 test 口径**，跨批次 test 的绝对数字不能与前四轮并列比较。
3. **可见性重排（v2，96 维）没有带来指标增益，但用 1/3 更少的维度持平 v1**：
   段级 edit 17.54 vs 19.89、F1@0.1 25.88 vs 25.58（中位数互有胜负），acc 略高（53.09 vs 52.97）。
   与之相对，v2 的特征密度显著改善：train 恒零通道 12/96（12%）vs v1 48/144（33%），
   低频 5 类从"散在 6 区几乎恒零"变成 3/3 全活跃（val/test 也一致改善）。**即：
   维度预算重排解决了表征冗余，但没有解决泛化——瓶颈仍在检测覆盖与批次漂移**，
   与 `INPUT_DESIGN_PROPOSAL.md` §0 判断 1 一致。
4. **跨批次 test 的证据价值**：project-18 是长短毛刷专项、与 train/val 不同录制批次，
   本轮数字应作为"跨批次泛化"的基线，而不是"同分布能力"的度量。后续方案（如归一化
   N0/N1/N2、ROI 图像通道）若要在新 test 上比，应**同时报旧口径（若保留）或明确标注
   跨批次**，否则会与前四轮混淆。
5. 段级 edit / F1@0.1 的"最优"在两个不同方案间分裂（global-40 段级最优、roi 系 acc 最优），
   说明当前 test 上**指标本身对方案不敏感**（各方案差异 < 2 个点，而 seed 间差异可达 8~10 点）。

## 第六轮：跨批次退化根因诊断（2026-09-18，新 test = project-18）

> 第五轮发现五个方案在新 test 上全部退化到近 idle。本轮用四组独立证据回答"为什么"，
> 并给出**首个真正跨批次**的评测口径。新增可复现工具
> [`tools/probe_split_shift.py`](../tools/probe_split_shift.py)（纯 CPU、只读数据、不训练）。

### 6.1 同一批 checkpoint 换 split：val（项目-16 同批次）vs test（项目-18 跨批次）

对第五轮的 15 个 run（5 策略 × 3 seed）的 best.pt 与 last.pt 各在 test 与 val 上评估一次
（共 60 次 `benchmark.cli.eval`：把配置的 `split_eval` 改成 `val` 得到 val 口径变体，
其余字段不动；原始表 `tmp/split_cmp/SPLIT_COMPARE.md`，逐 run JSON `tmp/split_cmp/split_compare.json`）：

| 策略 | ckpt | acc（test / val） | macro-F1（test / val） | flush（test / val） | sbc（test / val） | insert（test / val） |
|---|---|---:|---:|---:|---:|---:|
| bbox-40-global | best | 48.1 / 61.5 | 23.5 / 38.4 | 0.0 / 29.5 | 0.0 / 62.3 | 0.0 / 0.0 |
| bbox-40-global | last | 52.9 / 59.2 | 28.3 / 38.7 | 0.0 / 40.4 | 0.0 / 59.4 | 10.6 / 0.8 |
| bbox-40-hand | best | 51.1 / 69.0 | 45.2 / 67.1 | 0.0 / 0.0 | 0.0 / 44.6 | 0.0 / 0.0 |
| bbox-40-hand | last | 49.5 / 69.4 | 45.9 / 67.9 | 0.0 / 0.0 | 0.0 / 42.9 | 0.0 / 0.0 |
| bbox-80-global-hand | best | 52.1 / 64.1 | 19.7 / 54.9 | 0.0 / 41.0 | 0.0 / 65.7 | 5.7 / 0.0 |
| bbox-80-global-hand | last | 51.1 / 61.8 | 20.2 / 41.0 | 0.0 / 41.0 | 0.0 / 65.7 | 12.4 / 0.0 |
| roi-grid-144 | best | 53.0 / 59.6 | 26.1 / 32.6 | 0.0 / 0.0 | 0.0 / 12.0 | 5.7 / 0.0 |
| roi-grid-144 | last | 53.3 / 57.1 | 27.9 / 31.8 | 0.0 / 0.0 | 0.0 / 16.0 | 8.3 / 0.0 |
| roi-grid-96-v2 | best | 53.1 / 61.2 | 44.6 / 34.5 | 0.0 / 59.0 | 0.0 / 20.0 | 0.0 / 7.7 |
| roi-grid-96-v2 | last | 55.5 / 57.0 | 51.0 / 32.8 | 0.0 / 69.4 | 0.0 / 54.9 | 25.6 / 7.9 |

**要点**：五个特征契约、两种 checkpoint 选择**全部同向退化**——acc 差 8~18 个点、macro-F1 差
20~40 个点；`flush` 在 val 上可达 69.4%、`short_brush_cleaning` 可达 65.7%（视策略/checkpoint），
在 test 上**两者一律 0**。这排除了"某个特征方案选错了"这类解释。

### 6.2 逐视频拆解：退化集中在个别视频，不是"所有视频都差一点"

15 个 run 在 test 的逐帧预测（`tmp/split_cmp/PER_VIDEO.md`）：

| test 视频 | 帧数 | truth 非idle | 预测出非idle 的 run 数（/15） |
|---|---:|---:|---:|
| ac38ca6b | 535 | 266 | **14**（中位 208 帧） |
| 6d8c7af2 | 444 | 165 | 7 |
| 6d8b215c / 6f1a85e3 / 935f9e44 | 117 / 323 / 645 | 58 / 107 / 323 | 4 / 4 / 4 |
| 67edf4f9 / c7c853a4 | 230 / 64 | 106 / 18 | 1 / 1 |
| ca09e5b5 | 281 | 138 | **0** |

15 个 run 合计：预测 idle 35,683 帧（truth 21,870，1.63×）、flush **0**（truth 2,460）、
insert 1,348（0.13×）、withdraw 1,860（0.48×）、sbc 373（0.49×），另有 **321 帧被预测成
water_injection 而 test 真值为 0 帧**。即：不是"稍微差一点"，而是多数视频退化为全 idle。

### 6.3 特征层批次可分性：**没有**明显协变量漂移

域分类器（LDA，按视频分组 5 折 CV，`tools/probe_split_shift.py`）：

| 契约 | dim | train vs test AUC | train vs val AUC | \|SMD\|>1 通道占比 |
|---|---:|---:|---:|---:|
| bbox-40-global | 40 | 0.584 | 0.479 | 0.00% |
| bbox-40-hand | 40 | 0.563 | 0.545 | 0.00% |
| bbox-80-global-hand | 80 | 0.608 | 0.493 | 0.00% |
| roi-grid-144 | 144 | 0.526 | 0.549 | 0.69% |
| roi-grid-96-v2 | 96 | 0.474 | 0.541 | 0.00% |

AUC 全部落在 0.47~0.61 —— 特征的**边际分布**在两个批次间几乎不可分。
（bbox 系 AUC 0.56~0.61 略高于 ROI 系，说明 ROI 的量化统计更抗批次；但都不足以解释 6.1 的塌陷。）

### 6.4 结构与采样核查：排除时间错位 / 帧率不一致

| 核查项 | 结果 |
|---|---|
| 标签帧号步长 | train / val / test **全部为 4**（30fps → 7.5fps 均匀采样） |
| labels 行数 vs frames 检测文件数 | 逐视频完全一致（230/230、117/117、444/444、323/323、645/645、535/535、64/64、281/281） |
| 源视频帧率（由 clip 毫秒时间戳反推） | train 29.88~30.01、val 29.95~29.99、test **29.61~30.00** |

→ project-18 与 project-16 的采样口径、帧号语义、时长对位都一致，**不存在"标签错位 ×4"这类硬伤**。

### 6.5 逐帧线性探针：特征里有信号，但 flush / sbc **跨批次完全丢失**

用 train 拟合等先验多类 LDA（逐帧、无时序、无平滑），在 test 上：

| 契约 | flush | sbc | insert | withdraw | idle |
|---|---:|---:|---:|---:|---:|
| roi-grid-96-v2：val（同批次） | 62.3% | 53.1% | 17.6% | 3.4% | 52.4% |
| roi-grid-96-v2：test（跨批次） | **0.0%** | **0.0%** | 25.0% | 27.4% | 55.5% |
| roi-grid-144：test（ridge 0.1、train+val 拟合） | 6.1% | **58.8%** | **63.6%** | 24.3% | 42.2% |

test 上 roi-grid-96-v2 的 flush 帧去向：idle 88、withdraw 72、insert 4、**预测成 flush 0 帧** ——
即这批帧在 train 的特征空间里更接近 idle/withdraw。**结论：flush/sbc 的退化发生在"特征→类别"
关系上，不在时序模型能力上**（同一份特征在 val 上可得 62%/53%）。

另一个值得注意的反差：**逐帧线性探针在 test 上普遍强于训练好的 GRU**
（insert 25~63.6% vs 0~25.6%、withdraw 27~29% vs 0~23.2%、sbc 最高 58.8% vs 0%）。
即因果滑窗 + `min_duration=25` 平滑这套推理链在跨批次上是**负增益**；
建议把"逐帧线性探针"作为跨批次基线纳入评测（零训练成本），时序模型必须证明自己高于它。

### 6.6 结论与建议

1. **口径意义**：旧 test（project-16 的 #195/#199）同样来自训练所用批次——第五轮之前的所有
   "test 指标"实质都是同批次指标。project-18 是首个真跨批次 test，当前跨批次表现**不可用**
   （非 idle recall ≈ 0~27%）。
2. **退化不是输入分布漂移**（6.3）、不是错位（6.4）：更像"同一动作在两个批次里外观/标注语义
   不同"叠加"模型学到了批次内线索"。5 个特征契约同向退化、且线性探针能拿到的信号
   （insert/withdraw/sbc）远多于 GRU，说明**模型侧与推理侧还有可观改进空间**。
3. **flush 优先做标签审计**（10 分钟级、零成本）：val 29~69% vs test 0%，且 test 的 flush 帧
   在特征空间落在 idle/withdraw 附近。对照 `docs/features/INPUT_DESIGN_PROPOSAL.md` §1.4 的
   建议，先确认 project-18 的 flush 时间轴与可见动作一致，再决定是否重训/重标。
4. **基线要求**：后续任何新特征方案在 test 上比较时，必须同时报逐帧线性探针基线；否则无法
   区分"特征变好了"与"时序模型碰巧没塌"。
5. **复现命令**：

```bash
# 分布漂移 + 线性探针（纯 CPU，只读数据）
python tools/probe_split_shift.py --json tmp/probe_split_shift.json
# 五方案 × 3 seed 矩阵（新 test）
python tools/run_strategy_matrix.py --runs-dir runs/strategy_cmp_p18
# test/val 双口径对照（同一批 checkpoint）
sed 's/split_eval: test/split_eval: val/' framework/experiments/gru-actionmixed-auto-roi.yaml > /tmp/roi-val.yaml
python -m benchmark.cli.eval --config /tmp/roi-val.yaml \
    --ckpt runs/strategy_cmp_p18/gru-20260918-204416/checkpoints/best.pt --out-dir tmp/val_eval_demo
```

## 第七轮：推理侧平滑阈值消融——半数段级指标曾被协议吃掉（2026-09-18）

> 第六轮证明"特征没漂、模型没吃满信号"。本轮追到**评测协议本身**：滑窗推理的因果平滑
> `causal_decision(min_duration=25)` 要求候选类连续出现 25 帧（7.5fps ≈ 3.3 s）才切换输出，
> 而 project-18 的动作段普遍很短 —— 该阈值构成**召回上限**。本轮把它从硬编码改为可配置
> （`evaluation.smoothing_min_duration`，默认 25 保持历史行为），并对同一批 checkpoint
> 做零重训消融。

### 7.1 真实段长分布与阈值可达性

`python tools/probe_split_shift.py` 现会输出逐类段长与"≥阈值的帧占比"（召回上限）：

| split | flush（帧/段/中位） | ≥25 | insert | ≥25 | withdraw | ≥25 | sbc | ≥25 |
|---|---|---:|---|---:|---|---:|---|---:|
| train | 893 / 34 / 18 | 61% | 1289 / 22 / 64 | 98% | 437 / 19 / 24 | 63% | 488 / 26 / 10 | 59% |
| val | 183 / 12 / 8 | 56% | 495 / 8 / 53 | 100% | 206 / 8 / 22 | 38% | 175 / 10 / 6 | 63% |
| **test** | 164 / 7 / 22 | **52%** | 707 / 11 / 61 | **97%** | 259 / 9 / 27 | **91%** | 51 / 6 / 6 | **0%** |

→ 在 `min_duration=25` 下，**test 的 short_brush_cleaning 六个真实段全部 ≤19 帧**，该类召回在
"预测段与真值段对齐"的前提下不可能出现（实测五个方案确实全 0；严格说平滑态可跨越 id 边界
凑满 25 帧而生出预测段，但那样只会更低精度）；flush 有 48% 的帧落在 <25 帧的段里。这不是模型问题，是协议问题。

### 7.2 消融结果（同一批 best.pt，5 策略 × 3 seed，只改阈值）

| 策略 | edit（md=25 / 5 / 1） | F1@0.1（25 / 5 / 1） | flush（25 / 5 / 1） | insert | withdraw | sbc |
|---|---|---|---|---|---|---|
| bbox-40-global | 23.10 / 47.18 / **52.26** | 30.43 / 34.92 / 38.99 | 0.0 / 4.9 / 11.0 | 0.0 / 0.0 / 0.1 | 8.9 / 13.5 / 17.0 | 0.0 / 11.8 / 11.8 |
| bbox-40-hand | 15.62 / 37.84 / **46.37** | 23.53 / 37.38 / 36.36 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 23.2 / 21.2 / 22.8 | 0.0 / 23.5 / 9.8 |
| bbox-80-global-hand | 17.97 / 29.38 / **46.67** | 25.00 / 28.85 / 33.33 | 0.0 / 0.0 / 2.4 | 5.7 / 5.2 / 5.7 | 0.0 / 1.5 / 3.1 | 0.0 / 0.0 / 0.0 |
| roi-grid-144 | 19.89 / 41.13 / **40.99** | 25.58 / 41.54 / 39.10 | 0.0 / 3.7 / 3.0 | 5.7 / 9.6 / 11.0 | 0.0 / 3.1 / 4.6 | 0.0 / 0.0 / 0.0 |
| roi-grid-96-v2 | 17.54 / 37.74 / **39.55** | 25.88 / 33.96 / 30.26 | 0.0 / 0.0 / 0.0 | 0.0 / 6.2 / 5.2 | 14.3 / 6.9 / 10.0 | 0.0 / 0.0 / 0.0 |

（逐类列为 3 seed 中位 recall %；完整三张表见 `tmp/smooth_ablation/SMOOTHING_ABLATION.md`）

**要点**：

1. **段级 edit 全面翻倍**：17.5→39.6、19.9→41.0、23.1→**52.3**；F1@0.1 普遍 +5~14 个点。
   第五、六轮所有"新 test 上段级指标腰斩"的结论，有一半是**阈值 artifact**。
2. **sbc 从 0 变非零**（bbox-40-hand md=5 达 23.5%、global-40 md=1 达 11.8%）——与 7.1 的
   "0% 可达"完全对应：阈值一放松，这个类就回来了。
3. **flush 只回来一半**（0 → 0~11%）：与其 52% 的可达上限相称，剩下的缺口是真实的跨批次
   困难（第六轮：同一份特征在 val 上 62%）。
4. **insert 不是平滑的锅**：md=1 也只有 0.1~11%，而逐帧线性探针能到 25~64%（第六轮 6.5）
   → 时序模型在这类上的问题在训练/收敛，不在推理后处理。
5. idle recall 随阈值放松略降（84.5→80.7 等），acc 基本不变——即放松阈值是**用少量 idle
   精度换回大量段级正确性**，方向明确。

### 7.3 结论与协议建议

1. **任何跨方案比较都必须标注所用 `smoothing_min_duration`，并同时给出该口径下的召回上限**；
   矩阵汇总已自动附上"召回上限（非 idle 类，段长≥N）"一行（`tools/run_strategy_matrix.py`）。
2. **新 test 上建议以 md≤5 作为主口径、md=25 作为历史对照**：project-18 的段长分布与
   project-16 不同，沿用 25 会把协议差异误读成模型退化。
3. 后续新增数据/类别时，先跑 `tools/probe_split_shift.py` 看该类"阈值可达性"，
   再决定阈值——段长中位数可作默认值参考（test 全类中位 6~61 帧）。
4. 本轮把阈值做成配置项（默认值不变），旧结果可原样复现；测试见
   `framework/tests/test_causal_smoothing.py`。

## 第八轮：md=5 口径定版对照 + 逐帧线性探针升为一等参照（2026-09-18）

> 第七轮证明 md=25 会压指标，本轮把**评估口径做成矩阵工具的一等参数**，并在推荐口径
> （md=5）上给出五方案的定版排序；同时把"逐帧线性探针"接成矩阵里的**参照行**——它与模型
> 用同一个 min_duration、同一个冷启动填充、同一套 `temporal_metrics`，即"特征 + 相同后处理"
> 的端到端下界。工具：`python tools/run_strategy_matrix.py --runs-dir runs/strategy_cmp_p18
> --skip-train --smoothing-min-duration 5`（零重训，复用已有 checkpoint）。

### 8.1 五方案 × 3 seed（md=5，test = project-18）

| 策略 | acc | edit | F1@0.1 | F1@0.25 | idle | flush | insert | withdraw | sbc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bbox-40-global | 45.36 | **47.18** | 34.92 | 26.89 | 78.5 | 4.9 | 0.0 | 13.5 | 11.8 |
| bbox-40-hand | 51.12 | 37.84 | 37.38 | **29.91** | 86.1 | 0.0 | 0.0 | 21.2 | **23.5** |
| bbox-80-global-hand | 51.80 | 29.38 | 28.85 | 18.00 | 89.2 | 0.0 | 5.2 | 1.5 | 0.0 |
| roi-grid-144 | **51.99** | 41.13 | **41.54** | 26.89 | 86.0 | 3.7 | 9.6 | 3.1 | 0.0 |
| roi-grid-96-v2 | 51.53 | 37.74 | 33.96 | 22.64 | 89.0 | 0.0 | 6.2 | 6.9 | 0.0 |

### 8.2 同口径的线性探针参照（train 拟合 LDA，无时序）

| 参照 | acc | edit | F1@0.1 | F1@0.25 | idle | flush | insert | withdraw | sbc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear-probe（bbox-40-global） | 29.78 | 38.28 | 38.52 | 22.22 | 37.4 | 0.0 | 17.7 | 32.8 | **58.8** |
| linear-probe（bbox-40-hand） | 23.46 | 37.24 | 33.59 | 19.85 | 17.6 | **100.0** | 5.2 | **62.5** | 0.0 |
| linear-probe（bbox-80-global-hand） | 34.94 | 37.61 | 35.56 | 20.74 | 50.1 | 0.0 | 13.4 | 37.5 | 0.0 |
| **linear-probe（roi-grid-144）** | 49.37 | **50.63** | **45.26** | **33.58** | 71.7 | 7.3 | **26.4** | 22.8 | 0.0 |
| linear-probe（roi-grid-96-v2） | 44.71 | 49.80 | 39.74 | 33.11 | 68.1 | 0.0 | 16.5 | 27.0 | 0.0 |

（模型行为 3 seed 中位数；探针为确定性单点，无 seed 方差。两者口径、冷启动、指标实现完全一致。）

### 8.3 结论：**当前训练配方下的时序模型没有跑赢"特征 + 同后处理"的线性下界**

1. **段级**：`linear-probe（roi-grid-144）` 的 edit 50.63 / F1@0.1 45.26 / F1@0.25 33.58
   **高于全部五个 GRU 方案**（edit 最高 47.18、F1@0.1 最高 41.54）；其余四个探针也在
   33~38 的 edit 区间与模型持平。
2. **逐类**：探针在多个类上大幅领先 —— bbox-40-hand 探针 flush **100%**、withdraw 62.5%；
   bbox-40-global 探针 sbc **58.8%**；roi-144 探针 insert 26.4%。而所有 GRU 方案的
   flush ≤4.9%、insert ≤9.6%、sbc ≤23.5%。**即特征里有信号，GRU 没学出来**（与第六轮
   6.5 的定性判断一致，本轮给出同口径的定量证据）。
3. **特征契约的排序变得清晰**：roi-grid-144（显式空间分区）在两个口径下都是最强契约
   （模型与探针双料第一）；bbox-80-global-hand 最弱；`roi-grid-96-v2` 的可见性重排在
   探针口径下也接近 v1（39.74 vs 45.26 F1@0.1），但同样没有超越。
4. **下一步的重心应从"造新特征"转向"训练侧与标签侧"**：
   (a) 训练配方（早停/正则/类别权重/窗口长度/是否归一化）——探针说明上限远未触达；
   (b) project-18 的 flush 标签审计（第六轮遗留，flush 在 val 上 62% 而 test 上探针也仅 7.3%）；
   (c) 若要让时序建模真正加分，需要更长上下文或事件级目标，而不是继续堆几何特征。

### 8.4 工具行为（本轮固化）

- `--smoothing-min-duration N`：评估口径参数化（配置变体写 `<runs-dir>/_eval_cfg/`，
  结果写 `<runs-dir>/_eval_mdN/`，与默认口径互不污染）；
- 汇总表自动附 ①口径（从 `inference.smoothing` 解析）②逐类召回上限 ③线性探针参照行；
- `--no-probe-baseline` 可关闭探针；探针计算失败只告警、不让汇总整体失败。

## 第九轮：P2a 输入归一化——修的是"量纲病"，只对 ROI 契约有效（2026-09-18）

> 第八轮结论：GRU 没跑赢"特征 + 同后处理"的线性下界。本轮按 `INPUT_DESIGN_PROPOSAL.md`
> §2 P2a 落地输入归一化对照：GRU 历史上直接吃原始量纲，而 ROI 契约内部跨两个数量级
> （presence/count 0.1~0.7、max_area 1e-4~1e-2），在 wd=1e-4 下小量纲通道被系统性压制。
> 工具：`python tools/run_strategy_matrix.py --runs-dir runs/norm_zscore
> --strategies roi-grid-144,bbox-40-global --seeds 42,7,2026
> --set model.normalization=zscore --smoothing-min-duration 5`

### 9.1 实现（默认关闭，保历史口径）

- `model.normalization: none（默认）| zscore` + 可选 `model.norm_clip`（标准化后截断 ±clip）；
- 统计只按 **train split** 拟合（无泄漏），写入 buffer 并随 checkpoint 持久化，
  评估/在线推理自动复用同一变换；守卫 `std < 1e-3 → 1.0`（比 MS-TCN 的 1e-4 更严，
  理由见提案 §1.3：该契约下曾出现 |z|≈95.6）；
- checkpoint meta 记录 `normalizer` 字段（N0 为 `None`，诚实声明"未归一化"）；
- 单测：`framework/tests/test_gru_normalization.py`（7 项：直通不变、统计与守卫、
  归一化等价性、截断、工厂接线、非法值、配置白名单）。

### 9.2 结果（md=5，test = project-18，3 seed 中位数）

| 方案 | acc | edit | F1@0.1 | F1@0.25 | idle | flush | insert | withdraw | sbc | 非idle帧 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| roi-grid-144 **N0**（原样喂入） | **51.99** | 41.13 | **41.54** | 26.89 | 86.0 | 3.7 | 9.6 | 3.1 | 0.0 | 409 |
| roi-grid-144 **N1**（z-score） | 49.41 | **50.69** | 39.67 | **29.75** | 77.2 | **17.7** | **18.1** | 1.5 | 0.0 | 663 |
| linear-probe roi-144（同口径下界） | 49.37 | 50.63 | 45.26 | 33.58 | 71.7 | 7.3 | 26.4 | 22.8 | 0.0 | 846 |
| bbox-40-global **N0** | 45.36 | 47.18 | 34.92 | 26.89 | 78.5 | 4.9 | 0.0 | 13.5 | 11.8 | 437 |
| bbox-40-global **N1**（z-score） | 43.46 | 46.21 | 35.97 | 26.87 | 75.3 | 0.0 | 7.4 | 13.5 | 5.9 | 707 |

### 9.3 结论

1. **归一化对 ROI 契约是实质增益**：edit 41.13 → **50.69**（+23% 相对），F1@0.25 +2.9，
   flush recall 3.7 → **17.7**（≈5×）、insert 9.6 → **18.1**（≈1.9×），非 idle 预测帧
   409 → 663（坍缩缓解）。机制与提案 §1.3 的假设一致：z-score 后各维等权，max_area 通道
   不再被 wd 压制。
2. **归一化把"模型层"差距补平**：N1 的 edit 50.69 与线性探针下界 50.63 基本相等
   （F1@0.1 仍差 5.6、逐类 recall 仍差：探针 insert 26.4 / withdraw 22.8 vs 18.1 / 1.5）
   → 剩下的差距是**时序建模本身**（尤其 withdraw 的方向判别），不再是量纲/优化问题。
3. **不是万能增益**：bbox-40-global 用 N1 反而略降（edit 47.18 → 46.21，sbc 11.8 → 5.9）。
   该契约特征已在 0~1 量纲、无跨数量级问题——**按契约决策，不要无脑开**。
4. 与 `INPUT_DESIGN_PROPOSAL.md` §0 判断 3 互为印证：主线 GRU 此前完全无输入归一化，
   而 MS-TCN 一直有——历史上"架构对照"其实混着归一化口径差异，跨架构比较需注意。

### 9.4 工具侧修正（本轮附带）

- **评估统一改用 run 的 `config.resolved.json`**：`-S` 覆盖（如 `model.normalization`）
  会改变网络结构（buffer），用源配置评估会加载失败或按另一口径评估。已核对与源配置评估
  结果逐位一致（acc/edit 相同）。
- **口径变体改用 `apply_overrides` 生成**（此前按文本插入 `smoothing_min_duration`，
  对 JSON 形式的 resolved 配置静默失效，会"看着像 md=5、实际是 md=25"）。
- `--set KEY=VALUE`：把任意 `-S` 覆盖透传给训练（可多次）。

## 第十轮：窗口长度消融 + 方向判别探针——insert/withdraw 的信息不在特征与上下文里（2026-09-18）

> 第九轮把 ROI 契约的"量纲病"修好后，模型层与线性探针的差距集中在 `long_brush_withdraw`
> （探针 22.8% vs GRU 1.5%）。两个假设：①滑窗太短（16 帧 ≈2.1 s，而 insert/withdraw 真实段
> 中位 61/27 帧）②方向信息需要更长上下文。本轮用**窗口消融**直接验 ①，用**方向判别探针**
> 验 ②，并给出"还能往哪走"的结论。

### 10.1 窗口长度消融（roi-grid-144 + z-score 归一化，md=5，3 seed 中位数）

| 窗口 | acc | edit | F1@0.1 | F1@0.25 | idle | flush | insert | withdraw | sbc | 逐 seed edit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16（原配置） | 49.41 | **50.69** | 39.67 | 29.75 | 77.2 | 17.7 | **18.1** | 1.5 | 0.0 | 56.32 / 50.69 / 46.30 |
| 32 | 50.55 | 41.03 | 40.65 | **31.50** | 80.0 | 3.7 | 15.6 | 7.7 | 0.0 | 41.08 / 41.03 / 37.20 |
| 48 | 51.46 | 47.10 | **40.65** | 29.27 | 79.2 | 17.1 | 14.0 | **13.9** | 0.0 | 47.27 / 37.76 / 47.10 |

**要点**：三个窗口的 edit 排序 50.69 > 47.10 > 41.03 **非单调**，而同一臂内 seed 极差达
6~10 个点 —— 窗口长度在当前数据上**没有可辨增益**（acc 随窗口略升 = 更偏 idle）。逐类同样
在噪声内：insert 18.1→15.6→14.0（略降）、withdraw 1.5→7.7→13.9（绝对值仍近 0）、flush 来回跳。
`runs/win32/`、`runs/win48/`（各 3 run + 线性探针参照）。

### 10.2 方向判别探针：这对类**不可分**，且线索是"视频专属"的

`python tools/probe_direction.py`（本轮新增，纯 CPU、只读数据）：在 train 的
{insert, withdraw} 帧上拟合二分类 LDA，在 val/test 的同类帧上量 AUC；`居中-k`/`因果-k`
表示把前后 2k+1 / k+1 帧特征拼起来后再分类。

| 契约 | 上下文 | 维度 | AUC(train) | AUC(val) | AUC(test) |
|---|---|---:|---:|---:|---:|
| bbox-40-global | 单帧 | 40 | 0.678 | 0.478 | 0.500 |
| bbox-40-global | 居中-24 | 1960 | **0.977** | 0.430 | 0.474 |
| bbox-40-hand | 单帧 | 40 | 0.573 | 0.245 | 0.454 |
| bbox-40-hand | 居中-24 | 1960 | **0.954** | 0.377 | 0.533 |
| bbox-80-global-hand | 居中-24 | 3920 | **0.997** | 0.382 | 0.569 |
| roi-grid-144 | 单帧 | 144 | 0.726 | 0.485 | 0.367 |
| roi-grid-144 | 居中-24 | 7056 | **0.999** | 0.472 | 0.462 |
| roi-grid-96-v2 | 单帧 | 96 | 0.746 | 0.440 | 0.561 |
| roi-grid-96-v2 | 居中-24 | 4704 | **1.000** | 0.451 | 0.481 |

（全部 5 个契约 × 4 种上下文的完整表见 `tmp/direction_probe/all_contracts.json`；
单帧/因果-8/居中-8 的 AUC(test) 全部落在 0.32~0.57。）

**读法（三条）**：

1. **加长上下文没用**：把窗口从 1 帧加到 49 帧（居中-24），test AUC 只在 0.46~0.53 徘徊，
   离"可用"（>0.7）差得远 —— 方向信息不在帧间几何关系里。这与 `INPUT_DESIGN_PROPOSAL.md`
   §1.4 在 train 上的结论一致，且本轮把它扩到**新 test 批次**与**多种上下文长度**。
2. **train 上的高 AUC 是"记住视频"不是"学会方向"**：五个契约在 train 上居中-24 全部达到
   **0.95~1.000**（roi-96-v2 为 1.000），但同一判别方向在 **val（同批次、不同视频）只有
   0.245~0.485**、test 0.32~0.57 —— 连批次内换视频都不迁移。维度（最高 7056）远大于训练帧数
   （insert+withdraw 共约 1.7k 帧），完美分离只是记忆；而这些几何量编码的更像"是哪个视频/
   哪台镜子"，不是"刷子在进还是出"。
3. **多分类里的非零 recall 别当方向能力**：逐帧线性探针（roi-grid-144, test）的
   insert recall 39.2% / precision 43.9%、withdraw 25.5% / 26.8% —— "刷子在镜身内"这一
   **组级**可分性是真的（insert+withdraw 一起 vs idle），但 in/out 的切分基本是阈值噪声。
   报告逐类指标时应同时给 precision，否则会把组级可分性误读成方向判别。

### 10.3 结论：特征侧与上下文侧的工作到此为止，剩下三个杠杆

结合第七~十轮，跨批次 test 上的结论链条已经闭合：

1. **特征工程**：已试 5 个契约（40/40/80/144/96）+ 可见性重排 + 归一化；ROI-144 最强，
   归一化补齐量纲病后 edit ≈ 线性探针下界；**没有证据支持再加几何/运动学块**
   （提案 §1.4 + 本轮 §10.2 双重否证）。
2. **时序建模**：窗口 16/32/48 无可辨差异；GRU 已到达"特征 + 同后处理"的线性下界。
3. **因此剩下的杠杆是**：
   (a) **标签审计**（提案 §1.4 的零成本排除项）：insert/withdraw 的 in/out 切分若不与可见
       动作一致，任何模型都无法学；flush 同理（val 62% vs test 探针 7.3%）；
   (b) **像素通道**（提案 P1：按 ROI 取 crop embedding）——几何量已被证否，像素是唯一
       尚未被否证的信息源；
   (c) **检测覆盖**（提案 §1.1：8 类里 5 类出现率 <8%，`brush_tip_out` 仅 0.4%）——
       判别物看不见时，标签再准也学不到。

### 10.4 工具与复现

```bash
python tools/probe_direction.py --json tmp/dir.json            # 方向可分性 + 上下文效应
python tools/probe_direction.py --pair idle,flush --contract roi-grid-144
python tools/run_strategy_matrix.py --runs-dir runs/win32 --strategies roi-grid-144 \
    --set model.normalization=zscore --set train.window=32 --smoothing-min-duration 5
```

## 第十一轮：像素通道探针（提案 P1）+ 检测覆盖逐 split——三类失败的三种根因（2026-09-18）

> 第十轮把"再加几何/上下文"否掉了，剩下的候选是**像素**（提案 P1）与**检测覆盖**（提案 §1.1）。
> 本轮两件都做：① 在机制床 `datasets/cleansight-ActionMixed`（图/框/标签齐全、整帧 embedding
> 已预计算）做提案指定的四臂冻结特征线性探针；② 在**新数据**上按 split 复查检测覆盖，
> 回答"每个类的判别物到底在不在"。工具：`python tools/probe_pixel_channel.py
> {extract,evaluate}`（本轮新增，tracked）。

### 11.1 四臂对照（冻结 mobilenet_v3_small + 等先验 LDA；逐类为 recall/precision）

| 臂 | 维度 | split | macro-F1 | acc | flush | lb_insert | lb_withdraw | sbc |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A 整帧 224²（= E1 现状） | 576 | val | 55.9 | 52.6 | 96/86 | 27/92 | 61/29 | 46/78 |
| A 整帧 224² | 576 | test | 36.9 | 36.7 | 35/89 | 21/98 | n/a | **7/6** |
| B hand 裁剪（+presence） | 577 | test | 40.4 | 37.7 | 33/83 | 21/91 | n/a | 30/23 |
| C hand+scope 两槽（+presence） | 1154 | test | **43.6** | 39.7 | 34/92 | 20/94 | n/a | **41/45** |
| D ROI-144（文本特征，参照） | 144 | test | **55.6** | 45.3 | 35/88 | 38/79 | n/a | **82/36** |

（val 行同样是四臂都有：A 55.9 / B 47.4 / C 49.3 / D 52.6 macro-F1；完整表
`tmp/pixel_probe/P1_PROBE.md`。机制床 test 无 lb_withdraw 帧，故 n/a。）

**要点**：

1. **"裁剪比整帧好"这个方向被证实**：test 上 C 43.6 > B 40.4 > A 36.9（macro-F1），
   sbc 列 41 > 30 > **7** —— 整帧 embedding 对 sbc 几乎不可用（recall 7%/precision 6%），
   裁到手+镜身区域后恢复到 41/45。提案"判别物在整帧里只有 9~38 px"的判断在小目标上成立。
2. **但仍不及文本 ROI-144**：D 在同床 test 上 macro-F1 55.6、sbc 82/36，全面高于 B/C。
   按提案自己的判据（"B/C 明显高于 A **且 ≥ D** 才立项 E2"）→ **不满足，像素通道暂不立项**；
   尤其 auto 通道本身没有图像源，为它引入像素还要先解决图像链路。
3. **C 的价值在 precision**：sbc precision 45（C）vs 36（D）—— 裁剪像素更"准"但召回低，
   与"冻结分类 backbone 只见外观、不见动作"的预期一致。

### 11.2 检测覆盖逐 split：sbc 的"定义物"在新 test 批次里消失了

`tools/probe_input_features.py`（新数据 15,598 帧）+ 逐 split 复查：

| split | short_brush 总出现率 | **在 sbc 帧上的出现率** | syringe 在 flush 帧 | brush_tip_out 在 insert 帧 |
|---|---:|---:|---:|---:|
| train | 1.7% | **11.7%**（idle 1.5% 的 ~8×） | 5.7% | 0.5% |
| val | 3.0% | **26.9%**（idle 2.0% 的 ~13×） | 3.3% | 1.8% |
| **test** | 0.4% | **0.0%**（51 帧里一次都没检出） | 18.3% | 1.4% |

**读法（这条解释了很多）**：

- **sbc 在新 test 上从特征层就不可学**：定义物 `short_brush` 在 51 个 sbc 帧上**检出 0 次**，
  而 train/val 上是 8~13× 富集。任何几何特征、任何模型、任何窗口都无能为力 —— 这与第十/七轮
  观测到的 sbc 召回 0% 完全一致（第七轮 md=25 的"0% 可达"只是第二重打击）。
- **flush 不是覆盖问题**：其判别物 `syringe` 在 test 的 flush 帧上出现率 18.3%，
  比 train 的 5.7% 更高，却仍召回 0~35% → 指向**视觉/标注口径差异**（第六轮 val 62% vs test 0%
  的对照同向），标签审计优先级最高。
- **insert/withdraw 的判别物太稀**：`brush_tip_out` 在 test 的 insert 帧 1.4%、withdraw 帧 0%
  —— 2~3× 于 idle 的弱信号，配合第十轮"上下文不可分"，判定为**信息量不足**。

### 11.3 合并结论：三个类三种根因，对应三种不同行动

| 类 | 现象 | 根因（本轮证据） | 行动 |
|---|---|---|---|
| `short_brush_cleaning` | test 召回 0（全方案、全口径） | **检测覆盖**：定义物 short_brush 在 test 的 sbc 帧检出 0%（11.2） | 检测侧扩召回（逐类降阈值/换权重/补帧）或把 sbc 从 test 评测口径移出 |
| `flush` | val 62~96% → test 0~35% | **标签/视觉口径**：判别物在 test 反而更常见（18.3%） | **标签审计**（零成本，最高优先） |
| `long_brush_insert / withdraw` | 逐类 recall 近 0；方向不可分 | **信息不存在**：AUC≈0.5（第十轮），几何/上下文/像素裁剪都不解决（11.1） | 像素通道整体降权；先做标签审计确认 in/out 标注与可见动作一致 |

**对新数据（auto 通道）的可执行结论**：特征侧已经没有便宜的增益可取（5 契约 + 重排 + 归一化 +
窗口 + 像素通道全部试过）；**下一步收益最高的是检测覆盖与标签质量**，其次是给 auto 通道补图像源
后重估像素通道。

### 11.4 复现

```bash
python tools/probe_pixel_channel.py extract     # 抽 B/C 臂裁剪 embedding（CPU，约 2 分钟，结果缓存）
python tools/probe_pixel_channel.py evaluate    # 四臂线性探针对照
python tools/probe_input_features.py --root datasets/cleansight-ActionMixed-auto-lhh --json tmp/cov.json
```

## 第十二轮：同口径跨架构 / 跨 feed-mode 对照（GRU vs MS-TCN vs Transformer，2026-09-18）

> 提案 §1.3 指出：历史上的"架构对照"混着归一化口径差异（MS-TCN 一直做 z-score、主线 GRU 没有），
> 结论不可比。第九轮给 GRU 补上归一化后，本轮做一次**干净的同口径对照**：同一 ROI-144 契约、
> 同一配方（wd=1e-4 / dropout=0.2 / patience=4 / best_metric=val_f1_0.5 / epochs≤20）、
> 同一评测口径（**md=1**：全序列模型没有因果平滑，因此把 GRU 也放到"无最小时长平滑"下比较）、
> 3 seed 中位数，新 test（project-18）。

### 12.1 结果（test，3 seed 中位数；逐类为帧级 recall %）

| 架构 | 推理类别 | acc | edit | F1@0.1 | F1@0.25 | idle | flush | insert | withdraw | sbc | 非idle帧 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **GRU**（因果滑窗 w=16，+z-score） | 流式 | 49.91 | **35.75** | **32.50** | **25.37** | 75.7 | **19.5** | **18.4** | 1.9 | 0.0 | 520~710 |
| MS-TCN（全序列，非因果） | 离线 | **53.20** | 32.50 | 24.20 | 14.01 | 91.2 | 15.2 | 0.0 | 4.2 | 0.0 | 81~304 |
| Transformer（全序列，非因果） | 离线 | **55.25** | 13.70 | 19.75 | 12.35 | **100.0** | 0.0 | 0.0 | 0.0 | 0.0 | **0~243** |
| linear-probe（逐帧，参照） | — | 49.53 | 19.96 | 10.67 | 5.71 | 64.3 | 15.2 | **37.9** | **25.5** | **21.6** | 1060 |

### 12.2 结论

1. **因果流式 GRU 在本 test 上全面最好**：段级 edit 35.75 / F1@0.1 32.50 / F1@0.25 25.37
   三项第一，且非 idle 逐类召回（flush 19.5、insert 18.4）也是三个模型里最高 ——
   **跨批次场景下"流式小模型"没有被离线大模型打败**，反而更稳（也意味着它适合生产路径）。
2. **Transformer 的高 acc 是坍缩假象**：acc 55.25 最高，但 idle recall 100%、3 seed 里 2 个
   非 idle 预测帧为 0 —— 与第三/四轮旧的坍缩诊断同源（多数类友好指标被骗）。**报 acc 时必须
   同时给非 idle 帧数与逐类召回**。
3. **MS-TCN 居中**：acc 53.20 且不坍缩，但 insert 召回 0（对刷子活动整体判 idle），
   段级分数主要来自 flush/lb 段。
4. **线性探针的定位要分清**：md=1 下它的段级指标差（10.67，逐帧抖动无时间平滑），
   但逐类帧级召回最高（insert 37.9 / withdraw 25.5 / sbc 21.6）→ 它是**帧级可分性下界**，
   不是段级基线；段级指标奖励"时间持久性"，那正是模型提供的部分。

**口径caveat**：三架构共用的是**为 GRU 调出来的健康配方**；MS-TCN/Transformer 可能
在别的配方下更好（尤其 Transformer 的坍缩也许与 lr/epochs 有关），所以本表回答的是
"同一配方下谁更稳"，不是"各架构的最优水平"。CPU 口径，3 seed。

### 12.3 工具侧修正（本轮附带）

- **`--skip-train` 只 glob `gru-*`** 的 bug：MS-TCN/Transformer 的 run 目录以各自 `model.type`
  命名，之前不会被汇总进来（本轮修成扫所有含 `config.resolved.json` 的目录）。
- 探针参照对**全序列模型**不再套用滑窗冷启动与 min_duration（`pipeline != sliding_window_temporal`
  时 window=1、md=1），否则会人为压低参照（patch 后两个全序列策略的探针行完全一致，即为一致性自检）。

### 12.4 复现

```bash
python tools/run_strategy_matrix.py --runs-dir runs/arch_cmp --strategies roi-grid-144 \
    --seeds 42,7,2026 --set model.normalization=zscore --smoothing-min-duration 1
python tools/run_strategy_matrix.py --runs-dir runs/arch_cmp \
    --strategies mstcn-roi-144,transformer-roi-144 --seeds 42,7,2026 --smoothing-min-duration 1
python tools/run_strategy_matrix.py --runs-dir runs/arch_cmp --skip-train --smoothing-min-duration 1   # 合并汇总
```

## 第十三轮：可见性审计 + 归一化口径探针（2026-09-22）

> 新工具 [`tools/probe_segment_visibility.py`](../tools/probe_segment_visibility.py)：审计"每个动作类在当前检测特征里
> 能否被看见"，补上 `INPUT_DESIGN_PROPOSAL` §1.4 留下的**零成本待办**（"确认时间轴边界与可见动作是否一致"）。
> 纯 CPU、只读数据、不训练；LDA 一律在 train 上拟合，AUC 在 test 上量。

### 13.1 逐类可见性（roi-grid-144，train vs test）

| 动作类 | support(train) | support(test) | 段数(test) | AUC(train) | AUC(test) | 边界跳变（显著占比 / 效应量） | 最相关通道（Δ/σ） |
|---|---:|---:|---:|---:|---:|---|---|
| water_injection | 194 | 0 | 0 | 0.955 | n/a | n/a | hand.left-top.presence +1.58 |
| flush | 893 | 164 | 7 | 0.838 | **0.272** | 14%（0.44σ） | hand.left-top.presence +0.94 |
| long_brush_insert | 1,289 | 707 | 11 | 0.787 | 0.636 | 0%（0.62σ） | scope_control_body.mid-bottom.presence −0.66 |
| long_brush_withdraw | 437 | 259 | 9 | 0.847 | 0.586 | 0%（0.33σ） | hand.right-bottom.presence +1.10 |
| short_brush_cleaning | 488 | 51 | 6 | 0.849 | **0.762** | 0%（0.63σ） | scope_control_body.mid-bottom.max_area +0.74 |

三条读法：

1. **train 上所有类都可见（0.79–0.96），跨到 test 大面积退化**：flush 掉到 **0.272（比随机更差 = 判别方向失效）**，
   insert 0.636、withdraw 0.586；唯一稳住的是 sbc 0.762。
2. **flush 的机制被定位到通道级**：train 的 flush 帧里"左手出现在左上区域"的 presence 均值 **0.448**，
   而 test 的 flush 帧是 **0.000**（单变量 AUC 0.670 → 0.492）。标签语义没变，**视觉证据在新批次消失**——
   与第十一轮"标签/视觉口径"根因一致，并首次给出可量化的证据。
3. **sbc 需要修正结论**：定义物 `short_brush` 的通道在 test 上判别强度 ≈ **0.500（零信息）**，
   但 `scope_control_body.mid-bottom` 的 presence/max_area 在 test 上 AUC **0.703–0.727**（train 0.656–0.747，稳定）。
   即 sbc **不是"特征层不可学"，而是"可学线索换了通道"**（从定义物换到镜身）——§11.2 的
   "检测覆盖 0 → 特征层不可学"应修正为"定义物不可用、场景线索可用"。
4. **边界效应量 0.33–0.63σ、但显著占比 0%**：段内与段外确有差异，却没有形成"边界阶跃"——
   动作在检测上是渐变、标注边界落在渐变里。这解释了为什么段级边界指标（edit / F1@0.5）难提升，
   其中含数据侧原因，不只是模型问题。

### 13.2 单通道稳、组合崩（新判据）

按"逐检测类 × 统计量"聚合各通道在 test 上的最强判别强度（跨 4 个可评估类取均值）：

| 检测类 | presence | count | max_area |
|---|---:|---:|---:|
| hand | 0.676 | **0.703** | 0.657 |
| scope_control_body | 0.603 | 0.603 | 0.600 |
| scope_mid_section | 0.575 | 0.575 | 0.583 |
| scope_distal_end | 0.511 | 0.511 | 0.511 |
| syringe / air_gun / short_brush / brush_tip_out | 0.50–0.52 | 0.50–0.52 | 0.50–0.52 |

- **可迁移信号集中在 hand 与 scope_control_body / scope_mid_section**；四个"定义物"类（syringe、air_gun、
  short_brush、brush_tip_out）在 test 上几乎零信息。
- 单通道层面几乎没有"train 强 test 崩"：最大落差仅 **0.045**（scope_control_body.mid-bottom.max_area 0.747→0.703）。
- 但多变量 LDA 的 test AUC 掉到 0.27–0.76 → **失效发生在"通道组合"层面（组合方向过拟合 train），
  而不是单通道分布漂移**。可执行含义：**更少更稳的通道 + 更强正则**，可能比"144 维全给"更抗跨批次退化。

### 13.3 归一化口径探针（去视频电平）

对同一批特征做四种口径变换，再跑同样的"train 拟合 → test AUC"：

| 变体 | flush | insert | withdraw | sbc | 中位 |
|---|---:|---:|---:|---:|---:|
| ① 全局 z（现用） | 0.281 | 0.633 | 0.584 | 0.744 | **0.609** |
| ② 按视频 z | 0.528 | 0.640 | 0.562 | 0.593 | 0.578 |
| ④ 去视频均值 + 全局尺度 | **0.604** | 0.583 | 0.603 | 0.716 | 0.604 |
| ⑤ ①+② 拼接（288 维） | 0.646 | 0.505 | 0.472 | 0.563 | 0.534 |
| ⑥ ①+④ 拼接（288 维） | 0.185 | 0.445 | 0.501 | 0.417 | 0.431 |

- ②/④ 都把 **flush 从 0.28 修到 0.53/0.60**（当前最差类、AUC<0.5 = 误导性），代价是 sbc 0.744→0.593/0.716；
  中位数基本持平 → **归一化是"重新分配可迁移性"，不是普遍增益**。
- 拼接反而更差（0.534 / 0.431）：线性探针在更高维上过拟合，**下界不支持拼接**。
- 立项建议：若要试，选 **④ 作为附加视图**（而非替换全局 z），并用**带正则的模型**而不是探针裁决；
  flush 值得单独做一次对照（它是唯一 AUC<0.5 的类）。

### 13.4 留一检测类消融（初筛：8 类 × **seed 42/7 两点**，h32，lr 5e-4，60 轮）

| 变体 | edit | acc | F1@0.25 | 段数比 | 非 idle 帧 | flush R | insert R | withdraw R | sbc R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 基线（无遮罩） | 28.99 | 49.36 | 16.68 | 2.27 | 782 | 13.7 | 8.8 | 37.3 | 0.0 |
| 遮掉 **hand** | 28.35 | 51.67 | 17.40 | 2.21 | 692 | **32.6** | 9.1 | 31.5 | 0.0 |
| 遮掉 **scope_control_body** | **36.84** | 55.48 | 20.98 | 1.37 | **369** | 12.5 | 9.8 | 16.4 | 0.0 |
| 遮掉 **scope_mid_section** | 24.87 | 52.79 | 14.18 | 1.49 | 365 | 1.2 | 12.6 | **0.0** | 0.0 |
| 遮掉 scope_distal_end | 30.04 | 55.68 | 18.75 | 0.99 | **192** | 1.8 | 7.7 | 8.5 | 1.0 |
| 遮掉 **syringe** | 27.82 | 47.48 | 12.83 | 2.14 | 832 | **0.0** | 10.3 | 30.5 | 0.0 |
| 遮掉 air_gun | 30.16 | 51.42 | 17.59 | 2.08 | 778 | 13.4 | 9.3 | **46.3** | 0.0 |
| 遮掉 **short_brush** | 28.50 | 51.25 | 13.30 | **2.92** | 993 | 13.4 | **22.7** | 43.1 | 0.0 |
| 遮掉 **brush_tip_out** | 34.63 | 51.88 | 16.79 | 2.60 | 952 | 15.9 | **25.6** | 34.6 | 0.0 |

> ⚠️ **读法警告（本轮方法论收获）**：本表的 edit/acc **不能单独读**。看"非 idle 帧"一列——
> 遮掉 `scope_control_body` / `scope_mid_section` / `scope_distal_end` 后 edit 反而升高
> （36.84 / 24.87 / 30.04），同时非 idle 帧从 782 掉到 369 / 365 / **192**：那是**坍缩到 idle**
> （idle 段又长又好匹配），不是能力提升。**留一消融必须同时看非 idle 帧数与逐类 recall。**

读法（结合逐类 recall 才成立）：

1. **`syringe` 是 flush 的唯一依赖**：遮掉它 flush recall **13.7% → 0.0%**。与第九/十一轮"flush 的判别物
   是 syringe"一致，且首次给出"去掉即归零"的因果证据。
2. **`hand` 通道在压制 flush（初筛读数，已被 3 seed 修正）**：2 seed 下 flush recall 13.7% → 32.6%；
   **3 seed 复核只有 11.0% → 16.5%（§13.5）**——2 seed 把效应放大了约 3 倍。这一条正是"≥3 seed 且看配对"的
   反面教材，引用时以 §13.5 为准。
3. **`brush_tip_out` 与 `short_brush` 在压制 insert**：2 seed 下 insert recall 8.8% → **25.6% / 22.7%**，
   且活动量上升（952 / 993）；3 seed 组合复核确认方向（§13.5）。这两个是 §13.2 里"test 上零信息"的
   定义物通道——它们在 test 上制造误判。
4. **`scope_mid_section` / `scope_control_body` 承载 withdraw**：遮掉后 withdraw recall 37.3% → **0.0% / 16.4%**。
5. **"依赖 ≠ 可迁移"**：模型对 syringe（flush）、scope 系（withdraw）有强依赖，但这些通道在新批次的可迁移性
   并不高——**依赖越强，跨批次越脆**。这正好解释了 §13.1 的 AUC 崩塌。

**可执行的特征方案（"减法"而不是加法）**：候选删除/降权通道组 = `hand` + `short_brush` + `brush_tip_out`
（共 3 类 × 18 维 = 54 维，144 → 90 维），保留 `syringe`（flush）与 `scope_mid_section` / `scope_control_body`（withdraw）。
组合遮罩的 3 seed 复核已在 `runs/ablate-combo-*` 跑（见 §13.6）。

### 13.5 组合遮罩复核（3 seed：42/7/2026，h32，lr 5e-4，60 轮）

| 变体 | edit | acc | F1@0.25 | F1@0.5 | 段数比 | 非 idle 帧 | flush R | insert R | withdraw R | sbc R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 基线（全 144 维） | **33.91** | 50.63 | 17.91 | **10.95** | 1.77 | 599 | 11.0 | 14.3 | 16.6 | 0.0 |
| 遮掉 hand | 30.19 | 52.90 | 16.39 | 6.56 | 1.32 | 338 | 16.5 | 8.8 | 17.0 | 0.0 |
| **遮掉 hand+short_brush+brush_tip_out（54 维）** | 28.27 | 50.06 | **18.12** | 8.05 | 3.08 | **1148** | **25.0** | **24.2** | **40.9** | 0.0 |
| 反向对照：遮掉 syringe+scope_mid_section | 33.48 | 52.48 | 15.61 | 5.85 | 1.81 | 396 | **0.0** | 2.3 | 23.9 | 0.0 |

**结论：这是一个明确的"召回 vs 段级质量"权衡，而且方向可复现。**

1. **删掉 3 组干扰通道（144 → 90 维）显著提升稀类召回**：flush 11.0 → **25.0**、insert 14.3 → **24.2**、
   withdraw 16.6 → **40.9**（+14 / +10 / +24 点），非 idle 帧 599 → **1148**（不再偏 idle）。
2. **代价是段级质量**：edit 33.91 → 28.27、F1@0.5 10.95 → 8.05、段数比 1.77 → **3.08**（过分割）。
   即模型从"少说少错"变成"多说多中"，边界更碎。
3. **反向对照确立因果**：遮掉 `syringe+scope_mid_section` 后 flush recall **归零**、insert 掉到 2.3，
   反证这两组通道确实是 flush / insert 的依赖；而 §13.2 说它们在 test 上并不可迁移——
   **依赖越强，跨批次越脆**。
4. **方法论**：2 seed 的 `hand` 效应（+18.9）在 3 seed 下缩到 +5.5；`drop3` 的召回增益则在 3 seed 下
   依然成立（+14/+10/+24）。→ **初筛可以 2 seed，结论必须 ≥3 seed 且看逐类 recall，不看单一 edit。**

**立项建议（按目标二选一）**：
- 目标=召回（当前 sbc 0%、insert 14% 是主要痛点）→ 采用 `mask_targets: [hand, short_brush, brush_tip_out]`
  的 90 维契约，或用类别权重/阈值在解码侧达到同等效果（更便宜，不动特征）；
- 目标=段级指标（edit/F1@0.5）→ 不删通道，改在**过分割**上做功（T-MSE 权重、因果平滑 md）。

### 13.6 复现


```bash
# 可见性审计（纯 CPU、只读、不训练）
python tools/probe_segment_visibility.py --contract roi-grid-144 --split test --json tmp/visibility_roi144.json

# 留一 / 组合遮罩消融（mask_targets 支持逗号分隔多个检测类）
python tools/run_capacity_matrix.py --runs-dir runs/ablate-hand --hidden 32 --seeds 42,7 --epochs 60 \
    --set train.lr=0.0005 --set feature_schema.mask_targets=hand --no-eval-last
python tools/run_capacity_matrix.py --runs-dir runs/ablate-combo-drop3 --hidden 32 --seeds 42,7,2026 \
    --epochs 60 --set train.lr=0.0005 \
    --set feature_schema.mask_targets=hand,short_brush,brush_tip_out --no-eval-last
```

---

## 第十四轮：活动量旋钮 vs 特征删通道（2026-09-22）——推翻第十三轮的立项建议

> 第十三轮从留一消融推出"删掉 hand/short_brush/brush_tip_out 三组通道可提召回"。本轮用**损失侧的活动量旋钮**
> （新增配置项 `train.class_weight_clip`）做对照，结论是：**那个"召回增益"绝大部分是活动量效应，不是特征效应**；
> 同时找到一个**统计显著、零特征改动**的杠杆。

### 14.1 活动量旋钮是什么

`compute_class_weights` 按频率倒数算权重并截断到 `[lo, hi]`。idle 的原始倒数权重 ≈0.031，
默认下限 0.1 把它**抬高 3.2 倍**——这个"抬升"决定模型有多"敢说"非 idle。
新增 `train.class_weight_clip`（`[lo,hi]` 或 `"lo,hi"`，CLI 可传）后可对照：
下限 0.03 = 还原原始倒数频率（活动量↑）；0.2 = 更强强调多数类（活动量↓）。

### 14.2 5 个变体（h32、lr 5e-4、60 轮、3 seed 中位数；非 idle 真值 1181）

| 变体 | idle 权重 | edit | F1@0.1 | F1@0.25 | F1@0.5 | 段数比 | 非 idle 帧 | flush R | insert R | withdraw R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 基线（144 维） | 0.1 | 33.91 | 27.86 | 17.91 | 10.95 | 1.77 | 599 | 11.0 | 14.3 | 16.6 |
| **活动量↑ clip 0.03** | 0.03 | 23.81 | 18.51 | 13.87 | 5.87 | 3.59 | **1655** | 6.7 | **45.0** | **50.6** |
| 活动量↓ clip 0.2 | 0.2 | **36.13** | **34.07** | **22.22** | **11.29** | **0.85** | 218 | 7.3 | 13.7 | 2.7 |
| drop3 特征（90 维） | 0.1 | 28.27 | 24.79 | 18.12 | 8.05 | 3.08 | 1148 | **25.0** | 24.2 | 40.9 |
| drop3 + clip 0.2 | 0.2 | 30.92 | 25.00 | 15.91 | 7.95 | 1.23 | 354 | 10.4 | 10.9 | 13.9 |

- **活动量被旋钮单调控制**（非 idle 帧 218 / 599 / 1655 ↔ idle 权重 0.2 / 0.1 / 0.03）——机制确定。
- **drop3 与 clip 0.2 叠加没有协同**（edit 30.92 不如纯 clip 0.2，召回也回落）→ 两者作用在同一个"活动量"维度上。

### 14.3 配对检验（逐 (seed, 视频) Wilcoxon）——决定性的部分

| 变体 vs 基线 | 指标 | n | 胜/负 | 中位差 | p | 判读 |
|---|---|---:|---:|---:|---:|---|
| clip 0.03（活动量↑） | **insert recall** | 15 | **13 / 2** | **+27.9pp** | **0.0012** | **显著** |
| clip 0.03 | withdraw recall | 12 | 7 / 2 | +20.5pp | 0.074 | 边界 |
| clip 0.03 | flush recall | 6 | 3 / 2 | +0.5pp | 1.00 | 无变化 |
| clip 0.03 | edit | 23 | 10 / 13 | −4.20 | 0.22 | 不显著 |
| drop3 特征 | **flush recall** | 6 | **6 / 0** | **+15.7pp** | **0.031** | **显著** |
| drop3 特征 | insert recall | 15 | 5 / 4 | ±0.0pp | 0.57 | **不显著（推翻第十三轮）** |
| drop3 特征 | withdraw recall | 12 | 5 / 3 | ±0.0pp | 0.20 | 不显著 |
| clip 0.2（活动量↓） | edit / F1@0.25 | 21 | 13 / 8 | +3.72 / +1.59 | 0.31 / 0.58 | 不显著 |
| clip 0.2 | withdraw / flush recall | 12 / 6 | 1 / 4、1 / 3 | ±0.0、−1.7pp | 0.19 / 0.25 | 不显著（偏保守） |

### 14.4 结论（覆盖第十三轮的建议）

1. **"删通道提召回"不成立**：drop3 的 insert（+9.9 中位）与 withdraw（+24 中位）在配对检验下都**不显著**
   （p=0.57 / 0.20）——第十三轮读到的增幅来自 seed/视频混合。**drop3 唯一站得住的效果是 flush
   +15.7pp（p=0.031）**，值得单独保留为"flush 专项"候选，但不再作为通用特征方案立项。
2. **活动量旋钮才是可用的召回杠杆**：`train.class_weight_clip=0.03,5.0` 让 **insert 召回 +27.9pp
   （p=0.0012）**、withdraw +20.5pp（p=0.074），代价是 edit/F1@0.5 下降与过分割（段数比 3.59）。
   **零特征改动、一行 `-S`**。
3. **活动量↓ 给"段级指标"最好的一档**（edit 36.13 / F1@0.1 34.07 / F1@0.25 22.22 / 段数比 0.85，
   均为 h32 历史最好），但配对不显著，且把稀类召回压到很低——它是"保守策略"，不是改进。
4. **sbc 在全部 5 个变体里都是 0.0 召回**：确认它不可能靠特征/损失侧解决（与 §13.1 的"定义物不可见、
   可迁移线索在镜身通道"一致），只能动检测侧或标签侧。
5. **方法论（第二轮验证）**：留一/组合消融的"中位数提升"必须在**逐 (seed,视频) × 类** 上做配对检验，
   否则会把 seed 混合的假象当成特征效应（本轮 drop3 就是反例）。

### 14.5 下一步（按判读强度排序）

1. **叠加实验**：clip 0.03（显著提 insert/withdraw）+ drop3（显著提 flush）**作用在不同类上**，
   是目前唯一"两个显著效应各自成立"的组合 → 3 seed 复核是否可叠加。
2. **给 clip 0.03 配过分割对策**：换 `mstcn2`（自带 T-MSE 平滑）或调 `evaluation.smoothing_min_duration`，
   看能否保住召回的同时把段数比从 3.59 拉回 ~1.5。
3. **段级指标路线**：clip 0.2 的段级分数最好但不显著，值得在 mstcn2 h128/s4l10（当前最强模型）上复测。

### 14.6 复现

```bash
# 活动量旋钮（新增配置项 train.class_weight_clip，[lo,hi] 或 "lo,hi"）
python tools/run_capacity_matrix.py --runs-dir runs/cwclip-lo003 --hidden 32 --seeds 42,7,2026 \
    --epochs 60 --set train.lr=0.0005 --set train.class_weight_clip=0.03,5.0 --no-eval-last
python tools/run_capacity_matrix.py --runs-dir runs/cwclip-lo020 --hidden 32 --seeds 42,7,2026 \
    --epochs 60 --set train.lr=0.0005 --set train.class_weight_clip=0.2,5.0 --no-eval-last
pytest framework/tests/test_class_weight_clip.py -q
```

---

## 第十五轮：执行第十四轮的三项建议 + 序列去均值（2026-09-22）——显著效应全在"类级召回"，段级指标一条都没推上去

> 第十四轮 §14.5 留了三件下一步：①叠加 `clip 0.03` + drop3 特征；②给 `clip 0.03` 配过分割对策（换 mstcn2）；
> ③在最强模型（mstcn2 h128/s4l10）上复测 `clip 0.2`。本轮把三件全部跑完，并新增一个**特征侧旋钮**
> `model.sequence_normalization=demean`（序列去均值，动机是第十三轮 §13.3 探针：它把最差类 flush 的
> train→test 可迁移性从 0.281 修到 0.604）。
>
> **结论先行**：三项建议里两项被兑现（叠加成立、mstcn2 确实治住过分割），但**本轮 6 个显著效应全部落在
> 类级召回上，段级指标（edit / F1@IoU）一个都没显著提上去**；而且**我上一轮对 B 的读法是错的**（见 §15.3 第 1 条）。

### 15.1 八个组的汇总（官方 `metrics.summary`，3 seed 中位数；lr 5e-4、60 轮、144 维 ROI）

| 组 | 变体 | 参数量 | edit | F1@0.1 | F1@0.25 | F1@0.5 | acc | tp/fp/fn@0.5 | 段数比 | 非 idle 帧 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ① | mstcn h32（基线） | 38,118 | 33.91 | 27.86 | 17.91 | 10.95 | 50.63 | 11/117/62 | 1.77 | 599 |
| ② | mstcn2 h32/s2l5（**默认** clip） | 67,500 | 36.00 | 32.79 | 22.64 | 11.11 | **53.73** | 8/**63**/65 | **0.97** | 484 |
| A | mstcn h32 + clip 0.03 + drop3 | 38,118 | **36.54** | 23.78 | 16.08 | 6.29 | 45.28 | 9/204/64 | 2.92 | 1271 |
| B | mstcn2 h32/s2l5 + clip 0.03 | 67,500 | 36.28 | **36.48** | **25.16** | **13.50** | 47.40 | 11/79/62 | 1.23 | 1338 |
| ③ | mstcn2 h128/s4l10（**默认** clip，当前最强） | 3,312,664 | **51.47** | **44.78** | **35.82** | 12.00 | **55.25** | 9/**53**/64 | **0.84** | 671 |
| C | mstcn2 h128/s4l10 + clip 0.2 | 3,312,664 | 49.03 | 40.26 | 29.87 | 12.99 | 54.83 | 10/60/63 | 0.97 | 876 |
| D | mstcn h32 + demean | 38,118 | 30.19 | 20.06 | 14.42 | 5.64 | 37.21 | 8/237/65 | 3.37 | 1490 |
| E | mstcn h32 + demean + clip 0.03 | 38,118 | 32.60 | 22.37 | 14.45 | 5.42 | 39.18 | 8/214/65 | 3.04 | 1602 |

非 idle 真值 = **1181 帧**，真值总段数 = **73**。段数比 = 预测总段数 / 真值总段数（每 run 先算总比再取 seed 中位，
与第十四轮同法）。三项建议的对应：①→A、②→B、③→C；D/E 是本轮新增的 demean 臂。

读数：

- **B 的"好看"主要来自架构，不来自 clip**：同架构、**默认 clip** 的 ② 已经是 edit 36.00 / F1@0.1 32.79 /
  F1@0.25 22.64 / **fp 63**（比 ① 的 117 少 46%）/ acc 53.73（比 ① 高 3.1pp）。B 在 ② 之上只多换来
  F1@0.1 +3.7、F1@0.25 +2.5，代价是 acc −6.3pp、fp 63→79。**"B 全面优于基线"这个读法是错的**——
  真正的对比是 **② vs ①（换架构）**，B 只是"换架构 + 活动量↑"。
- **A（叠加）段级最差的一档**：edit 36.54 是 h32 家族最高，但 F1@0.5 只有 6.29（全场最低）、fp 204、段数比 2.92。
- **D/E（demean）把 acc 打到 37~39**——**低于全 idle 基线 55.25 达 16~18pp**，段数比 3.04~3.37，是"高召回低精度"的极端。
- **③ 仍是全场最强**，比本轮任何小模型高 13~15 edit；注意它 acc 55.25 恰好等于全 idle 基线（非 idle 仅 671 帧）
  ——最强模型的段级优势是在"敢说非 idle"的克制下取得的。

### 15.2 配对检验（逐 (seed, 视频) × 类 Wilcoxon）——本轮的判读全靠这张表

| 对比 | 指标 | n | 胜/负 | 中位差 | p | 判读 |
|---|---|---:|---:|---:|---:|---|
| **A** vs ① | **insert recall** | 15 | **13/2** | **+27.0pp** | **0.0020** | **显著** |
| **A** vs ① | **flush recall** | 6 | **6/0** | **+15.1pp** | **0.0312** | **显著** |
| A vs ① | edit | 24 | 16/8 | +5.1pp | 0.197 | 不显著 |
| A vs ① | F1@0.1 | 24 | 16/8 | +3.6pp | 0.114 | 不显著 |
| A vs ① | withdraw recall | 11 | 7/4 | +7.1pp | 0.765 | 不显著 |
| **B** vs ② | **insert recall** | 15 | **11/4** | **+17.8pp** | **0.0353** | **显著** |
| B vs ② | edit | 21 | 12/9 | +11.1pp | 0.614 | 不显著 |
| B vs ② | F1@0.1 / F1@0.25 | 20 / 21 | 12/8、14/7 | +4.0 / +6.1pp | 0.674 / 0.187 | 不显著 |
| **B** vs ① | **insert recall** | 15 | **14/1** | **+21.2pp** | **0.0015** | **显著** |
| B vs ① | F1@0.1 / F1@0.25 | 23 / 22 | 12/11、14/8 | +5.0 / +7.6pp | 0.140 / 0.138 | 不显著 |
| ② vs ① | F1@0.1 | 23 | 14/9 | +5.7pp | **0.065** | 边界 |
| ② vs ① | edit / F1@0.25 | 23 / 23 | 13/10、14/9 | +5.5 / +6.1pp | 0.540 / 0.267 | 不显著 |
| ② vs ① | withdraw recall | 6 | 1/5 | −15.2pp | 0.438 | 不显著 |
| **C** vs ③ | **insert recall** | 11 | **10/1** | **+8.4pp** | **0.0049** | **显著** |
| C vs ③ | edit | 20 | 7/13 | **−7.9pp** | 0.173 | 不显著（方向为差） |
| C vs ③ | F1@0.25 / F1@0.1 | 21 / 21 | 7/14、9/12 | −6.7 / −1.0pp | 0.321 / 0.539 | 不显著（方向为差） |
| C vs ③ | withdraw recall | 6 | 5/1 | +2.4pp | 0.0625 | 边界 |
| **D** vs ① | **insert recall** | 13 | **10/3** | **+23.0pp** | **0.0479** | **显著** |
| D vs ① | edit / F1@0.1 | 21 / 22 | 8/13、8/14 | −5.4 / −5.5pp | 0.355 / 0.210 | 不显著（方向为差） |
| D vs ① | flush recall | 6 | 4/2 | +8.5pp | 1.000 | **无变化** |
| **E** vs ① | **insert recall** | 13 | **13/0** | **+29.7pp** | **0.0002** | **显著** |
| E vs ① | edit / F1@0.1 / F1@0.25 | 24 | 12/12、12/12、13/11 | −1.4 / −1.3 / +1.6pp | 0.944 / 0.689 / 0.710 | 不显著 |
| E vs ① | flush / withdraw recall | 6 / 8 | 3/3、3/5 | −7.6 / −7.6pp | 0.813 / 1.000 | 不显著 |
| 全部 | sbc recall | 0~3 | — | — | — | 样本不足（仍恒为 0 召回） |

### 15.3 结论

1. **修正第十四轮立项时的乐观读法（本轮最重要的一条）**：B（mstcn2 s2l5 + clip 0.03）在段级上的"提升"
   主要属于**架构**（② vs ①：F1@0.1 +5.7pp，p=0.065 边缘；fp 117→63；acc +3.1pp），**clip 0.03 在其上只
   显著换来 insert 召回（+17.8pp，p=0.035），段级三项 p=0.19~0.67**。凡把 B 与 ① 直接比而说"clip 有用"的
   读法都不成立。（该架构差值在 §15.5 补齐到 **8 seed** 后**转为显著**：F1@0.1 p=0.0044、F1@0.25 p=0.0176。）
2. **§14.5(1) 成立：两个类级效应可以叠加**。A 同时拿到 **flush +15.1pp（6/0，p=0.031）** 与
   **insert +27.0pp（13/2，p=0.0020）**，两个效应作用在不同类上、各自独立显著——这是全仓唯一"两个显著
   召回效应共存"的配方。但代价是 F1@0.5 6.29（全场最低）与 fp 204。
3. **§14.5(2) 成立：mstcn2 的 T-MSE 确实治住了 `clip 0.03` 的过分割**。段数比 **3.59（第十四轮 clip 0.03 单用）
   → 0.97（②）/ 1.23（B）**，且非 idle 帧仍高（484 / 1338）。但"保住召回"只对 insert 成立**且幅度缩水**
   （+27.9pp → +17.8pp），withdraw 不再显著（第十四轮 +20.5pp p=0.074 → 本轮 +14.6pp p=0.195）。
4. **§14.5(3) 不成立：`clip 0.2` 在最强模型上不改善段级**（edit −7.9pp p=0.17，7 胜/13 负；F1@0.25 −6.7pp p=0.32），
   **反而显著提 insert 召回 +8.4pp（p=0.0049）**。→ **该旋钮在强模型上的作用方向与 h32 上相反**：
   h32 上 `clip 0.2` = 保守（压召回、段级最好），h128/s4l10 上 `clip 0.2` = 提 insert、段级变差。
   **"clip 0.2 是段级最优策略"不能跨容量推广**，该旋钮必须与容量一起调，不能固定。
5. **新增 demean（D/E）是纯召回杠杆，且第十三轮 §13.3 的探针预期没有兑现**：insert 召回 +23.0pp（p=0.048）/
   **+29.7pp（13/0，p=0.0002）** 显著，但 **flush 召回 4胜/2负 p=1.00 无变化**——探针说它把 flush 可迁移性
   0.281→0.604，模型层没兑现。**再次印证"探针 AUC ≠ 模型指标"**（同第十四轮 drop3 的教训）。
   另外 E 的 insert（+29.7pp）与 D（+23.0pp）相当 → **demean 与 clip 0.03 不叠加出额外收益**，两者作用在
   同一个"敢说非 idle"的维度上（与第十四轮 drop3 × clip 0.2 的发现同型）。
6. **本轮 6 个显著效应全部是类级召回（insert ×4、flush ×1、insert+flush ×1），段级指标 0 个显著。**
   结合 fp（117 → 63~237）与 acc（50.63 → 37.21~55.25）看：**"活动量/召回"这条轴已经走到"拿精度换召回"的
   边界**，继续在同一维度上调参不会带来段级收益。段级要再上一档只剩两条路：**换更大模型规模**
   （③ 比 ① 高 17.6 edit）或**动标签时间轴**。
7. **sbc 依旧 0 召回且样本不足**（n=0~3），与 §13.1 / §14.4 一致：它不在特征/损失侧的可解空间内。

### 15.4 下一步（按判读强度排序）

1. **~~把 ② 的架构收益做扎实~~（已执行，见 §15.5）**：补 seed 1/2/3/4/5 到 8 seed 后，**F1@0.1（p=0.0044）
   与 F1@0.25（p=0.0176）双双显著**，insert 召回 +17.3pp（p=0.019）——这是整条探索线上第一个显著的段级
   改进。**下一步应把 `mstcn2 h32/s2l5` 当作新基线**，在其上重估此前所有"活动量/特征"结论是否仍然成立。
2. **在 ③ 上把 clip 扫成单调轴**（0.03 / 0.05 / 0.1 / 0.2）：确认"强模型上 clip 方向反转"是否单调，
   若是则说明该旋钮应与容量联动，而不是当作固定超参。
3. **停止在"活动量"维度继续加杠杆**（本轮已用 5 个变体把这条轴走到底），把人力移到定版结论 F 节的
   **标签审计**与**检测侧扩召回**（sbc / flush 的根因既不在特征也不在损失）。
4. 段级指标的下一档提升：**更大模型规模**（③ 3.31M vs ① 38k，+17.6 edit）优先于任何特征侧尝试。

### 15.5 8 seed 复核：**架构切换是本轮唯一显著的段级改进**（补齐 seed 1/2/3/4/5 后）

§15.2 里 ② vs ① 的 F1@0.1 只是边界（p=0.065，3 seed），而 §15.4(1) 把它列为唯一"可能真提精度"的方向。
按此把**两个架构在同 5 个新 seed 上补齐到 8 seed**（42 / 7 / 2026 + 1 / 2 / 3 / 4 / 5，新旧批次配置逐字节
一致），结论**由"边缘"转为"显著"**：

| 指标 | ① mstcn h32（8 seed 中位） | ② mstcn2 h32/s2l5（8 seed 中位） | 配对 n | 胜/负 | 中位差 | p |
|---|---:|---:|---:|---:|---:|---:|
| **F1@0.1** | 27.54 | **31.36** | 59 | **37/22** | **+6.3pp** | **0.0044** |
| **F1@0.25** | 18.36 | **21.03** | 56 | **38/18** | **+7.1pp** | **0.0176** |
| **insert recall** | — | — | 25 | **16/9** | **+17.3pp** | **0.0192** |
| edit | 34.55（摆幅 [21.6, 44.5]） | 38.36（摆幅 [31.7, 40.4]） | 59 | 34/25 | +7.7pp | 0.1791 |
| F1@0.5 | 10.14 | 10.04 | — | — | ±0.0 | — |
| acc | 52.71 | 52.47 | — | — | −0.2pp | 持平 |
| fp@0.5（中位） | 93 | **58** | — | — | −38% | — |
| flush / withdraw recall | — | — | 10 / 10 | 3/7、3/7 | −1.7 / −8.2pp | 0.68 / 0.56 |

判读：

1. **这是整条"特征与精度"探索线上第一个统计显著的段级改进**（F1@0.1 p=0.0044、F1@0.25 p=0.0176），
   而且**没有动任何特征**——只是把 `mstcn` 换成 `mstcn2 h32/s2l5`（38,118 → 67,500 参数，仅 +77%）。
   机制侧对应 mstcn2 的生成+精化 stage / 双膨胀 / 深监督 / **T-MSE 平滑**（后者正是 §14.5(2) 治过分割的那一项）。
2. **edit 仍不显著（p=0.179），但 ② 的价值一半在"降方差"**：8 seed 下 ① 的 edit 摆幅 21.6~44.5（跨 seed 28.9），
   ② 只有 31.7~40.4（跨 seed 8.7）——**这是 edit 中位数提升却过不了配对检验的原因**（配对检验看逐视频差，
   不看跨 seed 摆幅）。对"可复现地拿到某个分数"而言，稳定本身就是收益。
3. **代价与收益**：acc 持平（52.71 → 52.47）、fp 段数中位 93 → 58（−38%）、insert 召回 +17.3pp；
   flush / withdraw 无显著变化，sbc 仍样本不足。
4. **一个方法论教训**：8 seed 的 ① 与 3 seed 的 ① 中位数并不同（edit 33.91 → 34.55、acc 50.63 → 52.71、
   非 idle 599 → 386）——**原 3 seed 恰好落在偏保守的一侧**，所以 §15.2 的"边界"是样本量不足而非效应不存在。
   → 后续任何"A vs B"的架构/特征对照，**≥5 seed 起步**，3 seed 只能用来筛掉明显更差的方案。
5. **回答本轮目标（"有什么提特征/提精度的方法"）**：本轮 5 个特征与损失侧变体全部只换来类级召回（§15.3 第 6 条），
   而**唯一显著的段级改进来自换架构**。结论：**先在 `mstcn2` 这一族里做配置搜索，再回到特征侧**——
   特征侧的边际收益已被第八~十五轮反复证伪。

### 15.6 复现

```bash
# 本轮全部证据（官方汇总 + 逐 (seed,视频)×类配对检验，一条命令可复核；工具见 `tools/compare_runs.py`）
python tools/compare_runs.py \
    --left "runs/capacity-lr0005-e60/h32/mstcn-*" --left "runs/seedext-h32/h32/mstcn-*" --left-label "mstcn h32" \
    --right "runs/mstcn2-cap-s2l5h32/h32/mstcn2-*" --right "runs/seedext-mstcn2s2l5h32/h32/mstcn2-*" \
    --right-label "mstcn2 s2l5 h32"

# A：clip 0.03 + drop3 叠加（drop3 = feature_schema.mask_targets 三组通道）
python tools/run_capacity_matrix.py --runs-dir runs/stack-clip003-drop3 --hidden 32 --seeds 42,7,2026 \
    --epochs 60 --set train.lr=0.0005 --set train.class_weight_clip=0.03,5.0 \
    --set feature_schema.mask_targets=hand,short_brush,brush_tip_out --no-eval-last

# B：mstcn2 s2l5 h32 + clip 0.03（② 为同配置去掉 --set class_weight_clip）
python tools/run_capacity_matrix.py --runs-dir runs/mstcn2-s2l5h32-clip003 --set model.type=mstcn2 \
    --set model.num_stages=2 --set model.num_layers=5 --hidden 32 --seeds 42,7,2026 \
    --epochs 60 --set train.lr=0.0005 --set train.class_weight_clip=0.03,5.0 --no-eval-last

# C：最强模型 + clip 0.2
python tools/run_capacity_matrix.py --runs-dir runs/mstcn2-s4l10h128-clip020 --set model.type=mstcn2 \
    --set model.num_stages=4 --set model.num_layers=10 --hidden 128 --seeds 42,7,2026 \
    --epochs 60 --set train.lr=0.0005 --set train.class_weight_clip=0.2,5.0 --no-eval-last

# D / E：序列去均值（E 再加 clip 0.03）
python tools/run_capacity_matrix.py --runs-dir runs/demean-clip010 --hidden 32 --seeds 42,7,2026 \
    --epochs 60 --set train.lr=0.0005 --set model.sequence_normalization=demean --no-eval-last
python tools/run_capacity_matrix.py --runs-dir runs/demean-clip003 --hidden 32 --seeds 42,7,2026 \
    --epochs 60 --set train.lr=0.0005 --set model.sequence_normalization=demean \
    --set train.class_weight_clip=0.03,5.0 --no-eval-last

# §15.5 的 8 seed 复核：两个架构在同 5 个新 seed 上补齐（保证逐 (seed,视频) 可配对）
python tools/run_capacity_matrix.py --runs-dir runs/seedext-h32 --hidden 32 --seeds 1,2,3,4,5 \
    --epochs 60 --set train.lr=0.0005 --no-eval-last
python tools/run_capacity_matrix.py --runs-dir runs/seedext-mstcn2s2l5h32 --set model.type=mstcn2 \
    --set model.num_stages=2 --set model.num_layers=5 --hidden 32 --seeds 1,2,3,4,5 \
    --epochs 60 --set train.lr=0.0005 --no-eval-last

pytest framework/tests/test_sequence_normalization.py -q
```

---

## 第十六轮：加宽 > 加深、T-MSE 是旋钮不是引擎、后处理与探针口径修正（2026-09-22）

> 承接 §15.5 的两条指示：①把 `mstcn2 h32/s2l5` 当新基线、在其上重估旧结论；②在 `mstcn2` 族里做配置搜索。
> 本轮把**配置网格（深度 vs 宽度）**、**T-MSE 剂量扫描**、**活动量旋钮在新基线上的重估**三件事做完；
> 并在核对"模型到底有没有跑赢线性探针"时发现**已交付的容量报告与图用错了探针口径**
> （把因果 md=5 的数当成离线口径的下界），于是补做了**离线后处理平滑**对照实验——
> 它同时暴露一个"用 test 选后处理参数能造出 22 分假增益"的坑。
>
> **一句话结论**：本轮唯一统计显著的精度杠杆是**加宽 `hidden`**；`num_stages/num_layers`、T-MSE 权重、
> 活动量旋钮都不是；**后处理平滑只是"时序建模"的替代品**，只在模型"话太多"时有用。

### 16.1 配置网格：加宽显著、加深不显著（同配方 lr 5e-4 / 60 轮 / 3 seed 中位数）

| 配置 | 参数量 | edit | F1@0.1 | F1@0.25 | F1@0.5 | acc | tp/fp/fn@0.5 | 段数比 | 非 idle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s2l5 / h32 | 67,500 | 36.00 | 32.79 | 22.64 | 11.11 | 53.73 | 8/63/65 | 0.97 | 484 |
| s4l10 / h32 | 213,784 | 40.59 | 36.21 | 24.14 | 5.94 | 51.95 | 3/34/70 | 0.58 | 567 |
| s4l10 / h64 | 837,144 | 34.19 | 32.84 | 23.88 | 8.96 | 47.90 | 6/46/67 | 0.73 | 695 |
| s2l5 / h128 | 1,007,244 | 46.45 | 41.29 | 33.59 | 14.19 | 53.66 | 11/71/62 | 1.12 | 778 |
| **s4l10 / h128** | **3,312,664** | **51.47** | **44.78** | **35.82** | 12.00 | **55.25** | 9/53/64 | 0.84 | 671 |

**宽度轴（`hidden` 32→128，配置固定）——全部显著：**

| 配对 | 指标 | n | 胜/负 | 中位差 | p |
|---|---|---:|---:|---:|---:|
| s4l10 h128 vs h64 | edit | 19 | 17/2 | **+16.67** | **0.0003** |
| s4l10 h128 vs h64 | F1@0.1 / F1@0.25 | 21 / 20 | 16/5、16/4 | **+9.68 / +13.71** | **0.0051 / 0.0012** |
| s4l10 h128 vs h32 | edit | 20 | 15/5 | **+22.59** | **0.0064** |
| s4l10 h128 vs h32 | F1@0.1 / F1@0.25 | 20 / 21 | 14/6、16/5 | **+10.13 / +9.50** | **0.0187 / 0.0112** |
| s4l10 h128 vs h32 | insert recall | 9 | 8/1 | **+25.56** | **0.0195** |
| s2l5 h128 vs h32 | edit / F1@0.1 | 21 / 22 | 15/6、16/6 | **+18.38 / +7.71** | **0.0142 / 0.0329** |
| s4l10 h64 vs h32 | edit | 18 | 7/11 | −11.11 | 0.6790（**不显著**） |

**深度轴（stage 数 / 层数加倍）——都不显著，且方向对 insert 有害：**

| 配对 | 指标 | n | 胜/负 | 中位差 | p |
|---|---|---:|---:|---:|---:|
| s4l10 vs s2l5 @h32 | edit / F1@0.1 | 16 / 16 | 10/6、9/7 | +7.72 / +2.22 | 0.7173 / 0.9399 |
| s4l10 vs s2l5 @h32 | insert recall | 10 | 3/7 | **−19.60** | 0.1055 |
| s4l10 vs s2l5 @h128 | edit / F1@0.25 | 23 / 23 | 12/11、15/8 | +5.56 / +8.80 | 0.3776 / 0.2068 |
| s4l10 vs s2l5 @h128 | insert recall | 12 | 3/9 | **−11.11** | 0.2334 |

→ **结论（顶替 §15.5 第 2 条的"架构收益"解释）**：`mstcn2` 的收益来自**宽度**而不是 stage/层数；
h64 与 h32 无差别（摆幅内），h128 才显著上一档。**加宽是"提精度"唯一被配对检验确认的动作**，
且 `s2l5 h128`（100 万参）已达 `s4l10 h128`（331 万参）的 90%（edit 46.45 vs 51.47）——**性价比最高的点是 s2l5/h128**。

> 本小节的配对行与 `docs/EXPERIMENT_REPORT_MSTCN_CAPACITY_20260922.md` §2.2 的同名行**同源同口径**
> （同一批 run、同一检验方法），此处重列是为了让特征侧文档自洽，不是第二份独立证据。

### 16.2 T-MSE 剂量扫描：它是"平滑旋钮"，不是 `mstcn2` 优势的来源

`mstcn2` 的 T-MSE 项 = 对逐 stage log-prob 的时间差平方惩罚（`model.tmse_weight`，默认 0.15；
`model.tmse_clip` 默认 4.0）。§15.5 曾猜"T-MSE 就是治过分割的那一项"——本轮消融否证了它。

**h32/s2l5（8 seed 中位数；w=0.05 仅 3 seed）**

| tmse_weight | edit | F1@0.1 | F1@0.25 | F1@0.5 | acc | tp/fp@0.5 | 段数比 | 非 idle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0（消融） | 32.68 | 28.77 | 19.46 | 8.62 | 51.38 | 9/86 | 1.25 | 1040 |
| 0.05 | 26.81 | 25.50 | 17.45 | 6.71 | 53.43 | 5/73 | 1.05 | 479 |
| **0.15（默认）** | 38.36 | 31.36 | 21.03 | 10.04 | 52.47 | 7/58 | 0.89 | 504 |
| 0.30 | 41.41 | 32.48 | 21.92 | 9.90 | 52.17 | 7/59 | 0.90 | 768 |
| 0.60 | 40.18 | **35.03** | **24.35** | **13.45** | **53.28** | 8/**39.5** | **0.65** | 636 |

- vs 消融（w=0）：**w=0.6 edit +8.0pp（30/18，p=0.0237 显著）**；**w=0.3 F1@0.1 +3.0pp（p=0.0336 显著）**；
  默认 w=0.15 反而**不显著**（edit +5.1，p=0.2711）。
- 代价一致：**insert 召回被压低**（默认 vs 消融 −18.8pp，p=0.0148 显著；w=0.6 −19.3pp，p=0.0515）。

**h128/s4l10（3 seed 中位数）**

| tmse_weight | edit | F1@0.1 | F1@0.25 | acc | tp/fp@0.5 | 段数比 | 非 idle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0（消融） | 48.85 | 33.33 | 24.68 | 52.94 | 7/74 | 1.11 | 832 |
| **0.15（默认）** | **51.47** | **44.78** | **35.82** | **55.25** | 9/53 | 0.84 | 671 |
| 0.60 | 39.38 | 37.21 | 30.89 | 54.15 | 6/43 | 0.68 | 746 |

- **在 h128 上消融 T-MSE 是显著掉分**：F1@0.1 −6.9pp（15/5，**p=0.0192**）、F1@0.25 −10.8pp（17/4，**p=0.0082**）、
  edit −8.5pp（p=0.1030）；而**在 h32 上消融只掉 5.1 且不显著**。
- 加倍到 0.6 在 h128 上更差（edit −12.1、F1@0.1 −7.6）→ **默认 0.15 在强配置上是内点最优**。

→ **结论**：T-MSE 是**容量相关**的平滑旋钮——模型越大越依赖它（h128 上不能删、也不能加），
小配置上删掉影响不大、加重可小幅提段级（但显著压 insert/flush 召回）。**默认值不要动**；
§15.5 里"T-MSE 是 mstcn2 优势来源"的猜测**不成立**：h32 上 w=0 与默认的 edit 是 32.68 vs 38.36 的中位差，
但配对 p=0.2711——那点差距是 seed 混合，不是机制。

### 16.3 活动量旋钮在新基线上的重估（`mstcn2 h32/s2l5`，8 seed）

| clip | edit | F1@0.1 | F1@0.25 | acc | tp/fp@0.5 | 段数比 | 非 idle | 配对判读（vs 默认 0.1） |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.03（活动量↑） | 36.39 | 29.37 | 23.26 | 48.37 | 11/98.5 | 1.47 | 1277 | **insert +19.4pp（24/10，p=0.0101）**、withdraw +21.2pp（p=0.1397） |
| **0.1（默认）** | 38.36 | 31.36 | 21.03 | 52.47 | 7/58.5 | 0.89 | 504 | — |
| 0.2（活动量↓） | 40.48 | 31.31 | 21.30 | 54.21 | 6.5/45 | 0.70 | 449 | **withdraw −34.6pp（2/8，p=0.0371）**、insert −10.6pp（p=0.0556） |

→ **行为与 `mstcn` 上完全同型**：一个单调的"召回 ↔ 精度"旋钮，**段级指标没有一个显著变化**
（clip 0.03 的 edit −0.02，p=0.976；F1@0.25 +8.68，p=0.119；clip 0.2 的 edit +3.64，p=0.606）。
**换架构没有改变这个旋钮的性质**——它继续只买召回。§15.3 第 6 条（"活动量这条轴已到边界"）在更强的基线上再次成立。

### 16.4 离线后处理平滑：只是"时序建模"的替代品 + 一个 test 选参的坑

**背景（口径修正的由头）**：`evaluation.smoothing_min_duration` **只实现在 `sliding_window_pipeline`**（因果 GRU）；
离线 `full_sequence_temporal`（`mstcn`/`mstcn2`）的正式评测是**逐帧 argmax、零平滑**。
而 §定版 §B 那组"线性探针 edit 50.63 / F1@0.1 45.26 / F1@0.25 33.58"是在 **`runs/strategy_cmp_p18` 的
`gru-*`（`pipeline=sliding_window_temporal`，window=16）** 上算的，探针因此带着**窗口冷启动 + md=5 迟滞平滑**。
详见 §16.5 的修正。于是本轮补做：**给离线模型/探针补上同类后处理，值多少分？**
（工具 `tools/probe_offline_postprocess.py`，纯后处理、不重训。）

| 对象 | 现状（逐帧 argmax） | 加"最小时长合并"后 | 判读 |
|---|---|---|---|
| 逐帧 LDA 探针（离线） | edit 9.01 / F1@0.1 12.78 / 段数 ~100 | **test 选 d=5 → edit 55.90 / F1@0.1 47.01** | 抖动被修掉，看似"反超模型" |
| 同上，**改在 val 上选 d** | val 最优 d=9 | **test edit 33.33 / F1@0.1 40.00** | **同一探针掉了 22.57 分** |
| 弱模型 `mstcn` h32（8 seed） | edit 31.64 | median k=3 → 33.33（32/13，**p=0.0508**） | 小幅受益（边界） |
| 强模型 `mstcn2` h128/s4l10 | edit 52.78 / 段数 7 | d=5 → 47.22；d=9 → 47.22（**降分**） | 已经干净，平滑只有害 |

**两条结论：**

1. **后处理平滑与"时序建模"是替代关系**：越抖的预测受益越大（探针 ~100 段/视频 → 收益最大），
   越干净的预测受害（强模型 7 段/视频 → 降分）。**模型层已经把平滑该做的事做掉了**——
   这也解释了 §定版 §3"时序模型跑不赢特征 + 后处理"为什么只在因果口径成立：那条下界的强度来自 md=5。
2. **后处理参数必须用 val 选，否则能造出 20 分级别的假增益**：同一个探针、同一套合并算子，
   **在 test 上选 d=5 → 55.90，在 val 上选 d=9 → 33.33**。val 最优与 test 最优不一致
   （val 偏好更强平滑），这正是"用 test 选参"最危险的形态。
   **在正当协议下（val 选参），最强模型 edit 52.78 显著高于探针 33.33（2/17，p=0.0013），
   而 F1@0.1（42.86 vs 40.00，p=0.5565）与 F1@0.25（34.88 vs 35.00，p=0.7089）与探针打平。**

### 16.5 口径修正：已交付报告与图里的"探针下界"用错了协议

**问题**：`docs/EXPERIMENT_REPORT_MSTCN_CAPACITY_20260922.md` §2.4 与
`docs/mstcn-capacity/figures/capacity_vs_segmental.png`（`tools/plot_capacity_curves.py` 里硬编码的
`PROBE_EDIT = 50.63` / `PROBE_F1_025 = 33.58`）把**因果口径**的探针值当成了**离线曲线**的下界，
据此得出的"模型刚刚追平/略超线性探针"**不成立**。

**证据**：`runs/strategy_cmp_p18` 下全部是 `gru-*`、`pipeline=sliding_window_temporal`、`window=16`；
`linear_probe_row` 对非因果配置强制 `window=1 / min_duration=1`，对因果配置才套 md 平滑。
capacity 摘要里**同口径**自动算出的探针行是 **acc 49.15 / edit 17.86 / F1@0.25 5.31**。

**修正后的正确读数**（离线同口径）：最强模型 edit **51.47 vs 17.86**（2.9 倍）、F1@0.25 **35.82 vs 5.31**（6.7 倍）
——**离线口径下模型远超逐帧线性下界**，"时序建模没用"的旧印象是跨协议比较的产物。
（报告 §2.4、§2.1、§3 与 `tools/plot_capacity_curves.py` 已同步修正。）

### 16.6 结论与下一步

1. **唯一显著的精度杠杆是加宽，但只到 h128**：`hidden` 32→128 在 s2l5 与 s4l10 上双双显著
   （edit p=0.0142 / 0.0064），h128 vs h64 更显著（p=0.0003）；**stage/层数加倍不显著且伤 insert**（−11~−20pp）。
   **h256 已在第十七轮 §17.5 结清：两个配置都更差**（s2l5 段级打平但 insert −12.6 p=0.0327；
   s4l10 的 F1@0.1/F1@0.25 显著下降）→ **h128 就是宽度峰值**。
2. **T-MSE 不要动默认值**：强配置上删它显著掉分（F1 p=0.0082~0.0192）、加倍更差；弱配置上加它只买精度、
   压召回。
3. **活动量旋钮在更强架构上性质不变**（只买召回，段级不显著）→ 想提段级不能再调它。
4. **后处理不是免费的精度**：只在模型抖动时有用，且参数必须 val 选；test 选参能骗出 22 分。
5. **下一批实验**：特征契约补充已在第十七轮完成（5 契约 × 2 宽度 + 通道子集探针）；
   剩下的是**标签侧/检测侧**（`sbc`/`flush` 的根因既不在特征也不在损失，见定版 §C）
   与第十七轮 §17.6 提出的 **presence-only 48 维契约**。

### 16.7 复现

```bash
# 任意"两组 run"的官方汇总 + 逐 (seed,视频)×类配对检验（本轮所有判读都用它）
python tools/compare_runs.py --left "<glob>" --left "<glob2>" --left-label A --right "<glob>" --right-label B

# 16.1 配置网格：加宽（h128 vs h64）
python tools/compare_runs.py --left "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" --left-label "s4l10 h128" \
    --right "runs/mstcn2-cap-s4l10h64/*/mstcn2-*" --right-label "s4l10 h64" --metrics edit,f1_01,f1_025,insert
# 16.1 配置网格：加深（s4l10 vs s2l5 @h128）
python tools/compare_runs.py --left "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" --left-label "s4l10 h128" \
    --right "runs/mstcn2-cap-s2l5h128/*/mstcn2-*" --right-label "s2l5 h128" --metrics edit,f1_01,f1_025,insert

# 16.2 T-MSE 剂量：w=0 消融 / w=0.6，都在 8 seed 上补齐后再比
python tools/run_capacity_matrix.py --runs-dir runs/round16-tmse-h32-w000 --set model.type=mstcn2 \
    --set model.num_stages=2 --set model.num_layers=5 --hidden 32 --seeds 42,7,2026 --epochs 60 \
    --set train.lr=0.0005 --set model.tmse_weight=0.0 --no-eval-last --no-probe-baseline
python tools/run_capacity_matrix.py --runs-dir runs/round16-tmse-h32-w00-seedext --set model.type=mstcn2 \
    --set model.num_stages=2 --set model.num_layers=5 --hidden 32 --seeds 1,2,3,4,5 --epochs 60 \
    --set train.lr=0.0005 --set model.tmse_weight=0.0 --no-eval-last --no-probe-baseline
python tools/compare_runs.py --left "runs/round16-tmse-h32-w000/*/mstcn2-*" \
    --left "runs/round16-tmse-h32-w00-seedext/*/mstcn2-*" --left-label "w=0" \
    --right "runs/mstcn2-cap-s2l5h32/h32/mstcn2-*" --right "runs/seedext-mstcn2s2l5h32/h32/mstcn2-*" \
    --right-label "w=0.15" --metrics edit,f1_01,f1_025,insert,flush
# （h128 的消融：把 --hidden 与 --set model.num_stages=4 --set model.num_layers=10 换上去，seeds 42,7,2026）

# 16.3 活动量旋钮在新基线上（8 seed）
python tools/run_capacity_matrix.py --runs-dir runs/round16-clip020 --set model.type=mstcn2 \
    --set model.num_stages=2 --set model.num_layers=5 --hidden 32 --seeds 42,7,2026,1,2,3,4,5 --epochs 60 \
    --set train.lr=0.0005 --set train.class_weight_clip=0.2,5.0 --no-eval-last --no-probe-baseline
python tools/compare_runs.py --left "runs/round16-clip020/*/mstcn2-*" --left-label "clip 0.2" \
    --right "runs/mstcn2-cap-s2l5h32/h32/mstcn2-*" --right "runs/seedext-mstcn2s2l5h32/h32/mstcn2-*" \
    --right-label "默认 clip" --metrics edit,f1_01,f1_025,insert,withdraw,flush

# 16.4 离线后处理平滑（含 val 选参的正当协议 + 探针对照）
python tools/probe_offline_postprocess.py --run "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" \
    --label "mstcn2 s4l10 h128" --median-k 1,3,5 --mindur-d 3,5,9 --probe \
    --compare-run "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" --compare-label "模型" --select-on-val
```

---

## 第十七轮：特征契约补充实验——换特征提取方式能买到什么（2026-09-22）

> 立项缺口：定版 §B 的特征契约排名（roi-144 > bbox-40-global > bbox-40-hand > roi-96-v2 >
> bbox-80-global-hand）**全部是 `gru`（因果、md=5）口径**得出的；扫过全部 run 后发现
> **离线模型只跑过 roi-144 与 bbox-40**，其余三个契约**从未用离线模型评测过**，更没在当前最优
> 架构上评测过。本轮补齐：**5 个契约 × 2 档宽度（mstcn2 s2l5 h128/h32）× 3 seed = 27 run**
> （roi-144 复用已有批次），并新增一个**通道语义 / 区域子集探针**回答"配方内部哪部分在携带信号"。
>
> **一句话结论：在当前最优宽度（h128）上，没有任何替代契约能提高段级指标；roi-144 真正稳赢的是
> `long_brush_insert` 的召回，而这与"insert 信号集中在 presence 通道"的探针结论一致。**

### 17.1 五个契约的同口径对照（mstcn2 s2l5，3 seed；官方 `metrics.summary` 中位）

**h128（当前最优宽度）**

| 契约 | dim | 参数量 | edit | F1@0.1 | F1@0.25 | F1@0.5 | acc | 非 idle | 段数比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **roi-144（基准）** | 144 | 1,007,244 | **46.45** | **41.29** | **33.59** | **14.19** | **53.66** | 778 | 1.12 |
| bbox-40（auto-v3） | 40 | 993,932 | 37.58 | 32.48 | 23.93 | 11.97 | 50.89 | 525 | 0.77 |
| bbox-40-hand | 40 | 993,932 | 51.05 | 33.94 | 25.17 | 9.27 | 46.72 | 626 | 1.10 |
| roi-96-v2（可见性重排） | 96 | 1,001,100 | 36.51 | 36.64 | 24.84 | 9.16 | 53.62 | 370 | 0.79 |
| bbox-80-global-hand | 80 | 999,052 | 43.42 | 31.25 | 21.33 | 8.33 | 51.57 | 545 | 1.05 |

**h32（同一批契约，低宽度）**

| 契约 | dim | 参数量 | edit | F1@0.1 | F1@0.25 | F1@0.5 | acc | 非 idle | 段数比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **roi-144（基准）** | 144 | 67,500 | 36.00 | 32.79 | 22.64 | 11.11 | 53.73 | 484 | 0.97 |
| bbox-40（auto-v3） | 40 | 64,172 | 32.94 | 32.56 | 21.71 | 12.61 | 52.44 | 406 | 0.68 |
| bbox-40-hand | 40 | 64,172 | 41.23 | 29.73 | 21.62 | 9.16 | 47.18 | 991 | 1.03 |
| roi-96-v2 | 96 | 65,964 | 32.28 | 37.14 | 26.98 | 10.77 | 52.63 | 847 | 0.78 |
| bbox-80-global-hand | 80 | 65,452 | 32.70 | 29.31 | 18.97 | 6.94 | 52.67 | 318 | 0.79 |

> 参数量跨契约几乎相同（h128 上 99.3 万 ~ 100.7 万，差 ≤1.3%）：输入维度只影响首末两个 1×1 投影，
> 所以这组比较**不受模型大小混淆**，测的就是特征契约本身。

### 17.2 配对检验：段级基本打平（除一个更差），**insert 召回是 roi-144 的稳健优势**

差值 = 该契约 − roi-144（负数 = 比 roi-144 差）；逐 (seed, 视频) × 类 Wilcoxon。

| 宽度 | 契约 | edit | F1@0.1 | F1@0.25 | **insert recall** |
|---|---|---|---|---|---|
| h128 | bbox-40 | −14.29（p=0.1625） | **−10.00（5/15，p=0.0032）** | −11.67（p=0.1540） | −11.93（3/10，p=0.0942） |
| h128 | bbox-40-hand | +8.33（p=0.6262） | +1.59（p=0.5235） | −0.12（p=0.4851） | **−28.15（1/13，p=0.0006）** |
| h128 | roi-96-v2 | **−18.80（6/14，p=0.0177）** | **−5.00（6/15，p=0.0263）** | −10.26（p=0.3205） | **−32.43（0/13，p=0.0002）** |
| h128 | bbox-80-global-hand | −10.07（p=0.2428） | −5.56（p=0.3155） | −4.79（p=0.7332） | −17.20（3/11，p=0.0676） |
| h32 | bbox-40 | +1.16（p=0.5627） | +4.76（p=0.3546） | −6.61（p=0.7112） | **−22.95（1/9，p=0.0352）** |
| h32 | bbox-40-hand | +11.11（p=0.2772） | +5.95（p=0.7841） | +2.12（p=0.6215） | **−22.95（2/8，p=0.0273）** |
| h32 | roi-96-v2 | −6.27（p=0.5277） | +4.76（p=0.8900） | −2.08（p=0.9632） | +6.30（7/6，p=0.8926） |
| h32 | bbox-80-global-hand | −6.38（p=0.7938） | −1.79（p=0.4330） | −8.57（p=0.2121） | **−22.95（2/8，p=0.0488）** |

读数：

1. **h128 上没有任何替代契约提高段级指标**；`roi-96-v2`（可见性重排）反而**显著更差**
   （edit −18.80 p=0.0177、F1@0.1 −5.00 p=0.0263）——它把维度预算从低频类挪给高频类的做法，
   在离线强模型上净亏。
2. **`long_brush_insert` 召回是 roi-144 唯一稳健的、跨宽度跨契约的优势**：8 个对比里 6 个显著更差
   （p=0.0002 ~ 0.0488），最极端的 `roi-96-v2 @h128` 是 **0/13 全负**。唯一例外是 `roi-96-v2 @h32`
   （+6.30，p=0.8926，不显著）。
3. **"段级中位数更高"会误导**：`bbox-40-hand` 的 edit 中位在 h128（51.05）与 h32（41.23）都**高于**
   roi-144（46.45 / 36.00），但配对检验都不显著（p=0.63 / 0.28），而它的 **acc 只有 46.72 / 47.18**
   （比 roi-144 低 7pp）且 insert 召回显著更差。**只看 edit 会得出"手部契约更好"的错误结论**——
   这是本仓库"必须同时报 acc + 逐类召回"规则的第 N 次生效。
4. **契约差异是容量相关的，但方向和 bbox-40 那组相反**：h32 上 bbox-40 与 roi-144 段级打平
   （+1.16 p=0.56），h128 上 roi-144 显著更好（F1@0.1 −10.00 p=0.0032）；
   即**增大宽度会放大"特征契约对不对"的差别**（与报告 §4.5"特征/容量互相抵消"并不矛盾——
   抵消的是"是否坍缩"，不是"契约质量"）。

### 17.3 复核报告 §4.5：h32 上的 "+14.29" 里混着"全 idle 坍缩"

用同口径重跑 `mstcn` 网格上的 40 vs 144 对照（`runs/mstcn-40d-cap` vs `runs/capacity-lr0005-e60`）：

| 宽度 | roi-144 edit | bbox-40 edit | edit 配对 | bbox-40 非 idle 帧 | bbox-40 acc |
|---|---:|---:|---|---:|---:|
| h32 | 33.91 | 26.14 | **+14.29（17/7，p=0.0199）** | **14**（真值 1181） | 55.17（≈全 idle 基线 55.25） |
| h128 | 41.79 | 22.35 | +11.16（13/7，p=0.0522） | 241 | 54.04 |
| h256 | 43.27 | 35.24 | +2.99（11/10，p=0.1491） | 402 | 50.78 |

→ **`mstcn` h32 上的 bbox-40 是"坍缩成全 idle"**（只输出 14 帧非 idle），roi-144 只是"没坍缩"；
  随着容量升高 bbox-40 的坍缩缓解（14 → 241 → 402），edit 差距同步收窄（+14.29 → +11.16 → +2.99）。
  换成**抑制坍缩的 `mstcn2`** 后，h32 上两者段级直接打平（+1.16，p=0.5627）。
  但**roi-144 的 insert 优势在两端都留着**：h256 上 +30.04pp（9/1，p=0.0059）、
  mstcn2 h32 上 +22.95pp（1/9 反向，p=0.0352）。
  **正确口径：特征契约买到的是"不坍缩 + 稀类召回"，不是"段级质量"。**

### 17.4 通道语义 / 区域子集探针（新工具 `tools/probe_channel_subsets.py`）

`roi-144` 的每类 18 维 = 6 区域 × `[presence, count, max_area]`。用逐帧 LDA（train 拟合、
test 上离线逐帧 argmax、同一套 `temporal_metrics`）做子集消融：

| 子集 | 维数 | edit | F1@0.1 | F1@0.25 | 配对（vs 全 144，逐视频） |
|---|---:|---:|---:|---:|---|
| all | 144 | 9.01 | 12.78 | 5.89 | —（基准） |
| channel=presence | 48 | 14.60 | 17.79 | 9.21 | +5.78（7/1，p=0.1953） |
| channel=count | 48 | 13.03 | 20.00 | 8.26 | +4.89（5/3，p=0.3828） |
| channel=max_area | 48 | 11.49 | 13.18 | 7.02 | +0.79（6/2，p=0.7422） |
| channel=presence+count | 96 | 9.58 | 11.88 | 7.12 | −0.16（3/5，p=0.4609） |
| region=1（单个网格区域） | 24 | 6.78 | 5.64 | 1.85 | −0.78（1/6，**p=0.0469**） |

逐类帧级 recall（%，同一 LDA 家族）：

| 类 | all | presence | count | max_area | presence+count |
|---|---:|---:|---:|---:|---:|
| flush | 21.6 | 21.6 | 21.6 | 21.6 | 21.6 |
| **long_brush_insert** | 41.3 | **68.8** | 65.1 | 57.6 | 53.3 |
| **short_brush_cleaning** | 16.7 | **73.0** | 64.4 | 19.7 | 64.4 |
| long_brush_withdraw | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| idle | 62.7 | 45.4 | 47.4 | 52.8 | 54.8 |

**通道语义冗余度**（逐 (检测类, 区域) 的 |Pearson r|，train split，32 个有效对照）：
`presence~count` 均值 **0.990**、`presence~max_area` **0.926**、`count~max_area` **0.920**。

读数：

1. **三类通道几乎在看同一件事**（|r| 0.92~0.99）→ 单通道 48 维子集在 LDA 下**不劣于** 144 维
   （presence 子集 edit +5.78，虽 p=0.195 未过显著，但方向一致且维数省 2/3）。
   **注意这不能直接推广到训练模型**：容量研究已证明 40→144 对训练模型有增益（§17.3），
   LDA 的"少即是多"来自协方差估计噪声与共线性，两者机制不同。
2. **`insert` 与 `sbc` 的信号集中在 presence 通道**（insert 41.3 → 68.8、sbc 16.7 → **73.0**），
   而 `flush` 对通道选择完全不敏感（恒为 21.6）、`withdraw` 在任何子集上都是 0。
   这与 §17.2 的"roi-144 的 insert 优势"**机制自洽**：roi-144 真正独有的是
   **8 类 × 6 区域的 presence 平面**（48 维），而 bbox 系契约把它压成了帧级
   `[presence, cx, cy, w, h]`（每类 5 维、无区域）。
3. **单区域子集**最多只能拿到 24 维的局部视图，最强的 region=1 也只有 edit 6.78 且显著差于全量
   → 空间分区必须**整组保留**才有用，砍区域等于砍掉大部分信息。

### 17.5 宽度轴封顶复核：**h128 就是峰值，h256 反而更差**（§16.6-5 的悬留项结清）

§16.1 证明"加宽是唯一显著杠杆"（h128 vs h64 p=0.0003），但当时没测 h256。本轮补齐两个配置的 h256
（各 3 seed，同配方），结论是**宽度收益在 h128 封顶，再往上只有害**：

| 配置 | 参数量 | edit | F1@0.1 | F1@0.25 | F1@0.5 | acc | fp@0.5 | 非 idle | 段数比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s2l5 / **h128** | 1,007,244 | **46.45** | **41.29** | **33.59** | **14.19** | 53.66 | 71 | 778 | 1.12 |
| s2l5 / h256 | 3,980,556 | 46.43 | 37.50 | 25.00 | 9.58 | 55.40 | 52 | 455 | 0.75 |
| s4l10 / **h128** | 3,312,664 | **51.47** | **44.78** | **35.82** | 12.00 | 55.25 | 53 | 671 | 0.84 |
| s4l10 / h256 | 13,178,904 | 42.65 | 35.90 | 25.64 | 12.21 | 54.72 | 50 | 592 | 0.79 |

配对检验（h256 − h128，逐 (seed,视频) × 类 Wilcoxon）：

| 配置 | edit | F1@0.1 | F1@0.25 | insert recall |
|---|---|---|---|---|
| s2l5 h256 vs h128 | −5.56（p=0.9187） | −3.01（p=0.3860） | −7.14（6/15，p=0.0581） | **−12.61（2/11，p=0.0327）** |
| s4l10 h256 vs h128 | −10.26（6/10，p=0.0623） | **−5.14（5/14，p=0.0401）** | **−12.16（3/17，p=0.0025）** | −5.61（1/8，p=0.0742） |

读数：

1. **`s2l5` 的宽度在 h128 完全饱和**：edit 46.43 vs 46.45（p=0.9187），但 F1@0.25 −7.14（p=0.0581）
   与 **insert −12.61（p=0.0327 显著）**——多出来的宽度没换来任何段级收益，只让模型更保守
   （非 idle 778 → 455、fp 71 → 52、acc 53.66 → 55.40，即向"全 idle"靠）。
2. **`s4l10` 的 h256 是净亏损**：F1@0.1（p=0.0401）、F1@0.25（p=0.0025）双双显著下降，
   edit 也 −10.26（p=0.0623）——**1283 万参数比 331 万参数更差**。
3. 机制上与 §16.3 的"活动量旋钮"同型：**再多的宽度只是把模型推向保守/欠分割**（段数比 0.84 → 0.79、
   非 idle 671 → 592），而不是改善段的检出或边界（见 §24：`edit` 本身不度量时长/边界）。
4. **可执行结论**：`mstcn2 s2l5 / h128`（100 万参）就是这条轴上的性价比峰值；
   `s4l10 / h128`（331 万参）仍是绝对最优（edit 51.47），但**不要再往 h256 以上加宽**。

### 17.6 结论与下一步

1. **特征契约在 h128 上的定论**：roi-144 仍是最优；`roi-96-v2` 显著更差；
   bbox 系（40/80/40-hand）段级打平但**insert 召回显著更差**。→ **不要再找"更好的契约"**，
   要找的是"**更省的同效契约**"。
2. **~~字段级别的下一步~~（已在第十八轮实施并被否证）**：把 `roi-144` 缩成 **`presence-only 48 维`**
   的提案**不成立**——契约已完整落地（recipe + catalog + 门禁 + 单测），但训练后
   **两档宽度都比 roi-144 差**（h128 F1@0.1 −7.04 p=0.0329；h32 F1@0.25 −7.69 p=0.0448，详见 §18）。
   教训：**线性探针的"少即是多"来自 LDA 的协方差估计噪声，不能外推到有正则的训练模型**；
   且省维度的实际收益只有 −1.2% 参数（前处理成本本来就在同一次 bbox 遍历里）。
3. **不要再试的方向**：继续造新契约（§17 已把 5 个契约在最优架构上排完、§18 又否证了缩维契约）、
   可见性重排类改动（v2 在强模型上显著更差）。
4. **召回侧**：insert 的瓶颈已定位到"区域 presence 平面的颗粒度"，sbc 则是**检测覆盖**
   （§13.1：定义物在 test 批次检出 0%）——后者只能靠检测侧解决。

### 17.7 复现

```bash
# 17.1/17.2 契约对照（每档宽度、每个契约各 3 seed；roi-144 复用已有批次）
python tools/run_capacity_matrix.py --runs-dir runs/round17-feat-h128-roi2 \
    --config mstcn-actionmixed-auto-roi.yaml \
    --set data.dataset_ref=temporal.actionmixed-auto-roi-v2 \
    --set feature_schema.version=actionmixed-roi-grid-v2 --set feature_schema.dim=96 \
    --set model.input_dim=96 --set model.type=mstcn2 --set model.num_stages=2 --set model.num_layers=5 \
    --hidden 128 --seeds 42,7,2026 --epochs 60 --set train.lr=0.0005 --no-eval-last --no-probe-baseline
# 其余契约同形：v3=actionmixed-bbox-8cls-v1/40、hand=actionmixed-bbox-hand-8cls-v1/40、
# gh=actionmixed-bbox-global-hand-8cls-v1/80（dim 同步改 feature_schema.dim 与 model.input_dim）
python tools/compare_runs.py --left "runs/round17-feat-h128-roi2/*/mstcn2-*" --left-label "roi-96-v2" \
    --right "runs/mstcn2-cap-s2l5h128/*/mstcn2-*" --right-label "roi-144" \
    --metrics edit,f1_01,f1_025,insert,withdraw,flush

# 17.3 报告 §4.5 的复核（mstcn 网格上的 40 vs 144）
python tools/compare_runs.py --left "runs/capacity-lr0005-e60/h32/mstcn-*" --left-label "roi-144" \
    --right "runs/mstcn-40d-cap/h32/mstcn-*" --right-label "bbox-40" --metrics edit,f1_01,insert

# 17.4 通道语义 / 区域子集探针
python tools/probe_channel_subsets.py --config-run "runs/mstcn2-cap-s2l5h128/*/mstcn2-*" \
    --json tmp/channel_subsets.json
pytest tests/test_probe_channel_subsets.py -q
```

---

## 第十八轮：presence-only 48 维契约（按 §17.6 提案实施）——**训练结果否证了探针的承诺**（2026-09-22）

> §17.6 第 2 条提出：把 `roi-144` 压成 **presence-only 48 维**（8 类 × 6 区域的 presence 平面），
> 依据是 §17.4 的探针结论（三通道 |r| = 0.990/0.926/0.920 高度冗余；presence 子集的 LDA edit
> 14.60 **高于**全量 144 维的 9.01）。本轮按仓库约定把该契约**完整落地并训练验证**：
> 新 mapping 常量 + recipe + catalog 登记 + 门禁 + 单测，`mstcn2 s2l5` h128/h32 各 3 seed。
>
> **结果：presence-48 在两档宽度上都比 roi-144 差**（h128 的 F1@0.1 −7.04，p=0.0329；
> h32 的 F1@0.25 −7.69，p=0.0448），**提案被否证**——"线性探针里更省 = 训练模型里更好"不成立。

### 18.1 实施内容（契约落地清单）

| 环节 | 改动 |
|---|---|
| recipe | `features/roi_bbox.py` 新增 `ROI_PRESENCE_VERSION` / `ROI_PRESENCE_DIM` / `build_roi_presence_frame_features`（实现为 roi-v1 的通道 0 切片，语义完全一致） |
| 导出与分发 | `features/__init__.py` 导出；`temporal/data.py` 新增 `roi_presence_recipe` 分支 |
| catalog | `framework/testsets.yaml` 新增 `temporal.actionmixed-auto-roi-presence-v1`（`input_dim: 48`、`feature_layout: {rows: 2, cols: 3, channels: 1}`）+ train/val/test 三条登记（复用同一批 manifest） |
| 门禁 | `python tools/validate_testsets.py --catalog framework/testsets.yaml` → **exit 0**，三条登记全 `[OK]` |
| 单测 | `framework/tests/test_roi_features_presence.py`（6 项，含**核心不变式**：presence 契约逐帧 == roi-v1 的通道 0 切片） |
| 训练 | `mstcn2 s2l5` × {h128, h32} × 3 seed（复用 `mstcn-actionmixed-auto-roi.yaml` + `-S` 覆盖） |

> 顺带修掉一个真缺陷：`tools/probe_channel_subsets.py` 此前**从未真正读 layout**（一直用代码默认的
> 2×3×3，只因 144 恰能被 18 整除才没暴露）；现改为**从 catalog 取 `feature_layout`**（单一事实源）。
> 修后回归自检：新契约的 `all` 子集指标（edit 14.60 / F1@0.1 17.79 / F1@0.25 9.21）与 roi-144 的
> `channel=presence` 子集**完全同值**，证明切片实现正确。

### 18.2 训练对照（mstcn2 s2l5，3 seed；官方 `metrics.summary` 中位）

| 配置 | 参数量 | edit | F1@0.1 | F1@0.25 | F1@0.5 | acc | tp/fp/fn@0.5 | 非 idle | 段数比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **roi-144 h128** | 1,007,244 | **46.45** | **41.29** | **33.59** | **14.19** | 53.66 | 11/71/62 | 778 | 1.12 |
| presence-48 h128 | 994,956 | 40.62 | 31.33 | 22.22 | 8.89 | 53.62 | 8/84/65 | 604 | 1.27 |
| **roi-144 h32** | 67,500 | **36.00** | **32.79** | **22.64** | **11.11** | 53.73 | 8/63/65 | 484 | 0.97 |
| presence-48 h32 | 64,428 | 32.93 | 31.72 | 19.31 | 5.52 | 54.04 | 4/68/69 | 515 | 0.99 |

配对检验（差值 = presence-48 − roi-144，逐 (seed,视频) × 类 Wilcoxon）：

| 宽度 | edit | F1@0.1 | F1@0.25 | F1@0.5 缺失部分 | insert recall |
|---|---|---|---|---|---|
| h128 | −12.91（7/15，p=0.1396） | **−7.04（6/16，p=0.0329）** | −9.97（7/13，p=0.4091） | fp 71→84、段数比 1.12→1.27 | −6.66（6/8，p=0.9032） |
| h32 | +0.74（9/8，p=0.8536） | −6.35（6/13，p=0.0509） | **−7.69（5/12，p=0.0448）** | tp 8→4、F1@0.5 11.11→5.52 | −5.59（6/7，p=0.3757） |

读数：

1. **两档宽度、五个段级指标方向一致为负**（其中两个显著、三个边界），**没有一档打平或更好**。
   recall 类指标（insert/withdraw）差异不显著，说明**丢掉的不是"检不检得到"，而是"段边界/精度"**
   （h128 上 fp 71→84、段数比 1.12→1.27；h32 上 tp 8→4、F1@0.5 11.11→5.52）——
   即 **`count` / `max_area` 提供的是"强度/量级"信息，presence（0/1）表达不了**。
2. **探针与训练模型的结论方向相反，这不是第一次**（第十四轮 drop3 的 flush 增益、第十五轮 demean 的
   flush 预期都同型）。机制上可解释：LDA 是**闭式无正则**估计，144 维里高度相关的通道会放大协方差
   估计噪声 → "少即是多"；而训练模型有正则与 z-score 归一化，**冗余通道仍能提供非线性交互与量级线索**。
   → **探针只用于"否掉明显没信号的提案"（AUC≈0.5 那一类），不能用来"立项更省的特征"**。
3. **省维度的收益本身也很小**：输入投影是 1×1 卷积，144→48 只省 96×hidden 参数
   （h128 上 1,007,244 → 994,956，**−1.2%**）；真正的成本在前处理侧（算 count / max_area），
   而那部分已在同一次 bbox 遍历里完成。**这条"省维度"路线本就收益有限**。
4. 契约本身**保留**（已登记、有单测、门禁通过）：它是"通道语义"这条轴上的一个已测点，
   后续若要检验"某通道是否冗余"可直接复用该 recipe 的实现方式（切片式契约）。

### 18.3 结论与下一步

1. **特征契约这条轴的定论收敛**：`roi-144`（z-score）仍是最优且**不要再改**——`roi-96-v2` 显著更差（§17.2）、
   bbox 系丢 insert 召回（§17.2）、`presence-48` 显著更差（本轮）。**"更好的契约/更省的契约"两个方向都已探尽。**
2. **§17.6 第 2 条提案正式作废**（探针依据不足）；§17.4 的通道冗余度结论仍然成立，但**只适用于线性探针口径**。
3. **特征侧的下一个（也是最后一个）方向不在"造契约"**，而在**上游检测覆盖**：
   §13.1/§14.4 已定位 `sbc` 的定义物（`short_brush`）在 test 批次检出率 0.0%——
   这是特征提取的**输入源**问题，只能在检测侧解决（逐类降阈值 / 换权重 / 补帧）。
4. 若还要在离线模型侧提精度，剩余手段只有**标签时间轴审计**（段边界精度的上限由标注决定；
   注意 `edit` 并不度量边界，见 §24）。

### 18.4 复现

```bash
# 门禁与单测
python tools/validate_testsets.py --catalog framework/testsets.yaml
pytest framework/tests/test_roi_features_presence.py -q          # 6 passed

# presence-48 契约训练（复用 roi 配置 + 覆盖 4 个字段）
python tools/run_capacity_matrix.py --runs-dir runs/round18-presence-h128 \
    --config mstcn-actionmixed-auto-roi.yaml \
    --set data.dataset_ref=temporal.actionmixed-auto-roi-presence-v1 \
    --set feature_schema.version=actionmixed-roi-grid-presence-v1 \
    --set feature_schema.dim=48 --set model.input_dim=48 \
    --set model.type=mstcn2 --set model.num_stages=2 --set model.num_layers=5 \
    --hidden 128 --seeds 42,7,2026 --epochs 60 --set train.lr=0.0005 \
    --no-eval-last --no-probe-baseline
# （h32 同形，改 --runs-dir 与 --hidden）

# 对照
python tools/compare_runs.py --left "runs/round18-presence-h128/*/mstcn2-*" --left-label "presence-48" \
    --right "runs/mstcn2-cap-s2l5h128/*/mstcn2-*" --right-label "roi-144" \
    --metrics edit,f1_01,f1_025,insert,withdraw

# 探针（现在从 catalog 读 layout）
python tools/probe_channel_subsets.py --config-run "runs/round18-presence-h128/*/mstcn2-*"
```

---

## 第十九轮：段级误差分解——"漏掉的段"里 52% 是标签认错，不是边界歪（2026-09-22）

> 承接 §18.3：特征契约这条轴已探尽，剩下的两个方向是**上游检测覆盖**与**标签时间轴**。
> 本轮先把这两条"上游"用数据定性（其中一条被数据可用性直接堵死），再新增一个**段级误差分解探针**，
> 回答一个此前只能猜的问题：段级 F1 上不去，**是类别认错了，还是边界框歪了**？
>
> **结论先行**：在当前最强模型上，**51.6% 的真值段连标签都认错**（flush 20/21、withdraw 25/27、
> sbc 18/18 全错）；剩下的是边界错——**标签对但边界偏移中位 22（首）/ 25.5（尾）帧**。
> 容量从 h32 涨到 331 万参，**只把边界偏移从 32/36 帧改到 22/25.5 帧，类别识别率几乎没动**
> （标签错 49.1% → 51.6%，IoU@0.5 匹配率 11.0% → 11.9%）。

### 19.1 两条"上游"方向的可用性核查（一条被数据堵死）

| 方向 | 核查 | 结论 |
|---|---|---|
| 检测侧扩召回（降阈值 / 换权重 / 补帧） | `datasets/cleansight-ActionMixed-auto-lhh/` 下 **`frames/` 只有 `.txt` bbox 文件、没有任何图像**（`ls frames/test/ \| sed 's/.*\.//' \| sort -u` → 仅 `txt`） | **当前数据上不可能做**：特征提取的输入源是"已固化的检测输出"，没有像素就无法重跑检测。要走这条路必须先补图像源（与 §定版 F-3、`IMAGE_FEATURE_TRAINING.md` §5.1 记录的阻塞一致） |
| 标签时间轴审计 | 逐 split 统计真值段时长（`labels/<split>/`，见下方命令） | **没有"test 标签碎片化"的域移**：段长中位 train 21 / val 22 / **test 26** 帧，≤5 帧的短段占比 10% / 11% / 11%。→ 不能把测试难度归因于"标注更碎"；真正的短段是 **sbc（各 split 中位仅 6~10 帧）** |

### 19.2 新工具：`tools/probe_boundary_error.py`（段级误差分解）

**动机**：正式评测里的 `boundary_mae` **只在已匹配（tp）的段上统计**——那是被 IoU 阈值筛过的
有偏子集（§18 实测：③ 在 IoU@0.5 上 boundary_mae=13.4 帧，而无偏统计是 22~25 帧，差近一倍）。
本探针在**全部真值段**上做无偏分解：

1. 每个真值段（同标签极大连续帧段）内取**多数预测标签**：等于真值 → 「标签对」，否则 → 「标签错」
   （标签错的段**无论边界多准都不可能匹配**，这是分解成立的前提）；
2. 标签对的段里取同类预测段中重叠最大者，算 IoU 与首尾偏移；
3. 于是 `fn@IoU = 标签错的段 + 标签对但 IoU 不达标的段`（**边界错**）。

口径：读 `artifacts/*.predictions.json`（与正式评测同源），帧单位同采样序列；纯分析、不训练。

### 19.3 结果：弱基线 vs 最强模型的分解对照

| 指标 | ① `mstcn` h32（8 seed，584 段） | ③ `mstcn2` s4l10 h128（3 seed，219 段） |
|---|---:|---:|
| 真值段总数 | 584 | 219 |
| **标签错（段内多数预测 ≠ 真值）** | 287（**49.1%**） | 113（**51.6%**） |
| IoU@0.25：匹配 / 标签对但边界不达标 / fn | 127 / 170 / 457 | 68 / 38 / 151 |
| IoU@0.50：匹配 / 标签对但边界不达标 / fn | 64 / 233 / **520** | 26 / 80 / **193** |
| **IoU@0.50 匹配率** | 11.0% | 11.9% |
| 标签对段的边界偏移中位（首 / 尾，帧） | 32 / 36 | **22 / 25.5** |
| 真值段时长中位（p25 / p75） | 26（13 / 52） | 26（13 / 52） |
| 匹配段中 `idle` 占比 | 59/64（92%） | 21/26（81%） |

逐类（③，真值段 / 标签错 / IoU@0.5 匹配 / 段长中位）：

| 类 | 真值段 | 标签错 | 匹配@0.5 | 段长中位 |
|---|---:|---:|---:|---:|
| idle | 120 | 27 | 21 | 24 |
| flush | 21 | **20** | 0 | 22 |
| long_brush_insert | 33 | 23 | 3 | 61 |
| long_brush_withdraw | 27 | **25** | 2 | 27 |
| short_brush_cleaning | 18 | **18** | 0 | 6.5 |

读数：

1. **段级失败的主因是类别识别，不是边界定位**：IoU@0.5 的 193 个漏检里 **113 个（59%）根本没认出类别**。
   容量/架构把边界偏移改善了 30%（32→22 帧），但**标签错比例几乎不变**（49.1% → 51.6%）、
   匹配率也只从 11.0% 爬到 11.9%。
2. **匹配段几乎全是 `idle`**（92% / 81%）：非 idle 段的段级匹配基本全丢——这与"只有段标签序列
   （`edit`）会涨、`F1@0.25` 各配置都不显著"（容量报告 §4.3，措辞已于 §24 修正）在机制上一致。
3. **`flush` / `withdraw` / `sbc` 是"标签错"重灾区**（20/21、25/27、18/18），而 `insert` 的段最长
   （中位 61 帧）却仍有 23/33 标签错 → 说明这几个类的**帧级线索本身不可用**
   （§定版 C：flush 是标签/视觉口径问题、insert/withdraw 是方向信息不存在、sbc 是检测覆盖为零）。
4. **修正一个会误导的指标用法**：`boundary_mae` 只能用来比较"已匹配段之间的对齐质量"，
   **不能**当作"模型的边界误差"来外推上限（无偏值大近一倍）。
5. 时长核查（§19.1）说明**难度不是标注碎片化**带来的：test 段长中位甚至比 train 长；
   真正短的是 sbc（6 帧 ≈ 0.8 s）——它在任何 split 都短，且检测覆盖为零。

### 19.4 结论与下一步

1. **离线模型侧的精度天花板已经清楚**：要动段级 F1，必须先解决"51.6% 的段标签认错"，
   而这几个类（flush / withdraw / sbc）的帧级线索已被第八~十八轮反复证伪——
   **继续在模型/特征/损失上调参不会改善它**。
2. **可做的只剩三件（按可行性排序）**：
   (a) **补图像源后重跑检测**（唯一能救 sbc 的路径，但依赖数据采集/下载，本轮已用证据确认阻塞点）；
   (b) **标签审计**：把 flush / withdraw 的时间轴与可见动作逐段核对（零算力，但要人工看视频）；
   (c) **接受现状并优化口径**：既然匹配段 92% 是 idle，段级指标对业务价值有限，
   应把主指标换成**逐类帧级召回 + 每类 precision**（评测口径已在 `docs/EVAL.md` 注册）。
3. **不再立项的方向**：新特征契约（§17/§18 已穷尽）、活动量旋钮（§16.3）、
   后处理平滑（§16.4）、更大宽度（§17.5 h256 是净亏）。

### 19.5 复现

```bash
# 19.3 段级误差分解（弱基线 / 最强模型）
python tools/probe_boundary_error.py --run "runs/capacity-lr0005-e60/h32/mstcn-*" \
    --run "runs/seedext-h32/h32/mstcn-*" --label "mstcn h32 基线（8 seed）" --json tmp/boundary-h32.json
python tools/probe_boundary_error.py --run "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" \
    --label "mstcn2 s4l10 h128（最强）" --json tmp/boundary-s4l10.json
pytest tests/test_probe_boundary_error.py -q          # 7 passed

# 19.1 真值段时长逐 split（标签审计的第一项）
python - <<'PY'
import statistics, collections
from pathlib import Path
import numpy as np
ROOT = Path('datasets/cleansight-ActionMixed-auto-lhh')
NAMES = ['idle','water_injection','flush','long_brush_insert','long_brush_withdraw','short_brush_cleaning']
for split in ('train','val','test'):
    lens=[]; per=collections.defaultdict(list)
    for f in sorted((ROOT/'labels'/split).glob('*.txt')):
        seq=[int(l.split()[1]) for l in f.read_text().splitlines() if len(l.split())==2]
        s=np.array(seq); start=0
        for i in range(1,len(s)+1):
            if i==len(s) or s[i]!=s[start]:
                lens.append(i-start); per[NAMES[s[start]]].append(i-start); start=i
    print(f"{split}: 段数={len(lens)} 中位={statistics.median(lens):.0f} "
          f"≤5帧={sum(1 for l in lens if l<=5)/len(lens)*100:.0f}%  " +
          " ".join(f"{k}={statistics.median(v):.0f}" for k,v in sorted(per.items())))
PY

# 19.1 检测侧可用性（无图像 → 无法重跑检测）
ls datasets/cleansight-ActionMixed-auto-lhh/frames/test/ | sed 's/.*\.//' | sort -u   # → txt
```

---

## 第二十轮：训练期目标随机遮罩（`augmentation.target_mask`）——首个被真正启用的增强，不奏效（2026-09-23）

> 立项：把全部 run 扫一遍后发现 **`augmentation.target_mask` 从未被启用过**
> （`gru-actionmixed*.yaml` 里有该块但 `enabled: false`，没有任何 run 的 `config.resolved.json` 带它）。
> 而本仓库最顽固的病灶正是**批次特定线索记忆**（第十三轮 §13.1：flush 依赖的 `hand.left-top`
> 在 train 帧出现率 0.448、test 帧 **0.000**；per-class train→test AUC 全面塌陷）。
> **静态删特征**（`feature_schema.mask_targets`，第十四轮）只换来 flush 召回 +15.7pp（p=0.031），
> 但**随机遮罩是正则而非删特征**——同一类在别的帧仍可见，模型只能学会"不依赖单帧的单一检测类"。
> 这是训练侧最后一个未测杠杆，本轮把它测掉。
>
> **结论：两个臂都比基线差**（`hand` p0.5：edit −10.56、F1@0.25 −10.79；`all` p0.5：insert precision
> −21.46 p=0.0398、acc 53.66→46.00）。**遮罩去掉的是真信号，不是噪声。**

### 20.1 两个臂与实现

| 臂 | 配置 | 遮罩目标 | probability |
|---|---|---|---|
| aug:hand | `framework/experiments/mstcn2-auto-roi-aug-hand.yaml` | `hand`（train→test 出现率漂移最大的类） | 0.5 |
| aug:all | `framework/experiments/mstcn2-auto-roi-aug-all.yaml` | 全部 8 个检测类 | 0.5 |

实现路径：`--set` 只能寻址两层（`augmentation.target_mask.probability` 不可达），因此按仓库惯例
**写成实验配置**。语义已直接验证（`apply_target_mask_augmentation`）：p=0.5 下 `hand` 块 100 帧中
**53 帧清零、其它类块原样、同 seed 可复现、`enabled:false` 直通**；两条流水线都会调用它
（`full_sequence_pipeline.py:223` / `sliding_window_pipeline.py:195`），只作用于 train。
模型/配方与基线一致：`mstcn2 s2l5 h128`、lr 5e-4、60 轮、3 seed。

### 20.2 结果（3 seed；官方 `metrics.summary` 中位，差值 = 该臂 − 基线）

| 配置 | acc | edit | F1@0.1 | F1@0.25 | F1@0.5 | tp/fp/fn@0.5 | 段数比 | 非 idle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **基线（无增强）** | 53.66 | **46.45** | **41.29** | **33.59** | **14.19** | 11/71/62 | 1.12 | 778 |
| aug:hand p0.5 | 54.15 | 42.34 | 38.17 | 22.06 | 8.19 | 7/56/66 | 0.86 | 702 |
| aug:all p0.5 | 46.00 | 40.71 | 36.81 | 23.38 | 11.04 | 9/81/64 | 1.23 | 849 |

| 配对（逐 (seed,视频)×类 Wilcoxon） | edit | F1@0.1 | F1@0.25 | insert | insert precision |
|---|---|---|---|---|---|
| aug:hand p0.5 | −10.56（7/13，p=0.1893） | −8.57（6/14，p=0.0826） | −10.79（6/12，p=0.0599） | +6.64（9/5，p=0.2676） | — |
| aug:all p0.5 | −7.54（9/13，p=0.1940） | −5.42（7/15，p=0.2113） | −10.09（9/13，p=0.2902） | −1.77（6/8，p=0.7148） | **−21.46（3/10，p=0.0398）** |

逐类帧级指标（官方 micro-pool 口径，各 run 中位；P / R / F1）：

| 类（support） | 基线 | aug:hand | aug:all |
|---|---|---|---|
| idle（1458） | 60.4 / 72.4 / 65.1 | **61.6 / 81.6 / 68.6** | 57.0 / 71.2 / 62.0 |
| flush（164） | 21.4 / 12.8 / 16.0 | **83.3 / 6.1 / 11.4** | **30.2 / 19.5 / 23.7** |
| long_brush_insert（707） | **56.2 / 32.2 / 42.5** | 45.1 / 36.1 / 41.3 | 47.9 / 28.9 / 36.4 |
| long_brush_withdraw（259） | 9.6 / 6.9 / 8.1 | 11.2 / 7.7 / 9.1 | 10.4 / 5.4 / 7.1 |
| short_brush_cleaning（51） | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

读数：

1. **两个臂的段级指标全面变差**（`hand` 臂的 F1@0.1/F1@0.25 已到边界显著 p=0.083/0.060；
   `all` 臂的 insert precision 显著掉 21.46pp p=0.0398，acc 掉 7.7pp）。
2. **机制上不是"噪声被去掉"，而是"真信号被去掉"**：`hand`（手部位置）本来就指示器械在哪、
   动作正在发生——遮掉它，模型在 flush 上从"多而糙"（P 21.4 / R 12.8）变成"少而准"
   （**P 83.3 / R 6.1**），F1 反而更低。**precision↑ + recall↓ 不是净收益**，
   与第十四~十六轮"活动量旋钮"同型。
3. **全类遮罩有个有意思的副作用**：flush 的 P/R/F1 **同时上升**（16.0 → 23.7），
   说明"依赖多类共同证据"确实降低了 flush 的误报；但代价是 insert（42.5 → 36.4）与 idle
   （65.1 → 62.0）变差，**净段级仍是下降**。
4. **结论：目标随机遮罩不是提精度的杠杆**（本轮把它排除）。至此训练侧的所有可调项
   （配方/容量/架构族/类别权重/T-MSE/序列归一化/增强）都已测过一轮。

### 20.3 结论与下一步

1. 训练侧与特征侧的可调空间已探尽（§14/§16/§17/§18/§20）；唯一还有正收益的仍然是
   **`mstcn2` + 宽度 h128**（§16.1/§17.5）。
2. 真正的瓶颈在**数据与标注**：§19.3 的分解显示 51.6% 的真值段连标签都认错，且这些类的
   帧级线索在八~二十轮里反复被证伪；sbc 则是**检测覆盖为零**（需要补像素源）。
3. 评测口径建议（§19.4c）：段级指标里 92% 的匹配来自 idle，业务判读应以
   **逐类帧级 P/R** 为主——本轮起 `tools/compare_runs.py --per-class` 已能直接产出该表。

### 20.4 复现

```bash
# 两个增强臂（配置里已写死 augmentation 块；--set 无法寻址三层键）
python tools/run_capacity_matrix.py --runs-dir runs/round20-aug-hand \
    --config mstcn2-auto-roi-aug-hand.yaml --hidden 128 --seeds 42,7,2026 --epochs 60 \
    --set train.lr=0.0005 --no-eval-last --no-probe-baseline
python tools/run_capacity_matrix.py --runs-dir runs/round20-aug-all \
    --config mstcn2-auto-roi-aug-all.yaml --hidden 128 --seeds 42,7,2026 --epochs 60 \
    --set train.lr=0.0005 --no-eval-last --no-probe-baseline

# 对照（汇总 + 配对 + 逐类 P/R/F1）
python tools/compare_runs.py --left "runs/round20-aug-hand/*/mstcn2-*" --left-label "aug:hand" \
    --right "runs/mstcn2-cap-s2l5h128/*/mstcn2-*" --right-label "基线" \
    --metrics edit,f1_01,f1_025,insert,insert:precision --per-class

# 遮罩语义自检（不训练）
PYTHONPATH=. python - <<'PY'
import numpy as np
from framework.cleansight_eval.temporal.data import apply_target_mask_augmentation
cfg = {"root": "datasets/cleansight-ActionMixed-auto-lhh", "frames_dir": "frames"}
aug = {"target_mask": {"enabled": True, "strategy": "frame_dropout", "targets": ["hand"], "probability": 0.5}}
out = apply_target_mask_augmentation([np.ones((100, 144), dtype=np.float32)], cfg, aug, seed=42)[0]
print("hand 块清零帧数:", int((out[:, :18].sum(axis=1) == 0).sum()), "/100；其它类原样:",
      bool((out[:, 18:] == 1).all()))
PY
```

---

## 第二十一轮：逐帧错误 × 距标签边界距离——78% 的错误远离边界（2026-09-23）

> §19.3 已证明"段级失败的主因是类别认错"；本轮把同一问题下沉到**逐帧**：错误究竟集中在
> 标签切换处（时序定位问题），还是散布在段内部（整段认错）？这决定"改时序建模"还是"改标签/特征"。
> 工具侧在 `tools/probe_boundary_error.py` 上新增该分节（距离只由**标签切换点**定义，视频首尾不算边界）。

### 21.1 结果：错误率在边界处约 ×1.2~1.35，但绝大多数错误在段内

| 距最近标签切换 | ① `mstcn` h32（21112 帧）错误率 | ③ `mstcn2` s4l10 h128（7917 帧）错误率 | ③ 占总错误 |
|---|---:|---:|---:|
| 0–1 帧 | 57.4% | **61.2%** | 9.7% |
| 2–3 帧 | 58.2% | **61.7%** | 12.1% |
| 4–7 帧 | 57.0% | 59.3% | 19.9% |
| 8–15 帧 | 57.1% | 57.3% | 27.4% |
| 16+ 帧 | 38.2% | **29.7%** | 30.9% |
| 整体错误率 | 48.3%（10199/21112） | 45.3%（3585/7917） | — |

读数：

1. **边界帧的错误率确实更高，但只高 1.2~1.35 倍**（③：61.2% vs 段内 29.7%；①：57.4% vs 38.2%）——
   与第一轮"边界效应 0.33~0.63σ、0% 显著"的定性判断一致：**过渡是渐变的，不是硬边**。
2. **但按"占总错误的比例"看，边界几乎不是主战场**：距切换 ≤3 帧的帧占 ③ 全部帧的 **16.1%**，
   只承载 **21.9%** 的错误；**78.1% 的错误发生在距边界 4 帧以外**（① 同向：19.3% vs 80.7%）。
   → **错误的大头是"整段认错"，不是"边界差几帧"**，与 §19.3 的 51.6% 段标签错率互相印证。
3. **扩容改善的是段内、不是边界**：③ 在 ≥16 帧处把错误率从 ① 的 38.2% 压到 **29.7%**，
   但在 0–3 帧处两者都在 57~62%（几乎没有改善）。**边界过渡帧对容量不敏感**——
   这类错误要么来自标注时间轴本身的不确定，要么来自模型缺乏"切换瞬间"的判别线索。
4. 因此：**想继续提帧级指标，收益在"段内整段识别"（对应的仍是那几个不可分的类），
   而边界帧已被证明对容量、特征契约、损失旋钮都不敏感**。

### 21.2 关键澄清：**宽度买到的是"时序结构"，不是"逐帧正确率"**

上面 ① vs ③ 的对比同时换了架构与容量，因此补一组**同架构同族、同 seed**（42/7/2026）的对照
（`mstcn2 s2l5` h32 vs h128）：

| 配置（同 3 seed） | 0–1 帧错误率 | 2–3 帧 | 16+ 帧 | **池化帧错误率** | **acc 中位** | **edit 中位** |
|---|---:|---:|---:|---:|---:|---:|
| s2l5 **h32** | 56.3% | 57.7% | 33.5% | **45.7%** | **53.73** | 36.00 |
| s2l5 **h128** | 59.3% | 59.7% | 38.7% | **48.0%** | **53.66** | **46.45** |

逐 seed 明细（acc / edit）：h32 = 53.73/31.66、56.46/39.55、52.75/36.00；
h128 = 53.66/43.63、53.69/52.37、48.65/46.45。

**帧级正确率两种口径下都没有随宽度改善**（acc 中位 53.73 vs 53.66 持平；池化帧错误率甚至 45.7% vs 48.0%
更差，因为 h128 有一个 48.65 的低 seed），**而 edit 中位从 36.00 涨到 46.45（配对 +18.38，15/6，p=0.0142）**。
这修正了 §21.1 第 3 条的措辞（"扩容改善段内"是跨架构对照的错觉）：

1. **加宽不提高逐帧正确率**，它提高的是**预测的时序组织**——段边界与真值段结构的对齐程度
   （edit 是段标签序列上的编辑距离，只看"段怎么排"，不看"每帧对不对"）。
   因此 §16.1/§17.5 的"h128 最优"说的是**段级结构最优**；若要逐帧精度，加宽没用。
2. **① vs ③ 的差异来自架构（`mstcn2` s4l10 的多 stage 精化），不是容量**：③ 在 16+ 帧处
   29.7% 明显低于 ① 38.2%，而同族 h32→h128 无此收益（33.5% → 38.7%，反而略差）。
3. 这解释了容量报告 §4.3"容量买到段边界、买不到段检出"的机制来源：**容量改善的是段的排布，
   与逐帧分类能力是两条独立的通路**——也让"段级指标涨了 ≠ 模型更准"成为可验证的结论，
   与业务判读直接相关（应看逐类帧级 P/R）。
4. **口径提醒**：h32 的 **8 seed** 中位（acc 52.47 / edit 38.36）是更宽的基线统计，
   与 h128 的 3 seed 不可混用；本小节的结论建立在**同 seed 配对**上。

### 21.3 结论与下一步

1. 三条独立证据（§19.3 段级分解、§21.1 逐帧画像、§17/§18 特征契约穷尽）共同指向同一结论：
   **离线模型侧的进一步提升不在模型/特征/损失，而在数据与标注**。
2. 可执行动作不变（§19.4）：补像素源重跑检测（救 sbc）→ 标签时间轴人工审计（flush/withdraw）
   → 评测主指标改为逐类帧级 P/R（`tools/compare_runs.py --per-class` 已支持）。
3. **本轮把"边界帧"这一项也从待办里划掉**：它对容量不敏感（57~62% 恒定），不再作为优化目标。

### 21.4 复现

```bash
# 逐帧错误 × 距边界距离（① 弱基线 / ③ 最强模型）
python tools/probe_boundary_error.py --run "runs/capacity-lr0005-e60/h32/mstcn-*" \
    --run "runs/seedext-h32/h32/mstcn-*" --label "mstcn h32 基线（8 seed）"
python tools/probe_boundary_error.py --run "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" --label "mstcn2 s4l10 h128"
pytest tests/test_probe_boundary_error.py -q          # 10 passed（含距离分桶与占比断言）
```

---

## 第二十二轮：当前配方下的三族架构重测——**mstcn2 > (mstcn ≈ GRU) > Transformer**（2026-09-23）

> 第十二轮的"因果 GRU > MS-TCN > Transformer（edit 35.75 / 32.50 / 13.70）"用的是**旧配方**
> （默认 lr 2e-3、20 轮、早停）；容量研究已证明配方决定成败（lr 5e-4 + 足额预算下离线模型才兑现）。
> 本轮把三族拉到**统一健康配方**（lr 5e-4 / 60 轮 / patience=null / wd 1e-4 / dropout 0.2 /
> roi-144 契约 / 3 seed），并处理"因果 vs 离线"的口径问题后重排。
>
> **结论：rank 变为 `mstcn2` >（`mstcn` ≈ 因果 `GRU`）> `Transformer`**，
> 且 **GRU 显著差于当前最强 `mstcn2` s4l10 h128（edit −15.14，8/14，p=0.0348）**、
> **Transformer 显著差于同口径的 `mstcn2`（edit −24.24，5/18，p=0.0062）**。

### 22.1 配方与口径

| 项 | 值 |
|---|---|
| 统一配方 | lr 5e-4、60 轮、`patience=null`、wd 1e-4、`model.dropout=0.2`、`best_metric=val_f1_0.5`、roi-144 契约、seed 42/7/2026 |
| 流水线 | GRU = `sliding_window_temporal`（**因果**：窗口冷启动 + `causal_decision` 平滑）；`mstcn`/`mstcn2`/`transformer` = `full_sequence_temporal`（**离线**：逐帧 argmax、**无平滑**，§16.5/`docs/EVAL.md` §5） |
| GRU 口径变体 | md=1（最接近"无平滑"，与离线侧可比）与 md=5（第七/八轮推荐的因果口径），**只换评估、不重训**（`_eval_mdN/`，见 22.4 的工具增强） |
| 已有对照 | `mstcn2` s2l5/s4l10 h128（§16.1）、`mstcn` h128（容量研究，同配方同 lr） |

### 22.2 谱系（3 seed 中位数；官方 `metrics.summary`）

| 臂 | 流水线 | acc | edit | F1@0.1 | F1@0.25 | 逐 seed edit 摆幅 |
|---|---|---:|---:|---:|---:|---:|
| **mstcn2 s4l10 h128** | 离线 | 55.25 | **51.47** | **44.78** | **35.82** | **0.64** |
| mstcn2 s2l5 h128（d0.2） | 离线 | **55.63** | 49.03 | 38.66 | 21.85 | — |
| mstcn2 s2l5 h128（d0.3 默认） | 离线 | 53.66 | 46.45 | 41.29 | 33.59 | — |
| `mstcn` h128（d0.08） | 离线 | 50.63 | 41.79 | 26.98 | 21.40 | — |
| **GRU**（md=1） | 因果 | 51.80 | 42.40 | 38.66 | 22.73 | 14.5（48.5/34.0/40.9） |
| **GRU**（md=5，推荐因果口径） | 因果 | 52.03 | 40.89 | 37.04 | 22.22 | 14.5 |
| **Transformer**（d64×2） | 离线 | 48.73 | 27.31 | 23.79 | 18.59 | 8.0 |

### 22.3 配对检验（逐 (seed,视频) × 类 Wilcoxon）

| 对比 | edit | F1@0.1 | F1@0.25 | insert |
|---|---|---|---|---|
| GRU md=5 vs **Transformer** | **+25.12（15/8，p=0.0089）** | **+18.16（16/6，p=0.0115）** | +3.90（p=0.1895） | **+13.92（10/0，p=0.0020）** |
| Transformer vs **mstcn2 s2l5 h128**（同为离线、dropout 对齐 0.2） | **−24.24（5/18，p=0.0062）** | **−16.89（6/18，p=0.0065）** | −8.57（p=0.1909） | **−27.31（0/10，p=0.0020）** |
| GRU md=1 vs **mstcn h128**（两侧均"无平滑"） | +14.29（16/7，p=0.2476） | +12.50（16/5，p=0.0958） | +9.22（p=0.1752） | −1.12（p=0.7002） |
| GRU md=1 vs **mstcn2 s2l5 h128** | −10.32（8/12，p=0.2043） | −9.24（p=0.3943） | −7.30（p=0.5706） | **−20.55（3/11，p=0.0067）** |
| GRU md=5 vs **mstcn2 s4l10 h128**（最强） | **−15.14（8/14，p=0.0348）** | −8.06（p=0.3554） | −8.04（p=0.0917） | −24.38（p=0.1099） |

**机制（逐类帧级，官方 micro-pool 口径；GRU md=1 vs mstcn2 s2l5 h128）**：

| 类 | GRU md=1（P / R / F1） | mstcn2 s2l5 h128 |
|---|---|---|
| idle | 56.2 / **88.1** / 68.6 | 60.4 / 72.4 / 65.1 |
| flush | **0.0 / 0.0 / 0.0** | 21.4 / 12.8 / 16.0 |
| long_brush_insert | 37.6 / **10.7** / 16.5 | 56.2 / **32.2** / 42.5 |
| long_brush_withdraw | 9.0 / 2.7 / 4.2 | 9.6 / 6.9 / 8.1 |

→ **因果 GRU 是"保守型"**：非 idle 预测帧只有 354~376（mstcn2 为 671~778），idle 召回更高（88.1），
但 **flush 完全不预测（F1=0）**、insert 召回只有 mstcn2 的 1/3。段数比 0.51（欠分割）vs mstcn2 的 0.84。

### 22.4 结论与下一步

1. **修正第十二轮**：在 lr 5e-4 + 60 轮 + roi-144 下，**`mstcn2` 全面领先**；
   `mstcn` 与因果 `GRU` 落在同一档（+14.29 但 p=0.25，不可判读），`Transformer` 显著最差。
   旧结论"GRU > MS-TCN"只在**旧配方 + 只比 `mstcn`（无 `mstcn2`）**的语境下成立。
2. **GRU 的稳定性也差一个数量级**：逐 seed edit 摆幅 14.5（48.5/34.0/40.9），
   而 `mstcn2` s4l10 h128 只有 0.64（容量报告 §4.4）。要"可复现的分数"，离线 `mstcn2` 仍是唯一选择。
3. **dropout 是一个未登记但有效的旋钮**（副作用观察）：`mstcn2` s2l5 h128 把 dropout 从默认 0.3
   降到 0.2 → edit 46.45→**49.03**、acc 53.66→**55.63**，但 **F1@0.25 33.59→21.85**（权衡，非免费）。
4. **架构轴到此结束**：三族已在同一配方与正确口径下排完；`mstcn2` 仍是最优，
   与 §16.1（宽度）、§17/§18（特征契约）、§19（数据/标签瓶颈）共同构成完整的结论面。

### 22.5 复现

```bash
# 22.1/22.2 训练（GRU + Transformer；md=5 变体评估）
python tools/run_strategy_matrix.py --runs-dir runs/round19-arch-roi \
    --strategies roi-grid-144,transformer-roi-144 --seeds 42,7,2026 --epochs 60 \
    --set train.lr=0.0005 --set train.patience=null --smoothing-min-duration 5 --no-probe-baseline
# md=1 变体（只评估，不重训）
python tools/run_strategy_matrix.py --runs-dir runs/round19-arch-roi --skip-train \
    --strategies roi-grid-144,transformer-roi-144 --smoothing-min-duration 1 --no-probe-baseline
# dropout 对齐的 mstcn2 臂
python tools/run_capacity_matrix.py --runs-dir runs/round19-mstcn2-d020 \
    --config mstcn-actionmixed-auto-roi.yaml --set model.type=mstcn2 --set model.num_stages=2 \
    --set model.num_layers=5 --set model.dropout=0.2 --hidden 128 --seeds 42,7,2026 --epochs 60 \
    --set train.lr=0.0005 --no-eval-last --no-probe-baseline

# 22.3 对照（口径变体评估用 --left-eval-dir 指定，预测按同名 stem 从 run 目录取）
python tools/compare_runs.py --left "runs/round19-arch-roi/gru-*" \
    --left-eval-dir "runs/round19-arch-roi/_eval_md1/gru-*" --left-label "GRU md=1" \
    --right "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" --right-label "mstcn2 最强" \
    --metrics edit,f1_01,f1_025,insert --per-class
pytest tests/test_compare_runs.py -q
```

---

## 第二十三轮：选点口径消融——现行默认 `val_f1_0.5` 与 test 几乎不相关（2026-09-23）

> 容量报告遗留 #3："全部结论基于 `val_f1_0.5` 选点；`val_f1_0.25` 等新词表已可用但未对比"。
> 本轮把它补掉：先在全仓 **326 个 run** 上量化"候选 val 指标 → test 表现"的迁移能力，
> 再在旗舰配置上重训三种选点口径做直接对照（选点只用 val，无 test 泄漏）。

### 23.1 val→test 迁移体检（326 run；工具 `tools/probe_selection_transfer.py`）

| val 指标 | n | val 中位 | test 中位 | **val−test 中位差** | Pearson r | **Spearman ρ** |
|---|---:|---:|---:|---:|---:|---:|
| `val_edit` | 326 | 51.51 | 33.94 | **+17.55** | 0.355 | 0.410 |
| `val_f1_0.1` | 216 | 30.25 | 29.70 | +1.24 | 0.613 | **0.623** |
| `val_f1_0.25` | 216 | 21.45 | 21.05 | +1.66 | 0.588 | 0.559 |
| **`val_f1_0.5`（现行默认）** | 326 | 9.92 | 8.21 | +2.01 | 0.286 | **0.199** |
| `val_acc` | 326 | 68.20 | 52.90 | **+14.76** | 0.723 | 0.257 |

选点轮次分布（跨 run 摆幅越大 = 该口径越不稳）：`val_edit` 中位第 34 轮（1–142）、
`val_f1_0.25` 第 45 轮（3–147）、`val_f1_0.5` 第 31 轮（1–148）、**`val_loss` 第 5 轮（1–32）**。

读数：

1. **现行默认 `val_f1_0.5` 与 test 的秩相关只有 ρ=0.199**——按它选点**几乎等于随机挑一轮**；
   `val_f1_0.1`（ρ=0.623）与 `val_f1_0.25`（ρ=0.559）的迁移能力好一倍以上。
2. **`val_acc` 是陷阱**：val 68.20 vs test 52.90（差 +14.76），且 ρ=0.257——
   它的峰值常出现在**第 1 轮**（idle 坍缩早峰，如 §22 的 s4l10 seed42：val_acc 峰值在 epoch 1）。
3. **`val_edit` 的偏移最大（+17.55）**：val 上读到 51.5 的 edit 在 test 上只有 33.9。
   这是 val（4 视频）与 test（8 视频）之间的**批次差异**，也是"用 val 选 edit 目标"需要谨慎的原因。
4. **`val_loss` 完全不能用于选点**：它的最小值几乎总在第 3~9 轮（之后一路回升），
   按它选点等于固定选"最早的最低点"（与段级目标无关）。这也解释了为什么本仓库把
   `best_metric` 默认设成段级指标而不是 loss。

### 23.2 三口径重训对照（同配方、仅换 `best_metric`；3 seed 中位数）

| 臂 | acc | edit | F1@0.1 | F1@0.25 | F1@0.5 |
|---|---:|---:|---:|---:|---:|
| s4l10 h128 · **`val_f1_0.5`（现行默认）** | **55.25** | 51.47 | **44.78** | **35.82** | 12.00 |
| s4l10 h128 · `val_edit` | 54.00 | **58.35** | 42.96 | 33.33 | 12.00 |
| s4l10 h128 · `val_f1_0.25` | 53.66 | 46.97 | 39.68 | 31.75 | 9.52 |
| s2l5 h128 · **`val_f1_0.5`（现行默认）** | 53.66 | 46.45 | **41.29** | **33.59** | 14.19 |
| s2l5 h128 · `val_edit` | 51.88 | **52.83** | 35.96 | 25.35 | 10.11 |

配对检验（差值 = 该口径 − 现行默认）：

| 对比 | edit | F1@0.1 | F1@0.25 |
|---|---|---|---|
| s4l10 · `val_edit` | **+11.11（9/2，p=0.0068）** | −0.75（p=0.7869） | +0.27（p=1.0000） |
| s4l10 · `val_f1_0.25` | −1.07（8/8，p=0.4690） | **−2.85（6/13，p=0.0446）** | **−8.81（5/15，p=0.0441）** |
| s2l5 · `val_edit` | +4.48（11/9，p=0.7012） | +2.78（p=0.9457） | −4.98（p=0.8983） |

逐 seed（s4l10 h128，test edit）：`val_edit` = 58.35 / 62.03 / 51.07；`val_f1_0.5` = 51.47 / 51.71 / 51.07；
`val_f1_0.25` = 48.03 / 46.97 / 46.07。

读数（**含必须说明的局限**）：

1. **选点口径能把 headline 数字移动 11 分以上**（同一次训练、只换"留哪个 epoch 的权重"）：
   s4l10 h128 的 test edit 在 46.97 ~ 58.35 之间浮动，跨度 11.4。**选点是一等口径参数，必须随结果一起报**。
   > **⚠️ 这 11 分买的是什么（第二十四轮补充）**：`edit` 的实现是**段标签序列**的 Levenshtein，
   > **对时长/边界完全免疫**。实测这两个被选中的 checkpoint 在**所有 IoU 阈值下段匹配计数都没变**
   > （tp@0.10/@0.25/@0.50 基本不动），说明 `val_edit` 选点改善的是"**段标签序列的顺序**"，
   > 而**不是边界精度**。详见 §24。
2. **每个指标倾向于"选对自己"**：`val_edit` 选点拿到最好的 test edit（58.35，+11.11，p=0.0068 显著），
   `val_f1_0.5` 选点拿到最好的 test F1@0.1/F1@0.25，而 `val_f1_0.25` 两边都不占优
   （F1@0.1 −2.85 p=0.0446、F1@0.25 −8.81 p=0.0441 显著更差）。
   这与 §23.1 的"跨 run 相关系数"**不矛盾但也不能互推**：那个 ρ 是在**异构配置池**上算的，
   而选点发生在**同一次训练的相邻 epoch 之间**，后者才是决定分数的场景。
3. **局限（必须写进结论）**：s4l10 上 seed 2026 的三个口径**峰值同为第 55 轮**，
   因此该臂的有效独立 seed 只有 **2 个**（7 与 42，两 seed 的 test edit 一致上升 +10.32 / +6.88）；
   s2l5 的复现只是**同向**（edit 中位 +6.4）而**配对不显著**（p=0.7012），
   且 `val_edit` 选点在 s2l5 上把 **insert 召回压低 6.70pp**（p=0.0803）。
   → 结论定为"**若 headline 是 edit，`val_edit` 选点是当前最好的选择，但只在旗舰配置上被显著证实**"，
   不是普适结论。
4. **可执行建议**：① 报告里显式写 `best_metric`；② 以 edit 为主指标时用 `val_edit` 选点；
   ③ 不要用 `val_loss`（§23.1 第 4 条）与 `val_acc`（val 68.2 / test 52.9）选点。

### 23.3 8 seed 复核：`val_edit` 选点的优势成立（且顺带提 recall）

按 §23.2 的局限（s4l10 上有效独立 seed 只有 2 个）补跑 5 个新 seed（1/2/3/4/5），两臂同 seed 配对，
合计 **8 seed × 两口径 = 16 run**（配方逐字节相同，仅 `best_metric` 不同）。

| 指标 | `val_edit` 选点（8 seed） | `val_f1_0.5` 选点（8 seed） |
|---|---:|---:|
| acc | 53.98 | **54.64** |
| **edit** | **51.08** | 49.13 |
| F1@0.1 | **40.01** | 38.69 |
| F1@0.25 | 31.61 | **32.84** |
| F1@0.5 | 12.02 | 11.97 |
| 非 idle | 692.5 | 613.5 |
| 段数比 | 0.90 | 0.85 |

配对检验（逐 (seed,视频) × 类，n = 42~45 对）：

| 指标 | 中位差 | 胜/负 | p |
|---|---:|---:|---:|
| **edit** | **+9.63** | **29/13** | **0.0107** |
| **insert recall** | **+5.46** | **18/5** | **0.0214** |
| F1@0.1 | +2.20 | 25/20 | 0.1810 |
| F1@0.25 | +1.10 | 22/21 | 0.6992 |

逐 seed 的 Δedit：**6/8 提升**（+11.44 / +10.32 / +8.06 / +6.88 / +1.96 / +0.45 / −0.42 / 0.00），
中位 Δ = **+4.42**；8 个 seed 里**只有 seed 2026 两口径选中同一轮**（都是第 55 轮 → 预测完全相同，
是恒等对），其余 7 个才是真正不同的选点。

结论（升级）：

1. **`val_edit` 选点是一条被验证的改进**：edit **+9.63（29/13，p=0.0107）**、
   insert 召回 **+5.46（18/5，p=0.0214）** 双双显著，F1@IoU 不变（±2pp 内、p≥0.18），
   代价只是 acc 小幅下降（54.64 → 53.98）。按 §24 的语义，edit 增益是"**段标签序列更对**"
   （不是边界更准），而 **insert 召回上升是实打实的收益**。
2. **它同时暴露"3 seed 结论"的风险**：原 3 seed 的基线 edit 中位 51.47（三个 seed 都挤在 51 附近），
   而 5 个新 seed 的基线中位只有 **44.80** → **原 3 seed 恰好落在偏乐观的一段**。
   合并 8 seed 后基线中位 49.13、`val_edit` 51.08——配对幅度 +9.63 比"中位差"更可信（配对看逐视频差）。
3. **定版建议**：以 edit 为主指标时，`best_metric` 用 `val_edit`；报告须写明选点口径（§23.2 第 1 条），
   并同时报段数比与非 idle（§24.3 第 3 条）。

### 23.4 复现

```bash
# 23.1 迁移体检（全仓 326 run，纯读产物）
python tools/probe_selection_transfer.py --json tmp/sel_transfer.json
pytest tests/test_probe_selection_transfer.py -q          # 4 passed

# 23.3 的 8 seed 复核（两臂同 seed 补齐；配方逐字节相同）
python tools/run_capacity_matrix.py --runs-dir runs/round22-seedext-default \
    --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
    --hidden 128 --seeds 1,2,3,4,5 --best-metric val_f1_0.5 --epochs 60 \
    --set train.lr=0.0005 --no-eval-last --no-probe-baseline
python tools/run_capacity_matrix.py --runs-dir runs/round22-seedext-edit \
    --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
    --hidden 128 --seeds 1,2,3,4,5 --best-metric val_edit --epochs 60 \
    --set train.lr=0.0005 --no-eval-last --no-probe-baseline
python tools/compare_runs.py \
    --left "runs/round21-sel-edit-s4l10h128/h128/mstcn2-*" --left "runs/round22-seedext-edit/h128/mstcn2-*" \
    --left-label "val_edit 选点(8 seed)" \
    --right "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" --right "runs/round22-seedext-default/h128/mstcn2-*" \
    --right-label "val_f1_0.5 选点(8 seed)" --metrics edit,f1_01,f1_025,insert

# 23.2 三口径重训（只换 best_metric，其余配方逐字节相同）
for bm in val_edit val_f1_0.25; do
  python tools/run_capacity_matrix.py --runs-dir runs/round21-sel-${bm}-s4l10h128 \
      --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
      --hidden 128 --seeds 42,7,2026 --best-metric $bm --epochs 60 --set train.lr=0.0005 \
      --no-eval-last --no-probe-baseline
done
```

---

## 第二十四轮：`edit` 的语义澄清——它度量"段标签序列"，对时长完全免疫（2026-09-23）

> 由头：§23.2 里 `val_edit` 选点把 test edit 从 51.47 提到 58.35（+11.11，p=0.0068），
> 但同一批 checkpoint 在**所有 IoU 阈值下的段匹配计数都没变**（seed7：tp@0.10 30→29、tp@0.25 24→21；
> seed42：tp@0.10 30→31、tp@0.25 24→26）。一个"段边界指标"不该出现这种组合——于是回到实现去看。

### 24.1 实现事实：`edit` 只看段的**标签序列**

`framework/cleansight_eval/core/metrics.py::edit_score`：

```python
pred_labels = [segment.label for segment in segments_from_labels(predictions)]   # 只取标签
truth_labels = [segment.label for segment in segments_from_labels(truths)]       # 丢弃时长/位置
return 1.0 - levenshtein(pred_labels, truth_labels) / max(len(pred_labels), len(truth_labels))
```

即：把逐帧标签折叠成段后**只保留每段的标签**，再做归一化 Levenshtein 相似度。
→ **时长、边界位置、段的起止全部被丢弃**；它度量的是"段标签的出现顺序"，
并对**多出一段 / 漏掉一段**重罚。spec `edit/levenshtein-item-macro-mean/percent/v3` 就是这条口径的版本号，
改语义必须同时 bump 版本。

### 24.2 实测（自造用例，`tests/test_metrics.py::EditSemanticsTests` 已固化为回归）

真值 `idle(20) flush(10) idle(20)`：

| 预测 | edit | F1@0.5 | acc | 说明 |
|---|---:|---:|---:|---|
| 与真值完全一致 | 100.0 | 100.0 | 100.0 | 基准 |
| `idle(38) flush(1) idle(11)` | **100.0** | 66.7 | 78.0 | **顺序对、时长全错 → edit 满分** |
| `… idle(10) insert(1) idle(9)`（多一段） | **60.0** | 75.0 | **98.0** | **逐帧几乎全对却重罚** |
| `idle(30)`（漏掉 flush） | 33.3 | 0.0 | 66.7 | 漏段重罚 |

### 24.3 对既有结论的影响（读法修正）

1. **§23.2 的 "+11.11 edit" 不是"边界更准"**，而是"**段标签序列更对**"（少一些错位的段/多段）。
   这与 tp/fp/fn 在各阈值不变**完全自洽**——两条证据现在互相印证。
2. **容量报告 §4.3 的措辞已修正**：原文"容量买到的是段边界"→ 应为"段标签序列的顺序"。
   §16.1/§17.5 的"h128 最优"、§21.2 的"宽度买时序结构"同样只在这一语义下成立。
3. **读 `edit` 的铁律（与既有"必须报非 idle 帧"合并）**：报 edit 时必须同时报
   **段数比**与**非 idle 帧**——因为 edit 只惩罚"段的数量与顺序"，一个把动作说得又少又长的模型
   可以拿到高 edit 而召回极低（第十四~十六轮已多次出现这种假增益）。
4. **要衡量边界精度**，用 `f1@0.25/0.5`（IoU 匹配）与无偏的首尾偏移（§19.2 的 `tools/probe_boundary_error.py`）；
   `boundary_mae` 只在已匹配段上统计，有偏，不能当典型误差。
5. 口径文档已同步：`docs/EVAL.md` §3.1 的 `edit` 行补上"丢弃时长、对多段重罚"的语义与实测数字。

### 24.4 复现

```bash
pytest tests/test_metrics.py -q          # 16 passed（含 EditSemanticsTests 三项）
PYTHONPATH=. python - <<'PY'
from framework.cleansight_eval.core.metrics import temporal_metrics
LAB = ['idle','flush','insert']
truth = ['idle']*20 + ['flush']*10 + ['idle']*20
for tag, pred in (('顺序对/时长全错', ['idle']*38+['flush']*1+['idle']*11),
                  ('多一段', ['idle']*20+['flush']*10+['idle']*10+['insert']+['idle']*9)):
    m = temporal_metrics({'v': pred}, {'v': truth}, LAB)
    print(tag, 'edit=', round(m['segment']['edit']*100,1), 'acc=', round(m['frame']['accuracy']*100,1))
PY
```

---

## 第二十五轮：可执行结论汇总——杠杆定版表与当前最佳配方（2026-09-23）

> 把"动过哪些杠杆、结论是什么、代价多少"压成一张表，作为后续实验与报告的唯一入口。
> **表内每个数字都在本轮用 `tools/compare_runs.py` 重新核验过**（逐 (seed,视频)×类 Wilcoxon）；
> 详细机制见各自轮次。

### 25.1 杠杆定版表（按"是否值得采用"排序）

| # | 杠杆 | 定版设置 | 效果（配对中位差 / 胜-负 / p） | 代价 | 依据 |
|---|---|---|---|---|---|
| 1 | **架构族** | `mstcn2`（离线全序列） | GRU(md=5) 相对它 edit **−15.14 / 8-14 / p=0.0348**；Transformer **−24.24 / 5-18 / p=0.0062** | 离线大模型训练更慢 | §22 |
| 2 | **宽度** | **h128** | h64→h128 edit **+16.67 / 17-2 / p=0.0003**；h32→h128 **+18.38 / 15-6 / p=0.0142** | 参数 ~15× | §16.1 / §17.5 |
| 3 | **特征契约** | **roi-grid-144（z-score）** | 替代契约的 insert 召回 −11.9 ~ **−32.4**（p=0.0002~0.094）；96 维重排版 edit −18.80 p=0.0177 | 144 维前处理 | §17.2 |
| 4 | **选点口径** | **`best_metric=val_edit`**（以 edit 为主指标时） | edit **+9.63 / 29-13 / p=0.0107**；insert 召回 **+5.46 / 18-5 / p=0.0214** | acc 54.64→53.98 | §23.3 |
| 5 | **T-MSE 权重** | 保持默认 `0.15` | 消融（w=0）在 h128 上 F1@0.1 **−6.93 / 5-15 / p=0.0192** | —（不要动） | §16.2 |
| 6 | **序列归一化** | 保持 `none` | `demean` 只买召回（insert +29.7pp，13-0，p=0.0002）但 acc −11pp、段级不显著 | — | §15 |
| 7 | 活动量旋钮 `class_weight_clip` | 保持默认 `0.1` | `0.03`：insert **+19.43 / 24-10 / p=0.0101**，但 acc/fp 变差、段级不显著 | 精度换召回 | §16.3 |
| 8 | dropout | **保持默认 `0.3`**（`0.2` 不用） | s2l5 上 `0.2` 看似有用（edit 46.45→49.03、acc→55.63）但 **F1@0.25 −11.75**；**旗舰 s4l10 h128 上 `0.2` 显著更差：F1@0.25 −9.52 / 6-15 / p=0.0384、insert −14.29 / 1-8 / p=0.0273**、edit −11.11（p=0.0883） | —（不要动） | §22.4 + 第二十六轮 |
| 9 | 更深（stage/层数加倍） | **不用** | edit +5.56（p=0.3776）、insert **−11 ~ −20pp** | — | §16.1 |
| 10 | h256 以上加宽 | **不用** | s4l10：F1@0.1 −5.14 p=0.0401、F1@0.25 −12.16 p=0.0025 | — | §17.5 |
| 11 | 离线后处理平滑 | **不用** | val 选参后 edit 45.30 → **33.33**（test 选参会造出 +22 分假增益） | — | §16.4 |
| 12 | 目标随机遮罩增强 | **不用** | insert precision **−21.46 / 3-10 / p=0.0398**；`all` 臂 acc −7.7pp | — | §20 |
| 13 | **多种子集成**（逐帧多数投票） | **不用** | 最好的 k=8 也只打平（edit −0.85 p=0.81，F1@0.25 +1.17 p=0.38）；k=2/4 明显更差（−6.74 / −1.29） | — | §27 |

### 25.2 当前最佳配方（可直接复跑）

```
模型   mstcn2  num_stages=4 num_layers=10 hidden=128          # 331 万参
数据   temporal.actionmixed-auto-roi-v1（roi-grid-144 + z-score）
配方   lr 5e-4 / 60 轮 / patience=null / wd 1e-4 / dropout 0.3（默认）
       class_weight_clip 默认 / tmse_weight 默认 0.15 / sequence_normalization none
选点   best_metric=val_edit            ← 定版新增项（§23.3）
```

8 seed 实测：**edit 中位 51.08**（默认选点 49.13）、F1@0.1 40.01 / F1@0.25 31.61 / F1@0.5 12.02 /
acc 53.98 / 段数比 0.90 / 非 idle 692.5；对默认选点 **edit +9.63（p=0.0107）、
insert 召回 +5.46（p=0.0214）**。
> 注意 **3 seed 读数是 58.35**（§23.2）——8 seed 才是可信幅度（原 3 seed 偏乐观，见 §23.3 第 2 条）。

### 25.3 业务判读建议（口径）

1. **别只看 edit**：它只比较"段标签序列"（丢弃时长，§24），且段级匹配 92% 来自 idle。
   建议主指标 = **逐类帧级 P/R + `f1@0.25`**，并同时报**段数比与非 idle 帧**。
2. 当前最佳模型的逐类帧级（官方 micro-pool，P / R / F1）：`idle` 60.7 / 82.0 / 69.8、
   `long_brush_insert` 46.3 / 32.0 / 37.8、`flush` 35.7 / 15.2 / 21.4、
   `long_brush_withdraw` 10.6 / 4.6 / 6.5、`short_brush_cleaning` 0 / 0 / 0。
3. **天花板已定位**（§19.3/§21）：52% 的真值段连标签都认错；78% 的错误远离标签边界，
   且对容量/架构/损失/增强**均不敏感**；`sbc` 是检测覆盖为 0（缺像素源，§19.1）。

### 25.4 尚未做（附原因）

| 项 | 阻塞 |
|---|---|
| 检测侧扩召回（降阈值/补帧） | `datasets/cleansight-ActionMixed-auto-lhh/frames/` **只有 bbox txt、无图像**，无法重跑检测 |
| 标签时间轴人工审计 | 需人工看视频（工具已备：`tools/probe_boundary_error.py` 给出逐段分解） |
| GRU / Transformer 容量梯度 | 架构族排序已定（§22），容量轴在 `mstcn`/`mstcn2` 上已饱和（§16/§17.5） |
| 分类（ROI）真实数据链路 | 评估仍是 in-sample（缓存由 train+val 构建），需给 test 单独建缓存 |

### 25.5 复现

```bash
# 25.1 逐项核验（任一行都可用 compare_runs 复跑，例：宽度与选点）
python tools/compare_runs.py --left "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" --left-label "h128" \
    --right "runs/mstcn2-cap-s4l10h64/*/mstcn2-*" --right-label "h64" --metrics edit,f1_01,insert
python tools/compare_runs.py \
    --left "runs/round21-sel-edit-s4l10h128/h128/mstcn2-*" --left "runs/round22-seedext-edit/h128/mstcn2-*" \
    --left-label "val_edit 选点(8 seed)" \
    --right "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" --right "runs/round22-seedext-default/h128/mstcn2-*" \
    --right-label "val_f1_0.5 选点(8 seed)" --metrics edit,f1_01,f1_025,insert

# 25.2 最佳配方（一行复跑）
python tools/run_capacity_matrix.py --runs-dir runs/best-recipe \
    --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
    --hidden 128 --seeds 42,7,2026,1,2,3,4,5 --best-metric val_edit --epochs 60 \
    --set train.lr=0.0005 --no-eval-last --no-probe-baseline
```

---

## 第二十六轮：杠杆叠加测试——选点效应稳健、dropout 0.2 不迁移（2026-09-23）

> 承接 §25.1 的第 4 行（选点）与第 8 行（dropout）：这两条是近年唯二"看起来是改进"的旋钮。
> 本轮把它们放到**旗舰配置 `mstcn2` s4l10 h128** 上做正交叠加测试（同配方，只改 `model.dropout`
> 与 `best_metric`），回答两个问题：dropout 0.2 在旗舰上是否同样有效？两个杠杆能否叠加出更好的配方？

### 26.1 四条臂（同配方；仅 dropout 与选点不同）

| 臂 | seed 数 | acc | edit | F1@0.1 | F1@0.25 | 非 idle | 段数比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旗舰默认 · d0.3 + `val_f1_0.5` | 8 | **54.64** | 49.13 | 38.69 | **32.84** | 613.5 | 0.85 |
| 旗舰 · d0.3 + `val_edit` | 8 | 53.98 | **51.08** | **40.01** | 31.61 | 692.5 | 0.90 |
| d0.2 + `val_f1_0.5` | 3 | 54.07 | 45.62 | 40.98 | 27.87 | 500 | 0.70 |
| d0.2 + `val_edit` | 3 | 50.06 | 47.72 | 39.46 | 28.90 | 835 | 1.16 |

配对检验（逐 (seed,视频) × 类）：

| 对比 | edit | F1@0.1 | F1@0.25 | insert recall |
|---|---|---|---|---|
| **dropout 0.2 vs 0.3**（同一默认选点，3 seed 对 3 seed） | −11.11（5/12，p=0.0883） | −5.01（p=0.5016） | **−9.52（6/15，p=0.0384）** | **−14.29（1/8，p=0.0273）** |
| **`val_edit` vs 默认选点**（同在 d0.2 下，3 seed） | +7.78（12/10，p=0.2902） | +1.59（p=0.9457） | −3.41（p=0.9308） | **+12.17（9/1，p=0.0137）** |
| 叠加臂（d0.2 + `val_edit`）vs 旗舰默认 | −5.56（7/12，p=0.3955） | −1.48（p=0.4628） | −8.06（p=0.1560） | +0.93（p=0.5771） |

### 26.2 结论

1. **dropout 0.2 不迁移到旗舰配置**：在 `s2l5` h128 上它看似有用（edit 46.45→49.03、acc→55.63），
   在 `s4l10` h128 上却**显著更差**——**F1@0.25 −9.52（p=0.0384）、insert 召回 −14.29（p=0.0273）**，
   edit 也 −11.11（p=0.0883）。机制与 §16.3/§17.5 同型：它把模型推向保守
   （非 idle 671→500、段数比 0.84→0.70），**又是一个"小配置有效、大配置有害"的旋钮**。
   → §25.1 第 8 行已改为**保持默认 dropout 0.3**。
2. **选点效应（`val_edit`）在第三个语境下再次复现**：同在 dropout 0.2 下，`val_edit` 相对默认选点
   **insert 召回 +12.17（9/1，p=0.0137 显著）**、edit +7.78（p=0.29 方向一致）。
   加上 §23.3 的 8 seed 旗舰（insert +5.46 p=0.0214、edit +9.63 p=0.0107），
   **"`val_edit` 选点提 edit 与 insert 召回"已在三组独立数据上同向**，是本仓库最稳的免费杠杆。
3. **两个杠杆不能叠加出更好的配方**：叠加臂对旗舰默认仍是**全面不优**（edit −5.56、F1@0.25 −8.06，
   均为不显著的负向）——因为 dropout 0.2 的损失大于选点的增益。
   → **§25.2 的最佳配方不变**：`mstcn2` s4l10 h128 + 默认 dropout + `best_metric=val_edit`。
4. **方法论**：本轮再次证明"旋钮效应必须逐配置复核"——本仓库已出现四次
   "小配置有效、大配置无效/有害"（T-MSE 权重 §16.2、活动量 clip §16.3、T-MSE 在 h32 vs h128、
   dropout 本轮），**任何旋钮都不应跨容量外推**。

### 26.3 复现

```bash
# 26.1 四条臂（d0.2 两臂为本轮新增；d0.3 两臂见 §23.3）
python tools/run_capacity_matrix.py --runs-dir runs/round23-d020-s4l10h128-base \
    --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
    --set model.dropout=0.2 --hidden 128 --seeds 42,7,2026 --best-metric val_f1_0.5 \
    --epochs 60 --set train.lr=0.0005 --no-eval-last --no-probe-baseline
python tools/run_capacity_matrix.py --runs-dir runs/round23-d020-s4l10h128-edit \
    --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
    --set model.dropout=0.2 --hidden 128 --seeds 42,7,2026 --best-metric val_edit \
    --epochs 60 --set train.lr=0.0005 --no-eval-last --no-probe-baseline

# 对照（dropout 2 与 选点 2 分别隔离）
python tools/compare_runs.py --left "runs/round23-d020-s4l10h128-base/h128/mstcn2-*" \
    --left-label "d0.2" --right "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" --right-label "d0.3" \
    --metrics edit,f1_01,f1_025,insert
python tools/compare_runs.py --left "runs/round23-d020-s4l10h128-edit/h128/mstcn2-*" \
    --left-label "d0.2+val_edit" --right "runs/round23-d020-s4l10h128-base/h128/mstcn2-*" \
    --right-label "d0.2+val_f1_0.5" --metrics edit,f1_01,f1_025,insert
```

---

## 第二十七轮：多种子预测集成——被广泛使用，但在本任务上无效（2026-09-23）

> 动机：本仓库所有结论都建立在"单 seed 训练 + 逐 seed 对比"上，而**集成**是业界最常用的免费提分手段，
> 此前**从未测过**。逐帧标签的天然集成方式是**多数投票**（同视频同帧位、N 个 seed 取多数类）。
> 本轮用已存的 8 seed 预测产物直接验证（零训练成本），工具 `tools/probe_seed_ensemble.py`。
>
> **结论：无效。** 最好的情形是打平（val_edit 臂 k=8：edit −0.85，p=0.81），
> 其余情形都是负向，且成员越少越差。原因见 §27.3。

### 27.1 口径与公平性

- 输入 `artifacts/*.predictions.json`（与正式评测同源）；投票平局取**标签 id 较小者**（可复现）；
- **对照是"同一批成员的平均"，不是"最好的单 seed"**——后者用 test 挑 seed 属于泄漏；
- 统计 = 逐视频配对 Wilcoxon；每个 k 枚举成员组合（k=2/3/4 枚举全部或抽 70 组，k=8 唯一），
  报**组合中位**与组合内配对的 Δ/p 中位。

### 27.2 结果（旗舰 `mstcn2` s4l10 h128，8 seed × 8 视频）

**默认选点臂**

| k | 组合数 | edit | F1@0.1 | F1@0.25 | 成员均值 edit | Δedit | p | ΔF1@0.25 | p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 28 | 38.10 | 34.85 | 25.00 | 46.55 | **−6.74** | 0.2188 | −5.44 | 0.0781 |
| 3 | 56 | 47.78 | 38.75 | 30.36 | 46.90 | +1.10 | 0.6328 | −1.55 | 0.4688 |
| 4 | 70 | 44.44 | 34.85 | 26.79 | 46.74 | −1.29 | 0.5781 | −4.18 | 0.2188 |
| 8 | 1 | 38.10 | 36.67 | 29.17 | 45.55 | **−7.65** | 0.2188 | −4.60 | 0.3750 |

**val_edit 选点臂**

| k | 组合数 | edit | F1@0.1 | F1@0.25 | 成员均值 edit | Δedit | p | ΔF1@0.25 | p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 28 | 52.22 | 37.98 | 31.10 | 55.54 | −3.85 | 0.5625 | −2.96 | 0.2969 |
| 3 | 56 | 52.78 | 38.97 | 31.84 | 55.32 | −3.70 | 0.3750 | −1.63 | 0.5781 |
| 4 | 70 | 50.00 | 38.01 | 32.50 | 55.83 | −5.20 | 0.3438 | −0.67 | 0.5781 |
| 8 | 1 | 55.56 | 40.00 | 37.86 | 56.19 | **−0.85** | 0.8125 | **+1.17** | 0.3750 |

（另测"投票后再做最小时长合并 d=5"——仍无效：默认臂 k=8 的 Δedit −3.19、k=2 −9.99，全部不显著。）

### 27.3 结论与机制

1. **逐帧多数投票在本任务上不成立**：edit 在两个臂上都没赢过"成员均值"，最好的 k=8 也只是打平；
   部分集成（k=2/4）明显更差。
2. **机制**：不同 seed 的**段边界不同**，逐帧投票会在分歧处反复翻转标签 → **段被打碎**；
   而本任务的段级指标对"多出段"极敏感（§24：edit 是段标签序列的 Levenshtein，对多一段重罚），
   于是共识预测反而比任何单个成员都差。投票后加最小时长合并只修复一部分，仍无法超过成员均值。
3. **唯一的方向性信号**：val_edit 臂 k=8 的 **F1@0.25 比成员均值 +1.17（p=0.375，不显著）**——
   与"共识会轻微平滑边界"一致，但幅度不足以作为采用理由。
4. **对结论的意义**：**"集成是免费提分"在本任务不成立**，不要把它列进改进清单；
   要提分仍只能走 §25.1 已验证的杠杆（架构 / 宽度 / 契约 / 选点）。

### 27.4 复现

```bash
# 两个臂各 8 seed 的集成曲线（含 k=2/3/4/8 与成员均值对照）
python tools/probe_seed_ensemble.py --run "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" \
    --run "runs/round22-seedext-default/h128/mstcn2-*" --label "默认选点臂 8 seed"
python tools/probe_seed_ensemble.py --run "runs/round21-sel-edit-s4l10h128/h128/mstcn2-*" \
    --run "runs/round22-seedext-edit/h128/mstcn2-*" --label "val_edit 臂 8 seed"
# 投票后合并（先验 d=5）：加 --merge-d 5
pytest tests/test_probe_seed_ensemble.py -q          # 7 passed
```

---

## 定版结论（2026-09-18，**新 test = project-18**，取代下方"多 seed 版"结论）

> 下方"结论（多 seed 版）""坍缩分析""下一步建议"三节写于 **2026-09-03/04**，口径是旧 test
> （project-16 的 #195/#199，同批次）。它们对当时的口径仍然成立，但**不要用来读新 test 的排序**。
> 新口径下的完整证据链见第五~十一轮；本节是收口版。

### A. 评测口径（三条铁律，任何新方案比较都必须满足）

1. **标注 `smoothing_min_duration`**：它同时是召回上限（第七轮：md=25 把 test 的 sbc 锁死在 0，
   段级 edit 被压掉约一半；md=5 为推荐主口径，md=25 仅作历史对照）。
2. **给出逐类 support + 召回上限 + precision**：support=0 的类（test 无 water_injection）不可评估；
   insert/withdraw 这类"组级可分、方向不可分"的类，只看 recall 会误读（第十轮 §10.2）。
3. **给出线性探针下界**：逐帧 LDA（train 拟合、同 min_duration、同冷启动、同 `temporal_metrics`）
   是"特征 + 后处理"能做到的水平；方案不高于它就没有收益（第八轮；矩阵汇总已自动附加）。

### B. 特征方案定版排名（新 test，md=5，3 seed 中位数）

| 排名 | 方案 | acc | edit | F1@0.1 | F1@0.25 | 备注 |
|---|---|---:|---:|---:|---:|---|
| 1 | **roi-grid-144 + z-score 归一化** | 49.41 | **50.69** | 39.67 | **29.75** | edit 与同口径探针下界（50.63）持平；flush 17.7 / insert 18.1 |
| 2 | bbox-40-global | 45.36 | 47.18 | 34.92 | 26.89 | 段级次优；sbc 11.8 |
| 3 | bbox-40-hand | **51.12** | 37.84 | 37.38 | 29.91 | acc 最高但段级一般 |
| 4 | roi-grid-96-v2（可见性重排） | 51.53 | 37.74 | 33.96 | 22.64 | 维度省 1/3，指标不优于 v1 |
| 5 | bbox-80-global-hand | 51.80 | 29.38 | 28.85 | 18.00 | 最弱 |
| — | linear-probe（roi-144，下界） | 49.37 | 50.63 | **45.26** | **33.58** | 逐类仍高于全部模型 |

**架构侧（第十二轮，同配方同口径、md=1）**：因果流式 **GRU > MS-TCN > Transformer**
（edit 35.75 / 32.50 / 13.70；Transformer acc 最高但那 2/3 seed 全 idle 坍缩）。
即在新 test 上**流式模型反而最稳**，离线大模型没有兑现优势。

> **⚠️ 已于第二十二轮修正**：该结论用的是**旧配方**（默认 lr 2e-3、20 轮、早停）且当时**没有 `mstcn2`**。
> 在 lr 5e-4 / 60 轮 / roi-144 / dropout 0.2 的统一配方下重排为
> **`mstcn2` >（`mstcn` ≈ 因果 `GRU`）> `Transformer`**：GRU(md=5) 显著差于 `mstcn2` s4l10 h128
> （edit −15.14，8/14，p=0.0348），Transformer 显著差于同口径 `mstcn2`（edit −24.24，5/18，p=0.0062）。
> 详见 §22。下面这张"流式最稳"的结论**仅适用于旧配方语境**。

### C. 跨批次现实：所有方案都退化，根因**分类别**

新 test（project-18）上非 idle 召回整体 0~27%，idle recall 84~92%。第十一轮把根因拆开：

| 类 | 根因 | 证据 |
|---|---|---|
| `short_brush_cleaning` | **检测覆盖**：定义物 `short_brush` 在 test 的 sbc 帧检出 **0.0%**（train 11.7% / val 26.9%） | 第十一轮 §11.2 |
| `flush` | **标签/视觉口径**：判别物 syringe 在 test flush 帧反而更常见（18.3% vs train 5.7%），召回却 0~35% | 第六轮 6.1 + 第十一轮 §11.2 |
| `insert` / `withdraw` | **信息不存在**：方向判别 AUC≈0.5（含 ±24 帧上下文），train 却能到 0.95~1.000（记忆化） | 第十轮 §10.2 |

### D. 已验证**无效**的方向（省人天，别再试）

运动学/几何非线性块（提案 §1.4 + 第十轮双重否证）· 参考系归一化 · 可见性重排（指标不增，
维度降 1/3）· 加长滑窗（16/32/48 无差异）· 像素裁剪通道（比整帧好，但仍不及文本 ROI-144，
第十一轮 §11.1）。

### E. 已验证**有效**的改动

ROI-144 契约（第六~八轮双口径第一）· **z-score 输入归一化**（第九轮：ROI 上 edit +23%，
且把模型层差距补平到探针下界；bbox 契约无效，按契约决策）· md≤5 评测口径（第七轮）·
线性探针作为一等基线（第八轮）· 逐帧线性探针/方向探针/覆盖探针三个诊断工具（第八/十/十一轮）。

### F. 下一步该动的三件事（按收益排序）

1. **标签审计**（零成本）：flush、insert/withdraw 的时间轴与可见动作是否一致；
2. **检测侧扩召回**：sbc 在 test 已从特征层不可学，只能靠检测（逐类降阈值/换权重/补帧）；
3. **给 auto 通道补图像源**后重估像素通道（机制床上 ROI 裁剪方向已证实，但需 ≥ ROI-144 才立项）。

## 结论（多 seed 版，取代此前单 seed 结论）——**历史：旧 test 口径（见上方定版结论）**

1. **seed 方差巨大**：同策略不同 seed 的段级指标可差 3~5 倍（全局 40：F1@0.1 28.6 vs 11.4），
   部分 seed 仍坍缩到近全 idle（非 idle 预测 0~8 帧）——**单 seed 结论全部不可靠**，
   此前所有单 seed 数字只代表一个样本；多 seed 是硬性要求。
2. **ROI 网格（144 维）是唯一 CPU/GPU 两轮均三 seed 全部不坍缩的策略**
   （CPU 中位 F1@0.1 31.8 / F1@0.25 22.7，其余策略 5.9~15.8；GPU 中位数见第四轮，全局 40
   与 roi-grid 接近、仅手部全坍缩）——空间分区特征的空间冗余使其对 seed/初始化最稳健，
   抗坍缩是跨设备成立的唯一结论；段级指标排序则锚定设备口径。
3. **新配方（dropout/早停/段级选择）没有消除坍缩的 seed 依赖性**：bbox 系在 seed 2026
   下仍坍缩（edit 6.93 = 全 idle 上下界）。坍缩根因（稀有类样本少 + test 分布漂移）未变，
   配方只能缓解不能根治；ROI 网格的特征形态（区域统计天然抗单帧噪声）才是抗坍缩主因。
4. 早停实际停在第 5~9 epoch，best 多在第 1~5 epoch——概念学习期依然很短，dropout 未显著
   延长；进一步延长有效训练需数据扩量或更强正则/调度。
5. 遗留：选择指标（val_f1_0.5 vs val_acc）对结果的影响未单独消融；last.pt 未对比。
6. 推理延迟四者无实质差异（p95 ≈ 2–2.6 ms，远低于 33 ms 帧预算）——延迟不构成选型约束。

## 坍缩分析与配方修复（第一轮的教训，第二轮的依据）

详见对话记录与 git log `1346ec5` 后的分析；要点：

- **坍缩机制**：稀有类样本少 + 类别权重极端（idle 0.032）+ 无正则 20 epoch → 稀有类被
  "记忆化"而非"概念化"；test（task#195/#199 全新视频）分布漂移使记忆全部失活 → 输出坍缩
  到唯一跨视频稳定的 idle；val_acc 选择机制（多数类友好）再放大退化解。
- **修复**：5 epoch + weight_decay=1e-4 让模型停留在概念学习期（val_loss 不再从 epoch 1
  单调飙升），坍缩解除、段级指标大幅回升。
- **遗留（已修复 2026-09-03）**：best.pt 按 val_acc 选择、GRU 无 dropout、无早停——
  三项已在代码层落地（`train.best_metric` 可选 val_acc/val_edit/val_f1_0.5、
  `gru.py` 支持 model.dropout、`train.patience` 按 val_loss 早停，默认均向后兼容；
  另类别权重归一化后截断至 [0.1, 5.0]）。单 seed 问题待多 seed 复跑。

## 下一步建议——**历史：旧 test 口径（现行建议见上方"定版结论"F 节）**

- **多 seed（42/7/2026）复跑第二轮配方**：单 run 噪声大（CPU/GPU 同配置结果不同），
  三策略 × 3 seed 取中位数后才能下最终结论。
- 代码级配方修复：GRU 加 dropout、best checkpoint 指标可选（按段级）、早停实现。
- 若确认 insert/withdraw 仍弱：加"区域级差分"等运动学通道（新 feature mapping 版本）。
- 换 mstcn/transformer 全序列模型看段级表现是否改变排序。
- 图像通道（形态 B，bbox + 冻结 backbone embedding）已接入并跑过机制床 E0/E1 对照：
  段级 edit/F1@0.25 略优、帧级宏指标与 air_injection recall 下降，未定论；
  见 [`EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md`](EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md)
  与 [`features/README.md`](features/README.md) §1.7/§2.1。
