# 实验报告：时序模型架构调研（面向「动作段检测准确率」）

- 日期：2026-09-27
- 触发：需求「寻找一系列时序模型架构，满足检测动作段准确率提升的要求」
- **验收口径（2026-09-27 用户确认）**：以「**动作段真的被检出**」为准——真值动作段的**段覆盖率与
  稀类段/帧召回**为主指标，**不用 `edit`**（`edit` 奖励少分段：全预测 idle 反而拿 13.70 地板分）。
  任何"提升"须经逐 (seed,视频)×类 配对 Wilcoxon 支持（5 seed 下 edit 的 MDE ≈ 9 分）。
  本报告 §1.1/§1.5 的指标批评与这一口径一致。
- 范围：**架构/解码轴**（容量轴与特征轴此前已探尽）。本轮新增 2 个架构的实测、2 个训练无关解码探针的实测、
  2 份文献调研
- 性质：诊断（零训练成本复算）+ 新架构训练对照（CPU，已与 GPU 官方读数交叉验证）+ 训练无关解码对照
- 配套文献调研：[`LITERATURE_SURVEY_OFFLINE_TAS_20260927.md`](../LITERATURE_SURVEY_OFFLINE_TAS_20260927.md)、
  [`literature_survey_streaming_tas_identity_20260927.md`](../literature_survey_streaming_tas_identity_20260927.md)

---

## 0. 一屏结论

> **直接回答需求**：本轮**没有找到**能显著提升动作段准确率的时序架构——两条最对症的新架构
> （动作性解耦 `actionness_tcn`、层级注意力 `asformer`）实测为"无显著增益"与"**显著更差**"；
> 两个训练无关的解码侧方案（约束 Viterbi、段级重标注的可行性前提）实测**否定**。
> 但诊断把"该往哪投"钉死到了一个可判读的位置：**瓶颈既不在编码器、也不在解码器，而在
> "模型有没有在输出里形成这个动作的假设"**——同一份特征上闭式 LDA 能拿到稀类召回，
> 331 万参数模型拿不到。详见结论③④⑤。
>
> ⚠️ **后续（同日第 2 轮）**：不换任何架构，只把**容量 × 类别权重**这一对训练配方改成
> `mstcn2 s2l5 h32 + class_weight_clip=0.03`（**6.75 万参数 = 旗舰的 1/49**），
> 段召回即 **+25.0pp（逐视频配对 p=0.0070）**、`sbc` 段召回 **0%→23%** ——
> 见 [`EXPERIMENT_REPORT_SEGMENT_COVERAGE_20260927.md`](EXPERIMENT_REPORT_SEGMENT_COVERAGE_20260927.md)。
> **本报告第 1 轮所有架构否定的结论都发生在 h128 + 默认权重这一个配置点上，不可跨配方外推。**
> 该复测已完成：**`asformer` 在新工作点上段召回 +33.3pp（p=0.0392 显著）、`flush` 17%→63%**
> （平衡分与 `mstcn2` 打平，属召回↔精确率取舍），而 `actionness_tcn` 在第二个配置点上仍不可区分。

| # | 结论 | 等级 |
|---|---|---|
| ① | **旗舰模型帧准确率 53.93 < 平凡基线「全预测 idle」55.25**；换逐类平均（平衡）准确率后 25.20 vs 20.00，**只高 5.2pp**。acc 54 是 idle 占比撑起来的假象 | 可判读（8 seed × 2639 帧复算） |
| ② | 真实失败模式不是"边界不准"，而是**多数动作段没被检出**：8 seed / 264 个真值动作段汇总，**只有 27.3% 的段有 ≥50% 的帧被认成"某个动作"**，**56.1% 的段 ≤10%**；逐 run 中位：段内帧被认成动作仅 **35.9%**（范围 24.7~48.3%） | 可判读（`tools/probe_segment_coverage.py`，8 seed） |
| ③ | **模型的输出分布里几乎没有目标类的假设**：在**真值动作帧**上，模型给正确类的概率——`flush` 0.157 / `insert` 0.312 / `withdraw` 0.072 / **`sbc` 0.005**，而给 idle 的是 0.777/0.558/0.524/**0.979** | 可判读（`tools/probe_evidence_mass.py`，4 seed） |
| ④ | **因此解码侧无解**：任何 Viterbi/时长/转移/语法约束只能重排模型已支持的假设，`P(真值类)=0.005` 时无法凭空造出它。已用**实测**验证：硬最短时长约束 Viterbi 把段数比从 1.05 打到 **0.11**、覆盖率 1.4%→**0%**（4 run，0/4 提升） | 可判读（`tools/probe_constrained_decoding.py`） |
| ⑤ | **而信息其实在特征里**：同样 144 维特征上，闭式逐帧 LDA 拿到 `sbc` 召回 **21.6%**、`withdraw` 25.5%、`insert` 39.2%（模型：0.0% / 6.6% / 31.5%）。**每个稀类都输给线性判别器** → 问题是"模型没把已有信息变成输出假设"，属**表示/目标函数**层 | 可判读（`run_strategy_matrix.linear_probe_row`） |
| ⑥ | **本轮两条新架构实测**：`actionness_tcn`（主干与基线逐参数同构，只换输出参数化）**无显著增益**（edit 中位差 −5.38pp 但 p=0.58；f1@0.1 p=0.13；且非 idle 帧从 716 降到 545、稀类召回更低）；显式监督版更差。`asformer`（层级注意力）**显著更差**：F1@0.1 **−11.96pp（p=0.0212）**、F1@0.25 **−10.91pp（p=0.0296）** | 可判读（逐 (seed,视频)×类 配对 Wilcoxon） |
| ⑦ | 唯一**确证不可解**的是 `insert`↔`withdraw`：4 特征契约 × 4 档上下文，test AUC 全部 0.32~0.57（train AUC 可达 1.000）→ 方向信息不在特征里 | 可判读（`probe_direction.py`） |
| ⑧ | 本 test 集统计功效极低：**5 seed 下 edit 的最小可检测效应 ≈ 9.1 分**；test 仅 **73 真值段（非 idle 33 个）**，一个段 ≈ 3pp。**任何 <9 分的 edit"提升"都不可当结论** | 可判读（由 8 seed 官方读数算 sd） |
| ⑨ | 与本轮结论一致的**外部证据**：受控 14 编码器扫描（ICRA 2026）显示**换编码器只动 2.2~8.9 分**（参数量跨 16.2×），而**去掉一个流水线阶段动 15.8~40.6 分**；且"更大的 checkpoint 反而更差"。容量与编码器双轴在文献侧同样被证否 | 文献可判读 |
| ⑩ | **按验收口径（段覆盖率）最终判决**：四条路线**全部未超过基线**——基线 ≥50% 段占比 **29.5%**，`asformer` 18.2%、`actionness` 24.8%、`clip 0.03` 27.9%（非 idle 帧却 +22%）；且第十四轮在 **h32** 上读到的 insert 召回 +27.9pp **不可跨容量外推到 h128**（配对检验 p=0.0215，方向反了） | 可判读（§3.4） |

### 投什么、不投什么（按证据强度）

| 序 | 方向 | 为什么 | 成本 |
|---|---|---|---|
| **1** | **修「证据形成」而不是修架构**：模型在有信息的情况下仍给出 P=0.005。先把目标函数侧**没试过的那一半**补上——`transition-level` 代价敏感（文献实测 class-level 的 CB/LA/Focal/τ-norm 增益 ≈0，而 transition-level 拿 +8.1/+8.1/+5.7） | 结论③⑤的直接对症；`insert`/`withdraw` 正是逆转移对 | 低（只改损失） |
| **2** | **给 144 维特征补时间内容**：现行契约只有 presence/count/max_area，**完全没有 Δ 项**。Funke et al. (MICCAI 2019) 在同样"帧准确率不动、edit 大涨"的形态下验证过：仅空间 → 加时空，**edit 41.4→64.0、F1@10 55.4→75.2，帧准确率 79.9→79.9** | 结论⑤的机制解释：模型缺的是"变化"这条通道 | 低（改特征契约 + 重训） |
| **3** | **补 `per_class_frame_accuracy` 指标口径** | 现行 acc 掩盖了结论① | 零 |
| **4** | **数据/标签侧**：Cholec80 学习曲线 **1 段视频 40.1% → 3 段 71.1% → 80 段 89.5%**；本项目 train 仅 14 视频/210 段，正处在最陡段。`sbc` 另需检测侧覆盖（frames 目录无像素源） | 结论⑤的下游 | 高（人力） |
| **5** | **架构侧仅剩一个值得押注的方向：改变"输出什么"而非"多大/多深"** —— UVAST 式（独立时长/对齐解码器 + 逐帧辅助损失）或 FACT 式（action-token 匹配损失）。文献：MS-TCN→DiffAct **+16.7 Edit**；UVAST 以**更少参数**超过 ASFormer（Breakfast Edit 77.2 vs 75.0）。**注意 UVAST 是 AGPL-3.0、仓库已归档**，需按论文复现机制而非用其代码 | 结论⑨：唯有"改变输出形式"的架构才有文献支撑的增益 | 中-高（重训 + 调参） |

---

## 1. 瓶颈定位（全部为已存盘产物的复算，零训练成本）

### 1.1 旗舰配方的真实读数 vs 平凡基线

配方：`mstcn2` `s4l10 h128`（331 万参）+ `roi-grid-144`(z-score)，lr 5e-4 / 60 轮 / `best_metric=val_edit`；
数据 `temporal.actionmixed-auto-roi-v1`，test = 8 视频 / 2,639 帧 / 73 真值段（非 idle 33 段）。
数字为 8 seed（`round21-sel-edit-s4l10h128` + `round22-seedext-edit`）汇总，指标走官方
`framework.cleansight_eval.core.metrics.temporal_metrics`。

| 方案 | **global acc** | **per-class acc（平衡）** | macro-F1 | edit | F1@0.1 | F1@0.25 | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **mstcn2 s4l10 h128** | 53.93 | **25.20** | 25.32 | **51.08** | **40.01** | 31.61 | 12.02 |
| 平凡：全预测 idle | **55.25** | **20.00** | 71.17 | 13.70 | 0 | 12.35 | 0 |
| 平凡：随机均匀 | 17.09 | — | 15.15 | 4.73 | 0 | 0 | 0 |
| 逐帧闭式 LDA 探针 | 49.15 | 32.45 | — | 17.86 | 10.39 | 5.31 | — |

1. **global acc 已被平凡基线反超**（53.93 < 55.25）。文献在长尾 TAS 上实测 global 与 per-class
   帧准确率相差 **18.6~20.1 个点**（Breakfast 上 MS-TCN 67.7% vs 49.1%）；TPAMI 2024 综述明确批评
   帧准确率 "**can be problematic under dataset imbalance** … **does not capture segment quality**"。
2. **`macro-F1` 同样不可用**：只在"模型预测过的类"上求平均，全 idle 只定义一个类 → 拿 71.17，
   模型只有 25.32。**该指标奖励"从不预测稀类"**。
3. **`edit` 是唯一拉开差距的指标**（51.08 vs 13.70），但它**丢弃时长、只看段标签序列**，全预测 idle
   每视频只有 1 段反而拿到 13.70 地板分——它奖励"少分段"。见 §1.5。

### 1.2 失败模式：动作段几乎没被覆盖

先排除两个易误判的假设：

| 假设 | 检验 | 结论 |
|---|---|---|
| 预测与真值**时间轴没对齐** | shift ∈ [−40, +40] 的帧准确率 | **否定**：53.40~55.79 平坦无峰 |
| 段**边界对、只是标签错** | ±5 帧容差的边界 P/R/F1 | **否定**：P 0.222 / R 0.206 / **F1 0.213** |

再以**真值动作段**为单位算"段覆盖"（段内被预测为任一非 idle 类的帧比例）。
**注意用占比而不要用中位数**：`flush` 与 `sbc` 的中位覆盖率恒为 0，导致合并后的中位数骑在
0 与正值之间的刀刃上（同一配方换一批 seed，合并中位在 0.0%~8.7% 之间跳），不可单独引用。

| 类 | 真值段 | **覆盖率中位** | **覆盖率 ≥50% 的段占比** | 真值段时长中位（帧） |
|---|---:|---:|---:|---:|
| long_brush_insert | 88 | 28% | ~42% | 61 |
| long_brush_withdraw | 72 | 39~65% | ~47% | 27 |
| flush | 56 | **0%** | **3~5%** | 22 |
| short_brush_cleaning | 48 | **0%** | **0%** | **6** |
| **合计（264 段）** | 264 | 刀刃值，勿引用 | **27.3%** | — |

分布（264 段汇总）：**56.1% 的段覆盖率 ≤10%**、64.8% ≤25%、仅 **27.3% ≥50%**、11.7% ≥80%。
逐 run 稳定性（8 seed）：≥50% 段占比 **27.3%**（15.2~36.4%）；段内帧被认成动作 **35.9%**（24.7~48.3%）。

典型 trace（test 视频 1）：

```
GT  : idle(26) flush(23) idle(9) flush(22) idle(8) flush(27) idle(15) flush(13) idle(5) flush(9) …
PRED: idle(2)  insert(3) insert(8) idle(11) flush(4) idle(85) flush(3) idle(31) flush(3) …
```

模型**在最强的类上也只打了个 2~4 帧的短促脉冲**，随后立刻回到 idle。这解释了全部现象：
帧召回低、段级 IoU 匹配不上，而 `edit`（只看顺序）却不算差。

### 1.3 逐类帧级召回：深度模型全面输给闭式 LDA

| 类 | test 真值帧 | **深度模型 recall** | **闭式 LDA recall** | 模型预测该类帧占比 |
|---|---:|---:|---:|---:|
| idle | 1,458 | 80.9% | 63.0% | 73.2% |
| flush | 164 | 9.1% | **15.2%** | 1.1% |
| long_brush_insert | 707 | 30.2% | **39.2%** | 17.5% |
| long_brush_withdraw | 259 | **5.8%** | **25.5%** | 5.0% |
| short_brush_cleaning | 51 | **0.0%** | **21.6%** | 3.2% |
| water_injection | 0 | n/a（test 无样本） | n/a | — |

> 深度模型在**每一个稀类**上都不如一个不看时序的线性判别器；优势全在 idle（+17.9pp）。
> `short_brush_cleaning` 最极端：真值帧召回 **0%**，却把 3.2% 的帧预测成该类（落在 idle 帧上）。
> 而 `sbc` vs idle 的单帧 LDA **test AUC = 0.831**（`bbox-40-global` 0.856）
> → **信息在特征里，是模型的判决/表示把它丢了。**

### 1.4 唯一确证不可解的一类：insert ↔ withdraw

`python tools/probe_direction.py`（二分类 LDA，train 拟合 / val·test 评估）：

| 特征契约 | 单帧 test AUC | 居中-24 test AUC | 居中-24 **train** AUC |
|---|---:|---:|---:|
| bbox-40-global | 0.500 | 0.474 | 0.977 |
| bbox-80-global-hand | 0.469 | 0.569 | 0.997 |
| **roi-grid-144（主线）** | **0.367** | 0.462 | **0.999** |
| roi-grid-96-v2 | 0.561 | 0.481 | 1.000 |

AUC 0.5 = 不可分。**四类契约 × 四档上下文，test 全部 0.32~0.57，train 可达 1.000**：
"能背训练集、泛化时该信息不存在"。**加长上下文、换架构都救不了这一对。**

### 1.5 指标的结构性偏向与统计功效

**(a) `edit` 奖励少分段。** 全预测 idle 只有 1 段/视频却拿到 13.70 地板分；`class_weight_clip=0.03`
虽让 insert 召回 **+27.9pp（p=0.0012）**，段数比却从 1.77 涨到 3.59、edit 从 33.91 掉到 23.81
（`FEATURE_STRATEGY_COMPARE.md` 第十四轮）。
> **判读纪律**：报 edit/F1@k 必须同时报**段数比**与**非 idle 帧数**。
> 文献同构：ProTAS（CVPR 2024）把 EgoProceL 的 edit 从 19.9 拉回 **48.8**，**帧准确率反而从 72.7 降到 70.2**。

**(b) 统计功效**（由 8 个官方 seed 算）：

| 指标 | 中位 | 跨 seed sd | **5 seed MDE（80% 功效）** | 8 seed MDE |
|---|---:|---:|---:|---:|
| acc | 53.98 | 1.78 | 2.96 | 2.06 |
| edit | 51.08 | 5.46 | **9.08** | 6.30 |
| F1@0.1 | 40.01 | 3.18 | 5.29 | 3.67 |
| F1@0.25 | 31.61 | 2.66 | 4.42 | 3.06 |
| F1@0.5 | 12.02 | 2.09 | 3.47 | 2.41 |

逐 seed edit 实测摆幅 **46.76 ~ 62.03**。**结论必须走逐 (seed, 视频) × 类配对检验**（本轮全部采用）。

**(c) 指标可比性提醒**：官方 TAS 代码的 `Edit` 只丢弃字面量为 `"background"` 的帧；本项目类别名是
`idle`。**因此 edit 51.08 不能与任何已发表数字直接比较。**

### 1.6 段级误差的精确三分（8 seed × 8 视频 = 584 个真值段）

| 失败类型 | 段数 | 占全部真值段 | 是否可解 |
|---|---:|---:|---|
| ① 动作段被吞成 idle | **198** | **33.9%** | **可解**（§1.3：信息在特征里） |
| ② 动作段认成别的动作（其中 insert↔withdraw 35 段） | 40 | 6.8% | **不可解**（§1.4） |
| ③ idle 段被误报成动作 | 64 | 11.0% | 可缓解（precision 侧） |
| **合计标签错** | **302** | **51.7%** | — |

> 复算：`python tools/probe_boundary_error.py --run "runs/round21-sel-edit-s4l10h128/h128/mstcn2-*" --run "runs/round22-seedext-edit/h128/mstcn2-*"`（给出 301/584）；"错到哪一类"由逐段多数标签额外复算。

### 1.7 训练目标侧一处既存偏差（登记）

`compute_class_weights` 用「除以最大权重」归一化，而最大权重由**只有 2 个 train 段、test 完全没有的
`water_injection`** 决定。实际生效权重：
`idle 0.100 / insert 0.151 / flush 0.217 / sbc 0.398 / withdraw 0.444 / water_injection 1.000`。
一个 test 上不存在的类拿走了权重上界，而 idle 仍占加权损失约 39%。

---

## 2. 架构系列：选型逻辑与实测判决

### 2.1 为什么"再换一个编码器"没用（本轮已用测量与文献双重固定）

| 轴 | 已试 | 结果 |
|---|---|---|
| 容量 | h32→h128→h256（MS-TCN/MS-TCN++） | h128 到顶、h256 有害 |
| 特征契约 | 5 个替代契约 | 无任何契约超过 `roi-grid-144` |
| 编码器 | GRU / 因果 TCN / Transformer / MS-TCN / MS-TCN++ / **ASFormer（本轮）** | `mstcn2` 最优；Transformer edit −24；**ASFormer F1@0.1 显著 −11.96pp（p=0.0212）** |
| 输出参数化 | **动作性解耦（本轮）** | **无显著增益**，且更保守（非 idle 帧 716→545） |
| 解码 | **硬最短时长约束 Viterbi（本轮）** | **崩溃**：段数比 1.05→0.11、覆盖率→0% |

> 文献侧的同一结论：受控 14 编码器扫描（参数量跨 16.2×）显示**换编码器只动 2.2~8.9 分**，
> 而**去掉一个流水线阶段动 15.8~40.6 分**，且"更大的 checkpoint 反而更差"（四个模型族全部如此）。

### 2.2 本轮实现的架构与实测判决

#### (1) `actionness_tcn` —— 动作性解耦的双分支输出 · **判决：无显著增益**

```
P(idle | t) = 1 − a(t)
P(c    | t) = a(t) · softmax(class_logits)[c],  c ≠ idle
```

- 立项逻辑：§1.2 的失败是"回落到 idle"，而 6 路 softmax 里"是不是动作"与"是哪个动作"共享概率质量。
- **主干与基线逐参数同构（3,312,664，完全相同）** → 干净的单变量对照。
- 组合出的 `log p` 已归一化，`CrossEntropyLoss` 内部 `log_softmax` 是恒等映射
  （实测 `max|log_softmax(logp) − logp| = 4.8e-7`），**流水线零改动**；`x=±100` 时概率和仍为 1。
- 代码 [`actionness.py`](../../framework/cleansight_eval/temporal/models/actionness.py)，注册名 `actionness_tcn`。
- **实测（§3）：不是解药。** 非 idle 帧反而更少、稀类召回更低。

#### (2) `asformer` —— 局部膨胀卷积 + 全局自注意力多级精化解码 · **判决：显著更差**

- ASFormer（BMVC 2021）相对纯 Transformer 的两处差异：①每个 block 先局部膨胀卷积再全局注意力
  （对抗小数据上的逐帧过拟合）；②多级精化解码器 + 交叉注意力 + 深监督。
- `hidden 128 / heads 4 / enc 6 / dec 4` → **2,567,174 参数**（基线的 78%）。
- 修掉一处隐式分支：解码层"是否投影输入"改为**显式 `project_input` 开关**，不再按张量末维推断
  （`num_classes == hidden` 时形状判断会静默跳过投影）。已用 `hidden=64/classes=64` 边界验证。
- 代码 [`asformer.py`](../../framework/cleansight_eval/temporal/models/asformer.py)，注册名 `asformer`。
- **实测（§3）：显著更差**，方向与既有 Transformer 结论一致。

两者均在 `temporal/models/__init__.py` 注册（`causal: False`）；专属 `model.*` 键（`heads` /
`num_encoders` / `num_decoders` / `actionness_aux_weight`）已登记进 `core/config.py` 的
`KNOWN_SECTION_KEYS`（未登记的键**报错**，不静默丢弃）。

### 2.3 训练无关解码侧：本轮已实测并否定

| 方案 | 文献预期 | **本轮实测** |
|---|---|---|
| **约束感知 Viterbi**（CAD, ICPR 2026）：按类时长界硬约束解码 | 消融显示增益来自**约束**（纯 Viterbi 39.2 → 约束版 48.9 edit） | **崩溃**：段数比 1.05→**0.11**、段覆盖率 1.4%→**0%**、edit 50.18→13.70（= 全 idle 地板分），4 run **0/4** 提升。自检通过（α=0 且关闭约束时 Viterbi 与 argmax **逐帧完全相同**，证明实现无误） |
| **段级 NCM 重标注 / 任何"逐段重判标签"** | 绕开有偏分类器 | **前提不成立**：`sbc` 的 `P(真值类)=0.005`，重判只能更自信地选错（见 §0 结论③④） |

**机理**（这是本轮最有价值的一条可判读结论）：硬最短时长约束要求"一旦进入某类就至少持续 D_c 帧"，
而 train 真值 q10 时长为 `insert 25 / withdraw 10 / flush 7 / sbc 4 / idle 5`。
模型的动作证据既弱（`P(真值类)` 0.005~0.31）又断续，**承诺 25 帧的代价远超脉冲上那几帧的收益**，
于是"始终留在 idle"成为约束下的全局最优。**时长约束不能创造本来不存在的证据。**

### 2.4 文献支撑、本轮未实测的候选（已逐篇核验）

| 候选 | 机制 | 为什么仍值得排在架构侧最前 | 来源 / license |
|---|---|---|---|
| **UVAST**（ECCV 2022） | 独立**时长/对齐解码器** + 逐帧辅助损失，把"逐帧分类"与"段序列"解耦 | **其论文原文点名了本项目的失败模式**："the encoder performs frame-wise classification with **high localization performance … but low discrimination performance**, i.e., over-segmentation with low Edit distance"（归因于训练视频少）。且**以更少参数超过 ASFormer**（Breakfast edit 77.2 vs 75.0；50Salads 83.9 vs 79.6） | [arXiv 2209.00638](https://arxiv.org/abs/2209.00638)；**AGPL-3.0 且仓库已归档 → 按论文复现机制，不用代码** |
| **FACT**（CVPR 2024） | 帧分支 + 可学习 action token + 匹配损失，强制一个 token 编码一整段 | **唯一按构造**建模段身份；GTEA（小数据）Edit **91.4** vs MS-TCN++ 83.5；1.9M 参数 < 我们的 3.3M。风险：一对一匹配有 token 预算下限 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html)；**MIT** |
| **KARI**（NeurIPS 2023） | PCFG 活动语法 + Earley 解析重写标签序列 | 50Salads MS-TCN **+4.3 edit / +5.1~7.9 F1**，帧准确率持平。风险：语法质量敏感（坏语法把 ASFormer 从 76.5 打到 58.3）；但本项目流程高度有序，可手工写 | [arXiv 2312.04266](https://arxiv.org/abs/2312.04266)；**MIT** |
| **transition-level 代价敏感**（BMVC 2024） | 只惩罚逆转移对，而非类频率 | 论文同表内 CB/LA/Focal/τ-norm 增益 ≈0，transition-aware 拿 **+8.1/+8.1/+5.7**；`insert`/`withdraw` 正是逆转移对——**本项目唯一没碰过的重加权维度** | [arXiv 2503.18358](https://arxiv.org/abs/2503.18358) |
| **状态空间模型（Mamba/SSM）因果版** | 线性时间可流式长记忆 | 给**在线**路径一个非 TCN 备选（`mstcn2` 是离线模型，在线当前只能退 GRU）。诚实说明：**不存在**"Mamba 作逐帧 TAS 主干"的可核验代表作 | OAD 方向有 MOAD (AAAI 2026) |

**命名纠错（避免重复检索）**：`UVAST` = **ECCV 2022** & **AGPL-3.0**；`FactFormer` / `ICF` / `SSTCN`
在 TAS 文献中**不存在**（应为 FACT / ICC / BaFormer 自用的 MS-TCN 基线）；`LTContext` = **ICCV 2023**
且 **CC BY-NC（非商用，硬阻断）**；`DiffAct++` = TPAMI 2025（付费墙）。

### 2.5 不推荐投入的方向（有反例）

- **继续加大容量**（h256 已证有害；文献"更大 checkpoint 反而更差"）。
- **多种子集成**（段边界不一致 → 投票把段打碎，edit 更差，已有实测）。
- **无监督后处理平滑**（median / min-duration，已实测无效；**本轮进一步证明带约束的结构化解码同样无效**）。
- **只靠"更敢说"提召回**（clip 0.03：召回 +27.9pp 但段级指标下降——换指标不是提精度）。
- **TimesFM 类时序基础模型**（已证伪）。
- **CDF "段级"正则**：其段级损失**身份无关**（拿已选中的类去比均匀目标，无跨类项），按构造无法修 §1.2 的失败。

### 2.6 诚实的天花板

- **HeiChole 基准**（Med Image Anal 2023，12 队同数据）：器械存在 F1 **38.5~63.8%**，**动作识别 F1 仅 21.8~23.3%**。本项目 144 维在信息类型上等价于"器械存在"，要预测的却是 insert/withdraw 这类**动词**——跨团队独立验证过的断崖。
- **标注一致性**：两位专家 + 手册，kappa **阶段 95.9% 但步骤只有 80.8%**；SAGES 共识点名 "cut vs seal / grasp vs clamp" 存在可预测歧义。（边界：均在手术域；TAS 域**没有**一致性研究。）
- **段密度**（ICRA 2026 受控研究）：GTEA 36.3 段/视频（1.94 s）最难、Breakfast 5.4 段/视频（20.95 s）最易。**本项目 test ≈27 段/视频、段长中位 26 帧 ≈3.5 s，落在"最难"一侧**——这是与模型选择无关的、独立发表的难度解释。
- **重新定位期望**：工业装配数据集 Assembly101（I3D 特征）上 SOTA 整体也只有 Acc 37–41 / Edit 30–33。本项目的 53.93/51.08 不是"模型坏了"。

---

## 3. 实测结果

> 口径：全部跑在 **CPU**（本沙箱无 GPU 透传）。已先用同一配方复算 CPU 基线并与 GPU 官方读数交叉验证
> （acc 54.09 vs 53.98、edit 50.18 vs 51.08），确认 CPU 与 GPU 一致，故 CPU 上的架构对照可外推。
> 训练预算、特征契约、选点口径（`val_edit`）、`lr 5e-4 / 60 轮` 与基线完全一致。
> 判读一律走 `tools/compare_runs.py` 的逐 (seed, 视频) × 类 Wilcoxon 配对检验。

### 3.1 四臂主表

| 架构 | 参数量 | seed | acc | edit | F1@0.1 | F1@0.25 | F1@0.5 | 段数比 | 非 idle 帧 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`mstcn2` s4l10 h128（基线·GPU 官方）** | 3,312,664 | 8 | 53.98 | **51.08** | **40.01** | 31.61 | 12.02 | 1.05 | — |
| **`mstcn2` s4l10 h128（基线·CPU 复算）** | 3,312,664 | 4 | 54.09 | **50.18** | **39.21** | 32.03 | 14.74 | 1.05 | 716 |
| `actionness_tcn`（aux=0.0） | 3,312,664 | 5 | 54.72 | 50.92 | 41.96 | 32.39 | 14.08 | 0.93 | 545 |
| `actionness_tcn`（aux=0.5） | 3,312,664 | 5 | 54.11 | 46.97 | 36.99 | 30.14 | 13.70 | 1.00 | 601 |
| `asformer` h128 e6d4 | 2,567,174 | 3 | 52.48 | 44.15 | 30.08 | 21.95 | 10.57 | 1.22 | 563 |
| 平凡：全 idle | — | — | **55.25** | 13.70 | 0 | 12.35 | 0 | 0.11 | 0 |
| 闭式 LDA 探针 | 0 | — | 49.15 | 17.86 | 10.39 | 5.31 | — | — | 1088 |

### 3.2 配对检验（差值 = 基线 − 候选；**负值 = 候选更好**）

**`actionness_tcn` aux=0.0 vs 基线**

| 指标 | n | 胜/负 | 中位差 | p | 判读 |
|---|---:|---:|---:|---:|---|
| edit | 24 | 9/15 | **−5.38pp** | 0.5774 | **不显著** |
| f1@0.1 | 26 | 10/16 | −6.43pp | 0.1291 | 不显著 |
| f1@0.25 | 26 | 12/14 | −1.84pp | 0.3276 | 不显著 |
| f1@0.5 | 24 | 14/10 | +3.08pp | 0.3596 | 不显著（反向） |
| insert | 15 | 10/5 | +6.54pp | 0.3591 | 不显著（反向） |
| flush | 7 | 5/2 | +8.62pp | 0.2969 | 样本不足 |

**`actionness_tcn` aux=0.5 vs 基线**

| 指标 | n | 胜/负 | 中位差 | p | 判读 |
|---|---:|---:|---:|---:|---|
| edit | 25 | 14/11 | +3.51pp | 0.4431 | **不显著但方向为负** |
| f1@0.1 | 28 | 15/13 | +3.35pp | 0.4652 | 同上 |
| f1@0.25 | 28 | 15/13 | +6.11pp | 0.7282 | 同上 |
| flush | 7 | 6/1 | +10.34pp | 0.1094 | 同上 |

**`asformer` vs 基线**

| 指标 | n | 胜/负 | 中位差 | p | 判读 |
|---|---:|---:|---:|---:|---|
| edit | 21 | 14/7 | +15.38pp | 0.1023 | 不显著（方向为负） |
| **f1@0.1** | 22 | **17/5** | **+11.96pp** | **0.0212** | **显著更差** |
| **f1@0.25** | 20 | **15/5** | **+10.91pp** | **0.0296** | **显著更差** |
| f1@0.5 | 20 | 10/10 | +1.62pp | 0.5217 | 不显著 |

### 3.3 逐类帧级召回（关键判读面）

| 架构 | idle | flush | insert | withdraw | sbc |
|---|---:|---:|---:|---:|---:|
| **`mstcn2`（基线）** | 81.6 | 16.8 | 31.5 | 6.6 | **0.0** |
| 闭式 LDA 探针（**信息下界**） | 63.0 | 15.2 | 39.2 | 25.5 | **21.6** |
| `actionness_tcn` aux=0.0 | 86.3 | 7.9 | 21.9 | 3.9 | 0.0 |
| `actionness_tcn` aux=0.5 | 83.8 | 2.4 | 25.3 | 6.9 | 0.0 |
| `asformer` | 83.5 | 8.5 | 19.7 | 5.8 | 0.0 |

> **两条新架构在"召回"这一侧没有一个超过基线**；`actionness_tcn` 的 idle 召回反而更高
> ——与"解耦动作性以提高非 idle 召回"的立项意图**方向相反**。

### 3.4 按验收口径（段覆盖率）的最终判决

**验收口径（用户 2026-09-27 确认）：以「动作段真的被检出」为准**——段覆盖率与段占比为主指标，
不用 `edit`。工具：`tools/probe_segment_coverage.py`（本轮新增）。

| 臂 | run | 动作段 | **≥10% 段占比** | **≥50% 段占比** | **≥80% 段占比** | 段内帧被认成动作 | 非 idle 帧 | 段数比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **基线 `mstcn2` h128** | 4 | 132 | 49.2% | **29.5%** | **13.6%** | **36.9%** | 2837 | 1.02 |
| `clip 0.03` h128（活动量旋钮） | 5 | 165 | 46.1% | 27.9% | 12.7% | 36.4% | **3450** | 0.79 |
| `actionness_tcn` aux=0 | 5 | 165 | 43.0% | 24.8% | 9.1% | 30.2% | 2810 | 0.92 |
| `asformer` | 3 | 99 | 48.5% | 18.2% | 6.1% | 28.5% | 1802 | 1.59 |
| 平凡：全 idle | — | — | 0% | 0% | 0% | 0% | 0 | 0.11 |

**没有任何一条臂在验收口径上超过基线。** 三条读法：

1. **`clip 0.03` 在本项目最强模型上不成立**：它让非 idle 帧 **+22%（2837→3450）**，
   但"段内帧被认成动作"只从 36.9% 变成 36.4%、≥50% 段占比从 29.5% 降到 27.9%
   ——**多出来的活动全是错位的**（虚检），不是检出的段。配对检验里 `insert` 召回
   **显著更差**（31.5%→26.0%，中位差 +6.54pp，**p=0.0215**）。
   第十四轮在 **h32** 上读到的 insert 召回 **+27.9pp（p=0.0012）** **不可跨容量外推**
   ——这正是本项目自己写下的方法论警告（`FEATURE_STRATEGY_COMPARE.md` 第十四/十九轮）。
2. **`asformer` 的段结构最碎**：≥50% 段占比 18.2%（基线 29.5%）、≥80% 仅 6.1%，
   段数比 1.59（基线 1.02）——注意力换来的不是更好的段，而是更多的碎段。
3. **基线自身离验收口径很远**：只有 **29.5%** 的动作段有 ≥50% 的帧被当作动作，
   而 §1.3 已证明这些信息**在特征里**（闭式 LDA 的稀类召回高于模型）。

### 3.5 证据质量与解码侧实测

**模型在真值动作帧上的输出概率**（`tools/probe_evidence_mass.py`，基线 4 seed 中位）：

| 真值类 | 帧数 | **P(真值类)** | **P(idle)** | argmax 命中 |
|---|---:|---:|---:|---:|
| idle | 1458 | 0.800 | 0.800 | 81.6% |
| flush | 164 | 0.157 | 0.777 | 16.8% |
| long_brush_insert | 707 | 0.312 | 0.558 | 31.5% |
| long_brush_withdraw | 259 | 0.072 | 0.524 | 6.6% |
| **short_brush_cleaning** | 51 | **0.005** | **0.979** | **0.0%** |

**约束感知 Viterbi 解码**（`tools/probe_constrained_decoding.py`，4 run，训练无关）：

| 解码 | acc | edit | F1@0.1 | F1@0.25 | F1@0.5 | 段数比 | 段覆盖率中位 |
|---|---:|---:|---:|---:|---:|---:|---:|
| argmax（基线） | 54.09 | 50.18 | 39.21 | 32.03 | 14.74 | 1.05 | 7.4% |
| 硬最短时长约束 Viterbi | 53.88 | **13.70** | 19.63 | 12.35 | 0.00 | **0.11** | **0.0%** |
| 自检：关闭约束（应 = argmax） | 54.04 | 52.62 | 46.98 | 33.56 | 12.08 | 1.04 | 13.5% |

> 第三行与第一行（同一 run）**逐帧完全相同** → 解码器实现正确，第二行的崩溃是**约束本身**导致的，
> 不是 bug。逐 run edit 提升 **0/4**。

---

## 4. 结论与建议

### 4.1 直接回答"找一系列能提升动作段准确率的时序架构"

**本轮没有找到。** 按用户确认的验收口径（**动作段真的被检出**，见 §3.4），四条路线全部未超过基线：

| 路线 | 代表 | 验收口径下的读数 | 判决 |
|---|---|---|---|
| 换编码器 | `asformer`（层级注意力 + 多级精化） | ≥50% 段占比 18.2% vs 基线 29.5%；F1@0.1 **p=0.0212 显著更差** | **否定** |
| 改输出参数化 | `actionness_tcn`（与基线逐参数同构） | ≥50% 段占比 24.8% vs 29.5%；非 idle 帧更少（545 vs 716/run） | **无增益** |
| 改解码 | 硬最短时长约束 Viterbi | 段数比 1.05→**0.11**、覆盖率→**0%**，0/4 提升 | **崩溃** |
| 调活动量 | `class_weight_clip=0.03` @h128 | 非 idle 帧 **+22%** 但覆盖率不涨（36.4% vs 36.9%）；insert 召回 **p=0.0215 显著更差** | **否定**（h32 结论不可跨容量外推） |

> 文献侧同一结论：受控 14 编码器扫描（参数量跨 16.2×）显示**换编码器只动 2.2~8.9 分**，
> 而**去掉一个流水线阶段动 15.8~40.6 分**；且"更大的 checkpoint 反而更差"。

### 4.2 真正的瓶颈（可判读）

**模型没有把已有的信息变成输出假设。** 三条互相独立的读数：

- 同一份 144 维特征上，**闭式 LDA 的稀类召回全面高于 331 万参数模型**（§1.3）；
- 模型在**真值动作帧**上给正确类的概率只有 0.005~0.31（§3.5）；
- 因此任何解码器（Viterbi / 时长 / 转移 / 语法）都无从下手——它们只能重排模型已支持的假设。

这不是"架构不够好"，而是**训练把可用的判别信息丢掉了**。外部佐证：HeiChole 的
"器械存在→动作动词"断崖（存在 F1 38.5~63.8% vs 动作 F1 21.8~23.3%）；以及"去掉一个流水线阶段
动 15.8~40.6 分"的受控实验。

### 4.3 建议动作（按性价比，均已按验收口径设计证伪判据）

| 序 | 动作 | 依据 | 成本 | 证伪判据（一律走配对检验） |
|---|---|---|---|---|
| 1 | **`transition-level` 代价敏感重加权**（只罚逆转移对） | 文献同表 class-level 增益 ≈0、transition-aware +8.1/+8.1/+5.7；`insert`/`withdraw` 是逆转移对，本项目从未动过。**注意：class-level 的活动量旋钮本轮已在 h128 上被证否（§3.4）** | 低（只改损失） | ≥50% 段占比不显著提升 → 放弃 |
| 2 | **特征补 Δ 项**（每 ROI 的 Δcount / Δmax_area） | 现行契约**完全没有时间导数**；Funke et al. 在"帧准确率不动、edit 大涨"的形态下验证过（edit 41.4→64.0）。§3.5 的 `sbc` P=0.005 最可能是缺了这条通道 | 低（新契约 + 重训） | 稀类段覆盖率不显著提升 → 放弃 |
| 3 | **补 `per_class_frame_accuracy` 与段覆盖率进指标注册表** | §1.1 的 acc/`macro-F1` 都会奖励"从不预测稀类"；验收口径需要可训练侧选点的 `val_segment_coverage` | 低 | —（口径修复） |
| 4 | **修 `compute_class_weights` 的归一化**（不要让 test 上不存在的类定权重上界） | §1.7 | 低 | 与 clip 同向则说明是同一旋钮（本轮已证该旋钮在 h128 无效） |
| 5 | **数据与标签**：追加标注视频；`sbc` 补像素源重跑检测；`flush`/`withdraw` 人工审计标签轴 | Cholec80 学习曲线 1→3→80 段：40.1→71.1→89.5 | 高（人力） | — |
| 6 | **架构侧仅剩一个方向**：改变"输出什么"（UVAST 式独立时长/对齐解码器，或 FACT 式 action-token 匹配） | MS-TCN→DiffAct **+16.7 edit**；UVAST 以更少参数超 ASFormer | 中-高 | ≥50% 段占比不显著提升 → 放弃 |

**不要做**：加容量、多种子集成、无监督后处理、纯编码器替换、解码侧结构约束、class-level 活动量旋钮
（以上均有本轮实测或既有实测的否定证据）。

---

## 5. 复现

```bash
# ① 段级误差分解与平凡基线（零训练成本，读已存盘 predictions）
python tools/probe_boundary_error.py --run "runs/round21-sel-edit-s4l10h128/h128/mstcn2-*" \
    --run "runs/round22-seedext-edit/h128/mstcn2-*"
python tools/probe_direction.py                                   # insert/withdraw：test AUC 0.32~0.57
python tools/probe_direction.py --pair idle,short_brush_cleaning  # sbc：test AUC 0.831（信息在）

# ② 逐帧闭式 LDA 参照（"特征里有多少信号"的下界）
python -c "import sys;sys.path[:0]=['tools','.'];from pathlib import Path;\
from run_strategy_matrix import linear_probe_row;print(linear_probe_row(Path('<run_dir>'), None))"

# ②b 验收口径：真值动作段的覆盖率（「动作段真的被检出了吗」）
python tools/probe_segment_coverage.py \
    --run "runs/arch-survey/baseline-mstcn2-cpu/h128/mstcn2-*" --label 基线 \
    --run "runs/arch-survey/cwclip003-h128-cpu/h128/mstcn2-*" --label "clip0.03" \
    --run "runs/arch-survey/actionness-aux0-cpu/h128/actionness_tcn-*" --label actionness

# ③ 证据质量：模型在真值动作帧上给了目标类多少概率（本轮新增，判"解码侧有没有空间"）
python tools/probe_evidence_mass.py --run "runs/arch-survey/baseline-mstcn2-cpu/h128/mstcn2-*"

# ④ 约束感知 Viterbi 解码（训练无关；本轮结论：硬时长约束崩溃）
python tools/probe_constrained_decoding.py --run "runs/arch-survey/baseline-mstcn2-cpu/h128/mstcn2-*"
python tools/probe_constrained_decoding.py --run "<one run>" --no-duration   # 自检：应等于 argmax

# ⑤ CPU 基线（与 GPU 官方读数交叉验证）
python tools/run_capacity_matrix.py --runs-dir runs/arch-survey/baseline-mstcn2-cpu \
    --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
    --hidden 128 --seeds 1,7,42,2026 --best-metric val_edit --epochs 60 \
    --set train.lr=0.0005 --no-eval-last --no-probe-baseline

# ⑥ 新架构对照（主干与基线逐参数同构 / 换编码器）
python tools/run_capacity_matrix.py --runs-dir runs/arch-survey/actionness-aux0-cpu \
    --set model.type=actionness_tcn --set model.num_stages=4 --set model.num_layers=10 \
    --set model.actionness_aux_weight=0.0 \
    --hidden 128 --seeds 42,7,2026,1,2 --best-metric val_edit --epochs 60 \
    --set train.lr=0.0005 --no-eval-last --no-probe-baseline

python tools/run_capacity_matrix.py --runs-dir runs/arch-survey/asformer-cpu \
    --set model.type=asformer --set model.heads=4 \
    --set model.num_encoders=6 --set model.num_decoders=4 \
    --hidden 128 --seeds 42,7,2026 --best-metric val_edit --epochs 60 \
    --set train.lr=0.0005 --no-eval-last --no-probe-baseline

# ⑥b 活动量旋钮在 h128 上复测（第十四轮的 h32 结论不可跨容量外推）
python tools/run_capacity_matrix.py --runs-dir runs/arch-survey/cwclip003-h128-cpu \
    --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
    --set train.class_weight_clip=0.03,5.0 \
    --hidden 128 --seeds 42,7,2026,1,2 --best-metric val_edit --epochs 60 \
    --set train.lr=0.0005 --no-eval-last --no-probe-baseline

# ⑦ 配对检验（必做：中位数差 <9 分不可判读）
python tools/compare_runs.py --left "runs/arch-survey/baseline-mstcn2-cpu/h128/mstcn2-*" \
    --left-label baseline --right "runs/arch-survey/asformer-cpu/h128/asformer-*" \
    --right-label asformer --per-class
```

---

## 6. 遗留（未做，含原因）

1. **`per_class_frame_accuracy` 与 `segment_coverage` 未进指标注册表**：本报告用
   `tools/probe_segment_coverage.py` 与一次性脚本算出，建议把段覆盖率登记进
   `framework/cleansight_eval/core/metrics.py` 的 `TEMPORAL_METRIC_SPECS`（带 `val_` 前缀），
   使验收口径可用于**训练侧选点**（现行 `best_metric` 选项里没有一个与验收口径对齐）。
2. **§4.3 的 1/2/4/6 四项未实测**：transition-level 重加权、Δ 特征契约、权重归一化修复、
   UVAST/FACT 式架构——本轮 CPU 时间用在 5 个训练批次 + 3 个探针上。
   > **补记（同日第 3 轮）**：本报告 §2.3 判定"段级重标注（S-NCM）前提不成立"的依据是
   > `P(真值类)=0.005`。该前提在第 2 轮改配方后**已不成立**（`sbc` 的 `P(真值类)` 升到 0.182），
   > S-NCM 因此可做，并在 **4/4 个 `mstcn2` 臂上显著提高 `F1@0.1`（+3.4 ~ +8.4）**——
   > 见 [`EXPERIMENT_REPORT_SEGMENT_NCM_20260927.md`](EXPERIMENT_REPORT_SEGMENT_NCM_20260927.md)。
3. **`sbc` 检测侧覆盖**：`datasets/cleansight-ActionMixed-auto-lhh/frames/` 只有 bbox txt、无图像，
   无法重跑检测；需先补像素源。
4. **`insert`/`withdraw` 标签轴**：§1.4 已证信息不在特征里；需人工看视频判断是标注约定问题还是确需方向特征。
5. **功效上限**：即便按本轮最敏感的口径（逐 (seed,视频)×类 配对），3~5 seed 也只能分辨
   10pp 量级的效应。§4.3 第 6 项若要做，建议直接上 8 seed。
6. **本轮新增了 2 条架构边界门禁违规（登记，未修）**：`tools/probe_evidence_mass.py` 与
   `tools/probe_constrained_decoding.py` 需要**加载 checkpoint 并前向**，因此 `import torch`，
   触发 `tests/test_architecture_boundaries.py::test_active_non_framework_code_does_not_execute_models`
   （该测试禁止 `benchmark/`、`tools/` 直接调用模型库）。这与既有的
   `tools/probe_pixel_channel.py`、`tools/run_strategy_matrix.py` 属**同一类**违规；
   周报已把"放宽 `tools/` 运行规则 vs 把执行模型的工具下沉到 `framework/`"登记为待决策项。
   **按该门禁的意图，正确解法是下沉**：把"读 checkpoint + 前向"搬进
   `framework/cleansight_eval/`，`tools/` 只留参数解析与打印。本轮未做，以免在门禁决策未定时
   单方面改动分层；**但这两条债务是新增的，须与既有违规一并处置**。
   （`tools/probe_segment_coverage.py` **不违规**——它只读已存盘 predictions，不加载模型。）

---

## 7. 引用来源

- 本项目既有结论：[`EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`](EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md)、
  [`EXPERIMENT_REPORT_MSTCN_CAPACITY_20260922.md`](EXPERIMENT_REPORT_MSTCN_CAPACITY_20260922.md)、
  [`FEATURE_STRATEGY_COMPARE.md`](../FEATURE_STRATEGY_COMPARE.md) 第十四/十五轮、
  [`MODELSET_OVERVIEW.md`](../MODELSET_OVERVIEW.md)
- 本轮文献调研（URL 已逐篇抓取核验）：[`LITERATURE_SURVEY_OFFLINE_TAS_20260927.md`](../LITERATURE_SURVEY_OFFLINE_TAS_20260927.md)、
  [`literature_survey_streaming_tas_identity_20260927.md`](../literature_survey_streaming_tas_identity_20260927.md)
- 关键外部文献：
  - [UVAST: Unified Video Action Segmentation with Timestamp Supervision (ECCV 2022)](https://arxiv.org/abs/2209.00638)
  - [FACT: Frame-Action Cross-Attention Temporal Modeling (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html)
  - [Cost-Sensitive Learning for Long-Tailed TAS / S-NCM (BMVC 2024)](https://arxiv.org/abs/2503.18358)
  - [Constraint-Aware Viterbi Decoding (ICPR 2026)](https://arxiv.org/abs/2605.10149)
  - [KARI: Activity Grammar Parsing (NeurIPS 2023)](https://arxiv.org/abs/2312.04266)
  - [ASFormer (BMVC 2021)](https://arxiv.org/abs/2110.08568)
  - [ProTAS: Progress-Aware Online Action Segmentation (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf)
  - [Exploring Vision-Language Models for Zero-Shot Action Segmentation (ICRA 2026)](https://arxiv.org/abs/2602.21406)
- 文档回写：[`usage/YAML_CONFIG.md`](../../usage/YAML_CONFIG.md)（新 `model.*` 键）、[`docs/README.md`](../README.md) §6 登记

---

## 补记（2026-10-02）：架构轴补上"每族自己的调参机会"——结论不变

**为什么补**：本报告的架构对照**全在同一配置点**上做（每族 1 个配置点、沿用 `mstcn2` 的配方），
所以严格说它否证的是"**在该配方下换架构就能更好**"。本轮把这句缺口补上：给每族 **1~7 个配置点**
（lr × 容量），并把训练预算放宽到 3 倍，再与参考同 seed 配对比较。

**规模与产物**：**27 个配置点 / 216 个 run**（`runs/arch-fair/`），全部 `roi-grid-v4` 契约、与参考臂
**同 seed 集**（42,7,2026,1,2,3,4,5 + 11,13,…,37）可逐 seed 配对。

| 族 | 配置点数 | 最好配置点 | acc | Δ vs 参考 | 胜/负 |
|---|---:|---|---:|---:|---|
| `mstcn`（单阶段） | 3 | h128 lr1e-4 | **54.89** | **−3.43** | 0/8 |
| `fact` | 2 | d256 lr5e-4 | 53.64 | −4.32 | 0/8 |
| `transformer` | 7 | d128 lr5e-4 +warmup/cosine | 53.41 | −4.17 | 0/8 |
| `clean_asformer`（**此前从未训练过**） | 2 | h64 lr5e-4 | 53.24 | −3.77 | 0/8 |
| `clean_bigru`（**此前从未训练过**） | 1 | h64 lr5e-4 | 52.01 | −6.26 | 0/8 |
| `asformer` | 4 | h64 lr1e-4 | 51.94 | −5.38 | 0/8 |
| `gru`（full_sequence，因果） | 1 | h128 lr5e-4 | 51.49 | −6.48 | 0/8 |
| `clean_mstcn_bilstm` | 2 | h256 lr5e-4 | 49.66 | −6.99 | 0/8 |

- **16 对 seed 复核**（最好三族各补 8 seed）：`mstcn` **54.41（−3.58，0/16，p=3.1e-05）**。
- **"欠训练"解释被同预算对照否掉**：挑战者与参考都给 **180 轮**——参考 **57.60 vs 57.59（无增益，
  60 轮即收敛）**；唯一受益者 `transformer` lr5e-5 从 51.53 → **53.19**，仍差 4.4pp。
  调度变体（warmup10+cosine@120 轮）transformer 53.41 / 参考 57.01 → **两边都吃不到红利**。
- **族内调参收益落在 seed 噪声里**：逐 seed Δ 最宽达 **[−10.2, +12.5]**，且"中位差"与"配对中位差"
  两个估计量在 `mstcn` 上**符号相反**（+0.06 / −1.40）。
- **附带发现**：唯一接近参考的恰是**同为 TCN 族的单阶段 `mstcn`** → **"多阶段精化"这个结构本身值 ≈3.6pp**。
- 两个**此前从未训练过**的架构（`clean_asformer` / `clean_bigru`）已补训，落在同档（53.24 / 52.01），
  不改变结论；`gru` 补了 full_sequence 口径的 8-seed 点（此前只有 3 seed 且是滑窗口径）。

**结论**：**架构不是杠杆**——在 **14 训练视频 / 9,575 帧 + `roi-grid-v4` 契约**下，给每族自己的调参机会
与 3 倍预算后，**没有任何一族的任何配置点接近 `mstcn2`**。**仍未覆盖**：更大的联合搜索空间
（warmup × 权重衰减 × 层数）、预训练/迁移路线、换契约后架构排序是否翻转。
