# 投票集成与拼接契约（2026-10-10）

> **本文件是过程文档**（生命周期 ≤ 1 周），长期有效的结论会折进
> [`../../../docs/experiments/`](../../experiments/) 与 `runs/v41-push/` 的产物。
> 目标：在 **v4.1 尺子**（15fps + 标签时间轴 ×1.25 + train 28 / val 7 / test 8）上提升时序模型的
> **帧准确率 `acc`** 与**编辑分 `edit`**。起点 = `runs/v41/` 的 22 臂 × 8 seed 全量重训读数
> （acc 64.91~76.27、edit 55.09~69.43）。

## 0. 结论速查

| # | 候选 | test 读数 | val 读数 | 判定 |
|---|---|---|---|---|
| 1 | **同契约 8-seed 多数投票** | acc **+1.50（9/11 臂为正）**、edit +0.40（7/11 为正） | acc **+2.27（11/11 臂为正）**、edit +3.32（7/11 为正） | ✅ **acc 有效杠杆**（val 与 test 同向、val 全为正）；edit 中位为正但两 split 常一正一负 → 不稳健 |
| 2 | 同契约投票 + 最小时长合并（d=5） | edit +3.88（7/11）、acc +1.44 | edit +1.70（7/11）、acc +2.28（11/11） | ⚠️ 合并把 test edit 从 +1.54 抬到 +3.88，但 val 上只有 +1.70 → 方向一致、幅度在噪声量级 |
| 3 | **跨契约两两投票**（各 4 seed） | 相对"较强父契约单 seed 中位"：acc +0.23（28/55，p=0.78）、edit +0.34（30/55，p=0.84） | acc **+2.44（62/66，p=8.8e-12）**、edit **−3.02（17/66，p=3.5e-07）** | ❌ **test 上的大 edit 增益（+9~13pp）不可采信**：val 上 edit 反向 |
| 4 | 拼接契约 `combo-v2`（272 维） | acc 75.34 / edit 68.64（8 seed） | acc **75.53**（13 臂最高）/ edit 71.75 | ⚠️ 与两个父契约**不可分**，未同时胜出；稀类召回下限 28.3 → 40.7 |
| 5 | 在线轨 `window` 16 → 32（15fps 后物理窗口减半） | acc 58.82 → **55.00**、edit 49.35 → 49.05（8 seed 配对） | — | ❌ **假设被否定**：加长窗口不补在线轨的 16~17pp 差距 |

一句话：**本轮唯一在 val 与 test 上同向、且统计上站得住的改进是「帧准确率」上的多成员投票
（val +2.3pp，11/11 臂）**；`edit` 的诱人读数全部来自 test 口径或 test 选点，val 不支持。

## 1. 口径（先说清楚，否则数字会被误读）

- **数据**：v4.1，revision `4aa9bf31a247…`；test 8 视频 / 5,272 帧，val 7 视频。
- **模型**：`mstcn2` s4l10 h128（3.31M 参），60 轮 / lr 5e-4 / wd 1e-4 / `best_metric=val_edit` /
  早停关闭，每臂 8 seed = {1,2,3,4,5,7,42,2026}。
- **指标**：`acc` = 逐帧 micro accuracy（合并全部视频帧）；`edit` = **逐视频段标签序列的
  归一化 Levenshtein 相似度，再对视频取 macro 平均**（对边界/时长不敏感，对"多出/漏掉一段"重罚）。
  两者都由 `framework.cleansight_eval.core.metrics.temporal_metrics` 复算，**与已存盘
  `evaluation.json` 逐位一致**（已核对：roi-v5 seed 1 acc 69.822/edit 72.989 ↔ 存盘 69.82/72.99）。
- **两套推理协议，不可混用**：
  - **单 seed 中位**（严格口径）：8 个独立 checkpoint 各自评测，取中位；
  - **多成员投票**（全栈口径）：成员逐帧多数投票后评测（平局取标签 id 较小者）。
  本文件所有"Δ"都注明相对哪个对照。
- **val 复评**：`benchmark.cli.eval` 原本只评 test；本轮把每个 run 的 `config.resolved.json`
  复制成 val 变体（只改 `data.split_eval`）复评 val，checkpoint 用**硬链接**放到
  `tmp/v41push/val_runs/` 以免把 `best.eval.md` / `EVALUATION_REPORT.md` 写回原 run。

## 2. 候选 1/2：同契约 8-seed 多数投票

| 臂 | test 单seed中位 acc | test 投票 acc | Δ | test 单seed中位 edit | test 投票 edit | Δ | val 单seed中位 acc | val 投票 acc | Δ | val 单seed中位 edit | val 投票 edit | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| feat-global-hand | 76.27 | 78.02 | +1.75 | 64.14 | 61.09 | -3.05 | 72.36 | 74.08 | +1.72 | 72.29 | 75.61 | +3.32 |
| feat-hand | 75.98 | 76.33 | +0.34 | 62.44 | 62.81 | +0.37 | 68.26 | 69.42 | +1.16 | 75.06 | 74.07 | -1.00 |
| feat-roi-v5 | 74.53 | 75.99 | +1.46 | 69.43 | 75.47 | +6.04 | 73.01 | 75.10 | +2.09 | 72.44 | 69.40 | -3.04 |
| feat-roi-v3 | 72.61 | 76.08 | +3.47 | 62.86 | 63.26 | +0.40 | 68.34 | 70.66 | +2.32 | 70.31 | 75.77 | +5.46 |
| feat-combo | 70.48 | 69.95 | -0.53 | 61.89 | 70.92 | +9.03 | 74.68 | 76.68 | +2.00 | 72.48 | 70.92 | -1.56 |
| feat-roi-v1 | 69.88 | 72.72 | +2.85 | 63.27 | 61.49 | -1.79 | 69.82 | 71.97 | +2.15 | 73.36 | 76.95 | +3.59 |
| feat-bbox40 | 69.38 | 72.91 | +3.54 | 62.30 | 68.12 | +5.82 | 72.67 | 75.21 | +2.55 | 75.39 | 80.76 | +5.37 |
| feat-roi-presence | 68.47 | 70.18 | +1.71 | 64.87 | 68.43 | +3.57 | 64.81 | 68.36 | +3.55 | 72.64 | 69.65 | -2.99 |
| feat-roi-delta | 67.26 | 68.17 | +0.91 | 62.94 | 61.18 | -1.76 | 70.72 | 73.07 | +2.36 | 74.93 | 76.27 | +1.35 |
| feat-roi-v2 | 65.04 | 64.81 | -0.23 | 63.12 | 62.22 | -0.90 | 70.51 | 72.78 | +2.27 | 73.93 | 80.75 | +6.82 |
| flagship-v41 | 64.91 | 66.41 | +1.50 | 62.64 | 64.17 | +1.53 | 70.24 | 73.48 | +3.24 | 74.55 | 78.47 | +3.93 |

**读数**：

- **acc**：test 9/11 臂为正（中位 +1.50），**val 11/11 臂为正（中位 +2.27）**。两个 split 同向
  → 这是本轮唯一"跨 split 一致"的杠杆。注意 val 的增益更大，说明 test 那批（project-18 长短毛刷专项）
  对集成不敏感。
- **edit**：test 7/11、val 7/11 为正（中位 +0.40 / +3.32），且**同一臂在两个 split 上常常一正一负**
  （roi-v5 +6.04 / −3.04；combo +9.03 / −1.56；global-hand −3.05 / +3.32）→ 与噪声不可分。
  **不应把某臂的 test edit 增益当作结论**。
- 机制解释：投票压制了个别 seed 的"多发段/漏发段"错误，对**帧级**判对率是稳定的正向；
  对**段标签序列**则取决于"多发段"是否恰好被多数成员压掉——这是不稳定的，故 edit 不稳健。

## 3. 候选 3：跨契约投票 —— test 上的大增益是**假象**

跨契约投票（每臂取 4 seed，共 8 成员）在 **test** 上出现了很漂亮的数字，例如
`roi-v5 + flagship` edit **78.95**、加 d=5 合并后 **82.60**（成员中较强的单 seed 中位是 69.43）。
但如果把同一批协议搬到 **val**：

| 对照 | test | val |
|---|---|---|
| Δ acc（相对"较强父契约的单 seed 中位"） | +0.23（28/55 正，p=0.78） | **+2.44（62/66 正，p=8.8e-12）** |
| Δ edit（同上） | +0.34（30/55 正，p=0.84）；加 d=5 后 +2.37（35/55，p=0.042） | **−3.02（17/66 正，p=3.5e-07）**；加 d=5 后 −1.21（19/55 正） |
| Δ acc（相对成员平均） | +6.88（53/55 正） | +1.44（59/66 正） |
| Δ edit（相对成员平均） | +0.47（34/55 正，p=0.24） | −1.57（22/66 正，p=0.0006） |

**判定**：跨契约投票**没有**比"同契约投票"多给出可交付的东西——

1. **edit**：val 上显著为负（−3.02，仅 17/66 个组合为正；加最小时长合并后仍为 −1.21，19/55）→ 按本仓库纪律
   （"test 涨、val 不支持一律不计交付"）**不采纳**；
2. **acc**：val 上 +2.44 显著，但 test 上相对较强父契约只有 +0.23（不显著）——
   即"用两个契约的成员投票"与"就用较强那个契约自己投票"在 test 上**打平**，而后者更省一半算力。
3. **协议排名的可迁移性也很差**：55 个跨契约组合的 test 与 val 读数 Spearman ρ 仅
   **+0.058（acc）/ +0.121（edit）**（Δ 口径 ρ = +0.353 / −0.085）→ **不能用 val 选出"最好的那一对"**，
   而按 test 选就是选点泄漏。这条与 v4.1 简报记录的"契约排名不迁移（ρ=−0.149）"互为印证。

> 方法学提醒：跨契约/大成员的"Δ 相对成员平均"很容易做出 +6~7pp 的漂亮数字，但那是因为
> 成员池里混进了较弱的成员（例如把 roi-v1 的 4 个 seed 混进来）。**对照必须写成
> "相对池内较强成员的单 seed 中位"**，否则会把"平均掉了弱成员"说成"集成有增益"。

## 4. 候选 4：拼接契约 `combo-v2`（单模型路线）

**立项**：v4.1 上没有全能契约——`roi-v5` edit 最高（69.43）且四类召回都 ≥0.55，
`bbox-global-hand` acc 最高（76.27）但 flush 0.239 / sbc 0.276；`combo-v1`（v4 ‖ boxset）
比其父契约 roi-v4 高 +5.57pp（v3 时代"拼接无增量"的结论未迁移）。
故把"最强 edit 契约 ‖ 最强 acc 契约"拼进一个主干：

- 左 `roi-grid-v5`（192）‖ 中 全画面 bbox 40 维 ‖ 右 手部区域 bbox 40 维 = **272 维**
  （`features/combo_v2.py`，登记 `temporal.actionmixed-auto-combo-v2-v41`）；
- 公式/骨架/lr/轮数/选点口径与 `mstcn2-auto-roi-v4-v41.yaml` **逐项一致**，唯一变量 = 契约与 `input_dim`。

**读数（8 seed 中位）**

| 契约 | 维度 | test acc | test edit | frame macro_f1 | f1@0.25 | val acc | val edit |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bbox-global-hand`（acc 父） | 80 | **76.35** | 64.45 | 62.83 | 65.77 | 72.36 | 72.29 |
| `roi-grid-v5`（edit 父） | 192 | 74.53 | **69.43** | **68.84** | **67.28** | 73.01 | 72.44 |
| **`combo-v2`（本轮）** | **272** | 75.34 | 68.65 | 68.81 | 64.76 | **75.53** | 71.75 |
| `roi-v4`（v4.1 旗舰） | 128 | 64.91 | 62.64 | 56.96 | 56.56 | 70.24 | 74.55 |

**逐类帧召回（8 seed 中位，%）** —— 这张表比总表更能说明拼接到底换到了什么：

| 契约 | idle | flush | long_brush_insert | long_brush_withdraw | short_brush_cleaning |
|---|---:|---:|---:|---:|---:|
| `bbox-global-hand` | 89.2 | 30.1 | **82.0** | 61.1 | **28.3** |
| `roi-grid-v5` | 90.4 | **55.9** | 67.6 | 55.6 | 56.3 |
| **`combo-v2`** | **91.4** | 40.7 | 70.1 | **60.0** | **66.9** |
| `roi-v4`（旗舰） | 86.9 | 36.8 | 51.0 | 39.3 | 52.4 |

即：`combo-v2` 把 `global-hand` 最弱的 `sbc`（28.3 → **66.9**）与 `roi-v5` 最弱的 `insert`
（67.6 → 70.1）各补了一部分，**五类召回的下限从 28.3 抬到 40.7**，代价是 `flush`
（55.9 → 40.7）与 `insert` 的峰值回落到两者之间。

**配对检验**（`tools/compare_runs.py`，逐 (seed, 视频) × 类 Wilcoxon）

| 对照 | edit 中位差 | p | f1@0.25 | p | acc（逐 seed 配对） |
|---|---:|---:|---:|---:|---:|
| vs `roi-grid-v5` | +3.33 | 0.559 | +0.57 | 0.836 | +0.81（中位差） |
| vs `bbox-global-hand` | **+7.69** | 0.328 | **+6.41** | 0.091 | −0.93 |

**判定**：`combo-v2` **落在两个父契约之间**，没有出现"同时拿到 acc 与 edit"的协同：

- vs `roi-v5`：acc +0.81 / edit −0.78 / macro_f1 −0.03 → **不可分**（与 v3 时代
  combo-v1 vs roi-v4 的结论同型）；
- vs `global-hand`：段级指标明显更好（edit +7.69、f1@0.25 +6.41）但 acc −0.93 → 换到的是
  "段结构 + 稀类召回"，不是"帧判对率"；
- 它在 **val acc 上是最高的一个臂（75.53）**，但 test acc 仍低于 `global-hand`——又一次
  val/test 排序不一致的例子。

**结论**：拼接路线在本轮**未证明有增益**（更准确说是"不可分"，判 <1.5pp 的差异需 ≥16~32 seed）；
唯一站得住的定性收益是**稀类召回的均衡性**（五类召回下限 28.3 → 40.7）。若要继续这条线，
应当先做**源域筛选**（train 侧视频级留出探针）而不是直接上 8 seed 训练。

## 5. 候选 5：在线轨 window 重定标（15fps 后物理窗口减半）—— **不成立**

**立项**：v4.1 把采样率从 7.5fps 提到 15fps，**同一份超参在时间轴上的语义变了**：
在线 GRU 轨的 `train.window: 16` 从 2.13 s 变成 **1.07 s**（感受野物理时长减半）。
上周简报把"超参重定标"列为第一候选变量。本轮做单变量对照：**只把 `window` 从 16 改成 32**
（= v3 时代的物理窗口），契约（`roi-grid-v4` 128 维）、`model.num_layers=3`、lr、轮数、
选点口径、seed 集全部不变，8 seed 配对。

**读数（test，8 seed，best.pt）**

| 臂 | acc 中位 | edit 中位 | F1@0.1 | F1@0.25 | F1@0.5 | 段数比 | 非 idle 帧 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `online-gru-roi-v4`（window 16，1.07 s） | **58.82** | 49.35 | 55.46 | 47.70 | 31.28 | 0.51 | 2758 |
| `online-gru-roi-v4-w32`（window 32，2.13 s） | 55.00 | 49.05 | 53.97 | 46.40 | 26.99 | 0.51 | 2154 |

**逐 seed 配对（8 对）**：Δacc 中位 **−2.61**（5/8 为负）、Δedit 中位 **+0.79**（5/8 为正）；
`tools/compare_runs.py` 的逐 (seed,视频)×类 检验：edit +7.69（p=0.43）、f1@0.25 −0.38（p=0.54）、
f1@0.5 −1.58（p=0.85）——**段级指标不可分**。

**判定**：**窗口加倍不解决问题**，acc 反而略降（在 seed 摆幅内）→
"在线轨弱是因为 15fps 后窗口太短"这个假设**被否定**。在线轨（58.8）与离线轨（74.5~76.3）
之间 16~17pp 的差距**不能靠加长窗口补**，更可能来自因果约束（只能看历史 + 逐窗口末帧决策）
与训练样本构造本身。下一步应转向"**用离线教师蒸馏一个因果学生**"或"改滑窗推理协议"，
而不是继续扫 window。

## 6. 复现

```bash
source ../CleanSightBackend/.venv/bin/activate

# ① 把一批已训练 run 在 val 上复评（不训练；checkpoint 走硬链接，原 run 不被改动）
python tools/eval_runs_on_split.py --split val \
  --run "runs/v41/feat-hand/h128/mstcn2-*" --workers 3 \
  --out-root tmp/v41push/val_evals --work-root tmp/v41push/val_runs

# ② 同契约 / 跨契约投票（读已存盘产物，不训练）
python tools/probe_cross_contract_vote.py \
  --arm feat-roi-v5="runs/v41/feat-roi-v5/h128/mstcn2-*" \
  --arm feat-global-hand="runs/v41/feat-global-hand/h128/mstcn2-*" \
  --arm flagship-v41="runs/v41/flagship-v41/h128/mstcn2-*" \
  --json tmp/v41push/vote_test.json
python tools/probe_cross_contract_vote.py --source val --merge-d 5 \
  --arm feat-roi-v5="tmp/v41push/val_runs/feat-roi-v5-*" \
  --arm feat-global-hand="tmp/v41push/val_runs/feat-global-hand-*" \
  --json tmp/v41push/vote_val.json

# ③ combo-v2 训练（8 seed，与 v4.1 各臂同配方）
python tools/run_capacity_matrix.py --runs-dir runs/v41-push/combo-v2 \
  --config mstcn2-auto-combo-v2-v41.yaml --hidden 128 \
  --seeds 42,7,2026,1,2,3,4,5 --epochs 60 --best-metric val_edit --no-probe-baseline

# ④ 配对对照
python tools/compare_runs.py --left "runs/v41-push/combo-v2/h128/mstcn2-*" --left-label combo-v2 \
  --right "runs/v41/feat-roi-v5/h128/mstcn2-*" --right-label roi-v5 --metrics acc,edit,f1_025,f1_01
```

产物：`runs/v41-push/combo-v2/`（8 run + `CAPACITY_SUMMARY.md`）、`tmp/v41push/`（val 复评产物与
JSON 结果；`tmp/` 为 scratch 可删）。**两个新增工具已入库**：
[`tools/eval_runs_on_split.py`](../../../tools/eval_runs_on_split.py)（任意 split 复评，
隔离原 run）与 [`tools/probe_cross_contract_vote.py`](../../../tools/probe_cross_contract_vote.py)
（同契约/跨契约投票探针，对照口径写死为"较强父契约的单 seed 中位"）。

> 踩坑记录（值得记住）：**第一次 val 复评直接把 --ckpt 指向原 run 的 checkpoint**，
> 结果 `benchmark.cli.eval` 把 val 版 `best.eval.md` / `EVALUATION_REPORT.md` **写回了原 run**，
> 并且把 val 的 predictions artifact 落进了原 run 的 `artifacts/`（含 7 个视频，与 test 的 8 个混在一起，
> 会让"取最新产物"的分析脚本读错 split）。已删除错产物并**重跑该 run 的 test 评测**恢复两份报告
> （现为 test 口径，acc 80.52 / edit 68.80，与该 run 原读数一致）。`tools/eval_runs_on_split.py`
> 用硬链接从机制上避免这类污染。
