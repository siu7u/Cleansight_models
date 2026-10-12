# 在线轨换契约 + 新契约上的投票协议（2026-10-11）

> **本文件是过程文档**（生命周期 ≤ 1 周）。两个主题都属于"把离线侧已验证的杠杆搬到别处"：
> ① 在线（滑窗因果）轨此前跑的全是**离线读数最差**的契约；② 投票协议在 round-1 只在 `acc`
> 上站得住，在新契约上要重测。

## 1. 在线轨换契约（进行中）

**问题**：v4.1 的三个在线臂用的契约，在**离线**读数上恰好是最差的一档——

| 在线臂（`sliding_window_temporal` + gru 3 层 + `window:16`） | 契约 | 同契约的**离线** test acc / edit |
|---|---|---:|
| `online-gru-roi-v4`（在线最好，acc 58.82 / edit 49.35） | `roi-grid-v4` 128 维 | 64.91 / 62.64（离线垫底） |
| `online-gru-bbox40`（55.45 / 41.52） | 全画面 bbox 40 维 | 69.38 / 62.30 |
| `online-gru-roi-v1`（50.45 / 47.65） | `roi-grid-v1` 144 维 | 69.88 / 63.27 |

而 v4.1 上把离线 acc/edit 拉起来的两条契约（`hand-v2` 80→77.75、`global-hand-v2` 79.15）
**从未在在线轨上训练过**。本批把在线轨换到这两条 v2 手部契约，其余逐项不变
（`gru` 3 层 / `window:16` / 60 轮 / `val_edit` 选点 / 同一 seed 集），产物：

- `runs/v41-push/online-gru-hand-v2/`（40 维）
- `runs/v41-push/online-gru-global-hand-v2/`（80 维）

**在线-离线差距的分布（换契约之前）**：不是"只丢稀类"，而是整体平移——

| 类 | 在线 `roi-v4` | 离线 `global-hand-v2` | 差 |
|---|---:|---:|---:|
| idle | 69.9 | 90.9 | −21.0 |
| flush | 48.8 | 51.2 | −2.4 |
| long_brush_insert | 53.9 | 81.1 | −27.2 |
| long_brush_withdraw | 44.6 | 66.0 | −21.4 |
| short_brush_cleaning | 17.7 | 50.4（`combo-v3`） | −32.7 |
| **acc / edit** | **58.82 / 49.35** | **79.15 / 70.41** | **−20.3 / −21.1** |

（round-1 已经用 8 seed 配对否掉"窗口太短"这条解释：`window:16`→`32` 反而 acc −2.61。）

### 1.1 结果（`online-gru-hand-v2`，8 seed；`best.pt`）

| 在线臂 | 契约 | test acc | test edit | f1@0.1 | f1@0.25 | f1@0.5 | `last.pt` acc/edit |
|---|---|---:|---:|---:|---:|---:|---:|
| `online-gru-roi-v4`（旧最好） | `roi-grid-v4` 128 | 58.82 | 49.35 | 55.46 | 47.70 | 31.28 | 60.48 / 46.95 |
| **`online-gru-hand-v2`** | **`bbox-hand-v2` 40** | **59.53** | **55.14** | **56.90** | 43.97 | 32.00 | **64.06 / 55.20** |
| `online-gru-global-hand-v2` | `bbox-global-hand-v2` 80 | 59.84 | 48.29 | 56.57 | **48.56** | 31.28 | 59.89 / 47.83 |

配对检验（`online-gru-hand-v2` − `online-gru-roi-v4`，逐 (seed,视频)×类）：
**edit +9.40pp（30/12，p=0.0042）**、**f1@0.1 +15.02（37/10，p=0.0001）**、
**f1@0.5 +11.26（32/15，p=0.0002）** 均显著；f1@0.25 +6.28（p=0.23）。
→ **只换契约（其余逐项不变）就让在线轨的 edit 显著上一个台阶**，acc 小幅 +0.71。

**注意 80 维并不更好**：离线 acc 冠军契约（`global-hand-v2` 80 维，离线 79.15）
在在线轨上只有 59.84 / 48.29，与 128 维的 `roi-v4` 基本持平；**在线轨偏好最省的 40 维手部契约**。
（一个合理解释：滑窗只喂 16 帧，参数量/维度更大的模型在同样 60 轮里更难拟合，
但本批没有做容量对照，这只是假设。）

### 1.2 附带发现：在线轨的 `val_edit` 选点选错了 checkpoint

`best.pt` 由 val（7 视频）的 `val_edit` 选出、`last.pt` 是固定 60 轮的终点。四个在线臂：

| 在线臂 | best acc/edit | last acc/edit | Δacc | Δedit |
|---|---:|---:|---:|---:|
| `online-gru-hand-v2`（本轮） | 59.53 / 55.14 | **64.06 / 55.20** | **+4.53** | +0.06 |
| `online-gru-roi-v4` | 58.82 / 49.35 | 60.48 / 46.95 | +1.66 | −2.40 |
| `online-gru-roi-v1` | 50.45 / 47.65 | 55.22 / 42.64 | +4.77 | −5.01 |
| `online-gru-bbox40` | 55.45 / 41.52 | 56.53 / 43.82 | +1.09 | +2.30 |

**四个臂的 `last.pt` 在 acc 上全部优于 `best.pt`（+1.09 ~ +4.77）**，而 edit 有正有负；
本轮最好的臂上两者 edit 基本持平（55.20 vs 55.14）。
→ 在线轨上 `val_edit` 选点**系统性压低 acc 且不换来 edit**（与离线轨相反：离线 23 臂的
best−last 在 acc 上中位仅 +0.26）。

**可交付读数（在线轨）**：`online-gru-hand-v2` 的 `last.pt` = **acc 64.06 / edit 55.20**，
相对旧最好在线臂（58.82 / 49.35）是 **+5.24 / +5.85**。
`last.pt` 属于"固定预算、不经 val 选点"的协议，规则本身不看 test，故不构成选点泄漏；
反过来用 val 选点在这条轨道上已被证明是噪声（EVAL.md §9.2：val 侧 ρ≈0.2）。

## 2. 新契约上的投票协议（零训练成本，已完成）

round-1 的结论是"同契约多成员投票只在 `acc` 上站得住，`edit` 两 split 常一正一负"。
在三条新契约上重测（16 成员投票 = 该契约的全部 seed）：

| 契约 | split | 单 seed 中位 acc / edit | 16 成员投票 acc / edit | Δacc | Δedit |
|---|---|---:|---:|---:|---:|
| **`global-hand-v2`（80）** | test | 79.15 / 66.37 | **80.65 / 76.24** | +1.51 | **+9.87** |
| `global-hand-v2` | val | 75.18 / 75.15 | 76.23 / 75.89 | +1.05 | +0.74 |
| **`combo-v3`（232）** | test | 77.40 / 70.41 | **80.10 / 73.40** | +2.70 | +2.99 |
| **`combo-v3`** | val | 77.91 / 75.78 | **79.93 / 78.78** | +2.03 | **+3.00** |
| `hand-v2`（40） | test | 77.75 / 66.35 | 78.93 / 68.53 | +1.18 | +2.18 |
| `hand-v2` | val | 77.74 / 76.36 | 78.78 / 73.19 | +1.04 | −3.17 |

**读数**：

1. **`combo-v3` 的 16 成员投票在 val 与 test、acc 与 edit 四个格子上全部为正**
   （+2.03 / +3.00 / +2.70 / +2.99）——这是本 session 第一次出现"投票协议在两个 split 上
   两个指标同时提升"，round-1 的"edit 不稳健"结论**在新契约上不再成立**。
   落到读数上：**test 80.10 / 73.40、val 79.93 / 78.78**（16 成员）。
2. **`global-hand-v2` 的投票把 test edit 从 66.37 抬到 76.24（+9.87）**，但 val 上 edit 只 +0.74
   → 与 round-1 的老毛病同型（test 上的 edit 增益可能不迁移），**引用时必须两个 split 一起给**。
3. **跨契约投票仍然不行**：`combo-v3 + hand-v2` test acc +2.12 但 edit −8.22（1/8 视频为正）；
   `combo-v3 + global-hand-v2` edit −5.77 → **第四次复现**"跨契约的 edit 增益不迁移"。
4. `hand-v2` 的投票在 val 上 edit −3.17 → 单契约投票也**不是**无条件正向，要看契约。

## 3. 复现

```bash
source ../CleanSightBackend/.venv/bin/activate

# 在线轨换 v2 契约（约 1.5 h/臂，8 seed）
bash tmp/v41push/run_online_v2.sh run
#   → runs/v41-push/online-gru-{hand-v2,global-hand-v2}/

# 投票协议（读产物，秒级）
python tools/probe_cross_contract_vote.py \
  --arm combo-v3="runs/v41-push/feat-combo-v3/h128/mstcn2-*" \
  --arm hand-v2="runs/v41-push/feat-hand-v2/h128/mstcn2-*" \
  --arm global-hand-v2="runs/v41-push/feat-global-hand-v2/h128/mstcn2-*"
python tools/probe_cross_contract_vote.py --source val \
  --arm combo-v3="tmp/v41push/val_runs/feat-combo-v3-mstcn2-*" \
  --arm hand-v2="tmp/v41push/val_runs/feat-hand-v2-mstcn2-*" \
  --arm global-hand-v2="tmp/v41push/val_runs/feat-global-hand-v2-mstcn2-*"
```

## 4. 遗留

1. **在线轨的读数与判定**（§1 待补）：若换契约有效，下一步是"离线教师 → 因果学生蒸馏"
   （在线轨每类召回整体低 20~30pp，说明缺的是长时上下文，而不是某类特征）。
2. **投票协议的落地形态**：`combo-v3` 的 16 成员投票是当前最好的协议读数，但它是
   "16 倍推理成本 + 需要 16 个 checkpoint"，是否作为交付口径要单独决策。
