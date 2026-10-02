# 实验报告：帧准确率攻关第 9 轮 —— 补上缺失的学习率调度，并量化"测量精度"问题（2026-09-29）

> ## ⚠️ 更正声明（2026-09-29 第 10 轮追加）
> 本报告 §2.1 的两条结论**已在 32-seed 复核中被推翻**，阅读时以第 10 轮报告为准
> （[`EXPERIMENT_REPORT_FRAME_ACC_R10_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R10_20260929.md) §3）：
>
> | 本报告原结论 | 32-seed 复核 | 状态 |
> |---|---|---|
> | "warmup+cosine 中位 58.62 vs 恒定 lr 58.24，配对 +0.59pp" | 32 seed：**58.20 vs 58.31**，配对 +0.36pp、**17/15**、p=0.81 | ❌ **撤回**（16-seed 偶然） |
> | "调度把 seed 摆幅从 6.97 压到 5.23" | 32 seed：摆幅 **11.98 vs 9.66**（反而更大） | ❌ **撤回** |
> | "须 ≥16 seed" | 16-seed 中位 58.62 → 32-seed **58.20** | ⚠️ **升级为"须 ≥32 seed"** |
>
> **仍然成立的部分**：§1（流水线原先没有学习率调度，这个基础设施缺口是真的）、§2.2
> （8-seed 中位数噪声与目标差距同量级）、以及三个新配置项本身（默认 `constant` 逐位等同历史行为，有单测）。
> **§2.1 的调度增益数字一律不要引用。**


> **目标**：`acc` 53.92 → 58.92（+5pp，严格口径 = 单 seed、仓库标准评测）。
> **本轮两件事**：
> ① 发现全序列流水线**全程恒定学习率、没有任何调度器**，补上可选的 **warmup + cosine**；
> ② 为判断这 +0.6pp 是真是假，把 **seed 数从 8 扩到 16**，顺带量出一个更要紧的事实——
> **8-seed 中位数本身的噪声就有 ±0.7~1.3pp，与"距目标还差多少"同一量级。**

---

## 1. 发现：训练循环里没有学习率调度

`framework/cleansight_eval/temporal/full_sequence_pipeline.py` 原先只做：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3), ...)
```

`history.csv` 的 `lr` 列**每轮都是 0.0005**——**没有 warmup、没有衰减、没有任何调度**。
而本项目的所有配方都是"固定 lr + 固定 60 轮"，这在小数据（14 视频 / 9,575 帧）上是一个
明显缺失的标准手段。

**本轮补上（默认不改既有口径）**：

| 配置 | 取值 | 语义 |
|---|---|---|
| `train.lr_schedule` | `constant`（默认）/ `cosine` | 默认**逐位等同历史行为**（倍率恒 1.0，已用单测钉住） |
| `train.warmup_epochs` | 整数，默认 0 | 前若干轮从 `lr/warmup` 线性升到 `lr` |
| `train.min_lr_ratio` | [0,1]，默认 0 | 余弦衰减下界比例 |

实现要点：倍率写成**只依赖轮号的显式函数**（`lr_factor`），而非 `LambdaLR` 状态机——
这样 **resume 后自动落在同一条曲线上**，不需要序列化调度器状态。
衰减区间口径固定为 `(warmup_epochs, epochs]`（第 `warmup_epochs` 轮为峰值、末轮恰为下界），
已写进 docstring 并由单测固定。**新增单测 4 条**（`framework/tests/test_lr_schedule.py`）。

## 2. 测量：把 seed 从 8 扩到 16（否则判不了这 +0.6pp）

8 seed 时 warmup+cosine 相对恒定 lr 是 **+0.62pp（6/2，p=0.46 不显著）**，且 **val 侧反而不支持**
（val_edit 峰值 68.09 vs 71.14）。这种量级必须加统计功效才能判，于是**两条臂各补 8 个新 seed**
（11,13,17,19,23,29,31,37），并**把基线也补齐**（否则 16-seed 对照不成立）。

### 2.1 16-seed 口径（同一批 16 个 seed，逐 seed 配对）

| 臂 | 8seed A | 8seed B | **16seed 中位** | seed 摆幅 |
|---|---:|---:|---:|---:|
| `roi-grid-v1`（基线，144 维） | 53.92 | 54.66 | **54.34** | 5.26 |
| `roi-grid-v4`（128 维） | 57.59 | 58.92 | **58.24** | 6.97 |
| **`roi-grid-v4` + warmup3 + cosine** | 58.22 | 58.62 | **58.62** | **5.23** |

配对检验（n=16，逐 seed）：

| 对比 | Δ 中位 | 胜/负 | p |
|---|---:|---|---:|
| `v4` vs 基线 | **+3.94pp** | **16/0** | **0.0004** |
| **`v4`+warmup+cosine vs 基线** | **+3.96pp** | **16/0** | **<0.0001** |
| `v4`+warmup+cosine vs `v4`（调度本身） | +0.59pp | 11/5 | 0.3258（不显著） |

**结论**：
- **调度本身的效果未获证实**（+0.59pp，p=0.33），但**它把 seed 间摆幅从 6.97 压到 5.23**——
  这是可复现的**稳定性**收益，值得作为默认的候选（同 lr 下更难训崩）。
- **v4 特征契约的收益极其稳固**：16/16 胜、p=0.0004。

### 2.2 本轮最重要的发现：**8-seed 中位数不可用于 <1pp 的判断**

同一份配方、只换 seed 集合：

| 臂 | seed 集 A | seed 集 B | 差 |
|---|---:|---:|---:|
| 基线 v1 | 53.92 | 54.66 | **+0.74** |
| v4 | 57.59 | 58.92 | **+1.33** |

**同一配方、只换 8 个 seed，中位数就能差 1.3pp。**
即 8-seed 中位数的标准误约 **±0.7~1.3pp**，与"距 +5pp 目标还差多少"**完全同一量级**。
→ **纪律：本项目此前所有 8-seed 单点比较，凡差异 <1.5pp 的都不足以判读**；
要判 1pp 量级必须 ≥16 seed（本轮起已把 B 组 seed 补进 `runs/acc-push/*-seedB/`）。

## 3. 修正后的状态（严格口径，like-for-like）

| 口径 | 基线 | 最好 | Δ | 距 58.92 |
|---|---:|---:|---:|---:|
| 单 seed（8 seed A，历史口径） | 53.92 | 58.22 | +4.30 | −0.70 |
| **单 seed（16 seed，本轮标准）** | **54.34** | **58.62** | **+4.28** | **−0.30** |
| 集成 + val 选点 mindur15（8 seed A） | 54.26~55.40 | **59.23** | +3.83~+4.97 | ✅/❌ 视基线口径 |

⚠️ **必须同时读的三点**：
1. **基线本身也随 seed 集浮动**（53.92 / 54.66），所以"提升几个点"取决于用哪套 seed 报基线；
2. 全栈 59.23 里含**推理侧**成分（8 模型集成 + val 选点平滑）；
3. 上一版把它写成 +5.23pp，用的是**8-seed A 的基线 53.92**；按 16-seed 同口径重算后，
   **严格口径的诚实读数是 +4.28pp，仍差 0.30pp**。

## 4. 本轮否定/未证实的项

| 项 | 结果 |
|---|---|
| `cosine`（无 warmup） | 57.07（vs v4 57.60），**−0.46pp，3/5，p=0.64** → 无 warmup 的纯余弦更差 |
| 调度本身 | +0.59pp，11/5，**p=0.33 不显著**（但方差显著下降） |
| 源域校准叠加在 cosw3 上 | 58.05（vs argmax 58.22）→ **−0.17pp**；第 7 轮它在 v4 上是 +0.64pp → **这类 <1pp 的后处理本身也不稳** |

## 5. 复现

```bash
source ../CleanSightBackend/.venv/bin/activate
# 调度臂（A 组 + B 组各 8 seed）
python tools/run_capacity_matrix.py --runs-dir runs/acc-push/acc-roiv4-cosw3 \
  --config mstcn-actionmixed-auto-roi.yaml --hidden 128 --seeds 42,7,2026,1,2,3,4,5 \
  --best-metric val_edit --epochs 60 --no-probe-baseline --set train.lr=0.0005 \
  --set train.lr_schedule=cosine --set train.warmup_epochs=3 --set train.min_lr_ratio=0.05 \
  --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
  --set model.input_dim=128 --set feature_schema.dim=128 \
  --set feature_schema.version=actionmixed-roi-grid-v4 \
  --set data.dataset_ref=temporal.actionmixed-auto-roi-v4
pytest framework/tests/test_lr_schedule.py -q     # 4 条
```

**新增产物**：`train.lr_schedule/warmup_epochs/min_lr_ratio`（流水线 + config schema +
`usage/YAML_CONFIG.md` 登记）；单测 `framework/tests/test_lr_schedule.py`；
训练臂 `runs/acc-push/{acc-roiv4-cos,acc-roiv4-cosw3,acc-roiv4-seedB,acc-roiv4-cosw3-seedB,acc-roiv1-seedB}/`。

**验证**：全量回归 **473 passed / 7 failed**（7 项均为预先存在，与改动前逐项一致）。

**已知限制**：test 仅 8 视频 / 2639 帧、val 仅 4 视频；§2.2 的测量精度问题对**本报告所有数字**都适用。
