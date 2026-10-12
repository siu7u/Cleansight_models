# 时序后处理（median / mindur）在新契约上还值不值得开（2026-10-12）

> 过程性文档，属本周（2026-10-10 ~ 10-16）。分支：`feat/temporal-sbc`。
> 回应 [`README.md`](README.md) §4 遗留第 5 条：「段数比 1.10 → 0.95 后，'最小时长合并'的收益
> 预期变小，需重跑才知道还值不值得开」。免 GPU，只读已存盘预测产物。

## 0. 一句话结论

**不值得开。** 三个新契约上按**交付口径**（后处理参数在**模型自己的 val 产物**上选）跑下来：
`hand-v2` 与 `global-hand-v2` 的 val 最优就是 **argmax（不做后处理）**；`combo-v3` 选出
`median k=5` 但 val 只涨 **+0.35**，套到 test 是 **edit +0.00 / f1@0.5 −0.20**，逐视频配对
**30 胜 49 负、中位 −5.98、p=0.0205（显著更差）**。

而**只看 test** 会得到完全相反的印象（`global-hand-v2` 的 `median k=15` 把 edit 从 68.42
抬到 **76.47**，配对 p=0.0023 "显著"）——那是**用 test 选参**选出来的，不可交付。
这是本仓库「test 涨、val 不支持不计交付」这条纪律的**第 5 次**复现。

## 1. 读数

### 1.1 test 侧曲线（**用 test 选参**，仅供机制参考，不是交付读数）

16 seed，基线 = 离线逐帧 argmax、零平滑：

| 契约 | 基线 test edit | 最好的 test 设置 | test edit | 逐视频配对 |
|---|---:|---|---:|---|
| `combo-v3` | 71.43 | median k=15 | 76.47 | n=55 胜/负=37/18 p=0.0671 |
| `combo-v3` | 71.43 | mindur d=9 | 76.39 | n=62 胜/负=40/22 p=0.1212 |
| `hand-v2` | 70.00 | mindur d=9 | 72.79 | n=54 胜/负=31/23 p=0.3241 |
| **`global-hand-v2`** | **68.42** | **median k=15** | **76.47** | **n=33 胜/负=25/8 p=0.0023** |
| `global-hand-v2` | 68.42 | mindur d=9 | 76.47 | n=49 胜/负=27/22 p=0.2173 |

单看这张表，"后处理是 +5~+8pp 的 edit 杠杆、且对 `global-hand-v2` 配对显著"的结论很诱人。

### 1.2 交付口径：后处理参数在**模型自己的 val 产物**上选，再套到 test

val 产物来自 `tools/eval_runs_on_split.py` 的复评（`tmp/v41push/val_runs/<contract>-*`，
16 seed × 7 视频；与模型按 val 选 checkpoint 同口径）：

| 契约 | val 最优设置 | val edit（argmax → 最优） | test edit（基线 → 套用） | test f1@0.5 | 逐视频配对 |
|---|---|---:|---:|---:|---|
| `combo-v3` | **median k=5** | 77.43 → 77.78（**+0.35**） | 71.43 → 71.43（**+0.00**） | 48.53 → 48.33（−0.20） | **30 胜 49 负，中位 −5.98，p=0.0205（显著更差）** |
| `hand-v2` | **argmax** | 78.17 → 78.17（+0.00） | 70.00 → 70.00 | 50.00 → 50.00 | —（未改动） |
| `global-hand-v2` | **argmax** | 77.53 → 77.53（+0.00） | 68.42 → 68.42 | 56.25 → 56.25 | n=95，中位 −2.38，p=0.5724 |

**val 上后处理几乎没有空间**（三个契约的 val 最优 treble 在 77.4~78.2 之间，argmax 已经吃到），
而 test 上"看起来很香"的那几档在 val 上都是**更差**的（`global-hand-v2` 的 `median k=15`
val edit 只有 71.96，比 argmax 的 77.53 低 5.57）。

### 1.3 机制

与 §4 遗留的预期一致：手部区域锚点 v2 之后**段数比已从 1.10 降到 0.95**，也就是"碎片段"
基本被模型自己解决了。此时 median/mindur 这类**只为合并短段而存在**的后处理，剩下的作用
主要是**把真实的短动作段吃掉**——`combo-v3` 套 `median k=5` 后 test `f1@0.5` 掉 0.20、
逐视频配对中位 −5.98，正是这个方向。val 侧的 `mindur d=15` 更是断崖（77.43 → 66.67），
说明它已经把长动作段也动到了。

**结论：这条杠杆在本任务的 v4.1 + 手部 v2 盘面上已经关闭**，不必再开。

## 2. 一个工具陷阱（已修）

`tools/probe_offline_postprocess.py --select-on-val` **不是**给模型选后处理参数：它走
`select_threshold_on_val()` → `probe_on_split(..., "split_val")`，算的是**用 train 拟合的
逐帧 LDA 线性探针**在 val 上的曲线，用途是回答"『平滑』与『时序建模』各自值多少"。
直接把它读成"模型在 val 上选了 median k=15"会得到错误结论——本次就差点据此写出
"val 选参导致 test 从 71 崩到 43"（43.30 其实是**探针**的分，不是模型的）。

- 已在本工具新增 `--select-on-val-model`（配合 `--val-root`），用**模型自己的 val 产物**
  做交付口径的选参，并在 `--select-on-val` 的 help 里写明它用的是探针曲线；
- 模型的 val 曲线与 test 曲线现在同表可比（§1.2 即由该 flag 产出）。

## 3. 复现命令

```bash
# 1) 先有模型的 val 复评产物（checkpoint 硬链接隔离，不污染原 run）
python tools/eval_runs_on_split.py --run "runs/v41-push/feat-combo-v3/h128/mstcn2-*" --split val

# 2) 交付口径：val 选参、test 上报（本文件 §1.2 即此命令）
python tools/probe_offline_postprocess.py \
    --run "runs/v41-push/feat-combo-v3/h128/mstcn2-*" --label combo-v3 \
    --select-on-val-model --val-root "tmp/v41push/val_runs/feat-combo-v3-*"

# 3) 对照：test 侧曲线（§1.1，仅供机制参考，不可作交付读数）
python tools/probe_offline_postprocess.py --run "runs/v41-push/feat-combo-v3/h128/mstcn2-*" --label combo-v3
```

## 4. 对当前交付盘面的影响

**没有影响**——后处理不开，`combo-v3` 的交付读数就是既有的
**test 77.40 / 70.41、val 77.91 / 75.78**（16 seed 中位）。本轮的贡献是**把一条候选杠杆
用交付口径关掉**，并修掉一个会导致误判的工具陷阱。
