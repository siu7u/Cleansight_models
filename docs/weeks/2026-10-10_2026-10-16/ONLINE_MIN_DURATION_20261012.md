# 在线轨：一个从未被扫过的硬上限——`smoothing_min_duration`（2026-10-12 摸底）

> 分支：`feat/temporal-online-md`。**本文只做摸底与机制验证，尚未产出在线轨读数**。

## 0. 一句话

在线轨（滑窗因果 GRU）的推理后处理里，`causal_decision(..., min_duration=25)` 的
`min_duration` 是一个**硬召回上限**：候选类必须**连续出现 ≥ min_duration 帧**才会被写成
`stable`，因此**短于该值的真实动作段在输出里根本不可能出现**。
而 **6 个在线臂全部用的是默认值 25**（`evaluation.smoothing_min_duration` 一路为 `None`）——
这是一个**从未被扫过的配置默认值**，而在线轨与离线轨 16~17pp 的差距正是本任务最大的一块未解空间。

## 1. 机制：合成验证（CPU、秒级、可复现）

给一段"后验明确偏向 `sbc` 的 19 帧连续段"（其余为 idle），扫 `min_duration`：

| `min_duration` | 输出里稳定为 sbc 的帧数 |
|---:|---:|
| **25（现行默认）** | **0** |
| 19 | 19 |
| 15 | 19 |
| 10 | 19 |
| 5 | 19 |
| 1 | 19 |

即 `min_duration=25` 下，**即使该类后验连续 19 帧压倒性地高，也一帧都写不出去**；
门槛一降到 ≤19，整段立刻全部恢复。上限是**精确**的，不是统计性的。

复现：

```python
import sys, torch
sys.path.insert(0, "framework")
from cleansight_eval.temporal.util import causal_decision
# 沿 t 给 20..38 帧 sbc 的 logits[5]=8.0、其余 idle 的 logits[0]=8.0
# pending=None, stable=0, count=0 起步，逐帧调用并按 min_duration 扫描
```

## 2. 这条门槛直接卡住了 `sbc`

`framework/cleansight_eval/temporal/util.py` 的 docstring 自己就写明了这条上限，并给了实测：

> `min_duration` … 它同时是**召回上限**——真实段短于该值的动作在输出里不可能出现
> （2026-09-18 实测：project-18 test 的 short_brush_cleaning 六段全部 **≤19 帧**，
> `min_duration=25` 下召回恒为 0）。

而 `sbc`（short_brush_cleaning）正是**离线侧最难、也是本轮重点攻的类**：
它的 oracle 天花板量过是 **+2.37 acc / +12.91 edit**，离线侧靠锚点 v2 + 拼接把它从 0 补到 50.4。
在线轨若因一个默认常数而结构性拿不到它，那么"在线轨只丢稀类"的解释就要改写——
本任务此前把在线轨的差距描述为**整体平移**（idle −21.0、insert −27.2、withdraw −21.4、
sbc −32.7），但**没有单独查过这段差距里有多少是这条硬门槛造成的**。

## 3. 为什么这是一个"便宜且未探索"的杠杆

- **从未被扫过**：扫 6 个在线臂的 `config.resolved.json`，
  `smoothing_min_duration` **全部为 `None`**（即走代码默认 25）：
  `online-gru-{roi-v4, bbox40, roi-v1, hand-v2, global-hand-v2, roi-v4-w32}`。
- **它是纯推理后处理**：不需要重新训练，只需在评测时覆盖
  `evaluation.smoothing_min_duration` 重跑一遍推理（每个 arm 十几分钟量级）。
- **口径已具备**：`min_duration` 同时影响**所有类**，所以不能只按 sbc 挑——
  必须在 **val** 上按交付指标选、在 test 上报（与本任务既有的选点纪律一致）。

## 4. 下一步（交接，需在 GPU 空闲时做）

1. 取 2~3 个在线臂，在 **val** 上扫 `smoothing_min_duration ∈ {25, 19, 15, 10, 5}`；
   看交付指标（`edit` / f1 家族 / 逐类召回）随门槛的变化曲线。
2. 按 val 选一个门槛，在 test 上报；**重点看 `sbc` 召回是否从 0 起身**，
   以及为了它付出的 `idle`/`flush` 代价。
3. 若 val 支持，则在线轨的"稀类天花板"这条解释被证实，并可给出一组
   "在线轨交付配置"；若 val 不支持，则记一条负结果，把这条假设关掉——
   无论哪个方向都有价值，因为**此前没人查过**。

**注意**：本机此前 YOLO 全量训练在跑（占 GPU），故本文只做到机制验证这一步，
在线臂的实际扫描留到 GPU 空闲时执行。
