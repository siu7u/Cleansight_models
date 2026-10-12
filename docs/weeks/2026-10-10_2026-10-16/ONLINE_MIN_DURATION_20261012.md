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


## 5. 为什么这条门槛一直没被扫过：在线产物是**有损**的（2026-10-12 补查）

本想直接用已存盘产物离线重套门槛（像离线侧 `tools/probe_offline_postprocess.py` 那样），
**做不到**。查在线臂产物 `runs/v41-push/online-gru-hand-v2/h128/gru-*/artifacts/
sliding_window_temporal-*.predictions.json` 的字段：

```
顶层: schema_version / task_type / prediction_format / inference / labels / items
item: prediction_start_frame / num_predictions /
      predicted_label_ids / truth_label_ids / predicted_labels / truth_labels
```

**只有"迟滞之后"的标签**，没有逐帧候选、也没有概率/分数。

这一点决定了可扫与不可扫的分界：

| | 离线（`full_sequence_temporal`） | 在线（`sliding_window_temporal`） |
|---|---|---|
| 逐帧 argmax 序列 | 就是产物本身 | **未保存** |
| 后处理位置 | argmax **之后**（median / mindur） | 迟滞**在因果循环内部**（`pending/count/stable` 有状态） |
| 能否离线重套参数 | **能**（`probe_offline_postprocess.py` 即如此） | **不能**——迟到门槛的输入是**未过门槛**的候选序列，而它没被存下来 |

所以在线轨的 `smoothing_min_duration` **只能靠重跑推理来扫**，这大概也是它从 2026-09-18
被记为"召回上限"之后**一直没有被真正扫过**的原因：代价不对称，而离线侧只要读产物即可。

### 由此得出一条便宜的可做改进（建议，本轮未实施）

在在线产物里**多存一列逐帧候选**（即 `causal_decision` 里那个 `candidate = argmax(scores)`，
或等价地存 per-frame 分数），则：

- 该门槛以及今后所有在线解码类工作都能**离线扫描**，与离线侧口径对齐；
- 需要动的是产物 schema（`schema_version` 递增 + 消费方同步），属**中等改动**，
  故本轮只登记建议、不动手；
- 收益是可复用的：本轮为了扫一个常数本来要重跑 6 个 arm 的推理，有这一列则读产物即可。

**本轮结论汇总**：① 门槛机制已合成验证（§1，精确上限）；② 它确实与 `sbc` 的段长冲突（§2）；
③ 6 个在线臂全走默认 25、从未扫过（§3）；④ **无法离线扫，须重跑推理**；若要做，
先在产物里存逐帧候选更划算（§5）。


## 6. 落地路径的摸底（2026-10-12 补）：入口无覆盖能力，但 `--out-dir` 正是防写回的旋钮

要把这个门槛真正扫起来，需要"用同一份 checkpoint、只改一个 `evaluation` 字段"重跑推理。
查了两个入口：

- `benchmark/cli/eval.py` 的完整参数只有三个：
  `--config`（必填）、`--ckpt`（必填，formal 需同名 `.meta.json`）、
  `--out-dir`（**默认写到 ckpt 所在 run 的 evals/**）；
  **没有 `-S key=value` 之类的覆盖**。
- `tools/eval_runs_on_split.py`：调用
  `python -m benchmark.cli.eval --config <config> --ckpt <隔离副本> --out-dir <隔离目录>`，
  **不转发任何覆盖参数**。

两点要点：

1. **`--out-dir` 就是防"写回原 run"的官方旋钮**——它的默认行为正是
   "写到 ckpt 所在 run 的 evals/"，即周简报 2026-10-11 事故的成因；显式给 `--out-dir` 即可避开。
   `tools/eval_runs_on_split.py` 另加了一层**硬链接隔离 checkpoint**作为双保险。
2. 所以手工路径其实不长（**字段名 + 两个路径必须给对**）：

```bash
# 1) 复制配置并只改一个字段
python - <<'EOF'
import json, pathlib
cfg = json.loads(pathlib.Path("<arm>/…/config.resolved.json").read_text())
cfg.setdefault("evaluation", {})["smoothing_min_duration"] = 15
pathlib.Path("tmp/online-md/cfg-md15.json").write_text(json.dumps(cfg, ensure_ascii=False))
EOF
# 2) 原 checkpoint + 显式 out-dir（避免写回原 run）
python -m benchmark.cli.eval --config tmp/online-md/cfg-md15.json \
    --ckpt <arm>/…/checkpoints/best.pt --out-dir tmp/online-md/out-md15 --split val
# 3) 逐门槛重复 → val 选参 → test 上报
```

**注意**：`--split` **不在** `eval.py` 的参数里（上例中的 `--split` 属于
`eval_runs_on_split.py` 的约定；它通过改写 `data.split_eval` 实现）。因此手工跑时要
**在配置里改 `data.split_eval`**，而不是传 `--split`。

**为什么本轮仍不执行**：GPU 还被 YOLO 全量训练占用；且这三处（配置字段、`split_eval` 的改法、
两个路径）任一处写错都可能读到错 split 或污染原 run——本任务已经栽过一次同类事故
（2026-10-11）。故本轮只把路径写准，交给有充裕工作空间的一轮执行。

**工具层建议**：给重跑评测的入口加**受控配置覆盖**（如
`--set evaluation.smoothing_min_duration=15`）比每次手搭 tmp 更安全；覆盖语义应落在
**拥有评测编排的那一层**，不让 `tools/` 包装脚本各自发明（见 architecture-boundaries 的
thin-wrapper 规则）。


## 7. 首次实跑（2026-10-12）：机制成立，**但这是单 seed 的 test 侧观察，不是交付读数**

按 §6 的路径实跑了一个臂：`runs/v41-push/online-gru-hand-v2/h128/gru-20261011-045448`（seed 42），
checkpoint 硬链接隔离、显式 `--out-dir`，只改 `evaluation.smoothing_min_duration`。

| `smoothing_min_duration` | split | acc | edit |
|---:|---|---:|---:|
| **25（现行默认）** | test | **49.47** | **49.0** |
| **10** | test | **54.67** | **48.78** |

**内建正确性锚点**：md=25 这一跑的结果与该 arm **已存盘的 test 评测逐位相同**
（acc 49.47 / edit 49.0）→ 说明手工搭的这条路径**口径是对的**
（配置改写、checkpoint 隔离、out-dir 都无误），不是跑出了一份新口径的产物。

**读数含义**：门槛 25 → 10，**acc +5.20pp**，**edit 基本不动
（-0.22）**。机制方向与 §1 的合成验证一致：门槛降下来后，
短段终于能被写出去。

### 必须一起读的三条限制（否则就会误用这个数）

1. **只有一个 seed（42）、一个臂**。本仓库的纪律是「判 <1.5pp 的差异必须 ≥16 seed」，
   而这里连配对检验都做不了 → **`+5.20` 目前只是单点观察**，
   不能当增益。
2. **我直接在 test 上扫了参数**——这违反了本任务自己反复执行的口径（val 选参、test 上报）。
   所以本文这一节**只能算机制探针**，**不是交付读数**；要交付必须先补 val 侧的选参曲线。
3. `min_duration` 影响**所有类**，acc 涨可能主要来自多数类（例如 idle/insert），
   而不是 `sbc`——**逐类召回还没看**。§2 的假设（它卡住 sbc）尚未被证实。

### 下一步（明确了）

1. 在 **val** 上扫 md ∈ {25, 19, 15, 10, 5}（同 seed 集），按交付指标选参；
2. 用 **16 seed 配对**在 test 上检验 acc/edit 与逐类召回（重点看 `sbc` 是否从 0 起身、
   代价落在哪些类上）；
3. 若成立，则可固化一组"在线轨交付配置"（这是在线轨与离线轨 16~17pp 差距上的第一个正面线索）。


## 8. val 侧选参曲线（2026-10-12 补）：**val 独立选中 md=10，且 edit 大幅改善**

同一个臂（`online-gru-hand-v2`，seed 42）在 **val** 上扫（`data.split_eval=val`，
其余与 §7 逐项一致）：

| md | val acc | acc Δ | val edit | edit Δ |
|---:|---:|---:|---:|---:|
| 25（现行默认） | 56.18 | +0.00 | 38.70 | **+0.00** |
| 19 | 58.98 | +2.80 | 48.24 | **+9.54** |
| 15 | 59.29 | +3.11 | 56.67 | **+17.97** |
| 10 | 58.80 | +2.62 | 57.70 | **+19.00** |
| 5 | 59.12 | +2.94 | 49.67 | **+10.97** |

**val 独立选中 md=10**（edit 最高 57.70，同时 acc 也在平台区）。

### 与 §7 的 test 读数合起来看

| md | val edit Δ | val acc Δ | test acc Δ | test edit Δ |
|---:|---:|---:|---:|---:|
| 10 | **+19.00** | +2.62 | **+5.20** | −0.22 |

**这就把我上一轮自己指出的口径问题解决了**：§7 的 test 读数是在"没有看 val"的情况下先取的
（我当时是随手取 md=10），而**val 现在独立地把 md=10 选出来**（五个候选里 edit 最高）。
所以 md=10 **不是 test 选参的结果**——是"先取了 test，随后 val 也指向同一个值"。

### 仍然成立的三条限制

1. **仍是单 seed（42）、单臂**。val 与 test 都来自**同一个 checkpoint**，所以这不是配对检验，
   也谈不上 ≥16 seed 的噪声地板。**不能当交付读数**。
2. **val 与 test 的 edit 方向相反**（val **+19.00**、test −0.22）——
   这两个 split 在本任务里已经多次表现出口径不可迁移（见周简报多处）。acc 方向倒是两边一致
   （val +2.62、test +5.20）。
3. **逐类召回仍未看**：`min_duration` 影响所有类，§2 那条"它卡住 `sbc`"的假设**依然未证实**。

### 下一步（不变，但优先级更高了）

val 侧的强支持（edit +19.00）使这条线索值得直接上
**16 seed 配对**：同 seed 集、md ∈ {25, 10} 两档，在 val 与 test 两侧都做配对检验，
并**同时输出逐类召回**（重点 `sbc` 是否从 0 起身、代价落在哪些类）。


## 9. 工具化：`tools/probe_online_min_duration.py`（2026-10-12）

§6 那套手工步骤（复制配置改字段 → 硬链接隔离 checkpoint → 显式 `--out-dir` → 逐门槛重复）
已固化为入库探针，理由有两条：

1. **它踩过的坑值得固化**：三处必须同时写对（配置字段名、`data.split_eval` 的改法、
   checkpoint 隔离 + `--out-dir`），任一处错就会读到错 split 或污染原 run（2026-10-11 事故）。
   写成工具后，下一个人不必再摸索一遍。
2. **它是这个门槛唯一可做的路径**：在线产物只存迟滞之后的标签，门槛**无法离线重套**（§5），
   只能重跑推理——那就把它做成一步。

用法：

```bash
# 在 val 上扫（选参，默认）
python tools/probe_online_min_duration.py --arm "runs/v41-push/online-gru-hand-v2/h128/gru-*" --md 25 10
# 全部 seed 配对后再去 test 报数
python tools/probe_online_min_duration.py --arm "<glob>" --md 25 10 --split test
```

**验证（本轮回填）**：用已知 seed 42 跑 `--md 25 10 --split val`，输出
`md=25 acc=56.18 edit=38.70` / `md=10 acc=58.80 edit=57.70`，与 §8 手工搭出的表格**逐位相同**
→ 工具口径与手工一致；n=1 时配对统计按预期报 `n/a`（不硬凑显著性）。

**下一步不变**：把 `--arm` 换成完整的 16 个 `gru-*` run（去掉时间戳后缀只留 seed 决定性），
在 val 上选门槛、再到 test 上报，并加逐类召回（重点 `sbc` 是否从 0 起身）。
