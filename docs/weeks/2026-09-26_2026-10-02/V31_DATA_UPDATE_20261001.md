# 数据集 v3.1 接入与即时复现（2026-10-01）

> 触发：上游 ModelScope `lhh010/cleansight-ActionMixed-auto` 于 **2026-09-29** 更新为 **v3.1**
> （2026-09-28 标注修订）。本文件记录**接入处置、实测差异、重训结果**。
> 相关：catalog 条目见 [`framework/testsets.yaml`](../../../framework/testsets.yaml)，
> 脚本见 `tmp/v31/`，产物 `runs/v31/`。

## 1. 处置原则：新版本并列，不覆盖 v3

- **不覆盖** `datasets/cleansight-ActionMixed-auto-lhh`——它钉住 2026-09 全部历史 run 的复现性
  （本周 1,069 个 run、全部 11 课题读数都建立在 v3 上）。
- 新数据落到 **`datasets/cleansight-ActionMixed-auto-v31`**（走仓库自带下载器，git-lfs 正常）：

```bash
python -m framework.cleansight_eval.cli.dataset --preset actionmixed-auto \
    --dataset lhh010/cleansight-ActionMixed-auto \
    --output datasets/cleansight-ActionMixed-auto-v31
```

## 2. 实测差异（逐文件 sha256 + 逐行比对；脚本 `tmp/v31/`）

| 维度 | v3（`-lhh`） | v3.1（`-v31`） | 说明 |
|---|---|---|---|
| test | 8 视频 | 8 视频 | **逐字节未改**（标签与 `frames/` 检测框都相同）→ **跨版本 test 读数同尺子可比** |
| train | 14 视频 | 14 视频 | 换了一个：−`f173153a`（#198 遮挡 flush → `occlusion/`）、+`f809e944`（#203 补录，266 帧/flush 52） |
| val | 4 视频 | **3 视频** | −`152453e5`（#207，只有废弃动作 water_injection → `deprecated/`） |
| 标签改动 | — | 4 个 train + 1 个 val | **#201 `39da2635` 194/294 帧 = 66%**、#205 `c1367d51` 125 帧、#202 `4cc6a009` 36 帧、#193/194 `071eb2d6` 14 帧；val `e8ea5bb7` 1 帧 |
| 仅换行符 | CRLF | LF | 8 个 train + 2 个 val 文件**逐行内容相同**，仅 CRLF→LF |
| `frames/`（检测框） | — | — | **全部视频未变** → 特征差异只来自动作标签侧 |

**⚠️ 选点风险**：val 由 4 视频降为 **3 视频**，`best_metric=val_edit` 的选点噪声进一步变大
（`docs/EVAL.md` 记录：val 4 视频时 `val_f1@0.5` 与 test 的 Spearman ρ 仅 0.199）。

## 3. 登记（按"数据变更=新版本"纪律）

| 项 | 内容 |
|---|---|
| manifest | `benchmark/manifests/actionmixed-auto-v31/{train,val,test}.txt`（14/3/8，取自 `labels/<split>/` 目录） |
| revision | `875d73d168ca95e8ba2b2e5d0ee9fc97e9a73f7dabaaa45a48eb51aa10f7c313`（= 三 split manifest 原文拼接 sha256，口径与其它条目一致，已用旧值 `6375eba9…` 反向验证） |
| catalog 数据集 | `temporal.actionmixed-auto-v31`（bbox 40d）、`temporal.actionmixed-auto-roi-v1-v31`（144d）、`temporal.actionmixed-auto-roi-v4-v31`（128d） |
| catalog split | 上述 3 个 ref × train/val/test = 9 条 |
| 门禁 | `python tools/validate_testsets.py --catalog framework/testsets.yaml` → **ok=true, errors=0**（61 个 split 条目） |
| 运行自检 | 新臂 run 的 `evals` 中 `dataset_version=cleansight-actionmixed-auto-*-v31`、`revision=875d73d1…`、`registered=true`、`validation_errors=[]` |

## 4. 重训与结论（mstcn2 s4l10 h128 / lr5e-4 / wd1e-4 / 60 轮 / `best_metric=val_edit`，与本周 acc-push 臂同口径）

产物：`runs/v31/roiv1{,‑b}/`、`runs/v31/roiv4{,‑b}/`（**16 对 seed** = A 批 8 + B 批 8，
与旧臂同 seed 集，可逐 seed 配对）；对照脚本 `tmp/v31/compare_v31.py`。

### Q1 契约增益是否仍成立 —— **成立，幅度在噪声内**

| 数据 | v1（144d） | v4（128d） | Δ 中位 | 胜/负 | p |
|---|---:|---:|---:|---|---:|
| v3（旧，16 seed） | 54.34 | 58.24 | **+3.94** | 16/0 | 0.00044 |
| **v3.1（新，16 seed）** | **52.99** | **57.51** | **+3.77** | **16/0** | **3.05e-05** |

→ **标注修订（#201 的 66% 重标 + 换掉一个 train 视频 + val 4→3）没有动摇"v4 契约优于 v1"这一结论**；
幅度差 −0.17pp，远小于 16-seed 的分辨力（1.35pp）。

### Q2 标签修订对绝对读数的影响 —— 帧准不变，**段级指标系统性下降 3.6~5pp**

| 契约 | 指标 | v3 | v3.1 | Δ 中位 | 胜/负 | p |
|---|---|---:|---:|---:|---|---:|
| v1 | acc | 54.34 | 52.99 | −0.80 | 7/8 | 0.073 |
| v1 | **edit** | 50.52 | 46.75 | **−4.59** | 3/13 | **0.0027** |
| v1 | **f1@0.25** | 31.32 | 25.83 | **−5.05** | 1/15 | **9.2e-05** |
| v1 | f1@0.5 | 12.31 | 9.27 | −2.89 | 3/13 | 0.0042 |
| v4 | acc | 58.24 | 57.51 | −0.68 | 5/11 | 0.13 |
| v4 | **edit** | 43.09 | 37.36 | **−4.97** | **0/16** | **3.1e-05** |
| v4 | **f1@0.25** | 32.31 | 29.80 | **−3.60** | 4/12 | **0.018** |
| v4 | f1@0.5 | 14.02 | 12.02 | −1.67 | 4/12 | 0.034 |

→ **帧准确率对标签修订不敏感（−0.7~0.8pp，边缘不显著），但段级指标系统性变差**，
且 v4 的 `edit` 是 **16/16 全部变差**。**引用本周任何绝对读数时都必须注明是 v3 口径。**

### Q3 标签到底改了什么（train 迁移矩阵，共 369 帧 = 3.8%）

| v3 → v3.1 | 帧数 |
|---|---:|
| `water_injection` → `flush` | 129 |
| `flush` → `idle` | 125 |
| `water_injection` → `idle` | 65 |
| `short_brush_cleaning` → `idle` | 47 |
| `idle` → `sbc` | 3 |

按视频：`39da2635`（#201）194 帧、`c1367d51`（#205）125、`4cc6a009`（#202）36、`071eb2d6` 14。
**`insert` / `withdraw` 的帧一帧未动**（1289 / 437 → 1289 / 437）；val 仅 1 帧（idle→flush）。

### Q4 逐类（v4 契约，v3 → v3.1 的 test 指标）—— 含一个**未解释**的回归

| 类 | 指标 | v3 | v3.1 | Δ |
|---|---|---:|---:|---:|
| `long_brush_insert` | F1 | 42.18 | 30.24 | **−11.11** |
| `long_brush_insert` | 召回 | 32.04 | 21.57 | −11.95 |
| `long_brush_withdraw` | F1 | 11.55 | 13.16 | +3.72 |
| `idle` | 召回 | 88.00 | 91.50 | +4.29 |
| `flush` | F1 | 0.00 | 0.00 | 0.00 |

**⚠️ 未解释点**：train 的 `insert` 标签**一帧未改**，但 test 上 insert F1 掉 11pp（0/16 seed 全负）。
已排除的解释：
- **不是选点漂移**：两版的 `best_epoch` 中位相同（50.5）、逐 seed 差 −22~+23 居中于 0；
  v3.1 的 `val_edit` 最佳值反而更高（71.95 vs 70.72）。
- **不是 insert 监督变少**：帧数完全相同。

剩下的候选解释（**均未验证**）：① 换掉/新增的那两个视频改变了训练序列的上下文分布；
② `water→flush` 的 129 帧让模型把 flush 相关的时序模式重新分配，间接挤占了 insert 的判别边界；
③ `flush→idle`/`sbc→idle` 使模型整体更保守（idle 召回 +4.3 印证），代价落在 insert 上。

## 5. 下一步

1. **定位 insert 回归**（本轮最值得追的一条）：建两个中间数据集做隔离实验——
   ① 只换视频（v3 标签 + 视频替换）、② 只改标签（v3 视频 + v3.1 标签），各 16 seed，
   即可判断回归来自"视频替换"还是"标签重分配"。成本约 2×20 分钟 GPU。
2. **是否把默认配置切到 v3.1**：建议**保留旧 YAML 指向 v3**（历史可比），
   新增 v3.1 用法以 `-S data.dataset_ref=...-v31` 注入；需要你确认口径。
3. **段级指标下降（−3.6~5pp）作为独立问题跟踪**：若下游消费段级事件，这是比帧准更重要的变化。
4. **flush 专项**（本周遗留）：v3.1 把唯一的遮挡 flush 训练视频移出，该类的可救性需重估。
5. 已扩到 16 对 seed；若要判 <1.5pp 的差异需 32+ seed（见剖析文档 §1.4 的分辨力表）。
