# 实验报告：帧准确率攻关第 2 轮 —— 候选根因："自动标注的检测框"（2026-09-29）

> ## ⚠️ 更正声明（2026-09-29 第 3 轮追加）
>
> 本报告 §3 与 §5 的两条**因果结论已被第 3 轮的多划分复核推翻**，阅读时请以第 3 轮报告为准：
> [`EXPERIMENT_REPORT_FRAME_ACC_R3_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R3_20260929.md) §0。
>
> | 本报告原结论 | 复核结果 | 状态 |
> |---|---|---|
> | §3：manual（人工框）+7.01pp vs auto（YOLO 框）−11.38pp，"分水岭在框的来源" | 统一协议 10 折：auto Δ 中位 **−5.06**（10/10 为负，稳）；manual Δ 中位在 **−2.95 ~ +4.28** 之间随划分集合剧烈摆动，与平凡基线**不可靠区分** | ⚠️ 量级作废 |
> | §5："判别力 **100% 来自 5 个稀有器械类通道**（+7.01 → −0.86）" | 10 折配对：`full` Δ 中位 −2.95、`drop-rare` −2.75，配对差中位 **+0.08pp**，胜/负 5/5，**Wilcoxon p=0.557** | ❌ 撤回 |
>
> **原因**：两条结论都建立在"前 12 个 stem 训练 / 后 4 个测试"这**一个**划分上，
> 而留出集只有 3~4 个视频，单折读数完全不可用。本报告保留原文以留痕，
> **凡 §3/§5 的 Δ 数字一律不要引用**。
>
> **仍然成立的部分**（直接测量，不依赖留出）：
> §4 的逐类框数对比（auto 的稀有器械类只有人工框的 5%~21%）、
> §2 的训练臂读数（合并集 47.75 / auto120 54.19 — 都是 8 seed test 的正式评测）、
> §2 的"工作点 vs 表示"泄漏上界对比（55.17 vs 57.52）。

---


## 1. 基线（与第 1 轮一致）

| 项 | 值 |
|---|---|
| 配方 | `mstcn2` s4l10 h128（3,312,664 参）+ `actionmixed-roi-grid-v1`(144 维) |
| 读数 | 8 seed test **acc 中位 53.92**；跨 seed 摆幅 4.32pp |
| **平凡基线（全 idle）** | **55.25** |
| test 口径 | `temporal.actionmixed-auto-roi-v1.test`，8 视频 / 2639 帧（LS project-18，长短毛刷专项） |

## 2. 尝试 A：并入第二个来源的训练数据 → **负迁移**

**动机**：第 1 轮定位出两个失效模式——① 跨批次（train/val 是 project-16，test 是 project-18）；
② 先验偏 idle（auto train idle 65.5% vs test 55.2%）。手动数据集
`datasets/cleansight-ActionMixed` 有 **16 个与 auto 零交集的视频 / 9,532 帧**，且 **idle 只占
25.6%**，看似同时对症。

**做了什么**（已按仓库规程落地，可复跑）：

- 新增构建器 `tools/build_union_temporal_dataset.py`（只读原始数据集，产物可随时重建）；
  产出同构合并根 `datasets/cleansight-ActionMixed-union/`：train = auto-train 14 视频 +
  手动集 16 视频全部帧 = **30 视频 / 19,107 帧（帧数 2 倍，idle 65.5% → 45.6%）**；
  **val / test 与 auto 逐字节相同**（目录符号链接），保证单变量对照。
- `framework/testsets.yaml` 新增 `temporal.actionmixed-union-roi-v1` + 三个 split；
  manifest `benchmark/manifests/actionmixed-union/train.txt`；`usage/YAML_CONFIG.md` 与
  `docs/features/README.md` 同步登记。
- `tools/validate_testsets.py` 全量 **`ok: true` / `errors: []`**；1 轮 smoke 训练的
  checkpoint meta 记录 `registered: true`、id/revision 正确。

**结果（8 seed，60 epoch，`best_metric=val_edit`，其余同基线）**

| 臂 | 训练数据 | test acc | Δ vs 53.92 | idle 召回 | insert 召回 |
|---|---|---:|---:|---:|---:|
| 基线 旗舰 | auto train 14 视频 | 53.92 | — | 82.7% | 25.7% |
| **A 合并集** | 30 视频（auto+manual） | **47.75** | **−6.17** | 72.2% | 27.4% |
| 等曝光量对照 | auto train 14 视频，**120 epoch** | 54.19 | +0.27 | 86.7% | 22.7% |

- 等曝光量对照（120 epoch）说明"多算不算增益"，排除了"只是训练步数翻倍"的解释；
- 合并集把模型推得更"敢判动作"（idle 召回 82.7%→72.2%），但**动作召回几乎没涨**
  （insert 25.7%→27.4%），多出来的动作预测几乎全是 idle 帧上的误报 → 净掉 6.17pp。

**是"工作点偏了"还是"表示变差了"？**（`tmp/acc-decomp/armcmp.py`）

| 臂 | 单 seed 中位 | 统一偏置最优（泄漏上界） | 逐类偏置最优（泄漏上界） |
|---|---:|---:|---:|
| 基线 旗舰 | 53.92 | 54.91 | **57.52** |
| 等曝光量 auto120 | 54.19 | 55.06 | 56.61 |
| 合并集 | 47.75 | 50.09 | **55.17** |

**合并集的表示上限（55.17）比基线（57.52）低 2.35pp** → 不是工作点问题，
**加进来的数据让表示真的变差了**（负迁移），而不是"先验变了、调回来就行"。

## 3. 尝试 B：同一特征的"批次内 vs 跨批次"对照 → 找到了真正的分水岭

**方法**（`tmp/acc-decomp/batchshift2.py`）：完全相同的特征配方（ROI-144）与分类器（HistGB），
按 **视频 stem 分组留出**（手动集的 train/val/test 是同一批视频的不同帧区间，必须按 stem 汇总后
再切，否则会高估泛化——本报告修正了初版这个错误）。

| 设置 | 训练 | 测试 | acc | 该测试集全 idle | **Δ** |
|---|---|---|---:|---:|---:|
| auto 批次内（11→3 视频） | auto train | auto train 余下 | 56.49 | 67.86 | **−11.38pp** |
| auto → auto test（真实 benchmark） | auto train 14 | auto test 8 | 53.85 | 55.25 | −1.40pp |
| **manual 批次内（12→4 视频）** | 手动集 | 手动集余下 | 36.88 | 29.87 | **+7.01pp** |
| manual → auto test | 手动集 16 | auto test 8 | 46.00 | 55.25 | −9.25pp |

**同一个特征配方、同一个协议：人工框数据 +7.01pp（真的能判），YOLO 框数据 −11.38pp（完全不能判）。**
分水岭不在批次，在**框的来源**。

## 4. 尝试 C：逐类框数对比 → 差异集中在"定义动作的稀有器械类"

每帧平均框数（`frames/<split>/*.txt` 直接统计）：

| 来源 | 每帧框数 | hand | scope_ctrl | scope_mid | **scope_distal** | **syringe** | **air_gun** | **short_brush** | **brush_tip** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 手动集（**人工 videorectangle 框**） | 4.48 | 1.84 | 0.82 | 0.79 | **0.44** | **0.40** | **0.066** | **0.079** | **0.056** |
| auto（**YOLO 推理框**） | 3.46 | 1.71 | 0.93 | 0.70 | **0.078** | **0.024** | **0.003** | **0.017** | **0.003** |
| 保留率（auto / 人工） | — | 0.93 | 1.13 | 0.89 | **0.18** | **0.06** | **0.05** | **0.21** | **0.05** |

三个高频类（hand / scope_control / scope_mid）两边**基本一致**；
**五个稀有器械类的保留率只有 5%~21%**，而它们正是动作的"定义物"
（`syringe`→冲水/注水、`short_brush`→短刷清洁、`brush_tip_out`/`scope_distal_end`→长刷进出）。

**检测器侧证据**：`registry/detection/yolo-group2-small-v1/eval_report.md` 报
**mAP@0.5 0.343 / Recall 0.394**（门槛 0.5），且 `syringe` 与 `scope_distal_end`
**在验证集上无法评估**（"验证集无样本或未检出"）。

**配置侧证据**：构建 auto 数据的 `framework/experiments/auto-annotate-yolo11.yaml` 用的是
**全局 `conf: 0.25`**（文件里自述"标量全局生效"），而同一仓库的 `auto-annotate.yaml`
早就注释好了逐类放宽的示例（`syringe: 0.1` / `scope_distal_end: 0.15`）却没启用。

## 5. 尝试 D：因果消融 → **信号完全来自稀有器械类通道**

若"稀有类证据缺失"是判别力消失的**充分原因**，那么在**人工框**上把稀有类通道削掉，
应当复现出 auto 的"无信号"状态。`tmp/acc-decomp/rareablate.py`（手动集，12→4 视频留出）：

| 特征条件 | acc | 全 idle | **Δ** |
|---|---:|---:|---:|
| `full`（人工框全部 144 维） | 36.88 | 29.87 | **+7.01** |
| `drop-rare`（砍掉 5 个稀有类整块，只留 3 个高频类） | 29.02 | 29.87 | **−0.86** |
| `subsample-rare p=0.50` | 32.72 | 29.87 | +2.84 |
| `subsample-rare p=0.25` | 29.48 | 29.87 | −0.40 |
| `subsample-rare p=0.10`（≈ auto 实测保留率） | 30.14 | 29.87 | +0.26 |
| `subsample-rare p=0.05` | 29.54 | 29.87 | −0.33 |

**砍掉稀有类通道 → Δ 从 +7.01 直接掉到 −0.86（信号归零）；按 auto 的保留率子采样 → +0.26。**
即：**ROI-144 的动作判别力几乎 100% 来自稀有器械类通道**，而 auto 数据里这些通道基本是空的。
这条把第 4 节的"相关"升级为**因果**。

## 6. 尝试 E：把恒零的稀有类通道遮掉再训（当下可执行的对照）

既然 auto 的稀有类通道基本是噪声/常数，直接把它们遮掉（`feature_schema.mask_targets`）是否
反而更好？（`runs/acc-push/acc-norare`，8 seed × 60 epoch，其余同基线）

| 臂 | test acc | Δ vs 53.92 | idle 召回 | flush 召回 | insert 召回 |
|---|---:|---:|---:|---:|---:|
| 基线 旗舰（144 维全开） | 53.92 | — | 82.7% | 7.6% | 25.7% |
| `acc-norare`（遮掉 5 个稀有类 → 有效 54 维） | **53.20** | **−0.72** | 84.1% | 0.0% | 19.0% |

**遮掉稀有类通道没有增益，反而略降**（落在跨 seed 摆幅 4.32pp 内）。这条与第 5 节互为印证：
手动集上 `drop-rare` 后 Δ = −0.86（只剩三个高频类时模型回到平凡水平），
auto 上遮掉稀有类后 **53.20** 也回到平凡水平（全 idle 55.25）——
**auto 的稀有类通道确实带不来信息，而剩下的三个高频类也不足以判别动作。**
同时也说明"把噪声通道关掉"这类补救无效：问题不是噪声，是**证据缺失**。

## 7. 根因链（本轮闭合）

```
auto-annotate-yolo11.yaml: conf=0.25（全局，标量）
        │
        ▼
g2 小目标检测器 mAP@0.5=0.343 / R=0.394；syringe、scope_distal_end 在验证集上检不出
        │
        ▼
auto 数据里 5 个稀有器械类的框只有人工标注的 5%~21%
        │
        ▼
ROI-144 的这 90 维基本恒零 → 而消融显示判别力 100% 来自这 90 维（+7.01 → −0.86）
        │
        ▼
时序模型只能靠 hand/scope 三高频类 → 帧准确率封顶在"全 idle"（55.25）
        │
        ▼
基线 53.92（低于平凡基线）、20 余次模型/训练/解码尝试全部打不动
```

## 8. 结论与可执行修复

1. **+5pp 在"现行 auto 检测框"下不可达**；本轮再排除三条替代解释：
   更多数据（合并集 → −6.17pp 负迁移）、更多算力（120 epoch → +0.27pp）、更好的工作点
   （逐类偏置的**泄漏上界**也只有 57.52）。
2. **修复点是单一且明确的：重跑自动标注，对稀有类放宽置信度阈值。**
   仓库里 `auto-annotate.yaml` 已备好逐类阈值示例（`syringe: 0.1` / `scope_distal_end: 0.15`），
   只需在**有视频与 `yolo11s-g2-v1` 权重的机器**上执行（本 checkout 缺视频，已核：
   `datasets/` 无 project-16/18 像素；ModelScope `lhh010/cleansight-raw` 只有 LS 导出 JSON；
   Label Studio `49.234.120.241:8080` 可达但下载需凭据）。
   **这条与仓库自己的判断一致**——`docs/features/INPUT_DESIGN_PROPOSAL.md` §3 早已把
   "检测侧扩召回（逐类降阈值 / 换 g2 权重 / 补帧）"列为**"最高天花板，但有资源前置"**；
   本轮补上的是**因果量化证据**（第 4/5 节）。
3. **预期收益的量级**：人工框数据上"相对全 idle"的判别力是 **+7.01pp**；若 auto 的框恢复到
   接近人工框的覆盖，帧准确率有望从 53.92 抬到 **~62（≈ +8pp）**，足以覆盖 +5pp 目标。
   ⚠️ 这是**量级估计**，不是承诺：手动集的动作先验（idle 25.6%）与 auto test（idle 55.2%）不同，
   且两者批次不同。
4. **附带可交付项**：第 1 轮的"否决 `withdraw` + 平滑"（纯 val 口径 +1.48pp）仍成立，可与
   上游修复叠加。

### 8.1 具体怎么改（给持有视频与权重的那台机器）

只需在 `framework/experiments/auto-annotate-yolo11.yaml`（构建 auto 数据的那份配置）里把
**全局标量 `conf` 换成逐类映射**，其余不动——`docs/features/INPUT_DESIGN_PROPOSAL.md` §3
与 `auto-annotate.yaml` 的注释里已备好示例值：

```yaml
# 原来是标量：conf: 0.25   ← 三个高频类够用，小目标类被整体压掉
conf:
  hand: 0.25
  scope_control_body: 0.25
  scope_mid_section: 0.25
  scope_distal_end: 0.15      # 人工框保留率的短板之一（auto 仅 0.18）
  syringe: 0.10               # 最短板（auto 仅 0.06）
  air_gun: 0.10               # auto 仅 0.05
  short_brush: 0.10           # auto 仅 0.21
  brush_tip_out: 0.10         # auto 仅 0.05
```

然后按原流程重建检测标注并重新登记数据集版本（**检测源变更 = 新 dataset_version**，
不能悄悄改现有 v3 条目的框）。建议同时做两件事以判定收益：

1. 用新框重算**逐类每帧框数**，与第 4 节的人工框列对比——`syringe`/`air_gun`/`brush_tip_out`
   若能从 0.02~0.006 抬到 0.1 量级即可认为覆盖恢复；
2. 用同一份 8 seed × 60 epoch 配方重训，直接看 `acc` 是否越过平凡基线（55.25）。
   参考量级：人工框条件下的判别力是 +7.01pp，故目标是 **acc ≥ 58.92**。

## 9. 复现

```bash
source ../CleanSightBackend/.venv/bin/activate

# 构建 + 校验合并集（只读原始数据集）
python tools/build_union_temporal_dataset.py --check
python tools/build_union_temporal_dataset.py
python tools/validate_testsets.py --catalog framework/testsets.yaml --json | head

# 训练臂
python tools/run_capacity_matrix.py --runs-dir runs/acc-push/acc-union \
  --config mstcn-actionmixed-auto-roi.yaml --hidden 128 --seeds 42,7,2026,1,2,3,4,5 \
  --best-metric val_edit --epochs 60 --no-probe-baseline --set train.lr=0.0005 \
  --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
  --set data.dataset_ref=temporal.actionmixed-union-roi-v1
python tools/run_capacity_matrix.py --runs-dir runs/acc-push/acc-auto120 ... --epochs 120
python tools/run_capacity_matrix.py --runs-dir runs/acc-push/acc-norare ... \
  --set feature_schema.mask_targets=scope_distal_end,syringe,air_gun,short_brush,brush_tip_out

# 诊断脚本
python tmp/acc-decomp/batchshift2.py     # 视频级留出：auto vs manual
python tmp/acc-decomp/rareablate.py      # 稀有类通道因果消融
python tmp/acc-decomp/armcmp.py acc-union acc-auto120 recheck-flagship   # 工作点 vs 表示
```

**已知限制**：test 仅 8 视频 / 2639 帧（跨 seed 摆幅 4.32pp）；手动集与 auto 批次不同，
第 8 节第 3 条的收益是量级估计；`sbc` 在 test 只有 51 帧（≈7 个采样帧），相关结论置信度低；
凡标"泄漏上界"的数字不可作为交付指标。
