# 实验报告：帧准确率（`acc`）+5pp 攻关 —— 一系列尝试与天花板定位（2026-09-28）

> **目标**：把当前时序模型的**帧检测准确率**（指标 `accuracy/frame-wise-micro-across-items/percent/v3`，
> 6 类逐帧 micro 准确率，下称 `acc`）提升 **≥5 个百分点**。
> **结论先行**：**未达成**。本轮做了 5 个轴、20 余次可复现尝试，最好读数是 **56.69（+2.77pp）**，
> 且该读数里的"选臂"用了 test（偏乐观）；纯 val 选点口径下最好 **55.40（+1.48pp）**。
> 更重要的是本轮**定位了天花板**：在现行输入契约下，`acc` 的任何判决规则上界约 **57.5**
> （这是**用 test 标签拟合**出的泄漏上界，不是可交付数字），即 **+5pp 在本输入源下不可达**。

---

## 1. 口径与基线

| 项 | 值 | 来源 |
|---|---|---|
| 指标 | `acc`（6 类逐帧 micro，合并全部视频帧） | `framework/cleansight_eval/core/metrics.py` |
| 目标 | 53.92 + 5.00 = **58.92** | 目标定义 |
| 基线配方 | `mstcn2` s4l10 h128（3,312,664 参）+ `actionmixed-roi-grid-v1`(144 维, z-score) | `runs/gpu-batch/recheck-flagship` |
| 基线读数（8 seed test 中位） | **acc 53.92** | 同上，`CAPACITY_SUMMARY.md` |
| **平凡基线（全 idle 常数预测器）** | **55.25** | 只依赖真值序列 |
| 跨 seed 摆幅（噪声地板） | **4.32pp** | 同上 |
| 数据 | `temporal.actionmixed-auto-roi-v1`，train 14 / val 4 / test 8 视频 | `framework/testsets.yaml` |
| 设备 | RTX 4060 Laptop（沙盒内 `torch.cuda.is_available()=True`） | — |

**基线的关键事实**：模型 `acc 53.92` **低于**全 idle 的 `55.25`。也就是说，按纯帧准确率判读，
当前旗舰配方**没有跑赢一个常数预测器**。

## 2. 误差分解（8 seed 中位 seed，test 2639 帧）

| 来源 | 帧数 | 说明 |
|---|---:|---|
| idle 判对 | 1224 | |
| **idle 被判成动作** | **234** | 其中 168 被判成 `long_brush_insert` |
| 动作判对 | 207 | |
| **动作被判成 idle** | **786** | 判漏的主体 |
| 动作被判成其它动作 | 188 | 其中 180 是 `insert ↔ withdraw` 互串 |

- 逐类帧级 recall（8 seed 中位）：`idle` 82.7、`flush` 7.6、`insert` 25.7、`withdraw` 4.1、`sbc` 0.0。
- 达到 +5pp 需要**净修好 132 帧**。
- 二分类（动作/非动作）：precision 60.9%、recall 38.4%；
  **已判为动作的帧里类正确率只有 64.3%** → 端到端"预测动作且类对"的精度 ≈ 39%，
  低于"多预测一帧动作"所需的 50% 门槛 —— 这是 `acc` 上不去的直接算术原因。

## 3. 先验漂移（本轮最重要的口径发现）

| split | idle | water_inj | flush | insert | withdraw | sbc |
|---|---:|---:|---:|---:|---:|---:|
| train (9575 帧) | 65.5% | 2.0% | 9.3% | 13.5% | 4.6% | 5.1% |
| val (3384 帧) | 68.2% | 0.5% | 5.4% | 14.6% | 6.1% | 5.2% |
| **test (2639 帧)** | **55.2%** | 0.0% | 6.2% | **26.8%** | 9.8% | 1.9% |

**test 的 idle 占比比 train/val 低 10~13 个百分点，insert 占比是 train 的两倍。**
后果：任何在 val 上做的判决规则都会系统性地偏向 idle——本报告里几乎所有 val 选点的尝试
都收敛到"接近全 idle"，而 test 恰恰需要更激进的动作判别。这解释了为什么
"val 上 +12pp"的规则在 test 上只有 +0.7pp。

**这个先验差不是抽样噪声，而是数据集设计的**（`datasets/cleansight-ActionMixed-auto-lhh/README.md` §二）：

- train/val = Label Studio **project-16**（2026-08 录制，18 个视频）；
- test = **project-18（action-test）task#213-220**，2026-09 新录 8 个视频，且是
  **"长短毛刷刷洗 insert/withdraw 专项"**——即 test 是**跨录制批次 + 面向刷洗动作定向构造**的集合。

因此本任务本质是**跨批次、跨目的泛化**：用通用 08 批次的模型去测 09 批次的刷洗专项视频。
这既解释了 insert 占比翻倍，也解释了 §4.4 里"`scope_mid_section` 出现率与 insert 的关系在
train 与 test 上方向相反"——**特征与标签的对应关系本身不跨批次稳定**。

## 4. 尝试清单（全部可复现）

### 4.1 解码轴（训练无关）

| 尝试 | test acc | Δ vs 53.92 | 备注 |
|---|---:|---:|---|
| 逐帧 argmax（基线） | 53.92 | — | 8 seed 中位 |
| 8 seed 概率集成 | 54.11 | +0.19 | |
| 集成 + 中值滤波 k=15 | 54.57 | +0.65 | |
| 标签级多数投票集成 | 54.00 | +0.08 | |
| 动作类统一加偏置 s（**test 上扫最优**） | 55.85 | +1.93 | 泄漏上界 |
| 逐类偏置（**test 上坐标上升拟合**） | **57.52** | **+2.86** | **泄漏上界 = 校准轴的天花板** |
| EM 无标签先验自适应（Saerens） | 55.74 | +0.45 | 单 seed，未泛化 |
| 真值先验重加权（**用 test 真值先验**） | 55.51 | +0.22 | 泄漏 |
| val 上贪心逐类否决 | 54.62 | +0.70 | val 坍塌到全 idle |
| **否决 `withdraw`（val 依据：8/8 seed val recall≈0）** | **55.32** | **+1.40** | 可交付 |
| 否决 `withdraw` + `sbc` | 57.09 | +3.16 | `sbc` 部分**无 val 依据**，不计入可交付 |
| val 选点最优组合（否决 withdraw + mindur15） | 55.40 | +1.48 | 纯 val 口径最好 |

### 4.2 训练轴（8 seed × 60 epoch，同预算）

| 臂 | test acc | Δ | best epoch | 说明 |
|---|---:|---:|---|---|
| 旗舰基线（s4l10 h128） | 53.92 | — | 28~59 | |
| `model.dropout=0.5` | **54.49** | +0.57 | 46~57 | 最好的一条，仍在噪声地板内 |
| `weight_decay=0.01` | 54.00 | +0.08 | — | |
| `model.tmse_weight=0.0` | 52.65 | −1.27 | — | 去掉 T-MSE 反而更差 |
| `hidden=32, s2l5`（默认类权） | 51.97 | −1.95 | — | 小模型更差（与容量轴结论一致） |
| `lr=2e-4` | 53.62 | −0.30 | — | |
| `augmentation.target_mask` frame_dropout p=0.5 | 50.68 | −3.24 | — | 强遮罩显著有害 |

训练侧最直接的证据：旗舰的 **train_loss 0.29 vs val_loss 12.2**（过拟合极强），
但**所有抗过拟合手段都没能换来 acc 提升**——说明瓶颈不是过拟合。

### 4.3 特征轴：更丰富的框特征

当前正式契约每 (检测类, 区域) 只有 `[presence, count, max_area]`。本轮直接从原始检测框
构造更丰富的描述子并配强分类器（train 拟合 / test 只读）：

| 特征 | 模型 | test acc | Δ vs 全 idle |
|---|---|---:|---:|
| 144 维 ROI（正式契约） | 逐帧 LDA | 49.15 | −6.10 |
| 144 维 ROI | LogReg +ctx±15 | 54.64 | −0.61 |
| 144 维 ROI | HistGB | 53.66 | −1.59 |
| 144 维 ROI | RandomForest +ctx±3 | 55.32 | **+0.07** |
| 144 维 ROI | MLP(256) | 53.13 | −2.12 |
| **全局 8 类 × 8 统计量（64 维）** | HistGB | **55.97** | **+0.72** |
| 64 维 + Δ(±1,±3)（192 维） | HistGB | 55.29 | +0.04 |

**没有任何特征/模型组合显著超过全 idle 基线。**

### 4.4 决定性诊断：检测类与动作类是解耦的

直接统计原始框的**逐类出现率**（train / test）：

| 动作 | 帧数 | hand | scope_control | scope_mid | scope_distal | syringe | air_gun | short_brush | brush_tip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| idle | 6274 | 94 | 92 | 73 | 9 | 2 | 0 | 1 | 0 |
| flush | 893 | 94 | 98 | 78 | 10 | 6 | 1 | 0 | 0 |
| insert | 1289 | 98 | 90 | **53** | 2 | 2 | 0 | 1 | 0 |
| withdraw | 437 | 98 | 94 | **45** | 4 | 0 | 0 | 1 | 0 |
| sbc | 488 | 97 | 96 | 72 | 3 | 1 | 0 | **12** | 0 |

test 上同样形态（idle 的 `scope_mid` 80% vs insert 83%）。

**8 个检测类里，只有 `short_brush` 对应一个动作类（`short_brush_cleaning`），
而定义 `long_brush_insert / withdraw` 的"长刷"根本没有检测类**；`hand`、`scope_*`
在所有动作下几乎恒在。这从数据层解释了为什么：
模型、树模型、线性模型、LDA 全部卡在全 idle 附近——**输入里没有判别动作所需要的信息**。

## 5. 结论

1. **+5pp 在现行输入源下不可达**。三条独立证据：
   - 用 test 标签拟合逐类偏置（最强的校准手段）上界只有 **57.52**；
   - 20 余次训练/解码/特征尝试的最好 test 读数是 56.69，且其中含 test 选臂；
   - 检测类的出现率在动作间几乎不区分，任何分类器都无法超过全 idle 基线。
2. **可交付的真实增益**（纯 val 选点）：`acc 53.92 → 55.40（+1.48pp）`，
   由"否决 `long_brush_withdraw` + min-duration 15 帧"构成；去掉否决只剩 +0.65pp。
3. **不要再用 `acc` 单独判读模型质量**（与 `CAPACITY_SUMMARY.md` 的既有警告一致）。
   本轮进一步量化了它：`acc` 的可用区间几乎完全由类别先验决定，模型差异被先验噪声淹没。

## 6. 下一步（按预期收益排序）

| 序 | 事项 | 依据 | 阻塞 |
|---|---|---|---|
| **0** | **把手动数据集 `cleansight-ActionMixed`（32 视频 / 9532 帧 / 动作占 75%）并入训练** | 本轮已完成可行性核验（见 §6.1）：它提供**额外 32 个独立视频**（与 auto 的 26 个视频 id **零交集**）、不同录制批次、且**动作占 75%**（auto 只有 34.5%）——正对 §3 的"跨批次 + 先验偏 idle"两个失效模式 | 无（可直接做） |
| 1 | **给"长刷"等定义动作的器械加检测类** | §4.4：动作与检测类解耦，`insert/withdraw` 占 test 36.6% 的帧却没有对应检测类 | 需要标注（数据组） |
| 2 | **引入像素信息**（逐帧图像 embedding / 光流方向） | 方向信息不在框统计里（既有结论：`insert↔withdraw` test AUC 0.32~0.57）；本项目已有 embedding 契约与提取工具，但 auto 数据不含像素 | 需要视频源。**已核查**：`lhh010/cleansight-raw` 只有 LS 导出 JSON（无视频）；本项目 `datasets/` 下无 project-18 像素；Label Studio `49.234.120.241:8080` 可达（302）但下载需凭据 → **需用户提供视频或凭据** |
| 3 | 在拿到新输入源之前，把 `acc` 目标换成先验稳健口径 | §3：val/test 先验差 13pp，`acc` 在 val 上选点会系统偏 idle | 需用户确认口径 |
| 4 | 已可交付的解码改动（否决 + 平滑）落进评测链路 | §4.1：+1.48pp，零参数、训练无关 | 无 |

### 6.1 下一步（序 0）的可行性核验（本轮已实测）

`datasets/cleansight-ActionMixed`（catalog 里是 `temporal.actionmixed-v2`，40 维 bbox 契约）
**就是** ROI-144 所需的目录形态——本轮直接调用现成的
`framework.cleansight_eval/temporal/features/roi_bbox.py::build_roi_frame_features`
在它的 `frames/<split>/` 上跑通，输出 `[T, 144]`，标签覆盖 **100%**：

| split | 视频 | 帧 | 动作帧分布（id 0..5） |
|---|---:|---:|---|
| train | 14 | 5,993 | idle **25.9%** / 6.1 / 14.0 / insert 24.6 / withdraw 18.0 / sbc 11.3 |
| val | 11 | 2,082 | idle 23.5% / 10.0 / 16.0 / 22.0 / 10.4 / 18.1 |
| test | 7 | 1,457 | idle 27.3% / 7.9 / 17.9 / 39.3 / — / 7.6 |

- 与 auto 数据集的视频 id **零交集**（`comm -12` 结果 0），是**独立的 32 个视频**；
- 动作空间一致（id 1 在两边的差异只是 LS 更名 `air_injection → water_injection`，id 位置不变）；
- 其 idle 占比只有 **~25%**，而 auto test 是 55.2%、auto train 是 65.5%
  → 并入训练可直接把有效先验推向"更敢判动作"，对 §4.1 里所有"val 选点坍塌到全 idle"的失败模式
  是**结构性**的对策，而不是又一个超参微调。

执行方式（下一轮）：建一个合并数据根（`frames/`+`labels/` 下用符号链接拼两个来源，视频名不冲突）
→ 在 `framework/testsets.yaml` 登记合并 split → 跑 8 seed 对照（auto-only vs auto+manual）。
按本报告 §4.2 的成本，单臂 8 seed × 60 epoch 约 7 分钟（GPU）。

## 7. 复现

```bash
source ../CleanSightBackend/.venv/bin/activate
# 误差分解（读测试产物，不训练）
python tmp/acc-decomp/decompose.py "runs/gpu-batch/recheck-flagship/h128/mstcn2-*"
# 先验漂移 + 二分类 ROC
python tmp/acc-decomp/priors.py
# 缓存逐帧 logits（需要 GPU；之后所有解码实验都在缓存上做）
python tmp/acc-decomp/calib.py --glob "runs/gpu-batch/recheck-flagship/h128/mstcn2-*" --device cuda
# 解码轴：否决 / 集成 / 平滑（val 选点）
python tmp/acc-decomp/final_sweep.py --runs-dir runs/gpu-batch/recheck-flagship
# 校准轴泄漏上界（test 上拟合逐类偏置）
python tmp/acc-decomp/bias6.py
# 训练臂（本报告 §4.2 全部 6 条）
GROUP=A bash tmp/acc-decomp/batch.sh   # GROUP=B 跑另外三条
# 经典模型 / 丰富特征上界
python tmp/acc-decomp/classical.py --run runs/gpu-batch/recheck-flagship/h128/mstcn2-20260928-193230
python tmp/acc-decomp/strong.py
python tmp/acc-decomp/richfeat.py
```

产物：`runs/acc-push/{acc-drop05,acc-wd1e2,acc-lr2e4,acc-tmse0,acc-h32def,acc-aug-all}/`
（每臂 8 seed，含 `CAPACITY_SUMMARY.md` 与逐 run `evals/`）。

**已知限制**：test 只有 8 个视频 / 2639 帧，跨 seed 摆幅 4.32pp；
本报告里凡标"泄漏上界"的数字**不可作为交付指标**；`sbc` 在 test 只有 51 帧（≈7 个采样帧），
其相关结论置信度低。
