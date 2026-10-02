# 帧准确率攻关 12 轮浓缩报告（2026-09-28 ~ 2026-09-29）

> **一句话**：12 轮、16 条轴、约 100 次训练对照之后，**唯一有效杠杆是"特征契约的通道语义 + 维度预算分配"**：
> 帧准确率（`acc`）从 **53.92 → 58.31**（32 seed 同口径，配对 **32/0，p<1e-6**，**+4.41pp**），
> 距 **+5pp** 目标还差约 **0.5pp**；其余 15 条轴全部 ≤ +0.6pp 或负向。
> 详细过程见 [`EXPERIMENT_REPORT_FRAME_ACC_R2…R12_20260929.md`](EXPERIMENT_REPORT_FRAME_ACC_R4_20260929.md) 与第 1 轮的 [`EXPERIMENT_REPORT_FRAME_ACC_20260928.md`](EXPERIMENT_REPORT_FRAME_ACC_20260928.md)。

---

## 1. 目标与口径

| 项 | 值 |
|---|---|
| 目标 | 时序模型**帧检测准确率** `acc`（6 类逐帧 micro，`accuracy/frame-wise-micro-across-items/percent/v3`）**提升 ≥5pp** |
| 基线 | `mstcn2` s4l10 h128 + `roi-grid-v1`(144 维)，**acc 53.92**（8 seed 中位；32 seed 复测 **53.90**） |
| **平凡基线（全 idle 常数预测器）** | **55.25** —— 基线**低于它** |
| 数据 | `temporal.actionmixed-auto-roi-v1`，train 14 / val 4 / test 8 视频（test 2,639 帧，LS project-18） |
| 设备 | RTX 4060 Laptop（沙盒内 GPU 可用） |

**test 批次是"跨批次 + 定向构造"的**：train/val 来自 LS `project-16`（2026-08），
test 来自 `project-18`（2026-09，**"长短毛刷"专项**）——idle 占比 65.5 / 68.2 / **55.2%**，
`insert` 占比 13.5 / 14.6 / **26.8%**。这条先验差是全系列多次误判的根源。

## 2. 最终定标（32 seed，两臂同协议）

| 口径 | 基线 `roi-grid-v1` | **`roi-grid-v4`** | 同口径改进 |
|---|---:|---:|---:|
| **单模型严格**（仓库标准评测，逐帧 argmax） | 53.90 | **58.31** | **+4.41pp**（配对 **+4.47**，**32/0**，p<1e-6） |
| 全栈（32 成员集成 + val 选点最小时长） | 54.83 | **59.30** | **+4.47pp** |

- **同口径改进稳定在 +4.4~4.5pp，与推理协议无关**（集成/平滑对两臂增益相近）；
- 全栈的**绝对值 59.30 高于目标 58.92**，但**基线在同协议下是 54.83**——
  "59.30 − 53.92 = +5.38pp" **混用了两套协议，不应引用**；
- 32-seed 中位数 bootstrap 95% CI：`v4` **[57.50, 58.85]**；配对 Δ 的 CI **[+3.33, +5.11]**。

## 3. 有效杠杆：特征契约演进（唯一一条）

> 机制：**把特征维度预算从"按类别表整齐分配"改为"按检测证据在哪儿分配"。**
> auto 数据里 5 个稀有器械类的框只有人工标注的 **5%~21%**，却占掉了 v1 的 90/144 维。

| 契约 | 布局 | 通道 | 维度 | acc（8 seed A） | 结论 |
|---|---|---|---:|---:|---|
| `roi-grid-v1`（原基线） | 8 类统一 2×3 | presence,count,max_area | 144 | 53.92 | — |
| **`roi-grid-v2`** | 高频 3 类 3×3、低频 5 类全局 | 同上 | 96 | **56.48** | **+2.56pp（8/0）** |
| `roi-grid-v3` | 高频 3 类 4×4 | 同上 | 159 | 49.30 | −4.62pp（更细反而崩） |
| **`roi-grid-v4`（最终采用）** | 高频 3 类 3×3、低频全局 | **count,max_area,w,h** | 128 | **57.59** | **相对 v1 +3.67pp（8/0，p=0.0078）** |
| `roi-grid-v5` | 同 v4 + 格内 dx,dy | | 192 | 57.05 | 无增益（配对 p=0.55） |
| `boxset-v1`（原始框集合）+ 集合编码器模型 | 每帧前 8 个框 | one-hot,cx,cy,w,h | 96 | 45.02 | **−7.84pp（0/8）** |
| `combo-v1`（v4 ++ 框集合） | 拼接 | | 224 | 57.67 | 与 v4 不可分（p=0.84） |
| `bbox-global-hand` | 全局 + 手部区域 | | 80 | 49.38 | −4.55pp |

**v4 的关键发现**：`presence ≡ count>0` **完全冗余**（剔除后 30/30 折逐位相同），
于是把该通道位换成**最大面积框的宽、高**——补回 `max_area = w×h` 丢掉的**长宽比**。

**边界已高精度扫完**（30 折逐折配对，分辨力 ≤0.4pp）：

| 轴 | 候选数 | 结果 |
|---|---:|---|
| 高频网格（2×2~6×6、3×4/4×3、逐类非均匀） | 17 | 无候选优于 v4（最好 +0.06pp，p=0.51） |
| 通道集合（含 sum/mean/长宽比/次大框面积/多实例指示） | 11 | 无候选显著优于 v4（最好 +0.37pp，p=0.47） |
| 低频类布局（1×1/2×2/3×3） | 3 | 1×1 最优 |
| 低频类是否保留 | 2 | 保留更好（遮掉 −0.91pp，p=0.20） |

## 4. 已排除的 16 条轴

| 轴 | 最好结果 |
|---|---|
| 时序架构（GRU/TCN/Transformer/MS-TCN/++/ASFormer/actionness/FACT） | 无一在段结构不崩前提下胜出 |
| 解码（S-NCM、约束 Viterbi、类别否决、中值/最小时长） | ≤ +1.5pp |
| 校准（val 拟合 / train 拟合 / EM 无标签先验） | val 拟合 **−3.01pp**；train 拟合 +0.64pp；EM +0.30pp |
| 集成（同契约 / 跨契约 / 扩到 16 成员） | ≤ +0.97pp，且 val 多不支持 |
| 跨 seed **权重平均** | **坍缩到 55.25（= 平凡基线）** |
| 正则（dropout / weight decay） | ≤ +0.6pp |
| 损失（T-MSE / transition 重加权 / **标签平滑**） | 标签平滑 **−0.84pp（p=0.0035）** |
| 容量（h32/h64/h128/h256） | 加参数无用 |
| 预算（60 / 120 轮） | −0.45pp |
| 学习率调度（cosine / warmup / 退火到 0） | 32 seed 下 **17/15，p=0.81**，无差异 |
| 选点口径（`val_edit` / `val_acc` / `val_f1@0.1` / `val_f1@0.25`） | 换口径在 val 上更差 |
| 更多数据（并入手动数据集 16 视频 / 9,532 帧） | **−6.17pp 负迁移** |
| 学习式表示（集合编码器） | **−7.84pp（0/8）** |
| 原始框拼接 | 与 v4 不可分（p=0.84） |
| 格内亚格位置（dx,dy） | 无增益（p=0.55） |
| 布局 × 通道联合 | 无候选胜出（分辨力 ≤0.4pp） |

## 5. 工程侧改动（默认值一律逐位等同历史行为，且有测试）

| 改动 | 位置 | 说明 |
|---|---|---|
| **学习率调度** | `train.lr_schedule`(`constant`/`cosine`) + `warmup_epochs` + `min_lr_ratio` | 流水线原先**全程恒定 lr、无任何调度**；倍率只依赖轮号，**resume 天然落在同一曲线** |
| **标签平滑** | `train.label_smoothing` | 与 `transition_loss_weight` **互斥**（同时开直接报错，不静默忽略）；实测否定，默认保持 0 |
| **新契约族校验** | `catalog.py`：`BOXSET_FEATURE_MAPPING_PREFIX`（槽位数×(类数+4)）、`COMBO_FEATURE_MAPPING_PREFIX`（`feature_parts` 分部和） | 与既有 ROI 分支同模式的**声明式维度校验** |
| **penultimate 挂钩点** | `inference.py` `HIDDEN_HOOKS` 支持**点路径**（`core.refines`） | 包装型模型（框集合编码器）也能挂表示层 |
| **新契约/模型** | `features/{roi_bbox_v3,v4,v5,boxset,combo}.py`、`models/boxset.py` | 均已登记 catalog + 写入 `usage/YAML_CONFIG.md` 与 `docs/features/README.md` |
| **新增测试** | `test_roi_features_v{3,4,5}.py`、`test_boxset.py`（含**置换不变性**）、`test_lr_schedule.py` | 累计新增 **32 条**（本轮系列内新增，全部通过） |

**回归状态**：`pytest framework/tests tests` → **473 passed / 7 failed**，7 项**均为改动前既有失败**（逐项核对过）；
`tools/validate_testsets.py` 全量 `ok: true / errors: []`。

## 6. 方法学教训与 4 处自我更正

| # | 更正 | 触发 |
|---|---|---|
| 1 | 第 2 轮"人工框 +7.01pp vs YOLO 框 −11.38pp → 分水岭在框来源"**量级作废**；"判别力 100% 来自稀有类通道"**撤回**（10 折配对 p=0.557） | 用了**单个** 12/4 划分 |
| 2 | 第 3 轮"降 conf 阈值没用"**作废**——那是**截断假象**（JSON 的非零最小 conf 恰为 0.2500） | 先测前提再断言 |
| 3 | 第 7 轮"val 拟合校准 +0.76pp"**作废**——把 seed 0 的拟合参数套用到全部 seed | **凡"拟合—应用"必须逐 seed 对齐** |
| 4 | 第 9 轮"warmup+cosine +0.59pp、降方差"**撤回**——32 seed 下 58.20 vs 58.31，17/15，p=0.81 | 16 seed 仍会误判 |

**三条可复用纪律**：

1. **8-seed 中位数噪声 ±0.7~1.3pp**（同一配方只换 seed 集：基线 53.92/54.66，v4 57.59/58.92）
   → **判 <1.5pp 的差异必须 ≥16 seed，判 <0.5pp 须 ≥32 seed**；
2. **凡"拟合—应用"必须逐 seed 对齐**，跨 seed 套用会造出假增益；
3. **"test 涨、val 不支持"的候选一律不计交付**——本系列有 4 个这样的候选被 val 否决。

## 7. 交付物与复现

**文档**：`docs/experiments/frame-acc/EXPERIMENT_REPORT_FRAME_ACC_20260928.md`（第 1 轮）+ `…_R2…R12_20260929.md`（11 份分轮报告）
+ 本浓缩报告 + 周简报 `docs/weeks/2026-09-26_2026-10-02/README.md`。

**运行产物**：`runs/acc-push/` 下 **37 个批次目录、376 个 run**（含 `acc-roiv{2,3,4,5}`、`acc-boxset`、`acc-combo`、
`acc-roiv4-{cosw3,ls005,nocold,cos-seedB,seedB,seedC}`、`acc-roiv1-seed{B,C}` 等），
每条臂 8~32 seed，均带 `CAPACITY_SUMMARY.md` 与逐 run `evals/`。

**金牌配置（复现最优点）**：

```bash
source ../CleanSightBackend/.venv/bin/activate
python tools/run_capacity_matrix.py --runs-dir runs/acc-push/champion \
  --config mstcn-actionmixed-auto-roi.yaml --hidden 128 \
  --seeds 42,7,2026,1,2,3,4,5,11,13,17,19,23,29,31,37 \
  --best-metric val_edit --epochs 60 --no-probe-baseline \
  --set train.lr=0.0005 --set model.type=mstcn2 --set model.num_stages=4 \
  --set model.num_layers=10 --set model.input_dim=128 \
  --set feature_schema.dim=128 --set feature_schema.version=actionmixed-roi-grid-v4 \
  --set data.dataset_ref=temporal.actionmixed-auto-roi-v4
```

**分析脚本**：`tmp/acc-decomp/`（`decompose / priors / calib / final_sweep / bias6 / classical / strong /
richfeat / lagscan{,2} / armselect / round4_final / multifold / batchshift{,2} / rareablate /
confrecall / confcurve / channelscreen{,2} / jointscreen / layoutscreen / crossens / decoding_v4 / perclass`）。

## 8. 阻塞与恢复条件

**阻塞**：like-for-like 口径下距 +5pp 还差约 **0.5pp**，而本地可试杠杆已全部排除；
32-seed 标准误约 0.3pp —— **继续加 seed 只会把 +4.4pp 收得更准，不会变成 +5pp**。

**恢复需要二选一**：

- **(a) 口径决策**：若认可"32 成员集成 + val 选点平滑"属于"目前的训练和模型"的一部分，
  则该协议下绝对值 **59.30 > 58.92**，可进入 CARD / `pin.yaml` / 交付报告整理；
- **(b) 提供资源**：目标批次（LS `project-16/18`）的**视频**或 **LS 凭据**——
  用于唯一未被证伪的方向：**换输入源**（逐帧像素特征，或对目标批次重新人工标注检测框）。
  **已核实**：本 checkout 无视频；ModelScope `cleansight-raw` 只有 LS 导出 JSON；
  LS `49.234.120.241:8080` 可达但下载需凭据。

**已知限制**：test 仅 8 视频 / 2,639 帧、val 仅 4 视频；本报告"无候选优于 v4"的结论分辨力为 **≥0.4pp**
（不能排除更小的真实增益）。

---

## 补记（2026-10-02）：三条限定 + 一次数据版本变更（引用本报告数字前必读）

### 1. 统计单元：seed 级 p 值不能当视频级用

本报告全部显著性都是 **seed 级配对**（n=32，p=7.9e-07）。同一批模型换到**视频级**统计：
逐视频 Δ 帧池化 **8/8 全正（p=0.0078）**、逐 seed 取中位 **7/8（符号检验 p=0.0703）**；
**单视频 `ac38ca6b` 约占增益一半**。→ **引用"提升"必须写明统计单元**。

### 2. 工作点：`acc` 的 +4.47pp 里约 0.9pp 来自"更保守的决策点"

两臂 argmax 落在**不同动作帧预算**上（v1 23.97% / v4 20.06%，真值 44.75%）。拉到**同一预算**：
**+3.56pp（29/3，p=1.4e-07）** 仍成立；要求"报得和真值一样多"时 **+0.98pp（p=0.087 不显著）**。
→ 同时注意本报告已记录的段级代价：`edit` **−8.32pp**（7/25，p=4.4e-05）、**`flush` 召回 7.93% → 0.00%**。

### 3. 分辨力：本报告的门槛表低估了所需 seed

实测配对设计的可检测效应：**8 seed = 1.91pp、16 = 1.35pp、32 = 0.95pp、128 = 0.48pp**
→ 原"< 0.5pp 用 ≥32 seed"应改为"**判 0.5pp 需 ≈128 seed**"。
另：以视频为簇算 `acc` 的 **ICC 0.014~0.018 → 设计效应 5.6~7.0 → 2639 帧只值 376~470 个独立样本**；
两条**独立** run 相比的最小可检测差 ≈ **9.1pp**——本系列能测出 +4.47pp **完全依赖 32-seed 配对**。

### 4. 数据版本变更（2026-10-01）：契约结论复现，但**段级指标系统性下降**

上游数据集更新为 **v3.1**（标注修订：#201 重标 66%、train 换 1 个视频、val 4→3；**test 8 视频逐字节未改**
→ 跨版本 test 同尺子）。在 v3.1 上重训两臂（16 对 seed）：

| | v3（本报告口径） | v3.1 | 变化 |
|---|---:|---:|---|
| 契约增益（v4 − v1） | +3.94pp（16/0） | **+3.77pp（16/0，p=3.05e-05）** | 复现 |
| `acc`（v4 契约） | 58.24 | 57.51 | −0.68（p=0.13 不显著） |
| **`edit`** | 43.09 | **37.36** | **−4.97（0/16，p=3.1e-05）** |
| `f1@0.25` | 32.31 | 29.80 | −3.60（p=0.018） |

→ **帧准确率对标签修订不敏感，但段级指标系统性变差 4~5pp**；**引用本报告任何绝对读数必须注明数据版本**。
另有一个**未解释**的回归：`insert` F1 **−11.11pp**（0/16 全负），而 train 的 insert 标签**一帧未改**；
已排除选点漂移（两版 best_epoch 中位同为 50.5）与监督变少（帧数相同）。详见
[`../../weeks/2026-09-26_2026-10-02/V31_DATA_UPDATE_20261001.md`](../../weeks/2026-09-26_2026-10-02/V31_DATA_UPDATE_20261001.md)。

### 5. 离线模型的上线边界（本系列全部结论的适用范围）

本报告所有读数的主语是**整段非因果 `mstcn2`**。把已训模型按"每帧只能看过去 W 帧"窗口化部署
（W=8~645，自检 8/8 通过）：**非 idle 预测率 18.30% → 0.00~1.17%**，准确率塌到平凡基线 **55.25%**
（W=128 时低于基线 1~2.8pp）→ **不是掉几个点，是功能消失**；流式上线必须换因果架构重训。
