# 输入设计提案：先量可分性，再改输入

> **状态：提案（2026-09-14），未跑任何训练对照，正文不含新的模型指标。**
> 本文只做两件事：①用可复跑的探针把"当前输入到底携带多少信息"量出来；
> ②据此给出本周值得投的 1~2 个方向，以及**有证据否掉**的方向。
> 全部数字来自 `tools/probe_input_features.py`（纯 CPU、只读 `labels/` + `frames/`，
> 几分钟跑完，命令见文末附录），数据集为 `datasets/cleansight-ActionMixed-auto-lhh`
> （v3，12,959 个标签帧）。
>
> 相关文档：[特征契约索引](./README.md)、[图像特征训练流程](./IMAGE_FEATURE_TRAINING.md)、
> [bbox 系策略对照](../FEATURE_STRATEGY_COMPARE.md)。

## 0. 结论先行

| # | 判断 | 依据（本文实测） |
|---|---|---|
| 1 | **当前输入的天花板由检测覆盖决定，不由特征工程决定**：8 个检测类里 4 个几乎不出现；判别 insert/withdraw 的 `brush_tip_out` 只出现 0.4% | §1.1、§1.2 |
| 2 | **insert vs withdraw 在当前输入下基本不可分**：7 个手写候选量帧级 AUC 全在 0.43~0.56，4 秒离线平滑后仍 ≈0.5 | §1.4 |
| 3 | **ROI-144 有 38% 的维度恒零，存活通道量纲跨 ~75 倍，且主线的 GRU 完全没有输入归一化**（MS-TCN 有） | §1.3 |

由此本周推荐两个方向：

- **P1（回答"视觉特征怎么 embedding"）**：把图像通道从"整帧压 224"改成"按检测框取 ROI"，
  本周交付 = 机制床上的**冻结特征线性探针对照**（1 小时级），用证据决定 E2/E3 是否立项。
- **P2（回答"归一化"+"特征怎么组装"）**：输入归一化对照（N0/N1/N2）+ ROI 维度预算重排
  （按实测可见性分配），两件都不依赖新数据，GPU 上各分钟级，当天可出多 seed 中位数。

**明确暂缓**：新增 bbox 运动/几何非线性块（速度、距离一阶差分、轴向投影）——§1.4 已给出否证
证据；把宝贵的人天先花在 P1/P2 与检测覆盖上。

## 1. 体检数据

### 1.1 检测覆盖（12,959 帧）

| 检测类 | 出现率 | 检测类 | 出现率 |
|---|---:|---|---:|
| hand | 95.2% | syringe | 2.0% |
| scope_control_body | 91.4% | short_brush | 2.0% |
| scope_mid_section | 69.8% | brush_tip_out | 0.4% |
| scope_distal_end | 7.9% | air_gun | 0.3% |

每帧框数分布（中位 4）：`{0:12, 1:134, 2:1325, 3:4700, 4:6001, 5:772, 6:15}`。

**读法**：特征里"有 8 个类"是契约写法，实际**只有 hand / scope_control_body / scope_mid_section
三类是稠密信号**，其余 5 类合计贡献不到 11% 的帧。这不是特征工程的锅，是检测召回的天花板。

### 1.2 动作 × 检测类出现率（当前输入"看得见"什么）

| action | 帧数 | hand | scope_ctl | scope_mid | scope_distal | syringe | air_gun | short_brush | brush_tip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| idle | 8528 | 94.6% | 89.7% | 70.5% | 8.4% | 1.8% | 0.4% | 1.5% | 0.4% |
| water_injection | 211 | 100.0% | 98.1% | **97.6%** | 17.5% | **7.1%** | 2.4% | 0.0% | 0.0% |
| flush | 1083 | 95.1% | **98.8%** | 77.0% | 9.4% | 5.5% | 0.5% | 0.3% | 0.0% |
| long_brush_insert | 1784 | 94.9% | 90.4% | **60.0%** | 5.8% | 1.4% | 0.1% | 0.7% | 0.8% |
| long_brush_withdraw | 643 | 98.3% | 94.7% | **56.8%** | 6.7% | 0.3% | 0.0% | 0.9% | 0.3% |
| short_brush_cleaning | 710 | 97.9% | 97.5% | 78.9% | 2.4% | 0.7% | 0.0% | **15.6%** | 0.0% |

**读法**：
- 有真实富集的信号：`syringe` 在 water_injection 上 7.1% vs idle 1.8%（4×）；`short_brush` 在
  sb_cleaning 上 15.6% vs idle 1.5%（10×）——**稀疏但是真信号**，值得保住。
- `brush_tip_out` 在 insert 上 0.8% vs idle 0.4%（2×）——**基本等于噪声**，而它正是长刷插入
  定义物；`scope_mid` 在 insert/withdraw 上反而是全库最低（60.0%/56.8%，idle 70.5%），
  说明"刷子进去时镜身中段被遮挡"是这两个动作唯一稳定的可见痕迹。
- insert 与 withdraw 两行的所有列几乎一样 → §1.4 的不可分结论在覆盖层面就已经注定。

### 1.3 通道体检（ROI-144，train 9,142 帧）

| 指标 | 数值 |
|---|---|
| 恒零通道 | **54 / 144 = 38%** |
| 存活通道 std | p25 = 0.0096，中位 = 0.033，max = 0.716（**max/p25 ≈ 74×**） |
| MS-TCN 归一化口径下 z-score | max\|z\| = **95.6**，217 个格子 \|z\| > 20 |

逐类活跃维（每类 18 维）：hand 18/18、scope_control_body 15/18、scope_mid_section 15/18、
scope_distal_end 12/18、syringe 12/18、**short_brush 9/18、air_gun 6/18、brush_tip_out 3/18**。

**读法**：
- `presence/count` 通道量纲是 0.1~0.7，`max_area` 通道量纲是 1e-4~1e-2 ——同一向量内跨两个
  数量级。在 `weight_decay=1e-4` 下，小量纲通道需要更大的权重才能被同等利用，而 wd 对大权重
  惩罚更重，**等价于系统性压制 `max_area` 通道**（这是待验证的机制假设，不是结论）。
- 主线 GRU/Transformer 直接吃原始量纲，**只有 MS-TCN 内部做 z-score**（`models/mstcn.py`
  `fit_normalization`）——即 §3.3 的架构对照历来是"一边归一化、一边没归一化"，结论不可比。
- MS-TCN 的守卫是 `std < 1e-4 → 1.0`，本契约下已出现 \|z\|=95.6（近零方差通道被放大）。
  若给 GRU 加同样的归一化，**不能照抄这个守卫**。

### 1.4 可分性：class 3（insert, n=1784）vs class 4（withdraw, n=643）

| 候选量（bbox 能推出的非线性量） | AUC | AUC（4s 离线平滑） | 中位（insert / withdraw） |
|---|---:|---:|---|
| hand↔scope_mid 距离 | 0.543 | 0.560 | 0.171 / 0.159 |
| hand↔scope_control 距离 | 0.445 | 0.455 | 0.087 / 0.101 |
| scope_mid 图像 y 坐标 | 0.458 | 0.456 | 0.360 / 0.363 |
| scope 长度（control↔mid） | 0.558 | 0.546 | 0.223 / 0.179 |
| hand 面积 | 0.533 | 0.519 | 0.042 / 0.039 |
| scope_mid 一阶差分 | 0.476 | 0.434 | −0.00004 / 0.00039 |
| hand 在 scope 轴上的投影 | 0.469 | 0.428 | 0.040 / 0.029 |

**读法**：全部候选量 AUC ≈ 0.5（最好 0.558），**离线居中平滑 31 帧（≈4 秒）后也没有改善**。
即"手在向镜身靠近还是远离"这类运动学量，在当前检测覆盖下**帧级信噪比不足以支撑判别**；
把它显式加进输入，大概率只是给 13 个训练视频加了若干噪声维度。

> 口径说明：AUC 只度量单变量可分性，AUC≈0.5 不等于"模型一定学不出来"（多变量组合仍可能
> 有信息）；但它足以否掉"把该量显式加进输入"这一类提案。这也与既有实测一致：
> `FEATURE_STRATEGY_COMPARE.md` 记录 insert/withdraw 的逐类 recall 跨 seed 为 0~56%、
> 大多数 seed 为 0。
>
> 另一个尚未排除的解释是**标注本身**：建议花 10 分钟看 2~3 段 insert/withdraw 视频，
> 确认时间轴边界与可见动作一致（零成本排除项，优先做）。

## 2. 本周推荐

### P1：图像通道按 ROI 取，而不是整帧挤压（回答"视觉特征如何 embedding"）

**假设**：E1 的"段级略好、逐类 recall 崩"不是"图像没用"，而是**输入的尺度取错了**。

**依据**：
- 判别物在 224×224 输入里小到看不见。按面积换算等效边长：hand ≈ 38 px、
  brush_tip_out ≈ 14 px、short_brush ≈ 11 px、**scope_distal_end ≈ 9 px**。
  整帧 resize 后，模型看到的主要是背景、光照与镜身外观——这与 E1 实测（段级 edit
  42.49→45.22、F1@0.25 28.00→34.62 略优；但 frame macro-F1 46.63→35.66、
  `air_injection` recall 86.96→0.00、`long_brush_insert` 55.94→5.59 崩塌）完全吻合：
  整帧 embedding 学到的更像"场景/光照"，在 6k 训练帧上过拟合，换视频即失效。
- bbox 侧已被 §1.4 证明无 insert/withdraw 信号 → **像素是唯一还没被证伪的信息源**。

**设计（若立项）**：固定语义槽位、每帧少量 crop，而不是 8 类全裁（4 类 <2% 出现率，
裁出来也是背景）：
- 槽 1 = hand 区域（复用 `hand_bbox.py` 的最大 hand 框 ×1.5 扩张锚点规则，口径一致性优先）；
- 槽 2 = scope 区域（`scope_control_body ∪ scope_mid_section ∪ scope_distal_end` 并集框 ×1.2）；
- 缺失槽 → 零向量 + 一个 presence 标志（沿用"空帧全零"语义）；每帧 K 次冻结 backbone 前向
  （先做简单版；后续可换一次 spatial map + RoI pooling 省算力）；
- 因果、无状态、只看当前帧；新契约 `actionmixed-bbox-embed-roi-<backbone>-v1`，
  `mask_targets` 只作用 bbox 块（沿用既有 tail-block 约定，见 §4 工程注意）。

**本周交付（1 小时级，不训练时序模型）**：在机制床 `datasets/cleansight-ActionMixed`
（9,532 帧，图/框/标签齐全，整帧 mobilenet_v3_small embedding 已预计算）上做
**冻结特征 + 线性探针**对照：

| 臂 | 特征 |
|---|---|
| A | 整帧 224² embedding（= E1 现状，576 维） |
| B | hand crop embedding（576 维） |
| C | hand + scope 两槽 concat（1152 维） |
| D | bbox-144（ROI 契约，作为"当前主线看不看得见"的参照） |

指标：帧级 macro-F1 + 逐类 recall（重点 insert / withdraw / sb_cleaning）。

**判据**：若 B/C 在 insert/withdraw/sb_cleaning 上明显高于 A 且 ≥ D → ROI-crop embedding 立项
（E2）；若 B/C ≈ A ≈ D → 图像方向整体降权，省下 E2/E3 的人天，把力气转回检测覆盖。

**前置/风险**：机制床与正式 v3 不同源（结论只能定性）；正式 v3 无像素（`IMAGE_FEATURE_TRAINING.md`
§5.1 核心阻塞），要走正式通道需先解决图像源。

### P2：输入向量本身的体检与重排（回答"归一化"+"怎么组装"）

两件独立、都不依赖新数据、都能当天出多 seed 中位数（ROI 特征下 CPU ≈9 s/epoch、GPU ≈2 s/epoch）。

**P2a 输入归一化对照**（改模型输入层，契约不变，不需要新数据集登记）

| 臂 | 内容 |
|---|---|
| N0 | 现状（原样喂入，ROI-144 基线数字已有） |
| N1 | 训练集逐维 z-score（复用 `mstcn.fit_normalization` 模式；**守卫提到 1e-3 或对 z 做 clip(±5)**，理由见 §1.3） |
| N2 | 固定契约量纲对齐（`count/3`、`sqrt(area)`、坐标去均值——不含拟合统计，部署最简） |

判据：段级 edit / F1@0.1~0.5 中位数 + 坍缩 seed 数；附带产出首层权重范数对比，验证或否掉
"wd 压制 max_area 通道"的机制假设。N1 vs N2 的差异回答"起作用的是统计归一化，还是量纲对齐"。

**P2b ROI 维度预算重排（roi-grid-v2）**

依据：54/144 恒零；hand 18/18 活跃而 brush_tip_out 只有 3/18。当前"8 类 × 6 区域 × 3 通道"
是**按类别表整齐分配**，不是按实测可见性分配。

设计：高频类（hand 双实例 / scope_control_body / scope_mid_section）细化网格（3×3 或 4×4，
提高空间分辨率），低频 4 类（scope_distal_end / syringe / air_gun / short_brush / brush_tip_out）
合并为**全局 1 区域** `[presence, count, max_area]`。总维度可保持 ~144 或更低。

**为什么不直接删低频类**：它们在关键动作上有 4~10× 富集（§1.2），信号真实但稀疏；
散在 6 个区域时每格几乎恒零，合并成全局通道反而更稠密、更抗漏检。

判据：段级指标中位数 + 逐类 recall（尤其靠 hand+scope 区分的水相关类 flush / sb_cleaning）。

## 3. 暂缓与否掉的方向（附证据，省队友人天）

| 方向 | 状态 | 依据/解锁条件 |
|---|---|---|
| bbox 运动/几何非线性块（速度、Δ距离、轴向投影、pair 特征） | **暂缓** | §1.4：AUC 0.43~0.56，平滑后仍 ≈0.5。仓库历史设计（legacy 64 维含 `dcx/dcy`）与 CLEAN 离线特征（`clean_bbox_v2` 的 speed/pair/priors）都做过，但那套口径服务于后端 CLEAN 模型，未在本数据上验证过增益 |
| 参考系归一化（以 hand/scope 为原点的坐标系） | **暂缓** | 理论上更"语义"，但几何量本身在当前覆盖下无判别力（§1.4）；等 P1/P2 结论后再评估 |
| 置信度（conf）通道复活 | **排队** | 真实且便宜：`outputs/annotations/*.json` 的轨迹逐帧带 `conf`，`convert.py` 只写 5 列把它丢了。但单独加 conf 不解决覆盖问题——应与"逐类降阈值重建检测"一起做 |
| 检测侧扩召回（逐类降阈值 / 换 g2 权重 / 补帧） | **最高天花板，但有资源前置** | `auto-annotate.yaml` 里已有注释掉的逐类阈值示例（`syringe: 0.1` 等）。本 checkout 缺视频与 `yolo11s-g2-v1` 权重（`legacy/yolo-detection/pipeline/versioned_weights/` 只有 large-v1/v2/v3 与 small-v3），需在 **-lhh 机器**上执行 |

## 4. 工程注意（给后续任何 tail block 改动）

`apply_target_mask_augmentation` 目前按 `特征维 ÷ 检测类数` 推导每类块宽。ROI-144 下块宽 18；
若在其后追加运动块使总维变成 176（= 8 × 22）**能被整除**，会**静默错切**——不是报错，
而是遮错列。任何 tail block 都必须像图像 embedding 契约那样**显式声明尾部宽度**
（`feature_schema` 里给 `tail_dim`/`motion_dim` 之类的显式字段），让遮罩只作用前缀块。
这条是 `IMAGE_FEATURE_TRAINING.md` §3.5 已有的教训，本次预算重排同样适用。

## 5. 附录：复现命令

```bash
# 全部体检数字（检测覆盖 / 动作签名 / 通道量纲 / 可分性）
/home/caizh/programming/python_code/CleanSightBackend/.venv/bin/python \
  tools/probe_input_features.py --root datasets/cleansight-ActionMixed-auto-lhh \
  --json tmp/probe_v3.json

# 换类别对（默认 3,4 = long_brush_insert vs long_brush_withdraw）
python tools/probe_input_features.py --root <数据集> --pair 1,5
```

探针只读 `labels/` 与 `frames/`，纯 CPU，不训练，不改任何数据。
