# 实验报告：帧准确率攻关第 8 轮 —— "学习式表示"路线整体否定（2026-09-29）

> **目标**：`acc` 53.92 → 58.92。
> **起点**：`roi-grid-v4`（通道语义替换，128 维）单 seed **57.59**，全栈（+集成 +val 选点平滑）**59.15**。
> **本轮结果**：把前 7 轮都没碰过的**"不再用手工 ROID 直方图、让模型自己学表示"**这条路线完整测掉，
> **两条都否定**——手工 ROI 先验在这个数据规模上打不过，且原始框信息也没有直方图之外的增量。

---

## 1. 为什么做这条路线

前 7 轮全部在**手工 ROI 直方图**里腾挪（网格 2×3→3×3→4×4、通道 `p,c,a`→`c,a,w,h`、
维度预算分配、格内亚格位置），已系统性饱和。直方图有一个**结构性损失**：

> 它把每帧压成"(检测类, 区域) 的计数与极值"，**丢掉了框与框之间的关系**
> （哪些类同时出现、谁大谁小、谁在谁旁边）。

于是本轮把**每帧的原始框集合**直接交给模型，由**可学习、置换不变**的编码器自行学表示。

## 2. 新增的两个契约与一个模型（均已登记 + 单测）

| 产物 | 内容 |
|---|---|
| `features/boxset.py`（`actionmixed-boxset-v1`，**96 维**） | 每帧面积最大的前 8 个框按面积降序填充；每槽 `[one-hot(8) | cx | cy | w | h]`=12 维。**填充判据内建**（空槽 one-hot 全零），坐标不标准化 |
| `models/boxset.py`（`model.type=boxset_mstcn2`） | 逐槽共享 MLP + **掩码 max/mean 池化**（置换不变）→ 复用 `MSTCN2` 主干（多 stage 深监督 + T-MSE）；`forward` / `compute_loss` 转发 |
| `features/combo.py`（`actionmixed-combo-v1`，**224 维**） | 左 128 维 `roi-grid-v4` ++ 右 96 维框集合；用于回答"原始框有没有直方图之外的增量" |

**工程侧**：catalog 新增两个契约族的**声明式维度校验**（`BOXSET_FEATURE_MAPPING_PREFIX`：
槽位数 ×(类数+4)；`COMBO_FEATURE_MAPPING_PREFIX`：`feature_parts` 分部和），
`HIDDEN_HOOKS` 支持**点路径**（`core.refines`）以便给包装型模型挂 penultimate 表征。
**单测 8 条**，含**置换不变性**（槽顺序无关）与空帧有限性。

## 3. 结果：两条都否定

| 臂 | 契约 | 模型 | micro acc 中位 | 摆幅 | vs 基线 53.92 |
|---|---|---:|---:|---:|---:|
| `acc-roiv4`（参照） | roi-grid-v4 128 | mstcn2 | **57.59** | **2.77** | +3.67 |
| **`acc-boxset`** | boxset 96 | **boxset_mstcn2** | **45.02** | 9.09 | **−7.84（0/8，p=0.0078）** |
| **`acc-combo`** | combo 224 | mstcn2 | **57.67** | 4.50 | +3.75（vs v4 **−1.29，3/5，p=0.8438**） |

### 3.1 学习式集合编码器：显著更差（−7.84pp，0/8）

`boxset_mstcn2` 的逐类召回：idle 75.7%、insert 8.5%、withdraw 8.5%、flush 0%。
**它连"每类在哪儿、有几个、多大"这层先验都要从零学，而训练集只有 14 个视频 / 9,575 帧。**
手工 ROI 直方图把这层先验**编码进了特征**，这正是它在小数据下的价值。

### 3.2 拼接：与 v4 无显著差异 → **原始框没有可测增量**

`combo`（= v4 + 原始框集合）中位 57.67 与 v4 的 57.59 几乎相同，而**配对检验 Δ 中位 −1.29pp、
胜/负 3/5、p=0.8438**——即**两者不可分**。
→ **直方图并没有"丢掉有用的东西"**：框间关系在本任务的数据规模下没有带来可测增益。
这同时反过来说明 3.1 的失败是**优化/先验问题**，不是"信息被丢掉"。

## 4. 第 8 轮后的完整状态（不变）

| 口径 | 读数 | Δ vs 基线 53.92 | 达标 |
|---|---:|---:|---|
| 严格：单 seed、仓库标准评测（argmax） | **57.59** | **+3.67** | ❌ |
| 单 seed + `train` 源域校准 | 58.24 | +4.32 | ❌ |
| 集成 + 源域校准 | 58.62 | +4.70 | ❌ |
| **集成 + val 选点 mindur15（全栈）** | **59.15** | **+5.23** | ✅ |
| 目标 | 58.92 | +5.00 | — |
| 平凡基线（全 idle） | 55.25 | +1.33 | — |

## 5. 已排除的路线汇总（8 轮）

| 轴 | 结论 |
|---|---|
| 时序架构（8 类） | 不是杠杆（架构调研轮） |
| 解码 / 校准 / 集成 / 平滑 | 单点 ≤ +1.5pp；`val` 拟合校准有害、`train` 源域校准 +0.64pp |
| 训练（正则 / 损失 / 容量 / 预算 / 选点口径） | 全部 ≤ +0.6pp 或负向 |
| 更多数据（跨来源合并） | −6.17pp 负迁移 |
| 手工特征布局（21 个）/ 通道子集 / 通道语义 | **有效**：`roi-grid-v4` +3.67pp（8/0，p=0.0078） |
| 格内亚格位置（v5） | 无增益 |
| **学习式集合编码器（本轮）** | **−7.84pp，否定** |
| **原始框拼接（本轮）** | **与 v4 不可分，无增量** |
| 输入源（像素 / 目标批次重标注） | 未试（需资源） |

## 6. 复现

```bash
source ../CleanSightBackend/.venv/bin/activate
# 学习式集合编码器
python tools/run_capacity_matrix.py --runs-dir runs/acc-push/acc-boxset \
  --config mstcn-actionmixed-auto-roi.yaml --hidden 128 --seeds 42,7,2026,1,2,3,4,5 \
  --best-metric val_edit --epochs 60 --no-probe-baseline --set train.lr=0.0005 \
  --set model.type=boxset_mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
  --set model.input_dim=96 --set feature_schema.dim=96 \
  --set feature_schema.version=actionmixed-boxset-v1 \
  --set data.dataset_ref=temporal.actionmixed-auto-boxset-v1
# 拼接对照：把上面两条 --set 换成 combo 契约 + model.type=mstcn2 + input_dim=224
pytest framework/tests/test_boxset.py -q     # 8 条（含置换不变性）
```

**新增产物**：`features/boxset.py`、`features/combo.py`、`models/boxset.py`；
登记 `temporal.actionmixed-auto-{boxset,combo}-v1`；单测 `framework/tests/test_boxset.py`（8 条）；
catalog 两个新契约族校验分支；`HIDDEN_HOOKS` 点路径支持。

**验证**：`validate_testsets.py` 全量 `ok: true / errors: []`；
全量回归 **469 passed / 7 failed**——其中 7 项均为**预先存在**的失败；
本轮曾因新模型未登记 penultimate 挂钩点而使失败数升到 8，**已按测试要求修复**（这正是该测试的目的）。

**已知限制**：test 仅 8 视频 / 2639 帧、val 仅 4 视频；框集合契约只保留面积最大的前 8 个框
（auto 每帧平均 3.5 个框，绝大多数帧无损）；两轮源域筛选均显示 **+1pp 量级差异噪声过大**，
不足以单独作为选型依据。
