# 特征方案 v4 设计：废弃目标剔除 · 命名对齐 · 指标体系重构（2026-09-17）

> 三点修改决策（维护人 2026-09-17 确认）：① scope_distal_end / short_brush 已废弃，
> 不作核心特征；② syringe / air_gun / brush_tip_out **保留**为事件线索类，但
> **不得用作定位判据**（不作坐标锚点/参考基准，只被 scope 轴定位）；③ v4 器械轴
> 主锚点 `control_body → mid_section`。第 4 点修改意见待补。
> 上游结果与演进史见 [`FEATURE_LAB.md`](FEATURE_LAB.md)；本方案为 feature_mapping
> 升版（v4），按纪律全部重训重评，不与旧版本混评。

## 1. 证据：检测类可靠性量化（train 9575 / val 3384 / test 2639 帧）

| 检测类 | train | val | test | 判读 |
|---|---:|---:|---:|---|
| hand | 94.8% | 96.1% | 97.9% | 核心可靠 |
| scope_control_body | 92.8% | 87.4% | 85.2% | 核心可靠 |
| scope_mid_section | 69.7% | 70.3% | 78.6% | 可靠，v4 主锚点 |
| **scope_distal_end** | 7.8% | 7.9% | 12.0% | ❌ 剔除：镜体常在画面却检测不出，通道近乎恒零 |
| **short_brush** | 1.7% | 3.0% | 0.4% | ❌ 剔除：刷洗动作期间也检测不出 |
| syringe | 2.4% | 0.7% | 1.8% | ⚠️ 保留：事件线索（只在注水动作出现） |
| air_gun | 0.3% | 0.4% | 0.6% | ⚠️ 保留：事件线索（冲洗） |
| brush_tip_out | 0.3% | 0.7% | 0.6% | ⚠️ 保留：事件线索（出刷） |

**稀疏≠废弃的原则**：distal/short_brush 是"应可见却检测不出"（噪声通道）；
syringe/air_gun/brush_tip_out 是"本就短暂出现"（presence≈0 → 出现即强判别信号）。

**器械轴可用性**：ctrl+mid 同帧共现 63~69%，而 distal 仅 8~12%——v3 的
`ctrl→distal` 主轴绝大多数时间在走回退链（实际生效的是 mid 轴）。
**v4 将 mid 轴转正**：主锚点可用率 8% → 65%+，回退链简化、可解释性更好。

## 2. v4 特征规格（契约名见 §3）

### 2.1 类集合与角色

| 角色 | 类 | 允许的编码 |
|---|---|---|
| 核心几何 | hand（双 slot）、scope_control_body、scope_mid_section | 完整几何编码（轴系坐标/面积比/速度/缺失龄） |
| 事件线索 | syringe、air_gun、brush_tip_out | presence + **被 scope 轴定位**的坐标 + 面积比；**禁止**：作轴锚点、作任何参考基准、作 pair 距离的基准端 |
| 剔除 | scope_distal_end、short_brush | 无任何通道 |

### 2.2 器械轴（v4）

- **主轴**：`control_body → mid_section`（沿轴单位向量 + 轴长）
- 回退链（依序）：ctrl+mid 同帧 → 前一帧轴方向 × 当前 ctrl 中心平移 → 前一帧完整轴
  （前向填充，带缺失龄）→ 竖直默认 (0,1)
- 面积参考基准：`mid_section` 面积（缺失回退 ctrl 面积）——v3 用 distal 面积，随剔除废止

### 2.3 块结构（实现落点 `features/clean_bbox_v4.py`，新模块不动 v3）

| 块 | 维度 | 内容 |
|---|---:|---|
| hand_count | 1 | 每帧手框数 |
| hand × 2 slot | 2×8=16 | presence/轴系坐标÷轴长/面积比对数/速度/缺失龄（同 v3 hand 块口径） |
| ctrl | 5 | presence/面积比/速度/缺失龄 + 轴长（轴定义端点，坐标恒 (0,0) 不编码） |
| mid | 6 | presence/轴系坐标(=轴长端点)/面积比/速度/缺失龄 |
| 事件类 × 3 | 3×4=12 | presence/along/÷轴长/perp÷轴长/面积比对数（无速度——事件帧太少） |
| pair × 3 | 3×3=9 | hand-ctrl、hand-mid、ctrl-mid 距离（÷轴长）；**砍掉全部以 distal/short_brush/事件类为基准端的 pair** |
| 时间 | 3 | 同 v3 |
| **合计** | **62** | 对比 v3 113（砍 51 维：2 类几何块 18 + 相关 pair 12 + 参考基准重编码） |

> 62 维为设计目标值，实现时以单测锁定（差 1~2 维可接受，写入契约名维度字段）。

### 2.4 配套契约（同轮升版）

| 契约 | 维度 | 说明 |
|---|---:|---|
| v4 主特征（上表） | 62 | 定版候选主契约 |
| v4 ⊕ roi6 | 62+108=170 | ROI 网格同步砍 2 类：144→108（6 类 × 6 区 × 3 通道） |
| v2' ⊕ v4（可选） | — | v2 剔除废弃通道后的对照（若需要复验 v2⊕v3→v2'⊕v4 的互补性） |

## 3. 命名对齐（名称 ↔ 实现 ↔ 维度一一对应）

**规范**：`ama-v{N}-{编码}-{核心类数}c-{dim}d`（ama=ActionMixed-auto 数据域）。
维度入名，一名一实现；所有契约在 `docs/features/README.md` 登记表维护
名称/维度/模块/测试/登记处五列。

| 旧名（混用现状） | 新名 | 维度 |
|---|---|---:|
| actionmixed-bbox-8cls-v1 | ama-v1-bbox-8c-40d | 40 |
| actionmixed-roi-grid-v1 → v4 版 | ama-v4-roi-6c-108d | 108 |
| actionmixed-bbox-hand-8cls-v1（手部域，已淘汰线） | 退役不再重命名 | 40 |
| clean_bbox_v2_top1_impute（113/121/249 一名多维度） | ama-v2-clean-8c-113d（121/249 变体随探针淘汰退役） | 113 |
| clean_bbox_v3_scope_frame | ama-v3-scope-8c-113d | 113 |
| actionmixed-cleanv2v3-concat-v1 | ama-v3-concat23-8c-226d | 226 |
| actionmixed-cleanv3-roi-concat-v1（已证伪） | 退役 | 257 |
| cnn 系（S2/S3，已证伪） | 退役 | — |
| **v4 新契约** | **ama-v4-scope-6c-62d / ama-v4-scoperoi-6c-170d** | 62/170 |

> 旧名在历史 run/文档中不回改（历史可追溯），新代码只认新名；
> `feature_names_for_version` 同时注册新旧映射一个过渡期。

## 4. 评测指标体系（六维能力矩阵）

| 能力维度 | 指标 | 来源 | 动作 |
|---|---|---|---|
| 段识别与排序 | edit score（macro/视频） | 已有 | 保留为主线 |
| 边界定位精度 | F1@IoU(0.1/0.25/0.5) + **start/end MAE 入 summary** | 已有（MAE 在 details） | MAE 升一级 |
| 逐类判别 | per-class recall + **idle-excluded macro-F1** | 需新增聚合 | 新增（当前 macro 被 68% idle 失真） |
| 关键混淆对 | **insert↔withdraw 互混率**（占真值帧比例） | details 可算 | 新增为 summary 指标 |
| 稀有事件敏感 | water_injection / syringe 出现→命中曲线 | 需定义 | 二期（数据补齐后） |
| 部署就绪 | 流式一致性（temporal_feed_mode）+ per-tick 延迟 | 框架已有未跑 | 定版前必测 |

**评测口径升级**：test split（project-18，insert/withdraw 专项）已入数据集——
v4 定版评测 = val（同分布）+ test（专项）双报告；补 test 的 manifest/catolog 登记。

## 5. 实施清单（按序，含门禁）

1. `features/clean_bbox_v4.py` + 单测（块结构锁定 62 维、轴回退链、事件类"被定位不作判据"约束的断言测试）
2. ROI v4（roi_bbox 增 6 类版）+ 命名注册（新旧映射）+ `features/README.md` 登记表
3. testsets 登记：`temporal.actionmixed-auto-v4`（train/val）+ **test split 登记**（manifest test.txt + revision 重算 + validate 全绿）
4. 指标扩展：idle-excluded macro / insert-withdraw 互混率入 summary（benchmark/core 改动 + 单测）
5. 门禁全绿后跑矩阵：v4-62 vs v4⊕roi6-170 vs v3-113 vs v2⊕v3-226（当前最优）× 3 seed
6. 判据：v4 需同时 ≥ v2⊕v3（33.85/34.62/27.03）才定版替换；若 v4 < v2⊕v3，
   结论为"废弃通道剔除有损"（届时 v2⊕v3 中的 distal/short_brush 通道反而值得消融复查）
7. 汇总回写 FEATURE_LAB §8 + 本文档结果节

## 6. 风险与开放问题

- **62 维是否信息不足**：砍掉 51 维后 GRU 输入骤减，若 v4 明显劣化，需消融定位
  （是废弃通道其实在供信息，还是块结构砍太狠）
- v2' 对照（剔除废弃通道的 v2）暂列可选，视第 5 步结果决定是否跑
- 第 4 点修改意见（用户待补）
- 流式一致性评测依赖 streaming 推理路径，若 CleanSightBackend 侧有变化需先对齐
