# v4 特征维度明细（复核用，2026-09-19）

> 本文是 `ama-v4-scope-6c-60d` 与 `ama-v4-roi-6c-108d` 的**逐列清单**，
由实现的列名导出（`clean_v4_feature_names()`），与代码一一对应，非手工估算。
> 设计依据见 [`FEATURE_SCHEME_V4_DESIGN.md`](FEATURE_SCHEME_V4_DESIGN.md)。

## 0. 公共定义（先读这个）

- **器械轴（v4 主锚点转正）**：轴向量 a = (mid_center − ctrl_center)/|mid_center − ctrl_center|，
  轴长 L = |mid_center − ctrl_center|；ctrl+mid 同帧可用率 63~69%（v3 的 ctrl→distal 仅 8~12%）。
  轴缺失时回退：前一帧轴方向 → 竖直默认 (1,0)；L 前向填充（无有效值回退序列中位数）。
- **along/across**：目标中心相对 ctrl 中心的位置 r 在轴/垂轴上的分量 ÷ L：
  along = r·a / L，across = r·a⊥ / L（a⊥ = (−a_y, a_x)）。clip 到 [−4, 4]。
- **log_area_ratio**：log((area + 1e-4) / (ref_area + 1e-4))，clip 到 [−6, 6]。
  ref_area 基准 = **mid 面积**（v3 用 distal，随剔除废止）；mid 缺失帧回退 ctrl 面积，
  再前向填充，无有效值回退 0.02。
- **speed**：帧间中心位移 × fps，clip [0,5]/5（v2 口径复用）。
- **missing_age**：连续缺失帧数（÷6 封顶）；**imputed**：短缺失插值标记（v2 插值器复用）。
- **候选数 candidate_count**：该类当帧检测框数，clip [0,3]/3。
- **事件线索类约束**（决策 ②）：syringe/air_gun/brush_tip_out 只被轴定位（along/across 是它们
  **相对 scope 轴**的坐标，scope 轴是判据，不是它们）；它们不作任何锚点/基准/pair 端。

## 1. ama-v4-scope-6c-60d 逐块明细（60 = 1+16+7+9+15+9+3）

### 块 A：hand_count（1 维）

| 列 | 含义 |
|---|---|
| hand_count | 当帧 hand 检测框数，clip[0,3]/3 |

### 块 B：hand 双 slot（2×8 = 16 维）
按 v2 的评分排序取 top2 手框（评分 = conf×√area − 0.15×与上帧中心距离），各 8 通道：

| 通道 | 公式 |
|---|---|
| present | 该 slot 是否有效（含短缺失插值后） |
| conf | 检测置信度（标注无置信度时 = 配置默认值） |
| along / across | 手框中心相对 ctrl 的轴系坐标 ÷ L，clip ±4 |
| log_area_ratio | 手框面积相对 mid 面积对数比，clip ±6 |
| speed | 帧间位移×fps 归一 |
| missing_age / imputed | 缺失时长 / 插值标记 |

列名：hand_top1_{8通道}、hand_top2_{8通道}。缺席帧 conf/along/across/log/speed 清零
（保留 present/missing_age/imputed，与 v2/v3 一致）。

### 块 C：scope_control_body（1+6 = 7 维）
轴的定义端点——along/across **恒为 0 不编码**（编码无信息），其余通道保留：

| 列 | 说明 |
|---|---|
| candidate_count | ctrl 检测框数 |
| present / conf | 同上 |
| log_area_ratio | ctrl 面积相对基准（基准缺失时自身就是回退基准，此列退化为常数段，保留以维持块结构稳定） |
| speed / missing_age / imputed | 同上 |

### 块 D：scope_mid_section（1+8 = 9 维）
轴的另一端点，完整 8 通道（**along ≈ 1.0 本身携带轴长归一信息**，across 恒 0 但保留通道
一致性）：candidate_count + present/conf/along/across/log_area_ratio/speed/missing_age/imputed。

### 块 E：事件线索类 × 3（3×5 = 15 维）
syringe / air_gun / brush_tip_out 各：candidate_count + present + along + across + log_area_ratio。
**无 conf/speed/missing_age/imputed**（事件帧占比 <2.5%，速度与插值通道无统计意义）；
坐标为**被 scope 轴定位**的相对坐标（决策 ② 约束，模块结构保证）。

### 块 F：pair × 3（3×3 = 9 维）
仅核心类 pair（v2 的 7 组中 4 组以废弃类/事件类为基准端，全部砍除）：

| pair | 3 列 |
|---|---|
| hand↔ctrl / hand↔mid / ctrl↔mid | valid（双有效）/ dist（欧氏距离 ÷ √2）/ delta（帧间差 clip ±1） |

hand 取与对方更近的 slot。距离÷√2 归一（与 v2/v3 口径一致，不依赖轴）。

### 块 G：时间（3 维）
t_norm（0→1 线性）/ t_sin / t_cos（周期 1）——序列位置先验，与 v2/v3 相同。

### 与 v3（113 维）的精确对照账目（v3 = 1+16+8×9+7×3+3）

v3 的 8 个非 hand 对象块（各 count+8=9 维）：ctrl / mid / distal / short_brush /
**long_brush** / syringe / air_gun / brush_tip_out。其中 **long_brush 不在 auto 检测表**
（frames/data.yaml 只有 8 类，无 long_brush）——它的 9 维在全部 auto 数据上**恒为零**，
是死块；v2 的 7 组 pair 也**全部**以废弃类或恒零类为端点（hand-short_brush、
hand-long_brush、brush_tip_out-distal、short_brush-ctrl、long_brush-mid、
air_gun-distal、syringe-distal）。

| 变化 | 维度 |
|---|---:|
| v3 基线 | 113 |
| − distal_end 块（废弃：presence 8~12%） | −9 |
| − short_brush 块（废弃：presence 0.4~3%） | −9 |
| − long_brush 块（**auto 表无此类，恒零死块**，纯死重剔除） | −9 |
| − ctrl 块 9→7（along/across 恒 0 不编码） | −2 |
| − 事件类 3 组 9→5（砍 conf/speed/missing_age/imputed，事件帧 <2.5% 无统计意义） | −12 |
| − mid 块（不变：count+8） | 0 |
| − pair 7 组（全涉废弃/恒零类）→ 3 组核心 pair（hand-ctrl/hand-mid/ctrl-mid，**新增设计**，v2 原表没有核心 pair） | −12 |
| 时间/hand 块不变 | 0 |
| **合计** | **113 − 53 = 60** |

> 复核路径：§1 各块宽求和 1+16+7+9+15+9+3 = 60；或用本账目 113−53 = 60；
> 或跑 `clean_v4_feature_names()` 数列名。三条路径一致。

## 2. ama-v4-roi-6c-108d 明细

ROI v1（144 维）原样保留网格与通道，仅类表 8→6：

| 项 | 值 |
|---|---|
| 网格 | 2 行 × 3 列 = 6 区域（画面均分，行优先） |
| 通道/区域 | [presence, count, max_area]（同 v1：区域内该类是否有框 / 框数 / 最大框面积 w×h） |
| 类表 | v4 6 类（8 类 id 重映射：0→0,1→1,2→2,4→3,5→4,7→5；**3(distal)/6(short_brush) 丢弃**） |
| 布局 | class-major：每类 6 区域 × 3 通道 = 18 维，6 类 = **108** |
| 实现 | `features/roi_v4.py`：逐帧重映射临时文件后复用 v1 构建器（n_classes=6） |

## 3. 维度自锁（防漂移）

- 模块内断言：矩阵宽度 ≠ 60/108 直接 AssertionError；
- 契约名内嵌维度（-60d/-108d），名实不符在加载层即暴露；
- 单测（`framework/tests/test_clean_bbox_v4.py`，随实现提交）锁定列名全表与块边界索引。