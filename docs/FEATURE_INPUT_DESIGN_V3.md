# 时序模型输入特征设计提案与 v3 验证计划

> 背景：现行 `clean_bbox_v2` 族（113/121/249 维）特征提取方案评测结果不理想。
> 本文记录 2026-09 输入设计评审结论，并落第一步（v3）的验证计划。
> 结论落地后同步 `docs/features/README.md` 索引与 `FEATURE_STRATEGY_COMPARE.md`。

## 1. 设计提案（按预期收益/改动成本排序）

### P1 · scope 相对坐标系 + 尺度归一 + 分块归一（第一步，已实施）

**问题**：内镜画面存在推拉镜头/平移，绝对 `cx, cy, area` 非平稳——同一动作在不同
zoom 下特征值不同，checkpoint 级全局 z-score 无法修正。

**方案**：
- 以 `scope_control_body → scope_distal_end`（缺席时回退 `scope_mid_section`）定义器械轴，
  所有目标位置投影为沿轴分量 `along` / 垂轴分量 `across`，除以轴长 `ref_len`；
- 面积改为相对 `scope_distal_end` 面积的对数比 `log((area+eps)/(ref_area+eps))`；
- pair 距离不再用硬编码 `sqrt(2)` 上限，而是相对尺度（距离/轴长天然有界，仍 clip）；
- 归一化：所有连续列有界（clip），flag 类（present/imputed/missing_age）保持原值，
  不再依赖全局 z-score；模型入口 LayerNorm 承担残余尺度差异。

### P2 · bbox 非线性几何特征（第二步，待做）

- `log` 面积（P1 已含）+ 保留 `w, h` 分列的长宽比 `w/h`（v2 只存 `w*h`，比例信息丢失）；
- pair 增加 IoU 与 containment：`brush_tip_out` 插入 `scope_distal_end` 框的带符号深度
  （沿镜轴、按轴长归一），直接对应"刷尖在镜内/镜外"的业务语义；
- 相对运动：pair 相对速度向量（模 + 方向 sin/cos）；单目标加速度、速度方向；
- 速度按尺度归一：`speed / sqrt(area)`；
- 补 `presence_age`（连续出现时长），与 `missing_age` 对称；
- 修复：imputed 帧的 `delta` 由插值轨迹算出假速度，应置 0 或乘 `(1-imputed)`。

### P3 · business_priors 从输入改为辅助监督（第三步，待做）

8 个 prior 是人工拼出的"疑似标签"，作输入等于固化先验、易捷径学习且对检测噪声脆弱。
改为 aux head（低权重 BCE 多任务），主干自由组合底层特征。
同时消融 `center_window`（w5/w15 均值，136 维）——dilated conv 本身做多尺度时间聚合，
大概率冗余。

### P4 · 目标 token 化（set encoder，第四步，待做）

每目标一个 token（类型 embedding + MLP(8 维 track) + hand 槽位 embedding），
pair 单独 relation token，一层 attention 聚合。置换不变、hand 多实例不依赖
top1/top2 启发式槽位选择、加新类别不改输入维度。与 `[B,T,F]` 接口兼容的渐进路径：
先在特征管线侧做（每目标定长块 + 块内语义不变），再改模型侧。

### 配套评估方法

- 每步独立 feature version + 独立 benchmark 报告，逐块消融 + permutation importance；
- 检测端瓶颈（g2 上 brush_tip_out/short_brush mAP≈0）不在特征端解决范围，若确认瓶颈
  在检测召回，优先回到检测器/ROI 融合。

## 2. 第一步验证计划（P1 → `clean_bbox_v3_scope_frame`）

| 项 | 值 |
|---|---|
| feature version | `clean_bbox_v3_scope_frame` |
| 维度 | 113（块结构与 v2 base 一一对应，便于逐块对照） |
| 代码 | `features/clean_bbox_v3.py`（复用 v2 槽位选择/短缺失插值） |
| 分发 | `temporal/data.py` load_split clean 分支 |
| 训练配置 | `experiments/mstcn-clean-v3.yaml`（与 v2 对照同模型同超参） |
| 评价口径 | 与 v2 基线同 split 同 seed 对比帧级 acc / edit / F1@0.1 |

### 布局（113 维）

- `hand_count` (1)
- hand slot1/2 × 8：`[present, conf, along, across, log_area_ratio, speed, missing_age, imputed]`
- 8 个非 hand 目标 × (candidate_count + 8)（同上 8 通道）
- 7 个 pair × `[valid, dist_norm, delta]`
- `[t_norm, t_sin, t_cos]`

### scope 轴回退链

axis 依次取 `distal-control` → `distal-mid` → `mid-control` → 上一帧有效值 → `(1,0)`；
`ref_len`/`ref_area` 沿有效帧前向填充，序列内无有效值时回退中位数 / 常数
（0.35 / 0.02）。连续分量 clip 到 ±4 / ±6，保证有界。

### 成功标准

1. 单元测试：维度/列名稳定、空帧有限值、scope 缺席回退、load_split 分发；
2. 训练不劣于 v2 基线（同模型同超参，帧级 acc 与段级 edit 至少持平）；
3. 若达标准则进入 P2；若明显更差，检查 scope 检测质量（scope 类 presence 不足时
   回退链退化到常数轴，需先确认 scope 检出率）。

## 3. 已知限制

- 五列 bbox 无置信度时使用 `detection_confidence_default`，只能做 exploratory 评测
  （与 v2 相同，不冒充正式结果）；
- v3 未登记独立 dataset revision（复用 v2 家族的数据，`feature_schema.version` 区分）。
