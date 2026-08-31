# 特征提取范围三策略横向对比（bbox 编码固定）

> 实验目的：bbox 特征编码固定为每类最大框 `[presence, cx, cy, w, h]`，只改变**提取范围**，
> 横向对比三种策略对动作识别的表现。全部为 GRU（hidden=128, 3 层）、20 epoch、seed 42、
> 同一 v3 数据三 split（train 13 / val 3 / test 2），仅特征契约不同。

## 策略与特征契约

| 策略 | feature_mapping | 维度 | 说明 |
|---|---|---|---|
| A 整个画面 | `actionmixed-bbox-8cls-v1` | 40 | 基线，全局坐标 |
| B 仅手部周围 | `actionmixed-bbox-hand-8cls-v1` | 40 | 只编码面积最大 hand 框扩张 1.5 倍区域内的框，坐标相对区域归一化；无 hand 全零 |
| C 全局+手部 | `actionmixed-bbox-global-hand-8cls-v1` | 80 | A 与 B 拼接 |

实现：`framework/cleansight_eval/temporal/features/hand_bbox.py`；登记：
`temporal.actionmixed-auto-hand-v1` / `temporal.actionmixed-auto-global-hand-v1`
（revision 与 v3 相同）；配置：`framework/experiments/gru-actionmixed-auto{,-hand,-global-hand}.yaml`。

## 数据侧事实（v3 val，1926 帧）

- 无 hand 帧占比 4.9%（手部特征全零）；hand 类 presence 95.1%。
- 关键差异：`scope_control_body`/`scope_mid_section` 全局 presence 84.7%/72.2%，
  但**手部区域内只有 14.2%/15.9%**——手部策略会丢掉大部分 scope 类信号。
- 稀有类（syringe/air_gun/brush_tip_out）手部区域内 presence ≤ 1%。

## 结果

### 正式 test（锚定 task#195/#199，2 视频；test 仅含 idle/long_brush_insert/long_brush_withdraw）

| 策略 | dim | acc | edit | F1@0.1 | F1@0.25 | F1@0.5 | forward p95 (CPU) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C 全局+手部 | 80 | **59.76** | 16.02 | 11.11 | 5.56 | **5.56** | 2.02 ms |
| A 整个画面 | 40 | 50.34 | **25.97** | **23.81** | **14.29** | 0.00 | 2.18 ms |
| B 仅手部 | 40 | 48.02 | 11.47 | 11.11 | 5.56 | 0.00 | 2.60 ms |

运行目录（未入库）：`tmp/compare_strategies/gru-20260831-{192341,192623,192907}/`，
evaluation.json 含完整溯源（数据集/特征契约/指纹）。

### 训练期最佳 val（history.csv）

| 策略 | best val acc | best val edit | best val F1@0.5 |
|---|---:|---:|---:|
| C 全局+手部 | 56.35 (epoch 15) | 15.67 | 9.09 |
| B 仅手部 | 34.24 (epoch 10) | 11.23 | 4.79 |
| A 整个画面 | 22.49 (epoch 2) | 24.78 | 5.44 |

## 初步结论（20 epoch 单 seed，方向性信号，非最终结论）

1. **"全局+手部"在帧级 acc 上明显领先**（val 56.4 vs 22.5；test 59.8 vs 50.3），说明手部相对信息
   对帧级判别有正贡献。
2. **段级指标（edit/F1@0.1）仍是基线最好**——手部通道把 scope 类信号大幅压缩后，段边界质量下降；
   "两个都提取"恢复全局信息但仍不如基线段级表现。
3. **仅手部策略整体最弱**（test acc/edit 双低）——与数据侧事实一致：动作判别依赖 scope 类在
   画面其他区域的上下文，单纯手部区域信息不足。
4. 推理延迟三者无实质差异（p95 ≈ 2-2.6 ms，均远低于 33 ms 帧预算）。

## 下一步建议

- 多 seed（如 42/7/2026）复跑确认方向；调大 epoch 或用早停看收敛。
- 若"全局+手部"稳定占优，可考虑把手部区域扩张倍数、锚点规则（多 hand 取最大 vs 求和）纳入
  新版本迭代。
- 段级指标下滑需单独排查：可叠加目标遮罩/类别权重，或换 mstcn 全序列模型看段级表现。
