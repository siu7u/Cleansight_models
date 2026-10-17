# 分割决策简报交叉评估：与 feature-scheme 工作线的对齐（2026-10-17）

> 对象：同事的 SEGMENTATION_DECISION_BRIEF_20261017.md（分支 feat/mstcn-segmentation，未推送）。
> 结论：两线高度交汇，我们已有其层 1 的 60% 实现与层 2 的完整否定结论。

## 一、交汇点

| 他们的需求/计划 | 我们的现状 |
|---|---|
| 层 1：decode_segments 单一实现（禁各写一份折叠） | ✅ **已建成**——framework/cleansight_eval/temporal/postprocess.py（apply_v12，配置驱动，predict() 一行启用，逐位+端到端双验证） |
| 层 1：段 artifact + decode_config | 🟡 部分具备——评测产物已含逐帧标签与 v12 段输出；独立 segments 文件未做（半天） |
| 层 2：软时长 Viterbi / 滞回 / 最小驻留重测 | ✅ **已扫完（§22，45+ 配置）**——Viterbi 全败（-6.7+，一阶转移表达不了段长先验）；滞回是 Pareto 权衡（edit+3.3↔F1-7.3）；真正增益在逐类 min_len（+15.2） |
| 三套段口径不一致问题 | ✅ 架构性解决——我们的 decode 是单实现多配置 |
| 段 F1 / 边界 F1 注册正式指标 | ✅ 段级 F1@k 已注册；边界 F1 未做（可补） |
| "标注膨胀"（test 32.3% 帧、该区 acc 45%） | ❌ **我们未量化**——比"过渡帧模糊"更系统，解释边界 P90=97 帧；与 trans 训练失败互证 |

## 二、对其拍板问题的回答

**下游消费段**。C7（离线最优，test edit 71.8）输出即 [start, end, label]，与 SegmentFact 同构；
v1.2 后处理产物形态一致。层 1 应做，且我们分支已有六成。

## 三、对齐动作

1. postprocess.py 共享为 decode_segments 基准实现（合并或抄设计）；
2. §22 否定结论移交（Viterbi/滞回/门控扫描数据）——省其数周；
3. 联合量化标注膨胀于 v4.1（我方 test_probs npz 现成）；
4. 后端集成用名字断言 + feature_schema 版本号机制（对齐其"6 位错 5 位"教训）。

## 四、对我们 TODO 的影响

- 新增：段 artifact 独立产物 + 边界 F1 指标（层 1 补完，~1 天）；
- 新增：v4.1 标注膨胀量化（联合任务）；
- C7 生产化封装与后端打通可借用其层 4 路线图。