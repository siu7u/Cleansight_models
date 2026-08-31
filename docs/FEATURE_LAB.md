# 特征提取方案实验（feature-lab）

> 启动：2026-08-31 · 本周目标：**实验确定图像特征提取方案**（长短毛刷刷洗为重点考察动作）
> 实验产物区：`outputs/feature-lab/`（gitignored）；方案正式落点：`framework/cleansight_eval/temporal/features/`

## 1. 实验数据

| 数据源 | 状态 | 用途 |
|---|---|---|
| **action-test**（LS project 18） | 已建项目+模板（复制自 project-16，6 类 timeline），**待采集上传视频** | 小规模试验：长短毛刷刷洗测试数据 |
| `temporal.actionmixed-auto-v3` | ✅ 就绪 | 基线对比/全量验证 |

action-test 流转：采集上传 → LS 标注（仅 timeline，沿用 project-16 规范）→ 导出 → `annotate run`（yolo11 配置）→ `convert` → 实验数据集（独立目录，不进 v3 catalog，除非正式并入）。

## 2. 候选方案

| 方案 | 来源 | 特征 | 状态 |
|---|---|---|---|
| A. clean_bbox_v2 | 现行 | 8 类 × 5 项统计 = 40 维 | 基线 |
| B. 规则组（Rule 机制） | causal-model（队友实验） | 自由裁切：坐标/大小/双物体距离/轨迹方向等 | 待迁移评估 |
| C. 组合/新设计 | feature-lab | 待实验 | — |

## 3. causal-model 机制迁移清单（吸收后冻结为参考实现）

- [ ] **Rule 规则包装器**：`Rule(fn, object_ids=(...), dropout=...)`——规则函数接口 `(detections, id1, id2...) → (feat, ok)`，统一缺省值管理
- [ ] **manifest.json 特征规则记录**：特征定义与训练产物同存（可追溯、可复现），对齐 pin.yaml 的 feature_mapping 纪律
- [ ] **dropout 特征遮罩**：保存未遮罩特征+manifest 记录，训练时按需构建 mask（特征消融实验利器）
- [ ] **TorchScript 保存/加载**（Trainer/Predictor）与 online(step) 流式推理接口——评估流式一致性时参考

迁移落点：`framework/cleansight_eval/temporal/features/`；迁移时保持 actionmixed-bbox 特征语义兼容或显式升 feature_mapping 版本。

## 4. 评测口径（对齐 BENCHMARK_SEGMENTATION.md §7）

- **重点看 insert vs withdraw 混淆矩阵**（毛刷测试数据正是 P0 对比对）
- 边界定位误差（模糊渐变边界段）
- 方案间对比：同数据同模型（GRU 先行）只换特征

## 5. 纪律

- 特征集一旦变化 → feature_mapping 升版本（按 YAML_CONFIG/注册规范），模型需重训
- action-test 数据不与 v3 混用目录；若正式并入数据集走 manifest 三件套流程
