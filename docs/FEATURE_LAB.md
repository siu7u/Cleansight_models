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

## 6. 四特征集 3-seed 矩阵结果（2026-09-12，CPU/WSL 口径）

> 按 `docs/FEATURE_SCHEME_EVAL_PLAN.md` §4 执行：GRU（hidden=128, 3 层），健康配方
>（wd=1e-4 / dropout=0.2 / patience=4 / best_metric=val_f1_0.5），数据 v3
>（train 14 / val 4，eval split=val），seed 42/7/2026，12 run 全部完成并评估。
> 运行目录 `runs/strategy_matrix/`；**单机 CPU/WSL exploratory 口径，未设固定 testset。**

| 特征集 | median edit | median F1@0.1 | median F1@0.25 | median F1@0.5 | median frame mIoU |
|---|---:|---:|---:|---:|---:|
| **roi-144**（B0b 参照） | **24.83** | **24.24** | **18.18** | 2.02 | 11.35 |
| bbox-40（B0a 基线） | 23.35 | 23.66 | 17.20 | **4.40** | **20.93** |
| global-hand-80（S1） | 18.75 | 21.74 | 13.04 | 2.20 | 17.30 |
| hand-40（S1 退化组） | 14.56 | 13.79 | 9.20 | 2.30 | 17.62 |

结论（3-seed 中位数，val 仅 4 视频方差不小，谨慎解读）：

- **ROI 网格段级指标仍最优**（edit / F1@0.1 / F1@0.25 均第一），复现了分支先行实验的排序；
- **纯手部特征明确劣于全局 bbox**，且 global+hand 拼接（80 维）仍不敌基线——与数据侧事实一致
  （scope 类在手部区域 presence 仅 14~16%，手部 box 通道稀释 scope 信号）；
- 依方案判据：S1（box 锚定手部）**未通过**「同时优于 B0a 与 B0b」，ROI 网格暂胜；
- 待跑：S2（bbox ⊕ 全帧 CNN embedding）与 S3（手部+全局视觉），手部贡献的最终判断待
  CNN 全局特征加入后重新评估。

### 6.1 S2/S3 结果（2026-09-12 补充，CPU/WSL 口径）

> 实施：18 个源视频（`outputs/videos-p16/`）按标签帧号抽帧（12959 帧，0 缺失）→
> resnet18 逐帧 embedding → train 拟合 PCA-80（EVR=15.6%，域外特征方差分散）→
> 与几何特征拼接。契约 `actionmixed-bbox-cnn-resnet18-v1`（120 维）/
> `actionmixed-bbox-hand-cnn-v1`（160 维）；实施细节见
> [`FEATURE_SCHEME_S2_S3_SPEC.md`](FEATURE_SCHEME_S2_S3_SPEC.md)。
> **注意**：首轮因帧图路径/扩展名与 `extract_embeddings.py` 约定不符导致 embedding 全零，
> 已作废并在链路中加入自动门禁（逐视频非零行占比 ≥99%）后重跑。

| 特征集 | median edit | median F1@0.1 | median F1@0.25 | median frame mIoU |
|---|---:|---:|---:|---:|
| roi-144（B0b 参照） | 24.83 | **24.24** | **18.18** | 11.35 |
| bbox-40（B0a 基线） | 23.35 | 23.66 | 17.20 | **20.93** |
| hand+bbox+cnn-160（S3） | **25.22** | 22.92 | 12.50 | 14.63 |
| global-hand-80（S1） | 18.75 | 21.74 | 13.04 | 17.30 |
| bbox+cnn-120（S2） | 16.01 | 18.37 | 10.42 | 15.28 |
| hand-40 | 14.56 | 13.79 | 9.20 | 17.62 |

S2/S3 判据结论：

- **S2 未通过**：edit 16.01 / F1@0.25 10.42 全面低于基线与 ROI——ImageNet resnet18 的
  全帧外观特征在内镜域未带来增益（风险 ② 域差距应验），反而稀释几何信号；
- **S3 未通过**：edit 25.22 单项最高（超 roi +0.39，在 seed 方差内），但 F1@0.25 12.50
  远低于 roi 18.18，「同时优于 B0a 与 B0b」不成立；edit 上的微弱优势不足以支持 160 维
  与 CNN 部署成本；
- **手部通道最终判断**：S3(25.22) > S2(16.01) 说明手部几何通道在 CNN 特征加持下有正贡献，
  但仍未越过 ROI 网格——box 锚定手部线关闭；若未来重启手部线，应改用关键点级特征
  （MediaPipe 等）而非 hand box 区域统计；
- **定版建议**：特征选型定格 **roi-144（actionmixed-roi-grid-v1）**——因果、无状态、
  纯几何零部署成本，段级指标全面第一；升正式前需 GPU 多 seed 复跑（本轮为 CPU
  exploratory 口径，val 仅 4 视频且 PCA/评估均在低算力环境）。
