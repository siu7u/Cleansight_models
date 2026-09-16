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

## 6. 特征集矩阵结果（2026-09-15 统一健康配方后定稿，CPU/WSL 口径）

> 按 `docs/FEATURE_SCHEME_EVAL_PLAN.md` §4 执行：GRU（hidden=128, 3 层）、健康配方
>（wd=1e-4 / dropout=0.2 / patience=4 / best_metric=val_f1_0.5），数据 v3
>（train 14 / val 4，eval split=val）。**单机 CPU/WSL exploratory 口径，未设固定 testset。**
> bbox/hand/global-hand 三基线于 2026-09-15/16 按统一配方重跑（3 seed 全部完成）；
> roi/S2/S3 为 2026-09-12 原生健康配方 run（3 seed）。

| 特征集 | n | median edit | median F1@0.1 | median F1@0.25 | median frame mIoU |
|---|---:|---:|---:|---:|---:|
| **roi-144**（B0b 参照） | 3 | **24.83** | **24.24** | **18.18** | 11.35 |
| bbox-40（B0a 基线） | 3 | 18.53 | 23.66 | 15.05 | **20.37** |
| global-hand-80（S1） | 3 | 17.06 | 21.98 | 14.74 | 19.12 |
| hand-40（S1 退化组） | 3 | 15.81 | 17.98 | 13.48 | 17.52 |

> F1@0.5 各组均个位数且 seed 方差大，从略。混杂前的旧数字见 §6.2.2——bbox-40 旧值
> edit 23.35 系 val_acc 选型偏差抬高，统一配方后回落至 18.53。

结论（统一健康配方后，**roi-144 领先幅度扩大**）：

- **ROI 网格段级指标全面第一**，且与基线差距从 ~1.5 扩大到 **+6.3 edit / +1.6 F1@0.25**；
- **bbox-40 基线被混杂配方显著高估**（见 §6.2.2：旧 run 的 best.pt 是 epoch1 的 val_acc 解）；
- **纯手部特征与 global+hand 拼接均劣于全局 bbox**——与数据侧事实一致（scope 类在手部
  区域 presence 仅 14~16%，手部 box 通道稀释 scope 信号）；
- 依方案判据：S1（box 锚定手部）**未通过**「同时优于 B0a 与 B0b」；

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

### 6.2 训练充分性分析与配方混杂审计（2026-09-12）

#### 6.2.1 用 loss 曲线判断「训练是否充分」（18 个 run 全量审计）

| 组 | 实际 epochs | train_loss（首→末） | val_loss 最低点 | 早停 | 判读 |
|---|---|---|---|---|---|
| roi-144 / S2 / S3 | 5~6（早停触发） | 1.2 → 0.70~0.88（未收敛） | **ep1~2**，其后单调恶化 | ✅ | 训练充分：ep2 起即过拟合，加 epoch 无益 |
| bbox-40 / hand-40 / global-hand-80 | 20（跑满） | 1.4 → 0.23~0.53（尾部每 3ep 仍降 3~7%） | ep3~5，其后持续恶化（如 1.09→1.83） | ❌ 未启用 | 同理：val 早在 ep5 见底，瓶颈不在优化步数 |

**结论：本轮不存在「训练不足」问题。** 两组都表现为 train 仍在下降、val 早已回头——
瓶颈是**泛化**，而非训练轮数。可归因于：

1. **val 分布失衡**：idle 占 68.20%，water_injection 仅 17 帧（0.50%），val_loss 被 idle
   主导且稀有类噪声极大（见 [`FEATURE_LAB_CLASS_ACCURACY.md`](FEATURE_LAB_CLASS_ACCURACY.md)）；
2. **模型/特征容量与任务不匹配**：纯几何特征可线性区分的动作模式有限，段级边界定位难题
   （insert/withdraw 互混）不是靠多训几个 epoch 能解决的；
3. **val 仅 4 视频**：单视频的分布偏移即可主导整条 val_loss 曲线，早停点因此抖动。

#### 6.2.2 配方混杂问题（本轮审计的重要发现）

审计早停行为不一致时发现：**只有 roi 的 YAML 携带健康配方**（dropout=0.2 /
weight_decay=1e-4 / patience=4 / best_metric=val_f1_0.5），而 bbox-40、hand-40、
global-hand-80 三个基线 YAML 均缺这四键——它们实际按以下口径训练：

- 无 dropout、无 weight_decay（过拟合无约束）；
- **best.pt 按 `val_acc` 选择**——`status.json` 证实 bbox-40 seed42 选中的是
  **epoch1** 的模型（val_acc=64.89），即偏向 idle 的早期解；
- `patience` 缺失 → 无早停，硬跑满 20 epoch。

这正是 `docs/FEATURE_STRATEGY_COMPARE.md` 已诊断并修复过的「val_acc 选型偏爱 idle 坍缩解」
问题，但三个基线 YAML 未同步修复，导致第一轮矩阵中**基线与 ROI 的对比口径不一致**。

**处置（2026-09-15 完成）**：

1. 三份 YAML 已补齐健康配方（dropout 置于 `model` 段，weight_decay/patience/best_metric
   置于 `train` 段），提交 `2d04bf7`；首次重跑验证修复生效（`gru-20260915-160858`：
   `best_metric=val_f1_0.5`、按 val_loss 早停于 ep7、best.pt 选在 ep3）；
2. **统一配方重跑已全部完成**（3 配置 × 3 seed = 9 run + 评估全绿，2026-09-16 收官）；
   执行中发现并修复两个编排缺陷：
   ① torch 线程超订阅（4 进程 × 16 线程 → 50 倍减速，限 OMP/MKL=4 修复）；
   ② 并行训练同秒启动撞出同名 run 目录导致 checkpoint 污染（改为**训练串行 + 评估并行**，
   污染目录已删除）；
3. 数据集同步迁移 WSL 原生文件系统（symlink），eval 从 ~20 分钟降到 ~10 分钟，
   **指标等价性已验证（14/14 项 summary 完全一致）**；
4. §6 表格已替换为统一配方数字：**bbox-40 基线从 23.35 回落到 18.53，roi-144 领先
   扩大至 +6.3 edit**——混杂不但没有推翻结论，反而强化了 roi-144 的优势；S1 判据
   （未通过）维持不变。

> 遗留：GPU 多 seed 正式复跑仍未做。
