# 实验报告：方案b · v3 六类动作正式轮（四特征策略 × 3 seed，CPU+GPU 双设备）

> 日期：2026-09-05 · 目的：为「方案b」（六类动作全量分类，**重点保证长短毛刷与 flush
> 动作的分类评测覆盖**）提供可汇报的正式训练/测试事实与复现命令。
> 数据为 v3 自动标注集（`temporal.actionmixed-auto-v3`，18 视频）；李海豪的 action-test
> 长短毛刷/flush 新数据尚未到位，test 的 flush/sb_cleaning 覆盖待其补齐（见 §6）。

## 1. 数据与评测口径

| 项 | 值 |
|---|---|
| 数据集 | `cleansight-ActionMixed-auto-v3`（含 task#204 sbc 修正；revision `b7edb874…`） |
| 划分 | train 13 / val 3 / test 2；test 锚定团队指定 task#195/#199（`1b2c95ff`/`5b181b9b`） |
| 动作类 | 6 类：idle / water_injection / flush / long_brush_insert / long_brush_withdraw / short_brush_cleaning |
| test 类别覆盖 | 仅 idle(1140) / lb_insert(536) / lb_withdraw(215)；**flush、sb_cleaning 帧级 support=0** → 正式逐类指标按 missing 处理，不冒充 0 |
| 特征策略 | 全局 40（`actionmixed-bbox-8cls-v1`）/ 仅手部 40（`actionmixed-bbox-hand-8cls-v1`）/ 全局+手部 80（`actionmixed-bbox-global-hand-8cls-v1`）/ ROI 网格 144（`actionmixed-roi-grid-v1`） |
| 模型/配方 | GRU 滑窗（hidden 128×3、window 16、因果平滑）；健康配方 wd=1e-4 / dropout=0.2 / patience=4 / best_metric=val_f1_0.5 / epochs≤20（实际 5~9 早停） |
| 多 seed | 42 / 7 / 2026，取中位数 |
| 设备口径 | CPU 轮 2026-09-03（自动化会话）；GPU 轮 2026-09-04（RTX 4060 Laptop / cuDNN 91002 / fp32，数据根 `-lhh`）——**正式数字锚定 GPU**，跨设备只作定性 |

## 2. 正式结果（GPU 口径中位数；test 锚定 task#195/#199）

| 策略 | 中位 edit | 中位 F1@0.1 | 中位 F1@0.25 | 坍缩 seed（非idle预测=0） |
|---|---:|---:|---:|---:|
| **ROI 网格 144**（正式基线） | **23.6** | **19.5** | **14.6** | 0/3 |
| 全局 40 | 18.4 | 20.0 | 15.0 | 0/3 |
| 全局+手部 80 | 16.0 | 11.8 | 5.9 | 1/3 |
| 仅手部 40 | 6.9 | 11.8 | 5.9 | **3/3** |

逐 seed 全表见 `runs/strategy_compare_gpu/STRATEGY_SUMMARY.md`（12 run，全部 `succeeded`，
每个 run 带 formal evaluation.json + checkpoint 报告）。CPU 口径（2026-09-03，
`runs/strategy_compare/`）：ROI 网格 144 中位 F1@0.1 31.8 / F1@0.25 22.7、0/3 坍缩，其余
策略 ≤15.8（排序仅在 CPU 成立）。两次 roi-grid GPU 复跑（`runs/formal_roi_20260905/` 与
矩阵内）中位数完全一致：edit/F1@0.1/F1@0.25 = 23.59/19.51/14.63。

## 3. 逐类与覆盖事实（长短毛刷/flush 视角）

- **test 可测类（长刷）**：lb_insert/lb_withdraw 帧级 recall 逐 seed 波动大——lb_insert 最高
  26%（GPU roi-grid seed 42），lb_withdraw 最高 40%（CPU roi-grid seed 7；GPU 最高 24%），
  多数 seed 为 0——长刷类仍是当前最难判别的动作，与数据分布（test 全新视频、idle 占 60%）
  一致。
- **test 不可测类（flush/sb_cleaning）**：v3 test 零覆盖；val 中间口径（2026-09-05 补测，
  12 个 GPU best.pt，仅可见性参考、非独立测试，运行目录 `tmp/planb_val_evals/`）：
  flush 帧级 recall 跨策略/seed 0~86%（global-40/global+hand-80 部分 seed ≥70%），
  sb_cleaning 0~52%，lb_insert 几乎全 0、lb_withdraw 个别 seed 53%——模型见过这些类但
  不稳定，**可靠逐类结论必须等 action-test 新数据补 test 覆盖后重测**。

## 4. 策略结论（含"手部 ROI 融合"实测）

1. **ROI 网格 144 是唯一 CPU/GPU 均三 seed 零坍缩的策略**（抗坍缩跨设备成立），GPU edit
   中位领先 → **方案b 正式基线维持 roi-grid-144 + 健康配方**。
2. **手部 ROI 融合无增益**：仅手部 40 在 GPU 三 seed 全坍缩（非 idle 预测 0 帧）；全局+手部
   80 仍有 1/3 坍缩 seed、中位 F1@0.1（11.8）与仅手部持平。原因：scope 类在手部区域内
   presence 仅 ~15%（v3 val 实测），动作判别上下文在区域外。若 action-test 新数据中动作
   集中于手部附近，可再验一次手部通道，但按现有证据不作为主线。
3. seed 方差大（同策略跨 seed 指标差 3~5 倍）、设备差异大于单轮噪声 → 单 seed/跨设备结论
   不可靠，正式数字必须锚定设备口径、多 seed 取中位数。

## 5. 复现命令（仓库根目录，CleanSightBackend venv）

```bash
# 一键矩阵（四策略 × seed 42/7/2026：训练 + formal 评测 + STRATEGY_SUMMARY.md）
python tools/run_strategy_matrix.py --runs-dir runs/strategy_compare_gpu --seeds 42,7,2026
# 单策略单 seed 训练
python -m framework.cleansight_eval.cli.train --config framework/experiments/gru-actionmixed-auto-roi.yaml --seed 42
# 单 checkpoint formal 评测
python -m benchmark.cli.eval --config framework/experiments/gru-actionmixed-auto-roi.yaml \
    --ckpt runs/<run>/checkpoints/best.pt
# 数据/目录校验
python tools/validate_testsets.py --catalog framework/testsets.yaml --json
```

产物：`runs/strategy_compare_gpu/<run>/`（checkpoints/best.pt+meta、evals/*.evaluation.json、
checkpoints/best.eval.md、EVALUATION_REPORT.md）、`STRATEGY_SUMMARY.md`；逐类/逐 seed 口径
文档：`docs/FEATURE_STRATEGY_COMPARE.md`（第三/四轮）、`docs/features/IMAGE_FEATURE_TRAINING.md` §3.3。

## 6. 遗留与下一步（外部依赖）

- **action-test 长短毛刷/flush 数据（李海豪）**：到位后 `annotate run` → `convert` → 重划
  `benchmark/manifests/actionmixed-auto/` → 升 catalog（testsets.yaml revision + 注释）→
  `validate_testsets.py` → 重跑 §5 矩阵，补 test 的 flush/sb_cleaning 逐类覆盖。
- **v3b 重划分（2026-09-05 远端 de07b1b，未定稿）**：train/val 互换 071eb2d6↔e8ea5bb7 +
  4 个 train 视频标注修正；test 不变（指标可比）。远端 frames 存在重复陈旧文件
  （`frames/train/e8ea5bb7-*` 436 个、`frames/val/071eb2d6-*` 261 个）、README/task_ids 未
  同步 → 待数据侧清理确认后采纳（干跑产物 `tmp/planb_v3b_upgrade/`：manifest + 新 revision
  `a52950e5…`）。
