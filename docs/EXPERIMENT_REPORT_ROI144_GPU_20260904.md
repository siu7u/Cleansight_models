# 实验报告：ROI 网格 144 正式配方 · GPU 复跑（3 seed）

- 日期：2026-09-04
- 目的：在 **GPU 环境**上复跑正式训练方案（此前 09-03 正式轮跑在 CPU，自动化会话无 GPU），
  同时验证注册表改指 **`datasets/cleansight-ActionMixed-auto-lhh`** 后链路与结果。
- 一句话结论：**ROI 网格 144 + 健康配方在 GPU / -lhh 数据上三 seed 全部零坍缩**，批间完全
  确定性复现；但逐 seed 数字与 CPU 轮漂移显著（seed 42 提升、seed 7/2026 回落），
  **正式基线数字必须锚定设备口径**（GPU 全策略矩阵待跑后定版）。

## 1. 实验设置

| 项 | 值 |
|---|---|
| 数据 | `datasets/cleansight-ActionMixed-auto-lhh`（自动通道 v3，含 task#204 sbc 修正；train 13 / val 3 / test 2 视频） |
| 测试集 | 锚定 task#195/#199：1,891 帧 = idle 1,140 + long_brush_insert 536 + long_brush_withdraw 215（water/flush/sb_cleaning 零覆盖） |
| 特征 | `actionmixed-roi-grid-v1`：8 检测类 × 2×3 网格 × [presence, count, max_area] = 144 维，class-major |
| 模型 | GRU hidden=128 × 3 层，6 类，因果滑窗 |
| 配方 | wd=1e-4、dropout=0.2、patience=4（val_loss 早停）、epochs≤20、best=val_f1_0.5、类别权重截断 [0.1, 5.0] |
| 设备 | NVIDIA RTX 4060 Laptop GPU · CUDA 12.8 · cuDNN 91002 · torch 2.8.0+cu128 · fp32 |
| 命令 | `python tools/run_strategy_matrix.py --runs-dir runs/formal_roi_20260905 --strategies roi-grid-144 --seeds 42,7,2026` |
| 产物 | `runs/formal_roi_20260905/`（每 seed 一个 run 目录，含 checkpoints/evals/history/env.json） |

> 执行了两批完全相同命令（12:38 与 12:56，注册表 12:52 改指 -lhh，故第一批读官方目录、
> 第二批读 -lhh）；两批**逐 seed 指标全等** → ① GPU 环境内确定性复现；② -lhh 与官方目录
> 内容在运行级等价（此前已字节级 diff 一致）。

## 2. 结果（GPU 轮，逐 seed）

| seed | 训练 epochs | best val_f1_0.5 (ep) | acc | edit | F1@0.1 | F1@0.25 | F1@0.5 | 非 idle 预测帧 | 帧级 F1：idle / insert / withdraw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 8 | 10.14 (ep6) | 50.77 | **55.63** | **44.90** | **36.73** | 8.16 | 477 | 64.4 / **27.2** / — |
| 7 | 8 | 12.70 (ep1) | 34.37 | 21.21 | 19.05 | 14.29 | 9.52 | 769 | 53.0 / — / 11.8 |
| 2026 | 6 | 8.62 (ep2) | 54.10 | 23.59 | 19.51 | 14.63 | 0.00 | 152 | 70.8 / 0.0 / 2.5 |
| **中位数** | — | — | 50.77 | **23.59** | **19.51** | **14.63** | 8.16 | — | — |

（帧级 F1 为 test 上 GT 非零类；"—"= 该类 0 TP。段级匹配：F1@0.5 时 TP=2/2/0，GT 段共 32。）

## 3. 与 CPU 轮（09-03）对比

| seed | CPU edit / F1@0.1 / F1@0.25 | GPU edit / F1@0.1 / F1@0.25 | 方向 |
|---|---:|---:|---|
| 42 | 53.25 / 33.33 / 20.83 | 55.63 / 44.90 / 36.73 | ↑ |
| 7 | 23.59 / 28.57 / 23.81 | 21.21 / 19.05 / 14.29 | ↓ |
| 2026 | 28.35 / 31.82 / 22.73 | 23.59 / 19.51 / 14.63 | ↓ |
| **中位数** | 28.35 / 31.8 / 22.7 | 23.59 / 19.51 / 14.63 | 中位下滑（seed 7/2026 拖累） |

- 数据、配方、命令完全一致（数据根内容逐字节相同）；差异来源 = **设备**：训练轨迹不同
  （seed 42 CPU 早停 ep6 → GPU ep8），同环境内则完全确定（批间全等）。
- 设备差异（F1@0.1 最大 ~13 点）大于单轮 seed 噪声的度量误差——**跨设备比较只能定性**。

## 4. 诊断与观察

1. **坍缩状态**：6/6 run 非 idle 预测帧 152–769 > 0，ROI 144 的抗坍缩稳健性**跨设备成立**
   （对照：bbox 系其它策略 09-03 CPU 轮 2/3 seed 坍缩到 0~8 帧）。
2. **每 seed"学会的动作"不同**（帧级 F1）：seed 42 会 insert（27.2）不会 withdraw；
   seed 7 会一点 withdraw（11.8）不会 insert；seed 2026 两样都不会（insert 0.0）——
   这就是"同策略换 seed 段级指标差 3~5 倍"的机制层解释：稀有类被部分 seed 学会、部分遗忘。
3. **acc 误导的活例**：seed 2026 的 acc 最高（54.10，test 60% 是 idle），却 F1@0.5=0、
   非 idle 仅 152 帧——"永远多猜 idle"得分最高；段级指标才是主线（与文档结论一致）。
4. **段边界是当前短板**：F1@0.1/0.25 尚可、F1@0.5 几乎全 miss（TP 2/32）——预测段与 GT
   段重叠但边界/起止对不齐；这是"段级精确匹配"能力的量化证据，后续调优方向（如
   更细粒度帧分类一致性、后处理修边界）可对照此基线。
5. **best 多停在早期 epoch**（ep1/2/6，训练 6–8 ep 被早停）——概念学习期短，与 09-03
   观察一致；数据扩量/更强正则仍是延长有效训练的主要途径。

## 5. 结论与遗留

- ROI 网格 144 + 健康配方为当前**唯一跨设备（CPU/GPU）三 seed 零坍缩**的特征方案；D6 主线
  决策不受本轮影响。
- 正式基线数字**待 GPU 全策略矩阵（4 策略 × 3 seed，同设备）**后定版并回写
  `FEATURE_STRATEGY_COMPARE.md` / `IMAGE_FEATURE_TRAINING.md`（当前文档中位数为 CPU 轮，
  已加设备注）。
- 本实验两批 6 run 全部 `succeeded`，可作为 GPU 口径的 roi-grid-144 复现证据。

## 6. 复现与产物索引

```bash
# 复现（GPU 交互终端，backend venv）：
python tools/run_strategy_matrix.py --runs-dir runs/formal_roi_20260905 \
    --strategies roi-grid-144 --seeds 42,7,2026
# 门禁预检：python tools/validate_testsets.py --catalog framework/testsets.yaml --json
```

| 物 | 路径 |
|---|---|
| run 目录 ×6 | `runs/formal_roi_20260905/gru-20260904-{123815,123837,123900,125601,125623,125645}/` |
| 汇总 | `runs/formal_roi_20260905/STRATEGY_SUMMARY.md` |
| best checkpoint | `…/checkpoints/best.pt`（seed 42 建议候选） |
| 溯源 | 每 run `env.json`（seed/device/cuda/cudnn/command/git）+ `config.resolved.json`（data.root/feature_schema） |

---

*数据均来自本机 run 产物（status.json / history.csv / evals/*.evaluation.json），未做任何外推。*
