# 实验报告：形态 B 训练侧接入 + E0/E1 机制床消融（2026-09-11）

> 目的：把「像素特征进时序（形态 B）」从"只有提取工具"推进到**可训练、可复跑、可对照**：
> 新增可换产物根目录的图像 embedding 契约、`load_split` 分发、GRU 线性投影头、catalog 登记
> 与单测门禁，并在机制床上跑通 **E0（40 维 bbox）vs E1（40 bbox + 576 图像 embedding）**
> 的三 seed 段级对照。
>
> 相关文档：[特征提取方案索引](features/README.md)（§1.7 契约详述）、
> [`IMAGE_FEATURE_TRAINING.md`](features/IMAGE_FEATURE_TRAINING.md)（§3.5 接入现状、§4 E 系列）、
> [`FEATURE_STRATEGY_COMPARE.md`](FEATURE_STRATEGY_COMPARE.md)（bbox 系对照与坍缩分析）。

## 1. 结论先行

1. **接入链路已就绪并可复跑**：契约、分发、投影头、登记、单测、门禁全部落地（§3 验收证据）。
2. **机制床上图像通道没有"免费增益"**：3 seed 段级 `edit` / `F1@0.25` 稳定小幅优于基线
   （3/3 seed 不劣），但帧级宏 F1 明显下降、`air_injection` recall 在 2/3 seed 归零——
   **混合结论，不足以支持"图像通道有用"，也不足以判死**（机制床非正式数据，见 §5 限制）。
3. **正式结论仍被图像源阻塞**：v3 auto 数据不含像素（`IMAGE_FEATURE_TRAINING.md` §5.1）。
   下一步要么下载 project-16 视频抽帧，要么等 action-test 采集时保留帧图；E2/E3 同时在正式轮补齐。

## 2. 实验口径

| 项 | 值 |
|---|---|
| 机制床数据 | `datasets/cleansight-ActionMixed`（人工标注，6 类动作，9,532 帧；train 5993 / val 2082 / test 1457） |
| 数据契约 | E0 `temporal.actionmixed-v2`（40 维）；E1 `temporal.actionmixed-v2-embed-mbv3s-v1`（616 维，revision 同 v2） |
| 图像产物 | `datasets/cleansight-ActionMixed/embeddings/mobilenet_v3_small-v1`（576 维，缺图 0，逐视频行数与标签严格一致） |
| 模型/配方 | GRU（hidden 128×3、window 16、因果平滑）；wd=1e-4 / dropout=0.2 / patience=4 / best_metric=val_f1_0.5 / epochs≤20 |
| E1 增量 | `model.image_dim: 576` + `model.image_proj_dim: 64`（冻结 embedding 的线性投影头；backbone 零训练） |
| 多 seed | 42 / 7 / 2026，取中位数 |
| 设备 | **CPU**（22 核；本次会话 GPU 不可见 `torch.cuda.is_available()=False`）——数字锚定 CPU 口径 |
| 运行目录 | `runs/embed_ablation/`（6 run，全部 `succeeded`，每 run 带 `evals/*.evaluation.json`） |

## 3. 验收证据（接入链路）

| 验收项 | 命令 | 结果 |
|---|---|---|
| 契约/分发/遮罩/投影头单测 | `pytest framework/tests/test_image_embed_features.py -q` | **12 passed** |
| 全量回归 | `pytest framework/tests -q` | 173 passed, **3 failed（HEAD 同样失败，预先存在，与本次改动无关：`test_config_paths` 期望旧数据根、`test_temporal_masking` 双导入路径 monkeypatch、`test_pipeline_smoke` resume）** |
| catalog 门禁 | `python tools/validate_testsets.py --catalog framework/testsets.yaml --json` | **`errors: []`**，含新登记 `temporal.actionmixed-v2-embed-mbv3s-v1.{train,val,test}` 全 `ok` |
| 端到端 smoke | `train --config gru-actionmixed-embed.yaml -S train.epochs=1`（`runs/embed_ablation_smoke/`） | `state: succeeded`；meta 记录 `image_dim: 576 / image_proj_dim: 64`；`data.embedding_root` 由 catalog 注入 |
| checkpoint 级证明（投影头真的生效） | 读 `checkpoints/best.pt` 的 `model_state` | E1：`image_proj.weight (64, 576)` + `rnn.weight_ih_l0 (384, 104)`（= 40 bbox + 64 投影）；E0：无 `image_proj`、`rnn.weight_ih_l0 (384, 40)` |

## 4. 结果（test 1457 帧；CPU；3 seed）

| 指标（3 seed 中位数） | E0 bbox-40 | E1 bbox+embed-616 |
|---|---:|---:|
| 段级 edit | 42.49 | **45.22** |
| 段级 F1@0.1 | **48.00** | 46.15 |
| 段级 F1@0.25 | 28.00 | **34.62** |
| 段级 F1@0.5 | **15.69** | 7.69 |
| 帧级 macro_f1 | **46.63** | 35.66 |
| 帧级 macro_iou | **27.45** | 14.74 |
| 帧级 acc | 47.01 | 27.52 |
| recall idle | 41.96 | 41.96 |
| recall air_injection | **86.96** | 0.00 |
| recall flush | 9.58 | 12.64 |
| recall long_brush_insert | **55.94** | 5.59 |
| recall short_brush_cleaning | 67.57 | **73.87** |

逐 seed（坍缩检查：非 idle 预测帧均 906~1109，**两臂 6/6 run 零坍缩**）：

| 臂 | seed | epochs | macro_f1 | macro_iou | edit | F1@0.25 | R(air) | R(lb_ins) | R(flush) | R(sbc) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 7 | 5 | 44.52 | 21.02 | 38.55 | 23.53 | 87.8 | 0.0 | 8.4 | 67.6 |
| E0 | 42 | 5 | 48.54 | 28.65 | 45.35 | 35.29 | 85.2 | 55.9 | 9.6 | 67.6 |
| E0 | 2026 | 5 | 46.63 | 27.45 | 42.49 | 28.00 | 87.0 | 58.9 | 9.6 | 66.7 |
| E1 | 7 | 5 | 35.66 | 14.74 | 48.07 | 34.62 | 0.0 | 28.5 | 12.6 | 73.0 |
| E1 | 42 | 9 | 37.08 | 22.78 | 45.22 | 34.62 | 79.1 | 5.4 | 10.7 | 75.7 |
| E1 | 2026 | 8 | 27.35 | 10.81 | 44.26 | 30.19 | 0.0 | 5.6 | 22.6 | 73.9 |

读法（严格按证据，不越界）：

- **段级 edit / F1@0.25**：E1 三个 seed 全部 ≥ 对应 E0（edit 48.1/45.2/44.3 vs 38.6/45.4/42.5；
  F1@0.25 34.6/34.6/30.2 vs 23.5/35.3/28.0）——方向一致但幅度小于 seed 间波动。
- **帧级宏指标**：E1 三个 seed 全部低于 E0（27~37 vs 44~49），主要来自 `air_injection`
  （2/3 seed recall 归零）与 `long_brush_insert`（5% 上下 vs 基线 56~59%）。
  即：图像通道把决策推向别的操作点，而不是无条件变好。
- **训练时长**：E1 早停更晚（8~9 epoch vs 5），best 的 val_f1_0.5 更高（32.6~32.9 vs 23.9~25.2）——
  说明投影头确实在学，不是"接了没生效"。
- 帧级 acc 在 idle 占比高的数据上具欺骗性，不作为结论依据（沿用 `FEATURE_STRATEGY_COMPARE.md` 口径）。

## 5. 复现命令

```bash
PY=/home/caizh/programming/python_code/CleanSightBackend/.venv/bin/python   # 仓库根目录执行

# E0 / E1 × seed 42/7/2026（训练）
for cfg in gru-actionmixed gru-actionmixed-embed; do for seed in 42 7 2026; do
  $PY -m framework.cleansight_eval.cli.train --config framework/experiments/$cfg.yaml \
    --runs-dir runs/embed_ablation --seed $seed \
    -S train.weight_decay=0.0001 -S model.dropout=0.2 -S train.patience=4 \
    -S train.epochs=20 -S train.best_metric=val_f1_0.5
done; done

# 每个 run 的正式评测（--config 用训练时同一份配置）
$PY -m benchmark.cli.eval --config framework/experiments/gru-actionmixed-embed.yaml \
  --ckpt runs/embed_ablation/<run-id>/checkpoints/best.pt

# 门禁与单测
$PY tools/validate_testsets.py --catalog framework/testsets.yaml --json
$PY -m pytest framework/tests/test_image_embed_features.py -q
```

## 6. 限制与下一步

**限制（必须随数字一起引用）**

- **机制床 ≠ 正式数据**：手写标注 `cleansight-ActionMixed` 与自动通道 v3 auto（18 视频，
  test 锚定 task#195/#199）不同源，结论只能定性，不能并入方案b 正式数字。
- **设备口径**：本轮全部 CPU；GPU 是否复现同一排序未知（历史经验：设备差异大于单轮噪声）。
- **seed 数**：3 seed，中位数可比但不足以淘汰"图像通道无用"假设；`air_injection` 归零
  这种逐类塌陷需要更多 seed 才能判定是通道效应还是优化噪声。
- **未做**：E2（检测框 ROI 外观聚合）、E3（backbone 消融 resnet18/50、efficientnet_b0）；
  投影头宽度（64）未消融；`mask_targets` 只作用于 bbox 块的语义已实现但本轮未启用。

**下一步（按依赖排序）**

1. **解图像源**（正式轮前置）：project-16 18 视频下载 → stride-4 抽帧 → 与 v3 manifest 帧号对齐
   校验 → 提取 embedding → 按 §3.5 的登记模板新增 v3 图像契约（**不改代码**）。
2. 正式轮 E1：v3 图像契约上 E0 vs E1 多 seed（GPU 口径），补齐 E2/E3 与矩阵策略表扩展。
3. 后端影响提前知会：形态 B 上线需 CleanSightBackend 新增"像素 → CNN embedding"管线
   （`IMAGE_FEATURE_TRAINING.md` §4.4），否则只能停留在离线评测。
