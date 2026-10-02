# CleanSight YOLO 优化工作流指南

本文是 YOLO 检测优化的唯一操作入口。训练/推理/评测全部走 framework + benchmark，
不保留独立优化脚本。

## 目标

- **group1_large**（hand / scope_control_body / scope_mid_section）：通过多方法对比，把
  整体 Precision 与 Recall 提升到 **≥ 0.7**。
- **group2_small**（syringe / air_gun / scope_distal_end / short_brush / brush_tip_out）：
  逐类分析后，**P/R 无法超过 0.3 的类从 YOLO 淘汰**。原先计划改用 ROI 图像特征融合
  （`roi_classification`），该方案已于 **2026-10-02 评估为不采用**（理由见
  [`MODELSET_OVERVIEW.md`](MODELSET_OVERVIEW.md) §9.1）——淘汰类目前**没有替代方案**。

## 1. 多方法优化实验（sweep）

```bash
# 单预设（先跑基线看方向）
python -m framework.cleansight_eval.cli.sweep --group group1_large --preset large_baseline

# 多个预设一起跑
python -m framework.cleansight_eval.cli.sweep --group group1_large \
    --preset large_baseline large_s large_s_960 large_m_960

# grid 搜索（模型 × 分辨率 × 增强）
python -m framework.cleansight_eval.cli.sweep --group group2_small --grid models resolutions

# 预览计划，不真跑
python -m framework.cleansight_eval.cli.sweep --group group1_large --preset large_m_960 --dry-run

# 指定设备
python -m framework.cleansight_eval.cli.sweep --group group2_small --preset small_s_1280_p2 --device 0

# 快速探针（--smoke）：截断 epochs/patience + fraction 子采样，仅用于预设方向对比
# （结果强制标注 SMOKE，不代表正式指标）；探针选出方向后，去掉 --smoke 对赢家跑全量
python -m framework.cleansight_eval.cli.sweep --group group1_large \
    --preset large_s large_s_960 large_m_960 --smoke
```

### 可用预设

| 组 | 预设 | 说明 |
|---|---|---|
| large | `large_baseline` | yolo11n + 640（基线） |
| large | `large_s` / `large_m` | 更大模型 |
| large | `large_s_960` / `large_s_1280` | 更高分辨率 |
| large | `large_m_960` | yolo11m + 960 + 强增强 + cos lr + label smoothing（推荐冲刺） |
| large | `large_s_freeze` | 冻结 backbone 10 层 |
| small | `small_baseline` | yolo11n + 640（基线） |
| small | `small_s_960` | yolo11s + 960 + 强增强 |
| small | `small_n_1280_p2` / `small_s_1280_p2` | 1280 + P2 特征头（小目标专项） |
| small | `small_m_1280` | yolo11m + 1280 + 强增强（最重） |
| small | `small_s_copy_paste` | copy_paste 增强（稀有类） |

### 输出

每个实验跑「训练 → val 评测」，指标与逐类 P/R/mAP50 写入
`runs/optimize_reports/optimize_<group>_<ts>.json|md`；命令行直接打印汇总与最佳实验。

## 2. 逐类阈值分析与淘汰决策（analyze）

```bash
python -m benchmark.cli.analyze \
    --config framework/experiments/yolo-clean-small.yaml \
    --ckpt runs/cleansight-yolo/opt-group2_small-*/weights/best.pt \
    --threshold 0.3
```

- 复用 framework `DetectionPipeline.predict` 的 native_metrics，不直接调 adapter。
- 输出三类决策：**保留**（P≥阈值且 R≥阈值）、**边界**（0.5×阈值~阈值，可再优化）、
  **淘汰**（<0.5×阈值 → 从检测器剔除；原计划的 ROI 特征融合替代方案已停用，见 §4）。
- 对淘汰类自动调用 `framework/cleansight_eval/detection/data_tools.build_trimmed_dataset`
  生成裁剪数据集 `datasets/cleansight-yolo/<group>_kept/`（过滤淘汰类 labels、重映射 class id）。
- 报告写入 `runs/small_analysis/analysis_<group>_<ts>.json|md`，含多 conf 扫描表。

## 3. 裁剪后重训（可选）

```bash
# 用裁剪数据集重训（保留类）
python -m framework.cleansight_eval.cli.sweep --group group2_small_kept --preset small_s_1280_p2
```

## 4. ~~淘汰类走 ROI 特征融合（roi_classification）~~ —— **已停用（2026-10-02）**

> **本方案已评估为不采用**，配置 `roi-fusion{,‑heldout}.yaml` 与模型别名 `feature_fusion` 均已移除，
> 实验产物已回收；`framework/cleansight_eval/classification/` 实现保留以备"去框化"改造复用。
>
> **三条理由**（详见 [`MODELSET_OVERVIEW.md`](MODELSET_OVERVIEW.md) §9.1）：
> ① **无法验收**——要救的 `brush_tip_out` 在留出集只有 **21 个正样本**，precision 的 95% CI 宽达
> **[0.38, 0.70]**，撑不起任何性能声明；② **循环依赖**——训练/评估都靠 GT 框，而它要救的类正是
> 检不出框的类，落地必须先"去框化"；③ **它服务的淘汰决策从未执行**——仓库里没有
> `group2_small_kept`，时序特征仍在用全部 8 个检测类。

原操作步骤（保留备查）：把 `data.classes` 换成 analyze 输出的淘汰类（如 `[air_gun, brush_tip_out]`），
从 `data.group_dir` 的 GT 框裁 ROI 正样本 + 随机背景负样本，训练 CNN backbone + MLP 多标签头。

## 5. 对比与验收

```bash
# 汇总所有评估结果（三态矩阵）
python -m benchmark.cli.matrix --runs runs

# 校验 catalog
python tools/validate_testsets.py --catalog framework/testsets.yaml --json
```

## 6. 硬件与预算建议

- RTX 4090：完整流水线约 3~4 天（早停生效）；可先跑 `large_baseline + large_s`（5~8 小时）
  看趋势再决定。
- RTX 4060 (8GB)：完整流水线约 2~3 周，不划算；只跑预算版（large_baseline + large_s +
  small_baseline，1~1.5 天），并把高分辨率预设的 `batch` 调小（640→8、960→4、1280→2）。
- 不要用 CPU 跑全量（一个 150 epoch 实验约 37 天）。
- 想更快：`epochs` 降到 100、`patience` 降到 15；多卡时并行跑多个预设。

## 7. 相关文件

| 能力 | 位置 |
|---|---|
| sweep 逻辑与预设 | `framework/cleansight_eval/detection/sweep.py` |
| sweep CLI | `framework/cleansight_eval/cli/sweep.py` |
| 数据集裁剪 | `framework/cleansight_eval/detection/data_tools.py` |
| analyze CLI | `benchmark/cli/analyze.py` |
| ROI 分类模型 | `framework/cleansight_eval/classification/` |
| 分类评估器 | `benchmark/evaluators/classification.py` |
| 示例配置（已删除） | ~~`framework/experiments/roi-fusion.yaml`~~（2026-10-02 随方案停用移除） |
