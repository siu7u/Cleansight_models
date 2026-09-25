# 报告图表库（`docs/figures/`）

> **用途**：周报 / 汇报 / 实验报告需要**用图佐证数字**。本目录是仓库内可复用图表的**唯一存放处**——
> 每张图都配一份同名 JSON 旁证（`.json`），记录**该图实际画出的数字 + 来源标注**，便于审计与复用。
> 实验报告专属的插图仍可留在各自的报告目录（先例：`docs/mstcn-capacity/figures/`），
> **跨报告复用、周报引用**的图放这里。

## 1. 现有图表

| 图 | 内容 | 数据来源 |
|---|---|---|
| [`fig1_capacity_vs_params.png`](fig1_capacity_vs_params.png) | 容量 vs 段级 edit（4 条配方曲线，误差棒 = seed 极差） | [`docs/mstcn-capacity/figures/capacity_vs_segmental.json`](../mstcn-capacity/figures/capacity_vs_segmental.json)（真实 run 级数据）+ 容量研究 §0/§1 |
| [`fig2_architecture_vs_metrics.png`](fig2_architecture_vs_metrics.png) | 架构族对照（edit / F1@0.1 / F1@0.25）+ **逐 seed 摆幅** | 特征杠杆报告 §2.3 |
| [`fig3_feature_contract_gap.png`](fig3_feature_contract_gap.png) | 各替代契约相对 `roi-grid-144` 的 **insert 召回缺口** + 配对 p 值 | 特征杠杆报告 §2.2 |
| [`fig4_selection_metric.png`](fig4_selection_metric.png) | 选点口径对照（`val_f1_0.5` / `val_edit` / `val_f1_0.25`） | 特征杠杆报告 §2.4 |
| [`fig5_timesfm_probe.png`](fig5_timesfm_probe.png) | TimesFM 四问：点预测 MAE / 区间校准 / 切换点 F1 / 提前预警命中率 | TimesFM 报告 §4 |
| [`fig6_serving_latency.png`](fig6_serving_latency.png) | 在线代价：TimesFM 单序列 vs 帧预算 vs GRU 单 tick（对数刻度） | TimesFM 报告 §4.5 + [`docs/INFERENCE_CHAIN_PERF.md`](../INFERENCE_CHAIN_PERF.md) |
| [`fig7_image_embed_e0_e1.png`](fig7_image_embed_e0_e1.png) | 图像 embedding E0 vs E1（段级升、帧级降的混合结果） | E1 报告 §4 |

## 2. 重新生成

```bash
PY=/home/caizh/programming/python_code/CleanSightBackend/.venv/bin/python   # 仓库根目录执行
MPLCONFIGDIR=$PWD/tmp/fig_mplcache PYTHONPATH=. $PY tools/plot_report_figures.py
# 只重画一张：
MPLCONFIGDIR=$PWD/tmp/fig_mplcache PYTHONPATH=. $PY tools/plot_report_figures.py --only fig2_architecture_vs_metrics
```

脚本：[`tools/plot_report_figures.py`](../../tools/plot_report_figures.py)。跑完会同时覆盖 `.png` 与 `.json`。

## 3. 数字一致性纪律（重要）

- 图中每个数字都**必须能在其 `.json` 的 `source` 指向的报告里查到原文**；
- 改图 = **先改报告，再改脚本**，两边必须一致；**不允许**在这里维护"比报告更新"的数字；
- 唯一例外：`fig1` 直接读既有真实 run 级数据（不复制、不转抄）；
- 单位换算（如报告记 `0.069 s`、图中呈现 `69 ms`）必须在旁证 JSON 的 `unit_note` 里写明。

## 4. 字体说明

本机**没有中文字体**，matplotlib 渲染中文会出豆腐块（脚本会抛 `Glyph ... missing from font` 警告）。
因此**图内标签一律英文**，中文解释写在引用该图的报告正文与图注中。若将来装上 CJK 字体
（如 Noto Sans CJK），可把脚本里的英文标签换成中文，并确认无字形警告。

## 5. 引用方式

在 Markdown 报告中用相对路径引用即可：

```markdown
![容量 vs 参数量](figures/fig1_capacity_vs_params.png)
```

> 引用时**务必在图注里复述口径**（seed 数 / 选点口径 / 设备 / 数据 split），
> 这是 [`docs/EVAL.md`](../EVAL.md) 与 [`usage/YAML_CONFIG.md`](../../usage/YAML_CONFIG.md) 的硬性要求。
