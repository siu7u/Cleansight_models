# feature-lab 结果可视化（图表资产）

> 生成日期 2026-09-23。全部数字来自 [FEATURE_LAB.md](../../FEATURE_LAB.md) §6–§10 已完成实验，
> 口径：GRU + 3-seed 中位数（除窗口实验 w32 n=2 / w64 n=1、LOVO 单折外），CPU/WSL exploratory，
> 指标均为越高越好。

## 图片清单

| 文件 | 内容 |
|---|---|
| `fig1_overview.png` | 准确率总览：特征方案演进（val edit）、val/test 双口径 edit 与 F1@0.25、LOVO 18 折分布 |
| `fig2_analysis.png` | 窗口长度实验（上下文瓶颈假设证伪）+ nodep 增益逐类 F1 分解 |
| `fig3_lovo_summary.png` | LOVO 18 折 GT vs 预测段汇总（按 edit 升序），红框标最难 fold12 (12.0) / 最易 fold09 (71.9) |
| `lovo_probs/foldXX_*.png` | 每折 held-out 视频的"各标签概率-时间"图：上=6 类 softmax 概率曲线 + GT 背景色带，下=GT vs 预测段条 |

## 生成方式（scripts/）

| 脚本 | 运行环境 | 作用 |
|---|---|---|
| `report_charts_gen.py` | Windows Python（需 matplotlib + 微软雅黑），仓库根目录执行 `python tmp/report_charts_gen.py` | 由 FEATURE_LAB 内嵌数字生成 fig1/fig2 |
| `lovo_prob_extract.py` | WSL `.venv/bin/python`（torch），复用框架 load_checkpoint/build_model/load_split | 对 18 折 best.pt 在 held-out 视频上逐窗推理，softmax 概率存 npz（中间产物在本地 tmp/，不入库） |
| `lovo_prob_charts_plot.py` | Windows Python | 读 npz 画 18 张单图 + fig3 |
| `lovo_summary_replot.py` | Windows Python | 只重画 fig3（秒级） |

注意：脚本内的输入/输出路径按其原位置（`tmp/`，本地工作区）写死；移出后运行需相应调整路径。
LOVO 概率的推理语义与训练期 `_evaluate_sliding_window` 一致（逐窗末帧、原始 softmax、无
causal_decision 平滑；冷启动前 window-1 帧置 idle=1.0，与 predict() 冷启动约定相同）。
