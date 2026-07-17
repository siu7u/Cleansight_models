# CleanSight 模型集任务状态

> 历史快照：本文保留 2026-07-05 的旧模型资产状态，不代表当前统一 train-eval framework 的实现
> 进度。当前状态以 [`docs/TRAIN_EVAL_IMPLEMENTATION_STATUS.md`](docs/TRAIN_EVAL_IMPLEMENTATION_STATUS.md)
> 为准，新使用方式见 [`MODELSET_USAGE_GUIDE.md`](MODELSET_USAGE_GUIDE.md)。

更新时间：2026-07-05

本文档按当前模型仓库实际产物记录任务进度。结论分为：

- 完成：已有可复现文件、报告或 registry 记录。
- 部分完成：流程已跑通，但指标未达标或仍缺线上验证。
- 待完成：尚未落地，或依赖后端/新数据继续推进。

## 总体结论

当前模型集已经完成基础仓库搭建、三类时序模型模板、YOLO 分组训练流水线、首轮 YOLO 训练验证、单模型 benchmark 骨架、3 分钟端到端 benchmark 评分器，以及 ModelScope 上传目录整理。

当前尚未完成生产晋升。主要原因是：

- 两个 YOLO 分组模型均已训练和验证，但验收结果为 FAIL。
- 三个时序模型均已训练和评估，但刷洗类召回未达到临时目标。
- 时序模型仍基于 `legacy-20d-v1` 历史 20 维特征，尚未接入新 YOLO 分组特征。
- 3 分钟端到端 benchmark 的评分器已跑通，但真实后端 prediction JSON 仍需由 CleanSightBackend 在线推理导出。

## T-M1 YOLO 训练仓库

状态：部分完成。

已完成：

- 已建立集中式 YOLO 仓库：`yolo-detection/`
- 已接入新 YOLO pipeline：`yolo-detection/pipeline/`
- 已完成数据视图 A 登记：`yolo-detection/data/DATASET_VIEW_A.md`
- 已按目标特性拆分两个检测分组：
  - `group1_large`：hand / scope_control_body / scope_mid_section
  - `group2_small`：syringe / air_gun / scope_distal_end
- 已完成首轮训练和验证：
  - `yolo-detection/pipeline/runs/group1_large/weights/best.pt`
  - `yolo-detection/pipeline/runs/group2_small/weights/best.pt`
- 已登记 registry：
  - `yolo-detection/registry/yolo-group1-large-v1/`
  - `yolo-detection/registry/yolo-group2-small-v1/`
- 已生成评估报告：
  - `yolo-detection/registry/yolo-group1-large-v1/eval_report.md`
  - `yolo-detection/registry/yolo-group2-small-v1/eval_report.md`
- 已整理 ModelScope 上传目录：
  - `modelscope_upload/yolo-group1-large-v1/`
  - `modelscope_upload/yolo-group2-small-v1/`

当前指标：

| 分组 | 状态 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| `group1_large` | FAIL | 0.522 | 0.181 | 0.594 | 0.501 |
| `group2_small` | FAIL | 0.343 | 0.200 | 0.351 | 0.394 |

待完成：

- 补充或重切 `group2_small` 验证集，解决 `syringe` / `scope_distal_end` 无法评估的问题。
- 提升 `scope_control_body`、`scope_mid_section`、`air_gun` 的召回。
- 指标达标后再登记生产可用 YOLO tag。
- 将晋升后的 YOLO 版本同步到时序模型 `pin.yaml`。

## T-M2 时序模型仓库模板

状态：完成模板，模型结果为部分完成。

已完成：

- 已建立三套时序模型仓库：
  - `temporal-gru/`
  - `temporal-causal-tcn/`
  - `temporal-transformer/`
- 每个仓库均包含模板要求字段：
  - `feature_mapping.py`
  - `build_testset.py`
  - `CARD.md`
  - `pin.yaml`
  - `REPORT.md`
- 已完成首轮训练和 registry 权重登记：
  - `temporal-gru/registry/gru-v1/gru-final-20260704-150629.pt`
  - `temporal-causal-tcn/registry/tcn-v1/tcn-final-20260704-160652.pt`
  - `temporal-transformer/registry/transformer-v1/transformer-final-20260704-161653.pt`
- 已完成逐类召回和混淆矩阵评估。
- 已整理 ModelScope 上传目录：
  - `modelscope_upload/temporal-gru-v1/`
  - `modelscope_upload/temporal-causal-tcn-v1/`
  - `modelscope_upload/temporal-transformer-v1/`

当前指标：

| 模型 | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GRU | 68.54 | 70.77 | 48.74 | 40.34 | 25.21 | 不晋升 |
| Causal TCN | 69.23 | 44.62 | 46.81 | 40.43 | 27.66 | 不晋升 |
| Transformer | 69.70 | 66.15 | 46.43 | 41.07 | 33.93 | 不晋升 |

待完成：

- 用新 YOLO 分组模型生成最终在线/离线同源特征。
- 重训三个时序模型并重新打榜。
- 补齐单 tick 延迟。
- 补齐离线-在线落差。

## T-M3 魔搭模型版本管理与一键复刻

状态：部分完成。

已完成：

- 每个时序模型仓库已包含 `pin.yaml`。
- YOLO registry 已记录数据视图、类别、训练配置、指标和评估报告。
- 已按 ModelScope 上传需要整理出本地目录：
  - `modelscope_upload/yolo-group1-large-v1/`
  - `modelscope_upload/yolo-group2-small-v1/`
  - `modelscope_upload/temporal-gru-v1/`
  - `modelscope_upload/temporal-causal-tcn-v1/`
  - `modelscope_upload/temporal-transformer-v1/`

待完成：

- 上传 ModelScope 并记录真实模型仓库地址、revision 或 tag。
- 完善 `pin.yaml` schema，使 dataset / yolo / temporal model / feature_mapping 都可解析。
- 实现一键复刻脚本：读取 `pin.yaml`，拉齐 dataset、YOLO、时序模型和映射版本。
- 增加 CARD 门禁校验脚本，检查延迟、召回、离线-在线落差是否齐全。

## Benchmark 状态

### 单模型 benchmark

状态：部分完成。

已完成：

- YOLO 单模型 benchmark 汇总：`benchmark/single_model/yolo_summary.md`
- 端到端评分脚本：`benchmark/e2e_3min/run_e2e_benchmark.py`
- 3 分钟 case：`benchmark/e2e_3min/cases/clean_001.yaml`

待完成：

- 时序单模型 benchmark 需要补齐或重新跑：`benchmark/single_model/run_temporal_benchmark.py`
- 延迟指标需要写回各模型 `CARD.md`。
- benchmark 输出应统一从脚本生成，减少手工维护。

### 3 分钟端到端 benchmark

状态：评分器完成，真实在线 prediction 待接入。

已完成：

- case 文件已建立：`benchmark/e2e_3min/cases/clean_001.yaml`
- 报告已生成：`benchmark/e2e_3min/reports/clean_001.md`
- 使用 prediction JSON 时，评分器可输出 PASS / FAIL、动作召回和阶段时间误差。

注意：

- 当前 `clean_001.md` 的 PASS 只能说明评分器在给定 prediction JSON 时能跑通。
- 真实端到端验收还需要 CleanSightBackend 对 3 分钟视频完成在线推理，并导出 `benchmark/e2e_3min/outputs/clean_001.prediction.json`。

## 原 TODO 对照

| 编号 | 主题 | 当前状态 | 证据位置 |
| --- | --- | --- | --- |
| M1 | YOLO 检测验收线 | 部分完成 | `yolo-detection/registry/*/eval_report.md` |
| M2 | YOLO 架构对照范围 | 待完成 | 当前仅完成分组 YOLO 首轮训练 |
| M3 | 时序候选模型优先级/打榜方式 | 部分完成 | 三个 `temporal-*/REPORT.md` 和 `CARD.md` |
| M4 | 特征通道定稿 | 部分完成 | `feature_mapping.py` 有契约骨架，但新 YOLO 特征未落地 |
| M5 | conf 阈值 / 抽帧率 | 部分完成 | `yolo-detection/pipeline/config.yaml`、`DATASET_VIEW_A.md` |
| M6 | P0 同源核查 | 待完成 | 尚未实现离线训练特征和在线推理特征共用同一 `step()` |
| M7 | 集中评测 harness | 部分完成 | `benchmark/` |
| M8 | 窗口大小 / 平滑参数 | 待完成 | 当前窗口主要为 64 |
| M9 | 指标口径校准 | 部分完成 | YOLO 与时序报告已有，但仍需和 backend bundle eval 对账 |

## 下一步建议

1. 先把本仓库代码和文档提交 GitHub，继续忽略 `.pt`、视频、数据集和训练输出。
2. 上传 `modelscope_upload/` 下的五个模型版本到 ModelScope。
3. 回填 ModelScope 地址和 revision 到各 `pin.yaml`。
4. 在 CleanSightBackend 生成真实 `clean_001.prediction.json`，完成 3 分钟端到端真实验收。
5. 实现离线训练特征和在线推理特征共用同一个 `step()`。
6. 用新 YOLO 特征重训时序模型，再更新 `CARD.md` 和 `REPORT.md`。
