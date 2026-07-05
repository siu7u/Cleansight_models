# 数据视图 A 引用说明

本文档记录 YOLO 检测 checkpoint 使用的数据源。模型仓库不能在已登记版本背后静默替换数据集。

## 数据源

- 数据集名称：CleanSight CLEAN 检测框数据集
- 来源仓库：本模型集 `yolo-detection/pipeline`
- 导出时间：2026-07-04 11:10
- 标注工具：Label Studio
- 数据视图：A - 检测框标注
- 视频数量：13
- 入选训练/验证视频数量：10
- 抽帧间隔：`stride=12`
- 切分真源：`yolo-detection/pipeline/splits.yaml`
- 配置真源：`yolo-detection/pipeline/config.yaml`

## 类别

当前 CLEAN 阶段检测套件按目标尺寸分为两个 YOLO 分组模型。

### group1_large

- `hand`
- `scope_control_body`
- `scope_mid_section`

### group2_small

- `syringe`
- `air_gun`
- `scope_distal_end`

未列入当前分组的 Label Studio 类别不会进入本次 YOLO 数据集。

## 已登记 YOLO 版本

| 版本 | 分组 | 权重 | 评估报告 |
| --- | --- | --- | --- |
| `yolo-group1-large-v1` | `group1_large` | `yolo-detection/pipeline/runs/group1_large/weights/best.pt` | `yolo-detection/registry/yolo-group1-large-v1/eval_report.md` |
| `yolo-group2-small-v1` | `group2_small` | `yolo-detection/pipeline/runs/group2_small/weights/best.pt` | `yolo-detection/registry/yolo-group2-small-v1/eval_report.md` |

## 当前验收状态

当前两个 YOLO 分组模型已完成首轮训练和验证，但均未通过验收。

| 分组 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `group1_large` | 0.522 | 0.181 | 0.594 | 0.501 | FAIL |
| `group2_small` | 0.343 | 0.200 | 0.351 | 0.394 | FAIL |

主要阻塞：

- `group1_large`：`scope_control_body` precision / recall 偏低，`scope_mid_section` recall 未达标。
- `group2_small`：整体指标偏低，`syringe` 与 `scope_distal_end` 当前无法评估，`air_gun` precision / recall 未达标。

## 规则

- 一个 YOLO checkpoint 版本必须指向唯一的数据视图版本。
- 如果新增数据或重标数据，必须登记新的数据源版本。
- `config.yaml` 中的 `groups` 顺序决定 YOLO class id，已训练版本不能重排类别。
- `splits.yaml` 是 train / val / test / e2e_test 切分的唯一真源。
- 由 YOLO checkpoint 生成的特征化数据属于时序模型仓库，不属于数据集引用文档。
- 当前权重文件不直接进入 git，应通过 ModelScope、Git LFS 或外部模型仓库登记。