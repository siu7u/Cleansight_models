# YOLO 评估报告：yolo-group1-large-v1

## 版本信息

- 检测分组：`group1_large`
- YOLO 版本：`yolo-group1-large-v1`
- 数据视图 A：`project-10-at-2026-07-04-11-10-13954db4`
- 权重：`legacy/yolo-detection/pipeline/runs/group1_large/weights/best.pt`
- 训练配置：`registry/yolo-group1-large-v1/train_config.yaml`
- 类别配置：`registry/yolo-group1-large-v1/classes.yaml`
- 来源流水线：`legacy/yolo-detection/pipeline`
- 验收报告：`legacy/yolo-detection/pipeline/runs/group1_large/acceptance_report.md`

## 类别

| class_id | 类别 |
| ---: | --- |
| 0 | hand |
| 1 | scope_control_body |
| 2 | scope_mid_section |

## 总体指标

| 指标 | 值 | 门槛 |
| --- | ---: | ---: |
| mAP@0.5 | 0.522 | >= 0.5 |
| mAP@0.5:0.95 | 0.181 | >= 0.3 |
| Precision | 0.594 | - |
| Recall | 0.501 | - |

## 逐类指标

| 类别 | Precision | Recall | mAP@0.5 | 备注 |
| --- | ---: | ---: | ---: | --- |
| hand | 0.782 | 0.714 | 0.773 | 通过当前逐类门槛 |
| scope_control_body | 0.188 | 0.249 | 0.125 | precision / recall 未达标 |
| scope_mid_section | 0.811 | 0.538 | 0.668 | recall 未达标 |

## 运行时指标

| 项目 | 值 |
| --- | --- |
| 部署设备 | NVIDIA GeForce RTX 4060 Laptop GPU |
| preprocess | 1.1 ms / image |
| inference | 2.0 ms / image |
| postprocess | 1.3 ms / image |

## 晋升结论

- 是否晋升：否
- tag：`yolo-group1-large-v1`
- 结论：训练和验证已完成，但未通过当前验收线。
- 阻塞问题：
  - 整体 mAP@0.5:0.95 未达标。
  - `scope_control_body` precision / recall 明显偏低。
  - `scope_mid_section` recall 未达标。

