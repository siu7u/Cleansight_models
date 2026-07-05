# YOLO 评估报告：yolo-group2-small-v1

## 版本信息

- 检测分组：`group2_small`
- YOLO 版本：`yolo-group2-small-v1`
- 数据视图 A：`project-10-at-2026-07-04-11-10-13954db4`
- 权重：`yolo-detection/pipeline/runs/group2_small/weights/best.pt`
- 训练配置：`registry/yolo-group2-small-v1/train_config.yaml`
- 类别配置：`registry/yolo-group2-small-v1/classes.yaml`
- 来源流水线：`yolo-detection/pipeline`
- 验收报告：`yolo-detection/pipeline/runs/group2_small/acceptance_report.md`

## 类别

| class_id | 类别 |
| ---: | --- |
| 0 | syringe |
| 1 | air_gun |
| 2 | scope_distal_end |

## 总体指标

| 指标 | 值 | 门槛 |
| --- | ---: | ---: |
| mAP@0.5 | 0.343 | >= 0.5 |
| mAP@0.5:0.95 | 0.200 | >= 0.3 |
| Precision | 0.351 | - |
| Recall | 0.394 | - |

## 逐类指标

| 类别 | Precision | Recall | mAP@0.5 | 备注 |
| --- | ---: | ---: | ---: | --- |
| syringe | - | - | - | 验证集无样本或未检出，无法评估 |
| air_gun | 0.351 | 0.394 | 0.343 | precision / recall 未达标 |
| scope_distal_end | - | - | - | 验证集无样本或未检出，无法评估 |

## 小目标召回

| 指标 | 值 |
| --- | --- |
| 小目标类别 | syringe / air_gun / scope_distal_end |
| 小目标 Recall | 当前仅 air_gun 可计算：0.394 |
| 备注 | syringe 与 scope_distal_end 缺少可评估结果，需补验证样本或检查 split |

## 运行时指标

| 项目 | 值 |
| --- | --- |
| 部署设备 | NVIDIA GeForce RTX 4060 Laptop GPU |
| preprocess | 2.6 ms / image |
| inference | 7.4 ms / image |
| postprocess | 0.9 ms / image |

## 晋升结论

- 是否晋升：否
- tag：`yolo-group2-small-v1`
- 结论：训练和验证已完成，但未通过当前验收线。
- 阻塞问题：
  - 整体 mAP@0.5 与 mAP@0.5:0.95 未达标。
  - `air_gun` precision / recall 未达标。
  - `syringe` 与 `scope_distal_end` 当前无法评估，验证集覆盖不足或未检出。

