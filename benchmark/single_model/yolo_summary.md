# YOLO 单模型 Benchmark 汇总

- 流水线：`yolo-detection/pipeline`
- 验证退出码：`2`
- 当前结论：两个分组均已训练和验证，但均未通过验收。

| 组 | 结论 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 报告 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| group1_large | FAIL | 0.522 | 0.181 | 0.594 | 0.501 | `yolo-detection/pipeline/runs/group1_large/acceptance_report.md` |
| group2_small | FAIL | 0.343 | 0.200 | 0.351 | 0.394 | `yolo-detection/pipeline/runs/group2_small/acceptance_report.md` |

## 逐类召回

### group1_large

| 类别 | Precision | Recall | mAP@0.5 |
| --- | ---: | ---: | ---: |
| hand | 0.782 | 0.714 | 0.773 |
| scope_control_body | 0.188 | 0.249 | 0.125 |
| scope_mid_section | 0.811 | 0.538 | 0.668 |

未达标项：

- 整体 mAP50-95 0.181 < 0.3
- scope_control_body recall 0.249 < 0.7
- scope_control_body precision 0.188 < 0.5
- scope_mid_section recall 0.538 < 0.7

### group2_small

| 类别 | Precision | Recall | mAP@0.5 |
| --- | ---: | ---: | ---: |
| syringe | NA | NA | NA |
| air_gun | 0.351 | 0.394 | 0.343 |
| scope_distal_end | NA | NA | NA |

未达标项：

- 整体 mAP50 0.343 < 0.5
- 整体 mAP50-95 0.200 < 0.3
- syringe 验证集无样本或未检出，无法评估
- scope_distal_end 验证集无样本或未检出，无法评估
- air_gun recall 0.394 < 0.7
- air_gun precision 0.351 < 0.5
