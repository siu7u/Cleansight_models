# 时序喂法 Benchmark：整段喂 vs 流式喂

本报告比较同一 checkpoint 在两种输入方式下的结果：

- 整段喂：一次输入完整特征序列 `[1, T, F]`。
- 流式喂：每 tick 输入最近 `window` 帧 `[1, window, F]`，只取最后一帧预测。

评估时裁掉前 `window - 1` 帧，使两种模式在同一帧范围上对比。

| 模型 | 视频数 | 最多帧数 | 输入维度 | Full Acc | Stream Acc | Full Edit | Stream Edit | Full F1@0.5 | Stream F1@0.5 | 一致率 | Stream p95 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gru | 4 | 全量 | 20 | 87.29 | 90.99 | 41.62 | 37.92 | 28.42 | 19.75 | 91.53% | 0.6350 ms |
| tcn | 4 | 全量 | 20 | 85.18 | 85.18 | 4.96 | 4.97 | 4.52 | 4.52 | 99.98% | 2.6282 ms |
| transformer | 4 | 全量 | 20 | 65.77 | 90.59 | 9.84 | 31.83 | 2.75 | 28.35 | 70.35% | 2.4183 ms |

## 逐类召回

### gru

| 类别 | Full Recall | Stream Recall |
| --- | ---: | ---: |
| Idle | 91.02% | 96.85% |
| Long_Brushing | 78.18% | 80.32% |
| Short_Brushing | 82.92% | 77.75% |

### tcn

| 类别 | Full Recall | Stream Recall |
| --- | ---: | ---: |
| Idle | 89.50% | 89.51% |
| Long_Brushing | 76.44% | 76.43% |
| Short_Brushing | 76.90% | 76.88% |

### transformer

| 类别 | Full Recall | Stream Recall |
| --- | ---: | ---: |
| Idle | 94.37% | 97.02% |
| Long_Brushing | 0.61% | 79.40% |
| Short_Brushing | 23.97% | 75.11% |

