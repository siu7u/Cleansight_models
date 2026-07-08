# 时序单模型 Benchmark 汇总

| 模型 | checkpoint | Idle Recall | Long Recall | Short Recall | 延迟 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| gru | `registry/gru-v1/gru-final-20260704-150629.pt` | 88.67% | 34.73% | 42.97% | 0.568 ms | OK |
| tcn | `registry/tcn-v1/tcn-final-20260704-160652.pt` | 90.14% | 32.33% | 48.48% | 1.568 ms | OK |
| transformer | `registry/transformer-v1/transformer-final-20260704-161653.pt` | 94.98% | 31.43% | 40.71% | 1.819 ms | OK |
