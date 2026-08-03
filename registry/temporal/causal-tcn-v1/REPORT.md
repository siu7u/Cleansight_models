# 评估报告：temporal-causal-tcn

## 实验设置

| 项目 | 值 |
| --- | --- |
| 模型 | TCNClassifier |
| 仓库 | `temporal-causal-tcn` |
| 数据视图 | `endo-project-v1` |
| YOLO 版本 | `yolo-v1` |
| 特征映射版本 | `legacy-20d-v1` |
| 特征维度 | 20 |
| 窗口长度 | 64 帧 |
| 类别数 | 3 |
| 标签 | `Idle` / `Long_Brushing` / `Short_Brushing` |
| 训练轮数 | 10 |
| 依赖 | `pytorch-tcn==1.2.3` |
| 训练设备 | 本轮训练为 CPU；初始训练时 PyTorch 未识别到 CUDA |
| 详细评估设备 | CUDA |

模型输入为因果窗口 `[B, 64, 20]`。TCN 配置为 `causal=True`，因此每个时间点只依赖当前帧和历史帧。

## 训练命令

从模型集根目录执行：

```bash
cd temporal-causal-tcn

PYTHONDONTWRITEBYTECODE=1 ../../CleanSightBackend/.venv/bin/python main.py \
  --mode full \
  --model tcn \
  --epochs 10 \
  --window 64 \
  --verbose \
  --output_dir experiments/tcn
```

## 实验结果

| 权重 | 参数量 | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 | 可视化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `registry/tcn-v1/tcn-final-20260704-160652.pt` | 67,587 | 69.23 | 44.62 | 46.81 | 40.43 | 27.66 | `experiments/tcn/tcn-20260704-160748.png` |

## 详细分类指标

从 `temporal-causal-tcn` 目录执行：

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 \
../../CleanSightBackend/.venv/bin/python \
../tools/eval_temporal_detailed.py \
  --repo . \
  --model tcn \
  --checkpoint registry/tcn-v1/tcn-final-20260704-160652.pt
```

说明：该指标使用批量 last-frame logits 直接分类，不包含 `causal_decision` 平滑。

| 类别 | 召回率 |
| --- | ---: |
| Idle | 90.14% |
| Long_Brushing | 32.33% |
| Short_Brushing | 48.48% |

混淆矩阵：行是真值，列是预测。

| 真值 \ 预测 | Idle | Long_Brushing | Short_Brushing |
| --- | ---: | ---: | ---: |
| Idle | 15162 | 763 | 895 |
| Long_Brushing | 2994 | 2394 | 2018 |
| Short_Brushing | 1919 | 44 | 1847 |

## 分类质量

| 指标 | 值 |
| --- | --- |
| 帧准确率 | 69.23 |
| Idle 召回率 | 90.14% |
| Long_Brushing 召回率 | 32.33% |
| Short_Brushing 召回率 | 48.48% |

## 分割稳定性

| 指标 | 值 |
| --- | --- |
| Edit | 44.62 |
| F1@0.1 | 46.81 |
| F1@0.25 | 40.43 |
| F1@0.5 | 27.66 |
| 过分割率 | 本轮未测 |
| 长短刷瞬切次数 | 本轮未测 |

## 在线行为

| 指标 | 值 |
| --- | --- |
| 因果性 | 是，TCN 配置为 `causal=True` |
| 感受野 | 64 帧 |
| 离线-在线落差 | 本轮未测 |
| 动作确认延迟 | 本轮未测 |
| 单 tick 延迟 | 本轮未测 |

## 结论

Causal TCN 是本轮参数量最小的模型，也是轻量在线部署方向的候选。它在三个模型中 `Short_Brushing` 召回最高，但 `Long_Brushing` 召回仍偏低。

当前 Edit 明显低于 GRU 和 Transformer，说明分段边界稳定性或长时序平滑能力不足。

## 后续工作

- 优先测量单 tick 延迟，验证轻量部署优势。
- 尝试窗口长度 32、96、128。
- 调整 TCN 通道数或平滑策略，提高 Edit 和刷洗召回。
- 新 YOLO 到位后，使用最终特征映射重新生成特征并重训。
