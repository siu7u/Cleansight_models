# 评估报告：temporal-transformer

## 实验设置

| 项目 | 值 |
| --- | --- |
| 模型 | TransformerClassifier |
| 仓库 | `temporal-transformer` |
| 数据视图 | `endo-project-v1` |
| YOLO 版本 | `yolo-v1` |
| 特征映射版本 | `legacy-20d-v1` |
| 特征维度 | 20 |
| 窗口长度 | 64 帧 |
| 类别数 | 3 |
| 标签 | `Idle` / `Long_Brushing` / `Short_Brushing` |
| 训练轮数 | 10 |
| 训练设备 | 本轮训练为 CPU；初始训练时 PyTorch 未识别到 CUDA |
| 详细评估设备 | CUDA |

模型输入为因果窗口 `[B, 64, 20]`。Transformer Encoder 使用因果注意力 mask，每一帧只能关注自己和历史帧。

## 训练命令

从模型集根目录执行：

```bash
cd temporal-transformer

PYTHONDONTWRITEBYTECODE=1 ../../CleanSightBackend/.venv/bin/python main.py \
  --mode full \
  --model transformer \
  --epochs 10 \
  --window 64 \
  --verbose \
  --output_dir experiments/transformer
```

## 实验结果

| 权重 | 参数量 | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 | 可视化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `registry/transformer-v1/transformer-final-20260704-161653.pt` | 400,515 | 69.70 | 66.15 | 46.43 | 41.07 | 33.93 | `experiments/transformer/transformer-20260704-161738.png` |

## 详细分类指标

从 `temporal-transformer` 目录执行：

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 \
../../CleanSightBackend/.venv/bin/python \
../tools/eval_temporal_detailed.py \
  --repo . \
  --model transformer \
  --checkpoint registry/transformer-v1/transformer-final-20260704-161653.pt
```

说明：该指标使用批量 last-frame logits 直接分类，不包含 `causal_decision` 平滑。

| 类别 | 召回率 |
| --- | ---: |
| Idle | 94.98% |
| Long_Brushing | 31.43% |
| Short_Brushing | 40.71% |

混淆矩阵：行是真值，列是预测。

| 真值 \ 预测 | Idle | Long_Brushing | Short_Brushing |
| --- | ---: | ---: | ---: |
| Idle | 15976 | 396 | 448 |
| Long_Brushing | 3468 | 2328 | 1610 |
| Short_Brushing | 2175 | 84 | 1551 |

## 分类质量

| 指标 | 值 |
| --- | --- |
| 帧准确率 | 69.70 |
| Idle 召回率 | 94.98% |
| Long_Brushing 召回率 | 31.43% |
| Short_Brushing 召回率 | 40.71% |

## 分割稳定性

| 指标 | 值 |
| --- | --- |
| Edit | 66.15 |
| F1@0.1 | 46.43 |
| F1@0.25 | 41.07 |
| F1@0.5 | 33.93 |
| 过分割率 | 本轮未测 |
| 长短刷瞬切次数 | 本轮未测 |

## 在线行为

| 指标 | 值 |
| --- | --- |
| 因果性 | 是，使用因果注意力 mask |
| 感受野 | 64 帧 |
| 离线-在线落差 | 本轮未测 |
| 动作确认延迟 | 本轮未测 |
| 单 tick 延迟 | 本轮未测 |

## 结论

Transformer 在整体指标上是当前最强的精度基线，Acc 和 F1@0.5 均为三者最高。但详细召回显示它明显偏向 `Idle`，对刷洗动作的召回仍不足。

因此它更适合作为当前上限参考，不应直接作为上线候选。上线前需要提升刷洗召回，并补测单 tick 延迟。

## 后续工作

- 测量部署机单 tick 延迟。
- 固定随机种子并在 GPU 可用环境下重跑。
- 尝试类别重采样或损失权重，提升刷洗召回。
- 新 YOLO 到位后，使用最终特征映射重新生成特征并重训。
