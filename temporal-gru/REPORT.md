# 评估报告：temporal-gru

## 实验设置

| 项目 | 值 |
| --- | --- |
| 模型 | GRUClassifier |
| 仓库 | `temporal-gru` |
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

模型输入为因果窗口 `[B, 64, 20]`，输出窗口最后一帧的动作类别。

## 训练命令

从模型集根目录执行：

```bash
cd temporal-gru

PYTHONDONTWRITEBYTECODE=1 ../../CleanSightBackend/.venv/bin/python main.py \
  --mode full \
  --model gru \
  --epochs 10 \
  --window 64 \
  --verbose \
  --auto_save \
  --save_dir checkpoints/gru \
  --export_dir registry/gru-v1 \
  --output_dir experiments/gru
```

## 已完成实验

| 实验 | 权重 | 参数量 | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 | 可视化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 代表性 checkpoint | `registry/gru-v1/gru-final-20260704-150629.pt` | 256,131 | 68.54 | 70.77 | 48.74 | 40.34 | 25.21 | `experiments/gru/gru-20260704-154904.png` |
| 第二次 10 epoch | `registry/gru-v1/gru-final-20260704-155233.pt` | 256,131 | 55.25 | 71.08 | 47.30 | 33.78 | 14.86 | `experiments/gru/gru-20260704-155246.png` |

## 详细分类指标

从 `temporal-gru` 目录执行：

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 \
../../CleanSightBackend/.venv/bin/python \
../tools/eval_temporal_detailed.py \
  --repo . \
  --model gru \
  --checkpoint registry/gru-v1/gru-final-20260704-150629.pt
```

说明：该指标使用批量 last-frame logits 直接分类，不包含 `causal_decision` 平滑。

| 类别 | 召回率 |
| --- | ---: |
| Idle | 88.67% |
| Long_Brushing | 34.73% |
| Short_Brushing | 42.97% |

混淆矩阵：行是真值，列是预测。

| 真值 \ 预测 | Idle | Long_Brushing | Short_Brushing |
| --- | ---: | ---: | ---: |
| Idle | 14915 | 684 | 1221 |
| Long_Brushing | 2288 | 2572 | 2546 |
| Short_Brushing | 2041 | 132 | 1637 |

## 分类质量

| 指标 | 值 |
| --- | --- |
| 帧准确率 | 68.54 |
| Idle 召回率 | 88.67% |
| Long_Brushing 召回率 | 34.73% |
| Short_Brushing 召回率 | 42.97% |

## 分割稳定性

| 指标 | 值 |
| --- | --- |
| Edit | 70.77 |
| F1@0.1 | 48.74 |
| F1@0.25 | 40.34 |
| F1@0.5 | 25.21 |
| 过分割率 | 本轮未测 |
| 长短刷瞬切次数 | 本轮未测 |

## 在线行为

| 指标 | 值 |
| --- | --- |
| 因果性 | 是，GRU 为单向结构 |
| 感受野 | 64 帧 |
| 离线-在线落差 | 本轮未测 |
| 动作确认延迟 | 本轮未测 |
| 单 tick 延迟 | 本轮未测 |

## 结论

GRU 可以作为因果基线，但当前刷洗召回仍低于 70% 的临时目标。模型对 `Idle` 预测较强，仍会把较多 `Long_Brushing` 和 `Short_Brushing` 帧误判为空闲。

目前使用 `registry/gru-v1/gru-final-20260704-150629.pt` 作为 GRU 代表结果。第二次 10 epoch 训练虽然 Edit 略高，但帧准确率和 F1@0.5 明显下降，不建议作为代表结果。

## 后续工作

- 测量单 tick 延迟，补齐模型卡门禁字段。
- 固定随机种子后重跑，降低 run-to-run 波动。
- 单独测试特征归一化对召回率的影响。
- 新 YOLO 到位后，使用最终特征映射重新生成特征并重训。
