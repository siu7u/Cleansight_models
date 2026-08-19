# 模型卡：auto-transformer-v1（自动标注数据通道）

## 版本钉定

- 模型版本：v1
- 模型类型：Transformer Encoder（全序列，非因果）
- 权重：`runs/transformer-20260817-170535/checkpoints/best.pt`
  （sha256: `b0fa7f241200f658dfa4...`，完整值见 `pin.yaml`）
- 数据视图：`temporal.actionmixed-auto-v1`（revision `9dd8fb79...`）
- 特征映射版本：`actionmixed-bbox-8cls-v1`（40 维，8 检测类 × 5）
- 标签：idle / air_injection / flush / long_brush_insert / long_brush_withdraw / short_brush_cleaning

## 上线门禁

- 因果性：否（全序列离线推理）
- 输入形状：`[1, T, 40]`，`input_dim=40`，`num_classes=6`，`max_len=2560`
- 参数量：见 run `config.resolved.json` 与 checkpoint meta
- 训练配置：`framework/experiments/transformer-actionmixed-auto.yaml`（30 epoch）

## 评估结果（formal，2026-08-17）

- 训练期验证集 best val_acc：16.18（epoch 1，之后过拟合）
- test split（98 帧、仅 long_brush_withdraw）：acc 0.0 / edit 0.0
  —— 见 `docs/AUTO_ANNOTATION.md`「正式训练结果」：0.0 是 test 集单类弱势类别的
  真实结果，不代表链路问题
- 评测报告：`runs/transformer-20260817-170535/evals/*.evaluation.json`、
  `runs/transformer-20260817-170535/checkpoints/EVALUATION_REPORT.md`

## 已知限制

- 检测特征仅覆盖 6 类（short_brush / brush_tip_out 恒零），40 维 v1 布局兼容
- 7 视频小数据下 Transformer 首轮达峰即过拟合；test 集 98 帧单类样本意义有限
- 本通道为新增实验通道，与手动标注 `temporal.actionmixed-v2` 并存，不替代

## 同视频消融参照（2026-08-17）

同一批 9 个视频、同一批公共帧、同一套动作标签（project-10），仅 bbox 特征来源不同
（人工框 vs YOLO 框）时，val_acc 对比：

| 模型 | 人工框 | YOLO 框 | 差距 |
|---|---:|---:|---:|
| MS-TCN | 78.51 | 57.74 | -20.8 |
| GRU | 73.94 | 65.15 | -8.8 |
| Transformer | 65.47 | 18.48 | -47.0 |

结论：人工标注框特征显著优于 YOLO 自动框特征，是自动标注通道当前的主要代价来源。
