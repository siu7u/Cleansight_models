# CleanSight 模型集

本仓库用于管理 CleanSight 清洗刷洗相关模型资产，包括 YOLO 目标检测模型、时序动作识别模型、模型卡、评估报告、版本钉定文件和评估工具。

本仓库不负责线上服务运行。线上推理、视频流接入、告警和可视化由相邻的 `../CleanSightBackend/` 负责。

CleanSightBackend: [查看服务后端](https://github.com/Jiadezhende/CleanSightBackend)


## 项目边界

```text
Cleansight_models/
  负责模型训练、评估、版本登记、模型卡和实验报告

../CleanSightBackend/
  负责在线加载模型、接入视频流、逐帧推理、可视化和告警
```

当前模型集仍处于实验和登记阶段。YOLO 新模型后续给到后，需要重新钉定 YOLO 版本、重新生成特征，并重跑时序模型评估。

## 目录结构

```text
.
├── yolo-detection/              # YOLO 检测模型集中管理仓库
│   ├── data/                    # 数据视图 A 引用说明
│   ├── scripts/                 # YOLO 训练、评估、预测脚本
│   ├── registry/yolo-v1/        # yolo-v1 版本登记
│   ├── templates/               # YOLO 评估报告模板
│   └── docs/PIPELINE.md         # 历史 YOLO -> MS-TCN 流程说明
├── temporal-gru/                # GRU 因果时序模型
├── temporal-causal-tcn/         # Causal TCN 因果时序模型
├── temporal-transformer/        # Transformer 因果时序模型
├── temporal-mstcn-offline/      # MS-TCN++ 离线上限参考
├── tools/                       # 跨模型评估和延迟测试工具
├── references/                  # Label Studio 等参考材料
└── CausalModel-master/          # 原始候选模型代码参考，不作为正式仓库入口
```

每个 `temporal-*` 仓库遵循同一结构：

```text
temporal-<model>/
├── feature_mapping.py           # YOLO 检测到时序特征的接口契约
├── build_testset.py             # 测试集窗口构造
├── pin.yaml                     # 数据、YOLO、特征映射和模型版本钉定
├── CARD.md                      # 模型卡，上线门禁
├── REPORT.md                    # 评估报告
├── model/                       # 模型结构
├── registry/                    # 晋升/登记权重
├── experiments/                 # 可视化和实验输出
└── scripts/                     # 校验脚本
```

## 当前数据与特征

当前三组时序实验使用的是历史 20 维 YOLO 特征：

```text
feature_mapping_version: legacy-20d-v1
feature_dim: 20
window: 64
labels: Idle / Long_Brushing / Short_Brushing
```

输入格式：

```text
原始特征文件：data/Endo_Project/features/*.npy，形状通常为 [F, T]
加载后转置： [T, F]
模型输入：   [B, 64, 20]
模型输出：   [B, 64, 3]
训练监督：   只使用窗口最后一帧 logits
```

当前 `feature_mapping.py` 已提供 64 维规范骨架，但本轮实验结果仍基于 `legacy-20d-v1`。新 YOLO 到位后，需要用最终特征映射重建特征并重训。

## 快速开始

以下命令均从模型集根目录执行。

### 训练 GRU

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
  --visualize \
  --output_dir experiments/gru
```

### 训练 Causal TCN

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

### 训练 Transformer

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

## 详细评估

`tools/eval_temporal_detailed.py` 用于输出逐类召回和混淆矩阵。示例：

```bash
cd temporal-causal-tcn

MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 \
../../CleanSightBackend/.venv/bin/python \
../tools/eval_temporal_detailed.py \
  --repo . \
  --model tcn \
  --checkpoint registry/tcn-v1/tcn-final-20260704-160652.pt
```

注意：该工具使用批量 last-frame logits 直接分类，不包含 `causal_decision` 平滑。

## 延迟测试

`tools/measure_temporal_latency.py` 用于测量单窗口前向延迟。示例：

```bash
cd temporal-transformer

PYTHONDONTWRITEBYTECODE=1 ../../CleanSightBackend/.venv/bin/python \
../tools/measure_temporal_latency.py \
  --repo . \
  --model transformer \
  --checkpoint registry/transformer-v1/transformer-final-20260704-161653.pt \
  --window 64 \
  --input-dim 20
```

延迟结果需要回填到对应 `CARD.md` 的 `单 tick 延迟` 字段。

## 当前实验结果

| 模型 | 权重 | 参数量 | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | `temporal-gru/registry/gru-v1/gru-final-20260704-150629.pt` | 256,131 | 68.54 | 70.77 | 48.74 | 40.34 | 25.21 |
| Causal TCN | `temporal-causal-tcn/registry/tcn-v1/tcn-final-20260704-160652.pt` | 67,587 | 69.23 | 44.62 | 46.81 | 40.43 | 27.66 |
| Transformer | `temporal-transformer/registry/transformer-v1/transformer-final-20260704-161653.pt` | 400,515 | 69.70 | 66.15 | 46.43 | 41.07 | 33.93 |

逐类召回：

| 模型 | Idle | Long_Brushing | Short_Brushing |
| --- | ---: | ---: | ---: |
| GRU | 88.67% | 34.73% | 42.97% |
| Causal TCN | 90.14% | 32.33% | 48.48% |
| Transformer | 94.98% | 31.43% | 40.71% |

当前结论：

- Transformer 的整体 Acc 和 F1@0.5 最高，适合作为当前精度上限参考。
- Causal TCN 参数量最小，适合作为轻量在线候选。
- 三个模型对刷洗动作召回都未达到 70% 临时目标，当前均不建议晋升上线。

## 文档索引

```text
temporal-gru/REPORT.md
temporal-gru/CARD.md
temporal-causal-tcn/REPORT.md
temporal-causal-tcn/CARD.md
temporal-transformer/REPORT.md
temporal-transformer/CARD.md
yolo-detection/data/DATASET_VIEW_A.md
yolo-detection/registry/yolo-v1/eval_report.md
yolo-detection/templates/eval_report_template.md
```

## YOLO 管理状态

`yolo-detection/` 已建立集中式 YOLO 管理骨架，并登记了 `yolo-v1` 的配置和报告模板。

当前 YOLO 评估仍未完成：

- `mAP@0.5`：TODO
- `mAP@0.5:0.95`：TODO
- 逐类 AP / Recall：TODO
- 小目标召回：TODO
- 单帧延迟：TODO

新 YOLO 模型给到后，需要补齐 `yolo-detection/registry/yolo-v1/eval_report.md` 和 `metrics.json`，再更新各时序模型的 `pin.yaml`。

## 上线前门禁

任何时序模型进入 CleanSightBackend 前，必须至少补齐：

- 参数量
- 因果性与感受野
- 单 tick 延迟
- 逐类召回，尤其是 `Long_Brushing` 和 `Short_Brushing`
- 离线-在线落差
- 与 YOLO 版本和特征映射版本的钉定关系

当前三模型的 `CARD.md` 已记录参数量、因果性和评估指标，但单 tick 延迟与离线-在线落差仍待测。

## 与 CleanSightBackend 的接入关系

模型训练和评估在本仓库完成。真正在线推理应接入：

```text
../CleanSightBackend/app/services/inference/workflows/
../CleanSightBackend/config/inference_config.yaml
```

上线时建议将晋升模型整理成 bundle：

```text
model-bundle/
├── temporal.pt
├── feature_mapping.py
├── pin.yaml
├── CARD.md
└── REPORT.md
```

然后由 CleanSightBackend 的推理 workflow 加载该 bundle。

## 后续 TODO

- 等待新 YOLO 模型，补齐检测评估报告。
- 用最终 YOLO + 标准 `feature_mapping.py` 重新生成特征。
- 重跑 GRU / Causal TCN / Transformer。
- 测量三个模型的单 tick 延迟。
- 增加窗口大小实验：32 / 64 / 96 / 128。
- 优化刷洗召回，重点关注 `Long_Brushing` 和 `Short_Brushing`。
- 建立统一 benchmark 输出，减少手工整理报告。
