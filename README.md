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

## 当前任务状态

详细进度见 `TASK_STATUS.md`。

当前已完成基础模型集搭建、三套时序模型模板、YOLO 分组训练流水线、首轮 YOLO 训练验证、registry 登记、benchmark 骨架和 ModelScope 上传目录整理。

当前仍未达到生产晋升状态：

- YOLO 两个分组模型均已训练和验证，但验收结果为 FAIL。
- 三个时序模型均已训练和评估，但刷洗类召回未达到临时目标。
- 时序模型仍基于 `legacy-20d-v1` 历史 20 维特征，尚未用新 YOLO 分组特征重训。
- 3 分钟端到端 benchmark 的评分器已跑通，但真实后端 prediction JSON 仍需由 CleanSightBackend 导出。

## 目录结构

```text
.
├── TASK_STATUS.md               # 当前任务完成情况和剩余 TODO
├── yolo-detection/              # YOLO 检测模型集中管理仓库
│   ├── data/                    # 数据视图 A 引用说明
│   ├── pipeline/                # 当前 YOLO 分组训练流水线
│   ├── registry/                # yolo-group*-v1 版本登记
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
TASK_STATUS.md
temporal-gru/REPORT.md
temporal-gru/CARD.md
temporal-causal-tcn/REPORT.md
temporal-causal-tcn/CARD.md
temporal-transformer/REPORT.md
temporal-transformer/CARD.md
yolo-detection/data/DATASET_VIEW_A.md
yolo-detection/registry/yolo-group1-large-v1/eval_report.md
yolo-detection/registry/yolo-group2-small-v1/eval_report.md
yolo-detection/templates/eval_report_template.md
```

## YOLO 管理状态

`yolo-detection/` 已建立集中式 YOLO 管理目录，并接入 `yolo-detection/pipeline/` 作为当前训练流水线。

当前 YOLO 已按目标特性拆为两个并行检测分组：

| 版本 | 分组 | 类别 | 权重 | 结论 |
| --- | --- | --- | --- | --- |
| `yolo-group1-large-v1` | `group1_large` | hand / scope_control_body / scope_mid_section | `yolo-detection/pipeline/runs/group1_large/weights/best.pt` | FAIL |
| `yolo-group2-small-v1` | `group2_small` | syringe / air_gun / scope_distal_end | `yolo-detection/pipeline/runs/group2_small/weights/best.pt` | FAIL |

当前 YOLO 验收结果：

| 分组 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 主要问题 |
| --- | ---: | ---: | ---: | ---: | --- |
| `group1_large` | 0.522 | 0.181 | 0.594 | 0.501 | `scope_control_body` 和 `scope_mid_section` 召回不足 |
| `group2_small` | 0.343 | 0.200 | 0.351 | 0.394 | 小目标整体不足，`syringe` / `scope_distal_end` 暂无法评估 |

报告位置：

- `yolo-detection/registry/yolo-group1-large-v1/eval_report.md`
- `yolo-detection/registry/yolo-group2-small-v1/eval_report.md`
- `benchmark/single_model/latest/yolo_summary.md`

权重文件不进入 git，已整理到 `modelscope_upload/` 供上传 ModelScope。

## 上线前门禁

任何时序模型进入 CleanSightBackend 前，必须至少补齐：

- 参数量
- 因果性与感受野
- 单 tick 延迟
- 逐类召回，尤其是 `Long_Brushing` 和 `Short_Brushing`
- 离线-在线落差
- 与 YOLO 版本和特征映射版本的钉定关系

当前三模型的 `CARD.md` 已记录参数量、因果性和评估指标，但单 tick 延迟与离线-在线落差仍待测。

CARD 上线门禁脚本：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/validate_card_gate.py temporal-gru/CARD.md
PYTHONDONTWRITEBYTECODE=1 python3 tools/validate_card_gate.py temporal-causal-tcn/CARD.md
PYTHONDONTWRITEBYTECODE=1 python3 tools/validate_card_gate.py temporal-transformer/CARD.md
```

任何模型上线前 `CARD.md` 必须补齐部署机实测运行延迟、感受域/因果性和模型参数量。字段为 `待测`、`TODO` 或非因果时，门禁失败；MS-TCN 这类非因果模型只能作为离线上限参考，不进入在线部署。

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

- 上传 `modelscope_upload/` 下的 YOLO 与时序权重到 ModelScope，并回填真实地址、revision 或 tag。
- 补充或重切小目标验证集，解决 `syringe` / `scope_distal_end` 无法评估的问题。
- 提升 YOLO 弱项类别召回，尤其是 `scope_control_body`、`scope_mid_section`、`air_gun`。
- 用新 YOLO 分组模型生成最终特征，并让离线训练特征与在线推理特征共用同一个 `step()`。
- 基于新特征重训 GRU / Causal TCN / Transformer。
- 测量三个时序模型的单 tick 延迟，并写回 `CARD.md`。
- 在 CleanSightBackend 中导出真实 `clean_001.prediction.json`，完成 3 分钟端到端真实验收。
- 完善 `pin.yaml` schema 和一键复刻脚本，支持按版本拉齐 dataset / YOLO / temporal model / feature_mapping。
