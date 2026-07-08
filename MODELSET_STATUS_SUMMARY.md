# CleanSight 模型集完成状况 Summary

## 1. 总体结论

当前 `Cleansight_models` 已经完成模型资产仓库的主体框架，包括 YOLO 分组训练、三类时序模型、单模型 benchmark、整段喂/流式喂 benchmark、ModelScope 上传目录、使用指南和项目质量规范。

但当前还没有达到生产晋升状态，主要原因是：

- YOLO 两个分组模型均已训练和验证，但验收结果仍为 FAIL。
- 时序模型仍基于旧版 `legacy-20d-v1` 20 维特征。
- 新版 64 维 `feature_mapping.py` 已有规范，但尚未用新 YOLO 输出生成特征并重训时序模型。
- 3 分钟端到端 benchmark 评分器已建立，但真实 Backend prediction JSON 尚未接入。

## 2. 已完成内容

### 2.1 模型集仓库结构

已建立模型集主仓库：

```text
Cleansight_models/
```

当前模型集职责包括：

- 模型训练
- 模型评估
- registry 登记
- 模型卡维护
- benchmark 结果管理
- ModelScope 上传目录整理

线上推理、视频流接入、告警和业务流程仍由相邻的 `CleanSightBackend` 负责。

### 2.2 YOLO 目标检测部分

已建立 YOLO 管理链路：

```text
yolo-detection/
cleansight-yolo-pipeline-main/
yolo-detection/pipeline/
```

当前 YOLO 已按目标特性拆分为两个分组：

| 分组 | 类别 | 说明 |
| --- | --- | --- |
| `group1_large` | hand / scope_control_body / scope_mid_section | 较大目标 |
| `group2_small` | syringe / air_gun / scope_distal_end | 较小目标 |

已完成内容：

- Label Studio 导出接入
- YOLO 数据集构建
- 分组训练
- 分组验证
- 验收报告生成
- registry 目录整理
- ModelScope 上传目录整理

### 2.3 时序模型部分

已建立三套时序模型仓库：

```text
temporal-gru/
temporal-causal-tcn/
temporal-transformer/
```

每个时序模型仓库已包含：

```text
feature_mapping.py
build_testset.py
CARD.md
pin.yaml
REPORT.md
model/
registry/
scripts/
```

三个模型均已完成首轮训练和评估，并登记 registry checkpoint。

### 2.4 Benchmark 部分

当前 benchmark 已分为三层：

```text
benchmark/single_model/
benchmark/temporal_feed_mode/
benchmark/e2e_3min/
```

各层定位：

| Benchmark | 作用 |
| --- | --- |
| `single_model` | 验证单个模型本身效果 |
| `temporal_feed_mode` | 验证整段喂与流式喂的一致性 |
| `e2e_3min` | 验证完整 3 分钟洗消流程 |

这样可以避免把单模型效果、流式推理效果和真实业务流程效果混在一起。

### 2.5 文档和规范

已补充：

```text
MODELSET_USAGE_GUIDE.md
cleansight-yolo-pipeline-main/YOLO_PIPELINE_SUMMARY.md
.codex/skills/modelset-quality/SKILL.md
```

其中 `modelset-quality` skill 用于约束后续开发：

- 明确模型输入输出
- 区分 benchmark 口径
- 维护 `CARD.md` / `pin.yaml`
- 检查 Backend 接入条件
- 要求类和函数具备中文功能说明
- 避免提交大文件、视频、checkpoint 和临时产物

## 3. 当前模型结果

### 3.1 YOLO 结果

当前 YOLO 两个分组均已完成训练和验证，但验收结果仍为 FAIL。

| 分组 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `group1_large` | 0.522 | 0.181 | 0.594 | 0.501 | FAIL |
| `group2_small` | 0.343 | 0.200 | 0.351 | 0.394 | FAIL |

主要问题：

- `group1_large` 中 `scope_control_body` 和 `scope_mid_section` 召回不足。
- `group2_small` 小目标整体效果不足。
- `syringe` 和 `scope_distal_end` 当前验证样本或检出不足，仍无法稳定评估。

### 3.2 时序模型结果

当前三个时序模型均完成首轮训练，但不建议上线。

| 模型 | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GRU | 68.54 | 70.77 | 48.74 | 40.34 | 25.21 | 不晋升 |
| Causal TCN | 69.23 | 44.62 | 46.81 | 40.43 | 27.66 | 不晋升 |
| Transformer | 69.70 | 66.15 | 46.43 | 41.07 | 33.93 | 不晋升 |

逐类召回：

| 模型 | Idle | Long_Brushing | Short_Brushing |
| --- | ---: | ---: | ---: |
| GRU | 88.67% | 34.73% | 42.97% |
| Causal TCN | 90.14% | 32.33% | 48.48% |
| Transformer | 94.98% | 31.43% | 40.71% |

主要问题：

- 刷洗动作召回偏低。
- `Long_Brushing` 和 `Short_Brushing` 未达到稳定上线要求。
- 当前模型仍基于旧版 20 维特征。

## 4. 流式 Benchmark 结论

`temporal_feed_mode` 已完成全量评测，用于比较同一 checkpoint 在整段喂和流式喂下的差异。

| 模型 | Stream Acc | 一致率 | Stream p95 延迟 | 结论 |
| --- | ---: | ---: | ---: | --- |
| GRU | 90.99% | 91.53% | 0.6350 ms | 流式效果最好 |
| TCN | 85.18% | 99.98% | 2.6282 ms | 在线/离线最稳定 |
| Transformer | 90.59% | 70.35% | 2.4183 ms | 对输入方式敏感 |

当前建议：

- GRU 作为在线效果优先基线。
- TCN 作为在线/离线一致性对照基线。
- Transformer 暂不作为首选在线模型，但可继续作为结构对照。

## 5. 当前最大缺口

### 5.1 新 YOLO 特征尚未闭环

当前新版 `feature_mapping.py` 设计为 64 维：

```text
8 个类别 × 每类 8 维 = 64 维
```

每类 8 维为：

```text
present, cx, cy, w, h, conf, dcx, dcy
```

但当前已训练的 GRU / TCN / Transformer v1 checkpoint 仍使用旧版 20 维特征：

```text
feature_mapping_version: legacy-20d-v1
feature_dim: 20
window: 64
```

因此不能直接把新 YOLO 生成的 64 维特征喂给当前 20 维 checkpoint。

### 5.2 端到端真实验收尚未完成

`benchmark/e2e_3min/run_e2e_benchmark.py` 评分器已经存在，但真实端到端验收还缺：

```text
benchmark/e2e_3min/outputs/clean_001.prediction.json
```

该文件需要由 `CleanSightBackend` 在线推理链路导出。

### 5.3 ModelScope 和复刻链路未完全落地

当前已整理本地上传目录，但仍需：

- 上传到 ModelScope
- 回填真实模型地址
- 回填 revision 或 tag
- 完善 `pin.yaml` schema
- 实现一键复刻脚本

## 6. Git 状态提醒

当前工作区还有待整理内容，提交前需要重点检查：

- 新增 `MODELSET_USAGE_GUIDE.md`
- 新增 `benchmark/temporal_feed_mode/`
- 新增 `benchmark/single_model/temporal_summary.md`
- 新增 `benchmark/single_model/temporal_summary.json`
- 修改多个 benchmark 和模型注释文件
- `benchmark/e2e_3min/cases/example.yaml` 当前显示为删除
- `benchmark/e2e_3min/reports/clean_001.md` 当前显示为删除

如果后续还需要 e2e benchmark，应确认 case 文件是否误删，并恢复或重新建立。

## 7. 下一步建议

优先级建议如下：

1. 整理当前 Git 工作区，确认需要提交和需要忽略的文件。
2. 恢复或重新建立 e2e benchmark case 文件。
3. 上传 `modelscope_upload/` 下的 YOLO 和时序模型到 ModelScope。
4. 回填 ModelScope 地址和 revision/tag 到各 `pin.yaml`。
5. 使用新 YOLO 分组模型生成 64 维同源特征。
6. 基于 64 维新特征重训 GRU / Causal TCN / Transformer。
7. 在 `CleanSightBackend` 中导出真实 `clean_001.prediction.json`。
8. 用 e2e benchmark 完成 3 分钟端到端真实验收。
9. 完善 `pin.yaml` schema 和一键复刻脚本。

## 8. 一句话汇报版

当前模型集已经完成基础仓库、YOLO 分组训练、三类时序模型、模型卡、版本钉定、单模型 benchmark、流式一致性 benchmark 和端到端评分器框架；但当前仍处于研发验证阶段，YOLO 和时序模型指标都未达到生产晋升要求。下一步重点是用新 YOLO 生成 64 维同源特征、重训时序模型，并接入 Backend 导出真实 3 分钟端到端 prediction 完成验收。
