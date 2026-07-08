# CleanSight 模型集汇报提纲

## 1. 汇报主线

本次汇报可以围绕四个问题展开：

```text
为什么要建立模型集？
目前模型集已经完成了什么？
当前实验结果说明了什么？
下一步还需要做什么？
```

一句话概括：

```text
当前模型集已经从零散模型整理成了一个可训练、可评估、可登记、可复现的模型资产仓库。现阶段重点不是直接上线，而是先打通 YOLO、时序模型和端到端 benchmark 的完整链路，并基于新 YOLO 特征完成下一轮可晋升模型训练。
```

## 2. 建立模型集的目的

模型集的目标是把模型相关工作从业务后端中拆出来，形成独立的模型资产仓库。

模型集负责：

```text
模型训练
模型评估
checkpoint 管理
模型卡 CARD.md
版本钉定 pin.yaml
benchmark 报告
ModelScope 上传准备
```

`CleanSightBackend` 负责：

```text
在线视频流接入
在线推理
告警
可视化
端到端业务流程
```

这样做的好处是：

- 模型训练和线上服务职责更清楚。
- 后续换模型、重训模型、上传模型更容易。
- 模型版本、数据版本和评测报告可以统一管理。
- 便于后续复现和对比不同模型方案。

## 3. 当前已经完成的内容

### 3.1 YOLO 目标检测部分

当前已经建立 YOLO pipeline，完成了从数据到验证报告的基础闭环：

```text
Label Studio 导出
        ↓
抽帧并转换 YOLO 数据集
        ↓
按目标分组训练 YOLO
        ↓
验证集推理
        ↓
生成验收报告
```

YOLO 当前按目标特性拆为两个分组：

| 分组 | 类别 | 说明 |
| --- | --- | --- |
| `group1_large` | hand / scope_control_body / scope_mid_section | 较大目标 |
| `group2_small` | syringe / air_gun / scope_distal_end | 较小目标 |

可以汇报为：

```text
YOLO 检测模型已经完成分组训练和验证流程，但当前两个分组还没有达到验收标准。
```

### 3.2 时序模型部分

当前已经建立三套时序模型仓库：

```text
temporal-gru/
temporal-causal-tcn/
temporal-transformer/
```

每个仓库都包含：

```text
feature_mapping.py
build_testset.py
CARD.md
pin.yaml
REPORT.md
model/
registry/
```

可以汇报为：

```text
我们已经完成 GRU、Causal TCN、Transformer 三类候选时序模型模板，并完成首轮训练和评估。
```

### 3.3 Benchmark 部分

当前 benchmark 分为三层：

| Benchmark | 作用 |
| --- | --- |
| `single_model` | 验证单个模型效果 |
| `temporal_feed_mode` | 验证整段喂和流式喂的差异 |
| `e2e_3min` | 验证完整 3 分钟洗消流程 |

这个分层的意义是：

```text
单模型效果好，不代表完整流程可用；
离线整段评测好，也不代表在线流式推理稳定。
```

因此需要把单模型、在线流式和端到端流程分开评测。

## 4. 当前实验结果

### 4.1 YOLO 结果

当前 YOLO 两个分组都能训练和验证，但验收结果仍为 FAIL。

| 分组 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `group1_large` | 0.522 | 0.181 | 0.594 | 0.501 | FAIL |
| `group2_small` | 0.343 | 0.200 | 0.351 | 0.394 | FAIL |

可以这样说明：

```text
group1_large 的 mAP@0.5 已经达到 0.522，但 mAP@0.5:0.95 和部分类别召回不足；
group2_small 的小目标检测还不稳定，尤其 syringe 和 scope_distal_end 的验证样本或检出不足。
```

结论：

```text
YOLO 当前流程已经跑通，但还不能作为生产版本晋升。
```

### 4.2 时序模型结果

三个时序模型都完成了首轮训练和评估：

| 模型 | Acc | Edit | F1@0.5 | 结论 |
| --- | ---: | ---: | ---: | --- |
| GRU | 68.54 | 70.77 | 25.21 | 不晋升 |
| Causal TCN | 69.23 | 44.62 | 27.66 | 不晋升 |
| Transformer | 69.70 | 66.15 | 33.93 | 不晋升 |

主要问题：

```text
Long_Brushing 和 Short_Brushing 的召回仍然偏低，当前还达不到稳定上线要求。
```

需要特别说明：

```text
当前三个时序模型仍然基于旧版 20 维特征，不是新 YOLO 生成的 64 维特征。
```

## 5. 流式 Benchmark 的重点发现

`temporal_feed_mode` benchmark 比较了同一 checkpoint 在两种输入方式下的差异：

```text
整段喂：一次输入完整序列 [1, T, F]
流式喂：每 tick 输入最近 window 帧 [1, window, F]
```

当前结果：

| 模型 | Stream Acc | 一致率 | 结论 |
| --- | ---: | ---: | --- |
| GRU | 90.99% | 91.53% | 流式效果最好 |
| TCN | 85.18% | 99.98% | 在线/离线一致性最好 |
| Transformer | 90.59% | 70.35% | 对输入方式敏感 |

可以汇报为：

```text
GRU 的流式准确率最高，适合作为在线效果优先基线；
TCN 的整段喂和流式喂一致性最高，适合作为在线/离线一致性对照基线；
Transformer 在流式模式下效果还可以，但整段喂和流式喂差异较大，暂时不作为首选在线模型。
```

## 6. 端到端 Benchmark 状态

当前 3 分钟端到端 benchmark 的评分器已经完成。

设计流程：

```text
定义 3 分钟标准 case
        ↓
Backend 跑真实视频或视频流
        ↓
导出 prediction JSON
        ↓
benchmark 对比 expected 和 prediction
        ↓
输出 PASS / FAIL / PENDING
```

评分器会检查：

- 最终 result 是否一致。
- 关键动作是否检出。
- 动作阶段起止时间误差是否在允许范围内。

当前状态：

```text
评分器已经具备，但真实 Backend 导出的 prediction JSON 还没有接入。
```

因此当前 e2e benchmark 仍处于：

```text
评分器完成，真实在线验收待接入。
```

## 7. 质量规范和文档

为了方便后续组员维护，已经补充：

```text
MODELSET_USAGE_GUIDE.md
MODELSET_STATUS_SUMMARY.md
cleansight-yolo-pipeline-main/YOLO_PIPELINE_SUMMARY.md
.codex/skills/modelset-quality/SKILL.md
```

`modelset-quality` 主要约束：

- 每个模型要写清 `input_dim`、`window`、`label mapping`、`checkpoint`。
- benchmark 要区分 smoke test 和正式结果。
- 类和函数要有中文注释说明。
- 模型版本要配 `CARD.md`、`pin.yaml` 和评测报告。
- 接入 Backend 前要验证路径、维度、类别映射和在线 smoke test。

## 8. 当前未完成内容

当前还没有完成生产晋升，主要缺口包括：

1. YOLO 指标还没有达标，尤其小目标和部分镜体类别召回不足。
2. 当前时序模型仍然基于旧版 20 维特征。
3. 新版 `feature_mapping.py` 设计为 64 维，但还没有用新 YOLO 输出重训时序模型。
4. 端到端 benchmark 还缺 Backend 导出的真实 `prediction JSON`。
5. ModelScope 上传和 `pin.yaml` 一键复刻链路还需要继续完善。

## 9. 下一步计划

建议按以下顺序推进：

```text
1. 整理当前仓库和 Git 状态，保证文档、脚本、benchmark 结果可提交。
2. 继续补充 YOLO 数据，重点提升小目标和低召回类别。
3. 用新 YOLO 输出生成 64 维同源特征。
4. 基于新特征重新训练 GRU、TCN、Transformer。
5. 将优先模型接入 CleanSightBackend。
6. 从 Backend 导出真实 3 分钟 prediction JSON。
7. 用 e2e benchmark 做完整流程验收。
8. 完善 ModelScope 上传、pin.yaml schema 和一键复刻脚本。
```

## 10. 可直接口头收尾

```text
目前模型集已经完成从模型训练、评估、版本登记到 benchmark 的基本闭环。当前阶段的重点不是直接上线，而是把 YOLO 检测、时序识别和端到端流程验证的链路打通。下一步会用新 YOLO 输出生成统一的 64 维特征，重训时序模型，并接入 Backend 做真实 3 分钟端到端验收。
```
