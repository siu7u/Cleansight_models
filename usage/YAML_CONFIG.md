# YAML 配置文档

本目录用于集中说明仓库内 YAML 文件的内容和功能，并提供快速定位链接。业务代码仍从各文件的
原路径读取配置；本文档只做索引，不复制配置，也不是第二份配置真源。

## 文档范围与维护规则

- 本文档逐一覆盖 `git ls-files '*.yaml' '*.yml'` 返回的所有 YAML。
- 新增、修改、移动、重命名或删除受 Git 跟踪的 YAML 时，必须同步更新本文档。
- 字段、默认值、约束、读取方或运行影响变化时，即使文件路径不变，也要更新对应说明。
- 被 `.gitignore` 排除的构建、训练和打包产物不纳入逐文件清单，统一在文末说明。

当前共收录 22 个受 Git 跟踪的 YAML。

## 1. Framework 实验配置

这些配置由 `framework.cleansight_eval.cli.train` 和 `framework.cleansight_eval.cli.eval` 读取，
`framework/cleansight_eval/core/config.py` 负责加载、默认值、相对路径解析和配置溯源，再由对应
Pipeline 校验并执行。

共同内容：

| 字段 | 内容与功能 |
|---|---|
| `schema_version` | 实验配置契约版本，当前为 `1`。 |
| `pipeline` | 选择检测、滑窗时序或全序列时序流程，确定训练和推理语义。 |
| `model` | 模型类型、输入/输出维度、网络规模、初始权重及 metadata 策略。 |
| `data` | `dataset_ref` 引用 benchmark catalog；catalog 解析数据根、类别和 manifest，实验只声明 train/val/eval split。没有引用的临时/合成配置仍可直接使用 `root`。 |
| `feature_schema` | 时序特征维度、mapping 版本、类别布局及可选固定目标遮罩。 |
| `augmentation` | 训练期数据增强；`target_mask` 只作用于 train，不作用于 val/test。 |
| `evaluation` | 正式/探索模式、预测保存、延迟及检测阈值；时序 testset 由 `dataset_ref + split_eval` 唯一推导，也可显式写 `testset_id` 做一致性断言。两条时序 Pipeline 默认启用测试 timeline。 |
| `train` | epoch、学习率、batch/window、早停、梯度裁剪和 resume 等参数。 |

| YAML | 主要内容 | 功能 |
|---|---|---|
| [`framework/experiments/gru-actionmixed.yaml`](../framework/experiments/gru-actionmixed.yaml) | GRU、`temporal.actionmixed-v2`、40 维 bbox 特征、6 类、16 帧窗口；含默认关闭的随机目标遮罩 | 滑窗、末帧监督、因果推理的时序参照实验；测试默认输出 GT/Prediction timeline。 |
| [`framework/experiments/mstcn-actionmixed.yaml`](../framework/experiments/mstcn-actionmixed.yaml) | MS-TCN、40 维输入、6 类和单 stage baseline 参数 | 全序列、逐帧监督的离线时序参照实验。 |
| [`framework/experiments/mstcn2-actionmixed.yaml`](../framework/experiments/mstcn2-actionmixed.yaml) | MS-TCN++ 的 stage、layer、dropout 和 T-MSE 参数 | 全序列多 stage 精化实验。 |
| [`framework/experiments/transformer-actionmixed.yaml`](../framework/experiments/transformer-actionmixed.yaml) | Transformer 的表示维度、head、layer、FFN 和最大长度 | 使用完整上下文的非因果全序列实验。 |
| [`framework/experiments/yolo-clean-large.yaml`](../framework/experiments/yolo-clean-large.yaml) | group1 大目标、YOLO11n、640 输入和 val testset | 大目标检测训练及探索性评估。 |
| [`framework/experiments/yolo-clean-small.yaml`](../framework/experiments/yolo-clean-small.yaml) | group2 小目标、YOLO11n、640 输入和 val testset | 小目标检测训练及探索性评估，重点观察逐类召回。 |
| [`framework/experiments/yolo-group1.yaml`](../framework/experiments/yolo-group1.yaml) | group1 单帧检测参照参数和外部权重兼容选项 | group1 检测的带注释参照实验。 |

## 2. Benchmark 数据集和 split

[`benchmark/testsets.yaml`](../benchmark/testsets.yaml) 是正式评估数据身份与 split 契约的真源，
由 `benchmark/core/testsets.py`、`tools/validate_testsets.py` 和 framework 评估溯源逻辑读取。

| 字段 | 内容与功能 |
|---|---|
| `schema_version` / `root` | 定义清单版本和相对路径解析根。 |
| `datasets` | 数据集级公共事实：family、版本、数据根或 manifest、feature mapping、维度和 labels。 |
| `revision` | 外部数据仓库的固定 revision；ActionMixed v2 当前钉定为完整 9,532 帧数据对应的 Git commit。 |
| `split_overlap_policy` | `error` 禁止同源跨 split；`frame` 允许同源但禁止具体帧重合；`allow` 关闭重叠门禁。 |
| `testsets` | split 身份、manifest、用途和可选预期样本。 |
| `purpose` | 区分训练、训练期验证、开发 benchmark、锁定 holdout 和 schema smoke。 |

当前内容登记 ActionMixed 时序 train/val/test、旧 Endo Project train/test、两组 YOLO val/test 和
一个端到端 smoke case。评估时据此记录数据集版本、split、重叠策略和 fingerprint。

ActionMixed v2 的 manifest 是训练与评测 loader 的唯一样本真源，并必须与对应
`labels/<split>` 目录严格一致。fingerprint 同时覆盖 manifest、动作标签、类别映射和逐帧 bbox
文本；新增样本、改动作标签或改检测框都会产生新 fingerprint。训练 checkpoint 记录数据集版本、
revision、train/val/eval split fingerprint 以及动作/检测映射摘要，resume 时 train fingerprint
不一致会被拒绝。

## 3. 端到端三分钟用例

这些文件由 `benchmark/e2e_3min/run_e2e_benchmark.py` 读取，也可被 testset manifest 引用。

共同内容：`case_id` 是稳定身份，`source_task_id` 记录标注来源，`video` 和 `duration_sec` 描述输入；
`expected.result` 表示整体结果，`required_actions` 用于动作检出检查，`phases` 保存真值起止时间并
计算时间线 IoU/precision/recall/F1，`allowed_time_error_sec` 控制边界误差容限。

| YAML | 内容与功能 |
|---|---|
| [`benchmark/e2e_3min/cases/example.yaml`](../benchmark/e2e_3min/cases/example.yaml) | 180 秒最小示例，用于 schema smoke，不代表正式 benchmark。 |
| [`benchmark/e2e_3min/cases/ed1f1353-clip_1781659288372_1781659325362.yaml`](../benchmark/e2e_3min/cases/ed1f1353-clip_1781659288372_1781659325362.yaml) | 37 秒短刷清洁片段的期望时间线。 |
| [`benchmark/e2e_3min/cases/4807dbbe-clip_1781659328328_1781659467929.yaml`](../benchmark/e2e_3min/cases/4807dbbe-clip_1781659328328_1781659467929.yaml) | 约 140 秒长刷插入/退出及注气阶段的期望时间线。 |
| [`benchmark/e2e_3min/cases/65d70028-clip_1781661552468_1781661702909.yaml`](../benchmark/e2e_3min/cases/65d70028-clip_1781661552468_1781661702909.yaml) | 约 150 秒长刷及多段冲洗阶段的期望时间线。 |

## 4. 旧模型统一目录

[`model_manager/models.yaml`](../model_manager/models.yaml) 由 `model_manager/catalog.py` 读取：

- `profiles` 抽取 YOLO 和 legacy 20D 时序模型的公共输入、输出及命令；
- `models` 登记模型 ID、工作目录、checkpoint、标签和 testset；
- `benchmarks` 保留旧专项 benchmark 命令。

它是旧训练/评估入口的兼容目录，不替代 framework 实验配置。

## 5. 时序模型版本 pin

三个 pin 都记录模型版本、数据来源、上游 YOLO、feature mapping、在线因果属性、感受野和输出
标签。各模型的 `scripts/validate_pin.py` 校验必需字段，模型管理及交付资料引用这些文件；它们
用于复现和部署追溯，不控制 framework 训练循环。

| YAML | 内容与功能 |
|---|---|
| [`temporal-gru/pin.yaml`](../temporal-gru/pin.yaml) | 固定 GRU v1 的 legacy-20d 输入、64 帧窗口和三类输出契约。 |
| [`temporal-causal-tcn/pin.yaml`](../temporal-causal-tcn/pin.yaml) | 固定 causal TCN v1 的数据、特征和在线契约。 |
| [`temporal-transformer/pin.yaml`](../temporal-transformer/pin.yaml) | 固定旧 Transformer v1 的数据、特征和运行契约。 |

其中为 `TODO` 的 repo、revision、hash 或 fps 表示版本尚未完全钉定，正式交付前需要补齐。

## 6. YOLO 数据构建流水线

| YAML | 读取方 | 内容与功能 |
|---|---|---|
| [`yolo-detection/pipeline/config.yaml`](../yolo-detection/pipeline/config.yaml) | `pipeline/utils/common.py` 及拉取、状态、构建、训练、验证脚本 | 中央配置：两组类别顺序、质检视频白名单、抽帧、未来样本切分参数、训练超参和验收阈值。类别只能末尾追加，以保护已有 class ID。 |
| [`yolo-detection/pipeline/splits.yaml`](../yolo-detection/pipeline/splits.yaml) | `pipeline/utils/split.py`、`00_status.py`、`02_build_dataset.py` | 视频 stem 到 train/val/test/e2e_test 的稳定分配真源；已有 assignment 不自动重排。 |

`config.yaml` 决定新视频如何处理，`splits.yaml` 保存已经发生的分配；其中 `val_ratio` 和 `seed`
需要保持一致。

## 7. YOLO registry 元数据

这些文件供评估报告、CARD/打包流程及人工发布检查使用，不是 YOLO 运行时训练配置。

| YAML | 内容与功能 |
|---|---|
| [`yolo-detection/registry/yolo-group1-large-v1/classes.yaml`](../yolo-detection/registry/yolo-group1-large-v1/classes.yaml) | 固定 group1 checkpoint 的 class ID：hand、scope control body、scope mid section。 |
| [`yolo-detection/registry/yolo-group1-large-v1/train_config.yaml`](../yolo-detection/registry/yolo-group1-large-v1/train_config.yaml) | 记录 group1 的架构、图像尺寸、训练超参、stride 和验收门槛。 |
| [`yolo-detection/registry/yolo-group2-small-v1/classes.yaml`](../yolo-detection/registry/yolo-group2-small-v1/classes.yaml) | 固定 group2 checkpoint 的 class ID：syringe、air gun、scope distal end。 |
| [`yolo-detection/registry/yolo-group2-small-v1/train_config.yaml`](../yolo-detection/registry/yolo-group2-small-v1/train_config.yaml) | 记录 group2 的训练事实和验收门槛；小目标重点审查 recall。 |

## 8. 生成或本地 YAML

以下文件受 `.gitignore` 排除，不进入上面的 22 文件清单：

- `yolo-detection/pipeline/datasets/**/data.yaml`：数据构建产生的 Ultralytics 清单；真源是 pipeline
  配置、稳定 split 和已导入数据。
- `yolo-detection/pipeline/raw/**/data.yaml`：外部数据随附的类别或动作映射；experiment/testset
  保存其路径和 mapping 版本。
- `**/runs/**/args.yaml`、`yolo-detection/experiments/**/args.yaml`：训练过程生成的参数快照。
- `modelscope_upload/**/*.yaml`：打包输出副本，应由受跟踪的 experiment、pin 或 registry 重新生成。
- `cleansight-yolo-pipeline-main/**/*.yaml`：本地兼容镜像；可维护真源位于
  `yolo-detection/pipeline/` 和 `yolo-detection/registry/`。

若生成文件以后转为受 Git 跟踪的稳定契约，必须将其加入本文档的逐文件清单。

## 更新检查

1. 将本文档清单与 `git ls-files '*.yaml' '*.yml'` 对照，确保无遗漏和失效链接。
2. 更新对应文件的主要内容、读取方、功能、默认值或不变量及运行影响。
3. 检查 schema、路径、类别顺序、feature mapping、split、checkpoint 和数据版本是否同步。
4. 运行相关 validator 或测试，避免把生成 YAML 误当成配置真源提交。
