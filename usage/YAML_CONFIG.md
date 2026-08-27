# YAML 配置文档

本目录用于集中说明仓库内 YAML 文件的内容和功能，并提供快速定位链接。业务代码仍从各文件的
原路径读取配置；本文档只做索引，不复制配置，也不是第二份配置真源。

## 文档范围与维护规则

- 本文档逐一覆盖 `git ls-files '*.yaml' '*.yml'` 返回的所有 YAML。
- 新增、修改、移动、重命名或删除受 Git 跟踪的 YAML 时，必须同步更新本文档。
- 字段、默认值、约束、读取方或运行影响变化时，即使文件路径不变，也要更新对应说明。
- 被 `.gitignore` 排除的构建、训练和打包产物不纳入逐文件清单，统一在文末说明。

当前共登记 33 个 YAML，其中包含 4 个与本地外部 checkpoint 配套的探索性配置、3 个历史
时序 checkpoint 的 framework 兼容实验、1 个组员接入外部时序权重时使用的模板和 1 个
YOLO 自动标注配置。

## 1. Framework 实验配置

这些配置由 `framework.cleansight_eval.cli.train` 和 `benchmark.cli.eval` 读取，
`framework/cleansight_eval/core/config.py` 负责加载、默认值、相对路径解析和配置溯源，再由对应
Pipeline 校验并执行。

共同内容：

| 字段 | 内容与功能 |
|---|---|
| `schema_version` | 实验配置契约版本，当前为 `1`。 |
| `pipeline` | 选择检测、滑窗时序或全序列时序流程，确定训练和推理语义。 |
| `model` | 模型类型、输入/输出维度、网络规模、初始权重及 metadata 策略。`allow_missing_meta: true` 只在 `exploratory` 生效：YOLO 由自身格式加载，时序模型按本段结构严格加载裸 state dict、常见包装、受限 NumPy metadata 包装或由 JIT API 提取参数的 TorchScript；`formal` 禁止该降级。 |
| `data` | `dataset_ref` 引用 benchmark catalog；catalog 解析数据根、类别和 manifest，实验只声明 train/val/eval split。没有引用的临时/合成配置仍可直接使用 `root`；CLEAN v2 recipe 用 `fps` 计算速度特征。 |
| `feature_schema` | 时序特征维度、mapping 版本、类别布局及可选固定目标遮罩。`class_order` 可把数据动作 ID 重排到 checkpoint 输出顺序；五列 bbox 缺 confidence 时，`detection_confidence_default` 只允许作为 exploratory 的显式替代值。 |
| `augmentation` | 训练期数据增强；`target_mask` 只作用于 train，不作用于 val/test。 |
| `evaluation` | 正式/探索模式、预测保存、延迟及检测阈值；时序 testset 由 `dataset_ref + split_eval` 唯一推导，也可显式写 `testset_id` 做一致性断言。两条时序 Pipeline 默认启用测试 timeline。 |
| `train` | epoch、学习率、batch/window、早停、梯度裁剪和 resume 等参数。 |

| YAML | 主要内容 | 功能 |
|---|---|---|
| [`framework/experiments/gru-actionmixed.yaml`](../framework/experiments/gru-actionmixed.yaml) | GRU、`temporal.actionmixed-v2`、40 维 bbox 特征、6 类、16 帧窗口；含默认关闭的随机目标遮罩 | 滑窗、末帧监督、因果推理的时序参照实验；测试默认输出 GT/Prediction timeline。 |
| [`framework/experiments/legacy-gru-v1.yaml`](../framework/experiments/legacy-gru-v1.yaml) | 历史 20 维 GRU、3 类、64 帧因果窗口及 `temporal.endo-project-v1` | 通过 framework 严格加载 registry 中 GRU v1 裸 state dict；无绑定 metadata，因此只允许 exploratory。 |
| [`framework/experiments/legacy-causal-tcn-v1.yaml`](../framework/experiments/legacy-causal-tcn-v1.yaml) | 历史三层 64 通道 Causal TCN 的精确结构和 20 维输入契约 | 供统一单模型与 feed-mode benchmark 调用 framework，不再动态导入历史目录。 |
| [`framework/experiments/legacy-causal-transformer-v1.yaml`](../framework/experiments/legacy-causal-transformer-v1.yaml) | 历史 128 维、3 层、4 head 因果 Transformer 的精确结构 | 与当前非因果 `transformer` 区分，保持旧 checkpoint 参数键和在线窗口语义。 |
| [`external_checkpoints/external-temporal-template.yaml`](../external_checkpoints/external-temporal-template.yaml) | 外部时序 checkpoint 的复制模板；用 `[选择]`、`[确认]`、`[自动]` 注释区分可选枚举、必须回查训练事实和 catalog 自动字段，并列出 Pipeline、注册模型、数据来源、feature mapping/维度、归一化及窗口候选 | 供组员为裸 `.pt` 建立配套 exploratory 配置；数据来源必须在 `dataset_ref` 与 `name + root` 中二选一，`REPLACE_WITH_*`/`0` 必须替换，注释示例不是通用默认值，YAML 也不能代替未实现的模型或特征代码。 |
| [`external_checkpoints/gru-v0.4.0/gru-v0.4.0.yaml`](../external_checkpoints/gru-v0.4.0/gru-v0.4.0.yaml) | 外部 TorchScript GRU v0.4.0、48 维输入、64 隐层、3 层和6类；启用无 metadata 的探索性加载 | 与同目录 `.pt` 配套保存已确认的网络结构；48维 feature mapping、类别顺序和训练窗口未确认前，不得作为正式评测配置。 |
| [`external_checkpoints/asformer-offline/best_asformer_offline_segmenter.yaml`](../external_checkpoints/asformer-offline/best_asformer_offline_segmenter.yaml) | 后端 CLEAN ASFormer、121维 v2+业务先验、checkpoint z-score 和外部类别顺序 | 配套评测同目录 best checkpoint；五列 bbox 的 confidence=1.0 替代使其仅为 exploratory。 |
| [`external_checkpoints/bigru-offline/best_bigru_offline_segmenter.yaml`](../external_checkpoints/bigru-offline/best_bigru_offline_segmenter.yaml) | 后端 CLEAN BiGRU、249维居中窗口统计+业务先验、完整序列离线推理 | 配套评测同目录 best checkpoint；模型与 centered feature 都是非因果。 |
| [`external_checkpoints/mstcn-bilstm-offline/best_ms_tcn_offline_segmenter.yaml`](../external_checkpoints/mstcn-bilstm-offline/best_ms_tcn_offline_segmenter.yaml) | 后端 CLEAN BiLSTM+MS-TCN、113维基础 v2、两级 refine | 配套评测同目录 best checkpoint；与 framework 原有简化 `mstcn` 是不同模型类型。 |
| [`framework/experiments/mstcn-actionmixed.yaml`](../framework/experiments/mstcn-actionmixed.yaml) | MS-TCN、40 维输入、6 类和单 stage baseline 参数 | 全序列、逐帧监督的离线时序参照实验。 |
| [`framework/experiments/mstcn2-actionmixed.yaml`](../framework/experiments/mstcn2-actionmixed.yaml) | MS-TCN++ 的 stage、layer、dropout 和 T-MSE 参数 | 全序列多 stage 精化实验。 |
| [`framework/experiments/transformer-actionmixed.yaml`](../framework/experiments/transformer-actionmixed.yaml) | Transformer 的表示维度、head、layer、FFN 和最大长度 | 使用完整上下文的非因果全序列实验。 |
| [`framework/experiments/yolo-clean-large.yaml`](../framework/experiments/yolo-clean-large.yaml) | group1 大目标、YOLO11n、640 输入和 val testset | 大目标检测训练及探索性评估。 |
| [`framework/experiments/yolo-clean-small.yaml`](../framework/experiments/yolo-clean-small.yaml) | group2 小目标、YOLO11n、640 输入和 val testset | 小目标检测训练及探索性评估，重点观察逐类召回。 |
| [`framework/experiments/roi-fusion.yaml`](../framework/experiments/roi-fusion.yaml) | ROI 分类流水线（`feature_fusion`）、hidden_dim 等网络参数 | 小目标/稀有类的 ROI 特征融合替代方案实验。 |
| [`framework/experiments/yolo11s-large-gl-eval.yaml`](../framework/experiments/yolo11s-large-gl-eval.yaml) | 队友 YOLO11s 大目标模型、group1_large test split、本地化 data_yaml | 队友模型在锁定 test split 上的正式评测配置。 |
| [`framework/experiments/yolo11s-small-zyh-eval.yaml`](../framework/experiments/yolo11s-small-zyh-eval.yaml) | 队友 YOLO11s 小目标模型、group2_small test split、本地化 data_yaml | 队友模型在锁定 test split 上的正式评测配置。 |
| [`framework/experiments/mstcn-autoannotate-smoke.yaml`](../framework/experiments/mstcn-autoannotate-smoke.yaml) | MS-TCN、40 维输入、6 类、10 epoch、`data.root` 指向自动标注转换数据 | 验证 YOLO 自动标注 → 时序训练全链路的 smoke 实验；数据未登记 dataset_ref，只允许 exploratory。 |
| [`framework/experiments/mstcn-actionmixed-auto.yaml`](../framework/experiments/mstcn-actionmixed-auto.yaml) | MS-TCN、40 维输入、6 类、30 epoch、`dataset_ref: temporal.actionmixed-auto-v1`（自动标注特征数据） | 自动标注数据上 MS-TCN 全序列正式训练；与 smoke 的区别是登记数据集、formal 评估。 |
| [`framework/experiments/gru-actionmixed-auto.yaml`](../framework/experiments/gru-actionmixed-auto.yaml) | GRU、40 维输入、6 类、16 帧窗口、`dataset_ref: temporal.actionmixed-auto-v1` | 自动标注数据上 GRU 滑窗正式训练；与历史 gru-actionmixed.yaml（人工标注）同超参，用于对照自动标注特征代价。 |
| [`framework/experiments/transformer-actionmixed-auto.yaml`](../framework/experiments/transformer-actionmixed-auto.yaml) | Transformer、40 维输入、6 类、`max_len: 2560`（覆盖 auto 数据最长约 2080 帧序列）、`dataset_ref: temporal.actionmixed-auto-v1` | 自动标注数据上 Transformer 全序列正式训练；与历史 transformer-actionmixed.yaml 同结构，max_len 上调以容纳更长序列。 |

## 2. Benchmark 数据集和 split

[`framework/testsets.yaml`](../framework/testsets.yaml) 是正式评估数据身份与 split 契约的真源，
由 `framework/cleansight_eval/core/catalog.py`、`tools/validate_testsets.py` 和 framework 评估溯源逻辑读取。

| 字段 | 内容与功能 |
|---|---|
| `schema_version` / `root` | 定义清单版本和相对路径解析根。 |
| `datasets` | 数据集级公共事实：family、版本、数据根或 manifest、feature mapping、维度和 labels。 |
| `revision` | 外部数据仓库的固定 revision；ActionMixed v2 当前钉定为完整 9,532 帧数据对应的 Git commit。 |
| `split_overlap_policy` | `error` 禁止同源跨 split；`frame` 允许同源但禁止具体帧重合；`allow` 关闭重叠门禁。 |
| `testsets` | split 身份、manifest、用途和可选预期样本。 |
| `purpose` | 区分训练、训练期验证、开发 benchmark、锁定 holdout 和 schema smoke。 |

当前内容登记 ActionMixed 时序 train/val/test（`temporal.actionmixed-v2`，人工标注）、
自动标注数据通道 `temporal.actionmixed-auto-v2`（YOLO 检测框 + 人工动作标签，26 个视频
train 17 / val 5 / test 4，检测源 yolo11s-g1/g2-v1，2026-08-27 线路 B 重建；历史 v1 见
registry 各 auto pin）、旧 Endo Project train/test、两组 YOLO val/test 和
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

## 4. 时序模型版本 pin

三个 pin 都记录模型版本、checkpoint SHA-256、统一数据挂载、feature mapping、在线因果属性、
感受野和输出标签。它们用于复现和部署追溯，不控制 framework 训练循环。

| YAML | 内容与功能 |
|---|---|
| [`registry/temporal/gru-v1/pin.yaml`](../registry/temporal/gru-v1/pin.yaml) | 固定 GRU v1 checkpoint、legacy-20d 输入、64 帧窗口和三类输出契约。 |
| [`registry/temporal/causal-tcn-v1/pin.yaml`](../registry/temporal/causal-tcn-v1/pin.yaml) | 固定 Causal TCN v1 checkpoint、数据、特征和在线契约。 |
| [`registry/temporal/causal-transformer-v1/pin.yaml`](../registry/temporal/causal-transformer-v1/pin.yaml) | 固定旧因果 Transformer v1 checkpoint、结构和运行契约。 |
| [`registry/temporal/auto-mstcn-v1/pin.yaml`](../registry/temporal/auto-mstcn-v1/pin.yaml) | 固定 auto 数据通道 MS-TCN v1：checkpoint、40 维 v1 特征、6 类输出、数据集 revision 636e6372。 |
| [`registry/temporal/auto-gru-v1/pin.yaml`](../registry/temporal/auto-gru-v1/pin.yaml) | 固定 auto 数据通道 GRU v1：checkpoint、40 维 v1 特征、16 帧窗口、6 类输出、数据集 revision 636e6372。 |
| [`registry/temporal/auto-transformer-v1/pin.yaml`](../registry/temporal/auto-transformer-v1/pin.yaml) | 固定 auto 数据通道 Transformer v1：checkpoint、40 维 v1 特征、max_len 2560、6 类输出、数据集 revision 636e6372。 |

## 5. Legacy YOLO 快照配置

| YAML | 读取方 | 内容与功能 |
|---|---|---|
| [`legacy/yolo-detection/pipeline/config.yaml`](../legacy/yolo-detection/pipeline/config.yaml) | 仅供审计旧 pipeline | 保存迁移前两组类别、抽帧、训练超参和验收阈值；活跃训练和评测不得读取。 |
| [`legacy/yolo-detection/pipeline/splits.yaml`](../legacy/yolo-detection/pipeline/splits.yaml) | 仅供审计旧 split | 保存迁移前视频 stem 分配；当前评测身份由 `framework/testsets.yaml` 管理。 |

legacy YAML 是冻结快照，不再接收新字段或成为数据/模型真源。

## 6. YOLO registry 元数据

这些文件供评估报告、CARD/打包流程及人工发布检查使用，不是 YOLO 运行时训练配置。

| YAML | 内容与功能 |
|---|---|
| [`registry/detection/yolo-group1-large-v1/classes.yaml`](../registry/detection/yolo-group1-large-v1/classes.yaml) | 固定 group1 checkpoint 的 class ID：hand、scope control body、scope mid section。 |
| [`registry/detection/yolo-group1-large-v1/train_config.yaml`](../registry/detection/yolo-group1-large-v1/train_config.yaml) | 记录 group1 的架构、图像尺寸、训练超参、stride 和验收门槛。 |
| [`registry/detection/yolo-group2-small-v1/classes.yaml`](../registry/detection/yolo-group2-small-v1/classes.yaml) | 固定 group2 checkpoint 的 class ID：syringe、air gun、scope distal end。 |
| [`registry/detection/yolo-group2-small-v1/train_config.yaml`](../registry/detection/yolo-group2-small-v1/train_config.yaml) | 记录 group2 的训练事实和验收门槛；小目标重点审查 recall。 |

## 7. 生成或本地 YAML

以下文件受 `.gitignore` 排除，不进入上面的 32 文件清单：

- `datasets/cleansight-yolo/**/data.yaml`：本地 YOLO 数据挂载及生成的 Ultralytics 清单；framework
  experiment 和 testset catalog 只引用该统一路径。
- `datasets/raw/**/*.yaml`：外部数据随附映射；不得作为仓库配置提交。
- `**/runs/**/args.yaml`、`legacy/**/experiments/**/args.yaml`：训练过程或历史环境快照。
- `legacy/**/raw/**/*.yaml`：随历史目录移动的本地原始数据，不受兼容性保证。
- `modelscope_upload/**/*.yaml`：打包输出副本，应由受跟踪的 experiment、pin 或 registry 重新生成。
- `cleansight-yolo-pipeline-main/**/*.yaml`：本地兼容镜像，不是当前 framework/benchmark 真源。

若生成文件以后转为受 Git 跟踪的稳定契约，必须将其加入本文档的逐文件清单。

## 8. YOLO 自动标注配置

[`framework/experiments/auto-annotate.yaml`](../framework/experiments/auto-annotate.yaml) 由
`framework.cleansight_eval.cli.annotate` 读取，两条自动标注模式共用：`run` 子命令把已训练
YOLO checkpoint 的逐帧检测序列化为 legacy 时序标注 JSON（与 Label Studio 导出同构，可被
历史 `lab.py` 消费）；`run-dataset` 子命令对图片帧序列数据集（images/ + 动作标签 labels/）
逐帧检测，产出与 `convert` 同构的时序训练数据（frames/ + labels/），供 `temporal/data.py`
消费。

| 字段 | 内容与功能 |
|---|---|
| `videos` | `run` 的默认视频文件或目录（相对仓库根）；CLI 未传 `--videos` 时使用，两者均缺失则明确报错。 |
| `checkpoints` | 参与标注的 checkpoint 列表；每项含权重 `path`（相对仓库根）和 `class_map`（本地类别 id → 全局类名，id 必须存在于权重 names 中）。 |
| `imgsz` / `conf` | 推理输入尺寸与置信度阈值；`conf` 可为标量或 `{类别: 阈值}` 字典（按最低阈值推理、逐类过滤），可被 CLI 参数覆盖。 |
| `top_k` | `run` 的每类别轨迹（slot）数；`hand` 默认 2 条，其他类别 1 条，与 clean_bbox_v2 的 slot 语义一致；`track` 启用后不生效。 |
| `frame_stride` / `batch_size` | 每 N 帧推理一次（中间帧沿用最近结果）与批量推理帧/图片数；帧采样可显著降低推理成本。 |
| `track` | `run` 是否启用 ByteTrack 实例跟踪（轨迹按 `(类别, 实例 id)` 组织）。 |
| `out_dir` | `run` 的标注 JSON 输出目录（默认 `outputs/annotations`，`outputs/` 整体被 Git 忽略），可被 CLI `--out` 覆盖。 |

[`framework/experiments/auto-annotate-yolo11.yaml`](../framework/experiments/auto-annotate-yolo11.yaml) 是上述 schema 的检测源变体：权重指向 `legacy/yolo-detection/pipeline/versioned_weights/yolo11s-g{1,2}-v1/best.pt`（源自本地 EXPERIMENTS default 变体，不入 Git），其中 g2 覆盖 5 类小目标（含 `short_brush`/`brush_tip_out`，这两维特征不再恒零）；`videos` 默认指向 `outputs/videos-auto26`（26 个已标注视频的硬链接暂存），`out_dir` 为 `outputs/annotations-yolo11`，与 legacy v3 产物目录 `outputs/annotations` 隔离防止混用。检测源差异按 annotation source 变更处理，背景见 `docs/TEMPORAL_DATASET_TRANSFORMATION_PLAN.md`（线路 B）。

`run-dataset` 复用上述 checkpoints/imgsz/conf/batch_size；输入为 CLI `--dataset` 指定的
数据集根（`images/<split>/<序列>-<帧号:06d>.jpg` 有序帧 + `labels/<split>/<序列>.txt`
动作标签），输出默认原地补写 `frames/`（`--out` 可重定向），检测类别固定为 8 类全局表
（`auto_annotate._constants.DETECTION_CLASSES`，与 `convert` 的 frames bbox 编号一致）。

## 更新检查

1. 将本文档清单与 `git ls-files '*.yaml' '*.yml'` 对照，确保无遗漏和失效链接。
2. 更新对应文件的主要内容、读取方、功能、默认值或不变量及运行影响。
3. 检查 schema、路径、类别顺序、feature mapping、split、checkpoint 和数据版本是否同步。
4. 运行相关 validator 或测试，避免把生成 YAML 误当成配置真源提交。
