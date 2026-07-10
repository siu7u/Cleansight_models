# CleanSight Models 项目流程图

本文档描述 `Cleansight_models` 当前的端到端模型资产流程。仓库边界是:本仓库负责训练、评估、版本登记和模型卡;`../CleanSightBackend` 负责在线加载、视频流推理、可视化和告警。

## 总览

```mermaid
flowchart TD
    A[数据来源] --> A1[Label Studio 导出 JSON]
    A --> A2[LS 原始视频]
    A --> A3[ModelScope cleansight-ActionMixed]

    A1 --> B[YOLO 数据流水线]
    A2 --> B
    A3 --> B

    B --> C1[group1_large YOLO<br/>hand / scope_control_body / scope_mid_section]
    B --> C2[group2_small YOLO<br/>syringe / air_gun / scope_distal_end]

    C1 --> D[YOLO 单模型评估<br/>val / test]
    C2 --> D
    D --> E[versioned_weights<br/>yolo-large-vN / yolo-small-vN]
    D --> F[YOLO registry<br/>CARD / metrics / eval_report / pin]

    C1 --> G[YOLO 检测结果]
    C2 --> G
    G --> H[feature_mapping<br/>检测结果 -> 时序特征]

    H --> I1[temporal-gru]
    H --> I2[temporal-causal-tcn]
    H --> I3[temporal-transformer]
    H --> I4[temporal-mstcn-offline<br/>离线上限参考]

    I1 --> J[时序单模型评估]
    I2 --> J
    I3 --> J
    I4 --> J

    J --> K[Benchmark 汇总]
    F --> K
    K --> L{上线门禁}

    L -->|PASS| M[整理模型 bundle]
    L -->|FAIL| N[继续补数据 / 调模型 / 重训]
    M --> O[CleanSightBackend 在线推理]
```

## YOLO 数据、训练和评估流程

```mermaid
flowchart TD
    A[raw/exports/*.json<br/>Label Studio 标注] --> B[00_status.py<br/>对账]
    C[raw/videos/*.mp4<br/>LS 视频] --> B
    B -->|缺视频| D[01_pull_data.py<br/>下载视频]
    D --> B
    B -->|质检通过| E[config.yaml<br/>only_videos]
    E --> F[00_status.py --assign<br/>写 splits.yaml]
    F --> G[02_build_dataset.py<br/>LS JSON + 视频 -> YOLO 数据集]

    H[raw/modelscope/cleansight-ActionMixed] --> I[02_import_modelscope_dataset.py<br/>images + frames -> 分组数据]
    G --> I

    I --> J[datasets/group1_large<br/>images/labels train val test]
    I --> K[datasets/group2_small<br/>images/labels train val test]

    J --> L[03_train.py group1_large<br/>训练 YOLO]
    K --> M[03_train.py group2_small<br/>训练 YOLO]

    L --> N[runs/group1_large/weights/best.pt]
    M --> O[runs/group2_small/weights/best.pt]

    N --> P[versioned_weights/yolo-large-vN/best.pt]
    O --> Q[versioned_weights/yolo-small-vN/best.pt]

    P --> R[04_validate.py group1_large<br/>val / test / --weights]
    Q --> S[04_validate.py group2_small<br/>val / test / --weights]

    R --> T[runs/group1_large/acceptance_report*.md]
    S --> U[runs/group2_small/acceptance_report*.md]
```

## YOLO 版本管理流程

```mermaid
flowchart TD
    A[03_train.py] --> B[runs/<group>/weights/best.pt<br/>运行权重]
    B --> C[versioned_weights/<model>-vN/best.pt<br/>候选权重]
    C --> D[04_validate.py --weights<br/>val 验收]
    D --> E[04_validate.py --split test --weights<br/>holdout 测试]
    E --> F{是否达标}

    F -->|否| G[保留报告<br/>继续补数据或重训]
    F -->|是| H[registry/yolo-group*-vN<br/>正式登记]

    H --> H1[CARD.md<br/>模型卡和训练历史]
    H --> H2[classes.yaml<br/>类别顺序]
    H --> H3[train_config.yaml<br/>训练配置]
    H --> H4[metrics.json<br/>机器可读指标]
    H --> H5[eval_report.md<br/>人工评估报告]
    H --> H6[pin.yaml<br/>数据/权重/版本钉定]

    H --> I[ModelScope 上传目录]
    H --> J[CleanSightBackend 配置引用]
```

当前已有自动化:

- `03_train.py` 会复制 `runs/<group>/weights/best.pt` 到 `versioned_weights/<model>-vN/best.pt`。
- `03_train.py` 会通过 `yolo-detection/pipeline/utils/card.py` 追加训练历史到 registry 的 `CARD.md`。
- `04_validate.py` 会写 `runs/<group>/acceptance_report.md`、`acceptance_report_test.md` 和 timestamp 归档报告。

当前仍需补齐自动化:

- PASS 后自动创建新的 `registry/yolo-group*-vN/`。
- 自动生成 `metrics.json`、`eval_report.md`、`pin.yaml`。
- 自动合并 test、部署机延迟、参数量和上线门禁结论。

## 时序模型流程

```mermaid
flowchart TD
    A[YOLO 检测结果] --> B[feature_mapping.py<br/>检测结果 -> 特征序列]
    B --> C[时序特征<br/>当前历史版本 legacy-20d-v1]
    C --> D[build_testset.py<br/>构造 window=64 样本]

    D --> E1[temporal-gru/main.py]
    D --> E2[temporal-causal-tcn/main.py]
    D --> E3[temporal-transformer/main.py]

    E1 --> F1[registry/gru-v1/*.pt]
    E2 --> F2[registry/tcn-v1/*.pt]
    E3 --> F3[registry/transformer-v1/*.pt]

    F1 --> G[tools/eval_temporal_detailed.py<br/>Acc / Edit / F1 / per-class recall]
    F2 --> G
    F3 --> G

    F1 --> H[tools/measure_temporal_latency.py<br/>部署机延迟]
    F2 --> H
    F3 --> H

    G --> I[CARD.md / REPORT.md]
    H --> I
    I --> J[pin.yaml<br/>dataset / yolo / feature_mapping / temporal checkpoint]
```

## Benchmark 和上线门禁

```mermaid
flowchart TD
    A[单模型 benchmark] --> A1[benchmark/single_model/run_yolo_benchmark.py]
    A --> A2[benchmark/single_model/run_temporal_benchmark.py]

    B[接口 benchmark] --> B1[benchmark/temporal_feed_mode<br/>full sequence vs streaming]
    B --> B2[feature_mapping 一致性<br/>待完善]

    C[端到端 benchmark] --> C1[benchmark/e2e_3min/run_e2e_benchmark.py]
    C1 --> C2[CleanSightBackend 导出 prediction JSON]

    A1 --> D[benchmark 报告]
    A2 --> D
    B1 --> D
    C1 --> D

    D --> E{上线门禁}
    E --> E1[指标达标]
    E --> E2[部署机延迟实测]
    E --> E3[参数量记录]
    E --> E4[因果性/感受域说明]
    E --> E5[checkpoint / dataset / feature_mapping pin 完整]

    E1 --> F{可上线?}
    E2 --> F
    E3 --> F
    E4 --> F
    E5 --> F

    F -->|是| G[CleanSightBackend 加载模型 bundle]
    F -->|否| H[阻塞上线<br/>继续训练/补数据/补报告]
```

## 一条典型操作链

```mermaid
sequenceDiagram
    participant LS as Label Studio / ModelScope
    participant YP as yolo-detection/pipeline
    participant YOLO as YOLO weights
    participant TM as temporal-* models
    participant BM as benchmark
    participant REG as registry / CARD / pin
    participant BE as CleanSightBackend

    LS->>YP: 导入 JSON / 视频 / ModelScope 数据
    YP->>YP: 00_status / 01_pull_data / 02_build_dataset
    YP->>YP: 02_import_modelscope_dataset
    YP->>YOLO: 03_train.py 训练分组检测模型
    YOLO->>YP: versioned_weights/yolo-*-vN/best.pt
    YP->>BM: 04_validate.py val/test 验收
    BM->>REG: PASS 后登记 registry 版本
    YOLO->>TM: 生成/更新 feature_mapping 输入
    TM->>TM: 训练 GRU / TCN / Transformer
    TM->>BM: 时序评估和延迟测试
    BM->>REG: CARD / pin / benchmark 报告补齐
    REG->>BE: 通过门禁后交付模型 bundle
```
