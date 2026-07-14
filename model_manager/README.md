# 模型统一管理接口

`model_manager/` 提供一个轻量统一入口，用于管理 YOLO 与时序模型的训练、评测和 benchmark。

该接口不重写各模型已有训练逻辑，而是用 `models.yaml` 登记所有模型，再由 `manager.py` 调用现有脚本。

## 设计目标

- 用一个清单登记所有模型。
- 用统一 CLI 管理 YOLO 和时序模型。
- 保留现有 `03_train.py`、`04_validate.py`、`temporal-*/main.py`。
- 默认只打印命令，避免误触发长时间训练。
- 后续可继续扩展 ModelScope、pin 校验和 Backend 接入检查。

## 模型清单

模型清单位于：

```text
model_manager/models.yaml
```

清单支持 `profiles`。公共字段写在 profile 中，单个模型只写差异项。

例如 YOLO 模型现在复用 `yolo_pipeline` profile；新增同类 YOLO 时通常只需要填：

- `id`
- `profile: yolo_pipeline`
- `target`
- `input.dataset`
- `input.labels`
- `evaluation.testset_id`

时序 v1 模型复用 `temporal_legacy_20d` profile；新增同类时序模型通常只需要填：

- `id`
- `profile: temporal_legacy_20d`
- `workdir`
- `target`
- `output.checkpoint`

profile 和模型条目会递归合并，模型条目优先。字符串中可使用 `{id}`、`{target}`、`{workdir}`、`{checkpoint}` 这几个占位符，解析时会自动展开。

合并后的每个模型记录包含：

- `id`：统一模型 id，例如 `temporal.gru`
- `family`：模型族，例如 `yolo` 或 `temporal`
- `adapter`：当前使用的脚本适配方式
- `workdir`：模型工作目录
- `input`：输入格式、维度、标签或数据集
- `output`：checkpoint、报告、CARD、pin 等产物
- `commands`：训练和评测命令

## 使用方式

从模型集根目录执行：

```bash
../CleanSightBackend/.venv/bin/python model_manager/manager.py list
```

检查登记产物：

```bash
../CleanSightBackend/.venv/bin/python model_manager/manager.py status
```

预览 GRU 训练命令：

```bash
../CleanSightBackend/.venv/bin/python model_manager/manager.py train --model temporal.gru
```

真正执行 GRU 训练：

```bash
../CleanSightBackend/.venv/bin/python model_manager/manager.py train --model temporal.gru --run
```

预览所有时序模型评测命令：

```bash
../CleanSightBackend/.venv/bin/python model_manager/manager.py eval --family temporal
```

预览 YOLO 完整流水线：

```bash
../CleanSightBackend/.venv/bin/python model_manager/manager.py pipeline --model yolo.group1_large
```

真正执行 YOLO 完整流水线：

```bash
../CleanSightBackend/.venv/bin/python model_manager/manager.py pipeline --model yolo.group1_large --run
```

运行集中 benchmark：

```bash
../CleanSightBackend/.venv/bin/python model_manager/manager.py benchmark temporal_feed_mode --run
```

## 当前登记模型

```text
yolo.group1_large
yolo.group2_small
temporal.gru
temporal.tcn
temporal.transformer
```

## 注意事项

- 不传 `--run` 时只打印命令，不会实际训练或评测。
- `pipeline` 目前只登记给 YOLO 模型，顺序为 `00_status.py`、`01_pull_data.py`、`00_status.py --assign`、`02_build_dataset.py`、`03_train.py`、`04_validate.py`。
- `pipeline --run` 会在任一步失败时停止；`04_validate.py` 验收 FAIL 时也会返回非零退出码，但报告仍会写到 `runs/<组>/acceptance_report.md`。
- 执行子命令时会自动读取模型集根目录 `.env`，用于给 `01_pull_data.py` 传入 `LS_HOST`、`LS_TOKEN` 等变量；日志只显示 key 名，不显示 token 值。
- 当前时序 v1 checkpoint 输入维度为 20，仍使用 `legacy-20d-v1`。
- 新版 YOLO 到 64 维 feature mapping 的训练闭环完成后，需要更新 `models.yaml` 中的 `input_dim`、`feature_mapping` 和 checkpoint。
- 如果改变数据集、类别、特征映射或 checkpoint，应同步更新 `CARD.md`、`pin.yaml` 和 benchmark 报告。
