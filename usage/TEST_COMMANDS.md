# 测试命令行简要教程

本仓库中的“测试”有两种含义：

- **模型评测**：加载 checkpoint，在 YAML 指定的 split/testset 上计算指标并生成报告。
- **代码测试**：使用 `pytest` 验证代码契约和冒烟流程。

以下命令默认从仓库根目录执行。

## 1. 准备环境

先激活已经安装项目依赖的 Python 环境。团队开发机可复用 Backend 的虚拟环境：

```bash
source ../CleanSightBackend/.venv/bin/activate
```

确认 CLI 可用：

```bash
python -m framework.cleansight_eval.cli.eval --help
```

## 2. 模型评测命令格式

统一格式：

```bash
python -m framework.cleansight_eval.cli.eval \
  --config <实验配置.yaml> \
  --ckpt <checkpoint.pt> \
  --out-dir <可选输出目录>
```

| 参数 | 是否必填 | 作用 |
|---|---|---|
| `--config` | 是 | 指定 Pipeline、模型结构、数据 split、testset 和评测参数。 |
| `--ckpt` | 是 | 指定被评测权重；必须与配置中的模型类型、维度和类别兼容。 |
| `--out-dir` | 否 | 自定义评测输出目录；缺省时写入 checkpoint 所属 run 的 `evals/`。 |

需要特别注意：`--ckpt` 只选择模型。时序测试集由 YAML 中的
`data.dataset_ref + data.split_eval` 唯一推导；检测测试集由 `data.eval_split` 和显式
`evaluation.testset_id` 决定。

## 3. 常用评测示例

### 3.1 GRU 滑窗时序模型

```bash
python -m framework.cleansight_eval.cli.eval \
  --config framework/experiments/gru-actionmixed.yaml \
  --ckpt runs/<gru-run-id>/checkpoints/best.pt
```

输入语义为 `[1, window, 40]`，逐窗推进并输出末帧预测。默认还会生成：

```text
runs/<gru-run-id>/viz/segmentation-test-p01.png
```

### 3.2 MS-TCN / Transformer 全序列模型

```bash
python -m framework.cleansight_eval.cli.eval \
  --config framework/experiments/mstcn2-actionmixed.yaml \
  --ckpt runs/<mstcn2-run-id>/checkpoints/best.pt
```

将配置替换为 `mstcn-actionmixed.yaml` 或 `transformer-actionmixed.yaml` 即可评测对应模型。
输入语义为 `[1, T, 40]`，属于离线完整序列评测，不代表生产流式延迟。

### 3.3 YOLO 检测模型

```bash
python -m framework.cleansight_eval.cli.eval \
  --config framework/experiments/yolo-clean-large.yaml \
  --ckpt runs/<yolo-run-id>/checkpoints/group1_large/weights/best.pt
```

YOLO checkpoint 比时序模型多一层 `<data.name>/weights/`。小目标模型使用
`framework/experiments/yolo-clean-small.yaml`，评测时重点查看逐类 recall 和漏检情况。

## 4. 评测前检查 testset

正式评测前先验证 testset 清单、manifest、fingerprint 和 split 重叠策略：

```bash
python tools/validate_testsets.py \
  --catalog benchmark/testsets.yaml \
  --json
```

`formal` 评测要求 testset 已登记、数据门禁通过，并且 checkpoint 具有可信的同名 metadata。
外部权重或临时数据应明确使用 `exploratory` 配置，不能将其结果描述为正式 benchmark。

## 5. 查看评测输出

一次评测通常生成：

```text
runs/<run-id>/
├── artifacts/*.predictions.json
├── evals/*.evaluation.json
├── evals/*.delivery.manifest.json
├── viz/segmentation-<split>-pNN.png
└── checkpoints/
    ├── <checkpoint>.eval.md
    └── EVALUATION_REPORT.md
```

时序 timeline 默认开启，每页最多 6 个视频。需要调整时，在实验 YAML 中设置：

```yaml
evaluation:
  visualize: true
  viz_per_page: 6
```

评测 CLI 当前不支持 `-S/--set` 临时覆盖；测试配置变化需要修改或新建实验 YAML。

## 6. 汇总多个评测结果

汇总所有 Pipeline：

```bash
python -m framework.cleansight_eval.cli.matrix --runs runs
```

只比较同类 Pipeline：

```bash
python -m framework.cleansight_eval.cli.matrix \
  --runs runs \
  --pipeline sliding_window_temporal
```

输出为 `matrix.json` 和 `matrix.md`，保留 computed、N/A 和 MISSING 三态，不生成跨任务综合分。

## 7. 代码测试命令

运行全部测试：

```bash
pytest -q
```

只运行 framework 测试：

```bash
pytest -q framework/tests
```

只运行一个测试文件：

```bash
pytest -q framework/tests/test_temporal_viz.py
```

只运行一个测试函数：

```bash
pytest -q \
  framework/tests/test_pipeline_smoke.py::test_end_to_end
```

若当前 shell 找不到 `pytest`，可直接使用已经安装依赖的虚拟环境：

```bash
../CleanSightBackend/.venv/bin/pytest -q framework/tests
```

带有合成小数据、单 epoch 或 `max-videos/max-frames` 限制的测试只能称为 smoke test，不能作为
正式模型质量结论。
