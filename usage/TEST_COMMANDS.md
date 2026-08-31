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

新机器可用组员工具一键检查/安装：

```bash
python tools/team_env.py          # 检查依赖
python tools/team_env.py --setup  # 在当前环境安装 framework/requirements.txt
python tools/team_env.py --setup-venv  # 创建仓库内 .venv 并安装
```

确认 CLI 可用：

```bash
python -m benchmark.cli.eval --help
```

## 1.1 组员快速训练（framework CLI --model）

训练入口简化（framework CLI）：一条命令一个模型，无需手动挑 yaml：

```bash
python -m framework.cleansight_eval.cli.train --list-models   # 列出所有可训模型
python -m framework.cleansight_eval.cli.train --model yolo11s --group group1_large  # YOLO 指定规模+组
python -m framework.cleansight_eval.cli.train --model gru      # GRU 时序
python -m framework.cleansight_eval.cli.train --model mstcn    # MS-TCN
python -m framework.cleansight_eval.cli.train --model feature_fusion -S data.classes=air_gun  # ROI 特征融合
```

手动训练生命周期（后台启动/进度/恢复/日志/评测，framework CLI）：

```bash
python -m framework.cleansight_eval.cli.manual start --model yolo11s --group group1_large --bg
python -m framework.cleansight_eval.cli.manual status
python -m framework.cleansight_eval.cli.manual resume      # 中断后从 last.pt 恢复
python -m framework.cleansight_eval.cli.manual eval        # 评测最新 best.pt
python -m framework.cleansight_eval.cli.manual logs -f     # 跟踪训练日志
```

数据下载与校验（framework 数据契约层 CLI）：

```bash
python -m framework.cleansight_eval.cli.dataset --preset all   # 一键下载训练所需全部数据集
python -m framework.cleansight_eval.cli.dataset --check        # 校验是否就绪
```

完整入门见 [`docs/TEAM_GUIDE.md`](../docs/TEAM_GUIDE.md)。

## 2. 模型评测命令格式

统一格式：

```bash
python -m benchmark.cli.eval \
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
python -m benchmark.cli.eval \
  --config framework/experiments/gru-actionmixed.yaml \
  --ckpt runs/<gru-run-id>/checkpoints/best.pt
```

输入语义为 `[1, window, 40]`，逐窗推进并输出末帧预测。默认还会生成：

```text
runs/<gru-run-id>/viz/segmentation-test-p01.png
```

### 3.2 MS-TCN / Transformer 全序列模型

```bash
python -m benchmark.cli.eval \
  --config framework/experiments/mstcn2-actionmixed.yaml \
  --ckpt runs/<mstcn2-run-id>/checkpoints/best.pt
```

将配置替换为 `mstcn-actionmixed.yaml` 或 `transformer-actionmixed.yaml` 即可评测对应模型。
输入语义为 `[1, T, 40]`，属于离线完整序列评测，不代表生产流式延迟。

组员只提供裸时序 `.pt` 时，先复制仓库提供的模板：

```bash
mkdir -p external_checkpoints/<model-id>
cp external_checkpoints/external-temporal-template.yaml \
  external_checkpoints/<model-id>/<model-id>.yaml
```

将 `.pt` 放进同一目录，逐项替换模板已启用字段中的 `REPLACE_WITH_*` 和 `0` 占位值。没有绑定 metadata 时保留
`evaluation.mode: exploratory` 和：

```yaml
model:
  allow_missing_meta: true
```

YAML 中的 Pipeline、模型类型、维度、层数、类别顺序、feature mapping、归一化和训练窗口必须
与权重来源一致。加载器接受
裸 state dict、常见包装或 TorchScript 归档；TorchScript 通过 JIT API 提取参数，之后仍严格检查
参数键和张量形状，结果会标记 checkpoint metadata 未绑定及实际格式。

配置完成后的统一命令：

```bash
python -m benchmark.cli.eval \
  --config external_checkpoints/<model-id>/<model-id>.yaml \
  --ckpt external_checkpoints/<model-id>/<model-id>.pt \
  --out-dir runs/external-<model-id>
```

后端 CLEAN 三种离线 best checkpoint 已配套 exploratory YAML，可直接进入统一完整序列评测：

```bash
python -m benchmark.cli.eval \
  --config external_checkpoints/asformer-offline/best_asformer_offline_segmenter.yaml \
  --ckpt external_checkpoints/asformer-offline/best_asformer_offline_segmenter.pt

python -m benchmark.cli.eval \
  --config external_checkpoints/bigru-offline/best_bigru_offline_segmenter.yaml \
  --ckpt external_checkpoints/bigru-offline/best_bigru_offline_segmenter.pt

python -m benchmark.cli.eval \
  --config external_checkpoints/mstcn-bilstm-offline/best_ms_tcn_offline_segmenter.yaml \
  --ckpt external_checkpoints/mstcn-bilstm-offline/best_ms_tcn_offline_segmenter.pt
```

这些配置把 ActionMixed 五列 bbox 的缺失 confidence 显式设为 `1.0`，只用于模型接入和统一
指标的 exploratory 对比；正式复现必须消费带真实检测 confidence/timestamp 的后端
`FrameFeature` 或原 offline-model feature store。

### 3.3 已迁移的历史时序模型

历史模型不再运行 `legacy/temporal-*/main.py`，而是使用 framework 兼容模型、顶层 registry
checkpoint 和统一 benchmark：

```bash
python -m benchmark.cli.eval \
  --config framework/experiments/legacy-gru-v1.yaml \
  --ckpt registry/temporal/gru-v1/gru-final-20260704-150629.pt

python -m benchmark.cli.eval \
  --config framework/experiments/legacy-causal-tcn-v1.yaml \
  --ckpt registry/temporal/causal-tcn-v1/tcn-final-20260704-160652.pt

python -m benchmark.cli.eval \
  --config framework/experiments/legacy-causal-transformer-v1.yaml \
  --ckpt registry/temporal/causal-transformer-v1/transformer-final-20260704-161653.pt
```

这些旧权重没有当前 checkpoint metadata 绑定，因此配置固定为 `exploratory`。运行前需要按
[`datasets/README.md`](../datasets/README.md) 挂载 `datasets/endo-project-v1`。

### 3.4 YOLO 检测模型

```bash
python -m benchmark.cli.eval \
  --config framework/experiments/yolo-clean-large.yaml \
  --ckpt runs/<yolo-run-id>/checkpoints/group1_large/weights/best.pt
```

YOLO checkpoint 比时序模型多一层 `<data.name>/weights/`。小目标模型使用
`framework/experiments/yolo-clean-small.yaml`，评测时重点查看逐类 recall 和漏检情况。

## 4. 评测前检查 testset

正式评测前先验证 testset 清单、manifest、fingerprint 和 split 重叠策略：

```bash
python tools/validate_testsets.py \
  --catalog framework/testsets.yaml \
  --json
```

`formal` 评测要求 testset 已登记、数据门禁通过，并且 checkpoint 具有可信的同名 metadata。
外部权重或临时数据应明确使用 `exploratory` 配置，不能将其结果描述为正式 benchmark。

## 4.1 YOLO 优化实验（sweep）与逐类分析（analyze）

YOLO 多方法优化实验（多预设 / grid 搜索，复用 framework 的 YoloAdapter）：

```bash
# 跑预设（基线 + 更强配置），dry-run 只预览不训练
python -m framework.cleansight_eval.cli.sweep \
  --group group1_large --preset large_baseline large_s --dry-run

# grid 搜索（模型 × 分辨率）
python -m framework.cleansight_eval.cli.sweep \
  --group group2_small --grid models resolutions
```

小目标逐类阈值分析与淘汰决策（消费 framework predict 的 native_metrics）：

```bash
python -m benchmark.cli.analyze \
  --config framework/experiments/yolo-clean-small.yaml \
  --ckpt <best.pt> \
  --threshold 0.3
```

ROI 特征融合（`roi_classification` 流水线）训练与评测：

```bash
python -m framework.cleansight_eval.cli.train --config framework/experiments/roi-fusion.yaml
python -m benchmark.cli.eval --config framework/experiments/roi-fusion.yaml --ckpt <checkpoint路径>
```

完整工作流见 [`docs/YOLO_OPTIMIZATION.md`](../docs/YOLO_OPTIMIZATION.md)。

时序模型 + ROI 空间特征（`actionmixed-roi-grid-v1`，144 维；与 40 维 bbox 基线同超参对照）：

```bash
python -m framework.cleansight_eval.cli.train --config framework/experiments/gru-actionmixed-auto-roi.yaml
python -m framework.cleansight_eval.cli.train --config framework/experiments/mstcn-actionmixed-auto-roi.yaml
python -m framework.cleansight_eval.cli.train --config framework/experiments/transformer-actionmixed-auto-roi.yaml
# 冒烟（1 epoch，验证数据链路）：
python -m framework.cleansight_eval.cli.train --config framework/experiments/gru-actionmixed-auto-roi.yaml -S train.epochs=1
```

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
python -m benchmark.cli.matrix --runs runs
```

只比较同类 Pipeline：

```bash
python -m benchmark.cli.matrix \
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
