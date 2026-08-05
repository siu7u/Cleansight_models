# 🚀 CleanSight 模型管理仓库 · 组员快速指南

> 本文是**组员上手唯一入口**：clone 后 5 分钟跑通第一个训练。
> 全部命令在仓库根目录执行。约 3 步：**装环境 → 下载数据 → 训练**。

---

## 0. 这个仓库是干什么的

CleanSight 的**模型训练与评估仓库**。作为组员，你在这里做两件事：

1. **训练模型**（YOLO 目标检测 / GRU / MS-TCN / Transformer / ROI 特征融合）
2. **评测模型**（在固定测试集上出指标报告）

> 仓库不负责在线推理和业务告警——那是 `CleanSightBackend` 的事。这里只产出模型与评估事实。

---

## 1. 获取代码

```bash
git clone https://github.com/siu7u/Cleansight_models.git
cd Cleansight_models
```

---

## 2. 准备环境（5 分钟）

```bash
# 一键检查依赖，缺什么会提示
python tools/team_env.py

# 推荐：用 --setup-venv 在仓库内建独立虚拟环境（新机器首选）
python tools/team_env.py --setup-venv
source .venv/bin/activate

# 或者：已有 python 环境直接装依赖
python tools/team_env.py --setup
```

> 💡 **GPU 机器**：有 NVIDIA 显卡就能用，训练自动选 GPU（`torch.cuda`）。
> 💡 **Mac**：自动用 MPS；**没有 GPU 的机器也能训练**，只是慢（冒烟/小实验可用）。

---

## 3. 下载数据（一次即可）

```bash
# 一键下载训练所需的全部数据集（YOLO + 时序），自动放到 datasets/ 正确位置
python tools/team_dataset.py --preset all

# 只看数据源清单 / 校验是否就绪
python tools/team_dataset.py --list-presets
python tools/team_dataset.py --check
```

| 数据集 | 用途 | 落盘位置 |
|---|---|---|
| `cleansight-yolo` | YOLO 检测（group1_large 大目标 / group2_small 小目标） | `datasets/cleansight-yolo/` |
| `cleansight-ActionMixed` | 时序模型（GRU/MS-TCN/Transformer 的 40 维特征） | `datasets/cleansight-ActionMixed/` |

> 数据默认不进入 Git（体积大），每台新机器下载一次即可。

---

## 4. 训练模型

### 4.1 看有哪些模型可以训练

```bash
python tools/team_train.py --list
```

### 4.2 训练速查表（复制即用）

| 想训练 | 命令 |
|---|---|
| **YOLO nano 大目标组** | `python tools/team_train.py --model yolo11n --group group1_large` |
| **YOLO small 大目标组** | `python tools/team_train.py --model yolo11s --group group1_large` |
| **YOLO medium 大目标组** | `python tools/team_train.py --model yolo11m --group group1_large` |
| **YOLO nano 小目标组** | `python tools/team_train.py --model yolo11n --group group2_small` |
| **GRU 因果滑窗** | `python tools/team_train.py --model gru` |
| **MS-TCN 全序列** | `python tools/team_train.py --model mstcn` |
| **MS-TCN++ 全序列** | `python tools/team_train.py --model mstcn2` |
| **Transformer 全序列** | `python tools/team_train.py --model transformer` |
| **ROI 特征融合** | `python tools/team_train.py --model feature_fusion -S data.classes=<类名>` |

### 4.3 调整训练参数（不用改文件）

```bash
# -S 点路径覆盖任意超参，例如：
python tools/team_train.py --model yolo11m --group group1_large \
    -S train.epochs=200 -S model.imgsz=960 -S train.batch=8
```

### 4.4 产物在哪里

训练结束会打印 checkpoint 路径，默认在：

```text
runs/<模型>-<时间戳>/
├── checkpoints/
│   ├── best.pt            # 最佳权重（交付用这个）
│   └── last.pt            # 最后一轮权重
├── config.resolved.json   # 本次训练的完整配置
├── history.csv / results.csv
└── status.json
```

---

## 5. 评测模型

```bash
python -m benchmark.cli.eval \
  --config framework/experiments/yolo-clean-large.yaml \
  --ckpt runs/<run>/checkpoints/group1_large/weights/best.pt
```

评测输出 mAP/P/R 与逐类指标报告，产物在 `runs/<run>/evals/`。

---

## 6. 常见问题

| 问题 | 解决 |
|---|---|
| `torch.cuda.is_available()` 为 False | 检查驱动 `nvidia-smi`；无 GPU 则用 CPU（慢） |
| 显存不足（OOM） | 调小 batch：`-S train.batch=4`，或降分辨率 `-S model.imgsz=640` |
| yolo11s/m 权重下载失败 | 首次训练需联网下载预训练权重（约 20-50MB），重试或换 `yolo11n` |
| 提示"数据集未就绪" | 先跑 `python tools/team_dataset.py --preset all` |
| 训练很慢 | 小实验先降 epoch：`-S train.epochs=5`；正式训练请用 GPU 机器 |
| matplotlib 缓存目录告警 | `export MPLCONFIGDIR=/tmp/matplotlib` |

---

## 7. 进阶（可选）

- **多模型对比实验**：`python -m framework.cleansight_eval.cli.sweep --group group1_large --preset large_baseline large_s`
- **小目标淘汰决策**：`python -m benchmark.cli.analyze --config framework/experiments/yolo-clean-small.yaml --ckpt <best.pt>`
- 完整优化工作流见 [`docs/YOLO_OPTIMIZATION.md`](YOLO_OPTIMIZATION.md)
- 架构细节见 [`docs/ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md)

---

*有问题先看本文档；解决不了问仓库维护者。*
