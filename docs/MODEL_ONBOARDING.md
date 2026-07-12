# 新模型接入参考手册

> 面向：要给评估框架加一个新模型的人。
> 覆盖三条流水线：**单帧检测** / **全序列时序** / **历史滑窗时序**。
> 结论先行：模型退化为**可替换的纯组件**，流水线固定；接新模型 = 加一个网络 + 注册一行（时序），
> 或只写配置（YOLO）。

---

## 0. 调度（三条流水线共用）

配置 YAML 里 `pipeline` 一个 key 决定调度，`model.type` 选具体模型：

| key | 选择什么 | 注册表 |
|-----|---------|--------|
| `pipeline` | 三条流水线之一 | `cli/_registry.py` `_PIPELINES` |
| `model.type` | 具体模型 | 时序：`temporal/models/__init__.py` `_MODELS`；检测：`detection/yolo.py` `_ADAPTERS` |

入口 `cli/train.py:main` / `cli/eval.py:main` → `get_pipeline(cfg["pipeline"])` → 交给对应流水线。

**关键认知**：监督/loss 语义属于**流水线**，不属于模型。全序列一律逐帧 CE、滑窗一律末帧 CE +
因果平滑。所以模型只需提供网络结构；同一个 `nn.Module`（如 GRU）在两条时序流水线里都能用，
差别由流水线决定。

---

# 一、时序模型（GRU / MS-TCN / …）

## 1.1 差异点集中在哪几个环节

跑一遍训练/评估，**只有"网络结构"随模型变**，其余全共享：

| 环节 | 变还是共享 | 位置 |
|------|-----------|------|
| 数据加载 / 特征化 | **共享** | `temporal/data.py:load_split` |
| 网络结构 | **模型变** | `temporal/models/<name>.py`（纯 `nn.Module`） |
| 训练前归一化（可选） | 模型可选提供 | 模型的 `fit_normalization(features)` 方法 |
| **loss 监督粒度**（末帧 vs 逐帧） | **流水线决定** | `sliding_window_pipeline` / `full_sequence_pipeline` |
| 切窗 / 组 dataset / 评估循环 / 平滑 | **流水线决定** | 同上两个流水线文件 |
| 是否测延迟 | **流水线决定** | 滑窗测、全序列 N/A |
| 训练主循环 / 优化器 / 梯度裁剪 | **共享**（各流水线内） | 两个流水线的 `train` |
| 指标（acc/edit/F1@k）/ 延迟测量 | **共享** | `temporal/metrics.py` |
| checkpoint 读写 / envelope / 完整性 | **共享** | `core/*` |

参照两个现成实现：
- **GRU**（因果）：无归一化；可用于**两条**时序流水线。
- **MS-TCN**（非因果）：`fit_normalization` 用训练集拟合 z-score 写入 buffer；只能用于**全序列**。

## 1.2 新接入一个时序模型：要实现什么

### A. 新建 `temporal/models/<你的名字>.py`

只写一个纯 `nn.Module`：输入 `[B, T, F]`、输出逐帧 logits `[B, T, C]`。**不写任何监督/喂入
逻辑**——那些是流水线的事。

```python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128, ...):
        super().__init__()
        ...
    def forward(self, x):        # x:[B,T,F] -> [B,T,C]
        ...

    # 可选：模型若需输入归一化，实现此方法，全序列流水线会 duck-type 调用
    def fit_normalization(self, features): ...
```

### B. 注册：`temporal/models/__init__.py` 的 `_MODELS`

```python
def _build_mynet(cfg):
    return MyNet(input_dim=cfg["input_dim"], num_classes=cfg["num_classes"], hidden=cfg.get("hidden", 128))

_MODELS = {
    ...,
    "mynet": {"build": _build_mynet, "causal": True},   # causal=True 才允许进滑窗流水线
}
```

- **因果**（只看历史+当前，可流式）→ `causal: True`，两条时序流水线都能用。
- **非因果**（看全序列）→ `causal: False`，只能进 `full_sequence_temporal`；滑窗流水线会拒绝。

### C. 写配置 YAML

```yaml
pipeline: sliding_window_temporal   # 或 full_sequence_temporal
model:
  type: mynet
  input_dim: 40
  num_classes: 6
  hidden: 128                       # 你的自定义超参
train: { epochs: 20, lr: 1.0e-3, batch_size: 32, window: 64 }   # 全序列不需要 batch_size/window
data: { root: /path, split_train: train, split_eval: val }
feature_schema: { dim: 40, version: actionmixed-bbox-8cls-v1 }
```

**不用改**：两个流水线文件、`data.py`、`metrics.py`、CLI、`core/*`。

---

# 二、YOLO（单帧检测）

检测流水线目前**只接 YOLO**。`YoloAdapter`（[yolo.py](../framework/cleansight_eval/detection/yolo.py)）
已封好 ultralytics 的训练/验证，并**兼任 data loader**（从 `data.yaml` 自持读图、组 batch）。

**结论：新增一个 YOLO 模型 = 只写一份配置 YAML，零代码。** 换的只是预训练权重和超参，
`model.type` 始终是 `yolo`，同一个 adapter 复用。

```yaml
pipeline: detection
model:
  type: yolo                 # 固定 yolo
  weights: yolo11s.pt        # 换这里：yolo11n/s/m/l.pt，或自定义结构 .yaml
  imgsz: 640
train: { epochs: 100, batch: 16, patience: 20 }
data:
  data_yaml: /path/data.yaml # cleansight-yolo-pipeline 产出的标准数据集
  name: mydataset            # 产物/run 命名
  eval_split: val
```

- `weights` 传 `.pt` 走微调；传结构 `.yaml` 走从头训。二者都由 ultralytics `YOLO(weights)` 直接消费。
- 入口与时序完全一致：`cli/train.py --config xxx.yaml`、`cli/eval.py`。

**要改代码的情况**（暂不做）：接 YOLO 之外的检测器（Faster R-CNN 等）才需要新写一个 adapter 类
并在 `detection/yolo.py` 的 `_ADAPTERS` 注册——需要时再说。

---

# 三、三条流水线对照速查

| | 全序列时序 | 滑窗时序 | YOLO 检测 |
|--|----------|---------|-----------|
| `pipeline` | `full_sequence_temporal` | `sliding_window_temporal` | `detection` |
| 新增模型改动 | 加 `nn.Module` + 注册一行 | 同左（须 `causal: True`） | 只写配置 |
| 监督口径 | 逐帧 CE | 末帧 CE + 因果平滑 | ultralytics 自持 |
| data loader | `data.py` 统一（40 维特征序列） | 同左 | adapter 自持（读 data.yaml） |
| 性能延迟 | N/A | 单 tick 实测 | N/A |
| 共享出口 | envelope / metrics / integrity / matrix | 同左 | 同左 |

> 设计意图：模型只管网络结构，监督与推理由流水线拥有；时序与检测两域数据格式不同、故意不
> 强行统一。接时序新模型走 `models/` 注册，接 YOLO 只调配置。
