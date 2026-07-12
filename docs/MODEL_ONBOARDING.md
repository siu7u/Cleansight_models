# 新模型接入参考手册

> 面向：要给评估框架加一个新模型的人。
> 覆盖：**temporal（时序）** 与 **YOLO（单帧检测）** 两纵。
> 结论先行：两纵是**故意不相交的两套契约**，不要试图统一。
> 检测纵目前只接 YOLO；接 YOLO 之外的检测模型是**未来的事**，需要时再另说。

---

## 0. 三层调度（两纵共用）

配置 YAML 里有三个 key 决定调度：

| key | 选择什么 | 注册表 |
|-----|---------|--------|
| `task` | 纵（temporal / detection） | `cli/_registry.py:14` `_VERTICALS` |
| `family` | 模型族 / 适配器 | temporal: `temporal/family/__init__.py:13`；detection: `detection/adapter.py:97` |
| `feeding` | 喂入 / 评估协议 | temporal: `temporal/feeding/__init__.py:14`；detection: 恒为 `single_frame` |

入口 `cli/train.py:main`、`cli/eval.py:main` → `get_vertical(cfg["task"])` → 交给对应 Orchestrator。之后两纵分道扬镳。

---

# 一、Temporal（时序）

时序纵有**两条正交的抽象轴**，这是它和 detection 最大的区别：

- **family（模型族）** = 网络结构 + 因果/非因果的 loss 契约。→ `temporal/family/`
- **feeding（喂入模式）** = 帧怎么切窗、怎么组 batch、怎么跑评估。→ `temporal/feeding/`

一个模型 = `family × feeding` 的组合。两者独立注册、独立选择。

## 1.1 差异点集中在哪几个环节

跑一遍训练/评估，**只有这几处随模型变**，其余全共享：

| 环节 | 变还是共享 | 位置 |
|------|-----------|------|
| 数据加载 / 特征化 | **共享** | `temporal/loader.py:load_split` |
| 网络结构 | **family 变** | `family.build_network` |
| 训练前预处理（如归一化） | **family 变** | `family.prepare` |
| 前向 | family 变（通常 `return model(x)`） | `family.forward` |
| **loss 监督粒度**（末帧 vs 逐帧） | **family 变（核心差异）** | `family.compute_loss` |
| 单帧推理（滑窗用） | family 变 | `family.predict_frame_logits` |
| checkpoint 元信息 | family 变（一般照抄样板） | `family.checkpoint_meta` |
| 切窗 / 组 dataset | **feeding 变** | `feeding.build_datasets` |
| 评估循环 / 平滑决策 | **feeding 变** | `feeding.evaluate` |
| 训练 batch_size | feeding 变 | `feeding.train_batch_size` |
| 是否测延迟 | feeding 变 | `feeding.requires_performance` |
| 训练主循环 / 优化器 / 梯度裁剪 | **共享** | `temporal/orchestration.py:train` |
| 指标（acc/edit/F1@k） | **共享** | `temporal/metrics.py:compute_temporal_metrics` |
| checkpoint 读写 / envelope / 完整性 | **共享** | `core/checkpoint.py`、`core/envelope.py`、`core/integrity.py` |

**一句话记忆**：family 决定"模型是什么、损失怎么算"；feeding 决定"帧怎么喂、怎么评"。

参照两个现成实现体会差异：
- **GRU**（因果）：`prepare` 空操作；`compute_loss` 只监督**窗口最后一帧**（`gru.py:55`）；配 `windowed_causal`。
- **MSTCN**（非因果）：`prepare` 用训练集拟合 z-score 写入 buffer（`mstcn.py`）；`compute_loss` 监督**全部帧**；配 `full_sequence`。

## 1.2 新接入一个 temporal 模型：要实现什么

### A. 新建 `temporal/family/<你的名字>.py`

单文件自足：网络定义 + 族契约类。族类需实现 **6 个方法 + 1 个类属性**（鸭子类型，无需继承基类）：

```python
class MyFamily:
    family_id = "myfamily"                 # 注册键，对应 cfg["family"]

    def build_network(self, model_cfg: dict) -> nn.Module: ...
        # 从 model_cfg 造网络（读 input_dim / num_classes / 自定义超参）

    def prepare(self, model: nn.Module, features: list) -> None: ...
        # 训练前钩子。不需要就空实现（GRU 即空操作）。
        # features 是训练集全部特征 [T_i, F] 的 list，可在此拟合归一化。

    def forward(self, model, x: torch.Tensor) -> torch.Tensor: ...
        # x:[B,T,F] -> logits:[B,T,C]，通常 return model(x)

    def compute_loss(self, logits, y, criterion) -> torch.Tensor: ...
        # 核心契约：
        #   因果/末帧监督:  criterion(logits[:, -1, :], y)        # y 是标量帧标签
        #   逐帧监督:      criterion(logits.reshape(-1,C), y.reshape(-1))

    def predict_frame_logits(self, model, x: torch.Tensor) -> torch.Tensor: ...
        # 滑窗评估每步调用。x:[1,window,F] -> 末帧 logits [C]

    def checkpoint_meta(self, model_cfg, feature_schema, extra: dict) -> dict: ...
        # 照抄 gru.py:64 样板即可：family/input_dim/num_classes/model/feature_schema + extra
```

> 注意 `compute_loss` 的 `y` 形状由 feeding 决定：`windowed_causal` 给标量（末帧标签），`full_sequence` 给 `[T]`。你的 `compute_loss` 必须和所选 feeding 对齐。

### B. 注册：`temporal/family/__init__.py:13`

```python
from .myfamily import MyFamily
_FAMILIES = { ..., MyFamily.family_id: MyFamily }
```

### C. 选 feeding（一般无需写代码）

- 模型**因果**（只看历史+当前，流式）→ `feeding: windowed_causal`
- 模型**非因果**（看全序列）→ `feeding: full_sequence`
- 只有需要全新喂入/评估协议时，才在 `temporal/feeding/` 加类（实现 `name` / `build_datasets` / `evaluate` / `requires_performance` / `train_batch_size`）并注册。

### D. 写配置 YAML

```yaml
task: temporal
family: myfamily
feeding: windowed_causal
model:
  input_dim: 40
  num_classes: 6
  hidden: 128           # 你的自定义超参
train: { epochs: 20, lr: 1.0e-3, batch_size: 32, window: 64 }
data: { root: /path, split_train: train, split_eval: val }
feature_schema: { dim: 40, version: actionmixed-bbox-8cls-v1 }
```

**不用改**：`orchestration.py`、`loader.py`、`metrics.py`、CLI、`core/*`。

---

# 二、YOLO（单帧检测）

检测纵目前**只接 YOLO**，`YoloAdapter`（[adapter.py:29](../framework/cleansight_eval/detection/adapter.py#L29)）已经封好 ultralytics 的训练/验证，并**兼任 data loader**（从 `data.yaml` 自持读图、组 batch）。feeding 恒为 `single_frame`（[orchestration.py:56](../framework/cleansight_eval/detection/orchestration.py#L56) 强制校验）。

**结论：新增一个 YOLO 模型 = 只写一份配置 YAML，零代码。** 换的只是预训练权重和超参，`family` 始终是 `yolo`，同一个 adapter 复用。

## 2.1 差异点集中在哪几个环节

YOLO 之间的差异**全部由配置表达**，代码路径完全共享：

| 环节 | 变还是共享 | 位置 |
|------|-----------|------|
| 模型规模 / 结构（n/s/m/l 或自定义 arch） | **变（走 `model.weights`）** | `cfg.model.weights` → `adapter.train` |
| 输入分辨率 | 变（走 `model.imgsz`） | `cfg.model.imgsz` |
| 训练超参（epochs/batch/patience） | 变（走 `cfg.train`） | `cfg.train` |
| 数据集 | 变（走 `data.data_yaml`） | `cfg.data.data_yaml` |
| 训练 / 验证实现 | **共享** | `YoloAdapter.train` / `.val` |
| 指标装配（mAP/P/R 逐类三态） | **共享** | `detection/metrics.py:build_detection_metrics` |
| feeding 语义 / 性能延迟(一律 N/A) | **共享** | `orchestration.py:SINGLE_FRAME_SEMANTICS` / `_na_performance` |
| envelope / 完整性 / matrix | **共享** | `core/*` |

## 2.2 新接入一个 YOLO 模型：只写配置

```yaml
task: detection
family: yolo                 # 固定 yolo
feeding: single_frame        # 固定 single_frame
model:
  weights: yolo11s.pt        # 换这里：yolo11n/s/m/l.pt，或自定义结构 .yaml
  imgsz: 640
train: { epochs: 100, batch: 16, patience: 20 }
data:
  data_yaml: /path/data.yaml # cleansight-yolo-pipeline 产出的标准数据集
  name: mydataset            # 产物/run 命名
  eval_split: val
```

- `weights` 传 `.pt` 走微调（从预训练权重继续训）；传结构 `.yaml`（如 `yolo11.yaml`）走从头训。二者都由 ultralytics `YOLO(weights)` 直接消费，无需改代码。
- 训练/评估入口与时序完全一致：`cli/train.py --config xxx.yaml`、`cli/eval.py`。

**要改代码的情况**（暂不做）：接 YOLO 之外的检测器（Faster R-CNN、detectron2 等）才需要新写一个 adapter 类并在 [adapter.py:97](../framework/cleansight_eval/detection/adapter.py#L97) `_ADAPTERS` 注册——需要时再说。

---

# 三、两纵对照速查

| | Temporal | YOLO 检测 |
|--|----------|-----------|
| 新增模型改动 | **写代码**：加 family 文件（6 方法）+ 注册 + 选 feeding | **只写配置**：换 weights/超参 |
| 抽象轴 | 两条：family × feeding | 一条：adapter（已封好，暂不动） |
| data loader | 框架 `loader.py` 统一 | adapter 自持（ultralytics 读 data.yaml） |
| feeding | 可选 windowed_causal / full_sequence | 恒 single_frame |
| 训练主循环 | 框架实现（共享） | ultralytics 自持（adapter 内） |
| 性能延迟 | windowed_causal 实测，full_sequence N/A | 一律 N/A |
| 共享出口 | envelope / metrics / integrity / matrix | 同左 |

> 设计意图：temporal 的 family 与 detection 的 adapter 是**两套不相交契约**，`adapter.py` 顶部注释明确"故意不强行统一"。所以时序接新模型走 family 样板，YOLO 接新模型只调配置。
</content>
