"""两条时序流水线共用的训练工具（与具体流水线无关）。

这里只放"对滑窗/全序列都成立"的通用件。样本容器（窗口/末帧、整段/逐帧）由各流水线
自持（见 ``sliding_window_pipeline`` / ``full_sequence_pipeline``），不在此列。
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from ..core.metrics import training_metric_keys


def causal_decision(last, pending, stable, count, num_classes: int | None = None,
                    min_duration: int = 25):
    """滑窗推理的因果平滑：转移先验 + 最小持续时长。

    这是推理后处理而不是评估指标。仅在三分类时应用类别转移先验；其他类别数退化为
    最小持续时长平滑。``num_classes`` 保留用于兼容历史调用。

    ``min_duration`` 是最小持续时长（帧）：候选类必须连续出现这么多帧才会切换 ``stable``。
    它同时是**召回上限**——真实段短于该值的动作在输出里不可能出现（2026-09-18 实测：
    project-18 test 的 short_brush_cleaning 六段全部 ≤19 帧，min_duration=25 下召回恒为 0）。
    默认 25 保持历史行为；实验配置 ``evaluation.smoothing_min_duration`` 可覆盖。
    """

    prob = torch.softmax(last, dim=-1).cpu().numpy()
    classes = len(prob)
    transition_prior = np.zeros((classes, classes))
    if classes == 3:
        idle_id, long_id, short_id = 0, 1, 2
        transition_prior[idle_id, idle_id] = 2.0
        transition_prior[long_id, long_id] = 2.0
        transition_prior[short_id, short_id] = 1.5
        transition_prior[long_id, short_id] = -1.0
        transition_prior[short_id, long_id] = -1.0

    scores = np.zeros(classes)
    for index in range(classes):
        scores[index] = np.log(prob[index] + 1e-8) + transition_prior[stable, index]
    candidate = int(np.argmax(scores))

    if candidate == pending:
        count += 1
    else:
        pending = candidate
        count = 1
    if count >= max(1, int(min_duration)):
        stable = pending if pending is not None else 0
    return pending, stable, count


# best checkpoint 可选指标（validation 字典键）**由指标注册表派生**，不再手写枚举：注册表里
# 声明了 ``training_key`` 的指标都可用于选点，因此训练选点口径与 ``benchmark.cli.eval`` 报出的
# 同名指标一一对应（val_f1_0.25 ↔ f1@0.25 等）。val_acc 对多数类友好、会偏爱 idle 坍缩解
# （2026-09 诊断），段级指标（edit/F1）更能代表动作质量；滑窗/全序列共用。
VALID_BEST_METRICS = frozenset(training_metric_keys())


# 训练选点**默认口径的唯一事实源**：两条训练管线、矩阵工具与实验 YAML 都必须指回这里。
# 定版依据（docs/FEATURE_STRATEGY_COMPARE.md §23.1 / §25.1 第 4 行 / §26.2）：326 个 run 实测
# 旧默认 ``val_f1_0.5`` 与 test 的 Spearman ρ 仅 **0.199**（按它选点近乎随机挑一轮），
# ``val_acc`` 的峰值常落在第 1 轮（idle 坍缩早峰）；换 ``val_edit`` 实测 edit **+9.63**
# （p=0.0107）、insert 召回 **+5.46**（p=0.0214，8 seed），且已在三组独立数据上同向复现。
# 历史口径：2026-09-23 之前 YAML 写死 ``val_f1_0.5``、代码兜底 ``val_acc`` —— 同一份实验
# 换个入口跑会存下不同的 best.pt，故收敛为单一常量。
DEFAULT_BEST_METRIC = "val_edit"


def resolve_best_metric(train_cfg) -> str:
    """解析 ``train.best_metric``：缺省用 :data:`DEFAULT_BEST_METRIC`，非法值立即报错。

    :param train_cfg: 训练配置块（``cfg["train"]``），可为 None 或空映射。
    :return: 已校验的选点指标名（``VALID_BEST_METRICS`` 之一）。
    :raises ValueError: 取值不在指标注册表声明的选点词表内。
    """

    raw = (train_cfg or {}).get("best_metric")
    if raw is None:
        return DEFAULT_BEST_METRIC
    if raw not in VALID_BEST_METRICS:
        raise ValueError(
            f"train.best_metric 必须是 {sorted(VALID_BEST_METRICS)} 之一，实际 {raw!r}"
        )
    return str(raw)


# 类别权重截断区间：归一化后夹取到 [LOWER, UPPER]。
# 诊断依据（2026-09）：idle 权重被频率倒数归一化压到 ~0.032，极端权重把优化焦点
# 全压在小类上、加速小类"记忆化"，是训练坍缩的推手之一；截断下限保留多数类的基本
# 梯度信号，上限防止单类权重失衡。
CLASS_WEIGHT_CLIP = (0.1, 5.0)


def resolve_class_weight_clip(raw) -> tuple[float, float]:
    """解析 ``train.class_weight_clip``：缺省用 :data:`CLASS_WEIGHT_CLIP`。

    接受 ``[lo, hi]`` 或 ``"lo,hi"`` 字符串（便于 CLI ``-S`` 传参）。下限越低，多数类
    （idle）的梯度信号越弱、模型越"敢说"非 idle——这是与特征侧"删通道"等价的**活动量旋钮**，
    用于对照实验；非法值立即报错，不静默退回默认。
    """

    if raw is None or raw == "":
        return CLASS_WEIGHT_CLIP
    if isinstance(raw, str):
        parts: list = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        raise ValueError(f"train.class_weight_clip 需为 [lo, hi] 或 \"lo,hi\"，实际 {raw!r}")
    if len(parts) != 2:
        raise ValueError(f"train.class_weight_clip 需恰好两个数，实际 {raw!r}")
    try:
        lower, upper = float(parts[0]), float(parts[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"train.class_weight_clip 必须是数字，实际 {raw!r}") from exc
    if not 0.0 < lower <= upper:
        raise ValueError(f"train.class_weight_clip 需满足 0 < lo <= hi，实际 ({lower}, {upper})")
    return (lower, upper)


def compute_class_weights(dataloader, num_classes: int | None = None, clip=None) -> dict:
    """按类别频率倒数计算并归一化的损失权重（迁移自 util.compute_class_weights）。

    归一化后按 ``clip``（缺省 :data:`CLASS_WEIGHT_CLIP` = [0.1, 5.0]）截断，避免极端
    多数/少数类权重失衡；``clip`` 可由 ``train.class_weight_clip`` 覆盖以做活动量对照。
    ``num_classes`` 非空时补全未出现类别（权重 0），避免缺类数据构造 CrossEntropyLoss
    时类别数不匹配。
    """

    counter = Counter()
    for _, y in dataloader:
        # 展平以兼容两种标签形态：末帧标量 [B] 与逐帧全序列 [B, T]。
        counter.update(np.asarray(y.cpu()).reshape(-1))
    total = sum(counter.values())
    weights = {cls: total / count for cls, count in counter.items()}
    max_w = max(weights.values())
    lower, upper = resolve_class_weight_clip(clip)
    normalized = {
        k: min(max(v / max_w, lower), upper) for k, v in weights.items()
    }
    if num_classes is not None:
        for cls in range(num_classes):
            normalized.setdefault(cls, 0.0)
    return normalized


# ---- 转移级代价敏感（transition-level cost-sensitive）----------------------------

def estimate_transition_rarity(labels_list, num_classes: int) -> np.ndarray:
    """从**训练集真值**统计类间转移频率，返回稀有度表 ``[C, C]``（取值 0~1）。

    定义 ``rarity[a, b] = 1 − log1p(counts[a, b]) / log1p(counts.max())``：

    - 高频转移（含 ``a == b`` 的段内转移，出现次数最多）→ 接近 **0**；
    - 从未出现的转移 → **1**；
    - 于是"罕见的类间转移"（如本项目 `insert` → `withdraw` 这类**逆转移对**）拿到高权重。

    这是文献（Cost-Sensitive Learning for Long-Tailed TAS, arXiv 2503.18358）所指的
    **transition-level** 代价敏感——该文同一张表里 class-level 重加权（CB / LA / Focal /
    τ-norm）增益 ≈0，而 transition-aware 拿到 +8.1 F1@25；两者作用在不同维度上。
    """

    counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    for labels in labels_list:
        arr = np.asarray(labels).reshape(-1)
        if arr.size < 2:
            continue
        for a, b in zip(arr[:-1], arr[1:]):
            counts[int(a), int(b)] += 1.0
    peak = counts.max()
    if peak <= 0:
        return np.zeros((num_classes, num_classes), dtype=np.float64)
    return 1.0 - np.log1p(counts) / np.log1p(peak)


class TransitionWeightedCE(torch.nn.Module):
    """类别加权 CE × **转移稀缺度**逐帧加权（可直接当作 ``criterion`` 传给模型）。

    逐帧总权重 = **类别权重 ×** 转移因子 ``(1 + strength × rarity[y_{t−1}, y_t])``，
    再按 **权重之和** 归一化（``Σ w·ce / Σ w``）——与 PyTorch ``CrossEntropyLoss(weight=…)``
    的 ``reduction='mean'`` 语义**逐位一致**。

    ⚠ **这一点必须小心**：``F.cross_entropy(weight=w, reduction='mean')`` 的分母是
    **权重之和**，而 ``reduction='none'`` 后再 ``.mean()`` 是**算术平均**，两者相差
    ``N / Σw``（本类第一版就写成后者，实测差 2 倍）。若不按权重之和归一，"开启旋钮"本身
    就会改变损失尺度，使"损失变大"与"优化变好"无法区分，污染与该旋钮无关的对照。
    已由 ``framework/tests/test_transition_loss.py`` 的零强度等价性测试钉住。

    首帧的 ``y_{t−1}`` 取自身。

    **做成 criterion 而非改模型**：逐帧权重只依赖真值 ``y``、不依赖模型输出，因此可以完全
    封装在 criterion 里；流水线把它传给模型后，连 ``mstcn2`` 那种自持配方
    （多 stage 深监督 + T-MSE）也会自动带上转移加权，无需改动任何模型文件。

    **前提与保护**：全序列流水线固定 ``batch_size=1``，故扁平后的 ``y`` 就是单条序列，
    相邻元素即真实转移、不存在跨序列伪转移，故 ``single_sequence`` 默认 ``True``。
    若将来改成分批训练，必须传 ``single_sequence=False``（退化为普通加权 CE），
    否则会在每条序列边界处引入 ``B−1`` 个错误转移。
    """

    def __init__(self, class_weights: torch.Tensor, rarity: np.ndarray,
                 strength: float, single_sequence: bool = True):
        super().__init__()
        if strength < 0:
            raise ValueError(f"transition strength 必须 ≥0，实际 {strength}")
        self.register_buffer("class_weights", class_weights.detach().clone().float())
        # rarity 必须与 class_weights **同设备**：流水线把类别权重张量直接放到 device，
        # 而 rarity 由 numpy 生成（默认 CPU）；不同设备会在 forward 的索引处报
        # "indices should be either on cpu or on the same device as the indexed tensor"。
        self.register_buffer(
            "rarity",
            torch.as_tensor(rarity, dtype=torch.float32, device=self.class_weights.device),
        )
        self.strength = float(strength)
        self.single_sequence = bool(single_sequence)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 不带 weight 的逐帧 CE，权重统一在下面合成，保证归一化口径只有一个。
        ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        weight = self.class_weights[target]
        if self.strength > 0 and self.single_sequence and ce.numel() >= 2:
            previous = torch.cat([target[:1], target[:-1]])
            weight = weight * (1.0 + self.strength * self.rarity[previous, target])
        total = weight.sum()
        if not torch.isfinite(total) or float(total) <= 0:
            # 全部权重为 0（例如所有目标类权重都是 0）时退回算术平均，避免除以 0。
            return ce.mean()
        return (ce * weight).sum() / total
