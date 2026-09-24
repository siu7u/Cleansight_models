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
