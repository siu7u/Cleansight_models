"""两条时序流水线共用的训练工具（与具体流水线无关）。

这里只放"对滑窗/全序列都成立"的通用件。样本容器（窗口/末帧、整段/逐帧）由各流水线
自持（见 ``sliding_window_pipeline`` / ``full_sequence_pipeline``），不在此列。
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch


def causal_decision(last, pending, stable, count, num_classes: int | None = None):
    """滑窗推理的因果平滑：转移先验 + 最小持续时长。

    这是推理后处理而不是评估指标。仅在三分类时应用类别转移先验；其他类别数退化为
    最小持续时长平滑。``num_classes`` 保留用于兼容历史调用。
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

    min_duration = 25
    if candidate == pending:
        count += 1
    else:
        pending = candidate
        count = 1
    if count >= min_duration:
        stable = pending if pending is not None else 0
    return pending, stable, count


# best checkpoint 可选指标（validation 字典键）。val_acc 对多数类友好、会偏爱 idle
# 坍缩解（2026-09 诊断），段级指标（edit/F1）更能代表动作质量；滑窗/全序列共用。
VALID_BEST_METRICS = frozenset({"val_acc", "val_edit", "val_f1_0.5"})


# 类别权重截断区间：归一化后夹取到 [LOWER, UPPER]。
# 诊断依据（2026-09）：idle 权重被频率倒数归一化压到 ~0.032，极端权重把优化焦点
# 全压在小类上、加速小类"记忆化"，是训练坍缩的推手之一；截断下限保留多数类的基本
# 梯度信号，上限防止单类权重失衡。
CLASS_WEIGHT_CLIP = (0.1, 5.0)


def compute_class_weights(dataloader, num_classes: int | None = None) -> dict:
    """按类别频率倒数计算并归一化的损失权重（迁移自 util.compute_class_weights）。

    归一化后按 ``CLASS_WEIGHT_CLIP`` 截断（默认 [0.1, 5.0]），避免极端多数/少数类
    权重失衡。``num_classes`` 非空时补全未出现类别（权重 0），避免缺类数据构造
    CrossEntropyLoss 时类别数不匹配。
    """

    counter = Counter()
    for _, y in dataloader:
        # 展平以兼容两种标签形态：末帧标量 [B] 与逐帧全序列 [B, T]。
        counter.update(np.asarray(y.cpu()).reshape(-1))
    total = sum(counter.values())
    weights = {cls: total / count for cls, count in counter.items()}
    max_w = max(weights.values())
    lower, upper = CLASS_WEIGHT_CLIP
    normalized = {
        k: min(max(v / max_w, lower), upper) for k, v in weights.items()
    }
    if num_classes is not None:
        for cls in range(num_classes):
            normalized.setdefault(cls, 0.0)
    return normalized
