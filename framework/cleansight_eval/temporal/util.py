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


def compute_class_weights(dataloader) -> dict:
    """按类别频率倒数计算并归一化的损失权重（迁移自 util.compute_class_weights）。"""

    counter = Counter()
    for _, y in dataloader:
        # 展平以兼容两种标签形态：末帧标量 [B] 与逐帧全序列 [B, T]。
        counter.update(np.asarray(y.cpu()).reshape(-1))
    total = sum(counter.values())
    weights = {cls: total / count for cls, count in counter.items()}
    max_w = max(weights.values())
    return {k: v / max_w for k, v in weights.items()}
