"""时序纵的跨模式训练工具（与具体喂入模式无关）。

这里只放"对任意喂入模式都成立"的通用件。样本容器（窗口/末帧、整段/逐帧）由各
喂入模式自持（见 ``feeding/*``），不在此列。
"""

from __future__ import annotations

from collections import Counter

import numpy as np


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
