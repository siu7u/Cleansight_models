"""时序任务的滑窗数据结构（任务层，与数据来源无关）。

``EndoDataset`` / ``build_dataset`` 是"窗口容器"，由喂入模式（windowed_causal）用来把
逐视频序列 ``[T, F]`` 打包成窗口样本；具体怎么从原始数据集读出 ``[T, F]`` 特征交由
``loader.py``（features 契约）负责。
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


class EndoDataset(Dataset):
    """因果滑窗数据集：样本为 ``[window, F]``，标签为窗口最后一帧类别。"""

    def __init__(self, features: np.ndarray, labels, window: int = 64):
        self.x = torch.from_numpy(features).float()
        self.y = torch.tensor(labels, dtype=torch.long)
        self.w = window

    def __len__(self):
        return len(self.x) - self.w + 1

    def __getitem__(self, idx):
        x = self.x[idx : idx + self.w]
        y = self.y[idx + self.w - 1]
        return x, y


def build_dataset(features: list[np.ndarray], labels: list[list[int]], idx: list[int], window: int = 64):
    return ConcatDataset([EndoDataset(features[i], labels[i], window) for i in idx])


def compute_class_weights(dataloader) -> dict:
    """按类别频率倒数计算并归一化的损失权重（迁移自 util.compute_class_weights）。"""

    counter = Counter()
    for _, y in dataloader:
        counter.update(y.cpu().numpy())
    total = sum(counter.values())
    weights = {cls: total / count for cls, count in counter.items()}
    max_w = max(weights.values())
    return {k: v / max_w for k, v in weights.items()}
