"""喂入模式层接口（train/eval 中立的共享轴，需求 §5.2 / §6.2 / §8.3）。

喂入模式描述"帧怎么按时间打包给模型"：窗口长度、因果性、状态/reset、读/监督哪一帧。
它是**训练与评估共享**的一条轴——训练选一个模式并在其下优化，评估选一组模式分别打分；
二者唯一的不同是选定模式之后的外层机器（反向传播 vs 打分出信封），与喂入模式正交。

full_sequence（窗口→∞）、windowed_causal（有界因果窗、读末帧）、single_frame（窗口=1
无状态）都是这条轴上的取值。评估必须反映真实喂入语义，不得用全序列结果冒充实时结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class FeedingResult:
    """一个喂入模式在测试集上的原始输出（评估侧）。"""

    video_preds: list[np.ndarray]  # 每个视频逐帧预测 id
    video_gts: list[np.ndarray]  # 每个视频逐帧真值 id
    semantics: dict = field(default_factory=dict)  # 窗口/reset/冷启动等语义描述


class FeedingMode(Protocol):
    name: str
    requires_performance: bool  # 该模式是否需要性能测量（§8.4）

    def evaluate(self, family, model, datasets, device) -> FeedingResult:
        """评估侧：在若干 per-video 数据集上按本喂入模式推理，返回逐帧预测与真值。"""
        ...

    # 训练侧（可选）：把逐视频序列按本喂入模式打包成训练数据集，作为"窗口/末帧"规格的
    # 单一真源。仅可用于训练的喂入模式实现（如 windowed_causal）；full_sequence /
    # single_frame 不实现——训练由各自任务另行组织。
    # def build_training_dataset(self, features, truths, idx, window): ...
