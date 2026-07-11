"""喂入结果类型（temporal 纵专属）。

喂入模式描述"帧怎么按时间打包给模型"：窗口长度、因果性、状态/reset、读/监督哪一帧。
它是**训练与评估共享**的一条轴——训练选一个模式并在其下优化，评估在同模式下打分；
二者唯一的不同是选定模式之后的外层机器（反向传播 vs 打分出信封），与喂入模式正交。

``FeedingResult`` 承载评估侧的原始逐帧输出。此前存在的 ``FeedingMode`` 跨域 Protocol
已删除：三个具体喂入类同处 temporal 纵、由 ``family.test``/orchestration 直接调用，
靠同名方法约定即可，不需要 Protocol；detection 的单帧语义不再借道这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeedingResult:
    """一个喂入模式在测试集上的原始输出（评估侧）。"""

    video_preds: list[np.ndarray]  # 每个视频逐帧预测 id
    video_gts: list[np.ndarray]  # 每个视频逐帧真值 id
    semantics: dict = field(default_factory=dict)  # 窗口/reset/冷启动等语义描述
