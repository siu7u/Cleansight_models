"""ActionMixed bbox 帧 → **原始框集合**的定长填充张量（actionmixed-boxset-v1，96 维）。

**为什么做这个契约（2026-09-29 第 8 轮）**：前 7 轮一直在**手工 ROI 直方图**里腾挪
（网格分辨率、通道语义、维度分配）。这条路线已饱和：
`roi-grid-v4`（通道语义替换）把单 seed 从 53.92 抬到 **57.60**，但
网格更细（4×4）−4.62pp、格内亚格位置（v5）无增益、容量/预算/集成/选点全部无增益。
手工直方图的一个**结构性损失**是：它把每帧的框压成"(类, 区域) 的计数与极值"，
**丢掉了框与框之间的关系**（哪些类同时出现、谁大谁小、谁在谁旁边）。
本契约改为把**每帧最多 K 个框的原始属性**按面积降序填充成定长张量，
由模型端的**可学习集合编码器**（`models/boxset.py`）自行学跨类关系——
这是与"再调一次 ROI 参数"定性不同的一次尝试。

**布局**（K=8 槽 × 12 维 = **96 维**）：每槽 ``[one-hot(class, 8) | cx | cy | w | h]``。
- 框按面积（``w×h``）降序取前 K 个；不足 K 个则**零填充**；
- **填充判据是内建的**：空槽的 one-hot 全零 → 模型用它算掩码，无需额外通道；
- 坐标是 YOLO 归一化坐标（0~1），不做任何缩放或标准化（由模型端 Linear 学）。

**语义边界（如实记录）**：只保留**面积最大的前 8 个框**——auto 数据每帧平均 3.5 个框，
所以绝大多数帧无损；被丢弃的是极少数帧里的第 9 个及以后的框。
因果、无状态、逐帧独立；空帧全零。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BOXSET_VERSION = "actionmixed-boxset-v1"
BOXSET_N_CLASSES = 8          # 检测类数（one-hot 宽度，与 frames/data.yaml 一致）
BOXSET_N_SLOTS = 8            # 每帧保留的框数上限
BOXSET_SLOT_DIM = BOXSET_N_CLASSES + 4   # one-hot + cx, cy, w, h = 12
BOXSET_DIM = BOXSET_N_SLOTS * BOXSET_SLOT_DIM  # = 96


def build_boxset_frame_features(
    txt_path: Path,
    n_classes: int = BOXSET_N_CLASSES,
    n_slots: int = BOXSET_N_SLOTS,
) -> np.ndarray:
    """一帧 bbox → ``[96]`` 填充张量（按面积降序的前 ``n_slots`` 个框）。

    :param txt_path: 该帧的 YOLO 检测框文本（每行 ``class cx cy w h``，归一化坐标）；
        文件不存在视为空帧（全零）。
    :param n_classes: 检测类别数（必须等于 one-hot 宽度 8）。
    :param n_slots: 槽位数（默认 8）。
    :return: ``float32`` 一维向量，长度 ``n_slots × (n_classes + 4)``；
        槽内布局 ``[one-hot | cx | cy | w | h]``，空槽全零。
    """

    if n_classes != BOXSET_N_CLASSES:
        raise ValueError(
            f"{BOXSET_VERSION} 的 one-hot 宽度固定为 {BOXSET_N_CLASSES} 类，实际数据集类别数={n_classes}"
        )
    feat = np.zeros((n_slots, n_classes + 4), dtype=np.float32)
    if not txt_path.exists():
        return feat.reshape(-1)
    rows = []
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c = int(float(parts[0]))
        if not (0 <= c < n_classes):
            continue
        cx, cy, w, h = (float(v) for v in parts[1:])
        rows.append((w * h, c, cx, cy, w, h))
    # 面积降序；面积相同时按类 id 稳定排序，保证同 seed 可复现
    rows.sort(key=lambda r: (-r[0], r[1]))
    for slot, (_area, c, cx, cy, w, h) in enumerate(rows[:n_slots]):
        feat[slot, c] = 1.0
        feat[slot, n_classes:n_classes + 4] = (cx, cy, w, h)
    return feat.reshape(-1)
