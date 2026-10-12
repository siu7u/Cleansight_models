"""ROI 网格 v5 + 「全局 + 手部」双通道 bbox 的**拼接**契约（actionmixed-combo-v2，272 维）。

**要回答的问题（v4.1 数据换代后的第一条候选）**：v4.1 全量重训（22 臂 × 8 seed）给出的
契约排名与 v3 时代**完全不同**（Spearman ρ=−0.149），且**没有一个契约全能**：

| 契约 | acc（8 seed 中位） | edit | 备注 |
|---|---:|---:|---|
| `roi-grid-v4`（v3 冠军、旗舰契约） | 64.91 | 62.64 | v4.1 上垫底 |
| **`bbox-global-hand`** | **76.27** | 64.14 | acc 最高（全画面框 + 手部区域相对坐标） |
| `roi-grid-v5` | 74.53 | **69.43** | edit / macro_f1 最高，四类召回都 ≥0.55 |
| `combo-v1`（v4 ++ boxset） | 70.48 | 61.89 | 拼接**高于**其父契约 `roi-grid-v4` +5.57pp |

两个观察把本契约推成第一候选：① `combo-v1` 在 v4.1 上比父契约高 **+5.57pp**
（v3 时代"拼接无增量"的结论未迁移）；② 在已存盘预测产物上做**跨契约多数投票**，
把 `roi-v5` 与 `global-hand` 一类成员投到一起能同时拿到两者的长处，说明两类信息互补。
本契约把这种互补**固化进单模型**：一个 mstcn2 主干同时看两段特征。

**布局（顺序即拼接顺序，与 `feature_parts` 一致）**：

- 左段 `roi-grid-v5`（192 维）：(类, 区域) 的 `[count, max_area, w, h, dx, dy]`——网格空间先验；
- 右段 `bbox-global-hand`（80 维）：40 维全画面「每类最大框 `[presence, cx, cy, w, h]`」
  ++ 40 维手部区域（最大 hand 框绕中心扩张 1.5×）内的同类量——绝对坐标 + 相对手的位置。

两段共用同一份 `frames/` 数据与同一批 split，故与 `roi-grid-v5` / `bbox-global-hand`
均构成**单变量对照**（只多出另一段）。因果、无状态、逐帧独立计算；改任一分段的布局或
通道语义 = 新 feature mapping 版本。

实现说明：三段分别由 `features/roi_bbox_v5.py`（ROI 段）、`temporal/data.py`
（`featurize_frame_bbox` 全画面段）与 `features/hand_bbox.py`（手部段）产出，
故本模块只声明版本/维度并做**拼接与维度自检**，逐帧取数的分派在 `data.py`
（与既有 `actionmixed-bbox-global-hand-8cls-v1` 契约同一模式）。
"""

from __future__ import annotations

import numpy as np

from .roi_bbox_v5 import ROI_GRID_V5_DIM

COMBO_V2_VERSION = "actionmixed-combo-v2"
COMBO_V2_GLOBAL_DIM = 40                                        # 全画面：8 类 × 5 维
COMBO_V2_HAND_DIM = 40                                          # 手部区域：8 类 × 5 维
COMBO_V2_ROI_DIM = ROI_GRID_V5_DIM                              # 192
COMBO_V2_PARTS: tuple[int, ...] = (COMBO_V2_ROI_DIM,
                                   COMBO_V2_GLOBAL_DIM,
                                   COMBO_V2_HAND_DIM)
COMBO_V2_DIM = sum(COMBO_V2_PARTS)                              # = 272


def build_combo_v2_frame_features(
    roi: np.ndarray,
    global_bbox: np.ndarray,
    hand: np.ndarray,
) -> np.ndarray:
    """把三段逐帧特征按 ``[roi | 全局 | 手部]`` 拼成一帧 ``[272]``。

    :param roi: `roi-grid-v5` 段，形状 ``[192]``。
    :param global_bbox: 全画面 bbox 段，形状 ``[40]``。
    :param hand: 手部区域段，形状 ``[40]``。
    :return: ``float32`` 一维向量，长度 :data:`COMBO_V2_DIM`。
    :raises ValueError: 任一段宽度与契约声明不符（防静默回落到错宽特征）。
    """

    parts = (np.asarray(roi), np.asarray(global_bbox), np.asarray(hand))
    widths = tuple(int(part.shape[-1]) for part in parts)
    if widths != COMBO_V2_PARTS:
        raise ValueError(
            f"{COMBO_V2_VERSION} 分段宽度不符：实际 {widths} != 契约 {COMBO_V2_PARTS}"
        )
    return np.concatenate(parts).astype(np.float32)
