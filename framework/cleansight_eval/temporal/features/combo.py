"""ROI 手工先验 + 原始框集合的**拼接**契约（actionmixed-combo-v1，224 维）。

**要回答的问题**：第 8 轮的"学习式集合编码器"（`boxset_mstcn2`）在 test 上只有 **45.02**
（比 `roi-grid-v4` 的 57.60 低 12.6pp），说明**在 14 个训练视频的规模下，让模型从零学
"每类在哪儿、有几个、多大"这件事，打不过手工 ROI 直方图给的先验**。

但那条实验**没有回答另一个问题**：原始框信息里**是否有 ROI 直方图没表达出来的增量**？
本契约把两者**拼接**喂给同一个 `mstcn2` 主干来检验：

- 左段 `roi-grid-v4`（128 维）：(类, 区域) 的 `[count, max_area, w, h]`——**强先验**；
- 右段 `boxset`（96 维）：每帧面积最大的前 8 个框的 `[one-hot | cx | cy | w | h]`
  ——**框间关系与精确坐标**，槽按面积降序（确定性，故顺序信息可用）。

两段共用同一份 `frames/` 数据、同一批 split，故与 v4 构成**单变量对照**（只多出右段 96 维）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .boxset import build_boxset_frame_features
from .roi_bbox_v4 import build_roi_grid_v4_frame_features, ROI_GRID_V4_DIM

COMBO_VERSION = "actionmixed-combo-v1"
COMBO_ROI_DIM = ROI_GRID_V4_DIM          # 128
COMBO_BOXSET_DIM = 96                     # 8 槽 × 12
COMBO_DIM = COMBO_ROI_DIM + COMBO_BOXSET_DIM  # = 224


def build_combo_frame_features(
    txt_path: Path,
    n_classes: int = 8,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """一帧 bbox → ``[224]``：左 128 维 ROI-4 通道特征 + 右 96 维框集合填充张量。

    :param txt_path: 该帧的 YOLO 检测框文本；文件不存在视为空帧（全零）。
    :param n_classes: 检测类别数（ROI 段与 one-hot 段都按 8 类定义）。
    :param mask_target_ids: 只作用于**左段**（右段是原始框，不做类级遮罩）。
    :return: ``float32`` 一维向量，长度 :data:`COMBO_DIM`。
    """

    roi = build_roi_grid_v4_frame_features(txt_path, n_classes=n_classes,
                                          mask_target_ids=mask_target_ids)
    box = build_boxset_frame_features(txt_path, n_classes=n_classes)
    return np.concatenate([roi, box]).astype(np.float32)
