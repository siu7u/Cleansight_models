"""ROI 网格 v5 与**手部区域 v2** 的拼接族（actionmixed-combo-v3 / v4）。

**要回答的问题**：v4.1 上两个契约各有一块别人拿不到的东西——

- `actionmixed-roi-grid-v5`（192 维）是**唯一能召回 `short_brush_cleaning` 的契约**（56.3%），
  因为它按类统计全画面的 `count / max_area / w / h / dx / dy`（含低频器械类）；
- `actionmixed-bbox-hand-v2` / `-global-hand-v2`（40 / 80 维，**并集锚点**）是帧准确率与
  `flush` / `insert` / `withdraw` 召回的最强驱动（test acc 77.75 / 79.15），但
  **`sbc` 召回恒为 0**（`hand-v2`）/ 28.3（`global-hand-v2`）。

于是把两者拼进一个主干，检验能否同时拿到 `acc` 与 `sbc`。

**两个版本**（同为"roi-v5 ‖ 手部 v2"，差别只在右段）：

| 契约版本 | 维度 | 分段 | 对照问题 |
|---|---:|---|---|
| `actionmixed-combo-v3` | 232 | `roi-grid-v5`(192) ‖ `bbox-hand-v2`(40) | 最省的组合：把 acc 冠军（40 维）加进 edit 冠军 |
| `actionmixed-combo-v4` | 272 | `roi-grid-v5`(192) ‖ `bbox-global-hand-v2`(80) | **与 round-1 的 `combo-v2` 同结构同维度**，唯一变量 = 手部段 v1 → v2 |

**不要期待它靠"更多检测框"补 `sbc`**：源域实测（2026-10-11，train 的 1,374 个 sbc 帧）
——有刷类（`short_brush` / `brush_tip_out`）框的帧只占 **17.2%**，而这些框**100% 已经落在
并集手部区域内**（扩 2.0 倍）。也就是说 `sbc` 的输入侧证据是**检测缺失**，不是"区域没圈到"；
把器械并进锚点区域带不来新信息（同批筛选中 `union_tools` / `union_brush` 锚点无增益）。
因此本契约能补 `sbc` 的唯一途径是**让模型用 ROI 的全局上下文**去学，而不是多几个框。

两段共用同一份 `frames/` 数据与同一批 split，故与三个父契约（`roi-v5` / `hand-v2` /
`global-hand-v2`）均构成单变量对照。因果、无状态、逐帧独立。
"""

from __future__ import annotations

import numpy as np

from .hand_bbox_v2 import (
    GLOBAL_HAND_V2_EXPAND,
    HAND_V2_EXPAND,
    build_hand_v2_frame_features,
)
from .roi_bbox_v5 import ROI_GRID_V5_DIM, build_roi_grid_v5_frame_features

COMBO_V3_VERSION = "actionmixed-combo-v3"
COMBO_V4_VERSION = "actionmixed-combo-v4"
COMBO_V3_PARTS: tuple[int, ...] = (ROI_GRID_V5_DIM, 40)        # roi-v5 ‖ hand-v2
COMBO_V4_PARTS: tuple[int, ...] = (ROI_GRID_V5_DIM, 40, 40)    # roi-v5 ‖ 全画面 ‖ hand-v2
COMBO_V3_DIM = sum(COMBO_V3_PARTS)                             # = 232
COMBO_V4_DIM = sum(COMBO_V4_PARTS)                             # = 272


def _check(parts: tuple[int, ...], expected: tuple[int, ...], version: str) -> None:
    if parts != expected:
        raise ValueError(f"{version} 分段宽度不符：实际 {parts} != 契约 {expected}")


def build_combo_v3_frame_features(
    txt_path,
    n_classes: int = 8,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """一帧 bbox → ``[232]``：``roi-grid-v5``(192) ‖ ``bbox-hand-v2``(40, 并集锚点、扩张 2.0)。

    :param txt_path: 该帧的 YOLO 检测框文本；文件不存在视为空帧（两段各自按空帧处理）。
    :param n_classes: 检测类别数（两段都按 8 类定义）。
    :param mask_target_ids: 类级遮罩，作用于两段（命中的类整块为零）。
    :return: ``float32`` 一维向量，长度 :data:`COMBO_V3_DIM`。
    """

    roi = build_roi_grid_v5_frame_features(txt_path, n_classes=n_classes,
                                           mask_target_ids=mask_target_ids)
    hand = build_hand_v2_frame_features(txt_path, n_classes=n_classes,
                                        mask_target_ids=mask_target_ids, expand=HAND_V2_EXPAND)
    _check((roi.shape[-1], hand.shape[-1]), COMBO_V3_PARTS, COMBO_V3_VERSION)
    return np.concatenate([roi, hand]).astype(np.float32)


def build_combo_v4_frame_features(
    global_block: np.ndarray,
    roi: np.ndarray,
    hand: np.ndarray,
) -> np.ndarray:
    """把 ``[roi-grid-v5(192) | 全画面 bbox(40) | bbox-hand-v2(40)]`` 拼成一帧 ``[272]``。

    全画面段由 `data.py` 的 ``featurize_frame_bbox`` 产出（与 `global-hand` 契约同一实现），
    故这里只做拼接与宽度自检——与 `combo_v2` 的分工相同。

    :param global_block: 全画面段，形状 ``[40]``。
    :param roi: ``roi-grid-v5`` 段，形状 ``[192]``。
    :param hand: ``bbox-hand-v2`` 段，形状 ``[40]``（并集锚点、扩张 1.5）。
    :return: ``float32`` 一维向量，长度 :data:`COMBO_V4_DIM`。
    """

    parts = (np.asarray(roi), np.asarray(global_block), np.asarray(hand))
    widths = tuple(int(part.shape[-1]) for part in parts)
    _check(widths, COMBO_V4_PARTS, COMBO_V4_VERSION)
    return np.concatenate(parts).astype(np.float32)


__all__ = [
    "COMBO_V3_DIM", "COMBO_V3_PARTS", "COMBO_V3_VERSION", "build_combo_v3_frame_features",
    "COMBO_V4_DIM", "COMBO_V4_PARTS", "COMBO_V4_VERSION", "build_combo_v4_frame_features",
    "GLOBAL_HAND_V2_EXPAND", "HAND_V2_EXPAND",
]
