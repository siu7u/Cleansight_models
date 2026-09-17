"""v3 ⊕ roi 拼接契约：clean_bbox_v3（113 维）与 ROI 网格（144 维）逐帧拼接 = 257 维。

见 docs/FEATURE_LAB_NEXT_STEPS.md §1.1-#2：v3 段级 edit 强（31.07）、roi 边界精度持平
且 frame mIoU 高一档，验证两者是否互补。两侧均为因果、无状态特征；v3 侧的跨帧
回退链由 build_clean_bbox_v3_features 在序列内前向填充（与单独使用时语义一致）。

- `actionmixed-cleanv3-roi-concat-v1`：113 + 144 = **257 维**
- 布局：左 113 维 = clean_bbox_v3 块结构，右 144 维 = roi 网格（类优先 × 6 区域 × 3 通道）
- 本契约为 exploratory 探针（无 catalog 登记，配置用显式 data.root）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .clean_bbox_v3 import build_clean_bbox_v3_features
from .roi_bbox import build_roi_frame_features

CLEAN_V3_ROI_VERSION = "actionmixed-cleanv3-roi-concat-v1"
CLEAN_V3_ROI_DIM = 257


def build_clean_v3_roi_features(
    frame_paths: list[Path],
    *,
    detection_mapping: dict,
    fps: float,
    confidence_default: float,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """逐视频序列 [T, 257]：v3 序列特征（含回退链）与逐帧 ROI 网格沿特征维拼接。"""

    v3_feats, _names, actual_version = build_clean_bbox_v3_features(
        frame_paths,
        detection_mapping=detection_mapping,
        fps=fps,
        confidence_default=confidence_default,
        mask_target_ids=mask_target_ids,
    )
    if v3_feats.shape[1] != 113:
        raise ValueError(f"v3 侧维度 {v3_feats.shape[1]} != 113")
    roi_feats = np.stack(
        [
            build_roi_frame_features(path, mask_target_ids=mask_target_ids)
            for path in frame_paths
        ]
    ).astype(np.float32)
    if roi_feats.shape[1] != 144:
        raise ValueError(f"roi 侧维度 {roi_feats.shape[1]} != 144")
    if len(roi_feats) != len(v3_feats):
        raise ValueError(
            f"v3 帧数 {len(v3_feats)} 与 roi 帧数 {len(roi_feats)} 不对齐"
        )
    del actual_version
    return np.concatenate([v3_feats, roi_feats], axis=1).astype(np.float32)
