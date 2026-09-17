"""v2 ⊕ v3 拼接契约：clean_bbox_v2（113 维绝对坐标）与 clean_bbox_v3（113 维 scope
相对坐标系）逐帧拼接 = 226 维。

见 docs/FEATURE_LAB_NEXT_STEPS.md §1.1-#2b 与 docs/FEATURE_LAB.md §7.2：v2 边界定位强
（F1@0.25 24.00 全场最高）、v3 段识别强（edit 31.07），是有数据支撑的互补组合——
区别于已失败的 v3⊕roi（roi 被 v3 全面支配，无互补基础）。

- 契约 actionmixed-cleanv2v3-concat-v1：113 + 113 = 226 维
- 布局：左 113 维 = v2 块结构，右 113 维 = v3 块结构（两者块结构一一对应）
- 两侧均为因果序列特征（v3 含跨帧回退链前向填充），逐视频对齐后拼接
- exploratory 探针（无 catalog 登记，配置用显式 data.root）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .clean_bbox_v2 import build_clean_bbox_features
from .clean_bbox_v3 import build_clean_bbox_v3_features

CLEAN_V2V3_VERSION = "actionmixed-cleanv2v3-concat-v1"
CLEAN_V2V3_DIM = 226


def build_clean_v2_v3_features(
    frame_paths: list[Path],
    *,
    feature_version_v2: str,
    detection_mapping: dict,
    fps: float,
    confidence_default: float,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """逐视频序列 [T, 226]：v2（top1_impute）与 v3（scope_frame）沿特征维拼接。"""

    v2_feats, _names, actual_v2 = build_clean_bbox_features(
        frame_paths,
        detection_mapping=detection_mapping,
        feature_version=feature_version_v2,
        fps=fps,
        confidence_default=confidence_default,
        mask_target_ids=mask_target_ids,
    )
    if v2_feats.shape[1] != 113:
        raise ValueError(f"v2 侧维度 {v2_feats.shape[1]} != 113")
    v3_feats, _names3, actual_v3 = build_clean_bbox_v3_features(
        frame_paths,
        detection_mapping=detection_mapping,
        fps=fps,
        confidence_default=confidence_default,
        mask_target_ids=mask_target_ids,
    )
    if v3_feats.shape[1] != 113:
        raise ValueError(f"v3 侧维度 {v3_feats.shape[1]} != 113")
    if len(v2_feats) != len(v3_feats):
        raise ValueError(
            f"v2 帧数 {len(v2_feats)} 与 v3 帧数 {len(v3_feats)} 不对齐"
        )
    del actual_v2, actual_v3
    return np.concatenate([v2_feats, v3_feats], axis=1).astype(np.float32)