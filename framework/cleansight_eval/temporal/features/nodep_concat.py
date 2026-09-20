"""消融契约：v2⊕v3 同编码、仅剔除废弃类检测行（226 维，数据侧消融）。

动机（docs/FEATURE_LAB.md §8.3-2）：v4 重构块结构后 val 回落（信息量损失）但
test 跨批次领先（过拟合源剔除）——两个效应混在一起。本消融保留 v2⊕v3 的
**编码与维度完全不变**（226），唯一变量 = 废弃类（scope_distal_end / short_brush /
long_brush）的检测行在读入层丢弃（这些块变为全零）。判读：
- val 若回到 33+ → v4 的 val 回落来自块结构重构（60 维砍太狠），非信息剔除；
- test 若明显优于 v2⊕v3 的 26.09 → 剔废弃的泛化收益与块结构无关。
契约 ama-v3-concat23-nodep-226d；探针阶段用显式 data.root，不登记 catalog。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .clean_bbox_v2 import build_clean_bbox_features
from .clean_bbox_v3 import build_clean_bbox_v3_features

NODEP_CONCAT_VERSION = "ama-v3-concat23-nodep-226d"
NODEP_CONCAT_DIM = 226
DEPRECATED = {"scope_distal_end", "short_brush", "long_brush"}

def build_nodep_concat_features(
    frame_paths: list[Path],
    *,
    detection_mapping: dict,
    fps: float,
    confidence_default: float,
    mask_target_ids: frozenset[int] = frozenset(),
):
    """逐视频 [T,226]：v2⊕v3 同编码，废弃类检测行读入层丢弃（块变全零）。"""
    filtered = {i: n for i, n in detection_mapping.items() if n not in DEPRECATED}
    v2_feats, _n1, _ver2 = build_clean_bbox_features(
        frame_paths, detection_mapping=filtered,
        feature_version="clean_bbox_v2_top1_impute",
        fps=fps, confidence_default=confidence_default,
        mask_target_ids=mask_target_ids,
    )
    v3_feats, _n2, _ver3 = build_clean_bbox_v3_features(
        frame_paths, detection_mapping=filtered,
        fps=fps, confidence_default=confidence_default,
        mask_target_ids=mask_target_ids,
    )
    if v2_feats.shape[1] != 113 or v3_feats.shape[1] != 113:
        raise ValueError(f"维度异常: v2={v2_feats.shape[1]} v3={v3_feats.shape[1]}")
    if len(v2_feats) != len(v3_feats):
        raise ValueError("v2/v3 帧数不对齐")
    feats = np.concatenate([v2_feats, v3_feats], axis=1).astype(np.float32)
    if feats.shape[1] != NODEP_CONCAT_DIM:
        raise AssertionError(f"拼接维度 {feats.shape[1]} != {NODEP_CONCAT_DIM}")
    return feats, None, NODEP_CONCAT_VERSION