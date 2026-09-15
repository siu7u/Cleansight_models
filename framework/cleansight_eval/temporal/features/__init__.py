"""时序特征 recipe 注册入口。"""

from __future__ import annotations

from .clean_bbox_v2 import (
    CLEAN_FEATURE_DIMS,
    build_clean_bbox_features,
    clean_feature_names,
)
from .clean_bbox_v3 import (
    CLEAN_V3_FEATURE_DIMS,
    CLEAN_V3_VERSION,
    build_clean_bbox_v3_features,
    clean_v3_feature_names,
)
from .hand_bbox import (
    GLOBAL_HAND_BBOX_VERSION,
    GLOBAL_HAND_FEATURE_DIM,
    HAND_BBOX_VERSION,
    HAND_FEATURE_DIM,
    build_hand_frame_features,
)
from .roi_bbox import (
    ROI_CHANNELS,
    ROI_FEATURE_DIM,
    ROI_FEATURE_VERSION,
    ROI_GRID_COLS,
    ROI_GRID_ROWS,
    ROI_N_REGIONS,
    build_roi_frame_features,
)
from .cnn_concat import (
    CNN_BBOX_VERSION,
    CNN_FEATURE_DIMS,
    CNN_HAND_VERSION,
    build_cnn_concat_frame,
    load_pca,
    project_embedding,
)


def feature_names_for_version(version: str) -> list[str] | None:
    """返回已注册 recipe 的列名；未知版本由其它 feature adapter 负责。"""

    if version in CLEAN_FEATURE_DIMS:
        return clean_feature_names(version)
    if version in CLEAN_V3_FEATURE_DIMS:
        return clean_v3_feature_names()
    return None


__all__ = [
    "CLEAN_V3_FEATURE_DIMS",
    "CLEAN_V3_VERSION",
    "CNN_BBOX_VERSION",
    "CNN_FEATURE_DIMS",
    "CNN_HAND_VERSION",
    "CLEAN_FEATURE_DIMS",
    "GLOBAL_HAND_BBOX_VERSION",
    "GLOBAL_HAND_FEATURE_DIM",
    "HAND_BBOX_VERSION",
    "HAND_FEATURE_DIM",
    "ROI_CHANNELS",
    "ROI_FEATURE_DIM",
    "ROI_FEATURE_VERSION",
    "ROI_GRID_COLS",
    "ROI_GRID_ROWS",
    "ROI_N_REGIONS",
    "build_clean_bbox_features",
    "build_clean_bbox_v3_features",
    "build_hand_frame_features",
    "build_roi_frame_features",
    "clean_feature_names",
    "clean_v3_feature_names",
    "feature_names_for_version",
]
