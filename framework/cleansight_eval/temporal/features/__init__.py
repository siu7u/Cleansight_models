"""时序特征 recipe 注册入口。"""

from .clean_bbox_v2 import (
    CLEAN_FEATURE_DIMS,
    build_clean_bbox_features,
    clean_feature_names,
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


def feature_names_for_version(version: str) -> list[str] | None:
    """返回已注册 recipe 的列名；未知版本由其它 feature adapter 负责。"""

    if version in CLEAN_FEATURE_DIMS:
        return clean_feature_names(version)
    return None


__all__ = [
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
    "build_hand_frame_features",
    "build_roi_frame_features",
    "clean_feature_names",
    "feature_names_for_version",
]
