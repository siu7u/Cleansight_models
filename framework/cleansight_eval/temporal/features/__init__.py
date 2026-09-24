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
from .image_embed import (
    EMBED_BBOX_DIM,
    EMBED_FEAT_DIM,
    IMAGE_EMBED_DIM,
    IMAGE_EMBED_VERSION,
    IMAGE_EMBED_VERSION_PREFIX,
    is_image_embed_version,
    load_video_embeddings,
    resolve_embed_dim,
    resolve_embedding_root,
)
from .roi_bbox import (
    ROI_CHANNELS,
    ROI_FEATURE_DIM,
    ROI_FEATURE_VERSION,
    ROI_GRID_COLS,
    ROI_GRID_ROWS,
    ROI_N_REGIONS,
    ROI_PRESENCE_DIM,
    ROI_PRESENCE_VERSION,
    build_roi_frame_features,
    build_roi_presence_frame_features,
)
from .roi_bbox_v2 import (
    ROI_GRID_V2_CHANNELS,
    ROI_GRID_V2_COLD_BLOCK,
    ROI_GRID_V2_DIM,
    ROI_GRID_V2_HOT_BLOCK,
    ROI_GRID_V2_HOT_CLASS_IDS,
    ROI_GRID_V2_HOT_COLS,
    ROI_GRID_V2_HOT_ROWS,
    ROI_GRID_V2_N_CLASSES,
    ROI_GRID_V2_VERSION,
    build_roi_grid_v2_frame_features,
    roi_grid_v2_block_dims,
)


def feature_names_for_version(version: str) -> list[str] | None:
    """返回已注册 recipe 的列名；未知版本由其它 feature adapter 负责。"""

    if version in CLEAN_FEATURE_DIMS:
        return clean_feature_names(version)
    return None


def block_dims_for_version(version: str | None, n_classes: int) -> list[int] | None:
    """返回**块宽不等**契约的逐类块宽；均匀契约返回 None（由调用方按总维推导）。

    遮罩/随机遮罩按"特征维 ÷ 检测类数"推导块宽，对 ROI 可见性重排契约
    （``actionmixed-roi-grid-v2``：高频类 27 维、低频类 3 维）会静默遮错列，
    因此该契约必须走本函数拿显式块宽（见 INPUT_DESIGN_PROPOSAL.md §4）。
    """

    if version == ROI_GRID_V2_VERSION:
        return roi_grid_v2_block_dims(n_classes)
    return None


__all__ = [
    "CLEAN_FEATURE_DIMS",
    "EMBED_BBOX_DIM",
    "EMBED_FEAT_DIM",
    "GLOBAL_HAND_BBOX_VERSION",
    "GLOBAL_HAND_FEATURE_DIM",
    "HAND_BBOX_VERSION",
    "HAND_FEATURE_DIM",
    "IMAGE_EMBED_DIM",
    "IMAGE_EMBED_VERSION",
    "IMAGE_EMBED_VERSION_PREFIX",
    "ROI_CHANNELS",
    "ROI_FEATURE_DIM",
    "ROI_FEATURE_VERSION",
    "ROI_GRID_COLS",
    "ROI_GRID_ROWS",
    "ROI_GRID_V2_CHANNELS",
    "ROI_GRID_V2_COLD_BLOCK",
    "ROI_GRID_V2_DIM",
    "ROI_GRID_V2_HOT_BLOCK",
    "ROI_GRID_V2_HOT_CLASS_IDS",
    "ROI_GRID_V2_HOT_COLS",
    "ROI_GRID_V2_HOT_ROWS",
    "ROI_GRID_V2_N_CLASSES",
    "ROI_GRID_V2_VERSION",
    "ROI_N_REGIONS",
    "ROI_PRESENCE_DIM",
    "ROI_PRESENCE_VERSION",
    "block_dims_for_version",
    "build_clean_bbox_features",
    "build_hand_frame_features",
    "build_roi_frame_features",
    "build_roi_grid_v2_frame_features",
    "build_roi_presence_frame_features",
    "clean_feature_names",
    "feature_names_for_version",
    "is_image_embed_version",
    "load_video_embeddings",
    "resolve_embed_dim",
    "resolve_embedding_root",
    "roi_grid_v2_block_dims",
]
