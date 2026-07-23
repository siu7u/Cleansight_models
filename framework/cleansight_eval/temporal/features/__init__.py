"""时序特征 recipe 注册入口。"""

from .clean_bbox_v2 import (
    CLEAN_FEATURE_DIMS,
    build_clean_bbox_features,
    clean_feature_names,
)


def feature_names_for_version(version: str) -> list[str] | None:
    """返回已注册 recipe 的列名；未知版本由其它 feature adapter 负责。"""

    if version in CLEAN_FEATURE_DIMS:
        return clean_feature_names(version)
    return None


__all__ = [
    "CLEAN_FEATURE_DIMS",
    "build_clean_bbox_features",
    "clean_feature_names",
    "feature_names_for_version",
]
