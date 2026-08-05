"""外部时序 checkpoint 的内嵌特征/标签/归一化契约适配。"""

from __future__ import annotations

from .features import feature_names_for_version


def configure_external_model(model, cfg: dict, meta: dict) -> None:
    """校验 checkpoint 内嵌契约并向模型注入 normalizer；缺字段不伪造。"""

    embedded = meta.get("_embedded_checkpoint") or {}
    if not embedded:
        return

    feature_schema = cfg.get("feature_schema") or {}
    configured_dim = feature_schema.get("dim")
    embedded_dim = embedded.get("feature_dim")
    if (
        embedded_dim is not None
        and configured_dim is not None
        and int(embedded_dim) != int(configured_dim)
    ):
        raise ValueError(
            f"checkpoint feature_dim={embedded_dim} 与配置 feature_schema.dim={configured_dim} 不一致"
        )

    configured_version = feature_schema.get("version")
    embedded_version = embedded.get("feature_version")
    if embedded_version is not None and configured_version != embedded_version:
        raise ValueError(
            f"checkpoint feature_version={embedded_version!r} 与配置 {configured_version!r} 不一致"
        )

    configured_classes = feature_schema.get("class_order")
    embedded_classes = embedded.get("class_names")
    if embedded_classes is not None and configured_classes is not None:
        if list(embedded_classes) != list(configured_classes):
            raise ValueError(
                "checkpoint class_names 与 feature_schema.class_order 不一致: "
                f"checkpoint={list(embedded_classes)}, configured={list(configured_classes)}"
            )

    embedded_names = embedded.get("feature_names")
    expected_names = (
        feature_names_for_version(str(configured_version))
        if configured_version is not None
        else None
    )
    if embedded_names is not None and expected_names is not None:
        if list(embedded_names) != expected_names:
            raise ValueError("checkpoint feature_names 与已注册 feature recipe 列顺序不一致")

    mean = embedded.get("normalizer_mean")
    std = embedded.get("normalizer_std")
    if (mean is None) != (std is None):
        raise ValueError("checkpoint normalizer_mean/normalizer_std 必须同时存在")
    if mean is not None:
        setter = getattr(model, "set_input_normalization", None)
        if setter is None:
            raise ValueError(
                f"模型 {type(model).__name__} 不支持 checkpoint 内嵌输入归一化统计"
            )
        setter(mean, std)
