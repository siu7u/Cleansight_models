"""后端 CLEAN 外部模型的结构、特征和安全 checkpoint 接入测试。"""

from pathlib import Path

import numpy as np
import pytest
import torch

from cleansight_eval.core.checkpoint import load_checkpoint
from cleansight_eval.temporal.data import load_split, resolve_external_temporal_meta
from cleansight_eval.temporal.external import configure_external_model
from cleansight_eval.temporal.features import (
    build_clean_bbox_features,
    clean_feature_names,
)
from cleansight_eval.temporal.models import build_model


CASES = [
    (
        "clean_mstcn_bilstm",
        113,
        "clean_bbox_v2_top1_impute",
        {"hidden": 64, "lstm_layers": 2, "tcn_layers": 6, "refine_stages": 2, "dropout": 0.15},
    ),
    (
        "clean_asformer",
        121,
        "clean_bbox_v2_top1_impute+business_priors",
        {"hidden": 64, "nhead": 4, "num_layers": 4, "dropout": 0.15},
    ),
    (
        "clean_bigru",
        249,
        "clean_bbox_v2_top1_impute+center_window+business_priors",
        {"hidden": 64, "num_layers": 3, "dropout": 0.15},
    ),
]

CLASS_ORDER = [
    "idle",
    "long_brush_insert",
    "long_brush_withdraw",
    "short_brush_cleaning",
    "flush",
    "air_injection",
]


@pytest.mark.parametrize(("model_type", "feature_dim", "version", "extra"), CASES)
def test_clean_external_numpy_checkpoint_loads_strictly(
    tmp_path, model_type, feature_dim, version, extra
):
    """只白名单 NumPy metadata，并验证三种网络都能 strict 加载和逐帧前向。"""

    model_cfg = {
        "type": model_type,
        "input_dim": feature_dim,
        "num_classes": len(CLASS_ORDER),
        "allow_missing_meta": True,
        **extra,
    }
    model = build_model(model_cfg)
    checkpoint = tmp_path / f"{model_type}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_type,
            "class_names": CLASS_ORDER,
            "feature_names": clean_feature_names(version),
            "feature_version": version,
            "feature_dim": feature_dim,
            "normalizer_mean": np.zeros((1, feature_dim), dtype=np.float32),
            "normalizer_std": np.ones((1, feature_dim), dtype=np.float32),
        },
        checkpoint,
    )
    cfg = {
        "pipeline": "full_sequence_temporal",
        "model": model_cfg,
        "feature_schema": {
            "dim": feature_dim,
            "version": version,
            "class_order": CLASS_ORDER,
        },
        "evaluation": {"mode": "exploratory"},
        "train": {},
    }
    state, meta = load_checkpoint(
        checkpoint,
        fallback_meta=resolve_external_temporal_meta(cfg, cfg["pipeline"]),
    )
    rebuilt = build_model(model_cfg)
    rebuilt.load_state_dict(state, strict=True)
    configure_external_model(rebuilt, cfg, meta)

    output = rebuilt(torch.zeros(1, 4, feature_dim))

    assert output.shape == (1, 4, len(CLASS_ORDER))
    assert meta["_embedded_checkpoint"]["feature_version"] == version


@pytest.mark.parametrize(("_model_type", "feature_dim", "version", "_extra"), CASES)
def test_clean_feature_recipes_have_stable_dimensions(
    tmp_path, _model_type, feature_dim, version, _extra
):
    """同一 bbox 序列按三个 recipe 生成稳定列名和113/121/249维矩阵。"""

    frame_paths = []
    for index in range(20):
        path = tmp_path / f"frame-{index:06d}.txt"
        path.write_text("0 0.5 0.5 0.2 0.3\n6 0.4 0.4 0.1 0.2\n", encoding="utf-8")
        frame_paths.append(path)
    features, names, actual_version = build_clean_bbox_features(
        frame_paths,
        detection_mapping={0: "hand", 6: "short_brush"},
        feature_version=version,
        fps=7.5,
        confidence_default=1.0,
    )

    assert features.shape == (20, feature_dim)
    assert names == clean_feature_names(version)
    assert actual_version == version
    assert np.isfinite(features).all()


def test_load_split_remaps_action_ids_to_checkpoint_class_order(tmp_path):
    """外部类别顺序与数据集ID不同时，truth和输出标签统一按checkpoint顺序。"""

    root = Path(tmp_path)
    (root / "labels" / "test").mkdir(parents=True)
    (root / "frames" / "test").mkdir(parents=True)
    (root / "labels" / "data.yaml").write_text(
        "nc: 6\nnames: {0: idle, 1: air_injection, 2: flush, 3: long_brush_insert, "
        "4: long_brush_withdraw, 5: short_brush_cleaning}\n",
        encoding="utf-8",
    )
    (root / "frames" / "data.yaml").write_text(
        "nc: 8\nnames: {0: hand, 1: scope_control_body, 2: scope_mid_section, "
        "3: scope_distal_end, 4: syringe, 5: air_gun, 6: short_brush, 7: brush_tip_out}\n",
        encoding="utf-8",
    )
    (root / "labels" / "test" / "video.mp4.txt").write_text("1 1\n", encoding="utf-8")
    (root / "frames" / "test" / "video.mp4-000001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    features, truths, id2name = load_split(
        {"root": str(root), "fps": 7.5},
        "test",
        feature_schema={
            "dim": 113,
            "version": "clean_bbox_v2_top1_impute",
            "class_order": CLASS_ORDER,
            "detection_confidence_default": 1.0,
        },
    )

    assert features[0].shape == (1, 113)
    assert truths[0].tolist() == [5]
    assert id2name[5] == "air_injection"
