"""clean_bbox_v3 scope 坐标系特征测试。"""

from pathlib import Path

import numpy as np

from cleansight_eval.temporal.data import load_split
from cleansight_eval.temporal.features import (
    CLEAN_V3_FEATURE_DIMS,
    CLEAN_V3_VERSION,
    build_clean_bbox_v3_features,
    clean_v3_feature_names,
    feature_names_for_version,
)


DETECTION_MAPPING = {
    0: "hand",
    1: "scope_control_body",
    2: "scope_mid_section",
    3: "scope_distal_end",
    4: "syringe",
    5: "air_gun",
    6: "short_brush",
    7: "brush_tip_out",
}

CLASS_ORDER = [
    "idle",
    "long_brush_insert",
    "long_brush_withdraw",
    "short_brush_cleaning",
    "flush",
    "air_injection",
]


def _write_frames(tmp_path: Path, frames: int = 20) -> list[Path]:
    paths = []
    for index in range(frames):
        scope = "" if index == 10 else (
            "1 0.3 0.4 0.08 0.06 0.8\n"
            "3 0.7 0.4 0.10 0.08 0.7\n"
        )
        path = tmp_path / f"frame-{index:06d}.txt"
        path.write_text(
            f"0 0.5 0.5 0.2 0.3 0.9\n{scope}4 0.6 0.45 0.05 0.05 0.6\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def test_v3_dimensions_and_names_are_stable(tmp_path):
    frame_paths = _write_frames(tmp_path)
    features, names, version = build_clean_bbox_v3_features(
        frame_paths,
        detection_mapping=DETECTION_MAPPING,
        fps=7.5,
        confidence_default=1.0,
    )

    assert version == CLEAN_V3_VERSION
    assert CLEAN_V3_FEATURE_DIMS == {CLEAN_V3_VERSION: 113}
    assert features.shape == (20, 113)
    assert names == clean_v3_feature_names()
    assert names == feature_names_for_version(CLEAN_V3_VERSION)
    assert np.isfinite(features).all()


def test_v3_scope_axes_fall_back_when_scope_missing(tmp_path):
    """第 10 帧清空 scope 检测：位置/尺度列应沿用前向回退且保持有限。"""
    frame_paths = _write_frames(tmp_path)
    features, names, _version = build_clean_bbox_v3_features(
        frame_paths,
        detection_mapping=DETECTION_MAPPING,
        fps=7.5,
        confidence_default=1.0,
    )
    index = {name: col for col, name in enumerate(names)}

    along = features[:, index["syringe_along"]]
    log_ratio = features[:, index["syringe_log_area_ratio"]]

    assert np.isfinite(along).all() and np.isfinite(log_ratio).all()
    assert np.abs(along).max() <= 4.0
    assert np.abs(log_ratio).max() <= 6.0
    # syringe 在 scope 缺席帧不缺席，along 不应整体塌缩为 0
    assert np.abs(along[9:12]).max() > 0.0


def test_v3_empty_frames_produce_zero_blocks(tmp_path):
    paths = [tmp_path / f"frame-{index:06d}.txt" for index in range(5)]
    features, names, version = build_clean_bbox_v3_features(
        paths,
        detection_mapping=DETECTION_MAPPING,
        fps=7.5,
        confidence_default=1.0,
    )

    index = {name: col for col, name in enumerate(names)}
    assert version == CLEAN_V3_VERSION
    assert np.isfinite(features).all()
    assert features[:, index["hand_count"]].max() == 0.0
    assert features[:, index["syringe_present"]].max() == 0.0


def test_load_split_dispatches_v3(tmp_path):
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
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    features, truths, id2name = load_split(
        {"root": str(root), "fps": 7.5},
        "test",
        feature_schema={
            "dim": 113,
            "version": CLEAN_V3_VERSION,
            "class_order": CLASS_ORDER,
            "detection_confidence_default": 1.0,
        },
    )

    assert features[0].shape == (1, 113)
    assert np.isfinite(features[0]).all()
    assert truths[0].tolist() == [5]
    assert id2name[5] == "air_injection"