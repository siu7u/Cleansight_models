"""clean_bbox_v4 / roi_v4 特征测试（docs/FEATURE_V4_DIM_SPEC.md 逐列锁定）。"""

from pathlib import Path

import numpy as np

from cleansight_eval.temporal.data import load_split
from cleansight_eval.temporal.features import (
    CLEAN_V4_FEATURE_DIMS,
    CLEAN_V4_VERSION,
    ROI_V4_FEATURE_DIM,
    ROI_V4_VERSION,
    build_clean_bbox_v4_features,
    build_roi_v4_frame_features,
    clean_v4_feature_names,
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


def _write_frames(tmp_path: Path, frames: int = 20, deprecated: bool = False) -> list[Path]:
    paths = []
    for index in range(frames):
        scope = "" if index == 10 else (
            "1 0.3 0.4 0.08 0.06 0.8\n"
            "2 0.6 0.4 0.10 0.07 0.7\n"
        )
        deprecated_rows = ("3 0.7 0.4 0.10 0.08 0.7\n" "6 0.5 0.6 0.05 0.05 0.6\n") if deprecated else ""
        path = tmp_path / f"frame-{index:06d}.txt"
        path.write_text(
            f"0 0.5 0.5 0.2 0.3 0.9\n{scope}{deprecated_rows}4 0.62 0.45 0.05 0.05 0.6\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def test_v4_dimensions_and_names_are_stable(tmp_path):
    frame_paths = _write_frames(tmp_path)
    features, names, version = build_clean_bbox_v4_features(
        frame_paths,
        detection_mapping=DETECTION_MAPPING,
        fps=7.5,
        confidence_default=1.0,
    )

    assert version == CLEAN_V4_VERSION == "ama-v4-scope-6c-60d"
    assert CLEAN_V4_FEATURE_DIMS == {CLEAN_V4_VERSION: 60}
    assert features.shape == (20, 60)
    assert names == clean_v4_feature_names()
    assert names == feature_names_for_version(CLEAN_V4_VERSION)
    assert np.isfinite(features).all()
    # 块边界锁定（docs/FEATURE_V4_DIM_SPEC.md §1）
    assert names[0] == "hand_count"
    assert names[1] == "hand_top1_present" and names[9] == "hand_top2_present"
    assert names[17] == "scope_control_body_candidate_count"
    assert names[24] == "scope_mid_section_candidate_count"
    assert names[33] == "syringe_candidate_count"
    assert names[48] == "hand_to_scope_control_body_valid"
    assert names[57] == "t_norm"


def test_v4_deprecated_classes_are_invariant(tmp_path):
    """决策①行为验证：帧里加/不加废弃类检测行，v4 特征必须逐位一致。"""
    base_dir = tmp_path / "base"
    dep_dir = tmp_path / "with_dep"
    base_dir.mkdir(); dep_dir.mkdir()
    feats_base, _, _ = build_clean_bbox_v4_features(
        _write_frames(base_dir, deprecated=False),
        detection_mapping=DETECTION_MAPPING, fps=7.5, confidence_default=1.0,
    )
    feats_dep, _, _ = build_clean_bbox_v4_features(
        _write_frames(dep_dir, deprecated=True),
        detection_mapping=DETECTION_MAPPING, fps=7.5, confidence_default=1.0,
    )
    np.testing.assert_array_equal(feats_base, feats_dep)
    # 列名层面：废弃类不得出现
    joined = " ".join(clean_v4_feature_names())
    assert "distal" not in joined
    assert "short_brush_" not in joined and "long_brush" not in joined


def test_v4_ctrl_block_has_no_axis_coords(tmp_path):
    """ctrl 是轴端点：along/across 恒 0 不编码（块 C 只 7 列）。"""
    joined = [n for n in clean_v4_feature_names()
              if n.startswith("scope_control_body_") and "_to_" not in n]
    assert "scope_control_body_along" not in joined
    assert "scope_control_body_across" not in joined
    assert len(joined) == 7  # candidate_count + present/conf/log_area_ratio/speed/missing_age/imputed


def test_v4_axis_fallback_keeps_finite(tmp_path):
    """第 10 帧清空 scope：v4 主轴 ctrl→mid 缺席，回退前帧轴，特征保持有界有限。"""
    frame_paths = _write_frames(tmp_path)
    features, names, _v = build_clean_bbox_v4_features(
        frame_paths, detection_mapping=DETECTION_MAPPING, fps=7.5, confidence_default=1.0,
    )
    index = {name: col for col, name in enumerate(names)}
    along = features[:, index["syringe_along"]]
    log_ratio = features[:, index["syringe_log_area_ratio"]]
    assert np.isfinite(along).all() and np.isfinite(log_ratio).all()
    assert np.abs(along).max() <= 4.0 and np.abs(log_ratio).max() <= 6.0
    # syringe 固定在 ctrl 右侧 ~0.32·L 处：轴正常帧的 along 应接近该几何值
    assert along[0] > 0.05


def test_v4_event_classes_no_speed_channels(tmp_path):
    """决策②结构验证：事件类仅 count+4 通道，无 speed/conf/missing/imputed。"""
    names = clean_v4_feature_names()
    for event in ("syringe", "air_gun", "brush_tip_out"):
        cols = [n for n in names if n.startswith(event + "_")]
        assert len(cols) == 5, f"{event} 块宽 {len(cols)} != 5"
        for banned in ("speed", "conf", "missing_age", "imputed"):
            assert not any(banned in c for c in cols), f"{event} 含禁止通道 {banned}"


def test_v4_empty_frames_zero_blocks(tmp_path):
    paths = [tmp_path / f"frame-{i:06d}.txt" for i in range(5)]
    features, names, version = build_clean_bbox_v4_features(
        paths, detection_mapping=DETECTION_MAPPING, fps=7.5, confidence_default=1.0,
    )
    index = {name: col for col, name in enumerate(names)}
    assert version == CLEAN_V4_VERSION and np.isfinite(features).all()
    assert features[:, index["hand_count"]].max() == 0.0
    assert features[:, index["syringe_present"]].max() == 0.0


def test_roi_v4_remap_and_dim(tmp_path):
    """ROI v4：108 维；废弃类(id 3/6)行丢弃；syringe(id 4)映射到 v4 id 3 的区域。"""
    p = tmp_path / "frame-000001.txt"
    p.write_text(
        "0 0.5 0.5 0.2 0.2\n"      # hand -> v4 id 0
        "3 0.7 0.4 0.1 0.1\n"      # distal 废弃丢弃
        "6 0.5 0.6 0.05 0.05\n"    # short_brush 废弃丢弃
        "4 0.15 0.2 0.05 0.05\n",  # syringe -> v4 id 3（第一行区域）
        encoding="utf-8",
    )
    feats = build_roi_v4_frame_features(p)
    assert feats.shape == (ROI_V4_FEATURE_DIM,) == (108,)
    # v4 id 0 (hand) 中心 (0.5,0.5) → col=int(0.5*3)=1,row=int(0.5*2)=1 → region 4
    assert feats[0 * 18 + 4 * 3 + 0] == 1.0
    # v4 id 3 (syringe 映射自 id 4) 第一区域 presence=1（区域行优先：cx=0.15,cy=0.2 落第 1 格）
    assert feats[3 * 18 + 0] == 1.0
    # distal/short_brush 无块可查（108 维只有 6 类）


def test_load_split_dispatches_v4(tmp_path):
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
        "0 0.5 0.5 0.2 0.2\n1 0.3 0.4 0.1 0.1\n2 0.6 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    features, truths, id2name = load_split(
        {"root": str(root), "fps": 7.5},
        "test",
        feature_schema={
            "dim": 60,
            "version": CLEAN_V4_VERSION,
            "class_order": CLASS_ORDER,
            "detection_confidence_default": 1.0,
        },
    )
    assert features[0].shape == (1, 60)
    assert np.isfinite(features[0]).all()
    assert truths[0].tolist() == [5] and id2name[5] == "air_injection"

    roi_feats, _, _ = load_split(
        {"root": str(root), "fps": 7.5},
        "test",
        feature_schema={"dim": 108, "version": ROI_V4_VERSION},
    )
    assert roi_feats[0].shape == (1, 108)