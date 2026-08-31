"""ROI 空间特征（actionmixed-roi-grid-v1）的 feature 层单元测试。"""

import numpy as np
import pytest

from cleansight_eval.temporal.data import (
    apply_target_mask_augmentation,
    load_split,
)
from cleansight_eval.temporal.features import (
    ROI_FEATURE_DIM,
    ROI_FEATURE_VERSION,
    build_roi_frame_features,
)


def _write_detection_mapping(root, names=("hand", "syringe", "air_gun")):
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True)
    lines = ["nc: %d" % len(names), "names:"]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    (frames_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_roi_frame_features_regions_and_channels(tmp_path):
    """按 2×3 网格断言 presence/count/max_area 与区域编号（行优先）。"""

    bbox_path = tmp_path / "frame.txt"
    bbox_path.write_text(
        "0 0.1 0.2 0.3 0.4\n"   # 类0 → 区域0（row0,col0），面积 0.12
        "0 0.5 0.6 0.1 0.2\n"   # 类0 → 区域4（row1,col1），面积 0.02
        "1 0.8 0.9 0.2 0.2\n"   # 类1 → 区域5（row1,col2），面积 0.04
        "1 0.8 0.9 0.5 0.5\n"   # 类1 → 区域5，面积 0.25（同区域取最大）
        "2 0.0 1.0 0.1 0.1\n",  # 类2 边界 cy=1.0 → 钳制 row1,col0 → 区域3
        encoding="utf-8",
    )

    feat = build_roi_frame_features(bbox_path, n_classes=3).reshape(3, 6, 3)

    np.testing.assert_allclose(feat[0, 0], [1.0, 1.0, 0.12], rtol=1e-6)
    np.testing.assert_allclose(feat[0, 4], [1.0, 1.0, 0.02], rtol=1e-6)
    np.testing.assert_allclose(feat[1, 5], [1.0, 2.0, 0.25], rtol=1e-6)
    np.testing.assert_allclose(feat[2, 3], [1.0, 1.0, 0.01], rtol=1e-6)
    assert feat[0, 1:4].sum() == 0.0
    assert feat[1, :5].sum() == 0.0


def test_roi_frame_features_empty_file_is_all_zero(tmp_path):
    bbox_path = tmp_path / "empty.txt"
    bbox_path.write_text("", encoding="utf-8")

    feat = build_roi_frame_features(bbox_path, n_classes=8)

    assert feat.shape == (ROI_FEATURE_DIM,)
    np.testing.assert_array_equal(feat, np.zeros(ROI_FEATURE_DIM, dtype=np.float32))


def test_roi_frame_features_masks_whole_class_block(tmp_path):
    bbox_path = tmp_path / "frame.txt"
    bbox_path.write_text(
        "0 0.1 0.1 0.3 0.4\n1 0.9 0.9 0.2 0.2\n",
        encoding="utf-8",
    )

    feat = build_roi_frame_features(
        bbox_path, n_classes=2, mask_target_ids=frozenset({0})
    ).reshape(2, 6, 3)

    np.testing.assert_array_equal(feat[0], np.zeros((6, 3), dtype=np.float32))
    np.testing.assert_allclose(feat[1, 5], [1.0, 1.0, 0.04], rtol=1e-6)


def test_load_split_roi_contract(tmp_path):
    """load_split 按 feature_schema.version 走 ROI recipe，输出 [T, 144]。"""

    _write_detection_mapping(tmp_path)
    (tmp_path / "labels" / "test").mkdir(parents=True)
    (tmp_path / "frames" / "test").mkdir(parents=True)
    (tmp_path / "labels" / "data.yaml").write_text(
        "nc: 1\nnames:\n  0: idle\n",
        encoding="utf-8",
    )
    (tmp_path / "labels" / "test" / "sample.mp4.txt").write_text(
        "1 0\n2 0\n",
        encoding="utf-8",
    )
    (tmp_path / "frames" / "test" / "sample.mp4-000001.txt").write_text(
        "0 0.1 0.1 0.3 0.4\n",
        encoding="utf-8",
    )
    (tmp_path / "frames" / "test" / "sample.mp4-000002.txt").write_text(
        "",  # 空帧 → 全零 144 维
        encoding="utf-8",
    )
    data_cfg = {
        "root": str(tmp_path),
        "labels_dir": "labels",
        "frames_dir": "frames",
        "action_mapping": "labels/data.yaml",
    }

    features, truths, id2name = load_split(
        data_cfg,
        "test",
        feature_schema={"dim": ROI_FEATURE_DIM, "version": ROI_FEATURE_VERSION},
    )

    assert features[0].shape == (2, ROI_FEATURE_DIM)
    frame1 = features[0][0].reshape(8, 6, 3)
    np.testing.assert_allclose(frame1[0, 0], [1.0, 1.0, 0.12], rtol=1e-6)
    np.testing.assert_array_equal(features[0][1], np.zeros(ROI_FEATURE_DIM))
    np.testing.assert_array_equal(truths[0], [0, 0])
    assert id2name == {0: "idle"}


def test_augmentation_block_width_follows_roi_contract(tmp_path):
    """目标遮罩在 ROI 特征上按 18 维整类块清零，而不是硬编码的 5 维。"""

    _write_detection_mapping(
        tmp_path,
        names=("hand", "scope_control_body", "scope_mid_section", "scope_distal_end",
               "syringe", "air_gun", "short_brush", "brush_tip_out"),
    )
    data_cfg = {"root": str(tmp_path), "frames_dir": "frames"}
    features = [np.ones((2, ROI_FEATURE_DIM), dtype=np.float32)]
    augmentation = {
        "target_mask": {
            "enabled": True,
            "strategy": "frame_dropout",
            "targets": ["syringe"],  # 检测类 ID=4，块 [72:90]
            "probability": 1.0,
        }
    }

    masked = apply_target_mask_augmentation(features, data_cfg, augmentation, seed=7)

    np.testing.assert_array_equal(masked[0][:, 72:90], np.zeros((2, 18), dtype=np.float32))
    assert masked[0][:, :72].sum() == 2 * 72
    assert masked[0][:, 90:].sum() == 2 * 54


def test_augmentation_rejects_non_class_multiple_width(tmp_path):
    """特征维不是检测类数整数倍时（如 CLEAN 113 维）立即报错而非静默错切。"""

    _write_detection_mapping(tmp_path)
    data_cfg = {"root": str(tmp_path), "frames_dir": "frames"}
    features = [np.ones((2, 113), dtype=np.float32)]
    augmentation = {
        "target_mask": {
            "enabled": True,
            "strategy": "frame_dropout",
            "targets": ["syringe"],
            "probability": 1.0,
        }
    }

    with pytest.raises(ValueError, match="整数倍"):
        apply_target_mask_augmentation(features, data_cfg, augmentation, seed=7)
