"""手部区域特征（actionmixed-bbox-hand-8cls-v1 / global-hand 80 维）的 feature 层单元测试。"""

import numpy as np

from cleansight_eval.temporal.data import load_split
from cleansight_eval.temporal.features import (
    GLOBAL_HAND_BBOX_VERSION,
    GLOBAL_HAND_FEATURE_DIM,
    HAND_BBOX_VERSION,
    HAND_FEATURE_DIM,
    build_hand_frame_features,
)


def test_hand_frame_features_region_encoding(tmp_path):
    """hand 框扩张 1.5 倍成区域：区域内框相对编码，区域外框被排除，hand 类在中心。"""

    bbox_path = tmp_path / "frame.txt"
    bbox_path.write_text(
        "0 0.5 0.5 0.2 0.2\n"   # hand：区域 = (0.35,0.35)-(0.65,0.65)，宽高 0.3
        "1 0.45 0.45 0.1 0.1\n"  # 类1 在区域内 → 相对坐标 (1/3, 1/3)，相对尺寸 1/3
        "2 0.9 0.9 0.05 0.05\n",  # 类2 在区域外 → 排除
        encoding="utf-8",
    )

    feat = build_hand_frame_features(bbox_path, n_classes=3).reshape(3, 5)

    np.testing.assert_allclose(feat[0], [1.0, 0.5, 0.5, 0.2 / 0.3, 0.2 / 0.3], rtol=1e-6)
    np.testing.assert_allclose(feat[1], [1.0, 1 / 3, 1 / 3, 1 / 3, 1 / 3], rtol=1e-6)
    np.testing.assert_array_equal(feat[2], np.zeros(5, dtype=np.float32))


def test_hand_frame_features_region_clipped_at_frame_edge(tmp_path):
    """手部区域被画面边界钳制时，相对坐标按钳制后的区域归一化。"""

    bbox_path = tmp_path / "frame.txt"
    bbox_path.write_text(
        "0 0.05 0.5 0.1 0.2\n"  # 区域 = (0, 0.35)-(0.125, 0.65)
        "1 0.1 0.5 0.05 0.05\n",  # 类1 区域内 → cx_rel=0.1/0.125=0.8
        encoding="utf-8",
    )

    feat = build_hand_frame_features(bbox_path, n_classes=2).reshape(2, 5)

    np.testing.assert_allclose(feat[0], [1.0, 0.4, 0.5, 0.1 / 0.125, 0.2 / 0.3], rtol=1e-6)
    np.testing.assert_allclose(feat[1], [1.0, 0.8, 0.5, 0.05 / 0.125, 0.05 / 0.3], rtol=1e-6)


def test_hand_frame_features_no_hand_is_all_zero(tmp_path):
    bbox_path = tmp_path / "frame.txt"
    bbox_path.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    feat = build_hand_frame_features(bbox_path, n_classes=8)

    assert feat.shape == (HAND_FEATURE_DIM,)
    np.testing.assert_array_equal(feat, np.zeros(HAND_FEATURE_DIM, dtype=np.float32))


def test_hand_frame_features_masks_whole_class_block(tmp_path):
    bbox_path = tmp_path / "frame.txt"
    bbox_path.write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.45 0.45 0.1 0.1\n",
        encoding="utf-8",
    )

    feat = build_hand_frame_features(
        bbox_path, n_classes=2, mask_target_ids=frozenset({1})
    ).reshape(2, 5)

    np.testing.assert_allclose(feat[0], [1.0, 0.5, 0.5, 2 / 3, 2 / 3], rtol=1e-6)
    np.testing.assert_array_equal(feat[1], np.zeros(5, dtype=np.float32))


def _write_synthetic_dataset(root):
    (root / "frames").mkdir(parents=True)
    (root / "frames" / "data.yaml").write_text(
        "nc: 3\nnames:\n  0: hand\n  1: syringe\n  2: air_gun\n", encoding="utf-8"
    )
    (root / "labels" / "test").mkdir(parents=True)
    (root / "frames" / "test").mkdir(parents=True)
    (root / "labels" / "data.yaml").write_text(
        "nc: 1\nnames:\n  0: idle\n", encoding="utf-8"
    )
    (root / "labels" / "test" / "sample.mp4.txt").write_text(
        "1 0\n", encoding="utf-8"
    )
    (root / "frames" / "test" / "sample.mp4-000001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.45 0.45 0.1 0.1\n2 0.9 0.9 0.05 0.05\n",
        encoding="utf-8",
    )


def test_load_split_hand_contract(tmp_path):
    """load_split 按 version 分发到手部 recipe，输出 [T, 40]。"""

    _write_synthetic_dataset(tmp_path)
    data_cfg = {
        "root": str(tmp_path),
        "labels_dir": "labels",
        "frames_dir": "frames",
        "action_mapping": "labels/data.yaml",
    }

    features, truths, _ = load_split(
        data_cfg,
        "test",
        feature_schema={"dim": HAND_FEATURE_DIM, "version": HAND_BBOX_VERSION},
    )

    assert features[0].shape == (1, HAND_FEATURE_DIM)
    frame = features[0][0].reshape(8, 5)
    np.testing.assert_allclose(frame[0], [1.0, 0.5, 0.5, 2 / 3, 2 / 3], rtol=1e-6)
    np.testing.assert_allclose(frame[1], [1.0, 1 / 3, 1 / 3, 1 / 3, 1 / 3], rtol=1e-6)
    np.testing.assert_array_equal(frame[2], np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(truths[0], [0])


def test_load_split_global_hand_contract(tmp_path):
    """global-hand 契约输出 [T, 80]：左半全局、右半手部。"""

    _write_synthetic_dataset(tmp_path)
    data_cfg = {
        "root": str(tmp_path),
        "labels_dir": "labels",
        "frames_dir": "frames",
        "action_mapping": "labels/data.yaml",
    }

    features, truths, _ = load_split(
        data_cfg,
        "test",
        feature_schema={"dim": GLOBAL_HAND_FEATURE_DIM, "version": GLOBAL_HAND_BBOX_VERSION},
    )

    assert features[0].shape == (1, GLOBAL_HAND_FEATURE_DIM)
    frame = features[0][0]
    global_block = frame[:40].reshape(8, 5)
    hand_block = frame[40:].reshape(8, 5)
    # 全局：类1 最大框在 (0.45, 0.45)；手部：类1 相对编码 1/3
    np.testing.assert_allclose(global_block[1], [1.0, 0.45, 0.45, 0.1, 0.1], rtol=1e-6)
    np.testing.assert_allclose(hand_block[1], [1.0, 1 / 3, 1 / 3, 1 / 3, 1 / 3], rtol=1e-6)
    np.testing.assert_array_equal(truths[0], [0])
