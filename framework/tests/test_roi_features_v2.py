"""ROI 可见性重排特征（actionmixed-roi-grid-v2，96 维）的 feature 层单元测试。

覆盖：高频/低频类的区域与块偏移、空帧全零、整类遮罩、块宽表、
随机遮罩按显式块宽切片（不按"总维 ÷ 类数"错切）、load_split 分发与维度。
"""

import numpy as np
import pytest

from cleansight_eval.temporal.data import (
    apply_target_mask_augmentation,
    load_split,
)
from cleansight_eval.temporal.features import (
    ROI_GRID_V2_DIM,
    ROI_GRID_V2_VERSION,
    block_dims_for_version,
    build_roi_grid_v2_frame_features,
    roi_grid_v2_block_dims,
)

# 块偏移：hand 0 / scope_control_body 27 / scope_mid_section 54 /
# scope_distal_end 81 / syringe 84 / air_gun 87 / short_brush 90 / brush_tip_out 93
BLOCK_OFFSETS = (0, 27, 54, 81, 84, 87, 90, 93)


def _write_detection_mapping(root, names=("hand", "syringe", "air_gun")):
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True)
    lines = ["nc: %d" % len(names), "names:"]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    (frames_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_block_dims_and_offsets():
    """块宽表为 3×27 + 5×3，且类别数不等于 8 时直接报错而非静默算错。"""

    assert roi_grid_v2_block_dims(8) == [27, 27, 27, 3, 3, 3, 3, 3]
    assert sum(roi_grid_v2_block_dims(8)) == ROI_GRID_V2_DIM == 96
    assert block_dims_for_version(ROI_GRID_V2_VERSION, 8) == [27, 27, 27, 3, 3, 3, 3, 3]
    assert block_dims_for_version("actionmixed-roi-grid-v1", 8) is None  # 均匀契约

    with pytest.raises(ValueError, match="8 个检测类"):
        roi_grid_v2_block_dims(3)


def test_high_freq_class_uses_3x3_regions(tmp_path):
    """高频类（hand）：3×3 网格行优先编号，count/max_area/presence 语义正确。"""

    bbox_path = tmp_path / "frame.txt"
    bbox_path.write_text(
        "0 0.1 0.2 0.3 0.4\n"   # hand → region0（row0,col0），面积 0.12
        "0 0.9 0.9 0.1 0.2\n"   # hand → region8（row2,col2），面积 0.02
        "0 0.9 0.9 0.5 0.5\n"   # hand → region8，面积 0.25（同区域取最大）
        "0 0.5 0.5 0.2 0.2\n",  # hand → region4（row1,col1）
        encoding="utf-8",
    )

    flat = build_roi_grid_v2_frame_features(bbox_path)
    block = flat[:27].reshape(9, 3)  # 高频类块宽 27 = 9 区域 × 3 通道

    np.testing.assert_allclose(block[0], [1.0, 1.0, 0.12], rtol=1e-6)
    np.testing.assert_allclose(block[4], [1.0, 1.0, 0.04], rtol=1e-6)
    np.testing.assert_allclose(block[8], [1.0, 2.0, 0.25], rtol=1e-6)
    assert block[[1, 2, 3, 5, 6, 7]].sum() == 0.0
    assert flat[27:].sum() == 0.0  # 其余 7 类块保持全零


def test_low_freq_class_collapses_to_single_global_region(tmp_path):
    """低频类（syringe=4）：所有区域合并为 1 个全局通道，密度不再被网格稀释。"""

    bbox_path = tmp_path / "frame.txt"
    bbox_path.write_text(
        "4 0.1 0.1 0.2 0.2\n"   # syringe（左上）
        "4 0.9 0.9 0.3 0.3\n",  # syringe（右下）→ 同一全局区域
        encoding="utf-8",
    )

    flat = build_roi_grid_v2_frame_features(bbox_path)
    block = flat[BLOCK_OFFSETS[4] : BLOCK_OFFSETS[4] + 3]

    np.testing.assert_allclose(block, [1.0, 2.0, 0.09], rtol=1e-6)
    # 其余类块全零
    assert flat[: BLOCK_OFFSETS[4]].sum() == 0.0
    assert flat[BLOCK_OFFSETS[4] + 3 :].sum() == 0.0


def test_empty_and_missing_file_are_all_zero(tmp_path):
    """空帧与缺失文件语义一致：全零 96 维。"""

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")

    for path in (empty, tmp_path / "missing.txt"):
        feat = build_roi_grid_v2_frame_features(path)
        assert feat.shape == (ROI_GRID_V2_DIM,)
        assert feat.dtype == np.float32
        np.testing.assert_array_equal(feat, np.zeros(ROI_GRID_V2_DIM, dtype=np.float32))


def test_mask_zeroes_only_the_target_class_block(tmp_path):
    """整类遮罩只清目标类的块（高频 27 维、低频 3 维），不影响其它类。"""

    bbox_path = tmp_path / "frame.txt"
    bbox_path.write_text(
        "0 0.1 0.1 0.3 0.4\n"
        "1 0.5 0.5 0.2 0.2\n"
        "4 0.9 0.9 0.2 0.2\n",
        encoding="utf-8",
    )

    feat = build_roi_grid_v2_frame_features(bbox_path, mask_target_ids=frozenset({4}))

    assert feat[BLOCK_OFFSETS[4] : BLOCK_OFFSETS[4] + 3].sum() == 0.0     # syringe 被遮
    assert feat[BLOCK_OFFSETS[0] : BLOCK_OFFSETS[0] + 27].sum() > 0.0     # hand 保留
    assert feat[BLOCK_OFFSETS[1] : BLOCK_OFFSETS[1] + 27].sum() > 0.0     # scope_ctl 保留


def test_load_split_dispatches_roi_v2_contract(tmp_path):
    """load_split 按 feature_schema.version 走 v2 recipe，输出 [T, 96]。"""

    _write_detection_mapping(
        tmp_path,
        names=("hand", "scope_control_body", "scope_mid_section", "scope_distal_end",
               "syringe", "air_gun", "short_brush", "brush_tip_out"),
    )
    (tmp_path / "labels" / "test").mkdir(parents=True)
    (tmp_path / "frames" / "test").mkdir(parents=True)
    (tmp_path / "labels" / "data.yaml").write_text("nc: 1\nnames:\n  0: idle\n", encoding="utf-8")
    (tmp_path / "labels" / "test" / "sample.mp4.txt").write_text("1 0\n2 0\n", encoding="utf-8")
    (tmp_path / "frames" / "test" / "sample.mp4-000001.txt").write_text(
        "0 0.1 0.1 0.3 0.4\n", encoding="utf-8"
    )
    (tmp_path / "frames" / "test" / "sample.mp4-000002.txt").write_text("", encoding="utf-8")
    data_cfg = {
        "root": str(tmp_path),
        "labels_dir": "labels",
        "frames_dir": "frames",
        "action_mapping": "labels/data.yaml",
    }

    features, truths, id2name = load_split(
        data_cfg,
        "test",
        feature_schema={"dim": ROI_GRID_V2_DIM, "version": ROI_GRID_V2_VERSION},
    )

    assert features[0].shape == (2, ROI_GRID_V2_DIM)
    np.testing.assert_allclose(features[0][0][:3], [1.0, 1.0, 0.12], rtol=1e-6)
    np.testing.assert_array_equal(features[0][1], np.zeros(ROI_GRID_V2_DIM))
    np.testing.assert_array_equal(truths[0], [0, 0])
    assert id2name == {0: "idle"}


def test_augmentation_uses_explicit_block_dims_for_roi_v2(tmp_path):
    """随机遮罩按显式块宽切片：syringe(类4) 落在 [84:87]，而不是均匀块的 [48:60]。"""

    _write_detection_mapping(
        tmp_path,
        names=("hand", "scope_control_body", "scope_mid_section", "scope_distal_end",
               "syringe", "air_gun", "short_brush", "brush_tip_out"),
    )
    data_cfg = {"root": str(tmp_path), "frames_dir": "frames"}
    features = [np.ones((2, ROI_GRID_V2_DIM), dtype=np.float32)]
    augmentation = {
        "target_mask": {
            "enabled": True,
            "strategy": "frame_dropout",
            "targets": ["syringe"],
            "probability": 1.0,
        }
    }

    masked = apply_target_mask_augmentation(
        features,
        data_cfg,
        augmentation,
        seed=7,
        feature_schema={"dim": ROI_GRID_V2_DIM, "version": ROI_GRID_V2_VERSION},
    )

    np.testing.assert_array_equal(masked[0][:, 84:87], np.zeros((2, 3), dtype=np.float32))
    assert masked[0][:, :84].sum() == 2 * 84
    assert masked[0][:, 87:].sum() == 2 * 9
