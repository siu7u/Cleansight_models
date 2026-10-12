"""拼接族 combo-v3 / combo-v4（roi-grid-v5 ‖ 手部区域 v2）的 feature 层单元测试。

覆盖：分段宽度与自检、拼接顺序、空帧全零、真数据端到端宽度 == 契约维度（232 / 272，
**这是加载器分派层的断言**，只测构造器测不到"分派被顶掉、静默回落到其它宽度"）。
"""

from pathlib import Path

import numpy as np
import pytest

from cleansight_eval.temporal.data import load_split
from cleansight_eval.temporal.features import (
    COMBO_V3_DIM,
    COMBO_V3_PARTS,
    COMBO_V3_VERSION,
    COMBO_V4_DIM,
    COMBO_V4_PARTS,
    COMBO_V4_VERSION,
    build_combo_v3_frame_features,
    build_combo_v4_frame_features,
)
from cleansight_eval.temporal.features.hand_bbox_v2 import build_hand_v2_frame_features
from cleansight_eval.temporal.features.roi_bbox_v5 import (
    ROI_GRID_V5_DIM,
    build_roi_grid_v5_frame_features,
)


def _write_frame(path: Path, rows) -> None:
    path.write_text("\n".join(f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in rows) + "\n",
                    encoding="utf-8")


def test_parts_and_dims():
    """两个契约的分段与总维度：232 = 192+40、272 = 192+40+40。"""

    assert COMBO_V3_PARTS == (192, 40) and COMBO_V3_DIM == 232
    assert COMBO_V4_PARTS == (192, 40, 40) and COMBO_V4_DIM == 272


def test_v3_layout_is_roi_then_hand_v2(tmp_path):
    """combo-v3 的左段逐位等于 roi-grid-v5 的输出、右段等于 hand-v2 的输出。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [(0, 0.4, 0.5, 0.2, 0.1), (0, 0.7, 0.5, 0.2, 0.1), (6, 0.8, 0.5, 0.1, 0.05)])
    feature = build_combo_v3_frame_features(path)
    roi = build_roi_grid_v5_frame_features(path)
    hand = build_hand_v2_frame_features(path, expand=2.0)
    assert feature.shape == (COMBO_V3_DIM,)
    assert np.array_equal(feature[:ROI_GRID_V5_DIM], roi)
    assert np.array_equal(feature[ROI_GRID_V5_DIM:], hand)


def test_v4_layout_order_is_roi_global_hand(tmp_path):
    """combo-v4 的顺序为 [roi | 全画面 | 手部 v2]（全画面段由调用方传入）。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.3, 0.3, 0.1, 0.1)])
    roi = build_roi_grid_v5_frame_features(path)
    hand = build_hand_v2_frame_features(path, expand=1.5)
    synthetic_global = np.arange(40, dtype=np.float32)
    feature = build_combo_v4_frame_features(synthetic_global, roi, hand)
    assert feature.shape == (COMBO_V4_DIM,)
    assert np.array_equal(feature[:ROI_GRID_V5_DIM], roi)
    assert np.array_equal(feature[ROI_GRID_V5_DIM:ROI_GRID_V5_DIM + 40], synthetic_global)
    assert np.array_equal(feature[-40:], hand)


def test_part_width_mismatch_raises():
    """分段宽度不符时显式报错，不静默拼出错宽特征。

    ``combo-v3`` 的分段由内部固定 recipe 产出，无法从外部传入错宽；可注入分段的是
    ``combo-v4``（全画面段由调用方给），故这里只测它。
    """

    roi = np.zeros(ROI_GRID_V5_DIM, dtype=np.float32)
    with pytest.raises(ValueError, match=COMBO_V4_VERSION):
        build_combo_v4_frame_features(np.zeros(39, dtype=np.float32), roi,
                                      np.zeros(40, dtype=np.float32))


def test_missing_file_is_all_zero(tmp_path):
    """无检测框文件时两段全零 → 整帧全零。"""

    feature = build_combo_v3_frame_features(tmp_path / "nope.txt")
    assert feature.shape == (COMBO_V3_DIM,)
    assert not feature.any()


@pytest.mark.parametrize("version,dim", [
    (COMBO_V3_VERSION, COMBO_V3_DIM),
    (COMBO_V4_VERSION, COMBO_V4_DIM),
])
def test_load_split_real_data_width_matches_contract(version, dim):
    """真数据端到端：加载器分派出的宽度 == 契约维度、帧数 == v4.1 test 的 5,272。"""

    root = Path(__file__).resolve().parents[2] / "datasets/cleansight-ActionMixed-auto-lhh"
    if not root.is_dir():
        pytest.skip("本地缺少 datasets/cleansight-ActionMixed-auto-lhh")

    data_cfg = {"root": str(root), "dataset_ref": None, "split_train": "train",
                "split_val": "val", "split_eval": "test"}
    feats, truths, id2name = load_split(
        data_cfg, "test", feature_schema={"dim": dim, "version": version})
    assert len(feats) == 8
    assert all(f.shape[1] == dim for f in feats)
    assert sum(len(t) for t in truths) == 5272  # 数据 v4.1（15fps）test split
    assert len(id2name) == 6
