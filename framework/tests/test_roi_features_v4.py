"""ROI 通道替换特征（actionmixed-roi-grid-v4，128 维）的 feature 层单元测试。

覆盖：块宽表与偏移、4 通道语义（count / max_area / 宽 / 高）、3×3 网格区域编号、
低频类全局 1 区、空帧全零、整类遮罩、类别数校验、`block_dims_for_version` 分派
（v2/v3/v4 各自正确、v1 均匀契约返回 None），以及真数据端到端 128 维。
"""

from pathlib import Path

import numpy as np
import pytest

from cleansight_eval.temporal.data import load_split
from cleansight_eval.temporal.features import (
    ROI_GRID_V4_DIM,
    ROI_GRID_V4_VERSION,
    block_dims_for_version,
    build_roi_grid_v4_frame_features,
    roi_grid_v4_block_dims,
)

# 块偏移：hand 0 / scope_control_body 36 / scope_mid_section 72 /
# scope_distal_end 108 / syringe 112 / air_gun 116 / short_brush 120 / brush_tip_out 124
BLOCK_OFFSETS = (0, 36, 72, 108, 112, 116, 120, 124)


def _write_frame(path: Path, rows) -> None:
    path.write_text("\n".join(f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in rows) + "\n",
                    encoding="utf-8")


def test_block_dims_and_offsets():
    """块宽表为 3×36 + 5×4 = 128；类别数不等于 8 时报错。"""

    assert roi_grid_v4_block_dims(8) == [36, 36, 36, 4, 4, 4, 4, 4]
    assert sum(roi_grid_v4_block_dims(8)) == ROI_GRID_V4_DIM == 128
    assert block_dims_for_version(ROI_GRID_V4_VERSION, 8) == [36, 36, 36, 4, 4, 4, 4, 4]
    # 其它契约的分派不受影响
    assert block_dims_for_version("actionmixed-roi-grid-v2", 8) == [27, 27, 27, 3, 3, 3, 3, 3]
    assert block_dims_for_version("actionmixed-roi-grid-v3", 8) == [48, 48, 48, 3, 3, 3, 3, 3]
    assert block_dims_for_version("actionmixed-roi-grid-v1", 8) is None

    with pytest.raises(ValueError, match="8 个检测类"):
        roi_grid_v4_block_dims(3)


def test_missing_file_is_all_zero(tmp_path):
    feat = build_roi_grid_v4_frame_features(tmp_path / "nope.txt")
    assert feat.shape == (ROI_GRID_V4_DIM,)
    assert not feat.any()


def test_channels_are_count_area_w_h(tmp_path):
    """4 通道依次是 count / max_area / 宽 / 高，且取**面积最大**的那个框的宽高。"""

    path = tmp_path / "f.txt"
    # 同一格内两个 hand 框：小框面积 0.01、大框面积 0.08 → max_area/宽/高 取大框的
    _write_frame(path, [(0, 0.1, 0.1, 0.1, 0.1), (0, 0.12, 0.12, 0.4, 0.2)])
    feat = build_roi_grid_v4_frame_features(path)
    hand = feat[BLOCK_OFFSETS[0]: BLOCK_OFFSETS[0] + 36].reshape(9, 4)
    assert hand[0][0] == pytest.approx(2.0)          # count
    assert hand[0][1] == pytest.approx(0.08)         # max_area = 0.4×0.2
    assert hand[0][2] == pytest.approx(0.4)          # 大框的宽
    assert hand[0][3] == pytest.approx(0.2)          # 大框的高
    assert int((hand[:, 0] > 0).sum()) == 1          # 只落在一个格


def test_hot_grid_region_indexing(tmp_path):
    """高频类按 3×3 行优先编号：cx=0.9/cy=0.9 → 区域 8。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [(0, 0.9, 0.9, 0.1, 0.1)])
    feat = build_roi_grid_v4_frame_features(path)
    hand = feat[BLOCK_OFFSETS[0]: BLOCK_OFFSETS[0] + 36].reshape(9, 4)
    assert hand[8][0] == pytest.approx(1.0)
    assert hand[0][0] == 0.0


def test_cold_class_is_global(tmp_path):
    """低频类只有 1 个区域（块宽 4），坐标不影响区域选择。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [(4, 0.05, 0.95, 0.1, 0.2), (4, 0.55, 0.55, 0.3, 0.3)])
    feat = build_roi_grid_v4_frame_features(path)
    syringe = feat[BLOCK_OFFSETS[4]: BLOCK_OFFSETS[4] + 4]
    assert syringe[0] == pytest.approx(2.0)   # count
    assert syringe[1] == pytest.approx(0.09)  # max_area
    assert syringe[2] == pytest.approx(0.3)   # 宽
    assert syringe[3] == pytest.approx(0.3)   # 高


def test_mask_target_ids_zeroes_whole_class(tmp_path):
    path = tmp_path / "f.txt"
    _write_frame(path, [(0, 0.1, 0.1, 0.2, 0.2), (1, 0.5, 0.5, 0.3, 0.3)])
    feat = build_roi_grid_v4_frame_features(path, mask_target_ids=frozenset({0}))
    assert not feat[BLOCK_OFFSETS[0]: BLOCK_OFFSETS[0] + 36].any()
    assert feat[BLOCK_OFFSETS[1]: BLOCK_OFFSETS[1] + 36].any()


def test_load_split_real_data_gives_128_dims():
    """真数据端到端：按 v4 契约加载 test split，8 视频 / 2639 帧 / 每条 128 维。"""

    root = Path(__file__).resolve().parents[2] / "datasets/cleansight-ActionMixed-auto-lhh"
    if not root.is_dir():
        pytest.skip("本地缺少 datasets/cleansight-ActionMixed-auto-lhh")

    data_cfg = {"root": str(root), "dataset_ref": None, "split_train": "train",
                "split_val": "val", "split_eval": "test"}
    feats, truths, id2name = load_split(
        data_cfg, "test", feature_schema={"dim": ROI_GRID_V4_DIM, "version": ROI_GRID_V4_VERSION})
    assert len(feats) == 8
    assert all(f.shape[1] == ROI_GRID_V4_DIM for f in feats)
    assert sum(len(t) for t in truths) == 2639
    assert len(id2name) == 6
