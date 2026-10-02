"""ROI 高分辨率分组特征（actionmixed-roi-grid-v3，159 维）的 feature 层单元测试。

覆盖：高频/低频类的区域与块偏移、4×4 网格的区域编号、空帧全零、整类遮罩、块宽表、
类别数校验、`block_dims_for_version` 分派（v2/v3 各自正确、v1 均匀契约返回 None）、
以及 `load_split` 能按本契约加载真实数据并给出 159 维。
"""

from pathlib import Path

import numpy as np
import pytest

from cleansight_eval.temporal.data import load_split
from cleansight_eval.temporal.features import (
    ROI_GRID_V3_DIM,
    ROI_GRID_V3_VERSION,
    block_dims_for_version,
    build_roi_grid_v3_frame_features,
    roi_grid_v3_block_dims,
)

# 块偏移：hand 0 / scope_control_body 48 / scope_mid_section 96 /
# scope_distal_end 144 / syringe 147 / air_gun 150 / short_brush 153 / brush_tip_out 156
BLOCK_OFFSETS = (0, 48, 96, 144, 147, 150, 153, 156)


def _write_frame(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    path.write_text("\n".join(
        f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in rows) + "\n", encoding="utf-8")


def test_block_dims_and_offsets():
    """块宽表为 3×48 + 5×3 = 159；类别数不等于 8 时直接报错而非静默算错。"""

    assert roi_grid_v3_block_dims(8) == [48, 48, 48, 3, 3, 3, 3, 3]
    assert sum(roi_grid_v3_block_dims(8)) == ROI_GRID_V3_DIM == 159
    assert block_dims_for_version(ROI_GRID_V3_VERSION, 8) == [48, 48, 48, 3, 3, 3, 3, 3]
    # v2 的分派不能被 v3 影响；v1 是均匀契约 → None
    assert block_dims_for_version("actionmixed-roi-grid-v2", 8) == [27, 27, 27, 3, 3, 3, 3, 3]
    assert block_dims_for_version("actionmixed-roi-grid-v1", 8) is None

    with pytest.raises(ValueError, match="8 个检测类"):
        roi_grid_v3_block_dims(3)


def test_missing_file_is_all_zero(tmp_path):
    """文件不存在视为空帧：全零、维度正确。"""

    feat = build_roi_grid_v3_frame_features(tmp_path / "nope.txt")
    assert feat.shape == (ROI_GRID_V3_DIM,)
    assert not feat.any()


def test_hot_class_region_indexing(tmp_path):
    """高频类按 4×4 行优先编号：cx=0.1/cy=0.1 → 区域 0；cx=0.9/cy=0.9 → 区域 15。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [(0, 0.1, 0.1, 0.2, 0.2), (0, 0.9, 0.9, 0.4, 0.5)])
    feat = build_roi_grid_v3_frame_features(path)
    hand = feat[BLOCK_OFFSETS[0]: BLOCK_OFFSETS[0] + 48].reshape(16, 3)
    assert hand[0].tolist() == [1.0, 1.0, pytest.approx(0.04)]
    assert hand[15].tolist() == [1.0, 1.0, pytest.approx(0.20)]
    assert int((hand[:, 0] > 0).sum()) == 2  # 其余区域无框


def test_cold_class_is_global(tmp_path):
    """低频类只有 1 个区域（块宽 3），坐标不影响区域选择。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [(4, 0.05, 0.95, 0.1, 0.1), (4, 0.55, 0.55, 0.2, 0.2)])
    feat = build_roi_grid_v3_frame_features(path)
    syringe = feat[BLOCK_OFFSETS[4]: BLOCK_OFFSETS[4] + 3]
    assert syringe.tolist() == [1.0, 2.0, pytest.approx(0.04)]  # presence / count / max_area


def test_mask_target_ids_zeroes_whole_class(tmp_path):
    """整类遮罩：被遮类的块全零，其它类不受影响。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [(0, 0.1, 0.1, 0.2, 0.2), (1, 0.5, 0.5, 0.3, 0.3)])
    feat = build_roi_grid_v3_frame_features(path, mask_target_ids=frozenset({0}))
    assert not feat[BLOCK_OFFSETS[0]: BLOCK_OFFSETS[0] + 48].any()
    assert feat[BLOCK_OFFSETS[1]: BLOCK_OFFSETS[1] + 48].any()


def test_load_split_real_data_gives_159_dims():
    """真数据端到端：按 v3 契约加载 test split，帧数 2639、每条 159 维。"""

    root = Path(__file__).resolve().parents[2] / "datasets/cleansight-ActionMixed-auto-lhh"
    if not root.is_dir():
        pytest.skip("本地缺少 datasets/cleansight-ActionMixed-auto-lhh")

    data_cfg = {"root": str(root), "dataset_ref": None, "split_train": "train",
                "split_val": "val", "split_eval": "test"}
    feats, truths, id2name = load_split(
        data_cfg, "test", feature_schema={"dim": ROI_GRID_V3_DIM, "version": ROI_GRID_V3_VERSION})
    assert len(feats) == 8
    assert all(f.shape[1] == ROI_GRID_V3_DIM for f in feats)
    assert sum(len(t) for t in truths) == 2639
    assert len(id2name) == 6
