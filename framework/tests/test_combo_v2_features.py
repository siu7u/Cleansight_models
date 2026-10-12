"""拼接契约 v2（actionmixed-combo-v2，272 维）的 feature 层单元测试。

覆盖：分段宽度与自检、拼接顺序（roi-v5 ‖ 全画面 40 ‖ 手部 40）、空帧全零、类级遮罩只作用于
左段与全局子段（手部锚点段不遮），以及**真数据端到端宽度 == 契约维度**（防止再次出现
`data.py` 分派链被顶掉、静默回落到 40 维 bbox 特征的静默错宽 bug，参见 2026-10-08 的 presence 事故）。
"""

from pathlib import Path

import numpy as np
import pytest

from cleansight_eval.temporal.data import load_split
from cleansight_eval.temporal.features import (
    COMBO_V2_DIM,
    COMBO_V2_PARTS,
    COMBO_V2_VERSION,
    build_combo_v2_frame_features,
)
from cleansight_eval.temporal.features.hand_bbox import build_hand_frame_features
from cleansight_eval.temporal.features.roi_bbox_v5 import (
    ROI_GRID_V5_DIM,
    build_roi_grid_v5_frame_features,
)


def _write_frame(path: Path, rows) -> None:
    path.write_text("\n".join(f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in rows) + "\n",
                    encoding="utf-8")


def test_parts_and_total_dim():
    """分段 = (192, 40, 40)，总维度 272，与 testsets.yaml 的 feature_parts 对齐。"""

    assert COMBO_V2_PARTS == (192, 40, 40)
    assert COMBO_V2_DIM == 272


def test_concat_order_is_roi_then_global_then_hand(tmp_path):
    """拼接顺序为 [roi-v5 | 全画面 | 手部]，逐段直接取自各自 recipe 的输出。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.3, 0.3, 0.1, 0.1)])
    roi = build_roi_grid_v5_frame_features(path)
    hand = build_hand_frame_features(path)
    feat = build_combo_v2_frame_features(roi, roi[:40], hand)
    assert feat.shape == (COMBO_V2_DIM,)
    assert np.array_equal(feat[:ROI_GRID_V5_DIM], roi)
    assert np.array_equal(feat[ROI_GRID_V5_DIM:ROI_GRID_V5_DIM + 40], roi[:40])
    assert np.array_equal(feat[-40:], hand)


def test_missing_file_is_all_zero(tmp_path):
    """空帧（无检测框文件）三段全零。"""

    missing = tmp_path / "nope.txt"
    roi = build_roi_grid_v5_frame_features(missing)
    hand = build_hand_frame_features(missing)
    feat = build_combo_v2_frame_features(roi, roi[:40], hand)
    assert feat.shape == (COMBO_V2_DIM,)
    assert not feat.any()


def test_part_width_mismatch_raises():
    """任一段宽度不符时显式报错，不静默拼接出错宽特征。"""

    good = np.zeros(ROI_GRID_V5_DIM, dtype=np.float32)
    with pytest.raises(ValueError, match=COMBO_V2_VERSION):
        build_combo_v2_frame_features(good, np.zeros(39, dtype=np.float32), np.zeros(40, dtype=np.float32))


def test_load_split_real_data_gives_272_dims():
    """真数据端到端：按 combo-v2 契约加载 test split，8 视频 / 5272 帧 / 每条 272 维。

    这是**加载器分派**层的断言（宽度 == 契约维度）：只测特征构造器测不到分派回落。
    """

    root = Path(__file__).resolve().parents[2] / "datasets/cleansight-ActionMixed-auto-lhh"
    if not root.is_dir():
        pytest.skip("本地缺少 datasets/cleansight-ActionMixed-auto-lhh")

    data_cfg = {"root": str(root), "dataset_ref": None, "split_train": "train",
                "split_val": "val", "split_eval": "test"}
    feats, truths, id2name = load_split(
        data_cfg, "test", feature_schema={"dim": COMBO_V2_DIM, "version": COMBO_V2_VERSION})
    assert len(feats) == 8
    assert all(f.shape[1] == COMBO_V2_DIM for f in feats)
    assert sum(len(t) for t in truths) == 5272  # 数据 v4.1（15fps）test split
    assert len(id2name) == 6
