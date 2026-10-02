"""`build_roi_grid_delta_features`（ROI-144 + Δ 时间导数，240 维）的单元测试。

覆盖四类不变量：

1. **布局与维度**：每 (检测类, 区域) 5 通道 ``[presence, count, max_area, d_count, d_max_area]``，
   8 类 × 6 区域 × 5 = 240，且**通道最内**（不是"前 144 维 base + 后 96 维 delta"）。
2. **与 ROI-144 的逐位一致性**：前 3 个通道必须与 `build_roi_frame_features` **完全相同**
   ——否则它与 ROI-144 的对照就不是单变量对照，所有比较都失去意义。
3. **差分语义**：首帧 Δ=0（无前帧）；`0→1` 得 `+1`；`1→1` 得 `0`；只对 count/max_area 取 Δ。
4. **边界**：空序列、单帧序列都不得崩溃。
"""

import numpy as np
import pytest

from framework.cleansight_eval.temporal.features import (
    ROI_DELTA_DIM,
    ROI_DELTA_VERSION,
    block_dims_for_version,
    build_roi_frame_features,
    build_roi_grid_delta_features,
)
from framework.cleansight_eval.temporal.features.roi_bbox import (
    ROI_CHANNELS,
    ROI_DELTA_CHANNELS,
    ROI_FEATURE_DIM,
    ROI_N_REGIONS,
)


def _write_frames(tmp_path, rows_per_frame):
    """把每帧的 bbox 行写成 txt，返回帧路径列表。"""

    paths = []
    for index, rows in enumerate(rows_per_frame):
        path = tmp_path / f"frame{index:03d}.txt"
        path.write_text("\n".join(rows) + "\n" if rows else "\n")
        paths.append(path)
    return paths


def _view(array: np.ndarray, n_classes: int = 8) -> np.ndarray:
    """[T, 240] → [T, C, 区域, 5]，便于按 (类, 区域, 通道) 索引。"""

    return array.reshape(len(array), n_classes, ROI_N_REGIONS, ROI_DELTA_CHANNELS)


def test_dimension_and_channel_arithmetic():
    assert ROI_DELTA_DIM == 8 * ROI_N_REGIONS * ROI_DELTA_CHANNELS == 240
    assert ROI_DELTA_DIM == ROI_FEATURE_DIM + 8 * ROI_N_REGIONS * 2
    assert ROI_CHANNELS == 3 and ROI_DELTA_CHANNELS == 5


def test_base_channels_match_roi144_bit_for_bit(tmp_path):
    """前 3 通道必须与 ROI-144 逐位相同——这是"单变量对照"的前提。"""

    paths = _write_frames(tmp_path, [[], ["0 0.2 0.3 0.1 0.1"],
                                     ["0 0.2 0.3 0.1 0.1", "1 0.8 0.5 0.2 0.2"]])
    out = _view(build_roi_grid_delta_features(paths))
    for index, path in enumerate(paths):
        expected = build_roi_frame_features(path).reshape(8, ROI_N_REGIONS, ROI_CHANNELS)
        assert np.array_equal(out[index, :, :, :3], expected), f"第 {index} 帧 base 部分不一致"


def test_first_frame_delta_is_zero(tmp_path):
    """首帧没有前帧，Δ 必须约定为 0（而不是复制自身或 NaN）。"""

    paths = _write_frames(tmp_path, [["0 0.2 0.3 0.1 0.1"], ["0 0.2 0.3 0.1 0.1"]])
    view = _view(build_roi_grid_delta_features(paths))
    assert np.allclose(view[0, :, :, 3:], 0.0)


def test_delta_semantics_on_appear_and_hold(tmp_path):
    """`0→1` 的 count 增量为 +1；保持不变时为 0；area 同理。"""

    paths = _write_frames(tmp_path, [[], ["0 0.2 0.3 0.1 0.1"], ["0 0.2 0.3 0.1 0.1"]])
    view = _view(build_roi_grid_delta_features(paths))
    assert view[1, 0, 0, 3] == pytest.approx(1.0)      # 出现：count +1
    assert view[1, 0, 0, 4] == pytest.approx(0.01)     # 出现：max_area +0.01
    assert view[2, 0, 0, 3] == pytest.approx(0.0)      # 保持：count 增量 0
    assert view[2, 0, 0, 4] == pytest.approx(0.0)      # 保持：area 增量 0


def test_delta_captures_disappearance(tmp_path):
    """消失（1→0）必须给出 **负** 增量——这是"方向"信息的来源。"""

    paths = _write_frames(tmp_path, [["0 0.2 0.3 0.1 0.1"], []])
    view = _view(build_roi_grid_delta_features(paths))
    assert view[1, 0, 0, 3] == pytest.approx(-1.0)
    assert view[1, 0, 0, 4] == pytest.approx(-0.01)


def test_presence_channel_has_no_delta(tmp_path):
    """只对 count / max_area 取 Δ；presence 是二值量，其差分已由 d_count 承载。"""

    paths = _write_frames(tmp_path, [[], ["0 0.2 0.3 0.1 0.1"]])
    view = _view(build_roi_grid_delta_features(paths))
    assert view.shape[-1] == 5
    # 第 4/5 通道（索引 3、4）是 Δ；不存在第 6 个 Δpresence 通道。
    assert view[1, 0, 0, 3] != 0.0 and view[1, 0, 0, 4] != 0.0


def test_empty_and_single_frame_sequences(tmp_path):
    """空序列返回 (0, 240)；单帧序列 Δ 全零。"""

    empty = build_roi_grid_delta_features([])
    assert empty.shape == (0, ROI_DELTA_DIM)

    single = _view(build_roi_grid_delta_features(_write_frames(tmp_path, [["0 0.2 0.3 0.1 0.1"]])))
    assert single.shape[0] == 1
    assert np.allclose(single[0, :, :, 3:], 0.0)


def test_uniform_contract_needs_no_explicit_block_dims():
    """Δ 契约是**均匀**的（每类 30 维），因此不应返回显式块宽。

    随机遮罩按「总维 ÷ 类数」推导块宽，240 ÷ 8 = 30 正确；若此处误返回分组块宽，
    遮罩会静默遮错列。
    """

    assert block_dims_for_version(ROI_DELTA_VERSION, 8) is None
