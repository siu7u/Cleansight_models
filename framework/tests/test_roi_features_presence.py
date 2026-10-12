"""ROI presence 平面契约（actionmixed-roi-grid-presence-v1，48 维）的 feature 层单元测试。

验收标准（verification-first）：
- 维度 = 8 类 × 6 区域 × 1 通道 = 48，且与 catalog 登记一致；
- **不变式**：presence 契约的每一帧 == roi-grid-v1（144 维）的通道 0 切片
  （错位、漏通道、类序颠倒都会被这条抓住）；
- 区域编号仍是行优先 2×3，空 bbox → 全零，mask_target_ids 遮整类 6 维块；
- 均匀契约：块宽 = 48 ÷ 8 = 6（遮罩按此推导，无需 block_dims_for_version 特殊处理）。
"""

from __future__ import annotations

import numpy as np
import pytest

from cleansight_eval.temporal.features import (
    ROI_CHANNELS,
    ROI_FEATURE_DIM,
    ROI_FEATURE_VERSION,
    ROI_N_REGIONS,
    ROI_PRESENCE_DIM,
    ROI_PRESENCE_VERSION,
    build_roi_frame_features,
    build_roi_presence_frame_features,
)

N_CLASSES = 8
MIXED_FRAME = (
    "0 0.1 0.2 0.3 0.4\n"    # 类0 → 区域0（row0,col0）
    "0 0.5 0.6 0.1 0.2\n"    # 类0 → 区域4（row1,col1）
    "1 0.8 0.9 0.2 0.2\n"    # 类1 → 区域5（row1,col2）
    "1 0.8 0.9 0.5 0.5\n"    # 类1 → 区域5（重复框：count=2，presence 仍 1）
    "2 0.0 1.0 0.1 0.1\n"    # 类2 边界 cy=1.0 → 钳制区域3
    "7 0.02 0.02 0.4 0.4\n"  # 最后一个检测类 → 区域0（校验类序不错位）
)


def test_dimension_matches_contract():
    assert ROI_PRESENCE_DIM == 48
    assert ROI_PRESENCE_DIM == N_CLASSES * ROI_N_REGIONS
    assert ROI_PRESENCE_DIM == ROI_FEATURE_DIM // ROI_CHANNELS  # 144 / 3
    assert ROI_PRESENCE_VERSION != ROI_FEATURE_VERSION


def test_presence_equals_channel_zero_of_roi_v1(tmp_path):
    """核心不变式：presence 契约 == roi-grid-v1 的通道 0（逐帧）。"""

    path = tmp_path / "frame.txt"
    path.write_text(MIXED_FRAME, encoding="utf-8")

    full = build_roi_frame_features(path, n_classes=N_CLASSES)
    presence = build_roi_presence_frame_features(path, n_classes=N_CLASSES)

    expected = full.reshape(N_CLASSES, ROI_N_REGIONS, ROI_CHANNELS)[:, :, 0].reshape(-1)
    assert presence.shape == (ROI_PRESENCE_DIM,)
    np.testing.assert_array_equal(presence, expected)
    # 抽查两个具体位置：类0区域0 与 类7区域0 都应为 1
    grid = presence.reshape(N_CLASSES, ROI_N_REGIONS)
    assert grid[0, 0] == 1.0
    assert grid[7, 0] == 1.0
    assert grid[1, 5] == 1.0
    assert grid[0, 1:4].sum() == 0.0


def test_empty_frame_is_all_zero(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    presence = build_roi_presence_frame_features(path, n_classes=N_CLASSES)

    assert presence.shape == (ROI_PRESENCE_DIM,)
    np.testing.assert_array_equal(presence, np.zeros(ROI_PRESENCE_DIM, dtype=np.float32))


def test_missing_file_is_all_zero(tmp_path):
    presence = build_roi_presence_frame_features(tmp_path / "not-there.txt", n_classes=N_CLASSES)
    np.testing.assert_array_equal(presence, np.zeros(ROI_PRESENCE_DIM, dtype=np.float32))


def test_mask_targets_masks_whole_six_dim_class_block(tmp_path):
    """遮罩按整类块（6 维 = 6 区域 × 1 通道）生效，不越界到邻居类。"""

    path = tmp_path / "frame.txt"
    path.write_text(MIXED_FRAME, encoding="utf-8")

    presence = build_roi_presence_frame_features(
        path, n_classes=N_CLASSES, mask_target_ids=frozenset({0})
    ).reshape(N_CLASSES, ROI_N_REGIONS)

    np.testing.assert_array_equal(presence[0], np.zeros(ROI_N_REGIONS, dtype=np.float32))
    assert presence[1, 5] == 1.0  # 邻居类不受影响
    assert presence[7, 0] == 1.0


def test_presence_channel_is_binary(tmp_path):
    """presence 语义 = count > 0，因此取值只能是 0/1（重复框不叠成 2）。"""

    path = tmp_path / "frame.txt"
    path.write_text(MIXED_FRAME, encoding="utf-8")

    presence = build_roi_presence_frame_features(path, n_classes=N_CLASSES)

    assert set(np.unique(presence).tolist()) <= {0.0, 1.0}
