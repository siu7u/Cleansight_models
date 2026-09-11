"""ActionMixed bbox 帧 → ROI 区域特征（actionmixed-roi-grid-v1）。

与 ``actionmixed-bbox-8cls-v1``（每类取最大框编码 [presence, cx, cy, w, h]）不同，
本 recipe 丢弃框内精确坐标，把画面按固定 2×3 网格划分为 6 个区域，对每个
(检测类, 区域) 统计三通道：

    [presence, count, max_area]

- presence: 该区域内该检测类是否有框（0/1）
- count:    该区域内该检测类的框数量（原始计数，小整数）
- max_area: 该区域内该检测类的最大框面积（YOLO 归一化坐标下的 w×h，0~1）

布局为 class-major：每类 18 维（6 区域 × 3 通道，区域按行优先 row-major），
8 类拼接成 144 维。空 bbox 文件 → 全零 144 维。坐标越界时按网格边界钳制。

本 recipe 因果、无状态，逐帧独立计算，离线/在线口径一致；只改变特征契约，
不改变原始数据（同一份 frames/ 检测框）。修改网格或通道数 = 新 feature mapping
版本（见 modelset-quality 的 Feature Mapping Rules）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROI_FEATURE_VERSION = "actionmixed-roi-grid-v1"
ROI_GRID_ROWS = 2
ROI_GRID_COLS = 3
ROI_CHANNELS = 3  # [presence, count, max_area]
ROI_N_REGIONS = ROI_GRID_ROWS * ROI_GRID_COLS  # 6
ROI_FEATURE_DIM = 8 * ROI_N_REGIONS * ROI_CHANNELS  # = 144


def build_roi_frame_features(
    txt_path: Path,
    n_classes: int = 8,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """一帧 bbox → ``[n_classes * 区域数 * 3]`` ROI 特征；指定目标的整类特征保持为零。

    每个 (类, 区域) 输出 ``[presence, count, max_area]``；区域按行优先编号。
    """

    feat = np.zeros((n_classes, ROI_N_REGIONS, ROI_CHANNELS), dtype=np.float32)
    if txt_path.exists():
        for line in txt_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
            if not (0 <= c < n_classes):
                continue
            if c in mask_target_ids:
                continue
            row = min(int(cy * ROI_GRID_ROWS), ROI_GRID_ROWS - 1)
            col = min(int(cx * ROI_GRID_COLS), ROI_GRID_COLS - 1)
            region = row * ROI_GRID_COLS + col
            feat[c, region, 1] += 1.0
            area = w * h
            if area > feat[c, region, 2]:
                feat[c, region, 2] = area
    feat[:, :, 0] = feat[:, :, 1] > 0  # presence = count > 0
    return feat.reshape(-1)  # [n_classes * 区域数 * 3]
