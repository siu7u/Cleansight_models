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


ROI_PRESENCE_VERSION = "actionmixed-roi-grid-presence-v1"
ROI_PRESENCE_DIM = 8 * ROI_N_REGIONS  # = 48（8 类 × 6 区域的 presence 平面）


def build_roi_presence_frame_features(
    txt_path: Path,
    n_classes: int = 8,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """一帧 bbox → ``[n_classes * 区域数]`` 的 **presence 平面**（等价 ROI-144 的通道 0）。

    立项依据（第十七轮 §17.4）：ROI-144 的三通道高度冗余（逐 (类,区域) |Pearson r| =
    0.990 / 0.926 / 0.920），且 ``insert`` / ``sbc`` 的判别信号集中在 presence 通道
    （LDA 探针：insert 41.3 → 68.8、sbc 16.7 → 73.0），presence-only 48 维的探针 edit（14.60）
    还高于全量 144 维（9.01）。本契约把 144 维压成 48 维，用于验证"省 2/3 输入维度不掉指标"。

    语义与 ROI-144 完全一致（同一份 frames/ 检测框、同样的 2×3 行优先区域划分、同样的
    ``mask_target_ids`` 整类遮罩），只是丢弃 ``count`` 与 ``max_area`` 两个通道。
    空 bbox 文件 → 全零 48 维。
    """

    full = build_roi_frame_features(txt_path, n_classes=n_classes, mask_target_ids=mask_target_ids)
    presence = full.reshape(n_classes, ROI_N_REGIONS, ROI_CHANNELS)[:, :, 0]
    return np.ascontiguousarray(presence.reshape(-1), dtype=np.float32)
