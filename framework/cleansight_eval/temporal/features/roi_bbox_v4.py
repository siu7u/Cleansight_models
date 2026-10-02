"""ActionMixed bbox 帧 → **通道语义替换**版 ROI 特征（actionmixed-roi-grid-v4，128 维）。

**立项依据（2026-09-29 第 6 轮源域实测）**：第 5 轮发现现行 v2 的 `presence` 通道
**完全冗余**（`presence ≡ count>0`：`2ch(count+max_area)` 与 `3ch` 的逐折 Δ 完全相同），
`max_area` 不可省（去掉 −2.63pp）。也就是说每个区域的**有效信息只有 2 个数**，
第三、四个通道槽位应当放**新信息**，而不是继续放冗余量。

源域筛选（auto train 14 视频、10 折视频级留出、HistGB；与 test 无关）比较了 11 种通道组合：

| 通道组合 | 维度 | Δ 中位 | 配对 vs `p,c,a` | p |
|---|---:|---:|---|---:|
| **`count,max_area,w,h`（本版）** | **128** | **−2.53** | **+0.82pp，8/2 胜** | **0.084** |
| `count,max_area,mean_area,dx,dy` | 160 | −2.45 | +0.83pp，6/4 | 0.432 |
| `count,max_area,mean_area` | 96 | −2.87 | +0.53pp，7/3 | 0.131 |
| `count,max_area,sum_area` | 96 | −3.29 | +0.35pp，7/2 | 0.055 |
| `p,c,a`（现行 v2，参考） | 96 | −3.40 | — | — |
| `c,a`（去掉冗余 presence） | 64 | −3.40 | ±0.00（逐折相同） | — |

**本版相对 v2 的单一变量**：把冗余的 `presence` 槽位换成**最大面积框的宽、高**（`w`、`h`）
——`max_area = w×h` 丢了**形状/长宽比**，而器械的长宽比是有判别力的（细长的内镜 vs 方正的刷子）。

**布局与 v2 逐项一致**：高频 3 类（hand / scope_control_body / scope_mid_section）3×3 网格
（9 区域），低频 5 类全局 1 区域；每类块宽 = 区域数 × 4。共 3×(9×4) + 5×(1×4) = **128 维**。
因果、无状态、逐帧独立，空帧全零；`mask_target_ids` 语义同前几版。

**块宽不等**（高频 36 / 低频 4），遮罩必须用 :func:`roi_grid_v4_block_dims` 的显式块宽切片。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROI_GRID_V4_VERSION = "actionmixed-roi-grid-v4"
# 通道语义（顺序即块内布局）：count / max_area / 最大面积框的宽 / 高
ROI_GRID_V4_CHANNELS: tuple[str, ...] = ("count", "max_area", "w", "h")
ROI_GRID_V4_HOT_CLASS_IDS: tuple[int, ...] = (0, 1, 2)
ROI_GRID_V4_HOT_ROWS = 3
ROI_GRID_V4_HOT_COLS = 3
ROI_GRID_V4_COLD_ROWS = 1
ROI_GRID_V4_COLD_COLS = 1
ROI_GRID_V4_N_CHANNELS = len(ROI_GRID_V4_CHANNELS)  # = 4
ROI_GRID_V4_HOT_BLOCK = ROI_GRID_V4_HOT_ROWS * ROI_GRID_V4_HOT_COLS * ROI_GRID_V4_N_CHANNELS  # 36
ROI_GRID_V4_COLD_BLOCK = ROI_GRID_V4_COLD_ROWS * ROI_GRID_V4_COLD_COLS * ROI_GRID_V4_N_CHANNELS  # 4
ROI_GRID_V4_N_CLASSES = 8
ROI_GRID_V4_DIM = (
    len(ROI_GRID_V4_HOT_CLASS_IDS) * ROI_GRID_V4_HOT_BLOCK
    + (ROI_GRID_V4_N_CLASSES - len(ROI_GRID_V4_HOT_CLASS_IDS)) * ROI_GRID_V4_COLD_BLOCK
)  # = 3×36 + 5×4 = 128


def roi_grid_v4_block_dims(n_classes: int = ROI_GRID_V4_N_CLASSES) -> list[int]:
    """返回每类特征块宽（高频类 36 / 低频类 4），供遮罩按类精确切片。"""

    if n_classes != ROI_GRID_V4_N_CLASSES:
        raise ValueError(
            f"{ROI_GRID_V4_VERSION} 按 {ROI_GRID_V4_N_CLASSES} 个检测类定义，"
            f"实际数据集类别数={n_classes}"
        )
    return [
        ROI_GRID_V4_HOT_BLOCK if c in ROI_GRID_V4_HOT_CLASS_IDS else ROI_GRID_V4_COLD_BLOCK
        for c in range(n_classes)
    ]


def build_roi_grid_v4_frame_features(
    txt_path: Path,
    n_classes: int = ROI_GRID_V4_N_CLASSES,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """一帧 bbox → ``[128]`` 通道替换版 ROI 特征；``mask_target_ids`` 中的类整块保持为零。

    :param txt_path: 该帧的 YOLO 检测框文本（每行 ``class cx cy w h``，归一化坐标）；
        文件不存在视为空帧（全零）。
    :param n_classes: 数据集检测类别数，必须等于 8（契约按 8 类定义）。
    :param mask_target_ids: 需要整类置零的类别 ID 集合（目标遮罩）。
    :return: ``float32`` 一维向量，长度 :data:`ROI_GRID_V4_DIM`；每类块内为
        ``[count, max_area, w, h]`` × 区域（高频类 9 区域行优先、低频类 1 区域）。
    """

    blocks = roi_grid_v4_block_dims(n_classes)
    offsets: list[int] = []
    running = 0
    for width in blocks:
        offsets.append(running)
        running += width

    n_hot_regions = ROI_GRID_V4_HOT_ROWS * ROI_GRID_V4_HOT_COLS
    counts = np.zeros((n_classes, n_hot_regions), dtype=np.float32)
    areas = np.zeros_like(counts)
    ws = np.zeros_like(counts)
    hs = np.zeros_like(counts)
    if txt_path.exists():
        for line in txt_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
            if not (0 <= c < n_classes) or c in mask_target_ids:
                continue
            if c in ROI_GRID_V4_HOT_CLASS_IDS:
                row = min(int(cy * ROI_GRID_V4_HOT_ROWS), ROI_GRID_V4_HOT_ROWS - 1)
                col = min(int(cx * ROI_GRID_V4_HOT_COLS), ROI_GRID_V4_HOT_COLS - 1)
                region = row * ROI_GRID_V4_HOT_COLS + col
            else:
                region = 0
            counts[c, region] += 1.0
            area = w * h
            if area > areas[c, region]:
                areas[c, region] = area
                ws[c, region] = w
                hs[c, region] = h

    feat = np.zeros(ROI_GRID_V4_DIM, dtype=np.float32)
    for c in range(n_classes):
        n_regions = n_hot_regions if c in ROI_GRID_V4_HOT_CLASS_IDS else 1
        block = feat[offsets[c]: offsets[c] + blocks[c]].reshape(n_regions, ROI_GRID_V4_N_CHANNELS)
        block[:, 0] = counts[c, :n_regions]
        block[:, 1] = areas[c, :n_regions]
        block[:, 2] = ws[c, :n_regions]
        block[:, 3] = hs[c, :n_regions]
    return feat
