"""ActionMixed bbox 帧 → **更高空间分辨率**的 ROI 特征（actionmixed-roi-grid-v3，159 维）。

**立项依据（2026-09-29 帧准确率攻关第 4 轮实测）**：把维度预算从"按类别表整齐分配"
改为"按证据在哪儿分配"能显著抬升帧准确率——

| 契约 | 高频 3 类每类块 | 低频 5 类每类块 | 维度 | 8 seed test `acc` 中位 |
|---|---|---:|---:|---:|
| `v1`（`roi_bbox.py`） | 2×3 网格 × 3 通道 = 18 | 18 | 144 | 53.92 |
| `v2`（`roi_bbox_v2.py`） | 3×3 网格 × 3 通道 = 27 | 3 | 96 | **56.48** |
| v2 + 遮掉低频 5 类 | 27 | 0 | 81（有效） | **56.91** |
| **本版 v3** | **4×4 网格 × 3 通道 = 48** | 3 | **159** | 见报告 |

配对检验：v2 与 v2-nocold 相对 v1 均 **8/8 胜、Wilcoxon p=0.0078**（Δ 中位 +2.37 / +2.82pp），
而两者彼此**无显著差异**（p=0.5469）——说明增益来自**高频类的空间分辨率**，
与低频类是否保留无关（低频类保持压缩，是为了不丢 syringe↔water_injection 这类稀疏但真实的信号）。

**与 v2 的唯一差别**：高频类网格由 3×3 提到 4×4（每类 27 → 48 维）。
其余语义（类顺序、通道语义、低频类全局 1 区域、因果无状态、空帧全零）与 v2 逐项一致，
因此这是一次**单变量外推**。

- presence: 该区域内该类是否有框（0/1）
- count:    该区域内该类的框数量
- max_area: 该区域内该类最大框面积（YOLO 归一化坐标下 w×h）

**块宽不等**（高频 48 / 低频 3），遮罩与随机遮罩必须用 :func:`roi_grid_v3_block_dims`
给出的显式块宽切片；按"总维 ÷ 类数"推导会静默遮错列（见 `INPUT_DESIGN_PROPOSAL.md` §4）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROI_GRID_V3_VERSION = "actionmixed-roi-grid-v3"
ROI_GRID_V3_CHANNELS = 3  # [presence, count, max_area]
# 高频类（frames/data.yaml 中的类别 ID）：hand / scope_control_body / scope_mid_section
ROI_GRID_V3_HOT_CLASS_IDS: tuple[int, ...] = (0, 1, 2)
ROI_GRID_V3_HOT_ROWS = 4
ROI_GRID_V3_HOT_COLS = 4
# 低频类仍合并为全局 1 区域（与 v2 相同，保证与 v2 的单变量对照）
ROI_GRID_V3_COLD_ROWS = 1
ROI_GRID_V3_COLD_COLS = 1
ROI_GRID_V3_HOT_BLOCK = ROI_GRID_V3_HOT_ROWS * ROI_GRID_V3_HOT_COLS * ROI_GRID_V3_CHANNELS  # 48
ROI_GRID_V3_COLD_BLOCK = ROI_GRID_V3_COLD_ROWS * ROI_GRID_V3_COLD_COLS * ROI_GRID_V3_CHANNELS  # 3
ROI_GRID_V3_N_CLASSES = 8  # 契约按 8 检测类定义（3 高频 + 5 低频）
ROI_GRID_V3_DIM = (
    len(ROI_GRID_V3_HOT_CLASS_IDS) * ROI_GRID_V3_HOT_BLOCK
    + (ROI_GRID_V3_N_CLASSES - len(ROI_GRID_V3_HOT_CLASS_IDS)) * ROI_GRID_V3_COLD_BLOCK
)  # = 3×48 + 5×3 = 159


def roi_grid_v3_block_dims(n_classes: int = ROI_GRID_V3_N_CLASSES) -> list[int]:
    """返回每类特征块宽（高频类 48 / 低频类 3），供遮罩与随机遮罩按类精确切片。

    契约按 8 类定义；``n_classes`` 只用于与数据集类别表交叉校验，不等于 8 时抛错，
    避免用错误类别数静默算出错误的块宽与总维度。
    """

    if n_classes != ROI_GRID_V3_N_CLASSES:
        raise ValueError(
            f"{ROI_GRID_V3_VERSION} 按 {ROI_GRID_V3_N_CLASSES} 个检测类定义，"
            f"实际数据集类别数={n_classes}"
        )
    return [
        ROI_GRID_V3_HOT_BLOCK if c in ROI_GRID_V3_HOT_CLASS_IDS else ROI_GRID_V3_COLD_BLOCK
        for c in range(n_classes)
    ]


def build_roi_grid_v3_frame_features(
    txt_path: Path,
    n_classes: int = ROI_GRID_V3_N_CLASSES,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """一帧 bbox → ``[159]`` 高分辨率 ROI 特征；``mask_target_ids`` 中的类整块保持为零。

    :param txt_path: 该帧的 YOLO 检测框文本（每行 ``class cx cy w h``，归一化坐标）；
        文件不存在视为空帧（全零）。
    :param n_classes: 数据集检测类别数，必须等于 8（契约按 8 类定义）。
    :param mask_target_ids: 需要整类置零的类别 ID 集合（目标遮罩）。
    :return: ``float32`` 一维向量，长度 :data:`ROI_GRID_V3_DIM`；每类块内为
        ``[presence, count, max_area]`` × 区域（高频类 16 区域行优先、低频类 1 区域）。
    """

    blocks = roi_grid_v3_block_dims(n_classes)
    offsets: list[int] = []
    running = 0
    for width in blocks:
        offsets.append(running)
        running += width

    n_hot_regions = ROI_GRID_V3_HOT_ROWS * ROI_GRID_V3_HOT_COLS
    counts = np.zeros((n_classes, n_hot_regions), dtype=np.float32)
    areas = np.zeros_like(counts)
    # 低频类只有 1 个区域，复用第 0 列，避免为两类布局各写一套索引
    if txt_path.exists():
        for line in txt_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
            if not (0 <= c < n_classes) or c in mask_target_ids:
                continue
            if c in ROI_GRID_V3_HOT_CLASS_IDS:
                row = min(int(cy * ROI_GRID_V3_HOT_ROWS), ROI_GRID_V3_HOT_ROWS - 1)
                col = min(int(cx * ROI_GRID_V3_HOT_COLS), ROI_GRID_V3_HOT_COLS - 1)
                region = row * ROI_GRID_V3_HOT_COLS + col
            else:
                region = 0  # 低频类合并为全局 1 区域
            counts[c, region] += 1.0
            area = w * h
            if area > areas[c, region]:
                areas[c, region] = area

    feat = np.zeros(ROI_GRID_V3_DIM, dtype=np.float32)
    for c in range(n_classes):
        n_regions = n_hot_regions if c in ROI_GRID_V3_HOT_CLASS_IDS else 1
        block = feat[offsets[c]: offsets[c] + blocks[c]].reshape(n_regions, ROI_GRID_V3_CHANNELS)
        block[:, 1] = counts[c, :n_regions]
        block[:, 2] = areas[c, :n_regions]
        block[:, 0] = block[:, 1] > 0  # presence = count > 0
    return feat
