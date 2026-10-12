"""ActionMixed bbox 帧 → 可见性重排 ROI 特征（actionmixed-roi-grid-v2，96 维）。

与 ``actionmixed-roi-grid-v1``（8 类 × 2×3 网格 × 3 通道 = 144 维，**按类别表整齐分配**）
不同，v2 按实测可见性分配维度预算（依据：``docs/features/INPUT_DESIGN_PROPOSAL.md``
§1.1 检测覆盖、§1.3 通道体检 54/144 恒零、§2 P2b 设计）：

- **高频 3 类**（hand / scope_control_body / scope_mid_section，出现率 95.2%/91.4%/69.8%）：
  3×3 网格 × 3 通道 = 每类 27 维，提高空间分辨率；
- **低频 5 类**（scope_distal_end / syringe / air_gun / short_brush / brush_tip_out，
  出现率 7.9%/2.0%/0.3%/2.0%/0.4%）：合并为**全局 1 区域** × 3 通道 = 每类 3 维。
  不直接删低频类的原因：它们在关键动作上有 4~10× 富集（syringe ↔ water_injection、
  short_brush ↔ short_brush_cleaning），信号真实但稀疏；散在 6 个区域时每格几乎恒零，
  合并成全局通道反而更稠密、更抗漏检。

布局 class-major，类顺序同 ``frames/data.yaml``（hand, scope_control_body, scope_mid_section,
scope_distal_end, syringe, air_gun, short_brush, brush_tip_out）：3 类 × 27 + 5 类 × 3 = 96 维。
每类块内通道语义同 v1：``[presence, count, max_area]``，区域按行优先编号。

- presence: 该区域内该类是否有框（0/1）
- count:    该区域内该类的框数量（原始计数，小整数）
- max_area: 该区域内该类最大框面积（YOLO 归一化坐标下 w×h，0~1）

空 bbox 文件 → 全零；坐标越界按网格边界钳制。本 recipe 因果、无状态、逐帧独立计算，
离线/在线口径一致，只改变特征契约、不改原始数据。

**块宽不等的注意**：本契约每类块宽为 27（高频）或 3（低频），不是"总维 ÷ 类数"的均匀切分，
所以遮罩/随机遮罩必须用 :func:`roi_grid_v2_block_dims` 给出的显式块宽切片——按均匀块宽
推导会静默遮错列（见 ``INPUT_DESIGN_PROPOSAL.md`` §4 工程注意）。
修改网格、类分组或通道数 = 新 feature mapping 版本。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROI_GRID_V2_VERSION = "actionmixed-roi-grid-v2"
ROI_GRID_V2_CHANNELS = 3  # [presence, count, max_area]
# 高频类（frames/data.yaml 中的类别 ID）：hand / scope_control_body / scope_mid_section
ROI_GRID_V2_HOT_CLASS_IDS: tuple[int, ...] = (0, 1, 2)
ROI_GRID_V2_HOT_ROWS = 3
ROI_GRID_V2_HOT_COLS = 3
# 低频类合并为全局 1 区域
ROI_GRID_V2_COLD_ROWS = 1
ROI_GRID_V2_COLD_COLS = 1
ROI_GRID_V2_HOT_BLOCK = (
    ROI_GRID_V2_HOT_ROWS * ROI_GRID_V2_HOT_COLS * ROI_GRID_V2_CHANNELS
)  # = 27
ROI_GRID_V2_COLD_BLOCK = (
    ROI_GRID_V2_COLD_ROWS * ROI_GRID_V2_COLD_COLS * ROI_GRID_V2_CHANNELS
)  # = 3
ROI_GRID_V2_N_CLASSES = 8  # 契约按 8 检测类定义（3 高频 + 5 低频）
ROI_GRID_V2_DIM = (
    len(ROI_GRID_V2_HOT_CLASS_IDS) * ROI_GRID_V2_HOT_BLOCK
    + (ROI_GRID_V2_N_CLASSES - len(ROI_GRID_V2_HOT_CLASS_IDS)) * ROI_GRID_V2_COLD_BLOCK
)  # = 3×27 + 5×3 = 96


def roi_grid_v2_block_dims(n_classes: int = ROI_GRID_V2_N_CLASSES) -> list[int]:
    """返回每类特征块宽（高频类 27 / 低频类 3），供遮罩与随机遮罩按类精确切片。

    契约按 8 类定义；``n_classes`` 只用于与数据集类别表交叉校验，不等于 8 时抛错，
    避免用错误类别数静默算出错误的块宽与总维度。
    """

    if n_classes != ROI_GRID_V2_N_CLASSES:
        raise ValueError(
            f"{ROI_GRID_V2_VERSION} 按 {ROI_GRID_V2_N_CLASSES} 个检测类定义，"
            f"实际数据集类别数={n_classes}"
        )
    return [
        ROI_GRID_V2_HOT_BLOCK if c in ROI_GRID_V2_HOT_CLASS_IDS else ROI_GRID_V2_COLD_BLOCK
        for c in range(n_classes)
    ]


def build_roi_grid_v2_frame_features(
    txt_path: Path,
    n_classes: int = ROI_GRID_V2_N_CLASSES,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """一帧 bbox → ``[96]`` 可见性重排 ROI 特征；``mask_target_ids`` 中的类整块保持为零。

    参数：
        txt_path: 该帧的 YOLO 检测框文本（每行 ``class cx cy w h``，归一化坐标）；
            文件不存在视为空帧（全零）。
        n_classes: 数据集检测类别数，必须等于 8（契约按 8 类定义）。
        mask_target_ids: 需要整类置零的类别 ID 集合（目标遮罩）。

    返回：``float32`` 一维向量，长度 :data:`ROI_GRID_V2_DIM`；每类块内为
    ``[presence, count, max_area]`` × 区域（高频类 9 区域行优先、低频类 1 区域）。
    """

    blocks = roi_grid_v2_block_dims(n_classes)
    offsets: list[int] = []
    running = 0
    for width in blocks:
        offsets.append(running)
        running += width

    counts = np.zeros((n_classes, ROI_GRID_V2_HOT_ROWS * ROI_GRID_V2_HOT_COLS), dtype=np.float32)
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
            if c in ROI_GRID_V2_HOT_CLASS_IDS:
                row = min(int(cy * ROI_GRID_V2_HOT_ROWS), ROI_GRID_V2_HOT_ROWS - 1)
                col = min(int(cx * ROI_GRID_V2_HOT_COLS), ROI_GRID_V2_HOT_COLS - 1)
                region = row * ROI_GRID_V2_HOT_COLS + col
            else:
                region = 0  # 低频类合并为全局 1 区域
            counts[c, region] += 1.0
            area = w * h
            if area > areas[c, region]:
                areas[c, region] = area

    feat = np.zeros(ROI_GRID_V2_DIM, dtype=np.float32)
    n_hot_regions = ROI_GRID_V2_HOT_ROWS * ROI_GRID_V2_HOT_COLS
    for c in range(n_classes):
        n_regions = n_hot_regions if c in ROI_GRID_V2_HOT_CLASS_IDS else 1
        block = feat[offsets[c] : offsets[c] + blocks[c]].reshape(n_regions, ROI_GRID_V2_CHANNELS)
        block[:, 1] = counts[c, :n_regions]
        block[:, 2] = areas[c, :n_regions]
        block[:, 0] = block[:, 1] > 0  # presence = count > 0
    return feat
