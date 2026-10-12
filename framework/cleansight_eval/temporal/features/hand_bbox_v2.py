"""手部区域 recipe **v2**：双锚点（全部 hand 框的并集）+ 可配扩张倍数。

**为什么要改**：v4.1 全量重训显示**手部区域特征是帧准确率的最强驱动**——全画面 40 维
`actionmixed-bbox-8cls-v1` acc **69.38**，加上手部段后（80 维）acc **76.27**（+6.89pp）。
但 v1 recipe 有两个**从未调过**的选择（`features/hand_bbox.py` 里写死）：

1. 锚点 = **面积最大的单个 hand 框**——左右手都检出时，区域只覆盖其中一只，
   **换手/换器械的瞬间区域会跳**；
2. `HAND_REGION_EXPAND = 1.5`——v1 之后没动过。

**源域筛选（2026-10-11，只碰 train：28 视频 / 27,092 帧、视频级 10 折留出、HistGB，
`tmp/v41push/hand_region_screen.py`）** —— Δ = 留出视频逐帧 acc − 该折全 idle 基线（中位）：

| 变体 | 维度 | Δ中位 | 配对 vs v1（1.5×最大框） |
|---|---:|---:|---|
| v1 全局+手部 1.5 × 最大框（现行） | 80 | +11.64 | — |
| 全局+手部 1.0 × 并集 | 80 | +12.25 | **+1.83pp，10/0，p=0.002** |
| 全局+手部 1.25 × 并集 | 80 | **+12.78** | **+1.20pp，10/0，p=0.002** |
| 全局+手部 2.0 × 并集 | 80 | +12.00 | +1.25pp，8/2，p=0.014 |
| 仅手部 1.5 × 最大框（v1） | 40 | +6.56 | −4.11pp，1/9，p=0.006 |
| 仅手部 2.0 × 并集 | 40 | +14.65 | +1.86pp，8/2，p=0.027 |
| 仅手部 3.0 × 并集 | 40 | **+15.08** | **+2.39pp，8/2，p=0.020** |

**结论**：锚点从"最大 hand 框"改成"**全部 hand 框的并集**"在两个维度上都带来源域增益
（10/0 或 8/2 折，p≤0.03）；仅手部 recipe 在并集锚点下从 +6.56 跳到 **+14.65**
（并集锚点等于把左右手都圈进区域，器械在手间传递时区域不再跳）。

**本模块定义的两个契约**（扩张倍数按源域筛选各自取点）：

| 契约版本 | 维度 | 锚点 | 扩张 | 对照臂（单变量差异） |
|---|---:|---|---:|---|
| `actionmixed-bbox-hand-8cls-v2` | 40 | 全部 hand 框并集 | 2.0 | `feat-hand`（v1：1.5 × 最大框） |
| `actionmixed-bbox-global-hand-8cls-v2` | 80 | 全部 hand 框并集 | 1.5 | `feat-global-hand`（v1：仅锚点不同） |

语义与 v1 完全一致（`[presence, cx_rel, cy_rel, w_rel, h_rel]` 每类 5 维、只看中心落在区域内的
框、无 hand 时整帧全零）；**只有区域的定义变了**。因果、无状态、逐帧独立；改锚点或扩张倍数
= 新 feature mapping 版本。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HAND_BBOX_V2_VERSION = "actionmixed-bbox-hand-8cls-v2"
GLOBAL_HAND_V2_VERSION = "actionmixed-bbox-global-hand-8cls-v2"
HAND_CLASS_ID = 0  # 与 v1 一致：`frames/data.yaml` 中 hand 的类别 ID
HAND_V2_EXPAND = 2.0        # 仅手部契约的扩张倍数（源域最优点附近）
GLOBAL_HAND_V2_EXPAND = 1.5  # 全局+手部契约沿用 v1 的倍数 → 与 v1 只差锚点
HAND_V2_DIM = 8 * 5         # = 40
GLOBAL_HAND_V2_DIM = 8 * 5 * 2  # = 80（全画面 40 ‖ 手部 40）


def _parse_boxes(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    """读一帧 bbox 文本，返回 ``[(class, cx, cy, w, h)]``（跳过非法行）。"""

    boxes: list[tuple[int, float, float, float, float]] = []
    if txt_path.exists():
        for line in txt_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
            boxes.append((c, cx, cy, w, h))
    return boxes


def _union_hand_region(
    boxes: list[tuple[int, float, float, float, float]], expand: float
) -> tuple[float, float, float, float] | None:
    """**全部 hand 框并集**绕其中心扩张 ``expand`` 倍后的区域 ``(x1, y1, x2, y2)``。

    只有 1 个 hand 框时与 v1 的"最大框 + 同倍数"逐位等价；无 hand 返回 None。
    区域按 0~1 画面钳制（钳制后区域比值会变，模型看到的是钳制后的相对坐标）。
    """

    hands = [(cx, cy, w, h) for c, cx, cy, w, h in boxes if c == HAND_CLASS_ID]
    if not hands:
        return None
    x1r = min(cx - w / 2 for cx, _cy, w, _h in hands)
    x2r = max(cx + w / 2 for cx, _cy, w, _h in hands)
    y1r = min(cy - h / 2 for _cx, cy, _w, h in hands)
    y2r = max(cy + h / 2 for _cx, cy, _w, h in hands)
    center_x, center_y = (x1r + x2r) / 2, (y1r + y2r) / 2
    half_w = (x2r - x1r) / 2 * expand
    half_h = (y2r - y1r) / 2 * expand
    return (max(0.0, center_x - half_w), max(0.0, center_y - half_h),
            min(1.0, center_x + half_w), min(1.0, center_y + half_h))


def build_hand_v2_frame_features(
    txt_path: Path,
    n_classes: int = 8,
    mask_target_ids: frozenset[int] = frozenset(),
    expand: float = HAND_V2_EXPAND,
) -> np.ndarray:
    """一帧 bbox → ``[n_classes*5]`` 手部区域 v2 特征（并集锚点）。

    :param txt_path: 该帧的 YOLO 检测框文本；文件不存在或无 hand 框时整帧全零。
    :param n_classes: 检测类别数（默认 8）。
    :param mask_target_ids: 类级遮罩，命中的类整块保持为零（与 v1 同义）。
    :param expand: 区域扩张倍数；**改这个值等于改 feature mapping 版本**，
        契约常量见 :data:`HAND_V2_EXPAND` / :data:`GLOBAL_HAND_V2_EXPAND`。
    :return: ``float32`` 一维向量，长度 ``n_classes*5``。
    """

    feat = np.zeros((n_classes, 5), dtype=np.float32)
    boxes = _parse_boxes(txt_path)
    region = _union_hand_region(boxes, expand)
    if region is None:
        return feat.reshape(-1)
    x1, y1, x2, y2 = region
    rw, rh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)

    best_area = np.zeros(n_classes, dtype=np.float32)
    for c, cx, cy, w, h in boxes:
        if not (0 <= c < n_classes) or c in mask_target_ids:
            continue
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):  # 只看中心落入区域的框
            continue
        area = w * h
        if area >= best_area[c]:
            best_area[c] = area
            feat[c] = (1.0, (cx - x1) / rw, (cy - y1) / rh, w / rw, h / rh)
    return feat.reshape(-1)
