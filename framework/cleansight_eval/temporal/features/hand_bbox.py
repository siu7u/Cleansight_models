"""ActionMixed bbox 帧 → 手部区域特征（actionmixed-bbox-hand-8cls-v1）。

与 ``actionmixed-bbox-8cls-v1``（整个画面内取每类最大框）不同，本 recipe 只关心
**手部周围**：以本帧面积最大的 hand 框（检测类 ID 0）为锚点，把该框绕中心扩张
``HAND_REGION_EXPAND`` 倍并钳制到画面内，得到手部区域；对每类取中心落在区域内的
最大面积框，编码 ``[presence, cx_rel, cy_rel, w_rel, h_rel]``：

- cx_rel/cy_rel: 框中心在手部区域内的相对坐标（0~1，区域左上角为原点）
- w_rel/h_rel:   框宽高相对手部区域的比值（框可越出区域，故可大于 1）

语义：presence = "该类目标是否出现在手部附近"，坐标 = "相对手的哪里"。无 hand 框时
整帧特征全零（与空帧语义一致）；hand 类自身恒有 presence=1 且位于区域中心。

本 recipe 因果、无状态、逐帧独立计算；修改扩张倍数或锚点规则 = 新 feature mapping
版本。"全局 + 手部"双通道契约（``actionmixed-bbox-global-hand-8cls-v1``，80 维）由
``temporal/data.py`` 以本 recipe 与 40 维全局 recipe 拼接而成，不在此实现。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HAND_BBOX_VERSION = "actionmixed-bbox-hand-8cls-v1"
GLOBAL_HAND_BBOX_VERSION = "actionmixed-bbox-global-hand-8cls-v1"
HAND_CLASS_ID = 0  # frames/data.yaml 中 hand 的类别 ID
HAND_REGION_EXPAND = 1.5  # 手部区域 = hand 框绕中心扩张的倍数（v1 固定）
HAND_FEATURE_DIM = 8 * 5  # = 40，与全局 bbox 契约同维度
GLOBAL_HAND_FEATURE_DIM = 8 * 5 * 2  # = 80，全局 + 手部拼接


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


def _hand_region(boxes: list[tuple[int, float, float, float, float]]) -> tuple[float, float, float, float] | None:
    """取面积最大 hand 框扩张后的区域 ``(x1, y1, x2, y2)``；无 hand 返回 None。"""

    hand: tuple[int, float, float, float, float] | None = None
    best_area = -1.0
    for box in boxes:
        c, cx, cy, w, h = box
        if c != HAND_CLASS_ID:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            hand = box
    if hand is None:
        return None
    _c, cx, cy, w, h = hand
    hw, hh = w / 2.0, h / 2.0
    x1 = max(0.0, cx - hw * HAND_REGION_EXPAND)
    y1 = max(0.0, cy - hh * HAND_REGION_EXPAND)
    x2 = min(1.0, cx + hw * HAND_REGION_EXPAND)
    y2 = min(1.0, cy + hh * HAND_REGION_EXPAND)
    return x1, y1, x2, y2


def build_hand_frame_features(
    txt_path: Path,
    n_classes: int = 8,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """一帧 bbox → ``[n_classes*5]`` 手部区域特征；指定目标的整类特征保持为零。

    无 hand 框时全零；hand 类自身编码为 ``[1, 0.5, 0.5, 1/扩张倍数, 1/扩张倍数]``
    （未受画面钳制时），其余类取中心落在手部区域内的最大面积框。
    """

    feat = np.zeros((n_classes, 5), dtype=np.float32)
    boxes = _parse_boxes(txt_path)
    region = _hand_region(boxes)
    if region is None:
        return feat.reshape(-1)
    x1, y1, x2, y2 = region
    rw, rh = x2 - x1, y2 - y1

    best_area = np.zeros(n_classes, dtype=np.float32)
    for c, cx, cy, w, h in boxes:
        if not (0 <= c < n_classes):
            continue
        if c in mask_target_ids:
            continue
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):  # 只看中心落入手部区域的框
            continue
        area = w * h
        if area >= best_area[c]:
            best_area[c] = area
            feat[c] = (
                1.0,
                (cx - x1) / rw,
                (cy - y1) / rh,
                w / rw,
                h / rh,
            )
    return feat.reshape(-1)  # [n_classes*5]
