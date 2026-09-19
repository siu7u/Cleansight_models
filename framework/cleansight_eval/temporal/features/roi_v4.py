"""ROI 网格 v4（6 核心类，108 维）——废弃目标剔除版。

与 v1（8 类 144 维）的差别：读入层把 8 类检测 id 重映射为 v4 的 6 类表
（丢弃 scope_distal_end=3、short_brush=6），网格（2x3）与通道（3）不变。
重映射：hand:0->0, ctrl:1->1, mid:2->2, syringe:4->3, air_gun:5->4, brush_tip_out:7->5。
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from .roi_bbox import build_roi_frame_features as _build_roi_v1

ROI_V4_VERSION = "ama-v4-roi-6c-108d"
ROI_V4_FEATURE_DIM = 108
V4_CLASS_REMAP = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 7: 5, 3: None, 6: None}

def _remap_frame_file(src: Path, dst: Path) -> None:
    """一帧 8 类 bbox 重写为 v4 6 类（废弃类行丢弃）；无检测则空文件。"""
    lines = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                mapped = V4_CLASS_REMAP.get(cid)
                if mapped is None:
                    continue
                lines.append(" ".join([str(mapped)] + parts[1:5]))
    dst.write_text("\n".join(lines), encoding="utf-8")

def build_roi_v4_frame_features(
    txt_path: Path,
    *,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """一帧 8 类 bbox -> v4 [108]（6 类 x 6 区域 x 3 通道）。mask 按 8 类原 id 传入。"""
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / txt_path.name
        _remap_frame_file(txt_path, tmp_path)
        v4_mask = frozenset(
            m for cid in mask_target_ids
            if (m := V4_CLASS_REMAP.get(cid)) is not None
        )
        feats = _build_roi_v1(tmp_path, n_classes=6, mask_target_ids=v4_mask)
    if feats.shape[0] != ROI_V4_FEATURE_DIM:
        raise AssertionError(f"roi v4 dim {feats.shape[0]} != {ROI_V4_FEATURE_DIM}")
    return feats