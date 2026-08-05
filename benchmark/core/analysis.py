"""评测分析（benchmark 域）：逐类阈值淘汰决策的纯函数。

与 framework 无依赖，可独立单测；``benchmark/cli/analyze.py`` 消费它。
"""

from __future__ import annotations

from typing import List, Tuple


def classify_classes(per_class: dict, threshold: float) -> Tuple[List[str], List[str], List[str]]:
    """
    将类别分为三组:
      - keep: P>=threshold 且 R>=threshold → 保留在 YOLO
      - borderline: P 或 R 在 [0.5*threshold, threshold) → 边界，可尝试优化
      - eliminate: P<0.5*threshold 或 R<0.5*threshold → 淘汰，转特征融合

    返回 (keep, borderline, eliminate)
    """

    keep, borderline, eliminate = [], [], []
    for cls_name, metrics in per_class.items():
        p = float(metrics.get("precision", 0.0))
        r = float(metrics.get("recall", 0.0))
        if "note" in metrics:
            eliminate.append(cls_name)
        elif p >= threshold and r >= threshold:
            keep.append(cls_name)
        elif p >= 0.5 * threshold and r >= 0.5 * threshold:
            borderline.append(cls_name)
        else:
            eliminate.append(cls_name)
    return keep, borderline, eliminate
