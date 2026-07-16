"""检测 pipeline 到 benchmark prediction artifact 的薄适配。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from benchmark.core.artifacts import build_detection_prediction_artifact
except ModuleNotFoundError:  # pragma: no cover - 从 framework 目录执行时触发
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from benchmark.core.artifacts import build_detection_prediction_artifact


def build_prediction_artifact(
    items: Mapping[str, Mapping[str, Any]],
    labels: Mapping[int | str, str],
    *,
    split: str,
    prediction_format: str,
) -> dict:
    """把 YOLOAdapter 的逐图输出转换为 benchmark 唯一检测 artifact schema。"""

    return build_detection_prediction_artifact(
        items=items,
        labels=labels,
        split=split,
        prediction_format=prediction_format,
    )
