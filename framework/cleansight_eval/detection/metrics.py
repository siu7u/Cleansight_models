"""检测指标历史兼容入口；唯一实现位于 benchmark.evaluators.detection。"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from benchmark.evaluators.detection import (
        SPEC_MAP50,
        SPEC_MAP50_95,
        SPEC_PRECISION,
        SPEC_RECALL,
        build_detection_metrics,
    )
except ModuleNotFoundError:  # pragma: no cover - 从 framework 目录执行时触发
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from benchmark.evaluators.detection import (
        SPEC_MAP50,
        SPEC_MAP50_95,
        SPEC_PRECISION,
        SPEC_RECALL,
        build_detection_metrics,
    )

__all__ = [
    "SPEC_MAP50",
    "SPEC_MAP50_95",
    "SPEC_PRECISION",
    "SPEC_RECALL",
    "build_detection_metrics",
]
