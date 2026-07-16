"""framework 的评估结果兼容导出。

正式 schema 和三态指标均由 ``benchmark.core.result`` 唯一定义。本模块只保留历史 import 路径
``EvalEnvelope``，使现有 pipeline、报告、矩阵和外部脚本可以渐进迁移。
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from benchmark.core.result import (
        SCHEMA_VERSION,
        EvaluationResult,
        MetricState,
        MetricValue,
    )
except ModuleNotFoundError:  # pragma: no cover - 从 framework 目录执行时触发
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from benchmark.core.result import (
        SCHEMA_VERSION,
        EvaluationResult,
        MetricState,
        MetricValue,
    )


# 历史名称只作为类型别名；不再拥有独立实现或 schema。
EvalEnvelope = EvaluationResult

__all__ = [
    "SCHEMA_VERSION",
    "EvaluationResult",
    "EvalEnvelope",
    "MetricState",
    "MetricValue",
]
