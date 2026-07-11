"""评估结果的公共信封（framework 层，与模型语义无关）。

需求 §10 要求严格区分三类情况，本模块用 `MetricState` 表达：

- ``NOT_APPLICABLE``：指标不适用于当前任务/执行模式（例如离线模型的实时延迟）；
- ``MISSING``：指标适用，但评估所需输入缺失或运行失败；
- ``COMPUTED``：指标适用且已成功计算。

禁止用 ``0`` 冒充 ``NOT_APPLICABLE``，也禁止把 ``MISSING`` 伪装成 ``NOT_APPLICABLE``。
每个 ``MetricValue`` 都可携带口径版本 ``spec``，用于区分同名但口径不同的指标（§9.2）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class MetricState(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    MISSING = "missing"
    COMPUTED = "computed"


@dataclass
class MetricValue:
    """单个指标的三态取值。"""

    state: MetricState
    value: Any = None
    spec: str | None = None
    reason: str | None = None

    @classmethod
    def computed(cls, value: Any, spec: str | None = None) -> "MetricValue":
        return cls(MetricState.COMPUTED, value=value, spec=spec)

    @classmethod
    def not_applicable(cls, reason: str | None = None, spec: str | None = None) -> "MetricValue":
        return cls(MetricState.NOT_APPLICABLE, reason=reason, spec=spec)

    @classmethod
    def missing(cls, reason: str | None = None, spec: str | None = None) -> "MetricValue":
        return cls(MetricState.MISSING, reason=reason, spec=spec)

    def display(self) -> str:
        """人读矩阵单元格显示：区分 N/A、MISSING、已计算值。"""

        if self.state is MetricState.NOT_APPLICABLE:
            return "N/A"
        if self.state is MetricState.MISSING:
            return "MISSING"
        return str(self.value)

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "value": self.value,
            "spec": self.spec,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricValue":
        return cls(
            state=MetricState(data["state"]),
            value=data.get("value"),
            spec=data.get("spec"),
            reason=data.get("reason"),
        )


@dataclass
class EvalEnvelope:
    """一次评估运行的结构化产出（机读 + 人读的单一来源）。

    同一个 checkpoint 在不同喂入模式（full_sequence / windowed_causal）下会产生各自
    独立的信封；矩阵层再把它们横向汇总。
    """

    family: str
    model_id: str
    task: str
    feeding: str
    checkpoint: str
    dataset: str
    feature_schema: dict = field(default_factory=dict)
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    performance: dict[str, MetricValue] = field(default_factory=dict)
    feeding_semantics: dict = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)
    num_params: int | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "model_id": self.model_id,
            "task": self.task,
            "feeding": self.feeding,
            "checkpoint": self.checkpoint,
            "dataset": self.dataset,
            "feature_schema": self.feature_schema,
            "num_params": self.num_params,
            "timestamp": self.timestamp,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "performance": {k: v.to_dict() for k, v in self.performance.items()},
            "feeding_semantics": self.feeding_semantics,
            "integrity": self.integrity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalEnvelope":
        return cls(
            family=data["family"],
            model_id=data["model_id"],
            task=data["task"],
            feeding=data["feeding"],
            checkpoint=data["checkpoint"],
            dataset=data["dataset"],
            feature_schema=data.get("feature_schema", {}),
            num_params=data.get("num_params"),
            timestamp=data.get("timestamp"),
            metrics={k: MetricValue.from_dict(v) for k, v in data.get("metrics", {}).items()},
            performance={k: MetricValue.from_dict(v) for k, v in data.get("performance", {}).items()},
            feeding_semantics=data.get("feeding_semantics", {}),
            integrity=data.get("integrity", {}),
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> "EvalEnvelope":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
