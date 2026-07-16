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

from .execution import format_params


SCHEMA_VERSION = 2


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

    ``model_type`` 标识可替换的模型组件（gru / mstcn / yolo），``pipeline`` 标识它所属的
    完整流水线（sliding_window_temporal / full_sequence_temporal / detection）——后者已同时
    编码了域与推理方式，故不再单列 task/feeding。矩阵层把异构信封横向汇总。
    """

    model_type: str
    model_id: str
    pipeline: str
    checkpoint: str
    dataset: str
    feature_schema: dict = field(default_factory=dict)
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    performance: dict[str, MetricValue] = field(default_factory=dict)
    inference_semantics: dict = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)
    num_params: int | None = None
    timestamp: str | None = None
    run: dict = field(default_factory=dict)
    testset: dict = field(default_factory=dict)
    checkpoint_info: dict = field(default_factory=dict)
    metric_details: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    limits: dict = field(default_factory=lambda: {"is_smoke": False})
    pending_artifacts: dict = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict:
        """按 schema v2 输出；内部仍保留扁平属性以兼容报告和矩阵代码。"""

        checkpoint = {"path": self.checkpoint, **self.checkpoint_info}
        return {
            "schema_version": SCHEMA_VERSION,
            "result_type": "model_evaluation",
            "run": self.run or {"created_at": self.timestamp},
            "model": {
                "type": self.model_type,
                "id": self.model_id,
                "num_params": self.num_params,
                "checkpoint": checkpoint,
            },
            "pipeline": self.pipeline,
            "testset": self.testset or {"dataset_version": self.dataset},
            "feature_schema": self.feature_schema,
            "metrics": {
                "summary": {k: v.to_dict() for k, v in self.metrics.items()},
                "details": self.metric_details,
            },
            "performance": {k: v.to_dict() for k, v in self.performance.items()},
            "inference": self.inference_semantics,
            "artifacts": self.artifacts,
            "limits": self.limits,
            "integrity": self.integrity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalEnvelope":
        """读取 schema v2，并兼容没有 ``schema_version`` 的历史 v1 envelope。"""

        if int(data.get("schema_version", 1)) >= 2:
            model = data.get("model") or {}
            checkpoint = model.get("checkpoint") or {}
            if isinstance(checkpoint, str):
                checkpoint = {"path": checkpoint}
            metrics = data.get("metrics") or {}
            summary = metrics.get("summary", {}) if "summary" in metrics else metrics
            run = data.get("run") or {}
            testset = data.get("testset") or {}
            return cls(
                model_type=model["type"],
                model_id=model["id"],
                pipeline=data["pipeline"],
                checkpoint=checkpoint.get("path", ""),
                dataset=str(testset.get("dataset_version") or testset.get("id") or ""),
                feature_schema=data.get("feature_schema", {}),
                num_params=model.get("num_params"),
                timestamp=run.get("created_at"),
                metrics={k: MetricValue.from_dict(v) for k, v in summary.items()},
                performance={k: MetricValue.from_dict(v) for k, v in data.get("performance", {}).items()},
                inference_semantics=data.get("inference", {}),
                integrity=data.get("integrity", {}),
                run=run,
                testset=testset,
                checkpoint_info={k: v for k, v in checkpoint.items() if k != "path"},
                metric_details=metrics.get("details", {}),
                artifacts=data.get("artifacts", {}),
                limits=data.get("limits", {"is_smoke": False}),
            )
        return cls(
            model_type=data["model_type"],
            model_id=data["model_id"],
            pipeline=data["pipeline"],
            checkpoint=data["checkpoint"],
            dataset=data["dataset"],
            feature_schema=data.get("feature_schema", {}),
            num_params=data.get("num_params"),
            timestamp=data.get("timestamp"),
            metrics={k: MetricValue.from_dict(v) for k, v in data.get("metrics", {}).items()},
            performance={k: MetricValue.from_dict(v) for k, v in data.get("performance", {}).items()},
            inference_semantics=data.get("inference_semantics", {}),
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
