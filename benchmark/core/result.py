"""benchmark 评估结果的唯一 schema 真源。

schema v2 同时承载模型评估事实和 benchmark 门禁结果：模型评估使用三态 ``MetricValue``；
只有确实执行门禁时才写 ``decision``。framework 只运行模型并通过兼容模块使用这里的结果类型，
不再维护第二套 envelope 定义。
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
ALLOWED_STATUS = {"PASS", "FAIL", "PENDING", "EXPLORATORY"}


class MetricState(str, Enum):
    """指标三态：已计算、不适用、应有但缺失。"""

    NOT_APPLICABLE = "not_applicable"
    MISSING = "missing"
    COMPUTED = "computed"


@dataclass
class MetricValue:
    """一个带状态、口径版本和缺失原因的指标值。"""

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
        if self.state is MetricState.NOT_APPLICABLE:
            return "N/A"
        if self.state is MetricState.MISSING:
            return "MISSING"
        return str(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "value": self.value,
            "spec": self.spec,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricValue":
        return cls(
            state=MetricState(data["state"]),
            value=data.get("value"),
            spec=data.get("spec"),
            reason=data.get("reason"),
        )


@dataclass
class EvaluationResult:
    """一次单模型评估的结构化结果。

    扁平属性供 framework 的报告和矩阵代码使用；``to_dict`` 统一输出 schema v2。模型、
    checkpoint、testset、feature schema、推理方式和 artifact 均保留版本追溯信息。
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
    decision: dict = field(default_factory=dict)
    benchmark: str | None = None
    task_type: str | None = None
    pending_artifacts: dict = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """输出正式 schema v2；空的门禁字段不写，避免把模型事实伪装成 PENDING。"""

        checkpoint = {"path": self.checkpoint, **self.checkpoint_info}
        payload: dict[str, Any] = {
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
                "summary": {key: value.to_dict() for key, value in self.metrics.items()},
                "details": self.metric_details,
            },
            "performance": {key: value.to_dict() for key, value in self.performance.items()},
            "inference": self.inference_semantics,
            "artifacts": self.artifacts,
            "limits": self.limits,
            "integrity": self.integrity,
        }
        if self.benchmark:
            payload["benchmark"] = self.benchmark
        if self.task_type:
            payload["task_type"] = self.task_type
        if self.decision:
            payload["decision"] = self.decision
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationResult":
        """读取 schema v2，并兼容 framework 历史无版本扁平 envelope。"""

        version = int(data.get("schema_version", LEGACY_SCHEMA_VERSION))
        if version == SCHEMA_VERSION:
            validate_result(data)
            if data.get("result_type") not in (None, "model_evaluation"):
                raise ValueError(f"EvaluationResult 不支持 result_type={data.get('result_type')!r}")
            model = data.get("model") or {}
            checkpoint = model.get("checkpoint") or {}
            if isinstance(checkpoint, str):
                checkpoint = {"path": checkpoint}
            metrics = data.get("metrics") or {}
            summary = metrics.get("summary", {}) if "summary" in metrics else metrics
            run = data.get("run") or {}
            testset = data.get("testset") or {}
            return cls(
                model_type=str(model.get("type") or ""),
                model_id=str(model.get("id") or ""),
                pipeline=str(data.get("pipeline") or ""),
                checkpoint=str(checkpoint.get("path") or ""),
                dataset=str(testset.get("dataset_version") or testset.get("id") or ""),
                feature_schema=dict(data.get("feature_schema") or {}),
                num_params=model.get("num_params"),
                timestamp=run.get("created_at"),
                metrics={key: MetricValue.from_dict(value) for key, value in summary.items()},
                performance={
                    key: MetricValue.from_dict(value)
                    for key, value in (data.get("performance") or {}).items()
                },
                inference_semantics=dict(data.get("inference") or {}),
                integrity=dict(data.get("integrity") or {}),
                run=dict(run),
                testset=dict(testset),
                checkpoint_info={key: value for key, value in checkpoint.items() if key != "path"},
                metric_details=dict(metrics.get("details") or {}),
                artifacts=dict(data.get("artifacts") or {}),
                limits=dict(data.get("limits") or {"is_smoke": False}),
                decision=dict(data.get("decision") or {}),
                benchmark=data.get("benchmark"),
                task_type=data.get("task_type"),
            )

        if version != LEGACY_SCHEMA_VERSION:
            raise ValueError(f"不支持 schema_version={version}")

        # framework schema v1：无 schema_version，模型身份和指标均为顶层字段。
        return cls(
            model_type=str(data["model_type"]),
            model_id=str(data["model_id"]),
            pipeline=str(data["pipeline"]),
            checkpoint=str(data["checkpoint"]),
            dataset=str(data["dataset"]),
            feature_schema=dict(data.get("feature_schema") or {}),
            num_params=data.get("num_params"),
            timestamp=data.get("timestamp"),
            metrics={key: MetricValue.from_dict(value) for key, value in (data.get("metrics") or {}).items()},
            performance={
                key: MetricValue.from_dict(value)
                for key, value in (data.get("performance") or {}).items()
            },
            inference_semantics=dict(data.get("inference_semantics") or {}),
            integrity=dict(data.get("integrity") or {}),
        )

    def write(self, path: str | Path) -> Path:
        """确定性写出 UTF-8 JSON。"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        validate_result(payload)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> "EvaluationResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def make_run_id(prefix: str) -> str:
    """生成可用于文件名与 CARD marker 的 UTC 运行编号。"""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", prefix).strip("-") or "evaluation"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slug}-{timestamp}"


def _benchmark_metrics(metrics: Mapping[str, Any], benchmark: str) -> dict[str, Any]:
    """把旧 benchmark 任意 metrics 映射为 v2 summary/details，不伪造复杂结构的标量含义。"""

    metric_spec = metrics.get("metric_spec") if isinstance(metrics.get("metric_spec"), Mapping) else {}
    summary: dict[str, dict[str, Any]] = {}
    details: dict[str, Any] = {}
    for name, value in metrics.items():
        if name == "metric_spec":
            details[name] = value
        elif isinstance(value, (str, int, float, bool)):
            spec = metric_spec.get(name) or f"benchmark/{benchmark}/v1"
            summary[name] = MetricValue.computed(value, spec=spec).to_dict()
        elif value is None:
            summary[name] = MetricValue.missing("benchmark 未产出该值").to_dict()
        else:
            details[name] = value
    return {"summary": summary, "details": details}


def build_result(
    *,
    benchmark: str,
    task_type: str,
    run_id: str,
    model: dict[str, Any] | None,
    testset: dict[str, Any],
    inference: dict[str, Any],
    metrics: dict[str, Any],
    status: str,
    reasons: list[str] | None = None,
    limits: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造通用 benchmark 结果 v2；旧 ``gates`` 输入映射为可选 ``decision``。"""

    result = {
        "schema_version": SCHEMA_VERSION,
        "result_type": "benchmark_evaluation",
        "benchmark": benchmark,
        "task_type": task_type,
        "run": {
            "id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "model": model,
        "testset": testset,
        "inference": inference,
        "metrics": _benchmark_metrics(metrics, benchmark),
        "performance": {},
        "limits": limits or {"is_smoke": False},
        "decision": {"status": status, "reasons": reasons or []},
        "artifacts": artifacts or {},
        "integrity": {},
    }
    validate_result(result)
    return result


def validate_result(result: Mapping[str, Any]) -> None:
    """校验正式 schema v2；模型事实与 benchmark 门禁按 result_type 分别检查。"""

    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"不支持 schema_version={result.get('schema_version')}")
    required = {
        "schema_version",
        "result_type",
        "run",
        "model",
        "testset",
        "inference",
        "metrics",
        "performance",
        "limits",
        "artifacts",
        "integrity",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"评估结果缺少字段: {missing}")

    result_type = result.get("result_type")
    if result_type == "benchmark_evaluation":
        for key in ("benchmark", "task_type", "decision"):
            if key not in result:
                raise ValueError(f"benchmark 结果缺少 {key}")
        status = (result.get("decision") or {}).get("status")
        if status not in ALLOWED_STATUS:
            raise ValueError(f"非法 decision.status={status}")
        testset = result.get("testset") or {}
        for key in ("id", "dataset_version", "split", "manifest_sha256"):
            if not testset.get(key):
                raise ValueError(f"testset 缺少 {key}")
    elif result_type == "model_evaluation":
        if not result.get("pipeline"):
            raise ValueError("model_evaluation 缺少 pipeline")
        model = result.get("model") or {}
        for key in ("type", "id"):
            if not model.get(key):
                raise ValueError(f"model 缺少 {key}")
        checkpoint = model.get("checkpoint") or {}
        checkpoint_path = checkpoint.get("path") if isinstance(checkpoint, Mapping) else checkpoint
        if not checkpoint_path:
            raise ValueError("model 缺少 checkpoint.path")
        decision = result.get("decision") or {}
        if decision and decision.get("status") not in ALLOWED_STATUS:
            raise ValueError(f"非法 decision.status={decision.get('status')}")
    else:
        raise ValueError(f"不支持 result_type={result_type!r}")

    metric_block = result.get("metrics") or {}
    if not isinstance(metric_block.get("summary", {}), Mapping):
        raise ValueError("metrics.summary 必须是映射")
    for name, value in metric_block.get("summary", {}).items():
        try:
            MetricValue.from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"指标 {name} 不是合法三态值: {exc}") from exc


def upgrade_legacy_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """把 benchmark schema v1 门禁结果转换为 v2；输入不原地修改。"""

    if result.get("schema_version") == SCHEMA_VERSION:
        upgraded = copy.deepcopy(dict(result))
        validate_result(upgraded)
        return upgraded
    if result.get("schema_version", LEGACY_SCHEMA_VERSION) != LEGACY_SCHEMA_VERSION:
        raise ValueError(f"不支持 schema_version={result.get('schema_version')}")
    if not result.get("benchmark") or not result.get("task_type"):
        raise ValueError("不是 benchmark schema v1 结果")
    decision = result.get("gates") or {}
    upgraded = build_result(
        benchmark=str(result["benchmark"]),
        task_type=str(result["task_type"]),
        run_id=str((result.get("run") or {}).get("id") or make_run_id(str(result["benchmark"]))),
        model=copy.deepcopy(result.get("model")),
        testset=copy.deepcopy(result.get("testset") or {}),
        inference=copy.deepcopy(result.get("inference") or {}),
        metrics=copy.deepcopy(result.get("metrics") or {}),
        status=str(decision.get("status") or "PENDING"),
        reasons=list(decision.get("reasons") or []),
        limits=copy.deepcopy(result.get("limits") or {}),
        artifacts=copy.deepcopy(result.get("artifacts") or {}),
    )
    legacy_created_at = (result.get("run") or {}).get("created_at")
    if legacy_created_at:
        upgraded["run"]["created_at"] = legacy_created_at
    validate_result(upgraded)
    return upgraded


def write_result(path: str | Path, result: Mapping[str, Any] | EvaluationResult) -> Path:
    """校验后写出 UTF-8 JSON，供报告、CARD、release gate 和归档共用。"""

    payload = result.to_dict() if isinstance(result, EvaluationResult) else dict(result)
    validate_result(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
