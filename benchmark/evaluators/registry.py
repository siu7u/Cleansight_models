"""PredictionOutput.pipeline 到 benchmark evaluator 的唯一分派表。"""

from __future__ import annotations

from typing import Any, Mapping

from . import classification, detection, temporal


_EVALUATORS = {
    "detection": detection.evaluate,
    "full_sequence_temporal": temporal.evaluate,
    "sliding_window_temporal": temporal.evaluate,
    "roi_classification": classification.evaluate,
}


def _pipeline_name(output: Any) -> str:
    if isinstance(output, Mapping):
        return str(output.get("pipeline") or "")
    return str(getattr(output, "pipeline", ""))


def get_evaluator(pipeline: str):
    """取得指定 pipeline 的事实评估器。"""

    if pipeline not in _EVALUATORS:
        raise KeyError(f"未注册的 benchmark evaluator: {pipeline}；已注册: {sorted(_EVALUATORS)}")
    return _EVALUATORS[pipeline]


def evaluate_prediction(output: Any, options: Mapping[str, Any] | None = None):
    """按 PredictionOutput.pipeline 分派，不要求 benchmark import framework 类型。"""

    return get_evaluator(_pipeline_name(output))(output, options)
