import ast
from pathlib import Path

import pytest

from benchmark.evaluators import evaluate_prediction
from benchmark.core.result import MetricState

EVALUATORS_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "evaluators"


def test_framework_pipelines_do_not_own_formal_evaluate_method():
    pytest.importorskip("torch")  # framework core 模块级依赖 torch；GPU/完整环境全量验证
    from framework.cleansight_eval.classification.pipeline import ClassificationPipeline
    from framework.cleansight_eval.detection.pipeline import DetectionPipeline
    from framework.cleansight_eval.temporal.full_sequence_pipeline import FullSequenceTemporalPipeline
    from framework.cleansight_eval.temporal.sliding_window_pipeline import SlidingWindowTemporalPipeline

    for pipeline in (
        DetectionPipeline(),
        FullSequenceTemporalPipeline(),
        SlidingWindowTemporalPipeline(),
        ClassificationPipeline(),
    ):
        assert callable(pipeline.predict)
        assert not hasattr(pipeline, "evaluate")


def test_benchmark_evaluator_consumes_plain_mapping_without_framework_import():
    output = {
        "model_type": "gru",
        "model_id": "gru-test",
        "pipeline": "full_sequence_temporal",
        "checkpoint": "best.pt",
        "dataset": "fixture-v1",
        "predictions": {"video-a": ["idle", "brush"]},
        "targets": {"video-a": ["idle", "brush"]},
        "labels": ["idle", "brush"],
        "feature_schema": {"dim": 2, "version": "fixture-v1"},
        "inference_semantics": {"mode": "full_sequence"},
        "num_params": 10,
        "metadata": {},
    }
    result = evaluate_prediction(output)
    assert result.metrics["acc"].state is MetricState.COMPUTED
    assert result.metrics["acc"].value == 100.0
    assert result.metrics["f1@0.5"].spec.endswith("v4; source=framework.cleansight_eval.core.metrics")
    assert result.metric_details["temporal"]["metric_spec"]["version"] == "interval-matching-v2"
    assert result.pending_artifacts["predictions"]["task_type"] == "temporal"

    # evaluator 只消费 plain mapping，不得 import framework 的类型/流水线；
    # 但可以消费 framework core 的纯指标/数据原语（benchmark → framework 单向依赖）。
    forbidden_framework_modules = (
        "framework.cleansight_eval.core.execution",
        "framework.cleansight_eval.core.pipeline",
        "framework.cleansight_eval.detection",
        "framework.cleansight_eval.temporal",
        "framework.cleansight_eval.classification",
    )
    allowed_suffix = "framework.cleansight_eval.core.metrics"
    for path in EVALUATORS_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _assert_allowed(alias.name, path, forbidden_framework_modules, allowed_suffix)
            elif isinstance(node, ast.ImportFrom) and node.module:
                _assert_allowed(node.module, path, forbidden_framework_modules, allowed_suffix)


def _assert_allowed(module, path, forbidden, allowed_suffix):
    if module in forbidden or module.startswith(tuple(f"{item}." for item in forbidden)):
        raise AssertionError(f"{path.relative_to(Path.cwd())} 不得 import framework 类型: {module}")
