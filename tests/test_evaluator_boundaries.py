from pathlib import Path

from benchmark.evaluators import evaluate_prediction
from benchmark.core.result import MetricState


def test_framework_pipelines_do_not_own_formal_evaluate_method():
    from framework.cleansight_eval.detection.pipeline import DetectionPipeline
    from framework.cleansight_eval.temporal.full_sequence_pipeline import FullSequenceTemporalPipeline
    from framework.cleansight_eval.temporal.sliding_window_pipeline import SlidingWindowTemporalPipeline

    for pipeline in (
        DetectionPipeline(),
        FullSequenceTemporalPipeline(),
        SlidingWindowTemporalPipeline(),
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
    assert result.metrics["f1@0.5"].spec.endswith("v4; source=benchmark.core.metrics")
    assert result.metric_details["temporal"]["metric_spec"]["version"] == "interval-matching-v2"
    assert result.pending_artifacts["predictions"]["task_type"] == "temporal"

    evaluator_sources = Path("benchmark/evaluators").glob("*.py")
    assert all("cleansight_eval" not in path.read_text(encoding="utf-8") for path in evaluator_sources)
