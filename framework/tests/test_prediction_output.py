"""模型执行边界：PredictionOutput 不携带指标口径，pipeline 统一暴露 predict。"""

import json

import torch

from cleansight_eval.core.execution import PredictionOutput, sample_callable_latency
from cleansight_eval.core.pipeline import Pipeline
from cleansight_eval.cli._registry import get_visualizer
from cleansight_eval.detection.pipeline import DetectionPipeline
from cleansight_eval.temporal.full_sequence_pipeline import FullSequenceTemporalPipeline
from cleansight_eval.temporal.sliding_window_pipeline import SlidingWindowTemporalPipeline


def test_prediction_output_is_metrics_agnostic_and_serializable():
    output = PredictionOutput(
        model_type="gru",
        model_id="gru-51k",
        pipeline="sliding_window_temporal",
        checkpoint="best.pt",
        dataset="fixture-v1",
        predictions={"video-a": ["idle", "brush"]},
        targets={"video-a": ["idle", "idle"]},
        labels=["idle", "brush"],
        timing={"samples_ms": [1.0, 1.2]},
    )

    payload = output.to_dict()
    assert payload["predictions"]["video-a"] == ["idle", "brush"]
    assert "metrics" not in payload
    assert "metric_details" not in payload
    json.dumps(payload)


def test_all_pipelines_expose_predict_contract():
    for pipeline in (
        DetectionPipeline(),
        FullSequenceTemporalPipeline(),
        SlidingWindowTemporalPipeline(),
    ):
        assert callable(pipeline.predict)


def test_all_pipelines_inherit_pipeline_contract():
    for pipeline_type in (
        DetectionPipeline,
        FullSequenceTemporalPipeline,
        SlidingWindowTemporalPipeline,
    ):
        assert issubclass(pipeline_type, Pipeline)
        assert isinstance(pipeline_type(), Pipeline)


def test_temporal_pipelines_share_visualizer_registration():
    """两种时序喂入语义共享呈现器，检测流水线不被错误套用 timeline。"""

    full_sequence = get_visualizer("full_sequence_temporal")
    sliding_window = get_visualizer("sliding_window_temporal")
    assert callable(full_sequence)
    assert sliding_window is full_sequence
    assert get_visualizer("detection") is None


def test_latency_sampler_keeps_raw_samples_and_scope():
    calls = {"count": 0}

    def tick():
        calls["count"] += 1

    timing = sample_callable_latency(
        tick,
        torch.device("cpu"),
        warmup=2,
        runs=3,
        scope="model_forward_test",
        context={"input_shape": [1, 4, 8]},
    )

    assert calls["count"] == 5
    assert timing["scope"] == "model_forward_test"
    assert timing["runs"] == 3
    assert len(timing["samples_ms"]) == 3
    assert timing["context"]["input_shape"] == [1, 4, 8]
