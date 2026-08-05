"""benchmark 对时序与检测 prediction artifact 的统一校验测试。"""

import pytest

from benchmark.core.artifacts import (
    build_detection_prediction_artifact,
    build_temporal_prediction_artifact,
    prediction_artifact_recomputable,
    validate_prediction_artifact,
)


def test_detection_artifact_is_valid_but_requires_external_truth_to_recompute():
    artifact = build_detection_prediction_artifact(
        items={
            "frame-0001.jpg": {
                "predictions": [
                    {"class_id": 0, "confidence": 0.9, "xywhn": [0.5, 0.5, 0.2, 0.3]}
                ]
            }
        },
        labels={0: "hand"},
        split="test",
    )

    validate_prediction_artifact(artifact)
    assert artifact["task_type"] == "detection"
    assert prediction_artifact_recomputable(artifact) is None


def test_detection_artifact_rejects_unknown_class():
    with pytest.raises(ValueError, match="未登记类别"):
        build_detection_prediction_artifact(
            items={
                "frame.jpg": {
                    "predictions": [
                        {"class_id": 2, "confidence": 0.8, "xywhn": [0.5, 0.5, 0.2, 0.2]}
                    ]
                }
            },
            labels={0: "hand"},
            split="test",
        )


def test_temporal_artifact_remains_self_recomputable():
    artifact = build_temporal_prediction_artifact(
        pred_by_item={"video-a": [0, 1]},
        truth_by_item={"video-a": [0, 1]},
        index_to_action={0: "idle", 1: "brush"},
        window=64,
        inference_mode="windowed_causal",
    )

    assert prediction_artifact_recomputable(artifact) is True
