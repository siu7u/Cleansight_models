"""时序 prediction artifact 的边界、对齐和复算测试。"""

from __future__ import annotations

from benchmark.core.artifacts import (
    build_temporal_prediction_artifact,
    temporal_metrics_from_prediction_artifact,
)


def _artifact(predictions: dict[str, list[int]], truths: dict[str, list[int]]) -> dict:
    return build_temporal_prediction_artifact(
        pred_by_item=predictions,
        truth_by_item=truths,
        index_to_action={0: "Idle", 1: "Long_Brushing"},
        window=64,
        inference_mode="windowed_causal",
    )


def test_predictions_artifact_preserves_video_boundaries() -> None:
    artifact = _artifact(
        {"video-a": [1, 1], "video-b": [0]},
        {"video-a": [1, 0], "video-b": [0]},
    )

    assert artifact["schema_version"] == 1
    assert set(artifact["items"]) == {"video-a", "video-b"}
    assert artifact["items"]["video-a"]["prediction_start_frame"] == 63
    assert artifact["items"]["video-a"]["predicted_labels"] == [
        "Long_Brushing",
        "Long_Brushing",
    ]
    assert artifact["items"]["video-b"]["truth_labels"] == ["Idle"]


def test_metrics_can_be_recomputed_from_artifact() -> None:
    artifact = _artifact(
        {"video-a": [1, 1], "video-b": [1, 1]},
        {"video-a": [1, 1], "video-b": [1, 1]},
    )

    metrics = temporal_metrics_from_prediction_artifact(artifact, thresholds=(0.5,))

    assert metrics["segment"]["num_items"] == 2
    assert metrics["segment"]["details_at_iou"]["0.50"]["tp"] == 2
