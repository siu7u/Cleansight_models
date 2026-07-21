"""时序测试 timeline 只消费预测事实，不触发模型推理。"""

from pathlib import Path

import pytest

from cleansight_eval.core.execution import PredictionOutput
from cleansight_eval.temporal.viz import render_prediction_timeline


def _prediction_output() -> PredictionOutput:
    return PredictionOutput(
        model_type="gru",
        model_id="gru-test",
        pipeline="sliding_window_temporal",
        checkpoint="best.pt",
        dataset="fixture-v1",
        predictions={"video-a": ["idle", "brush", "brush"]},
        targets={"video-a": ["idle", "idle", "brush"]},
        labels=["idle", "brush"],
        metadata={"split": "test"},
    )


def test_render_prediction_timeline_writes_test_png(tmp_path):
    paths = render_prediction_timeline(_prediction_output(), out_dir=tmp_path)

    assert paths == [tmp_path / "segmentation-test-p01.png"]
    assert paths[0].is_file()


def test_render_prediction_timeline_rejects_misaligned_sequences(tmp_path):
    output = _prediction_output()
    output.targets["video-a"] = ["idle"]

    with pytest.raises(ValueError, match="预测/真值长度不同"):
        render_prediction_timeline(output, out_dir=tmp_path)
    assert not list(Path(tmp_path).glob("*.png"))
