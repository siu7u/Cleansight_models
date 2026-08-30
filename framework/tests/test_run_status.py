"""run 状态文件覆盖可靠训练的可诊断输出。"""

import json

import pytest
import torch

from cleansight_eval.core.run import RunContext
from cleansight_eval.temporal.sliding_window_pipeline import _loss_is_finite


def test_run_status_records_exception(tmp_path):
    run = RunContext(tmp_path, label="gru", run_id="manual-run")

    run.write_status("running", stage="initializing")
    running = json.loads(run.status_path.read_text(encoding="utf-8"))
    assert running["state"] == "running"
    assert running["stage"] == "initializing"

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        run.write_exception_status(exc, epoch=3)

    failed = json.loads(run.status_path.read_text(encoding="utf-8"))
    assert failed["state"] == "failed"
    assert failed["epoch"] == 3
    assert failed["error"]["type"] == "RuntimeError"
    assert failed["error"]["message"] == "boom"
    assert "traceback" in failed["error"]


def test_run_context_rejects_path_like_run_id(tmp_path):
    with pytest.raises(ValueError):
        RunContext(tmp_path, label="yolo", run_id="../outside")


@pytest.mark.parametrize("value,expected", [(1.0, True), (float("nan"), False), (float("inf"), False)])
def test_loss_finite_guard(value, expected):
    assert _loss_is_finite(torch.tensor(value)) is expected
