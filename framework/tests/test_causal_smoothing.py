"""滑窗因果平滑 ``causal_decision`` 的最小持续时长语义与配置接线测试。

背景（2026-09-18 跨批次诊断）：``min_duration`` 同时是**召回上限**——真实段短于该值的
动作在输出里不可能出现（project-18 test 的 short_brush_cleaning 六段全部 ≤19 帧，
min_duration=25 下召回恒为 0）。因此它从硬编码改为可由
``evaluation.smoothing_min_duration`` 覆盖，这里锁住默认行为与覆盖行为。
"""

import torch

from cleansight_eval.core.config import load_config, validate_config
from cleansight_eval.temporal.util import causal_decision


def _logits(class_id: int, num_classes: int = 6, peak: float = 5.0) -> torch.Tensor:
    """构造"某一类显著占优"的单帧 logits。"""

    out = torch.zeros(num_classes)
    out[class_id] = peak
    return out


def _simulate(min_duration: int, class_id: int = 3, frames: int = 10) -> list[int]:
    """连续 frames 帧都指向同一类，返回逐帧的 stable 输出。"""

    pending, stable, count, out = None, 0, 0, []
    for _ in range(frames):
        pending, stable, count = causal_decision(
            _logits(class_id), pending, stable, count, min_duration=min_duration
        )
        out.append(stable)
    return out


def test_default_min_duration_is_25():
    """默认（不传 min_duration）保持历史行为：25 帧内不切换。"""

    assert _simulate(25) == [0] * 10
    # 显式不传参数与传 25 等价
    pending, stable, count, out = None, 0, 0, []
    for _ in range(30):
        pending, stable, count = causal_decision(_logits(3), pending, stable, count)
        out.append(stable)
    assert out[:24] == [0] * 24 and out[24] == 3


def test_min_duration_one_switches_immediately():
    """min_duration=1 等价于不做最小时长平滑（逐帧 argmax）。"""

    assert _simulate(1) == [3] * 10


def test_min_duration_five_switches_after_five_frames():
    """min_duration=5：第 5 帧起切换。"""

    assert _simulate(5) == [0, 0, 0, 0, 3, 3, 3, 3, 3, 3]


def test_min_duration_zero_falls_back_to_one():
    """非法值 0 不应导致"永不切换"，按 1 处理（调用方另有 ≥1 校验）。"""

    assert _simulate(0) == [3] * 10


def test_config_accepts_smoothing_min_duration(tmp_path):
    """evaluation.smoothing_min_duration 是合法配置键（白名单已登记）。"""

    validate_config({
        "schema_version": 1,
        "pipeline": "sliding_window_temporal",
        "model": {"type": "gru"},
        "data": {"dataset_ref": "temporal.actionmixed-auto-v3"},
        "evaluation": {"mode": "formal", "smoothing_min_duration": 1},
    })


def test_experiment_config_carries_smoothing_key(tmp_path):
    """配置装载后该键可被读取（缺省时由流水线回落到 25）。"""

    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(
        "schema_version: 1\npipeline: sliding_window_temporal\n"
        "model:\n  type: gru\n  input_dim: 96\n  num_classes: 6\n"
        "data:\n  dataset_ref: temporal.actionmixed-auto-roi-v2\n"
        "feature_schema:\n  dim: 96\n  version: actionmixed-roi-grid-v2\n"
        "evaluation:\n  mode: formal\n  smoothing_min_duration: 5\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["evaluation"]["smoothing_min_duration"] == 5
