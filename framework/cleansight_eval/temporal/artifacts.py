"""时序流水线到 benchmark prediction artifact 的过渡适配。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Sequence

try:
    from benchmark.core.artifacts import build_temporal_prediction_artifact
except ModuleNotFoundError:  # pragma: no cover - 从 framework 目录执行时触发
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from benchmark.core.artifacts import build_temporal_prediction_artifact


def build_prediction_artifact(
    pred_by_item: Mapping[str, Sequence[str]],
    truth_by_item: Mapping[str, Sequence[str]],
    labels: Sequence[str],
    *,
    window: int | None,
    inference_mode: str,
    prediction_start_frame: int = 0,
) -> dict:
    """把标签名序列编码成 benchmark 可复算的逐视频 artifact。"""

    name_to_id = {name: index for index, name in enumerate(labels)}
    pred_ids = {
        item: [name_to_id[value] for value in values]
        for item, values in pred_by_item.items()
    }
    truth_ids = {
        item: [name_to_id[value] for value in values]
        for item, values in truth_by_item.items()
    }
    return build_temporal_prediction_artifact(
        pred_by_item=pred_ids,
        truth_by_item=truth_ids,
        index_to_action={index: name for name, index in name_to_id.items()},
        window=window,
        inference_mode=inference_mode,
        prediction_start_frame=prediction_start_frame,
    )
