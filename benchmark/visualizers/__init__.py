"""PredictionOutput 的 benchmark 呈现注册表。"""

from __future__ import annotations

from collections.abc import Callable


def get_visualizer(pipeline: str) -> Callable | None:
    """按 Pipeline 名返回纯呈现函数；不加载 checkpoint，也不重新执行推理。"""

    if pipeline in {"full_sequence_temporal", "sliding_window_temporal"}:
        from benchmark.visualizers.temporal import render_prediction_timeline

        return render_prediction_timeline
    return None


__all__ = ["get_visualizer"]
