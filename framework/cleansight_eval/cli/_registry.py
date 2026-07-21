"""pipeline → 流水线类 的分派表（唯一同时 import 三条流水线的地方）。

CLI 的组合根（composition root）：core 不知道流水线的存在、各流水线互不 import，唯有此处
按 ``cfg["pipeline"]`` 把请求分派到对应流水线。所有流水线实现统一的 ``Pipeline`` 接口，
但各自保留数据组织、训练和预测语义；注册表只负责构造，不承载领域实现。
"""

from __future__ import annotations

from collections.abc import Callable

from ..core.pipeline import Pipeline
from ..detection.pipeline import DetectionPipeline
from ..temporal.full_sequence_pipeline import FullSequenceTemporalPipeline
from ..temporal.sliding_window_pipeline import SlidingWindowTemporalPipeline

_PIPELINES: dict[str, type[Pipeline]] = {
    DetectionPipeline.pipeline_name: DetectionPipeline,
    FullSequenceTemporalPipeline.pipeline_name: FullSequenceTemporalPipeline,
    SlidingWindowTemporalPipeline.pipeline_name: SlidingWindowTemporalPipeline,
}


def get_pipeline(pipeline: str) -> Pipeline:
    if pipeline not in _PIPELINES:
        raise KeyError(f"未注册的流水线: {pipeline}；已注册: {sorted(_PIPELINES)}")
    return _PIPELINES[pipeline]()


def get_visualizer(pipeline: str) -> Callable | None:
    """返回流水线的评测呈现函数；无可视化能力的流水线返回 ``None``。

    注册位于 CLI 组合根，避免 core 依赖具体领域。时序 renderer 延迟导入，使 YOLO
    训练和无图评测不必预先加载 matplotlib。
    """

    if pipeline in {
        FullSequenceTemporalPipeline.pipeline_name,
        SlidingWindowTemporalPipeline.pipeline_name,
    }:
        from ..temporal.viz import render_prediction_timeline

        return render_prediction_timeline
    return None
