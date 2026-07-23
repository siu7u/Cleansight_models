"""训练/推理 Pipeline 注册表（唯一同时 import 三条流水线的地方）。

framework 的训练入口和 benchmark 的评测入口都通过本表按 ``cfg["pipeline"]`` 获取模型
执行能力。各流水线实现统一的 ``Pipeline`` 接口，但各自保留数据组织、训练和预测语义；
注册表只负责构造，不承载评测、指标或可视化实现。
"""

from __future__ import annotations

from .pipeline import Pipeline
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
