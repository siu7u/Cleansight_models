"""pipeline → 流水线类 的分派表（唯一同时 import 三条流水线的地方）。

CLI 的组合根（composition root）：core 不知道流水线的存在、各流水线互不 import，唯有此处
按 ``cfg["pipeline"]`` 把请求分派到对应流水线。流水线靠**同名方法约定**被 duck-type 调用
（``validate_config`` / ``train`` / ``predict``）——这是编排（脊柱关切）的约定，不是模型
Protocol，故无需基类。
"""

from __future__ import annotations

from ..detection.pipeline import DetectionPipeline
from ..temporal.full_sequence_pipeline import FullSequenceTemporalPipeline
from ..temporal.sliding_window_pipeline import SlidingWindowTemporalPipeline

_PIPELINES = {
    DetectionPipeline.pipeline_name: DetectionPipeline,
    FullSequenceTemporalPipeline.pipeline_name: FullSequenceTemporalPipeline,
    SlidingWindowTemporalPipeline.pipeline_name: SlidingWindowTemporalPipeline,
}


def get_pipeline(pipeline: str):
    if pipeline not in _PIPELINES:
        raise KeyError(f"未注册的流水线: {pipeline}；已注册: {sorted(_PIPELINES)}")
    return _PIPELINES[pipeline]()
