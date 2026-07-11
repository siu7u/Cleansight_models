"""单帧无状态喂入模式（窗口=1，需求 §5.2）。

描述目标检测这类"每次只看一张图、无跨帧状态"的喂入语义。它不产出时序逐帧
数组，因此不实现 ``FeedingResult`` 那套 ``evaluate``；检测任务自持推理循环
（``ultralytics.val``），只消费本模式的 ``semantics`` 挂进信封说明喂入方式。
"""

from __future__ import annotations


class SingleFrameFeeding:
    name = "single_frame"
    requires_performance = False

    @property
    def semantics(self) -> dict:
        return {
            "mode": "single_frame",
            "sees": "one_image",
            "stateless": True,
            "windowing": "none",
            "cold_start": "n/a",
            "reset": "per_image",
            "note": "单帧无状态检测，逐图独立推理",
        }

    def evaluate(self, family, model, datasets, device):
        raise NotImplementedError(
            "single_frame 为单帧无状态语义；检测任务自持推理，仅消费本模式的 "
            "semantics，不经由 feeding.evaluate（该接口返回时序逐帧数组）。"
        )
