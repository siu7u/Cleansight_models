"""逐步有状态喂入模式（占位，需求 §11.4）。

有状态模型逐步接收输入并维护内部状态，语义与滑窗不同。本次不实现，仅登记
占位以说明扩展点：新增喂入模式只需在 temporal 纵内实现同名 evaluate/训练接口，
不影响其他模式。
"""

from __future__ import annotations

from .result import FeedingResult


class StatefulFeeding:
    name = "stateful"
    requires_performance = True

    def evaluate(self, family, model, datasets, device) -> FeedingResult:
        raise NotImplementedError(
            "stateful 喂入模式尚未实现；这是预留扩展点（见需求 §5.2 / §11.4）。"
        )
