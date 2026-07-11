"""时序喂入模式注册表（temporal 纵内部）。

新增模式在此登记，配置以名字引用。一个实验用顶层单个 ``feeding`` 字段表达喂入
模式，训练与评估共用同一个。这是**时序专属**的注册表：``single_frame`` 那种检测
语义不在此列 —— 检测纵自持推理，不借道喂入模式。
"""

from __future__ import annotations

from .full_sequence import FullSequenceFeeding
from .stateful import StatefulFeeding
from .windowed_causal import WindowedCausalFeeding

_MODES = {
    FullSequenceFeeding.name: FullSequenceFeeding,
    WindowedCausalFeeding.name: WindowedCausalFeeding,
    StatefulFeeding.name: StatefulFeeding,
}


def get_feeding(name: str):
    if name not in _MODES:
        raise KeyError(f"未注册的喂入模式: {name}；已注册: {sorted(_MODES)}")
    return _MODES[name]()
