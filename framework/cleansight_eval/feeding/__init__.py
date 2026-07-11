"""喂入模式注册表（train/eval 中立的共享轴）。新增模式在此登记，配置以名字引用。

一个实验用**顶层单个 ``feeding`` 字段**表达喂入模式，训练与评估共用同一个。
CLI/任务通过 ``get_feeding(name)`` 取用，不 import 具体模式。
"""

from __future__ import annotations

from .full_sequence import FullSequenceFeeding
from .single_frame import SingleFrameFeeding
from .stateful import StatefulFeeding
from .windowed_causal import WindowedCausalFeeding

_MODES = {
    FullSequenceFeeding.name: FullSequenceFeeding,
    WindowedCausalFeeding.name: WindowedCausalFeeding,
    SingleFrameFeeding.name: SingleFrameFeeding,
    StatefulFeeding.name: StatefulFeeding,
}


def get_feeding(name: str):
    if name not in _MODES:
        raise KeyError(f"未注册的喂入模式: {name}；已注册: {sorted(_MODES)}")
    return _MODES[name]()
