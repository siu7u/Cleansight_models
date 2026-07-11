"""时序模型族注册表（temporal 纵内部）。

新增时序族（Transformer / causal-TCN…）在此登记；纵内部通过 ``get_family(id)``
取用。这是**时序专属**的注册表，不与 detection 纵共享 —— 两类模型故意不强行
统一为同一族抽象。
"""

from __future__ import annotations

from .gru import GruFamily

_FAMILIES = {
    GruFamily.family_id: GruFamily,
}


def get_family(family_id: str):
    if family_id not in _FAMILIES:
        raise KeyError(f"未注册的时序模型族: {family_id}；已注册: {sorted(_FAMILIES)}")
    return _FAMILIES[family_id]()
