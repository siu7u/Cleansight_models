"""模型族注册表。新增族只需在此登记，CLI 通过 family_id 查找。"""

from __future__ import annotations

from .gru.family import GruFamily
from .yolo.family import YoloFamily


_FAMILIES = {
    GruFamily.family_id: GruFamily,
    YoloFamily.family_id: YoloFamily,
}


def get_family(family_id: str):
    if family_id not in _FAMILIES:
        raise KeyError(f"未注册的模型族: {family_id}；已注册: {sorted(_FAMILIES)}")
    return _FAMILIES[family_id]()
