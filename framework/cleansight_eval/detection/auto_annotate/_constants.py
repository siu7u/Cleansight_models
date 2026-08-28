"""自动标注共享常量：类别表与抽样帧率。

注：类别表目前在此维护一份（与 ``temporal/features/clean_bbox_v2`` 的检测目标
表、``framework/testsets.yaml`` 的数据集条目并行存在）。检测类顺序即
``frames/data.yaml`` 的 class_id 语义，改动需同步消费端。
"""

from __future__ import annotations

# 默认每类别轨迹数（slot 数）；hand 双实例与 clean_bbox_v2 的 hand slots 一致。
DEFAULT_TOP_K: dict[str, int] = {"hand": 2}

# 时序训练数据布局的类别表（与 temporal.actionmixed-v2 / testsets.yaml 一致）。
DETECTION_CLASSES = [
    "hand",
    "scope_control_body",
    "scope_mid_section",
    "scope_distal_end",
    "syringe",
    "air_gun",
    "short_brush",
    "brush_tip_out",
]
# 动作类别名与 Label Studio 同步(project-16 起 air_injection 更名 water_injection,
# action id 位置不变仍为 1,旧数据集产物中的 air_injection 为历史名)。
ACTION_CLASSES = [
    "idle",
    "water_injection",
    "flush",
    "long_brush_insert",
    "long_brush_withdraw",
    "short_brush_cleaning",
]
# 标签抽样帧率（与真实 ActionMixed 训练数据一致）。
TARGET_LABEL_FPS = 7.5
